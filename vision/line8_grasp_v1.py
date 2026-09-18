#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EXP3 LINE8 GRASP V1
八个固定点扫描、返回目标、单物体抓取测试。

功能：
1. 使用已有 YOLOv8 模型识别 bottle / tennis_ball
2. 当前槽位只看画面中央区域
3. 如果 YOLO 漏检，则用 pointX_nothing.png 空点基准图差分兜底瓶子
4. 连续多帧投票，判断当前点：
       bottle
       tennis_ball
       EMPTY
       UNCERTAIN
5. 可选横向移动，每次固定 222 mm
6. 可选返回第一个目标，复查后调用 v9 实测抓取闭环
7. 抓取前必须再次获得 YOLO 真实检测框
8. 只抓取并抬起，不运输、不分类放置
9. 默认仍为安全扫描模式，只有显式参数才会抓取
10. 保存每个槽位图片和最终 JSON 日志

场地：
现场实测相邻中心距 222 mm
"""

import argparse
import json
import os
import time
import traceback
from collections import Counter
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from robomaster import robot


# ============================================================
# 固定配置
# ============================================================

MODEL_DEFAULT = (
    "/home/adam/team21_ex3_ws/src/robo_ex3/"
    "bottle_tennisball_best.pt"
)

CAMERA_RES = "360p"

IMAGE_W = 640
IMAGE_H = 360

CLASS_NAMES = {
    "bottle",
    "tennis_ball",
}

# 八点几何：现场实测标定值 222 mm
SLOT_COUNT = 8
SLOT_PITCH_M = 0.222

# 搜索阶段只认中央通道
IMAGE_CENTER_X = 320.0

# 比最终抓取宽松，因为这里只负责：
# “当前槽是什么”，不是判断是否能夹。
CENTER_TOL_PX = 55.0

# YOLO：远距离水瓶较小，离线 24 张截图验证需要低阈值和较大输入尺寸。
YOLO_IMGSZ = 960
YOLO_CONF = 0.05

# 空点基准差分兜底：仅在 YOLO 没有中央候选时使用。
# Only compare the pickup area directly in front of the gripper.
BASELINE_ROI_X1 = 0.40
BASELINE_ROI_X2 = 0.60
BASELINE_ROI_Y1 = 0.35
BASELINE_ROI_Y2 = 0.92
BASELINE_GRAY_DIFF = 24
BOTTLE_DIFF_PIXELS = 350

# A larger mean means the camera/arm pose no longer matches the baseline.
# Baseline fallback must not classify an object when the baseline is invalid.
BASELINE_MAX_MEAN_DIFF = 14.0

# 每个槽位采样
SAMPLE_FRAMES = 9
REQUIRED_HITS = 4

# 到达槽位后等待底盘稳定
MOVE_SETTLE_S = 0.65

# 横移速度
LATERAL_SPEED = 0.12

# 默认沿当前实机配置的方向扫描
DEFAULT_LATERAL_SIGN = -1.0

ROOT = (
    "/home/adam/team21_ex3_ws/src/"
    "robo_ex3_real_test/"
    "robo_ex3-codex-jetson-dji-real-test"
)

LOG_ROOT = os.path.join(
    ROOT,
    "diagnostics",
    "line8_grasp_v1",
)


# ============================================================
# 图像
# ============================================================

def get_frame(camera):
    try:
        frame = camera.read_cv2_image(
            strategy="newest",
            timeout=3,
        )
    except TypeError:
        frame = camera.read_cv2_image(
            strategy="newest"
        )

    return frame


# ============================================================
# YOLO
# ============================================================

def detect_frame(model, frame, device):
    result = model.predict(
        source=frame,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        verbose=False,
        device=device,
        half=(device != "cpu"),
    )[0]

    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = str(result.names[cls_id])

        if class_name not in CLASS_NAMES:
            continue

        conf = float(box.conf[0])

        x1, y1, x2, y2 = [
            float(v)
            for v in box.xyxy[0].detach().cpu().tolist()
        ]

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        detections.append({
            "class": class_name,
            "conf": conf,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": cx,
            "cy": cy,
            "bottom_x": cx,
            "bottom_y": y2,
            "width": x2 - x1,
            "height": y2 - y1,
        })

    return detections


def load_empty_templates(root):
    templates = {}

    for slot in range(1, SLOT_COUNT + 1):
        path = os.path.join(
            root,
            f"point{slot}_nothing.png",
        )

        image = cv2.imread(path)

        if image is None:
            print(
                f"[BASELINE] missing empty template: {path}"
            )
            continue

        templates[slot] = image

    print(
        f"[BASELINE] loaded "
        f"{len(templates)}/{SLOT_COUNT} empty templates"
    )

    return templates


def baseline_diff_details(frame, empty_template):
    if empty_template is None:
        return 0, 0.0, None, None

    if empty_template.shape != frame.shape:
        empty_template = cv2.resize(
            empty_template,
            (frame.shape[1], frame.shape[0]),
        )

    h, w = frame.shape[:2]

    x1 = int(w * BASELINE_ROI_X1)
    x2 = int(w * BASELINE_ROI_X2)
    y1 = int(h * BASELINE_ROI_Y1)
    y2 = int(h * BASELINE_ROI_Y2)

    current = cv2.GaussianBlur(
        frame[y1:y2, x1:x2],
        (5, 5),
        0,
    )

    empty = cv2.GaussianBlur(
        empty_template[y1:y2, x1:x2],
        (5, 5),
        0,
    )

    diff = cv2.absdiff(
        current,
        empty,
    )

    gray = cv2.cvtColor(
        diff,
        cv2.COLOR_BGR2GRAY,
    )

    mask = (
        gray > BASELINE_GRAY_DIFF
    ).astype("uint8")

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3),
        ),
    )

    return (
        int(mask.sum()),
        float(gray.mean()),
        mask * 255,
        (x1, y1, x2, y2),
    )


def baseline_diff_score(frame, empty_template):
    pixels, mean, _, _ = baseline_diff_details(
        frame,
        empty_template,
    )
    return pixels, mean


def choose_center_detection(detections):
    """
    当前槽只认靠近画面中央的物体。

    若中央附近同时检测到多个候选：
    1. 优先离中心最近
    2. 再优先置信度高
    """

    candidates = [
        d for d in detections
        if abs(d["bottom_x"] - IMAGE_CENTER_X)
        <= CENTER_TOL_PX
    ]

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda d: (
            abs(d["bottom_x"] - IMAGE_CENTER_X),
            -d["conf"],
        ),
    )


# ============================================================
# 绘制调试图片
# ============================================================

def draw_debug(frame, detections, selected, slot, diff_pixels=None):
    canvas = frame.copy()

    # 中心线
    cv2.line(
        canvas,
        (int(IMAGE_CENTER_X), 0),
        (int(IMAGE_CENTER_X), IMAGE_H),
        (255, 255, 255),
        1,
    )

    # 当前有效中心区域
    left = int(IMAGE_CENTER_X - CENTER_TOL_PX)
    right = int(IMAGE_CENTER_X + CENTER_TOL_PX)

    cv2.rectangle(
        canvas,
        (left, 0),
        (right, IMAGE_H - 1),
        (200, 200, 200),
        1,
    )

    for d in detections:
        x1 = int(d["x1"])
        y1 = int(d["y1"])
        x2 = int(d["x2"])
        y2 = int(d["y2"])

        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            (160, 160, 160),
            1,
        )

        text = (
            f"{d['class']} "
            f"{d['conf']:.2f}"
        )

        cv2.putText(
            canvas,
            text,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if selected is not None:
        x1 = int(selected["x1"])
        y1 = int(selected["y1"])
        x2 = int(selected["x2"])
        y2 = int(selected["y2"])

        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            3,
        )

    cv2.putText(
        canvas,
        (
            f"SLOT {slot}"
            if diff_pixels is None
            else f"SLOT {slot} diff={diff_pixels}"
        ),
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas


def live_debug_slot(
    model,
    camera,
    device,
    slot,
    log_dir,
    empty_templates,
):
    """Continuously show why each frame is classified."""
    window_camera = "LINE8 LIVE - camera (Q quit, S save)"
    window_mask = "LINE8 LIVE - baseline diff mask"
    template = empty_templates.get(slot)
    save_index = 0

    print()
    print(f"[LIVE] slot={slot}; Q=quit, S=save")
    print("[LIVE] chassis/arm/gripper are disabled")

    while True:
        frame = get_frame(camera)
        if frame is None:
            continue

        detections = detect_frame(model, frame, device)
        selected = choose_center_detection(detections)
        diff_pixels, diff_mean, mask, roi = baseline_diff_details(
            frame,
            template,
        )

        if selected is not None:
            decision = selected["class"]
            source = f"YOLO {selected['conf']:.2f}"
            color = (0, 255, 0)
        elif (
            diff_mean <= BASELINE_MAX_MEAN_DIFF
            and diff_pixels >= BOTTLE_DIFF_PIXELS
        ):
            decision = "bottle"
            source = "BASELINE"
            color = (0, 0, 255)
        elif diff_mean > BASELINE_MAX_MEAN_DIFF:
            decision = "EMPTY"
            source = "BASELINE INVALID"
            color = (0, 165, 255)
        else:
            decision = "EMPTY"
            source = "BASELINE"
            color = (255, 255, 0)

        canvas = draw_debug(
            frame,
            detections,
            selected,
            slot,
            diff_pixels,
        )

        if roi is not None:
            x1, y1, x2, y2 = roi
            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                2,
            )

        cv2.rectangle(canvas, (0, 38), (640, 103), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            f"DECISION: {decision}  SOURCE: {source}",
            (12, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            (
                f"diff={diff_pixels}/{BOTTLE_DIFF_PIXELS} "
                f"mean={diff_mean:.1f}/{BASELINE_MAX_MEAN_DIFF:.1f}"
            ),
            (12, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if mask is None:
            mask_view = np.zeros((IMAGE_H, IMAGE_W), dtype=np.uint8)
            cv2.putText(
                mask_view,
                f"Missing point{slot}_nothing.png",
                (30, IMAGE_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                255,
                2,
                cv2.LINE_AA,
            )
        else:
            mask_view = cv2.resize(mask, (IMAGE_W, IMAGE_H))

        cv2.imshow(window_camera, canvas)
        cv2.imshow(window_mask, mask_view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("s"), ord("S")):
            save_index += 1
            camera_path = os.path.join(
                log_dir,
                f"live_slot_{slot}_{save_index}_camera.jpg",
            )
            mask_path = os.path.join(
                log_dir,
                f"live_slot_{slot}_{save_index}_mask.png",
            )
            cv2.imwrite(camera_path, canvas)
            cv2.imwrite(mask_path, mask_view)
            print(f"[LIVE] saved: {camera_path}")
            print(f"[LIVE] saved: {mask_path}")

    cv2.destroyAllWindows()


# ============================================================
# 单槽稳定识别
# ============================================================

def classify_slot(
    model,
    camera,
    device,
    slot,
    log_dir,
    empty_templates,
):
    votes = []
    best_frame = None
    best_detections = []
    best_selected = None
    best_conf = -1.0
    best_diff_pixels = 0

    print()
    print("=" * 60)
    print(f"[SLOT {slot}] 开始稳定识别")
    print("=" * 60)

    # 先丢几帧，避免拿到移动前旧画面
    for _ in range(3):
        get_frame(camera)
        time.sleep(0.08)

    for index in range(SAMPLE_FRAMES):
        frame = get_frame(camera)

        if frame is None:
            print(
                f"[SLOT {slot}] "
                f"frame {index + 1}/{SAMPLE_FRAMES}: "
                f"NO FRAME"
            )
            votes.append(None)
            time.sleep(0.10)
            continue

        detections = detect_frame(
            model,
            frame,
            device,
        )

        selected = choose_center_detection(
            detections
        )

        if selected is None:
            diff_pixels, diff_mean = baseline_diff_score(
                frame,
                empty_templates.get(slot),
            )

            best_diff_pixels = max(
                best_diff_pixels,
                diff_pixels,
            )

            baseline_valid = (
                diff_mean <= BASELINE_MAX_MEAN_DIFF
            )

            if (
                baseline_valid
                and diff_pixels >= BOTTLE_DIFF_PIXELS
            ):
                votes.append("bottle")

                print(
                    f"[SLOT {slot}] "
                    f"frame {index + 1}/{SAMPLE_FRAMES}: "
                    f"bottle by baseline "
                    f"diff={diff_pixels} "
                    f"mean={diff_mean:.1f}"
                )

            else:
                votes.append(None)

                state = (
                    "EMPTY candidate"
                    if baseline_valid
                    else "EMPTY (baseline invalid)"
                )

                print(
                    f"[SLOT {slot}] "
                    f"frame {index + 1}/{SAMPLE_FRAMES}: "
                    f"{state} "
                    f"diff={diff_pixels} "
                    f"mean={diff_mean:.1f}"
                )

        else:
            votes.append(
                selected["class"]
            )

            print(
                f"[SLOT {slot}] "
                f"frame {index + 1}/{SAMPLE_FRAMES}: "
                f"{selected['class']} "
                f"conf={selected['conf']:.2f} "
                f"x={selected['bottom_x']:.1f} "
                f"y={selected['bottom_y']:.1f}"
            )

            if selected["conf"] > best_conf:
                best_conf = selected["conf"]
                best_frame = frame.copy()
                best_detections = detections
                best_selected = dict(selected)

        # 如果完全没检测到，也留一张现场图
        if best_frame is None:
            best_frame = frame.copy()
            best_detections = detections
            best_selected = selected

        time.sleep(0.10)

    valid_votes = [
        v for v in votes
        if v is not None
    ]

    counts = Counter(valid_votes)

    bottle_hits = counts.get(
        "bottle",
        0,
    )

    tennis_hits = counts.get(
        "tennis_ball",
        0,
    )

    if not valid_votes:
        final_class = "EMPTY"

    else:
        winner, hits = counts.most_common(1)[0]

        if hits >= REQUIRED_HITS:
            final_class = winner
        else:
            final_class = "UNCERTAIN"

    print()
    print(
        f"[SLOT {slot}] votes: "
        f"bottle={bottle_hits}, "
        f"tennis_ball={tennis_hits}, "
        f"empty={votes.count(None)}"
    )

    print(
        f"[SLOT {slot}] RESULT = "
        f"{final_class}"
    )

    if best_frame is not None:
        debug = draw_debug(
            best_frame,
            best_detections,
            best_selected,
            slot,
            best_diff_pixels,
        )

        image_path = os.path.join(
            log_dir,
            f"slot_{slot}_{final_class}.jpg",
        )

        cv2.imwrite(
            image_path,
            debug,
        )

        print(
            f"[SLOT {slot}] saved: "
            f"{image_path}"
        )

    return {
        "slot": slot,
        "result": final_class,
        "bottle_hits": bottle_hits,
        "tennis_ball_hits": tennis_hits,
        "empty_frames": votes.count(None),
        "sample_frames": SAMPLE_FRAMES,
        "baseline_diff_pixels": best_diff_pixels,
        "bottle_diff_threshold": BOTTLE_DIFF_PIXELS,
        "baseline_max_mean_diff": BASELINE_MAX_MEAN_DIFF,
        "best_conf": (
            round(best_conf, 4)
            if best_conf >= 0
            else None
        ),
        "best_x": (
            round(
                best_selected["bottom_x"],
                1,
            )
            if best_selected
            else None
        ),
        "best_y": (
            round(
                best_selected["bottom_y"],
                1,
            )
            if best_selected
            else None
        ),
    }


# ============================================================
# 返回目标后：复用 v9 实测抓取闭环
# ============================================================

def run_single_grasp(
    ep,
    camera,
    model,
    target_class,
    log_dir,
    nav_only=False,
    hold_seconds=5.0,
    bottle_cross_up_px=12.0,
    bottle_close_step_mm=20.0,
):
    """Approach one verified target, optionally close the gripper and lift."""
    import grasp_v9 as grasp

    profile_path = grasp.PROFILE_DEFAULTS[target_class]
    if not os.path.isfile(profile_path):
        raise FileNotFoundError(
            f"缺少 {target_class} 抓取标定文件: {profile_path}"
        )

    profile = grasp.load_profile(profile_path)

    if target_class == "bottle":
        # Stop slightly farther away so the gripper does not push the bottle.
        # The cross is only the visible reference; bottle also has independent
        # ready thresholds in v9, so both must move together.
        profile["bottom_y_px"] = max(
            0.0,
            float(profile["bottom_y_px"]) - bottle_cross_up_px,
        )
        grasp.BOTTLE_MIN_BOTTOM_Y = max(
            0.0,
            float(grasp.BOTTLE_MIN_BOTTOM_Y) - bottle_cross_up_px,
        )
        grasp.BOTTLE_MIN_HEIGHT = max(
            1.0,
            float(grasp.BOTTLE_MIN_HEIGHT) - 10.0,
        )
        grasp.BOTTLE_MAX_HEIGHT = max(
            grasp.BOTTLE_MIN_HEIGHT + 1.0,
            float(grasp.BOTTLE_MAX_HEIGHT) - 10.0,
        )
        grasp.MIN_MOVE_M = max(
            0.005,
            float(bottle_close_step_mm) / 1000.0,
        )
        grasp.MID_MOVE_M = max(
            grasp.MIN_MOVE_M,
            min(0.03, grasp.MIN_MOVE_M * 1.5),
        )
        grasp.MAX_MOVE_M = max(
            grasp.MID_MOVE_M,
            min(0.04, grasp.MIN_MOVE_M * 2.0),
        )

        print(
            "[BOTTLE SAFETY] "
            f"cross up={bottle_cross_up_px:.0f}px, "
            f"ready bottom_y>={grasp.BOTTLE_MIN_BOTTOM_Y:.0f}, "
            f"height={grasp.BOTTLE_MIN_HEIGHT:.0f}"
            f"~{grasp.BOTTLE_MAX_HEIGHT:.0f}, "
            "close steps="
            f"{grasp.MIN_MOVE_M * 1000:.0f}/"
            f"{grasp.MID_MOVE_M * 1000:.0f}/"
            f"{grasp.MAX_MOVE_M * 1000:.0f}mm"
        )
    profiles = {target_class: profile}

    args = SimpleNamespace(
        target=target_class,
        dry_run=False,
        observe_max_steps=grasp.OBSERVE_MAX_STEPS,
        x_tol=grasp.X_TOL_PX,
        ball_y_tol=grasp.BALL_Y_TOL_PX,
    )

    arm = ep.robotic_arm
    chassis = ep.chassis
    gripper = ep.gripper

    vision = None
    subscribed = False
    stage = "INITIALIZE"
    result = {
        "target": target_class,
        "nav_only": bool(nav_only),
        "success": False,
        "stage": stage,
    }

    try:
        if not nav_only:
            gripper.sub_status(
                freq=5,
                callback=grasp.exp2.on_gripper_status,
            )
            subscribed = True

        # Approach always starts with an open gripper. Nav-only never closes it.
        gripper.open(power=grasp.OPEN_POWER)
        time.sleep(0.5)

        stage = "START_VISION"
        result["stage"] = stage

        vision = grasp.AsyncVision(
            camera=camera,
            model=model,
            target=target_class,
            infer_fps=6.0,
            display_fps=30.0,
        )
        vision.start()

        start_wait = time.monotonic()
        while vision.frame_seq == 0:
            if time.monotonic() - start_wait > 5.0:
                raise RuntimeError("5秒内没有收到抓取视频帧")
            if vision.is_stopped():
                raise RuntimeError("抓取视频线程已停止")
            time.sleep(0.02)

        stage = "APPROACH"
        result["stage"] = stage

        cls, selected_profile = grasp.pregrasp_with_recovery(
            arm=arm,
            chassis=chassis,
            vision=vision,
            profiles=profiles,
            args=args,
        )

        if cls != target_class or selected_profile is None:
            result["stage"] = "APPROACH_FAILED"
            return result

        if nav_only:
            result.update({
                "success": True,
                "stage": "READY_TO_GRASP",
            })
            print(
                "[GRASP TEST] 已到抓取位，"
                "nav-only 不关闭夹爪"
            )
            time.sleep(2.0)
            return result

        stage = "CLOSE_AND_LIFT"
        result["stage"] = stage

        ok = grasp.close_and_lift(
            ep=ep,
            arm=arm,
            gripper=gripper,
            class_name=target_class,
            vision=vision,
        )

        result.update({
            "success": bool(ok),
            "stage": "LIFTED" if ok else "GRASP_FAILED",
        })

        if ok and hold_seconds > 0:
            print(
                f"[GRASP TEST] 抓取成功，"
                f"保持 {hold_seconds:.1f}s"
            )
            time.sleep(hold_seconds)

        return result

    except Exception as exc:
        result.update({
            "success": False,
            "stage": stage,
            "error": repr(exc),
        })
        raise

    finally:
        try:
            chassis.drive_speed(
                x=0,
                y=0,
                z=0,
                timeout=0.2,
            )
        except Exception:
            pass

        if vision is not None:
            vision.stop()

        if subscribed:
            try:
                gripper.unsub_status()
            except Exception:
                pass

        result_path = os.path.join(
            log_dir,
            "grasp_result.json",
        )
        try:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[GRASP TEST] log: {result_path}")
        except Exception as exc:
            print("[GRASP TEST] log warning:", repr(exc))


# ============================================================
# 底盘
# ============================================================

def move_one_slot(
    chassis,
    lateral_sign,
    reason="到下一槽",
):
    distance = (
        SLOT_PITCH_M
        * lateral_sign
    )

    print()
    print(
        f"[MOVE] {reason}："
        f"y={distance:+.3f} m"
    )

    action = chassis.move(
        x=0,
        y=distance,
        z=0,
        xy_speed=LATERAL_SPEED,
    )

    action.wait_for_completed()

    # 明确停车
    try:
        chassis.drive_speed(
            x=0,
            y=0,
            z=0,
            timeout=0.2,
        )
    except Exception:
        pass

    time.sleep(MOVE_SETTLE_S)


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=MODEL_DEFAULT,
    )

    parser.add_argument(
        "--start-slot",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--end-slot",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lateral-sign",
        type=float,
        choices=[-1.0, 1.0],
        default=DEFAULT_LATERAL_SIGN,
    )

    parser.add_argument(
        "--move",
        action="store_true",
        help=(
            "允许底盘按照222mm横向扫描。"
            "不加此参数时绝不移动底盘。"
        ),
    )

    parser.add_argument(
        "--return-first-object",
        action="store_true",
        help=(
            "扫描后逐格返回第一个检测到的物体，"
            "并再识别一次。不操作机械臂和夹爪。"
        ),
    )

    parser.add_argument(
        "--grasp-first-object",
        action="store_true",
        help=(
            "返回并通过YOLO复查后，启用单物体抓取与抬起。"
            "必须与 --return-first-object 一起使用。"
        ),
    )

    parser.add_argument(
        "--grasp-nav-only",
        action="store_true",
        help=(
            "只完成靠近、对准和机械臂就位，"
            "不关闭夹爪。"
        ),
    )

    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=5.0,
        help="抓取并抬起后保持时间，默认5秒。",
    )

    parser.add_argument(
        "--bottle-cross-up-px",
        type=float,
        default=12.0,
        help="水瓶抓取十字向上偏移像素，默认12。",
    )

    parser.add_argument(
        "--bottle-close-step-mm",
        type=float,
        default=20.0,
        help="水瓶近距离最小前进步长，默认20mm。",
    )

    parser.add_argument(
        "--live-debug",
        action="store_true",
        help=(
            "Display camera, ROI, YOLO source and baseline diff mask. "
            "Never moves the robot."
        ),
    )

    args = parser.parse_args()

    if not (
        1 <= args.start_slot
        <= args.end_slot
        <= SLOT_COUNT
    ):
        raise ValueError(
            "slot 范围必须在 1~8"
        )

    if args.return_first_object and not args.move:
        parser.error(
            "--return-first-object 必须与 --move 一起使用"
        )

    if args.grasp_first_object and not args.return_first_object:
        parser.error(
            "--grasp-first-object 必须与 "
            "--return-first-object 一起使用"
        )

    if args.grasp_nav_only and not args.grasp_first_object:
        parser.error(
            "--grasp-nav-only 必须与 --grasp-first-object 一起使用"
        )

    if args.hold_seconds < 0:
        parser.error("--hold-seconds 不能小于0")

    if not 0 <= args.bottle_cross_up_px <= 40:
        parser.error("--bottle-cross-up-px 必须在0~40之间")

    if not 5 <= args.bottle_close_step_mm <= 40:
        parser.error("--bottle-close-step-mm 必须在5~40之间")

    if not os.path.isfile(args.model):
        raise FileNotFoundError(
            f"找不到模型: {args.model}"
        )

    stamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    log_dir = os.path.join(
        LOG_ROOT,
        stamp,
    )

    os.makedirs(
        log_dir,
        exist_ok=True,
    )

    print()
    print("=" * 60)
    print("EXP3 LINE8 GRASP V1")
    print("=" * 60)
    print(
        f"slots: "
        f"{args.start_slot}"
        f" -> "
        f"{args.end_slot}"
    )
    print(
        f"pitch: "
        f"{SLOT_PITCH_M * 1000:.0f} mm"
    )
    print(
        f"lateral sign: "
        f"{args.lateral_sign:+.0f}"
    )
    print(
        f"move enabled: "
        f"{args.move}"
    )
    if args.grasp_first_object:
        print("ARM: ENABLED AFTER YOLO RECHECK")
        print(
            "GRIPPER: "
            + (
                "DISABLED (NAV ONLY)"
                if args.grasp_nav_only
                else "ENABLED AFTER ALIGNMENT"
            )
        )
        print("FORWARD/BACK: ENABLED DURING APPROACH")
        print("ROTATION: ENABLED DURING ALIGNMENT")
    else:
        print("ARM: DISABLED")
        print("GRIPPER: DISABLED")
        print("FORWARD/BACK: DISABLED")
        print("ROTATION: DISABLED")
    print("=" * 60)

    device = (
        0
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[YOLO] device={device}"
    )

    model = YOLO(
        args.model
    )

    empty_templates = load_empty_templates(
        ROOT
    )

    ep = robot.Robot()

    camera_started = False

    results = []

    try:
        print(
            "[ROBOT] connecting..."
        )

        ep.initialize(
            conn_type="ap"
        )

        chassis = ep.chassis
        camera = ep.camera

        print(
            "[CAMERA] starting 360p..."
        )

        camera.start_video_stream(
            display=False,
            resolution=CAMERA_RES,
        )

        camera_started = True

        time.sleep(1.5)

        if args.live_debug:
            live_debug_slot(
                model,
                camera,
                device,
                args.start_slot,
                log_dir,
                empty_templates,
            )
            return

        # --------------------------------------------
        # 安全模式：
        # 不加 --move 时只检查当前一个槽位。
        # --------------------------------------------

        if not args.move:
            if (
                args.start_slot
                != args.end_slot
            ):
                print()
                print(
                    "[SAFE MODE] 未指定 --move。"
                )
                print(
                    "[SAFE MODE] "
                    "只检测 start-slot，"
                    "不会移动底盘。"
                )

            slot = args.start_slot

            result = classify_slot(
                model,
                camera,
                device,
                slot,
                log_dir,
                empty_templates,
            )

            results.append(result)

        else:
            # 八槽扫描
            current = args.start_slot

            while current <= args.end_slot:
                result = classify_slot(
                    model,
                    camera,
                    device,
                    current,
                    log_dir,
                    empty_templates,
                )

                results.append(result)

                if current >= args.end_slot:
                    break

                move_one_slot(
                    chassis,
                    args.lateral_sign,
                )

                current += 1

            if args.return_first_object:
                target = next(
                    (
                        item for item in results
                        if item["result"]
                        in {"bottle", "tennis_ball"}
                    ),
                    None,
                )

                if target is None:
                    print()
                    print(
                        "[RETURN] 没有可返回的物体，底盘保持停止"
                    )
                else:
                    target_slot = target["slot"]
                    return_steps = current - target_slot

                    print()
                    print(
                        f"[RETURN] target=slot {target_slot} "
                        f"class={target['result']} "
                        f"steps={return_steps}"
                    )

                    for step in range(return_steps):
                        move_one_slot(
                            chassis,
                            -args.lateral_sign,
                            reason=(
                                f"返回目标 "
                                f"{step + 1}/{return_steps}"
                            ),
                        )

                    print(
                        f"[RETURN] 已到 slot {target_slot}，"
                        "停车并复查"
                    )

                    verification = classify_slot(
                        model,
                        camera,
                        device,
                        target_slot,
                        log_dir,
                        empty_templates,
                    )

                    target["return_verification"] = {
                        "result": verification["result"],
                        "bottle_hits": verification["bottle_hits"],
                        "tennis_ball_hits": verification[
                            "tennis_ball_hits"
                        ],
                        "empty_frames": verification["empty_frames"],
                        "best_conf": verification["best_conf"],
                    }

                    if args.grasp_first_object:
                        verified_by_yolo = (
                            verification["result"] == target["result"]
                            and verification["best_conf"] is not None
                        )

                        if not verified_by_yolo:
                            print()
                            print(
                                "[GRASP SAFETY] 返回复查没有获得"
                                "同类YOLO真实检测框，取消抓取"
                            )
                            target["grasp"] = {
                                "success": False,
                                "stage": "ABORTED_NO_YOLO_RECHECK",
                            }
                        else:
                            print()
                            print("=" * 60)
                            print(
                                f"[GRASP] YOLO复查通过: "
                                f"slot={target_slot} "
                                f"class={target['result']} "
                                f"conf={verification['best_conf']:.2f}"
                            )
                            print("=" * 60)

                            target["grasp"] = run_single_grasp(
                                ep=ep,
                                camera=camera,
                                model=model,
                                target_class=target["result"],
                                log_dir=log_dir,
                                nav_only=args.grasp_nav_only,
                                hold_seconds=args.hold_seconds,
                                bottle_cross_up_px=(
                                    args.bottle_cross_up_px
                                ),
                                bottle_close_step_mm=(
                                    args.bottle_close_step_mm
                                ),
                            )

        # --------------------------------------------
        # 汇总
        # --------------------------------------------

        print()
        print("=" * 60)
        print("FINAL SLOT SUMMARY")
        print("=" * 60)

        for item in results:
            print(
                f"SLOT {item['slot']}: "
                f"{item['result']}"
            )

        summary_path = os.path.join(
            log_dir,
            "summary.json",
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                results,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print()
        print(
            "summary:",
            summary_path,
        )

        print("=" * 60)

    except KeyboardInterrupt:
        print()
        print(
            "[STOP] Ctrl+C"
        )

        try:
            ep.chassis.drive_speed(
                x=0,
                y=0,
                z=0,
                timeout=0.2,
            )
        except Exception:
            pass

    except Exception as exc:
        error_payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }

        print()
        print("[ERROR]", repr(exc))

        error_paths = [
            os.path.join(log_dir, "fatal_error.json"),
            "/home/adam/Desktop/line8_grasp_last_error.json",
        ]

        for error_path in error_paths:
            try:
                with open(error_path, "w", encoding="utf-8") as f:
                    json.dump(
                        error_payload,
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                print(f"[ERROR] saved: {error_path}")
            except Exception as log_exc:
                print(
                    f"[ERROR] cannot save {error_path}: "
                    f"{log_exc!r}"
                )

    finally:
        try:
            if camera_started:
                ep.camera.stop_video_stream()
        except Exception:
            pass

        try:
            ep.chassis.drive_speed(
                x=0,
                y=0,
                z=0,
                timeout=0.2,
            )
        except Exception:
            pass

        try:
            ep.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
