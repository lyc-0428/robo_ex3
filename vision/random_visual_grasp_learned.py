#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RoboMaster EP 实验三：
基于“成功抓取视觉样本 + 机械臂姿态”的随机位置自动靠近与抓取

核心思路
--------
1. 启动时机械臂 recenter，YOLO 先识别目标类别。
2. 根据类别读取对应学习文件：
       grasp_profile_tennis_ball_arm.json
       grasp_profile_bottle_arm.json
3. 把机械臂移动到该类别学习时的成功抓取姿态。
4. 重新识别目标，并使用“视觉闭环”控制底盘：
       - bottom_x 控制左右转向
       - bbox_width 控制前进 / 后退距离
   不再使用旧的 d=180/220mm 单应距离作为主要抓取距离。
5. 当目标的视觉状态与学习样本足够接近时：
       - nav-only：停止
       - 正常模式：直接闭合夹爪并判断是否夹住
6. 抓住后抬起。

为什么用 bbox_width
-------------------
摄像头固定在机械臂上，只要机械臂恢复到学习时的姿态，
同一类别物体的 bbox 宽度可作为稳定的相对距离指标：
    bbox 比学习值小 -> 目标更远 -> 前进
    bbox 比学习值大 -> 目标更近 -> 后退

这样不会再命令底盘做 22mm 之类过小、实机可能无响应的位移。

注意
----
- 本程序会移动底盘和机械臂。
- 第一次必须先用 nav-only 测试方向是否正确。
- bottle 的夹爪判定参数仍沿用 tennis_ball，抓瓶参数后续还可单独调。
"""

import argparse
import json
import math
import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from robomaster import robot

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from real import pick_place_lua_params as exp2


MODEL_DEFAULT = "/home/adam/team21_ex3_ws/src/robo_ex3/bottle_tennisball_best.pt"

PROFILE_DEFAULTS = {
    "tennis_ball": os.path.join(ROOT, "grasp_profile_tennis_ball_arm.json"),
    "bottle": os.path.join(ROOT, "grasp_profile_bottle_arm.json"),
}

CLASS_NAMES = ("tennis_ball", "bottle")

CAMERA_RES = "360p"
IMG_W = 640
IMG_H = 360
YOLO_SIZE = 320
CONF_DEFAULT = 0.35

# 视觉稳定
STABLE_FRAMES = 3
TARGET_TIMEOUT = 8.0

# 横向像素误差
X_TOL_PX = 14.0

# bbox 宽度相对误差容差
WIDTH_TOL_RATIO = 0.08

# 网球额外检查 bottom_y；瓶子底部接近图像下边缘，bottom_y 易饱和，不作为硬条件
BALL_Y_TOL_PX = 20.0

# 底盘动作
CHASSIS_XY_SPEED = 0.20
CHASSIS_Z_SPEED = 20

MIN_MOVE_M = 0.04
MID_MOVE_M = 0.06
MAX_MOVE_M = 0.08

SMALL_TURN_DEG = 3.0
MID_TURN_DEG = 6.0
LARGE_TURN_DEG = 10.0

MAX_SERVO_STEPS = 18

# 实机已验证：目标在画面右侧时，需要 z 为负方向转向
TURN_SIGN = -1.0

OPEN_POWER = 35
TEST_LIFT = 25.0
MAIN_LIFT = 60.0


def to_signed32(v):
    """
    RoboMaster SDK 在当前板子上出现 y 负值被当作 uint32 显示的问题：
    例如 4294967212 = 2^32 - 84，应解释为 -84。
    """
    if v is None:
        return None
    v = float(v)
    if v > 2147483647:
        v -= 4294967296.0
    return v


def load_profile(path):
    with open(path, "r", encoding="utf-8") as f:
        p = json.load(f)

    required = (
        "target",
        "bottom_x_px",
        "bottom_y_px",
        "bbox_width_px",
        "bbox_height_px",
        "arm_x_mm",
        "arm_y_mm",
    )
    for key in required:
        if key not in p:
            raise ValueError(f"profile 缺少字段: {key}")

    p["arm_x_mm"] = to_signed32(p["arm_x_mm"])
    p["arm_y_mm"] = to_signed32(p["arm_y_mm"])

    return p


def detect_one_frame(model, frame_bgr, target, use_cuda, conf):
    result = model.predict(
        source=frame_bgr,
        imgsz=YOLO_SIZE,
        conf=conf,
        verbose=False,
        device=0 if use_cuda else "cpu",
        half=use_cuda,
    )[0]

    detections = []

    if result.boxes is None:
        return None, []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        name = result.names[cls_id]
        score = float(box.conf[0])

        if name not in CLASS_NAMES:
            continue
        if target != "auto" and name != target:
            continue

        x1, y1, x2, y2 = [
            float(v) for v in box.xyxy[0].detach().cpu().tolist()
        ]

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        bottom_x = cx
        bottom_y = y2
        width = x2 - x1
        height = y2 - y1

        detections.append({
            "class": name,
            "conf": score,
            "box": (x1, y1, x2, y2),
            "center": (cx, cy),
            "bottom_x": bottom_x,
            "bottom_y": bottom_y,
            "width": width,
            "height": height,
        })

    detections.sort(key=lambda d: (-d["conf"], -d["width"]))
    best = detections[0] if detections else None
    return best, detections


class FPSMeter:
    def __init__(self, window=2.0):
        self.window = float(window)
        self.times = deque()

    def tick(self):
        now = time.monotonic()
        self.times.append(now)
        cutoff = now - self.window
        while self.times and self.times[0] < cutoff:
            self.times.popleft()

    def value(self):
        if len(self.times) < 2:
            return 0.0
        dt = self.times[-1] - self.times[0]
        if dt <= 0:
            return 0.0
        return (len(self.times) - 1) / dt


class AsyncVision:
    def __init__(
        self,
        camera,
        model,
        target,
        conf,
        infer_fps=6.0,
        display_fps=30.0,
    ):
        self.camera = camera
        self.model = model
        self.target = target
        self.conf = conf
        self.infer_fps = max(0.5, float(infer_fps))
        self.display_fps = max(5.0, float(display_fps))

        self.use_cuda = torch.cuda.is_available()

        self.stop_event = threading.Event()
        self.user_quit = False

        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.frame_seq = 0

        self.det_lock = threading.Lock()
        self.latest_best = None
        self.latest_detections = []
        self.det_seq = 0
        self.det_time = 0.0

        self.state_lock = threading.Lock()
        self.status = "starting..."
        self.profile = None

        self.capture_meter = FPSMeter()
        self.infer_meter = FPSMeter()
        self.display_meter = FPSMeter()

        self.threads = []

    def set_target(self, target):
        self.target = target

    def set_profile(self, profile):
        with self.state_lock:
            self.profile = profile

    def set_status(self, text):
        with self.state_lock:
            self.status = str(text)

    def start(self):
        self.threads = [
            threading.Thread(target=self._capture_loop, daemon=True),
            threading.Thread(target=self._inference_loop, daemon=True),
            threading.Thread(target=self._display_loop, daemon=True),
        ]
        for t in self.threads:
            t.start()

    def stop(self):
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=1.2)

    def is_stopped(self):
        return self.stop_event.is_set()

    def _capture_loop(self):
        while not self.stop_event.is_set():
            try:
                try:
                    frame = self.camera.read_cv2_image(
                        strategy="newest",
                        timeout=1,
                    )
                except TypeError:
                    frame = self.camera.read_cv2_image(strategy="newest")
            except Exception as e:
                print("camera warning:", repr(e))
                time.sleep(0.05)
                continue

            if frame is None:
                continue

            frame = np.ascontiguousarray(frame)

            with self.frame_lock:
                self.latest_frame = frame
                self.frame_seq += 1

            self.capture_meter.tick()

    def _get_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None, self.frame_seq
            return self.latest_frame.copy(), self.frame_seq

    def _inference_loop(self):
        min_period = 1.0 / self.infer_fps
        last_run = 0.0
        last_frame_seq = -1

        while not self.stop_event.is_set():
            now = time.monotonic()
            remain = min_period - (now - last_run)
            if remain > 0:
                time.sleep(min(remain, 0.01))
                continue

            frame, frame_seq = self._get_frame()
            if frame is None or frame_seq == last_frame_seq:
                time.sleep(0.005)
                continue

            last_frame_seq = frame_seq
            last_run = time.monotonic()

            try:
                best, detections = detect_one_frame(
                    self.model,
                    frame,
                    self.target,
                    self.use_cuda,
                    self.conf,
                )
            except Exception as e:
                print("YOLO warning:", repr(e))
                time.sleep(0.05)
                continue

            with self.det_lock:
                self.latest_best = best
                self.latest_detections = detections
                self.det_seq += 1
                self.det_time = time.monotonic()

            self.infer_meter.tick()

    def snapshot(self):
        with self.det_lock:
            best = None if self.latest_best is None else dict(self.latest_best)
            dets = [dict(x) for x in self.latest_detections]
            return self.det_seq, self.det_time, best, dets

    def current_det_seq(self):
        with self.det_lock:
            return self.det_seq

    def _display_loop(self):
        period = 1.0 / self.display_fps

        try:
            cv2.namedWindow(
                "EXP3 Learned Visual Grasp",
                cv2.WINDOW_NORMAL,
            )
            cv2.resizeWindow(
                "EXP3 Learned Visual Grasp",
                960,
                540,
            )

            while not self.stop_event.is_set():
                t0 = time.monotonic()

                frame, _ = self._get_frame()
                if frame is not None:
                    _, det_time, best, dets = self.snapshot()

                    with self.state_lock:
                        status = self.status
                        profile = None if self.profile is None else dict(self.profile)

                    shown = frame.copy()
                    h, w = shown.shape[:2]

                    for det in dets:
                        x1, y1, x2, y2 = [
                            int(round(v)) for v in det["box"]
                        ]
                        bx = int(round(det["bottom_x"]))
                        by = int(round(det["bottom_y"]))

                        is_best = best is not None and det == best
                        color = (0, 255, 0) if is_best else (0, 180, 255)

                        cv2.rectangle(
                            shown,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2,
                        )

                        cv2.circle(
                            shown,
                            (bx, by),
                            6,
                            (0, 0, 255),
                            -1,
                        )

                        text = (
                            f"{det['class']} {det['conf']:.2f} "
                            f"x={det['bottom_x']:.0f} "
                            f"y={det['bottom_y']:.0f} "
                            f"w={det['width']:.0f}"
                        )

                        cv2.putText(
                            shown,
                            text,
                            (x1, max(25, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48,
                            color,
                            1,
                        )

                    if profile is not None:
                        rx = int(round(profile["bottom_x_px"]))
                        ry = int(round(profile["bottom_y_px"]))

                        cv2.drawMarker(
                            shown,
                            (rx, ry),
                            (255, 0, 255),
                            markerType=cv2.MARKER_CROSS,
                            markerSize=28,
                            thickness=2,
                        )

                        cv2.putText(
                            shown,
                            f"LEARNED x={rx} y={ry} w={profile['bbox_width_px']:.0f}",
                            (10, 54),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.47,
                            (255, 0, 255),
                            1,
                        )

                    cv2.rectangle(
                        shown,
                        (0, 0),
                        (w, 32),
                        (0, 0, 0),
                        -1,
                    )

                    det_age = 0.0
                    if det_time > 0:
                        det_age = (time.monotonic() - det_time) * 1000

                    line = (
                        f"{status} | VIDEO {self.capture_meter.value():.1f} "
                        f"YOLO {self.infer_meter.value():.1f} "
                        f"DISPLAY {self.display_meter.value():.1f} "
                        f"age {det_age:.0f}ms"
                    )

                    cv2.putText(
                        shown,
                        line,
                        (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.44,
                        (255, 255, 255),
                        1,
                    )

                    cv2.imshow(
                        "EXP3 Learned Visual Grasp",
                        shown,
                    )

                    self.display_meter.tick()

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    self.user_quit = True
                    self.stop_event.set()
                    break

                used = time.monotonic() - t0
                remain = period - used
                if remain > 0:
                    time.sleep(remain)

        except Exception as e:
            print("display warning:", repr(e))
            self.stop_event.set()
        finally:
            try:
                cv2.destroyWindow("EXP3 Learned Visual Grasp")
            except Exception:
                pass


def locate_stable(vision, timeout=TARGET_TIMEOUT, stable_frames=STABLE_FRAMES):
    """
    只使用新的 YOLO 结果，返回稳定中位数。
    """
    start = time.monotonic()
    last_seq = vision.current_det_seq()
    history = []
    track_class = None

    while time.monotonic() - start < timeout:
        if vision.is_stopped():
            raise KeyboardInterrupt

        seq, _, best, _ = vision.snapshot()

        if seq <= last_seq:
            time.sleep(0.01)
            continue

        last_seq = seq

        if best is None:
            history = []
            track_class = None
            continue

        if track_class != best["class"]:
            history = [best]
            track_class = best["class"]
        else:
            history.append(best)

        history = history[-stable_frames:]

        if len(history) >= stable_frames:
            return {
                "class": history[-1]["class"],
                "conf": float(np.median([d["conf"] for d in history])),
                "bottom_x": float(np.median([d["bottom_x"] for d in history])),
                "bottom_y": float(np.median([d["bottom_y"] for d in history])),
                "width": float(np.median([d["width"] for d in history])),
                "height": float(np.median([d["height"] for d in history])),
            }

    return None


def turn_step_from_x_error(x_error):
    a = abs(x_error)

    if a > 90:
        mag = LARGE_TURN_DEG
    elif a > 45:
        mag = MID_TURN_DEG
    else:
        mag = SMALL_TURN_DEG

    # x_error > 0 = 目标在画面右侧
    return TURN_SIGN * math.copysign(mag, x_error)


def move_step_from_width_ratio(ratio):
    """
    ratio = 当前 bbox_width / 学习 bbox_width

    ratio < 1：目标更远，前进
    ratio > 1：目标更近，后退

    返回底盘 x 米。
    """
    if ratio < 0.72:
        return MAX_MOVE_M

    if ratio < 0.86:
        return MID_MOVE_M

    if ratio < (1.0 - WIDTH_TOL_RATIO):
        return MIN_MOVE_M

    if ratio > 1.35:
        return -MAX_MOVE_M

    if ratio > 1.18:
        return -MID_MOVE_M

    if ratio > (1.0 + WIDTH_TOL_RATIO):
        return -MIN_MOVE_M

    return 0.0


def visual_servo(chassis, vision, profile, args):
    ref_x = float(profile["bottom_x_px"])
    ref_y = float(profile["bottom_y_px"])
    ref_w = float(profile["bbox_width_px"])
    cls = profile["target"]

    for step in range(1, args.max_steps + 1):
        vision.set_status(f"SERVO {step}/{args.max_steps}")

        loc = locate_stable(
            vision,
            timeout=args.target_timeout,
            stable_frames=args.stable_frames,
        )

        if loc is None:
            print("没有稳定识别到目标，停止。")
            return None

        x_error = loc["bottom_x"] - ref_x
        y_error = loc["bottom_y"] - ref_y
        width_ratio = loc["width"] / ref_w

        print()
        print(
            f"[SERVO {step}] {loc['class']} conf={loc['conf']:.2f} | "
            f"x={loc['bottom_x']:.1f} ref={ref_x:.1f} err={x_error:+.1f}px | "
            f"y={loc['bottom_y']:.1f} ref={ref_y:.1f} err={y_error:+.1f}px | "
            f"w={loc['width']:.1f} ref={ref_w:.1f} ratio={width_ratio:.3f}"
        )

        # 先横向对准
        if abs(x_error) > args.x_tol:
            z = turn_step_from_x_error(x_error)

            direction = "右" if x_error > 0 else "左"

            print(
                f"目标在画面{direction}侧，转向 {z:+.1f}deg"
            )

            if args.dry_run:
                return loc

            chassis.move(
                x=0,
                y=0,
                z=float(z),
                z_speed=CHASSIS_Z_SPEED,
            ).wait_for_completed()

            time.sleep(0.30)
            continue

        # 再按 bbox 宽度控制前后
        dx = move_step_from_width_ratio(width_ratio)

        if dx != 0.0:
            direction = "前进" if dx > 0 else "后退"

            print(
                f"距离视觉校正：bbox ratio={width_ratio:.3f} "
                f"-> {direction} {abs(dx)*1000:.0f}mm"
            )

            if args.dry_run:
                return loc

            chassis.move(
                x=float(dx),
                y=0,
                z=0,
                xy_speed=CHASSIS_XY_SPEED,
            ).wait_for_completed()

            time.sleep(0.30)
            continue

        # 距离和横向都通过；网球再检查 y，瓶子 y 靠近画面底边不做硬限制
        if cls == "tennis_ball" and abs(y_error) > args.ball_y_tol:
            # width 已经对了但 y 差太大，说明姿态/地面条件可能和学习时不同
            print(
                f"WARN: bbox 宽度已匹配，但网球 bottom_y 误差仍为 "
                f"{y_error:+.1f}px，继续小步修正。"
            )

            dx = MIN_MOVE_M if y_error < 0 else -MIN_MOVE_M

            if args.dry_run:
                return loc

            chassis.move(
                x=float(dx),
                y=0,
                z=0,
                xy_speed=CHASSIS_XY_SPEED,
            ).wait_for_completed()

            time.sleep(0.30)
            continue

        print()
        print("==============================================")
        print("[OK] 已到学习到的成功抓取视觉位置")
        print(
            f"x_error={x_error:+.1f}px, "
            f"width_ratio={width_ratio:.3f}, "
            f"y_error={y_error:+.1f}px"
        )
        print("==============================================")

        vision.set_status("READY TO GRASP")
        return loc

    print("达到最大视觉闭环次数，停止。")
    return None


def configure_gripper(class_name):
    exp2.GRIP_POWER = 30
    exp2.GRIP_CLOSED_FAST = 2.0
    exp2.GRIP_STABLE_TIME = 2.5
    exp2.GRIP_STATUS_TIMEOUT = 8.0

    if class_name == "bottle":
        print("WARN: bottle 暂时沿用网球夹爪判定参数。")


def close_and_lift(ep, arm, gripper, class_name, vision):
    configure_gripper(class_name)

    print("关闭夹爪并判断...")
    vision.set_status("GRIPPER CLOSE")

    ok, detail = exp2.close_and_judge(gripper)
    print("grasp result:", ok, detail)

    if not ok:
        try:
            exp2.robot_signal_error(ep)
        except Exception:
            pass

        vision.set_status("GRASP FAILED")
        return False

    try:
        exp2.robot_signal_success(ep)
    except Exception:
        pass

    vision.set_status("GRASP SUCCESS / LIFT")

    print("抓取成功，开始抬起。")

    arm.move(
        x=0,
        y=TEST_LIFT,
    ).wait_for_completed()
    time.sleep(0.4)

    arm.move(
        x=0,
        y=MAIN_LIFT,
    ).wait_for_completed()
    time.sleep(0.4)

    print("物体已抬起。")
    return True


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--target",
        choices=["auto", "tennis_ball", "bottle"],
        default="auto",
    )

    p.add_argument("--model", default=MODEL_DEFAULT)

    p.add_argument("--tennis-profile", default=PROFILE_DEFAULTS["tennis_ball"])
    p.add_argument("--bottle-profile", default=PROFILE_DEFAULTS["bottle"])

    p.add_argument("--conf", type=float, default=CONF_DEFAULT)

    p.add_argument("--infer-fps", type=float, default=6.0)
    p.add_argument("--display-fps", type=float, default=30.0)

    p.add_argument("--stable-frames", type=int, default=STABLE_FRAMES)
    p.add_argument("--target-timeout", type=float, default=TARGET_TIMEOUT)

    p.add_argument("--x-tol", type=float, default=X_TOL_PX)
    p.add_argument("--ball-y-tol", type=float, default=BALL_Y_TOL_PX)

    p.add_argument("--max-steps", type=int, default=MAX_SERVO_STEPS)

    p.add_argument("--nav-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-object", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        print("ERROR: 模型不存在:", args.model)
        return 2

    profile_paths = {
        "tennis_ball": args.tennis_profile,
        "bottle": args.bottle_profile,
    }

    profiles = {}

    for cls, path in profile_paths.items():
        if os.path.isfile(path):
            profiles[cls] = load_profile(path)

            print(
                f"profile {cls}: "
                f"visual x={profiles[cls]['bottom_x_px']:.1f} "
                f"y={profiles[cls]['bottom_y_px']:.1f} "
                f"w={profiles[cls]['bbox_width_px']:.1f}; "
                f"arm x={profiles[cls]['arm_x_mm']:.1f} "
                f"y={profiles[cls]['arm_y_mm']:.1f}"
            )
        else:
            print("WARN: profile 不存在:", path)

    if args.target != "auto" and args.target not in profiles:
        print("ERROR: 目标类别没有对应 profile。")
        return 2

    use_cuda = torch.cuda.is_available()

    print("加载 YOLO...")
    model = YOLO(args.model)

    dummy = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    model.predict(
        dummy,
        imgsz=YOLO_SIZE,
        conf=args.conf,
        verbose=False,
        device=0 if use_cuda else "cpu",
        half=use_cuda,
    )
    print("YOLO OK")

    ep = robot.Robot()

    initialized = False
    camera_started = False
    subscribed = False
    vision = None
    gripper = None

    try:
        print("连接 RoboMaster...")
        ep.initialize(conn_type="ap")
        initialized = True
        print("RoboMaster SDK OK")

        arm = ep.robotic_arm
        chassis = ep.chassis
        gripper = ep.gripper
        camera = ep.camera

        gripper.sub_status(
            freq=5,
            callback=exp2.on_gripper_status,
        )
        subscribed = True

        try:
            exp2.robot_signal_idle(ep)
        except Exception:
            pass

        # 初始观察姿态
        print("机械臂 recenter...")
        arm.recenter().wait_for_completed()
        time.sleep(0.5)

        gripper.open(power=OPEN_POWER)
        time.sleep(0.5)

        print("启动视频流...")
        camera.start_video_stream(
            display=False,
            resolution=CAMERA_RES,
        )
        camera_started = True

        vision = AsyncVision(
            camera=camera,
            model=model,
            target=args.target,
            conf=args.conf,
            infer_fps=args.infer_fps,
            display_fps=args.display_fps,
        )
        vision.start()

        # 等第一帧
        t0 = time.monotonic()
        while vision.frame_seq == 0:
            if time.monotonic() - t0 > 5:
                raise RuntimeError("5秒内没有收到视频")
            time.sleep(0.02)

        # 第一步：识别类别
        vision.set_status("INITIAL DETECTION")

        first = locate_stable(
            vision,
            timeout=args.target_timeout,
            stable_frames=args.stable_frames,
        )

        if first is None:
            print("初始视野没有稳定识别到目标。")
            return 4

        cls = first["class"]

        if cls not in profiles:
            print(
                f"识别到 {cls}，但没有对应学习 profile，停止。"
            )
            return 4

        profile = profiles[cls]

        print()
        print("==============================================")
        print("识别类别:", cls)
        print(
            "恢复学习时机械臂姿态:",
            f"x={profile['arm_x_mm']:.1f}mm",
            f"y={profile['arm_y_mm']:.1f}mm",
        )
        print("==============================================")

        # 切换到确定类别，避免后续识别到另一类目标
        vision.set_target(cls)
        vision.set_profile(profile)
        vision.set_status("MOVE ARM TO LEARNED POSE")

        # 恢复该类别学习时的机械臂姿态
        arm.moveto(
            x=float(profile["arm_x_mm"]),
            y=float(profile["arm_y_mm"]),
        ).wait_for_completed()

        time.sleep(0.7)

        # 再次确保夹爪打开
        gripper.open(power=OPEN_POWER)
        time.sleep(0.3)

        # 开始闭环
        loc = visual_servo(
            chassis=chassis,
            vision=vision,
            profile=profile,
            args=args,
        )

        if loc is None:
            try:
                exp2.robot_signal_error(ep)
            except Exception:
                pass

            vision.set_status("TASK FAILED")
            return 5

        if args.nav_only or args.dry_run:
            try:
                exp2.robot_signal_success(ep)
            except Exception:
                pass

            vision.set_status("NAV COMPLETE")

            print("仅导航测试完成，不执行抓取。")
            time.sleep(3.0)
            return 0

        ok = close_and_lift(
            ep=ep,
            arm=arm,
            gripper=gripper,
            class_name=cls,
            vision=vision,
        )

        if not ok:
            return 6

        if args.keep_object:
            print("保持夹持 5 秒。")
            vision.set_status("KEEP OBJECT")
            time.sleep(5.0)

        vision.set_status("DONE")
        print("任务完成。")
        time.sleep(1.0)
        return 0

    except KeyboardInterrupt:
        print("\n用户中止。")
        return 130

    finally:
        if vision is not None:
            vision.stop()

        if subscribed and gripper is not None:
            try:
                gripper.unsub_status()
            except Exception:
                pass

        if camera_started:
            try:
                ep.camera.stop_video_stream()
            except Exception:
                pass

        if initialized:
            try:
                ep.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
