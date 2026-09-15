#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RoboMaster EP 实验三：
基于学习视觉位置的多目标优先抓取 + ALIGN_PREGRASP 稳定版

相对上一版主要修改：
1. 网球使用 YOLO imgsz=320、conf=0.35
2. 水瓶使用 YOLO imgsz=640、conf=0.25
3. auto 模式使用 imgsz=640、conf=0.25
4. 稳定识别由“连续3帧”改为“最近5次结果中同类至少出现3次”
5. 远距离阶段机械臂保持 recenter 观察姿态，不会一识别到目标就低头
6. 目标进入近距离切换区后，才移动到对应类别的学习抓取姿态
7. 网球用 bbox width 判断远近
8. 水瓶用 bbox height + bottom_y 联合判断远近，并严格复现成功抓取视觉状态
9. 继续保留 bottom_x 控制左右转向
10. 抓取姿态下若目标丢失，会自动抬回观察姿态继续搜索/靠近

注意：
- 本程序会移动底盘和机械臂。
- 第一次先使用 nav-only 选项测试。
- 水瓶夹爪判定参数仍暂时沿用网球参数。
"""

import argparse
import json
import math
import os
import sys
import threading
import time
from collections import Counter, deque

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

YOLO_SIZE_BALL = 320
YOLO_SIZE_BOTTLE = 640

CONF_BALL = 0.35
CONF_BOTTLE = 0.25

STABLE_WINDOW = 5
STABLE_HITS = 3
TARGET_TIMEOUT = 8.0

X_TOL_PX = 14.0

# 网球 bbox 尺寸比例容差
SIZE_TOL_RATIO = 0.08

# 水瓶成功抓取视觉状态（来自实机5次成功样本）
BOTTLE_MIN_HEIGHT = 272.0
BOTTLE_MAX_HEIGHT = 290.0
BOTTLE_MIN_BOTTOM_Y = 350.0
BOTTLE_X_TOL_PX = 12.0

# 水瓶分段靠近阈值
BOTTLE_FAR_H = 240.0
BOTTLE_MID_H = 260.0
BOTTLE_NEAR_H = 270.0

# 网球额外检查 bottom_y
BALL_Y_TOL_PX = 20.0

CHASSIS_XY_SPEED = 0.20
CHASSIS_Z_SPEED = 20

MIN_MOVE_M = 0.04
MID_MOVE_M = 0.06
MAX_MOVE_M = 0.08

SMALL_TURN_DEG = 3.0
MID_TURN_DEG = 6.0
LARGE_TURN_DEG = 10.0

MAX_SERVO_STEPS = 18

# ==========================================================
# 两阶段视觉状态机
# 阶段1：OBSERVE/APPROACH，机械臂保持 recenter，摄像头看远处
# 阶段2：PREGRASP，目标足够近后才切换到学习抓取姿态
# ==========================================================

OBSERVE_CENTER_X = 320.0
OBSERVE_X_TOL_PX = 28.0

# 进入抓取姿态的视觉大小阈值。
# 这是“切换阈值”，不是最终抓取阈值。
# 最终抓取仍使用学习 profile 的严格条件。
BALL_PREGRASP_WIDTH = 60.0
BOTTLE_PREGRASP_HEIGHT = 170.0

OBSERVE_FAR_MOVE_M = 0.08
OBSERVE_MID_MOVE_M = 0.06
OBSERVE_NEAR_MOVE_M = 0.04

OBSERVE_MAX_STEPS = 80
PREGRASP_RECOVERY_LIMIT = 2

# ==========================================================
# 低头后的实时 CLOSE_APPROACH
# ==========================================================

# 低头后不再依赖 seq / after_seq 门槛。
# 只要当前同类检测框足够新鲜，就立即参与控制。
MAX_DET_AGE_S = 1.0

# 近距离循环周期
CLOSE_LOOP_PERIOD_S = 0.10

# 连续多久看不到有效目标才判定本轮近距跟踪失败
CLOSE_LOST_TIMEOUT_S = 2.5

# 近距离阶段最大控制次数
CLOSE_MAX_STEPS = 30

# 机械臂动作超时：
# RoboMaster SDK 的 wait_for_completed() 默认 timeout=None，
# 如果机械臂已经物理到位但完成ACK没有回来，程序会永久卡在 PREGRASP / MOVE ARM。
ARM_ACTION_TIMEOUT_S = 3.0
ARM_SETTLE_AFTER_TIMEOUT_S = 0.35

# ==========================================================
# 远距离连续速度控制
# ==========================================================

# 远距离阶段不再使用一段段 chassis.move()，改用 drive_speed()
# 这样底盘不会反复“启动-刹车-启动”。
APPROACH_FAST_MPS = 0.16
APPROACH_MID_MPS = 0.12
APPROACH_SLOW_MPS = 0.07

# 边走边转
TURN_FAST_DPS = 18.0
TURN_MID_DPS = 10.0
TURN_SLOW_DPS = 6.0

# 速度平滑：每个控制周期最大变化量
MAX_DV_MPS = 0.04
MAX_DZ_DPS = 5.0

# 连续控制周期
CONTROL_PERIOD_S = 0.12

# 预抓取阈值附近提前减速
PREGRASP_SLOW_RATIO = 0.82

# 预抓取前必须先横向对准，防止目标还在画面边缘就低头。
PREGRASP_ALIGN_ENTER_RATIO = 0.85
PREGRASP_ALIGN_X_TOL_PX = 30.0
PREGRASP_ALIGN_REQUIRED_HITS = 2
PREGRASP_CREEP_MPS = 0.04
OBSERVE_LOST_TIMEOUT_S = 2.0


# ==========================================================
# 多目标优先抓取 + 分类放置
# ==========================================================

# 同一画面里优先抓最近物体：
# 固定观察姿态下，物体与桌面接触点 bottom_y 越大，通常越靠近机器人。
# bbox 面积只用于 bottom_y 接近时的次级排序。
NEAREST_BOTTOM_Y_WEIGHT = 1.0

# 默认最多分拣 6 个物体（实验要求至少 6 个目标）
DEFAULT_MAX_ITEMS = 6

# 连续多少轮看不到目标后结束任务
EMPTY_RETRY_LIMIT = 3

# 分类方向：
# 实机中 z>0 为左转，z<0 为右转。
TENNIS_DROP_TURN_DEG = 90.0
BOTTLE_DROP_TURN_DEG = -90.0

# 转向分类区后，向前走一点，放下后再退回来
DROP_FORWARD_M = 0.18
DROP_XY_SPEED = 0.18
DROP_Z_SPEED = 25

# 抓起后已经抬升约 85mm。
# 放置前下降一点，避免从太高处直接扔下。
DROP_LOWER_MM = 55.0
DROP_OPEN_POWER = 35
DROP_OPEN_TIME = 1.0

# ==========================================================
# RETURN_SEARCH + SCAN
# ==========================================================
# 每抓完并分类一个目标后，不直接在当前靠前位置继续找下一件。
# 先后退恢复更宽视野，再做左/中/右扫描。
RETURN_BACK_M = 0.25
RETURN_BACK_SPEED = 0.18

# 扫描角度（chassis.move）
# 当前实机 move()：左转为正，右转为负。
SCAN_LEFT_DEG = 25.0
SCAN_RIGHT_DEG = -25.0
SCAN_Z_SPEED = 20

SCAN_SETTLE_S = 0.45
SCAN_DETECT_TIMEOUT_S = 1.8
SCAN_WINDOW = 5
SCAN_HITS = 2

# ==========================================================
# 转向符号：RoboMaster SDK 两个接口的 z 方向约定在当前实机表现不同
# ==========================================================
#
# 远距离连续控制使用 chassis.drive_speed(z=...)
# 实机验证：
#   x_error < 0（目标在画面左边） -> z < 0 能正确向左修正
# 因此直接使用 x_error 符号。
TURN_SIGN_DRIVE = 1.0
#
# 近距离离散修正使用 chassis.move(z=...)
# 实机验证：
#   x_error < 0（目标在画面左边） -> chassis.move 需要 z > 0 才能向左修正
# 因此必须反号。
TURN_SIGN_MOVE = -1.0

OPEN_POWER = 35
TEST_LIFT = 25.0
MAIN_LIFT = 60.0


def to_signed32(v):
    """
    当前 RoboMaster SDK/板端环境中，机械臂负 y 可能被显示为 uint32。
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
        profile = json.load(f)

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
        if key not in profile:
            raise ValueError(f"profile 缺少字段: {key}")

    profile["arm_x_mm"] = to_signed32(profile["arm_x_mm"])
    profile["arm_y_mm"] = to_signed32(profile["arm_y_mm"])

    return profile


def infer_config(target):
    """
    根据当前检测目标选择推理配置。

    tennis_ball:
        imgsz=320, conf=0.35

    bottle:
        imgsz=640, conf=0.25

    auto:
        为了兼顾透明瓶子，直接用 640 / 0.25
    """
    if target == "tennis_ball":
        return YOLO_SIZE_BALL, CONF_BALL

    return YOLO_SIZE_BOTTLE, CONF_BOTTLE


def detect_one_frame(model, frame_bgr, target, use_cuda):
    imgsz, conf = infer_config(target)

    result = model.predict(
        source=frame_bgr,
        imgsz=imgsz,
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
            float(v)
            for v in box.xyxy[0].detach().cpu().tolist()
        ]

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        bottom_x = cx
        bottom_y = y2

        width = x2 - x1
        height = y2 - y1

        detections.append(
            {
                "class": name,
                "conf": score,
                "box": (x1, y1, x2, y2),
                "center": (cx, cy),
                "bottom_x": bottom_x,
                "bottom_y": bottom_y,
                "width": width,
                "height": height,
            }
        )

    # 多目标优先级：
    # 1) bottom_y 越大 -> 在固定观察姿态下通常越近
    # 2) 如果 bottom_y 很接近，则 bbox 面积更大的优先
    # 3) 最后用置信度打破平局
    detections.sort(
        key=lambda d: (
            -d["bottom_y"],
            -(d["width"] * d["height"]),
            -d["conf"],
        )
    )

    best = detections[0] if detections else None

    return best, detections


class FPSMeter:
    """FPS统计做线程保护。"""
    def __init__(self, window=2.0):
        self.window = float(window)
        self.times = deque()
        self.lock = threading.Lock()

    def tick(self):
        now = time.monotonic()
        with self.lock:
            self.times.append(now)
            cutoff = now - self.window
            while self.times and self.times[0] < cutoff:
                self.times.popleft()

    def value(self):
        with self.lock:
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
        infer_fps=6.0,
        display_fps=30.0,
    ):
        self.camera = camera
        self.model = model
        self.target = target

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
        with self.state_lock:
            self.target = target

    def get_target(self):
        with self.state_lock:
            return self.target

    def set_profile(self, profile):
        with self.state_lock:
            self.profile = profile

    def set_status(self, text):
        with self.state_lock:
            self.status = str(text)

    def start(self):
        self.threads = [
            threading.Thread(
                target=self._capture_loop,
                name="camera-capture",
                daemon=True,
            ),
            threading.Thread(
                target=self._inference_loop,
                name="yolo-inference",
                daemon=True,
            ),
            threading.Thread(
                target=self._display_loop,
                name="video-display",
                daemon=True,
            ),
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
                    frame = self.camera.read_cv2_image(
                        strategy="newest"
                    )

            except Exception as e:
                print("camera warning:", repr(e))
                time.sleep(0.05)
                continue

            if frame is None:
                continue

            # 当前 PyAV 解码链路已经验证可直接作为 BGR 使用
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

            wait = min_period - (now - last_run)

            if wait > 0:
                time.sleep(min(wait, 0.01))
                continue

            frame, frame_seq = self._get_frame()

            if frame is None or frame_seq == last_frame_seq:
                time.sleep(0.005)
                continue

            last_frame_seq = frame_seq
            last_run = time.monotonic()

            target = self.get_target()

            try:
                best, detections = detect_one_frame(
                    model=self.model,
                    frame_bgr=frame,
                    target=target,
                    use_cuda=self.use_cuda,
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
            best = (
                None
                if self.latest_best is None
                else dict(self.latest_best)
            )

            detections = [
                dict(x)
                for x in self.latest_detections
            ]

            return (
                self.det_seq,
                self.det_time,
                best,
                detections,
            )

    def current_det_seq(self):
        with self.det_lock:
            return self.det_seq

    def _display_loop(self):
        period = 1.0 / self.display_fps

        window_name = "EXP3 Grasp V9"

        try:
            cv2.namedWindow(
                window_name,
                cv2.WINDOW_NORMAL,
            )

            cv2.resizeWindow(
                window_name,
                960,
                540,
            )

            while not self.stop_event.is_set():
                loop_start = time.monotonic()

                frame, _ = self._get_frame()

                if frame is not None:
                    (
                        _,
                        det_time,
                        best,
                        detections,
                    ) = self.snapshot()

                    with self.state_lock:
                        status = self.status
                        profile = (
                            None
                            if self.profile is None
                            else dict(self.profile)
                        )
                        target = self.target

                    shown = frame.copy()
                    h, w = shown.shape[:2]

                    for det in detections:
                        x1, y1, x2, y2 = [
                            int(round(v))
                            for v in det["box"]
                        ]

                        bx = int(round(det["bottom_x"]))
                        by = int(round(det["bottom_y"]))

                        is_best = (
                            best is not None
                            and det == best
                        )

                        det_age_sec = (
                            max(0.0, time.monotonic() - det_time)
                            if det_time > 0
                            else 999.0
                        )

                        if det_age_sec > MAX_DET_AGE_S:
                            color = (120, 120, 120)
                        elif is_best:
                            color = (0, 255, 0)
                        else:
                            color = (0, 180, 255)

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
                            f"{det['class']} "
                            f"{det['conf']:.2f} "
                            f"x={det['bottom_x']:.0f} "
                            f"y={det['bottom_y']:.0f} "
                            f"w={det['width']:.0f} "
                            f"h={det['height']:.0f}"
                        )

                        cv2.putText(
                            shown,
                            text,
                            (
                                x1,
                                max(25, y1 - 8),
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.46,
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

                        if profile["target"] == "bottle":
                            ref_size = profile["bbox_height_px"]
                            ref_name = "h"
                        else:
                            ref_size = profile["bbox_width_px"]
                            ref_name = "w"

                        cv2.putText(
                            shown,
                            (
                                f"LEARNED x={rx} y={ry} "
                                f"{ref_name}={ref_size:.0f}"
                            ),
                            (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.47,
                            (255, 0, 255),
                            1,
                        )

                    imgsz, conf = infer_config(target)

                    cv2.rectangle(
                        shown,
                        (0, 0),
                        (w, 32),
                        (0, 0, 0),
                        -1,
                    )

                    det_age = 0.0

                    if det_time > 0:
                        det_age = (
                            time.monotonic() - det_time
                        ) * 1000.0

                    line = (
                        f"{status} | target={target} "
                        f"YOLO={imgsz}/{conf:.2f} | "
                        f"VIDEO {self.capture_meter.value():.1f} "
                        f"INF {self.infer_meter.value():.1f} "
                        f"DISP {self.display_meter.value():.1f} "
                        f"age {det_age:.0f}ms"
                    )

                    cv2.putText(
                        shown,
                        line,
                        (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (255, 255, 255),
                        1,
                    )

                    cv2.imshow(
                        window_name,
                        shown,
                    )

                    self.display_meter.tick()

                key = cv2.waitKey(1) & 0xFF

                if key in (
                    ord("q"),
                    ord("Q"),
                    27,
                ):
                    self.user_quit = True
                    self.stop_event.set()
                    break

                used = time.monotonic() - loop_start
                remain = period - used

                if remain > 0:
                    time.sleep(remain)

        except Exception as e:
            print("display warning:", repr(e))
            self.stop_event.set()

        finally:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass


def _priority_pick(detections, previous=None):
    """
    从当前画面的所有检测框中选择“当前要抓的目标”。

    首次：
        优先 bottom_y 最大，也就是视觉上最靠近机器人/画面底部的物体。

    已经开始跟踪：
        优先继续跟踪与上一帧位置相近的同一物体，避免多个同类框之间跳来跳去。
        如果找不到相近目标，再退回最近目标规则。
    """
    if not detections:
        return None

    # 已经锁定一个目标时，先做简单位置连续跟踪
    if previous is not None:
        same_class = [
            d for d in detections
            if d["class"] == previous["class"]
        ]

        if same_class:
            ranked = []
            for d in same_class:
                dx = abs(d["bottom_x"] - previous["bottom_x"])
                dy = abs(d["bottom_y"] - previous["bottom_y"])

                # 优先像素位置连续；同时轻微偏向 bottom_y 更大的目标
                continuity_cost = dx + 0.7 * dy - 0.08 * d["bottom_y"]
                ranked.append((continuity_cost, d))

            ranked.sort(key=lambda x: x[0])

            # 目标移动主要来自机器人自身运动，通常不会瞬间跳太远。
            best_cost, best_det = ranked[0]

            if (
                abs(best_det["bottom_x"] - previous["bottom_x"]) <= 140
                and abs(best_det["bottom_y"] - previous["bottom_y"]) <= 120
            ):
                return best_det

    # 首次选择/跟踪丢失：直接选最近目标
    return max(
        detections,
        key=lambda d: (
            d["bottom_y"],
            d["width"] * d["height"],
            d["conf"],
        ),
    )


def locate_stable(
    vision,
    timeout=TARGET_TIMEOUT,
    window_size=STABLE_WINDOW,
    required_hits=STABLE_HITS,
):
    """
    多目标稳定选择。

    与旧版不同：
    - 不再只看“最高置信度框”
    - 每一帧先从所有检测框中选择最近候选
    - 一旦开始跟踪，尽量保持同一个物体，避免多个同类目标间跳框
    - 最近 window_size 次里，至少 required_hits 次跟踪成功即返回中位数位置
    """
    start = time.monotonic()
    last_seq = vision.current_det_seq()

    history = deque(maxlen=window_size)
    previous = None

    while time.monotonic() - start < timeout:
        if vision.is_stopped():
            raise KeyboardInterrupt

        seq, _, _, detections = vision.snapshot()

        if seq <= last_seq:
            time.sleep(0.01)
            continue

        last_seq = seq

        candidate = _priority_pick(
            detections,
            previous=previous,
        )

        history.append(candidate)

        if candidate is not None:
            previous = candidate

        valid = [
            det
            for det in history
            if det is not None
        ]

        if len(valid) < required_hits:
            continue

        counts = Counter(
            det["class"]
            for det in valid
        )

        cls, hits = counts.most_common(1)[0]

        if hits < required_hits:
            continue

        same = [
            det
            for det in valid
            if det["class"] == cls
        ]

        return {
            "class": cls,
            "conf": float(
                np.median([d["conf"] for d in same])
            ),
            "bottom_x": float(
                np.median([d["bottom_x"] for d in same])
            ),
            "bottom_y": float(
                np.median([d["bottom_y"] for d in same])
            ),
            "width": float(
                np.median([d["width"] for d in same])
            ),
            "height": float(
                np.median([d["height"] for d in same])
            ),
        }

    return None




def get_latest_target_detection(
    vision,
    class_name,
    ref_x=None,
):
    """
    直接读取当前最新检测结果。

    不再要求 seq 必须递增，也不再等待 after_seq。
    只保留两个条件：
    1. 类别必须匹配；
    2. 检测结果年龄 <= MAX_DET_AGE_S。

    这样可以避免“画面已有绿色框，但控制线程因为等新 seq 而不动作”。
    """
    _, det_time, _, detections = vision.snapshot()

    if det_time <= 0:
        return None

    age = time.monotonic() - det_time

    if age > MAX_DET_AGE_S:
        return None

    same = [
        d
        for d in detections
        if d["class"] == class_name
    ]

    if not same:
        return None

    # 低头以后，目标在切换前已经完成横向对准。
    # 因此若同类有多个框，优先选最接近学习中心 ref_x 的那个，
    # 避免重新跳到旁边另一个同类物体。
    if ref_x is not None:
        det = min(
            same,
            key=lambda d: (
                abs(d["bottom_x"] - ref_x),
                -d["bottom_y"],
                -(d["width"] * d["height"]),
                -d["conf"],
            ),
        )
    else:
        det = max(
            same,
            key=lambda d: (
                d["bottom_y"],
                d["width"] * d["height"],
                d["conf"],
            ),
        )

    return {
        "class": det["class"],
        "conf": float(det["conf"]),
        "bottom_x": float(det["bottom_x"]),
        "bottom_y": float(det["bottom_y"]),
        "width": float(det["width"]),
        "height": float(det["height"]),
        "det_age": float(age),
    }


def close_approach(
    chassis,
    vision,
    profile,
    args,
):
    """
    机械臂低头后的实时近距离闭环。

    关键变化：
    - 不等待新的 seq。
    - 不使用 after_seq / last_seq。
    - 每个循环直接读取当前最新检测框。
    - 只要检测框年龄 <= MAX_DET_AGE_S，就立即计算并执行动作。
    - 如果短时间内暂时没有有效框，仅等待；超过 CLOSE_LOST_TIMEOUT_S 才恢复。
    """
    cls = profile["target"]

    ref_x = float(profile["bottom_x_px"])
    ref_y = float(profile["bottom_y_px"])

    if cls == "bottle":
        ref_size = float(profile["bbox_height_px"])
        size_name = "height"
    else:
        ref_size = float(profile["bbox_width_px"])
        size_name = "width"

    lost_since = None
    step = 0

    while step < CLOSE_MAX_STEPS:
        if vision.is_stopped():
            raise KeyboardInterrupt

        vision.set_status(
            f"CLOSE_APPROACH {step + 1}/{CLOSE_MAX_STEPS}"
        )

        loc = get_latest_target_detection(
            vision=vision,
            class_name=cls,
            ref_x=ref_x,
        )

        if loc is None:
            if lost_since is None:
                lost_since = time.monotonic()

            lost_time = time.monotonic() - lost_since

            print(
                f"[CLOSE] 暂无新鲜 {cls} 检测，"
                f"等待中 {lost_time:.1f}s"
            )

            # 没有新鲜目标时确保底盘停止
            try:
                chassis.drive_speed(
                    x=0,
                    y=0,
                    z=0,
                    timeout=0.5,
                )
            except Exception:
                pass

            if lost_time >= CLOSE_LOST_TIMEOUT_S:
                print(
                    "[CLOSE] 目标丢失超过限制，退出近距离闭环。"
                )
                return None

            time.sleep(CLOSE_LOOP_PERIOD_S)
            continue

        # 当前有新鲜框
        lost_since = None
        step += 1

        x_error = loc["bottom_x"] - ref_x
        y_error = loc["bottom_y"] - ref_y

        if cls == "bottle":
            current_size = float(loc["height"])
        else:
            current_size = float(loc["width"])

        size_ratio = current_size / ref_size

        print()
        print(
            f"[CLOSE {step}] {cls} conf={loc['conf']:.2f} "
            f"age={loc['det_age']*1000:.0f}ms"
        )
        print(
            f"  x={loc['bottom_x']:.1f}, "
            f"ref={ref_x:.1f}, err={x_error:+.1f}px"
        )
        print(
            f"  y={loc['bottom_y']:.1f}, "
            f"ref={ref_y:.1f}, err={y_error:+.1f}px"
        )
        print(
            f"  {size_name}={current_size:.1f}, "
            f"ref={ref_size:.1f}, ratio={size_ratio:.3f}"
        )

        # -----------------------------
        # 横向修正
        # -----------------------------
        x_tol = (
            BOTTLE_X_TOL_PX
            if cls == "bottle"
            else args.x_tol
        )

        if abs(x_error) > x_tol:
            z = turn_step_from_x_error(x_error)
            direction = "右" if x_error > 0 else "左"

            vision.set_status(
                f"CLOSE_APPROACH / TURN {direction}"
            )

            print(
                f"  ACTION: TURN {direction} {z:+.1f}deg"
            )

            if args.dry_run:
                return loc

            chassis.move(
                x=0,
                y=0,
                z=float(z),
                z_speed=CHASSIS_Z_SPEED,
            ).wait_for_completed()

            time.sleep(0.18)
            continue

        # -----------------------------
        # 水瓶精确近距逻辑
        # -----------------------------
        if cls == "bottle":
            h = float(loc["height"])
            bottom_y = float(loc["bottom_y"])

            ready = (
                abs(x_error) <= BOTTLE_X_TOL_PX
                and BOTTLE_MIN_HEIGHT <= h <= BOTTLE_MAX_HEIGHT
                and bottom_y >= BOTTLE_MIN_BOTTOM_Y
            )

            if ready:
                print()
                print("==============================================")
                print("[READY] 水瓶已达到成功抓取视觉状态")
                print(
                    f"x_error={x_error:+.1f}px, "
                    f"height={h:.1f}px, "
                    f"bottom_y={bottom_y:.1f}px"
                )
                print("==============================================")

                try:
                    chassis.drive_speed(
                        x=0,
                        y=0,
                        z=0,
                        timeout=0.5,
                    )
                except Exception:
                    pass

                vision.set_status("READY_TO_GRASP")
                return loc

            if h < BOTTLE_FAR_H:
                dx = MAX_MOVE_M
            elif h < BOTTLE_MID_H:
                dx = MID_MOVE_M
            elif h < BOTTLE_NEAR_H:
                dx = MIN_MOVE_M
            elif bottom_y < BOTTLE_MIN_BOTTOM_Y:
                dx = MIN_MOVE_M
            elif h > BOTTLE_MAX_HEIGHT:
                dx = -MIN_MOVE_M
            else:
                dx = (
                    MIN_MOVE_M
                    if h < ref_size
                    else -MIN_MOVE_M
                )

            direction = "前进" if dx > 0 else "后退"

            vision.set_status(
                f"CLOSE_APPROACH / {direction}"
            )

            print(
                f"  ACTION: {direction} {abs(dx)*1000:.0f}mm "
                f"(h={h:.1f}, bottom_y={bottom_y:.1f})"
            )

            if args.dry_run:
                return loc

            chassis.move(
                x=float(dx),
                y=0,
                z=0,
                xy_speed=CHASSIS_XY_SPEED,
            ).wait_for_completed()

            time.sleep(0.18)
            continue

        # -----------------------------
        # 网球精确近距逻辑
        # -----------------------------
        ball_ready = (
            abs(x_error) <= args.x_tol
            and abs(size_ratio - 1.0) <= SIZE_TOL_RATIO
            and abs(y_error) <= args.ball_y_tol
        )

        if ball_ready:
            print()
            print("==============================================")
            print("[READY] 网球已达到成功抓取视觉状态")
            print(
                f"x_error={x_error:+.1f}px, "
                f"width_ratio={size_ratio:.3f}, "
                f"y_error={y_error:+.1f}px"
            )
            print("==============================================")

            try:
                chassis.drive_speed(
                    x=0,
                    y=0,
                    z=0,
                    timeout=0.5,
                )
            except Exception:
                pass

            vision.set_status("READY_TO_GRASP")
            return loc

        dx = move_step_from_size_ratio(
            size_ratio
        )

        if dx == 0.0:
            if abs(y_error) > args.ball_y_tol:
                dx = (
                    MIN_MOVE_M
                    if y_error < 0
                    else -MIN_MOVE_M
                )
            else:
                dx = (
                    MIN_MOVE_M
                    if size_ratio < 1.0
                    else -MIN_MOVE_M
                )

        direction = "前进" if dx > 0 else "后退"

        vision.set_status(
            f"CLOSE_APPROACH / {direction}"
        )

        print(
            f"  ACTION: {direction} {abs(dx)*1000:.0f}mm"
        )

        if args.dry_run:
            return loc

        chassis.move(
            x=float(dx),
            y=0,
            z=0,
            xy_speed=CHASSIS_XY_SPEED,
        ).wait_for_completed()

        time.sleep(0.18)

    print("CLOSE_APPROACH 达到最大控制次数，停止。")
    return None



def _wait_arm_action(action, label):
    """
    RoboMaster Action.wait_for_completed(timeout) 支持超时返回。
    这里绝不允许机械臂 ACK 把整个视觉状态机永久卡住。
    """
    try:
        ok = action.wait_for_completed(
            timeout=ARM_ACTION_TIMEOUT_S
        )
    except TypeError:
        # 极端兼容：如果某个 SDK 版本不接受关键字参数
        ok = action.wait_for_completed(
            ARM_ACTION_TIMEOUT_S
        )

    if ok:
        print(f"[ARM] {label} completed")
        return True

    print(
        f"[ARM] WARNING: {label} wait_for_completed "
        f"超过 {ARM_ACTION_TIMEOUT_S:.1f}s。"
    )
    print(
        "[ARM] 如果机械臂已经物理到位，继续进入视觉闭环，"
        "不再永久阻塞。"
    )

    time.sleep(ARM_SETTLE_AFTER_TIMEOUT_S)
    return False


def move_arm_to_profile_pose(
    arm,
    profile,
    label,
):
    print(
        f"{label}: "
        f"arm.moveto("
        f"x={profile['arm_x_mm']:.1f}, "
        f"y={profile['arm_y_mm']:.1f})"
    )

    action = arm.moveto(
        x=float(profile["arm_x_mm"]),
        y=float(profile["arm_y_mm"]),
    )

    _wait_arm_action(
        action,
        label,
    )

    # 给相机画面一点稳定时间，但不再无限等ACK
    time.sleep(0.35)


def move_arm_to_observe_pose(arm):
    """
    远距离观察姿态。
    recenter 同样加入超时，避免恢复观察姿态时卡死。
    """
    print("机械臂 -> 远距离观察姿态 (recenter)")

    action = arm.recenter()

    _wait_arm_action(
        action,
        "recenter",
    )

    time.sleep(0.35)



def observe_size(loc):
    """
    观察姿态下用于判断远近的尺寸：
    网球用 bbox width，水瓶用 bbox height。
    """
    if loc["class"] == "bottle":
        return float(loc["height"]), "height"
    return float(loc["width"]), "width"


def pregrasp_threshold(class_name):
    if class_name == "bottle":
        return BOTTLE_PREGRASP_HEIGHT
    return BALL_PREGRASP_WIDTH


def observe_move_step(class_name, size_value):
    """
    远距离观察阶段的前进步长。
    这里不追求最终精确距离，只负责把目标送进 PREGRASP 范围。
    """
    threshold = pregrasp_threshold(class_name)
    ratio = size_value / threshold

    if ratio < 0.55:
        return OBSERVE_FAR_MOVE_M
    if ratio < 0.78:
        return OBSERVE_MID_MOVE_M
    return OBSERVE_NEAR_MOVE_M



def clamp_step(current, target, max_step):
    """
    限制每个周期的速度变化量，避免突然加速/急停。
    """
    delta = target - current

    if delta > max_step:
        delta = max_step
    elif delta < -max_step:
        delta = -max_step

    return current + delta


def stop_chassis_smooth(chassis, current_x=0.0, current_z=0.0):
    """
    平滑减速到 0。
    """
    x = float(current_x)
    z = float(current_z)

    for _ in range(6):
        x = clamp_step(x, 0.0, MAX_DV_MPS)
        z = clamp_step(z, 0.0, MAX_DZ_DPS)

        chassis.drive_speed(
            x=float(x),
            y=0,
            z=float(z),
            timeout=0.5,
        )

        if abs(x) < 0.01 and abs(z) < 1.0:
            break

        time.sleep(CONTROL_PERIOD_S)

    chassis.drive_speed(
        x=0,
        y=0,
        z=0,
        timeout=0.5,
    )

    time.sleep(0.15)


def desired_approach_speed(class_name, size_value):
    """
    根据目标相对“进入 PREGRASP 阈值”的接近程度设置前进速度。
    目标越接近阈值，速度越低。
    """
    threshold = pregrasp_threshold(class_name)

    if threshold <= 0:
        return APPROACH_SLOW_MPS

    ratio = size_value / threshold

    if ratio < 0.55:
        return APPROACH_FAST_MPS

    if ratio < PREGRASP_SLOW_RATIO:
        return APPROACH_MID_MPS

    return APPROACH_SLOW_MPS


def desired_turn_speed(x_error):
    """
    根据横向像素误差生成连续转向角速度。
    """
    a = abs(x_error)

    if a <= OBSERVE_X_TOL_PX:
        return 0.0

    if a > 100:
        mag = TURN_FAST_DPS
    elif a > 55:
        mag = TURN_MID_DPS
    else:
        mag = TURN_SLOW_DPS

    # drive_speed() 的 z 符号按实机验证：
    # 左边 x_error<0 -> z<0
    # 右边 x_error>0 -> z>0
    return TURN_SIGN_DRIVE * math.copysign(mag, x_error)



def approach_in_observe_pose(
    arm,
    chassis,
    vision,
    profiles,
    args,
):
    """
    远距离连续靠近 + ALIGN_PREGRASP。

    主线程是唯一的底盘/机械臂控制者；AsyncVision 三个线程只负责
    相机读取、YOLO推理和显示，不会发送任何运动命令，因此这里没有
    与视觉线程争抢 chassis/arm 的控制冲突。

    状态：
        APPROACH
            -> 目标接近切换阈值的 85%
        ALIGN_PREGRASP
            -> 停止高速前进，先把目标转到画面中央
            -> 若已经居中但距离还差一点，只以 0.04m/s 慢慢向前
            -> “距离达到阈值 + |x_error|<=30px” 连续2个新YOLO结果
        PREGRASP
            -> 平滑停车
            -> 才允许机械臂低头

    这样绝不会再出现 err=-200~-300px 但因为 bbox 够大就直接低头。
    """
    move_arm_to_observe_pose(arm)

    vision.set_target(args.target)
    vision.set_profile(None)
    vision.set_status("OBSERVE / APPROACH")

    locked_class = None
    previous_det = None

    current_x = 0.0
    current_z = 0.0

    last_det_seq = -1
    last_seen_time = time.monotonic()

    align_mode = False
    align_hits = 0
    step = 0

    try:
        while step < args.observe_max_steps:
            if vision.is_stopped():
                raise KeyboardInterrupt

            seq, det_time, _, detections = vision.snapshot()

            # 只在有新的YOLO结果时更新目标状态；运动速度命令本身可持续。
            if seq == last_det_seq:
                if time.monotonic() - last_seen_time > OBSERVE_LOST_TIMEOUT_S:
                    print("[APPROACH] 连续超过2秒没有新的有效目标，停止并恢复。")
                    stop_chassis_smooth(chassis, current_x, current_z)
                    return None, None, None
                time.sleep(0.02)
                continue

            last_det_seq = seq

            # 检测结果过旧，不参与控制。
            if det_time <= 0 or (time.monotonic() - det_time) > MAX_DET_AGE_S:
                continue

            # 初次从全部目标选最近；锁定后持续跟踪同一类别、相邻位置的框。
            if locked_class is None:
                candidate = _priority_pick(detections, previous=None)
            else:
                same = [d for d in detections if d["class"] == locked_class]
                candidate = _priority_pick(same, previous=previous_det)

            if candidate is None:
                if time.monotonic() - last_seen_time > OBSERVE_LOST_TIMEOUT_S:
                    print("[APPROACH] 锁定目标丢失超过2秒，停止并恢复。")
                    stop_chassis_smooth(chassis, current_x, current_z)
                    return None, None, None
                continue

            if locked_class is None:
                locked_class = candidate["class"]
                if locked_class not in profiles:
                    stop_chassis_smooth(chassis, current_x, current_z)
                    return None, None, None
                vision.set_target(locked_class)
                print(
                    f"[SMOOTH] 锁定最近目标: {locked_class}，"
                    "后续持续跟踪该物体"
                )

            if candidate["class"] != locked_class:
                continue

            previous_det = candidate
            last_seen_time = time.monotonic()
            step += 1

            profile = profiles[locked_class]

            loc = {
                "class": candidate["class"],
                "conf": float(candidate["conf"]),
                "bottom_x": float(candidate["bottom_x"]),
                "bottom_y": float(candidate["bottom_y"]),
                "width": float(candidate["width"]),
                "height": float(candidate["height"]),
            }

            x_error = loc["bottom_x"] - OBSERVE_CENTER_X
            size_value, size_name = observe_size(loc)
            switch_threshold = pregrasp_threshold(locked_class)
            size_ratio_to_switch = size_value / switch_threshold

            print()
            print(
                f"[SMOOTH {step}] {locked_class} conf={loc['conf']:.2f} | "
                f"x={loc['bottom_x']:.1f} err={x_error:+.1f}px | "
                f"{size_name}={size_value:.1f} switch={switch_threshold:.1f}"
            )

            # --------------------------------------------------
            # 进入 ALIGN_PREGRASP：接近阈值后先对准，不准直接低头
            # --------------------------------------------------
            if (
                not align_mode
                and size_ratio_to_switch >= PREGRASP_ALIGN_ENTER_RATIO
            ):
                align_mode = True
                align_hits = 0

                print(
                    "[ALIGN_PREGRASP] 已接近预抓取区域，"
                    "先停止前冲并横向对准。"
                )

                stop_chassis_smooth(
                    chassis,
                    current_x=current_x,
                    current_z=current_z,
                )
                current_x = 0.0
                current_z = 0.0

            if align_mode:
                vision.set_status(
                    f"ALIGN_PREGRASP {align_hits}/{PREGRASP_ALIGN_REQUIRED_HITS}"
                )

                # 若目标因为转向/检测波动明显又变远，退出ALIGN重新正常靠近。
                if size_ratio_to_switch < 0.72:
                    print("[ALIGN_PREGRASP] 目标又明显变远，返回 APPROACH。")
                    align_mode = False
                    align_hits = 0
                    continue

                # 先横向对准。偏差大时绝不向前。
                if abs(x_error) > PREGRASP_ALIGN_X_TOL_PX:
                    align_hits = 0
                    target_x = 0.0
                    target_z = desired_turn_speed(x_error)
                    current_x = 0.0
                    current_z = clamp_step(
                        current_z,
                        target_z,
                        MAX_DZ_DPS,
                    )

                    direction = "右" if x_error > 0 else "左"
                    print(
                        f"  ALIGN: 目标偏{direction} {x_error:+.1f}px，"
                        f"原地转向 z={current_z:+.1f}deg/s，不前进"
                    )

                    if not args.dry_run:
                        chassis.drive_speed(
                            x=0.0,
                            y=0,
                            z=float(current_z),
                            timeout=0.5,
                        )
                    time.sleep(CONTROL_PERIOD_S)
                    continue

                # 横向已经基本对准。
                current_z = clamp_step(
                    current_z,
                    0.0,
                    MAX_DZ_DPS,
                )

                if size_value < switch_threshold:
                    # 还差一点距离：只允许极慢直行，不再高速冲上去。
                    align_hits = 0
                    current_x = clamp_step(
                        current_x,
                        PREGRASP_CREEP_MPS,
                        MAX_DV_MPS,
                    )
                    print(
                        f"  ALIGN: 已居中但距离还差一点，"
                        f"仅慢速前进 x={current_x:.2f}m/s"
                    )

                    if not args.dry_run:
                        chassis.drive_speed(
                            x=float(current_x),
                            y=0,
                            z=float(current_z),
                            timeout=0.5,
                        )
                    time.sleep(CONTROL_PERIOD_S)
                    continue

                # 同时满足：距离够近 + 横向居中。
                align_hits += 1
                current_x = 0.0

                print(
                    f"  ALIGN READY {align_hits}/{PREGRASP_ALIGN_REQUIRED_HITS}: "
                    f"|x_error|={abs(x_error):.1f}px <= {PREGRASP_ALIGN_X_TOL_PX:.1f}, "
                    f"{size_name}={size_value:.1f} >= {switch_threshold:.1f}"
                )

                # 要求连续两个“新的YOLO结果”都满足，防止单帧bbox抖动误触发。
                if align_hits < PREGRASP_ALIGN_REQUIRED_HITS:
                    if not args.dry_run:
                        chassis.drive_speed(
                            x=0,
                            y=0,
                            z=float(current_z),
                            timeout=0.5,
                        )
                    time.sleep(CONTROL_PERIOD_S)
                    continue

                print()
                print("==============================================")
                print("[PREGRASP] 距离和横向位置均已确认")
                print(
                    f"class={locked_class}, x_error={x_error:+.1f}px, "
                    f"{size_name}={size_value:.1f}"
                )
                print("现在才允许停车并切换学习抓取姿态。")
                print("==============================================")

                stop_chassis_smooth(
                    chassis,
                    current_x=current_x,
                    current_z=current_z,
                )
                return locked_class, profile, loc

            # --------------------------------------------------
            # 正常远距离 APPROACH
            # --------------------------------------------------
            vision.set_status(
                f"SMOOTH APPROACH {step}/{args.observe_max_steps}"
            )

            target_x = desired_approach_speed(
                locked_class,
                size_value,
            )

            abs_x_err = abs(x_error)

            # 大角度偏离时先转，不允许一边严重偏航一边继续前冲。
            if abs_x_err > 120:
                target_x = 0.0
            elif abs_x_err > 70:
                target_x = min(target_x, APPROACH_SLOW_MPS)

            target_z = desired_turn_speed(x_error)

            current_x = clamp_step(
                current_x,
                target_x,
                MAX_DV_MPS,
            )
            current_z = clamp_step(
                current_z,
                target_z,
                MAX_DZ_DPS,
            )

            print(
                f"  drive_speed: x={current_x:.2f}m/s "
                f"z={current_z:+.1f}deg/s"
            )

            if args.dry_run:
                stop_chassis_smooth(chassis, current_x, current_z)
                return locked_class, profile, loc

            chassis.drive_speed(
                x=float(current_x),
                y=0,
                z=float(current_z),
                timeout=0.5,
            )

            time.sleep(CONTROL_PERIOD_S)

        print("连续观察/靠近阶段达到最大控制步数。")
        stop_chassis_smooth(chassis, current_x, current_z)
        return None, None, None

    except Exception:
        try:
            stop_chassis_smooth(chassis, current_x, current_z)
        except Exception:
            pass
        raise



def pregrasp_with_recovery(
    arm,
    chassis,
    vision,
    profiles,
    args,
):
    """
    状态机：

        OBSERVE / SMOOTH_APPROACH
            ↓
        PREGRASP
            ↓
        机械臂下降
            ↓
        直接进入 CLOSE_APPROACH
            ↓
        实时读取 latest detection
            ↓
        TURN / FORWARD / BACKWARD / READY

    不再使用 seq 门槛。
    """
    for attempt in range(
        PREGRASP_RECOVERY_LIMIT + 1
    ):
        cls, profile, _ = approach_in_observe_pose(
            arm=arm,
            chassis=chassis,
            vision=vision,
            profiles=profiles,
            args=args,
        )

        if cls is None:
            return None, None

        if args.dry_run:
            return cls, profile

        vision.set_target(cls)
        vision.set_profile(profile)
        vision.set_status("PREGRASP / MOVE ARM")

        move_arm_to_profile_pose(
            arm,
            profile,
            "进入学习抓取姿态",
        )

        print()
        print(
            "[PREGRASP] arm.moveto 阻塞已释放。"
        )
        print(
            "[PREGRASP] 机械臂已下降/到位或已超时放行，"
            "直接进入实时 CLOSE_APPROACH。"
        )

        vision.set_status("CLOSE_APPROACH")

        loc = close_approach(
            chassis=chassis,
            vision=vision,
            profile=profile,
            args=args,
        )

        if loc is not None:
            return cls, profile

        if attempt < PREGRASP_RECOVERY_LIMIT:
            print()
            print(
                "[RECOVERY] CLOSE_APPROACH 未完成，"
                "抬回观察姿态重新搜索。"
            )

            move_arm_to_observe_pose(arm)
            vision.set_profile(None)
            vision.set_target(args.target)

            time.sleep(0.35)
            continue

        return None, None

    return None, None


def turn_step_from_x_error(x_error):
    a = abs(x_error)

    if a > 90:
        mag = LARGE_TURN_DEG

    elif a > 45:
        mag = MID_TURN_DEG

    else:
        mag = SMALL_TURN_DEG

    # 注意：这里给 chassis.move(z=...) 使用。
    # 当前实机上 move() 与 drive_speed() 的转向符号相反：
    # 左边 x_error<0 -> z>0
    # 右边 x_error>0 -> z<0
    return (
        TURN_SIGN_MOVE
        * math.copysign(
            mag,
            x_error,
        )
    )


def move_step_from_size_ratio(ratio):
    """
    ratio = 当前视觉尺寸 / 学习视觉尺寸

    ratio < 1 -> 目标比学习状态小 -> 目标更远 -> 前进
    ratio > 1 -> 目标比学习状态大 -> 目标更近 -> 后退
    """
    if ratio < 0.72:
        return MAX_MOVE_M

    if ratio < 0.86:
        return MID_MOVE_M

    if ratio < (1.0 - SIZE_TOL_RATIO):
        return MIN_MOVE_M

    if ratio > 1.35:
        return -MAX_MOVE_M

    if ratio > 1.18:
        return -MID_MOVE_M

    if ratio > (1.0 + SIZE_TOL_RATIO):
        return -MIN_MOVE_M

    return 0.0


def visual_servo(
    chassis,
    vision,
    profile,
    args,
):
    cls = profile["target"]
    ref_x = float(profile["bottom_x_px"])
    ref_y = float(profile["bottom_y_px"])

    if cls == "bottle":
        ref_size = float(profile["bbox_height_px"])
        size_name = "height"
    else:
        ref_size = float(profile["bbox_width_px"])
        size_name = "width"

    for step in range(1, args.max_steps + 1):
        vision.set_status(f"SERVO {step}/{args.max_steps}")

        loc = locate_stable(
            vision,
            timeout=args.target_timeout,
            window_size=args.stable_window,
            required_hits=args.stable_hits,
        )

        if loc is None:
            print("没有稳定识别到目标，停止。")
            return None

        x_error = loc["bottom_x"] - ref_x
        y_error = loc["bottom_y"] - ref_y
        current_size = float(loc["height"] if cls == "bottle" else loc["width"])
        size_ratio = current_size / ref_size

        print()
        print(f"[SERVO {step}] {loc['class']} conf={loc['conf']:.2f}")
        print(f"  X: current={loc['bottom_x']:.1f} ref={ref_x:.1f} err={x_error:+.1f}px")
        print(f"  Y: current={loc['bottom_y']:.1f} ref={ref_y:.1f} err={y_error:+.1f}px")
        print(f"  {size_name}: current={current_size:.1f} ref={ref_size:.1f} ratio={size_ratio:.3f}")

        x_tol = BOTTLE_X_TOL_PX if cls == "bottle" else args.x_tol

        # 先横向对准
        if abs(x_error) > x_tol:
            z = turn_step_from_x_error(x_error)
            direction = "右" if x_error > 0 else "左"
            print(f"  目标在画面{direction}侧 -> 转向 {z:+.1f}deg")

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

        # 水瓶：严格复现成功抓取视觉状态
        if cls == "bottle":
            h = float(loc["height"])
            bottom_y = float(loc["bottom_y"])

            if h < BOTTLE_FAR_H:
                dx = MAX_MOVE_M
            elif h < BOTTLE_MID_H:
                dx = MID_MOVE_M
            elif h < BOTTLE_NEAR_H:
                dx = MIN_MOVE_M
            elif bottom_y < BOTTLE_MIN_BOTTOM_Y:
                dx = MIN_MOVE_M
            elif h > BOTTLE_MAX_HEIGHT:
                dx = -MIN_MOVE_M
            else:
                dx = 0.0

            if dx != 0.0:
                direction = "前进" if dx > 0 else "后退"
                print(
                    f"  水瓶精确距离校正: h={h:.1f}px, bottom_y={bottom_y:.1f}px "
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

            bottle_ready = (
                abs(x_error) <= BOTTLE_X_TOL_PX
                and BOTTLE_MIN_HEIGHT <= h <= BOTTLE_MAX_HEIGHT
                and bottom_y >= BOTTLE_MIN_BOTTOM_Y
            )

            if not bottle_ready:
                print("  水瓶尚未满足最终抓取条件，继续视觉闭环。")
                time.sleep(0.15)
                continue

            print()
            print("==============================================")
            print("[OK] 水瓶已严格复现成功抓取视觉状态")
            print(
                f"x_error={x_error:+.1f}px, "
                f"height={h:.1f}px, "
                f"bottom_y={bottom_y:.1f}px"
            )
            print("==============================================")
            vision.set_status("BOTTLE READY TO GRASP")
            return loc

        # 网球保持上一版逻辑
        dx = move_step_from_size_ratio(size_ratio)

        if dx != 0.0:
            direction = "前进" if dx > 0 else "后退"
            print(
                f"  距离视觉校正: {size_name} ratio={size_ratio:.3f} "
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

        if abs(y_error) > args.ball_y_tol:
            dx = MIN_MOVE_M if y_error < 0 else -MIN_MOVE_M
            direction = "前进" if dx > 0 else "后退"
            print(f"  网球 Y 误差仍为 {y_error:+.1f}px -> {direction} {abs(dx)*1000:.0f}mm")

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
        print("[OK] 网球已到学习得到的成功抓取视觉位置")
        print(
            f"x_error={x_error:+.1f}px, "
            f"width_ratio={size_ratio:.3f}, "
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
        print(
            "WARN: bottle 暂时沿用网球夹爪判定参数。"
        )


def close_and_lift(
    ep,
    arm,
    gripper,
    class_name,
    vision,
):
    configure_gripper(
        class_name
    )

    print(
        "关闭夹爪并判断..."
    )

    vision.set_status(
        "GRIPPER CLOSE"
    )

    ok, detail = exp2.close_and_judge(
        gripper
    )

    print(
        "grasp result:",
        ok,
        detail,
    )

    if not ok:
        try:
            exp2.robot_signal_error(ep)
        except Exception:
            pass

        vision.set_status(
            "GRASP FAILED"
        )

        return False

    try:
        exp2.robot_signal_success(ep)
    except Exception:
        pass

    vision.set_status(
        "GRASP SUCCESS / LIFT"
    )

    print(
        "抓取成功，开始抬起。"
    )

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

    print(
        "物体已抬起。"
    )

    return True




def _scan_detect_current_view(
    vision,
    timeout=SCAN_DETECT_TIMEOUT_S,
):
    """
    当前朝向下快速判断是否有目标。
    扫描阶段只要求较宽松的 2/5 命中，
    目的是决定“下一轮搜索应该朝哪个方向”，不是直接抓。
    """
    vision.set_target("auto")

    loc = locate_stable(
        vision,
        timeout=timeout,
        window_size=SCAN_WINDOW,
        required_hits=SCAN_HITS,
    )

    return loc


def return_search_and_scan(
    arm,
    chassis,
    vision,
):
    """
    每次分类放置后执行：

        1. 机械臂回高位观察姿态
        2. 向后退固定距离，恢复较宽视野
        3. 中间视角检测
        4. 左转约25°检测
        5. 右扫约50°到右侧检测
        6. 根据三个视角中“视觉上最近”的目标，
           最后把车朝向那个视角
        7. 下一轮从这个方向重新做最近目标选择

    注意：
    这里不是严格回放原路径，而是“恢复搜索视野 + 主动重观测”。
    这比抓完后停在靠前位置直接继续识别更稳。
    """
    print()
    print("============================================================")
    print("[RETURN_SEARCH] 恢复搜索位置并扫描剩余目标")
    print("============================================================")

    vision.set_status("RETURN_SEARCH / OBSERVE POSE")

    move_arm_to_observe_pose(arm)

    vision.set_target("auto")
    vision.set_profile(None)

    # 1) 后退，扩大视野
    vision.set_status("RETURN_SEARCH / BACK UP")

    print(
        f"[RETURN_SEARCH] 后退 {RETURN_BACK_M*100:.0f}cm，"
        "恢复更完整的目标视野"
    )

    chassis.move(
        x=-RETURN_BACK_M,
        y=0,
        z=0,
        xy_speed=RETURN_BACK_SPEED,
    ).wait_for_completed()

    time.sleep(0.35)

    candidates = []

    # 2) 中间视角
    vision.set_status("SCAN / CENTER")
    time.sleep(SCAN_SETTLE_S)

    center = _scan_detect_current_view(vision)

    if center is not None:
        candidates.append(("CENTER", 0.0, center))
        print(
            f"[SCAN] CENTER: {center['class']} "
            f"bottom_y={center['bottom_y']:.1f}"
        )
    else:
        print("[SCAN] CENTER: no target")

    # 3) 左侧视角
    vision.set_status("SCAN / LEFT")
    print(f"[SCAN] 左转 {SCAN_LEFT_DEG:.0f}deg")

    chassis.move(
        x=0,
        y=0,
        z=SCAN_LEFT_DEG,
        z_speed=SCAN_Z_SPEED,
    ).wait_for_completed()

    time.sleep(SCAN_SETTLE_S)

    left = _scan_detect_current_view(vision)

    if left is not None:
        candidates.append(("LEFT", SCAN_LEFT_DEG, left))
        print(
            f"[SCAN] LEFT: {left['class']} "
            f"bottom_y={left['bottom_y']:.1f}"
        )
    else:
        print("[SCAN] LEFT: no target")

    # 4) 从左侧扫到右侧：总共右转50度
    vision.set_status("SCAN / RIGHT")
    sweep_to_right = SCAN_RIGHT_DEG - SCAN_LEFT_DEG

    print(
        f"[SCAN] 从左侧扫到右侧 {sweep_to_right:+.0f}deg"
    )

    chassis.move(
        x=0,
        y=0,
        z=sweep_to_right,
        z_speed=SCAN_Z_SPEED,
    ).wait_for_completed()

    time.sleep(SCAN_SETTLE_S)

    right = _scan_detect_current_view(vision)

    if right is not None:
        candidates.append(("RIGHT", SCAN_RIGHT_DEG, right))
        print(
            f"[SCAN] RIGHT: {right['class']} "
            f"bottom_y={right['bottom_y']:.1f}"
        )
    else:
        print("[SCAN] RIGHT: no target")

    if not candidates:
        # 没找到目标，回正等待下一轮搜索
        print("[SCAN] 左/中/右都没发现稳定目标，回正。")

        chassis.move(
            x=0,
            y=0,
            z=-SCAN_RIGHT_DEG,
            z_speed=SCAN_Z_SPEED,
        ).wait_for_completed()

        time.sleep(0.30)

        vision.set_status("SCAN COMPLETE / NO TARGET")
        return None

    # 5) 三个视角里选择“视觉上最近”的目标
    # 主优先：bottom_y；次优先：目标视觉面积。
    best_view, best_angle, best_loc = max(
        candidates,
        key=lambda item: (
            item[2]["bottom_y"],
            item[2]["width"] * item[2]["height"],
            item[2]["conf"],
        ),
    )

    print()
    print(
        f"[SCAN] 下一轮推荐方向: {best_view}, "
        f"target={best_loc['class']}, "
        f"bottom_y={best_loc['bottom_y']:.1f}"
    )

    # 当前车头在 RIGHT(-25°)。
    # 转到所选视角的绝对相对角度。
    current_angle = SCAN_RIGHT_DEG
    correction = best_angle - current_angle

    if abs(correction) > 0.5:
        print(
            f"[SCAN] 调整到 {best_view} 视角: "
            f"{correction:+.1f}deg"
        )

        chassis.move(
            x=0,
            y=0,
            z=float(correction),
            z_speed=SCAN_Z_SPEED,
        ).wait_for_completed()

        time.sleep(0.30)

    vision.set_status(
        f"SCAN COMPLETE / NEXT {best_loc['class']}"
    )

    return best_loc



def sort_direction(class_name):
    """
    tennis_ball -> 左侧分类区
    bottle      -> 右侧分类区
    """
    if class_name == "tennis_ball":
        return TENNIS_DROP_TURN_DEG, "LEFT / 网球区"

    return BOTTLE_DROP_TURN_DEG, "RIGHT / 水瓶区"


def place_to_sort_side(
    arm,
    chassis,
    gripper,
    class_name,
    vision,
):
    """
    抓取成功后的分类动作：

        抓起
          ↓
        网球左转90° / 水瓶右转90°
          ↓
        前进到分类放置区
          ↓
        机械臂下降一点
          ↓
        松开夹爪
          ↓
        机械臂抬起
          ↓
        后退回原位置
          ↓
        转回原来的目标区域方向

    这样每次分拣后机器人仍能继续面对剩余目标。
    """
    turn_deg, zone_name = sort_direction(class_name)

    print()
    print("==============================================")
    print(f"[SORT] {class_name} -> {zone_name}")
    print("==============================================")

    vision.set_status(
        f"SORT {class_name} -> {zone_name}"
    )

    # 1. 转向对应分类区
    chassis.move(
        x=0,
        y=0,
        z=float(turn_deg),
        z_speed=DROP_Z_SPEED,
    ).wait_for_completed()

    time.sleep(0.35)

    # 2. 向分类区前进一点，避免物体落在机器人正旁边
    chassis.move(
        x=DROP_FORWARD_M,
        y=0,
        z=0,
        xy_speed=DROP_XY_SPEED,
    ).wait_for_completed()

    time.sleep(0.30)

    # 3. 把物体稍微放低
    try:
        _wait_arm_action(
            arm.move(
                x=0,
                y=-DROP_LOWER_MM,
            ),
            "sort lower",
        )
        time.sleep(0.30)
    except Exception as e:
        print("drop lower warning:", repr(e))

    # 4. 松开夹爪完成分类放置
    print(f"[SORT] 放置 {class_name}")
    gripper.open(power=DROP_OPEN_POWER)
    time.sleep(DROP_OPEN_TIME)

    # 5. 机械臂重新抬高，防止后退时碰到已放置物体
    try:
        _wait_arm_action(
            arm.move(
                x=0,
                y=DROP_LOWER_MM,
            ),
            "sort lift",
        )
        time.sleep(0.25)
    except Exception as e:
        print("drop lift warning:", repr(e))

    # 6. 退回放置前的位置
    chassis.move(
        x=-DROP_FORWARD_M,
        y=0,
        z=0,
        xy_speed=DROP_XY_SPEED,
    ).wait_for_completed()

    time.sleep(0.30)

    # 7. 转回原来面对目标区域的方向
    chassis.move(
        x=0,
        y=0,
        z=float(-turn_deg),
        z_speed=DROP_Z_SPEED,
    ).wait_for_completed()

    time.sleep(0.35)

    # 8. 夹爪打开；真正的“恢复视野 + 扫描”放到 RETURN_SEARCH 阶段
    gripper.open(power=OPEN_POWER)
    time.sleep(0.25)

    vision.set_target("auto")
    vision.set_profile(None)
    vision.set_status("SORT DONE / RETURN SEARCH")

    print("[SORT] 分类放置动作完成，准备恢复搜索视野。")



def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--target",
        choices=[
            "auto",
            "tennis_ball",
            "bottle",
        ],
        default="auto",
    )

    p.add_argument(
        "--model",
        default=MODEL_DEFAULT,
    )

    p.add_argument(
        "--tennis-profile",
        default=PROFILE_DEFAULTS[
            "tennis_ball"
        ],
    )

    p.add_argument(
        "--bottle-profile",
        default=PROFILE_DEFAULTS[
            "bottle"
        ],
    )

    p.add_argument(
        "--infer-fps",
        type=float,
        default=6.0,
    )

    p.add_argument(
        "--display-fps",
        type=float,
        default=30.0,
    )

    p.add_argument(
        "--stable-window",
        type=int,
        default=STABLE_WINDOW,
    )

    p.add_argument(
        "--stable-hits",
        type=int,
        default=STABLE_HITS,
    )

    p.add_argument(
        "--target-timeout",
        type=float,
        default=TARGET_TIMEOUT,
    )

    p.add_argument(
        "--auto-pose-timeout",
        type=float,
        default=3.0,
    )

    p.add_argument(
        "--x-tol",
        type=float,
        default=X_TOL_PX,
    )

    p.add_argument(
        "--ball-y-tol",
        type=float,
        default=BALL_Y_TOL_PX,
    )

    p.add_argument(
        "--max-steps",
        type=int,
        default=MAX_SERVO_STEPS,
    )

    p.add_argument(
        "--observe-max-steps",
        type=int,
        default=OBSERVE_MAX_STEPS,
    )

    p.add_argument(
        "--pregrasp-timeout",
        type=float,
        default=3.0,
    )

    p.add_argument(
        "--nav-only",
        action="store_true",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
    )

    p.add_argument(
        "--keep-object",
        action="store_true",
    )

    p.add_argument(
        "--sort-mode",
        action="store_true",
        help="连续抓取多个目标并按类别左右分拣",
    )

    p.add_argument(
        "--max-items",
        type=int,
        default=DEFAULT_MAX_ITEMS,
        help="本轮最多分拣多少个物体",
    )

    p.add_argument(
        "--empty-retries",
        type=int,
        default=EMPTY_RETRY_LIMIT,
        help="连续多少轮识别不到目标后结束分拣",
    )

    return p.parse_args()


def main():
    args = parse_args()

    if (
        args.stable_hits
        > args.stable_window
    ):
        print(
            "ERROR: stable-hits 不能大于 stable-window"
        )
        return 2

    if not os.path.isfile(
        args.model
    ):
        print(
            "ERROR: 模型不存在:",
            args.model,
        )
        return 2

    profile_paths = {
        "tennis_ball":
            args.tennis_profile,
        "bottle":
            args.bottle_profile,
    }

    profiles = {}

    for cls, path in profile_paths.items():
        if os.path.isfile(path):
            profiles[cls] = load_profile(
                path
            )

            p = profiles[cls]

            print()
            print(
                f"profile {cls}:"
            )

            print(
                f"  visual "
                f"x={p['bottom_x_px']:.1f} "
                f"y={p['bottom_y_px']:.1f}"
            )

            print(
                f"  bbox "
                f"w={p['bbox_width_px']:.1f} "
                f"h={p['bbox_height_px']:.1f}"
            )

            print(
                f"  arm "
                f"x={p['arm_x_mm']:.1f} "
                f"y={p['arm_y_mm']:.1f}"
            )

        else:
            print(
                "WARN: profile 不存在:",
                path,
            )

    if (
        args.target != "auto"
        and args.target not in profiles
    ):
        print(
            "ERROR: 指定类别没有对应 profile。"
        )
        return 2

    use_cuda = torch.cuda.is_available()

    print()
    print(
        "加载 YOLO..."
    )

    model = YOLO(
        args.model
    )

    # 预热 640，兼容 bottle 模式
    dummy = np.zeros(
        (IMG_H, IMG_W, 3),
        dtype=np.uint8,
    )

    model.predict(
        dummy,
        imgsz=YOLO_SIZE_BOTTLE,
        conf=CONF_BOTTLE,
        verbose=False,
        device=0 if use_cuda else "cpu",
        half=use_cuda,
    )

    print(
        "YOLO OK"
    )

    ep = robot.Robot()

    initialized = False
    camera_started = False
    subscribed = False

    vision = None
    gripper = None

    try:
        print(
            "连接 RoboMaster..."
        )

        ep.initialize(
            conn_type="ap"
        )

        initialized = True

        print(
            "RoboMaster SDK OK"
        )

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

        gripper.open(
            power=OPEN_POWER
        )

        time.sleep(0.5)

        print(
            "启动视频流..."
        )

        camera.start_video_stream(
            display=False,
            resolution=CAMERA_RES,
        )

        camera_started = True

        vision = AsyncVision(
            camera=camera,
            model=model,
            target=args.target,
            infer_fps=args.infer_fps,
            display_fps=args.display_fps,
        )

        vision.start()

        # 等第一帧
        start_wait = time.monotonic()

        while vision.frame_seq == 0:
            if (
                time.monotonic()
                - start_wait
                > 5.0
            ):
                raise RuntimeError(
                    "5秒内没有收到视频"
                )

            time.sleep(0.02)

        # ----------------------------------------------------
        # 多目标模式：
        #
        # 相机画面里可以同时出现多个网球/水瓶。
        # 每轮优先选择 bottom_y 最大（视觉上最近）的目标。
        #
        # 单次抓取：
        #   OBSERVE -> APPROACH -> PREGRASP -> PRECISE -> GRASP
        #
        # 分类模式：
        #   抓取成功
        #       ↓
        #   tennis_ball 左侧放置
        #   bottle      右侧放置
        #       ↓
        #   转回目标区
        #       ↓
        #   再次从剩余框中选最近目标
        # ----------------------------------------------------

        if not args.sort_mode:
            # 保留单目标测试模式
            cls, profile = pregrasp_with_recovery(
                arm=arm,
                chassis=chassis,
                vision=vision,
                profiles=profiles,
                args=args,
            )

            if cls is None or profile is None:
                try:
                    exp2.robot_signal_error(ep)
                except Exception:
                    pass

                vision.set_status("TASK FAILED")
                return 5

            print()
            print("==============================================")
            print("两阶段定位完成")
            print("目标类别:", cls)
            print("==============================================")

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

        # ====================================================
        # 连续自动分拣模式
        # ====================================================
        sorted_count = 0
        empty_count = 0
        tennis_count = 0
        bottle_count = 0

        # 多物体分拣始终允许同时识别两类
        args.target = "auto"
        vision.set_target("auto")
        vision.set_profile(None)

        print()
        print("============================================================")
        print("开始多目标自动分拣")
        print("优先级：bottom_y 最大的目标优先（最近优先）")
        print("网球 -> 左侧；水瓶 -> 右侧")
        print(f"最多处理 {args.max_items} 个目标")
        print("============================================================")

        while sorted_count < args.max_items:
            vision.set_target("auto")
            vision.set_profile(None)
            vision.set_status(
                f"SEARCH NEXT {sorted_count + 1}/{args.max_items}"
            )

            print()
            print(
                f"========== 第 {sorted_count + 1} 个目标 =========="
            )

            cls, profile = pregrasp_with_recovery(
                arm=arm,
                chassis=chassis,
                vision=vision,
                profiles=profiles,
                args=args,
            )

            if cls is None or profile is None:
                empty_count += 1

                print(
                    f"[SEARCH] 本轮没有找到可抓目标 "
                    f"({empty_count}/{args.empty_retries})"
                )

                # 回高位再看一次，避免因为上一轮姿态残留导致误判为空
                move_arm_to_observe_pose(arm)
                vision.set_target("auto")
                vision.set_profile(None)

                if empty_count >= args.empty_retries:
                    print()
                    print("连续多轮没有目标，认为分拣区域已经处理完成。")
                    break

                time.sleep(0.6)
                continue

            empty_count = 0

            print(
                f"[PRIORITY] 当前最近目标: {cls}"
            )

            if args.nav_only or args.dry_run:
                print(
                    "sort-mode + nav-only：已完成最近目标选择和靠近，"
                    "不执行抓取/分类。"
                )
                return 0

            ok = close_and_lift(
                ep=ep,
                arm=arm,
                gripper=gripper,
                class_name=cls,
                vision=vision,
            )

            if not ok:
                print(
                    "[GRASP] 本次抓取失败，抬回观察姿态后继续寻找。"
                )

                move_arm_to_observe_pose(arm)
                gripper.open(power=OPEN_POWER)
                vision.set_target("auto")
                vision.set_profile(None)
                time.sleep(0.5)
                continue

            # 抓取成功，执行左右分类
            place_to_sort_side(
                arm=arm,
                chassis=chassis,
                gripper=gripper,
                class_name=cls,
                vision=vision,
            )

            # 关键新增：
            # 放完以后不要直接在当前靠前位置找下一个。
            # 先后退恢复视野，再做左/中/右扫描，
            # 最终让车朝向下一批目标最有希望的方向。
            return_search_and_scan(
                arm=arm,
                chassis=chassis,
                vision=vision,
            )

            sorted_count += 1

            if cls == "tennis_ball":
                tennis_count += 1
            else:
                bottle_count += 1

            print()
            print(
                f"[PROGRESS] 已分拣 {sorted_count}/{args.max_items} | "
                f"tennis_ball={tennis_count} | bottle={bottle_count}"
            )

            # RETURN_SEARCH 已完成后退与左/中/右重观测。
            # 等画面稳定，再重新选择当前最近目标。
            time.sleep(0.6)

        try:
            exp2.robot_signal_success(ep)
        except Exception:
            pass

        vision.set_status("SORT COMPLETE")

        print()
        print("============================================================")
        print("自动分拣结束")
        print(f"总数        : {sorted_count}")
        print(f"tennis_ball : {tennis_count}")
        print(f"bottle      : {bottle_count}")
        print("============================================================")

        time.sleep(2.0)
        return 0

    except KeyboardInterrupt:
        print(
            "\n用户中止。"
        )
        return 130

    finally:
        if vision is not None:
            vision.stop()

        if (
            subscribed
            and gripper is not None
        ):
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
