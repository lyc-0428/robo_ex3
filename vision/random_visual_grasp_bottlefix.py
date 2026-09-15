#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RoboMaster EP 实验三：
基于学习视觉位置的随机抓取 —— 水瓶识别增强版

相对上一版主要修改：
1. 网球使用 YOLO imgsz=320、conf=0.35
2. 水瓶使用 YOLO imgsz=640、conf=0.25
3. auto 模式使用 imgsz=640、conf=0.25
4. 稳定识别由“连续3帧”改为“最近5次结果中同类至少出现3次”
5. 指定 tennis_ball / bottle 时，先恢复该类别学习时机械臂姿态，再开始识别
6. auto 模式 recenter 首次识别失败后，会依次尝试网球/水瓶学习姿态搜索
7. 网球用 bbox width 判断远近
8. 水瓶用 bbox height 判断远近（学习数据中 height 比 width 更稳定）
9. 继续保留 bottom_x 控制左右转向
10. 最小底盘前后移动 40 mm，避免实机对 20mm 左右的小位移无响应

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

# bbox 尺寸比例容差
SIZE_TOL_RATIO = 0.08

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

# 已按实机验证：目标在画面右侧时，z 需要为负
TURN_SIGN = -1.0

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

    # 先按置信度，再按 bbox 面积
    detections.sort(
        key=lambda d: (
            -d["conf"],
            -(d["width"] * d["height"]),
        )
    )

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

        window_name = "EXP3 BottleFix Learned Visual Grasp"

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

                        color = (
                            (0, 255, 0)
                            if is_best
                            else (0, 180, 255)
                        )

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


def locate_stable(
    vision,
    timeout=TARGET_TIMEOUT,
    window_size=STABLE_WINDOW,
    required_hits=STABLE_HITS,
):
    """
    最近 window_size 次 YOLO 结果中，
    同一类别至少出现 required_hits 次即认为稳定。

    None 会保留在窗口中，但不会把历史清零。
    """
    start = time.monotonic()
    last_seq = vision.current_det_seq()

    history = deque(maxlen=window_size)

    while time.monotonic() - start < timeout:
        if vision.is_stopped():
            raise KeyboardInterrupt

        seq, _, best, _ = vision.snapshot()

        if seq <= last_seq:
            time.sleep(0.01)
            continue

        last_seq = seq

        # None 也进入窗口
        history.append(best)

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
                np.median(
                    [d["conf"] for d in same]
                )
            ),
            "bottom_x": float(
                np.median(
                    [d["bottom_x"] for d in same]
                )
            ),
            "bottom_y": float(
                np.median(
                    [d["bottom_y"] for d in same]
                )
            ),
            "width": float(
                np.median(
                    [d["width"] for d in same]
                )
            ),
            "height": float(
                np.median(
                    [d["height"] for d in same]
                )
            ),
        }

    return None


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

    arm.moveto(
        x=float(profile["arm_x_mm"]),
        y=float(profile["arm_y_mm"]),
    ).wait_for_completed()

    time.sleep(0.7)


def initial_detect(
    arm,
    vision,
    profiles,
    args,
):
    """
    指定类别：
        先恢复该类别学习姿态，再识别。

    auto：
        先 recenter。
        若失败，则依次尝试网球学习姿态、水瓶学习姿态。
    """
    if args.target in (
        "tennis_ball",
        "bottle",
    ):
        profile = profiles[args.target]

        vision.set_target(args.target)
        vision.set_profile(profile)
        vision.set_status(
            f"INITIAL {args.target} / learned pose"
        )

        move_arm_to_profile_pose(
            arm,
            profile,
            "初始学习姿态",
        )

        loc = locate_stable(
            vision,
            timeout=args.target_timeout,
            window_size=args.stable_window,
            required_hits=args.stable_hits,
        )

        return loc

    # auto 模式
    vision.set_target("auto")
    vision.set_profile(None)
    vision.set_status("AUTO initial / recenter")

    print("AUTO 模式：先 recenter 搜索。")

    arm.recenter().wait_for_completed()
    time.sleep(0.7)

    loc = locate_stable(
        vision,
        timeout=args.auto_pose_timeout,
        window_size=args.stable_window,
        required_hits=args.stable_hits,
    )

    if loc is not None:
        return loc

    # recenter 没找到，再依次尝试两类学习姿态
    for cls in (
        "tennis_ball",
        "bottle",
    ):
        if cls not in profiles:
            continue

        profile = profiles[cls]

        vision.set_status(
            f"AUTO search pose: {cls}"
        )

        move_arm_to_profile_pose(
            arm,
            profile,
            f"AUTO 尝试 {cls} 姿态",
        )

        loc = locate_stable(
            vision,
            timeout=args.auto_pose_timeout,
            window_size=args.stable_window,
            required_hits=args.stable_hits,
        )

        if loc is not None:
            return loc

    return None


def turn_step_from_x_error(x_error):
    a = abs(x_error)

    if a > 90:
        mag = LARGE_TURN_DEG

    elif a > 45:
        mag = MID_TURN_DEG

    else:
        mag = SMALL_TURN_DEG

    return (
        TURN_SIGN
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

    ref_x = float(
        profile["bottom_x_px"]
    )

    ref_y = float(
        profile["bottom_y_px"]
    )

    # 网球用宽度；瓶子用高度
    if cls == "bottle":
        ref_size = float(
            profile["bbox_height_px"]
        )
        size_name = "height"

    else:
        ref_size = float(
            profile["bbox_width_px"]
        )
        size_name = "width"

    for step in range(
        1,
        args.max_steps + 1,
    ):
        vision.set_status(
            f"SERVO {step}/{args.max_steps}"
        )

        loc = locate_stable(
            vision,
            timeout=args.target_timeout,
            window_size=args.stable_window,
            required_hits=args.stable_hits,
        )

        if loc is None:
            print(
                "没有稳定识别到目标，停止。"
            )
            return None

        x_error = (
            loc["bottom_x"]
            - ref_x
        )

        y_error = (
            loc["bottom_y"]
            - ref_y
        )

        if cls == "bottle":
            current_size = float(
                loc["height"]
            )

        else:
            current_size = float(
                loc["width"]
            )

        size_ratio = (
            current_size
            / ref_size
        )

        print()
        print(
            f"[SERVO {step}] "
            f"{loc['class']} "
            f"conf={loc['conf']:.2f}"
        )

        print(
            f"  X: current={loc['bottom_x']:.1f} "
            f"ref={ref_x:.1f} "
            f"err={x_error:+.1f}px"
        )

        print(
            f"  Y: current={loc['bottom_y']:.1f} "
            f"ref={ref_y:.1f} "
            f"err={y_error:+.1f}px"
        )

        print(
            f"  {size_name}: "
            f"current={current_size:.1f} "
            f"ref={ref_size:.1f} "
            f"ratio={size_ratio:.3f}"
        )

        # 先左右转向
        if abs(x_error) > args.x_tol:
            z = turn_step_from_x_error(
                x_error
            )

            direction = (
                "右"
                if x_error > 0
                else "左"
            )

            print(
                f"  目标在画面{direction}侧 "
                f"-> 转向 {z:+.1f}deg"
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

        # 再调整前后距离
        dx = move_step_from_size_ratio(
            size_ratio
        )

        if dx != 0.0:
            direction = (
                "前进"
                if dx > 0
                else "后退"
            )

            print(
                f"  距离视觉校正: "
                f"{size_name} ratio={size_ratio:.3f} "
                f"-> {direction} "
                f"{abs(dx)*1000:.0f}mm"
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

        # 网球保留 bottom_y 二次确认
        # 瓶子 bottom_y≈356/360，接近图像下边界，因此不做硬限制
        if (
            cls == "tennis_ball"
            and abs(y_error)
            > args.ball_y_tol
        ):
            dx = (
                MIN_MOVE_M
                if y_error < 0
                else -MIN_MOVE_M
            )

            direction = (
                "前进"
                if dx > 0
                else "后退"
            )

            print(
                f"  网球 Y 误差仍为 "
                f"{y_error:+.1f}px "
                f"-> {direction} "
                f"{abs(dx)*1000:.0f}mm"
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

        print()
        print(
            "=============================================="
        )
        print(
            "[OK] 已到学习得到的成功抓取视觉位置"
        )
        print(
            f"class={cls} "
            f"x_error={x_error:+.1f}px "
            f"{size_name}_ratio={size_ratio:.3f} "
            f"y_error={y_error:+.1f}px"
        )
        print(
            "=============================================="
        )

        vision.set_status(
            "READY TO GRASP"
        )

        return loc

    print(
        "达到最大视觉闭环次数，停止。"
    )

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
        # 初始检测：
        # 指定类别时先去该类别学习姿态
        # auto 时 recenter -> 网球姿态 -> 水瓶姿态
        # ----------------------------------------------------

        first = initial_detect(
            arm=arm,
            vision=vision,
            profiles=profiles,
            args=args,
        )

        if first is None:
            print(
                "所有初始搜索姿态都没有稳定识别到目标。"
            )
            return 4

        cls = first["class"]

        if cls not in profiles:
            print(
                f"识别到 {cls}，"
                "但没有对应学习 profile。"
            )
            return 4

        profile = profiles[cls]

        print()
        print(
            "=============================================="
        )
        print(
            "识别类别:",
            cls,
        )
        print(
            "目标学习机械臂姿态:",
            f"x={profile['arm_x_mm']:.1f}mm",
            f"y={profile['arm_y_mm']:.1f}mm",
        )
        print(
            "=============================================="
        )

        # 锁定类别，后续不再被另一类别干扰
        vision.set_target(cls)
        vision.set_profile(profile)
        vision.set_status(
            f"LOCKED TARGET: {cls}"
        )

        # 不管 initial_detect 最后在哪个姿态，
        # 都再次确保恢复当前类别的成功学习姿态。
        move_arm_to_profile_pose(
            arm,
            profile,
            "恢复目标学习姿态",
        )

        gripper.open(
            power=OPEN_POWER
        )

        time.sleep(0.3)

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

            vision.set_status(
                "TASK FAILED"
            )

            return 5

        if (
            args.nav_only
            or args.dry_run
        ):
            try:
                exp2.robot_signal_success(ep)
            except Exception:
                pass

            vision.set_status(
                "NAV COMPLETE"
            )

            print(
                "仅导航测试完成，不执行抓取。"
            )

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
            print(
                "保持夹持 5 秒。"
            )

            vision.set_status(
                "KEEP OBJECT"
            )

            time.sleep(5.0)

        vision.set_status(
            "DONE"
        )

        print(
            "任务完成。"
        )

        time.sleep(1.0)

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
