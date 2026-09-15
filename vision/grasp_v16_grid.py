#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RoboMaster EP 实验三：3x3 网格多目标分类整理。
从下边中间起点进入；抓取后倒序退出到起点；网球放左外侧，水瓶放右外侧。

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

注意：网格宽度必须现场测量并通过 --grid-width-m 提供。
路线倒放依赖底盘动作和位置反馈；超出起点误差会停止任务。
"""

import argparse
import json
import math
import os
import sys
import threading
import time
from datetime import datetime
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

# ==========================================================
# V14：夹爪误识别为 bottle 的视觉排除区
# ==========================================================
# 只过滤 bottle，不过滤 tennis_ball。
#
# 坐标使用归一化比例 (x1, y1, x2, y2)，不会写死640x360。
# 当前区域只覆盖画面下方左右夹爪，不覆盖中央抓取通道。
#
# 只有当 bottle 检测框有 >=55% 面积落在排除区内时才丢弃，
# 因而真实水瓶即使靠近排除区边缘，也不会轻易被误删。
BOTTLE_GRIPPER_EXCLUDE_ENABLED = True
BOTTLE_GRIPPER_EXCLUDE_MIN_OVERLAP = 0.55

BOTTLE_GRIPPER_EXCLUDE_ZONES_NORM = (
    # 左夹爪
    (0.23, 0.72, 0.43, 1.00),
    # 右夹爪（顺便预防后续也被识别成bottle）
    (0.57, 0.72, 0.78, 1.00),
)

# 在视频窗口画出排除区，方便实机确认位置。
SHOW_GRIPPER_EXCLUDE_ZONES = True

# ==========================================================
# V15：最终抓取阶段的“目标实例锁”
# ==========================================================
#
# 优先级严格固定：
#
# SEARCH:
#   可以比较所有合法目标，选本轮要抓的目标类别。
#
# PIPELINE:
#   一旦选中 tennis_ball / bottle，就锁定类别，
#   YOLO 只保留这个类别。
#
# FINAL CLOSE:
#   第一次在最终机械臂姿态下确认到目标框后，
#   锁定“这个具体框对应的物体”。
#   后续只允许跟踪与上一帧锁定框位置连续的同一个物体。
#
# 如果锁定目标暂时消失：
#   停止/等待 -> 超时后本轮失败恢复。
#   绝对不自动跳到旁边另一个同类物体。
#
# 这些参数只控制“是否还是同一个目标”，
# 不修改网球/水瓶已有的抓取阈值和运动步长。
FINAL_LOCK_MAX_DX_PX = 95.0
FINAL_LOCK_MAX_DY_PX = 120.0
FINAL_LOCK_MIN_SIZE_RATIO = 0.45
FINAL_LOCK_MAX_SIZE_RATIO = 2.20

# 最终姿态第一次建立锁时允许较宽的范围。
# camera/arm 从 ALIGN 姿态切到最终抓取姿态后，
# 同一个目标的像素位置会发生一次较大的跳变。
FINAL_LOCK_ACQUIRE_MAX_X_ERR_PX = 180.0
FINAL_LOCK_ACQUIRE_MAX_Y_ERR_PX = 190.0

STABLE_WINDOW = 5
STABLE_HITS = 3
TARGET_TIMEOUT = 8.0

X_TOL_PX = 14.0

# 网球 bbox 尺寸比例容差
SIZE_TOL_RATIO = 0.08

# 水瓶成功抓取视觉状态
#
# V11 实机现象：
# - height=291~292、bottom_y≈356~360、xerr≈0 时，本来已经非常接近成功姿态，
#   但旧 MAX_HEIGHT=290 会强制后退40mm；
# - height≈288、bottom_y≈345 时，旧 bottom_y>=350 又会强制前进40mm；
# - 40mm 往返 + 3°近距转向导致明显“前后/左右抖动”。
#
# V12 只放宽 bottle 的最终窗口，并降低近距动作幅度。
BOTTLE_MIN_HEIGHT = 270.0
BOTTLE_MAX_HEIGHT = 296.0
BOTTLE_MIN_BOTTOM_Y = 345.0
BOTTLE_X_TOL_PX = 16.0

# 水瓶 close 阶段最低可信度。
# 太近时 YOLO 框容易抖，低可信度帧不参与底盘动作。
BOTTLE_CLOSE_MIN_CONF = 0.55

# 水瓶最终 READY 连续确认次数。
BOTTLE_READY_REQUIRED_HITS = 2

# 水瓶近距离专用的小步长，避免40mm往返振荡。
BOTTLE_FINE_FORWARD_M = 0.015
BOTTLE_FINE_BACK_M = 0.015
BOTTLE_VERY_FINE_FORWARD_M = 0.010

# 水瓶近距离专用小角度修正。
BOTTLE_FINE_TURN_DEG = 1.0
BOTTLE_MID_TURN_DEG = 2.0

# 用最近3个高可信检测做中值滤波，降低 bbox 抖动。
BOTTLE_FILTER_WINDOW = 3

# 水瓶分段靠近阈值
BOTTLE_FAR_H = 240.0
BOTTLE_MID_H = 260.0
BOTTLE_NEAR_H = 270.0

# 网球额外检查 bottom_y
BALL_Y_TOL_PX = 20.0

# ==========================================================
# V13：只修网球最终近距离闭环
# ==========================================================
# V12 实机日志中，网球已经完成：
#   NEAR_VIEW -> ALIGN READY 2/2 -> 最终学习抓取姿态
# 说明远距/中距逻辑是正常的。
#
# 真正失败点在 TENNIS_CLOSE：
# 最终姿态后 xerr≈+22.5px，球实际仍位于夹爪中央附近，
# 旧阈值 x_tol=14px 却强制执行 3° 转向；
# 同时 close 循环可能重复消费同一个 YOLO 结果，
# 因而会对同一帧连续转动，反而把球转丢。
#
# 本版只改 tennis close：
# - 最终横向容差放宽到 30px
# - 近距转向降为 1°/2°
# - 每个 YOLO seq 最多执行一次运动
# - 最终机械臂姿态切换后必须等待“新帧”再动作
# - 前后调整改为 15~40mm 小步长
TENNIS_CLOSE_X_TOL_PX = 30.0
TENNIS_CLOSE_Y_TOL_PX = 25.0
TENNIS_CLOSE_SIZE_TOL_RATIO = 0.10
TENNIS_CLOSE_MIN_CONF = 0.35

TENNIS_CLOSE_FINE_TURN_DEG = 1.0
TENNIS_CLOSE_MID_TURN_DEG = 2.0

TENNIS_CLOSE_FAR_FORWARD_M = 0.040
TENNIS_CLOSE_MID_FORWARD_M = 0.025
TENNIS_CLOSE_FINE_FORWARD_M = 0.015
TENNIS_CLOSE_FINE_BACK_M = 0.015

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
ARM_ACTION_TIMEOUT_S = 8.0
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
# 网球近距离中间观察姿态
# ==========================================================
# V9的问题：
# 网球在 recenter 观察姿态下接近到 width≈55~60px 时，
# 夹爪会逐渐遮挡目标；而机械臂只有最终 PREGRASP 才下降，
# 导致目标在第二次 ALIGN READY 前就丢失。
#
# V10改为：
# 远距离靠近 -> 接近ALIGN阈值 -> 停车
# -> 机械臂先相对下降一小段 -> 重新寻找网球
# -> 在这个“近距观察姿态”继续慢速靠近/对准
# -> 达到最终阈值后才进入学习抓取姿态。
#
# 只对 tennis_ball 使用，避免破坏已经能稳定抓取的 bottle。
BALL_NEAR_VIEW_DROP_MM = -45.0
BALL_NEAR_VIEW_SETTLE_S = 0.45
BALL_NEAR_VIEW_LOST_TIMEOUT_S = 3.0

# ==========================================================
# V11：网球 / 水瓶双独立抓取流水线
# ==========================================================
#
# 扫描阶段只负责判断“下一件抓什么”。
# 一旦类别确定，后面的靠近、机械臂姿态和近距离闭环完全分流：
#
#   tennis_ball -> tennis_pipeline()
#   bottle      -> bottle_pipeline()
#
# 不再让两类目标共用同一个 approach 状态机，避免网球近距下降逻辑
# 影响已经验证过的水瓶流程。
TARGET_SELECT_TIMEOUT_S = 3.0
TARGET_SELECT_WINDOW = 5
TARGET_SELECT_HITS = 2

# bottle 使用 640 推理，真实 FPS 较低，允许更长的观察丢失时间。
BOTTLE_OBSERVE_LOST_TIMEOUT_S = 3.0

# 两类目标分别保留自己的 ALIGN 参数，后续可独立调参。
TENNIS_ALIGN_ENTER_RATIO = 0.85
TENNIS_ALIGN_X_TOL_PX = 30.0
TENNIS_ALIGN_REQUIRED_HITS = 2

BOTTLE_ALIGN_ENTER_RATIO = 0.85
BOTTLE_ALIGN_X_TOL_PX = 30.0
BOTTLE_ALIGN_REQUIRED_HITS = 2


# ==========================================================
# 多目标优先抓取 + 分类放置
# ==========================================================

# 同一画面里优先抓最近物体：
# 固定观察姿态下，物体与桌面接触点 bottom_y 越大，通常越靠近机器人。
# bbox 面积只用于 bottom_y 接近时的次级排序。
NEAREST_BOTTOM_Y_WEIGHT = 1.0

# 默认最多分拣 6 个物体（实验要求至少 6 个目标）
DEFAULT_MAX_ITEMS = 6
MAX_TASK_FAILURES = 3
ROUTE_MOVE_TIMEOUT_S = 20.0
GRID_SIDE_SPEED = 0.20

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



def _rect_intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    return (ix2 - ix1) * (iy2 - iy1)


def bottle_box_is_on_gripper(
    box_xyxy,
    frame_width,
    frame_height,
):
    """
    判断 bottle 检测框是否主要落在夹爪固定像素区域。

    采用“检测框面积中有多少比例位于夹爪区”作为判断，
    比单纯看中心点更稳，也更不容易误删真实水瓶。
    """
    if not BOTTLE_GRIPPER_EXCLUDE_ENABLED:
        return False

    x1, y1, x2, y2 = [float(v) for v in box_xyxy]

    box_area = max(1.0, (x2 - x1) * (y2 - y1))

    for nx1, ny1, nx2, ny2 in BOTTLE_GRIPPER_EXCLUDE_ZONES_NORM:
        zone = (
            nx1 * frame_width,
            ny1 * frame_height,
            nx2 * frame_width,
            ny2 * frame_height,
        )

        overlap = _rect_intersection_area(
            (x1, y1, x2, y2),
            zone,
        )

        overlap_ratio = overlap / box_area

        if overlap_ratio >= BOTTLE_GRIPPER_EXCLUDE_MIN_OVERLAP:
            return True

    return False



def detect_one_frame(model, frame_bgr, target, use_cuda):
    imgsz, conf = infer_config(target)

    frame_h, frame_w = frame_bgr.shape[:2]

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

        # ------------------------------------------------------
        # V14：夹爪误检过滤
        # ------------------------------------------------------
        # 只对 bottle 生效。
        # 网球检测、网球抓取、水瓶抓取控制本身都不改。
        if (
            name == "bottle"
            and bottle_box_is_on_gripper(
                (x1, y1, x2, y2),
                frame_width=frame_w,
                frame_height=frame_h,
            )
        ):
            continue

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


class RouteError(RuntimeError):
    pass


class RecordedAction:
    def __init__(self, action, route, command):
        self.action = action
        self.route = route
        self.command = command

    def wait_for_completed(self, timeout=None):
        limit = ROUTE_MOVE_TIMEOUT_S if timeout is None else min(float(timeout), ROUTE_MOVE_TIMEOUT_S)
        if not self.action.wait_for_completed(timeout=limit):
            raise RouteError(f"底盘动作未在 {limit:.1f}s 内完成: {self.command}")
        self.route.steps.append(self.command)
        return True


class RouteChassis:
    """Records relative chassis commands from the grid entrance for reverse replay."""

    def __init__(self, chassis):
        self.raw = chassis
        self.steps = []
        self.recording = False
        self.active_speed = None

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def begin(self):
        if self.steps or self.active_speed:
            raise RouteError("上一段路线尚未返回起点")
        self.recording = True

    def _flush_speed(self, now=None):
        if self.active_speed is None:
            return
        x, y, z, started, timeout = self.active_speed
        elapsed = max(0.0, (time.monotonic() if now is None else now) - started)
        duration = min(elapsed, timeout)
        if duration >= 0.02 and (abs(x) + abs(y) + abs(z)) > 0.001:
            self.steps.append(("speed", x, y, z, duration))
        self.active_speed = None

    def drive_speed(self, x=0, y=0, z=0, timeout=0.5):
        now = time.monotonic()
        if self.recording:
            self._flush_speed(now)
        result = self.raw.drive_speed(x=x, y=y, z=z, timeout=timeout)
        if self.recording and (abs(x) + abs(y) + abs(z)) > 0.001:
            self.active_speed = (float(x), float(y), float(z), now, float(timeout))
        return result

    def move(self, x=0, y=0, z=0, xy_speed=0.5, z_speed=30):
        if self.recording:
            self._flush_speed()
        command = ("move", float(x), float(y), float(z), float(xy_speed), float(z_speed))
        action = self.raw.move(x=x, y=y, z=z, xy_speed=xy_speed, z_speed=z_speed)
        return RecordedAction(action, self, command) if self.recording else action

    def stop(self):
        self.drive_speed(x=0, y=0, z=0, timeout=0.5)

    def retrace(self):
        self.stop()
        self.recording = False
        steps = list(reversed(self.steps))
        try:
            for step in steps:
                if step[0] == "move":
                    _, x, y, z, xy_speed, z_speed = step
                    self.stop()
                    action = self.raw.move(
                        x=-x, y=-y, z=-z, xy_speed=xy_speed, z_speed=z_speed
                    )
                    if not action.wait_for_completed(timeout=ROUTE_MOVE_TIMEOUT_S):
                        raise RouteError(f"原路退出超时: {step}")
                else:
                    _, x, y, z, duration = step
                    self.raw.drive_speed(x=-x, y=-y, z=-z, timeout=max(0.5, duration + 0.2))
                    time.sleep(duration)
            self.stop()
        except Exception:
            self.stop()
            raise
        self.steps.clear()


class HomePoseMonitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = None
        self.home = None

    def callback(self, pose):
        with self.lock:
            self.latest = (tuple(float(v) for v in pose[:3]), time.monotonic())

    def capture_home(self, timeout=4.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self.latest and time.monotonic() - self.latest[1] < 1.0:
                    self.home = self.latest[0]
                    return
            time.sleep(0.05)
        raise RouteError("没有收到底盘位置反馈，不能执行连续原路返回")

    def assert_home(self, position_tol_m, angle_tol_deg):
        requested = time.monotonic()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with self.lock:
                sample = self.latest
            if sample and sample[1] >= requested:
                x, y, z = sample[0]
                hx, hy, hz = self.home
                distance = math.hypot(x - hx, y - hy)
                angle = abs((z - hz + 180.0) % 360.0 - 180.0)
                print(f"[HOME] position error={distance:.3f}m, heading error={angle:.1f}deg")
                if distance > position_tol_m or angle > angle_tol_deg:
                    raise RouteError("原路返回偏离起点；请重新放车到起点后再运行")
                return
            time.sleep(0.05)
        raise RouteError("底盘位置反馈中断，停止下一轮")


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

        window_name = "EXP3 Grasp V15"

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

                    # V14：把 bottle 夹爪排除区画出来，便于现场确认。
                    if (
                        SHOW_GRIPPER_EXCLUDE_ZONES
                        and BOTTLE_GRIPPER_EXCLUDE_ENABLED
                    ):
                        for zone_i, (
                            nx1,
                            ny1,
                            nx2,
                            ny2,
                        ) in enumerate(
                            BOTTLE_GRIPPER_EXCLUDE_ZONES_NORM,
                            start=1,
                        ):
                            zx1 = int(round(nx1 * w))
                            zy1 = int(round(ny1 * h))
                            zx2 = int(round(nx2 * w))
                            zy2 = int(round(ny2 * h))

                            cv2.rectangle(
                                shown,
                                (zx1, zy1),
                                (zx2, zy2),
                                (255, 0, 255),
                                1,
                            )

                            cv2.putText(
                                shown,
                                f"BOTTLE IGNORE {zone_i}",
                                (zx1 + 3, max(12, zy1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.38,
                                (255, 0, 255),
                                1,
                                cv2.LINE_AA,
                            )

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
                abs(best_det["bottom_x"] - previous["bottom_x"]) <= 80
                and abs(best_det["bottom_y"] - previous["bottom_y"]) <= 80
            ):
                return best_det

        return None

    # 仅在尚未锁定目标时选最近物体。
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
    max_age=None,
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

    if max_age is None:
        max_age = MAX_DET_AGE_S

    if age > max_age:
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
    raise RouteError(f"机械臂姿态未确认: {label}")





def acquire_final_target_lock(
    detections,
    class_name,
    ref_x,
    ref_y,
    ref_size,
    size_key,
):
    """
    FINAL CLOSE 第一次建立实例锁。

    这里只发生一次：
    从指定类别的所有框中，选出最接近“学习抓取参考视觉状态”的那个。

    一旦建立成功，后续不再重新按中心/最近目标选择，
    而是只做 frame-to-frame 连续跟踪。
    """
    same = [
        d
        for d in detections
        if d["class"] == class_name
    ]

    if not same:
        return None

    valid = []

    for d in same:
        dx = abs(float(d["bottom_x"]) - float(ref_x))
        dy = abs(float(d["bottom_y"]) - float(ref_y))

        size = max(1.0, float(d[size_key]))
        size_ratio = size / max(1.0, float(ref_size))

        if dx > FINAL_LOCK_ACQUIRE_MAX_X_ERR_PX:
            continue

        if dy > FINAL_LOCK_ACQUIRE_MAX_Y_ERR_PX:
            continue

        if not (
            FINAL_LOCK_MIN_SIZE_RATIO
            <= size_ratio
            <= FINAL_LOCK_MAX_SIZE_RATIO
        ):
            continue

        # 参考中心优先，其次尺寸接近。
        score = (
            dx
            + 0.65 * dy
            + 35.0 * abs(size_ratio - 1.0)
        )

        valid.append((score, d))

    if not valid:
        return None

    if len(valid) > 1:
        print(f"[FINAL_LOCK] 同类候选有 {len(valid)} 个，无法确定原目标；停止本轮")
        return None

    valid.sort(key=lambda item: item[0])

    return dict(valid[0][1])


def update_final_target_lock(
    detections,
    locked_det,
    class_name,
    size_key,
):
    """
    FINAL CLOSE 已建立锁后的严格连续跟踪。

    关键点：
    - 只找同一类别；
    - 必须与上一帧 locked_det 在像素位置和尺寸上连续；
    - 找不到时返回 None；
    - 不会 fallback 到“最近目标”或“学习中心附近另一个目标”。

    因此一旦进入最终抓取状态，就不会突然换抓旁边物体。
    """
    if locked_det is None:
        return None

    same = [
        d
        for d in detections
        if d["class"] == class_name
    ]

    if not same:
        return None

    prev_x = float(locked_det["bottom_x"])
    prev_y = float(locked_det["bottom_y"])
    prev_size = max(1.0, float(locked_det[size_key]))

    candidates = []

    for d in same:
        dx = abs(float(d["bottom_x"]) - prev_x)
        dy = abs(float(d["bottom_y"]) - prev_y)

        size = max(1.0, float(d[size_key]))
        size_ratio = size / prev_size

        if dx > FINAL_LOCK_MAX_DX_PX:
            continue

        if dy > FINAL_LOCK_MAX_DY_PX:
            continue

        if not (
            FINAL_LOCK_MIN_SIZE_RATIO
            <= size_ratio
            <= FINAL_LOCK_MAX_SIZE_RATIO
        ):
            continue

        # 位置连续性优先，尺寸变化作为辅助。
        continuity_cost = (
            dx
            + 0.70 * dy
            + 30.0 * abs(size_ratio - 1.0)
        )

        candidates.append((continuity_cost, d))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])

    return dict(candidates[0][1])



def tennis_turn_step_from_x_error(x_error):
    """
    网球最终近距离专用小角度转向。
    不使用通用 3/6/10°，避免最终抓取姿态下左右过冲。
    """
    a = abs(float(x_error))

    if a > 60.0:
        mag = TENNIS_CLOSE_MID_TURN_DEG
    else:
        mag = TENNIS_CLOSE_FINE_TURN_DEG

    return (
        TURN_SIGN_MOVE
        * math.copysign(
            mag,
            x_error,
        )
    )



def close_approach_tennis(
    chassis,
    vision,
    profile,
    args,
):
    """
    V13 网球最终近距离闭环。

    注意：本函数只处理 tennis_ball。
    bottle 的 approach / close / READY 阈值完全不经过这里。

    修复重点：
    1. 最终学习姿态后等待真正的新 YOLO 结果；
    2. 一个 det_seq 最多执行一次运动，禁止同一帧重复转向/前进；
    3. xerr 30px 内视为已经横向对准，优先继续前后距离调整；
    4. 近距离转向改为 1°/2°；
    5. 前后步长缩小，减少越过成功位置。
    """
    ref_x = float(profile["bottom_x_px"])
    ref_y = float(profile["bottom_y_px"])
    ref_w = float(profile["bbox_width_px"])

    lost_since = None

    # FINAL CLOSE 实例锁状态：
    # 一旦 lock_acquired=True，本轮不允许换到其他网球。
    lock_acquired = False
    locked_det = None

    # 关键：进入最终抓取姿态后，不允许立刻消费机械臂运动前的旧检测。
    last_seq = vision.current_det_seq()

    print(
        f"[TENNIS_CLOSE] 等待最终机械臂姿态后的新YOLO帧 "
        f"(start_seq={last_seq})"
    )

    step = 0

    while step < CLOSE_MAX_STEPS:
        if vision.is_stopped():
            raise KeyboardInterrupt

        vision.set_status(
            f"TENNIS_CLOSE {step + 1}/{CLOSE_MAX_STEPS}"
        )

        seq, det_time, _, detections = vision.snapshot()

        # --------------------------------------------------------
        # 必须是“新的”推理结果
        # --------------------------------------------------------
        if seq == last_seq:
            if lost_since is None:
                lost_since = time.monotonic()

            if time.monotonic() - lost_since >= CLOSE_LOST_TIMEOUT_S:
                print(
                    "[TENNIS_CLOSE] 等待新YOLO结果超时，"
                    "退出本轮网球近距离闭环。"
                )
                return None

            time.sleep(CLOSE_LOOP_PERIOD_S)
            continue

        # 从现在开始，这个 seq 无论是否采取动作，都只处理一次。
        last_seq = seq

        if det_time <= 0:
            continue

        age = time.monotonic() - det_time

        if age > MAX_DET_AGE_S:
            continue

        # --------------------------------------------------------
        # FINAL TARGET INSTANCE LOCK
        # --------------------------------------------------------
        # 第一次新帧：建立锁；
        # 后续：只跟踪这个锁定实例，绝不跳到另一个网球。
        if not lock_acquired:
            det = acquire_final_target_lock(
                detections=detections,
                class_name="tennis_ball",
                ref_x=ref_x,
                ref_y=ref_y,
                ref_size=ref_w,
                size_key="width",
            )

            if det is not None:
                locked_det = det
                lock_acquired = True
                lost_since = None

                print(
                    "[FINAL_LOCK][tennis_ball] 已锁定当前网球实例："
                    f"x={float(det['bottom_x']):.1f}, "
                    f"y={float(det['bottom_y']):.1f}, "
                    f"w={float(det['width']):.1f}"
                )
            else:
                if lost_since is None:
                    lost_since = time.monotonic()

                if time.monotonic() - lost_since >= CLOSE_LOST_TIMEOUT_S:
                    print(
                        "[FINAL_LOCK][tennis_ball] "
                        "最终姿态下没有找到可锁定的当前网球。"
                    )
                    return None

                time.sleep(CLOSE_LOOP_PERIOD_S)
                continue

        else:
            det = update_final_target_lock(
                detections=detections,
                locked_det=locked_det,
                class_name="tennis_ball",
                size_key="width",
            )

            if det is None:
                if lost_since is None:
                    lost_since = time.monotonic()

                print(
                    "[FINAL_LOCK][tennis_ball] "
                    "锁定网球本帧暂时丢失；忽略其他目标，不切换。"
                )

                if time.monotonic() - lost_since >= CLOSE_LOST_TIMEOUT_S:
                    print(
                        "[FINAL_LOCK][tennis_ball] "
                        "锁定目标丢失超时，本轮失败恢复。"
                    )
                    return None

                time.sleep(CLOSE_LOOP_PERIOD_S)
                continue

            locked_det = det
            lost_since = None

        conf = float(det["conf"])

        if conf < TENNIS_CLOSE_MIN_CONF:
            print(
                f"[TENNIS_CLOSE] seq={seq} conf={conf:.2f} "
                f"< {TENNIS_CLOSE_MIN_CONF:.2f}，本帧不动作。"
            )
            continue

        lost_since = None
        step += 1

        bottom_x = float(det["bottom_x"])
        bottom_y = float(det["bottom_y"])
        width = float(det["width"])

        x_error = bottom_x - ref_x
        y_error = bottom_y - ref_y
        width_ratio = width / ref_w

        print()
        print(
            f"[TENNIS_CLOSE {step}] seq={seq} conf={conf:.2f} "
            f"xerr={x_error:+.1f}px "
            f"yerr={y_error:+.1f}px "
            f"width={width:.1f}/{ref_w:.1f} "
            f"ratio={width_ratio:.3f}"
        )

        # --------------------------------------------------------
        # 1. 横向
        #
        # V12截图中 xerr=+22.5px 时球实际上仍位于夹爪中间，
        # 所以不再用14px阈值强制3°转向。
        # --------------------------------------------------------
        if abs(x_error) > TENNIS_CLOSE_X_TOL_PX:
            z = tennis_turn_step_from_x_error(x_error)
            direction = "右" if x_error > 0 else "左"

            print(
                f"  TENNIS ACTION: FINE TURN "
                f"{direction} {z:+.1f}deg"
            )

            chassis.move(
                x=0,
                y=0,
                z=float(z),
                z_speed=CHASSIS_Z_SPEED,
            ).wait_for_completed()

            # 下一次循环必须等待另一个新 seq。
            time.sleep(0.12)
            continue

        # --------------------------------------------------------
        # 2. READY
        # --------------------------------------------------------
        ready = (
            abs(x_error) <= TENNIS_CLOSE_X_TOL_PX
            and abs(width_ratio - 1.0) <= TENNIS_CLOSE_SIZE_TOL_RATIO
            and abs(y_error) <= TENNIS_CLOSE_Y_TOL_PX
        )

        if ready:
            try:
                chassis.drive_speed(
                    x=0,
                    y=0,
                    z=0,
                    timeout=0.5,
                )
            except Exception:
                pass

            print()
            print("==============================================")
            print("[TENNIS READY] 网球已达到稳定抓取视觉状态")
            print(
                f"x_error={x_error:+.1f}px, "
                f"width_ratio={width_ratio:.3f}, "
                f"y_error={y_error:+.1f}px"
            )
            print("==============================================")

            vision.set_status("TENNIS READY_TO_GRASP")
            return {
                "class": "tennis_ball",
                "conf": conf,
                "bottom_x": bottom_x,
                "bottom_y": bottom_y,
                "width": width,
                "height": float(det["height"]),
                "det_age": float(age),
            }

        # --------------------------------------------------------
        # 3. 前后距离
        #
        # 当前截图 width=74 / ref=98 => ratio≈0.755。
        # 这时真正需要的是继续向前，而不是因为 xerr=22.5px
        # 连续原地转3°。
        # --------------------------------------------------------
        if width_ratio < 0.80:
            dx = TENNIS_CLOSE_FAR_FORWARD_M
            reason = "网球仍明显偏远"

        elif width_ratio < (1.0 - TENNIS_CLOSE_SIZE_TOL_RATIO):
            dx = TENNIS_CLOSE_MID_FORWARD_M
            reason = "网球接近目标尺寸"

        elif width_ratio > 1.20:
            dx = -TENNIS_CLOSE_FAR_FORWARD_M
            reason = "网球明显过近"

        elif width_ratio > (1.0 + TENNIS_CLOSE_SIZE_TOL_RATIO):
            dx = -TENNIS_CLOSE_FINE_BACK_M
            reason = "网球略近"

        else:
            # 尺寸已经基本正确，只利用 bottom_y 做极小修正。
            if y_error < -TENNIS_CLOSE_Y_TOL_PX:
                dx = TENNIS_CLOSE_FINE_FORWARD_M
                reason = "尺寸合适但目标仍偏上，微量前进"
            elif y_error > TENNIS_CLOSE_Y_TOL_PX:
                dx = -TENNIS_CLOSE_FINE_BACK_M
                reason = "尺寸合适但目标偏下，微量后退"
            else:
                # 理论上应由 READY 捕获，保护性停止。
                print("  TENNIS ACTION: HOLD，等待下一帧确认。")

                try:
                    chassis.drive_speed(
                        x=0,
                        y=0,
                        z=0,
                        timeout=0.5,
                    )
                except Exception:
                    pass

                time.sleep(0.12)
                continue

        direction = "前进" if dx > 0 else "后退"

        print(
            f"  TENNIS ACTION: {direction} "
            f"{abs(dx)*1000:.0f}mm ({reason})"
        )

        chassis.move(
            x=float(dx),
            y=0,
            z=0,
            xy_speed=CHASSIS_XY_SPEED,
        ).wait_for_completed()

        # 下一次运动必须等新 YOLO seq。
        time.sleep(0.12)

    print("[TENNIS_CLOSE] 达到最大近距控制次数。")
    return None


def bottle_turn_step_from_x_error(x_error):
    """
    水瓶最终近距离阶段专用转向。

    V11 使用通用 3/6/10°，近距离下 3° 都偏大；
    V12 改成 1/2°，减少左右来回越过中心线。
    """
    a = abs(float(x_error))

    if a > 40.0:
        mag = BOTTLE_MID_TURN_DEG
    else:
        mag = BOTTLE_FINE_TURN_DEG

    return (
        TURN_SIGN_MOVE
        * math.copysign(
            mag,
            x_error,
        )
    )



def close_approach_bottle(
    chassis,
    vision,
    profile,
    args,
):
    """
    V12 水瓶最终近距离闭环。

    核心原则：
    1. 水瓶和网球继续完全独立。
    2. 水瓶近距不再使用40mm大步长和3°大转向。
    3. 对最近3个高可信框做中值滤波，减小YOLO框抖动。
    4. height 是主距离判据，bottom_y 只作为辅助；
       不允许“height已经过大但bottom_y偏小”时继续前冲。
    5. READY 连续2个高可信结果成立才抓，避免偶发框直接触发。
    """
    ref_x = float(profile["bottom_x_px"])
    ref_y = float(profile["bottom_y_px"])
    ref_h = float(profile["bbox_height_px"])

    lost_since = None
    ready_hits = 0
    history = deque(maxlen=BOTTLE_FILTER_WINDOW)

    # FINAL CLOSE 实例锁状态：
    # 一旦锁定当前瓶子，本轮绝不跳到旁边另一个 bottle。
    lock_acquired = False
    locked_det = None
    last_seq = vision.current_det_seq()
    step = 0

    while step < CLOSE_MAX_STEPS:
        if vision.is_stopped():
            raise KeyboardInterrupt
        vision.set_status(
            f"BOTTLE_CLOSE {step + 1}/{CLOSE_MAX_STEPS} "
            f"READY {ready_hits}/{BOTTLE_READY_REQUIRED_HITS}"
        )

        # --------------------------------------------------------
        # FINAL TARGET INSTANCE LOCK
        # --------------------------------------------------------
        seq, det_time, _, detections = vision.snapshot()
        if seq <= last_seq:
            if lost_since is None:
                lost_since = time.monotonic()
            if time.monotonic() - lost_since >= BOTTLE_OBSERVE_LOST_TIMEOUT_S:
                return None
            time.sleep(CLOSE_LOOP_PERIOD_S)
            continue
        last_seq = seq
        step += 1

        age = (
            time.monotonic() - det_time
            if det_time > 0
            else 999.0
        )

        if age > 1.5:
            detections = []

        if not lock_acquired:
            det = acquire_final_target_lock(
                detections=detections,
                class_name="bottle",
                ref_x=ref_x,
                ref_y=ref_y,
                ref_size=ref_h,
                size_key="height",
            )

            if det is not None:
                locked_det = det
                lock_acquired = True
                lost_since = None

                print(
                    "[FINAL_LOCK][bottle] 已锁定当前水瓶实例："
                    f"x={float(det['bottom_x']):.1f}, "
                    f"y={float(det['bottom_y']):.1f}, "
                    f"h={float(det['height']):.1f}"
                )
            else:
                ready_hits = 0

                if lost_since is None:
                    lost_since = time.monotonic()

                try:
                    chassis.drive_speed(
                        x=0,
                        y=0,
                        z=0,
                        timeout=0.5,
                    )
                except Exception:
                    pass

                if (
                    time.monotonic() - lost_since
                    >= BOTTLE_OBSERVE_LOST_TIMEOUT_S
                ):
                    print(
                        "[FINAL_LOCK][bottle] "
                        "最终姿态下没有找到可锁定的当前水瓶。"
                    )
                    return None

                time.sleep(CLOSE_LOOP_PERIOD_S)
                continue

        else:
            det = update_final_target_lock(
                detections=detections,
                locked_det=locked_det,
                class_name="bottle",
                size_key="height",
            )

            if det is None:
                ready_hits = 0
                history.clear()

                if lost_since is None:
                    lost_since = time.monotonic()

                print(
                    "[FINAL_LOCK][bottle] "
                    "锁定水瓶本帧暂时丢失；忽略其他目标，不切换。"
                )

                try:
                    chassis.drive_speed(
                        x=0,
                        y=0,
                        z=0,
                        timeout=0.5,
                    )
                except Exception:
                    pass

                if (
                    time.monotonic() - lost_since
                    >= BOTTLE_OBSERVE_LOST_TIMEOUT_S
                ):
                    print(
                        "[FINAL_LOCK][bottle] "
                        "锁定目标丢失超时，本轮失败恢复。"
                    )
                    return None

                time.sleep(CLOSE_LOOP_PERIOD_S)
                continue

            locked_det = det
            lost_since = None

        loc = {
            "class": "bottle",
            "conf": float(det["conf"]),
            "bottom_x": float(det["bottom_x"]),
            "bottom_y": float(det["bottom_y"]),
            "width": float(det["width"]),
            "height": float(det["height"]),
            "det_age": float(age),
        }

        # --------------------------------------------------------
        # 低可信度框只观察，不驱动底盘
        # --------------------------------------------------------
        if float(loc["conf"]) < BOTTLE_CLOSE_MIN_CONF:
            ready_hits = 0
            history.clear()

            print()
            print(
                f"[BOTTLE_CLOSE {step}] conf={loc['conf']:.2f} "
                f"< {BOTTLE_CLOSE_MIN_CONF:.2f}，"
                "本帧不执行运动，等待更稳定检测。"
            )

            try:
                chassis.drive_speed(
                    x=0,
                    y=0,
                    z=0,
                    timeout=0.5,
                )
            except Exception:
                pass

            time.sleep(0.12)
            continue

        raw_x_error = float(loc["bottom_x"]) - ref_x
        raw_h = float(loc["height"])
        raw_bottom_y = float(loc["bottom_y"])

        history.append(
            (
                raw_x_error,
                raw_h,
                raw_bottom_y,
            )
        )

        # 最近3个高可信框中值滤波；
        # 初期不足3个时就使用已有样本。
        x_error = float(np.median([v[0] for v in history]))
        h = float(np.median([v[1] for v in history]))
        bottom_y = float(np.median([v[2] for v in history]))

        print()
        print(
            f"[BOTTLE_CLOSE {step}] conf={loc['conf']:.2f} "
            f"raw: xerr={raw_x_error:+.1f}px "
            f"h={raw_h:.1f} bottom_y={raw_bottom_y:.1f}"
        )
        print(
            f"  FILTER: xerr={x_error:+.1f}px "
            f"height={h:.1f}/{ref_h:.1f} "
            f"bottom_y={bottom_y:.1f}"
        )

        # --------------------------------------------------------
        # 1. 横向：近距离使用 1°/2° 小角度
        # --------------------------------------------------------
        if abs(x_error) > BOTTLE_X_TOL_PX:
            ready_hits = 0
            history.clear()

            z = bottle_turn_step_from_x_error(x_error)
            direction = "右" if x_error > 0 else "左"

            print(
                f"  BOTTLE ACTION: FINE TURN "
                f"{direction} {z:+.1f}deg"
            )

            chassis.move(
                x=0,
                y=0,
                z=float(z),
                z_speed=CHASSIS_Z_SPEED,
            ).wait_for_completed()

            time.sleep(0.18)
            continue

        # --------------------------------------------------------
        # 2. READY：窗口适当放宽 + 连续2次确认
        # --------------------------------------------------------
        ready_now = (
            abs(x_error) <= BOTTLE_X_TOL_PX
            and BOTTLE_MIN_HEIGHT <= h <= BOTTLE_MAX_HEIGHT
            and bottom_y >= BOTTLE_MIN_BOTTOM_Y
        )

        if ready_now:
            ready_hits += 1

            print(
                f"[BOTTLE READY CHECK] "
                f"{ready_hits}/{BOTTLE_READY_REQUIRED_HITS} "
                f"(xerr={x_error:+.1f}, "
                f"h={h:.1f}, bottom_y={bottom_y:.1f})"
            )

            # 第一次READY先完全静止，再等下一次视觉确认。
            try:
                chassis.drive_speed(
                    x=0,
                    y=0,
                    z=0,
                    timeout=0.5,
                )
            except Exception:
                pass

            if ready_hits >= BOTTLE_READY_REQUIRED_HITS:
                print(
                    "[BOTTLE READY] 连续确认成功，"
                    "停止修正并进入抓取。"
                )
                vision.set_status("BOTTLE READY_TO_GRASP")
                return loc

            time.sleep(0.18)
            continue

        ready_hits = 0
        history.clear()

        # --------------------------------------------------------
        # 3. 前后距离控制
        #
        # 重要修复：
        # 先判断 height 是否已经过大，再考虑 bottom_y。
        # V11 是先 bottom_y<350 就前进，哪怕 height 已经301，
        # 会造成明显前后振荡。
        # --------------------------------------------------------

        # 已明显过近：小步后退
        if h > BOTTLE_MAX_HEIGHT:
            dx = -BOTTLE_FINE_BACK_M
            reason = "height偏大，优先小步后退"

        # 还明显很远：保持较大的靠近步长
        elif h < BOTTLE_FAR_H:
            dx = MID_MOVE_M
            reason = "仍较远"

        elif h < BOTTLE_MID_H:
            dx = MIN_MOVE_M
            reason = "中距离"

        # 接近最终窗口，只允许15mm小步前进
        elif h < BOTTLE_MIN_HEIGHT:
            dx = BOTTLE_FINE_FORWARD_M
            reason = "接近READY窗口，15mm微调"

        # height已经在READY窗口内，但bottom_y略低：
        # bottom_y容易抖，只允许10mm非常小的前进，绝不再40mm冲。
        elif bottom_y < BOTTLE_MIN_BOTTOM_Y:
            dx = BOTTLE_VERY_FINE_FORWARD_M
            reason = "height已合适，仅bottom_y略低，10mm微调"

        else:
            # 理论上会被 ready_now 捕获；
            # 留一个保护分支，保持停止而不是盲目移动。
            print(
                "  BOTTLE ACTION: HOLD "
                "(已接近READY区域，等待下一帧确认)"
            )

            try:
                chassis.drive_speed(
                    x=0,
                    y=0,
                    z=0,
                    timeout=0.5,
                )
            except Exception:
                pass

            time.sleep(0.15)
            continue

        direction = "前进" if dx > 0 else "后退"

        print(
            f"  BOTTLE ACTION: {direction} "
            f"{abs(dx)*1000:.0f}mm "
            f"({reason}; h={h:.1f}, bottom_y={bottom_y:.1f})"
        )

        chassis.move(
            x=float(dx),
            y=0,
            z=0,
            xy_speed=CHASSIS_XY_SPEED,
        ).wait_for_completed()

        time.sleep(0.20)

    print("[BOTTLE_CLOSE] 达到最大近距控制次数。")
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





def move_arm_to_ball_near_view(arm):
    """
    网球专用的“近距离中间观察姿态”。

    不是最终抓取姿态，只是在 recenter 基础上相对下降一小段，
    让相机/夹爪几何关系发生变化，避免网球靠近后被夹爪挡住。

    该动作只由主控制线程调用，不会与 AsyncVision 线程争抢机械臂。
    """
    print(
        f"[NEAR_VIEW] 网球已接近，机械臂先相对下降 "
        f"{abs(BALL_NEAR_VIEW_DROP_MM):.0f}mm，继续寻找目标。"
    )

    action = arm.move(
        x=0,
        y=BALL_NEAR_VIEW_DROP_MM,
    )

    _wait_arm_action(
        action,
        "tennis_ball near-view lower",
    )

    time.sleep(BALL_NEAR_VIEW_SETTLE_S)


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



def _latest_class_candidate(
    vision,
    class_name,
    previous=None,
    max_age=MAX_DET_AGE_S,
    unique_if_unlocked=False,
):
    """
    从最新YOLO结果中只取指定类别。
    """
    seq, det_time, _, detections = vision.snapshot()

    if det_time <= 0:
        return seq, None

    if (time.monotonic() - det_time) > max_age:
        return seq, None

    same = [
        d for d in detections
        if d["class"] == class_name
    ]

    if not same:
        return seq, None

    if previous is not None:
        size_key = "height" if class_name == "bottle" else "width"
        return seq, update_final_target_lock(same, previous, class_name, size_key)

    if unique_if_unlocked and len(same) != 1:
        return seq, None

    return seq, _priority_pick(
        same,
        previous=previous,
    )


def select_nearest_target_class(
    arm,
    vision,
    profiles,
    args,
):
    """
    扫描/选择阶段只做一件事：
    在高位观察姿态下选出当前最近目标的类别。

    选完后立刻分流，不在这里执行任何类别专属抓取动作。
    """
    move_arm_to_observe_pose(arm)

    vision.set_target("auto")
    vision.set_profile(None)
    vision.set_status("SELECT TARGET CLASS")

    loc = locate_stable(
        vision,
        timeout=TARGET_SELECT_TIMEOUT_S,
        window_size=TARGET_SELECT_WINDOW,
        required_hits=TARGET_SELECT_HITS,
    )

    if loc is None:
        return None, None

    cls = loc["class"]

    if cls not in profiles:
        return None, None

    print(
        f"[DISPATCH] 当前最近目标={cls} -> "
        f"{'TENNIS PIPELINE' if cls == 'tennis_ball' else 'BOTTLE PIPELINE'}"
    )

    return cls, loc


def approach_tennis_pipeline(
    arm,
    chassis,
    vision,
    profile,
    args,
    seed=None,
):
    """
    网球专用远距/中距流水线。

    recenter高位观察
      -> width接近阈值
      -> 停车
      -> 机械臂相对下降45mm（NEAR_VIEW）
      -> 重新识别网球
      -> 慢速对准/靠近
      -> 返回给最终学习抓取姿态
    """
    move_arm_to_observe_pose(arm)

    vision.set_target("tennis_ball")
    vision.set_profile(None)
    vision.set_status("TENNIS / APPROACH")

    previous = seed
    last_seq = -1
    last_seen = time.monotonic()

    current_x = 0.0
    current_z = 0.0

    near_view_lowered = False
    align_mode = False
    align_hits = 0

    for step in range(1, args.observe_max_steps + 1):
        if vision.is_stopped():
            raise KeyboardInterrupt

        seq, candidate = _latest_class_candidate(
            vision=vision,
            class_name="tennis_ball",
            previous=previous,
            max_age=MAX_DET_AGE_S,
            unique_if_unlocked=near_view_lowered,
        )

        if seq == last_seq:
            lost_limit = (
                BALL_NEAR_VIEW_LOST_TIMEOUT_S
                if near_view_lowered
                else OBSERVE_LOST_TIMEOUT_S
            )

            if time.monotonic() - last_seen > lost_limit:
                print("[TENNIS] 目标更新超时。")
                stop_chassis_smooth(chassis, current_x, current_z)
                return None

            time.sleep(0.02)
            continue

        last_seq = seq

        if candidate is None:
            lost_limit = (
                BALL_NEAR_VIEW_LOST_TIMEOUT_S
                if near_view_lowered
                else OBSERVE_LOST_TIMEOUT_S
            )

            if time.monotonic() - last_seen > lost_limit:
                print("[TENNIS] 网球丢失。")
                stop_chassis_smooth(chassis, current_x, current_z)
                return None

            continue

        previous = candidate
        last_seen = time.monotonic()

        x_error = float(candidate["bottom_x"]) - OBSERVE_CENTER_X
        width = float(candidate["width"])
        ratio = width / BALL_PREGRASP_WIDTH

        print(
            f"[TENNIS {step}] xerr={x_error:+.1f}px "
            f"width={width:.1f} switch={BALL_PREGRASP_WIDTH:.1f} "
            f"near_view={near_view_lowered}"
        )

        if (
            not align_mode
            and ratio >= TENNIS_ALIGN_ENTER_RATIO
        ):
            align_mode = True
            align_hits = 0

            stop_chassis_smooth(chassis, current_x, current_z)
            current_x = 0.0
            current_z = 0.0

            if not near_view_lowered:
                vision.set_status("TENNIS / NEAR_VIEW")
                move_arm_to_ball_near_view(arm)

                near_view_lowered = True
                previous = None
                last_seq = vision.current_det_seq()
                last_seen = time.monotonic()

                print(
                    "[TENNIS] 已进入网球中间观察姿态，"
                    "现在重新寻找网球。"
                )
                continue

        if align_mode:
            vision.set_status(
                f"TENNIS / ALIGN {align_hits}/{TENNIS_ALIGN_REQUIRED_HITS}"
            )

            # 网球已下降到中间观察姿态后，禁止退回高速APPROACH。
            if abs(x_error) > TENNIS_ALIGN_X_TOL_PX:
                align_hits = 0
                current_x = 0.0

                target_z = desired_turn_speed(x_error)
                current_z = clamp_step(
                    current_z,
                    target_z,
                    MAX_DZ_DPS,
                )

                direction = "右" if x_error > 0 else "左"

                print(
                    f"  TENNIS ALIGN: 偏{direction} {x_error:+.1f}px，"
                    "只转向，不前进"
                )

                chassis.drive_speed(
                    x=0,
                    y=0,
                    z=float(current_z),
                    timeout=0.5,
                )

                time.sleep(CONTROL_PERIOD_S)
                continue

            current_z = clamp_step(
                current_z,
                0.0,
                MAX_DZ_DPS,
            )

            if width < BALL_PREGRASP_WIDTH:
                align_hits = 0
                current_x = PREGRASP_CREEP_MPS

                print(
                    f"  TENNIS ALIGN: 已居中，慢速前进 "
                    f"{current_x:.2f}m/s"
                )

                chassis.drive_speed(
                    x=current_x,
                    y=0,
                    z=float(current_z),
                    timeout=0.5,
                )

                time.sleep(CONTROL_PERIOD_S)
                continue

            align_hits += 1
            current_x = 0.0

            print(
                f"  TENNIS ALIGN READY "
                f"{align_hits}/{TENNIS_ALIGN_REQUIRED_HITS}"
            )

            if align_hits < TENNIS_ALIGN_REQUIRED_HITS:
                chassis.drive_speed(
                    x=0,
                    y=0,
                    z=float(current_z),
                    timeout=0.5,
                )
                time.sleep(CONTROL_PERIOD_S)
                continue

            stop_chassis_smooth(chassis, current_x, current_z)
            return candidate

        # 高位远距离阶段
        target_x = desired_approach_speed(
            "tennis_ball",
            width,
        )

        if abs(x_error) > 120:
            target_x = 0.0
        elif abs(x_error) > 70:
            target_x = min(
                target_x,
                APPROACH_SLOW_MPS,
            )

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

        chassis.drive_speed(
            x=float(current_x),
            y=0,
            z=float(current_z),
            timeout=0.5,
        )

        time.sleep(CONTROL_PERIOD_S)

    stop_chassis_smooth(chassis, current_x, current_z)
    return None


def approach_bottle_pipeline(
    arm,
    chassis,
    vision,
    profile,
    args,
    seed=None,
):
    """
    水瓶专用远距/中距流水线。

    这条线路刻意不包含任何网球逻辑：
    - 不执行 BALL_NEAR_VIEW_DROP_MM
    - 不使用 tennis width 阈值
    - 不进入 tennis near-view

    recenter高位观察
      -> bottle height 接近170px
      -> 横向对准
      -> 慢速靠近
      -> height>=170 且居中连续2帧
      -> 进入水瓶最终学习抓取姿态
    """
    move_arm_to_observe_pose(arm)

    vision.set_target("bottle")
    vision.set_profile(None)
    vision.set_status("BOTTLE / APPROACH")

    previous = seed
    last_seq = -1
    last_seen = time.monotonic()

    current_x = 0.0
    current_z = 0.0

    align_mode = False
    align_hits = 0

    for step in range(1, args.observe_max_steps + 1):
        if vision.is_stopped():
            raise KeyboardInterrupt

        seq, candidate = _latest_class_candidate(
            vision=vision,
            class_name="bottle",
            previous=previous,
            # 640推理更慢
            max_age=1.5,
        )

        if seq == last_seq:
            if time.monotonic() - last_seen > BOTTLE_OBSERVE_LOST_TIMEOUT_S:
                print("[BOTTLE] 目标更新超时。")
                stop_chassis_smooth(chassis, current_x, current_z)
                return None

            time.sleep(0.02)
            continue

        last_seq = seq

        if candidate is None:
            if time.monotonic() - last_seen > BOTTLE_OBSERVE_LOST_TIMEOUT_S:
                print("[BOTTLE] 水瓶丢失。")
                stop_chassis_smooth(chassis, current_x, current_z)
                return None
            continue

        previous = candidate
        last_seen = time.monotonic()

        x_error = float(candidate["bottom_x"]) - OBSERVE_CENTER_X
        height = float(candidate["height"])
        ratio = height / BOTTLE_PREGRASP_HEIGHT

        print(
            f"[BOTTLE {step}] xerr={x_error:+.1f}px "
            f"height={height:.1f} switch={BOTTLE_PREGRASP_HEIGHT:.1f}"
        )

        if (
            not align_mode
            and ratio >= BOTTLE_ALIGN_ENTER_RATIO
        ):
            align_mode = True
            align_hits = 0

            stop_chassis_smooth(chassis, current_x, current_z)
            current_x = 0.0
            current_z = 0.0

            print(
                "[BOTTLE ALIGN] 已进入水瓶预抓取区域。"
            )

        if align_mode:
            vision.set_status(
                f"BOTTLE / ALIGN {align_hits}/{BOTTLE_ALIGN_REQUIRED_HITS}"
            )

            # 如果检测明显又变远，水瓶可回到普通APPROACH。
            if ratio < 0.72:
                align_mode = False
                align_hits = 0
                continue

            if abs(x_error) > BOTTLE_ALIGN_X_TOL_PX:
                align_hits = 0
                current_x = 0.0

                target_z = desired_turn_speed(x_error)
                current_z = clamp_step(
                    current_z,
                    target_z,
                    MAX_DZ_DPS,
                )

                direction = "右" if x_error > 0 else "左"

                print(
                    f"  BOTTLE ALIGN: 偏{direction} {x_error:+.1f}px，"
                    "只转向，不前进"
                )

                chassis.drive_speed(
                    x=0,
                    y=0,
                    z=float(current_z),
                    timeout=0.5,
                )

                time.sleep(CONTROL_PERIOD_S)
                continue

            current_z = clamp_step(
                current_z,
                0.0,
                MAX_DZ_DPS,
            )

            if height < BOTTLE_PREGRASP_HEIGHT:
                align_hits = 0
                current_x = PREGRASP_CREEP_MPS

                print(
                    f"  BOTTLE ALIGN: 已居中，慢速前进 "
                    f"{current_x:.2f}m/s"
                )

                chassis.drive_speed(
                    x=current_x,
                    y=0,
                    z=float(current_z),
                    timeout=0.5,
                )

                time.sleep(CONTROL_PERIOD_S)
                continue

            align_hits += 1
            current_x = 0.0

            print(
                f"  BOTTLE ALIGN READY "
                f"{align_hits}/{BOTTLE_ALIGN_REQUIRED_HITS}"
            )

            if align_hits < BOTTLE_ALIGN_REQUIRED_HITS:
                chassis.drive_speed(
                    x=0,
                    y=0,
                    z=float(current_z),
                    timeout=0.5,
                )

                time.sleep(CONTROL_PERIOD_S)
                continue

            stop_chassis_smooth(chassis, current_x, current_z)
            return candidate

        # 水瓶远距离阶段
        target_x = desired_approach_speed(
            "bottle",
            height,
        )

        if abs(x_error) > 120:
            target_x = 0.0
        elif abs(x_error) > 70:
            target_x = min(
                target_x,
                APPROACH_SLOW_MPS,
            )

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

        chassis.drive_speed(
            x=float(current_x),
            y=0,
            z=float(current_z),
            timeout=0.5,
        )

        time.sleep(CONTROL_PERIOD_S)

    stop_chassis_smooth(chassis, current_x, current_z)
    return None



def pregrasp_with_recovery(
    arm,
    chassis,
    vision,
    profiles,
    args,
):
    """
    V11 核心调度器。

    扫描到什么类别，就进入哪一条完全独立的抓取线路：

        tennis_ball:
            select -> tennis approach -> tennis near-view
            -> final tennis pose -> tennis close

        bottle:
            select -> bottle approach
            -> final bottle pose -> bottle close

    两条线路之间不共享 near-view/align 状态。
    """
    last_failure = None
    for attempt in range(PREGRASP_RECOVERY_LIMIT + 1):

        # 1) 先确定本轮抓哪一类
        if args.target in ("tennis_ball", "bottle"):
            cls = args.target
            seed = None
        else:
            cls, seed = select_nearest_target_class(
                arm=arm,
                vision=vision,
                profiles=profiles,
                args=args,
            )

        if cls is None:
            return None, None, last_failure or "not_found"

        profile = profiles[cls]

        print()
        print("============================================================")
        print(
            f"[PIPELINE] 本轮进入 "
            f"{'网球' if cls == 'tennis_ball' else '水瓶'} 独立线路"
        )
        print("============================================================")

        # 2) 类别专属 approach
        if cls == "tennis_ball":
            approach_loc = approach_tennis_pipeline(
                arm=arm,
                chassis=chassis,
                vision=vision,
                profile=profile,
                args=args,
                seed=seed,
            )
        else:
            approach_loc = approach_bottle_pipeline(
                arm=arm,
                chassis=chassis,
                vision=vision,
                profile=profile,
                args=args,
                seed=seed,
            )

        if approach_loc is None:
            last_failure = "approach_failed"
            print(f"[PIPELINE] {cls} approach失败。")

            if attempt < PREGRASP_RECOVERY_LIMIT:
                move_arm_to_observe_pose(arm)
                vision.set_target("auto" if args.target == "auto" else args.target)
                vision.set_profile(None)
                time.sleep(0.35)
                continue

            return None, None, last_failure

        if args.dry_run or args.nav_only:
            return cls, profile, approach_loc

        # 3) 类别自己的最终学习机械臂姿态
        vision.set_target(cls)
        vision.set_profile(profile)
        vision.set_status(
            f"{cls.upper()} / FINAL ARM POSE"
        )

        move_arm_to_profile_pose(
            arm,
            profile,
            f"{cls} 最终学习抓取姿态",
        )

        time.sleep(0.25)

        # 4) 类别专属最终近距离闭环
        if cls == "tennis_ball":
            loc = close_approach_tennis(
                chassis=chassis,
                vision=vision,
                profile=profile,
                args=args,
            )
        else:
            loc = close_approach_bottle(
                chassis=chassis,
                vision=vision,
                profile=profile,
                args=args,
            )

        if loc is not None:
            return cls, profile, loc

        print(f"[PIPELINE] {cls} final close失败。")
        last_failure = "close_failed"

        if attempt < PREGRASP_RECOVERY_LIMIT:
            move_arm_to_observe_pose(arm)
            vision.set_profile(None)
            vision.set_target("auto" if args.target == "auto" else args.target)
            time.sleep(0.35)
            continue

        return None, None, last_failure

    return None, None, last_failure or "not_found"



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

def configure_gripper_tennis():
    """
    网球夹爪参数。
    """
    exp2.GRIP_POWER = 30
    exp2.GRIP_CLOSED_FAST = 2.0
    exp2.GRIP_STABLE_TIME = 2.5
    exp2.GRIP_STATUS_TIMEOUT = 8.0


def configure_gripper_bottle():
    """
    水瓶夹爪参数。

    当前先保持之前已经成功抓到水瓶时使用的参数，
    但入口已经与网球彻底分开，后续可只改水瓶而不影响网球。
    """
    exp2.GRIP_POWER = 30
    exp2.GRIP_CLOSED_FAST = 2.0
    exp2.GRIP_STABLE_TIME = 2.5
    exp2.GRIP_STATUS_TIMEOUT = 8.0


def configure_gripper(class_name):
    if class_name == "bottle":
        configure_gripper_bottle()
    else:
        configure_gripper_tennis()



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

    _wait_arm_action(arm.move(x=0, y=TEST_LIFT), "first lift")

    time.sleep(0.4)

    _wait_arm_action(arm.move(x=0, y=MAIN_LIFT), "main lift")

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



def place_outside_grid(arm, chassis, gripper, class_name, vision, args):
    """Sidestep from the entrance beyond the outer line, release, and return."""
    held_status = str(exp2.gripper_status["value"]).strip().lower()
    held_fresh = time.time() - exp2.gripper_status.get("ts", 0.0) < 1.0
    if held_status == "opened" and held_fresh:
        raise RouteError("搬运中夹爪已张开，不能按成功抓取放置")
    side = 1.0 if class_name == "tennis_ball" else -1.0
    lateral = side * args.left_y_sign * (
        args.grid_width_m / 2.0 + args.side_clearance_m
    )
    zone = "LEFT" if side > 0 else "RIGHT"
    print(f"[SORT] {class_name} -> {zone}, y={lateral:+.3f}m")
    vision.set_status(f"SORT / {zone}")

    _wait_chassis_action(chassis.move(y=lateral, xy_speed=GRID_SIDE_SPEED), "to sort side")
    _wait_arm_action(arm.move(x=0, y=-DROP_LOWER_MM), "grid drop lower")

    exp2.gripper_status["value"] = "unknown"
    exp2.gripper_status["ts"] = 0.0
    if gripper.open(power=DROP_OPEN_POWER) is False:
        raise RouteError("夹爪张开命令未被接受")
    opened_deadline = time.monotonic() + 4.0
    while time.monotonic() < opened_deadline:
        status = str(exp2.gripper_status["value"]).strip().lower()
        fresh = time.time() - exp2.gripper_status.get("ts", 0.0) < 1.0
        if status == "opened" and fresh:
            break
        time.sleep(0.1)
    else:
        raise RouteError("夹爪未报告 opened，放置结果无法确认")

    _wait_arm_action(arm.move(x=0, y=DROP_LOWER_MM), "grid drop lift")
    _wait_chassis_action(chassis.move(y=-lateral, xy_speed=GRID_SIDE_SPEED), "from sort side")
    vision.set_target("auto")
    vision.set_profile(None)
    vision.set_status("HOME / NEXT TARGET")
    return zone


def _wait_chassis_action(action, label):
    if not action.wait_for_completed(timeout=ROUTE_MOVE_TIMEOUT_S):
        raise RouteError(f"底盘动作未完成: {label}")


def write_task_event(log_file, **event):
    event["time"] = datetime.now().isoformat(timespec="seconds")
    log_file.write(json.dumps(event, ensure_ascii=False) + "\n")
    log_file.flush()


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
        help="兼容旧参数：只测试接近和退出，但底盘与机械臂仍会运动",
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

    p.add_argument("--grid-width-m", type=float, default=0.0,
                   help="网格两条外侧边线之间的宽度，分拣模式必须填写")
    p.add_argument("--side-clearance-m", type=float, default=0.25,
                   help="车中心越过网格外侧边线的额外距离")
    p.add_argument("--left-y-sign", type=int, choices=(-1, 1), default=-1,
                   help="现场左移对应的 y 符号；SDK 示例为 -1")
    p.add_argument("--max-task-failures", type=int, default=MAX_TASK_FAILURES)
    p.add_argument("--home-pos-tol-m", type=float, default=0.15)
    p.add_argument("--home-angle-tol-deg", type=float, default=12.0)
    p.add_argument("--log-dir", default=os.path.expanduser("~/exp3_logs"))

    return p.parse_args()


def main():
    args = parse_args()

    if args.sort_mode and (
        args.grid_width_m <= 0
        or args.side_clearance_m <= 0
        or args.grid_width_m / 2 + args.side_clearance_m > 5.0
        or args.max_items <= 0
        or args.empty_retries <= 0
        or args.max_task_failures <= 0
        or args.home_pos_tol_m <= 0
        or args.home_angle_tol_deg <= 0
    ):
        print("ERROR: 请填写有效网格宽度、边线余量和正整数任务上限。")
        return 2

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

    if args.sort_mode and any(cls not in profiles for cls in CLASS_NAMES):
        print("ERROR: 连续分类必须同时提供网球和水瓶两个抓取 profile。")
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
    chassis = None
    log_file = None
    home_monitor = None
    position_subscribed = False

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
        chassis = RouteChassis(ep.chassis) if args.sort_mode else ep.chassis
        gripper = ep.gripper
        camera = ep.camera

        gripper.sub_status(
            freq=5,
            callback=exp2.on_gripper_status,
        )

        subscribed = True

        if args.sort_mode:
            home_monitor = HomePoseMonitor()
            if not ep.chassis.sub_position(cs=1, freq=5, callback=home_monitor.callback):
                raise RouteError("底盘位置订阅失败")
            position_subscribed = True
            home_monitor.capture_home()

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

        # Select the visually nearest object, then return to the same entrance
        # before every lateral sorting trip.

        if not args.sort_mode:
            # 保留单目标测试模式
            cls, profile, reason = pregrasp_with_recovery(
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
        failure_count = 0
        tennis_count = 0
        bottle_count = 0

        os.makedirs(args.log_dir, exist_ok=True)
        log_path = os.path.join(
            args.log_dir, f"exp3_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        log_file = open(log_path, "w", encoding="utf-8")
        print("任务日志:", log_path)
        write_task_event(log_file, event="task_start", grid_width_m=args.grid_width_m,
                         side_clearance_m=args.side_clearance_m, left_y_sign=args.left_y_sign,
                         home_pose=home_monitor.home)

        # 多物体分拣始终允许同时识别两类
        args.target = "auto"
        vision.set_target("auto")
        vision.set_profile(None)

        print()
        print("============================================================")
        print("开始多目标自动分拣")
        print("优先级：bottom_y 最大的目标优先（最近优先）")
        print("网球 -> 网格左边线外；水瓶 -> 网格右边线外")
        print(f"最多处理 {args.max_items} 个目标")
        print("============================================================")

        while sorted_count < args.max_items:
            chassis.begin()
            vision.set_target("auto")
            vision.set_profile(None)
            vision.set_status(
                f"SEARCH NEXT {sorted_count + 1}/{args.max_items}"
            )

            print()
            print(
                f"========== 第 {sorted_count + 1} 个目标 =========="
            )

            cls, profile, reason = pregrasp_with_recovery(
                arm=arm,
                chassis=chassis,
                vision=vision,
                profiles=profiles,
                args=args,
            )

            if cls is None or profile is None:
                move_arm_to_observe_pose(arm)
                chassis.retrace()
                home_monitor.assert_home(args.home_pos_tol_m, args.home_angle_tol_deg)
                vision.set_target("auto")
                vision.set_profile(None)
                write_task_event(log_file, event="selection_failed", reason=reason)
                if reason == "not_found":
                    empty_count += 1
                    print(f"[SEARCH] 未找到目标 ({empty_count}/{args.empty_retries})")
                    if empty_count >= args.empty_retries:
                        break
                else:
                    empty_count = 0
                    failure_count += 1
                    print(f"[TASK] {reason} ({failure_count}/{args.max_task_failures})")
                    if failure_count >= args.max_task_failures:
                        raise RouteError("定位连续失败，停止分拣")
                time.sleep(0.6)
                continue

            empty_count = 0

            print(
                f"[PRIORITY] 当前最近目标: {cls}"
            )
            write_task_event(log_file, event="target_selected", class_name=cls,
                             recognition=reason)

            if args.nav_only or args.dry_run:
                move_arm_to_observe_pose(arm)
                chassis.retrace()
                home_monitor.assert_home(args.home_pos_tol_m, args.home_angle_tol_deg)
                write_task_event(log_file, event="navigation_test", class_name=cls)
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
                chassis.retrace()
                home_monitor.assert_home(args.home_pos_tol_m, args.home_angle_tol_deg)
                vision.set_target("auto")
                vision.set_profile(None)
                failure_count += 1
                write_task_event(log_file, event="grasp_failed", class_name=cls)
                if failure_count >= args.max_task_failures:
                    raise RouteError("抓取连续失败，停止分拣")
                time.sleep(0.5)
                continue

            write_task_event(log_file, event="grasp_success", class_name=cls)
            chassis.retrace()
            home_monitor.assert_home(args.home_pos_tol_m, args.home_angle_tol_deg)

            zone = place_outside_grid(
                arm=arm,
                chassis=chassis,
                gripper=gripper,
                class_name=cls,
                vision=vision,
                args=args,
            )
            home_monitor.assert_home(args.home_pos_tol_m, args.home_angle_tol_deg)
            move_arm_to_observe_pose(arm)
            write_task_event(log_file, event="placement_released", class_name=cls,
                             zone=zone, result="gripper_opened_home_returned")

            sorted_count += 1
            failure_count = 0

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
        write_task_event(log_file, event="task_complete", placed_count=sorted_count,
                         tennis_count=tennis_count, bottle_count=bottle_count)

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
        if log_file is not None:
            write_task_event(log_file, event="interrupted")
        return 130

    except Exception as e:
        print("任务停止:", repr(e))
        if initialized:
            try:
                exp2.robot_signal_error(ep)
            except Exception:
                pass
        if vision is not None:
            vision.set_status("TASK STOPPED")
        if log_file is not None:
            write_task_event(log_file, event="exception", detail=repr(e))
        return 7

    finally:
        if chassis is not None:
            try:
                chassis.drive_speed(x=0, y=0, z=0, timeout=0.5)
            except Exception:
                pass
        if log_file is not None:
            log_file.close()
        if position_subscribed:
            try:
                ep.chassis.unsub_position()
            except Exception:
                pass
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
