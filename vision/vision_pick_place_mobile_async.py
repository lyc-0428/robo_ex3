#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RoboMaster EP 实验三：异步视频优先版

结构：
    相机线程：一直以最快速度读取最新帧，不等待 YOLO
        ↓
    显示线程：默认 30 FPS 显示最新视频帧
        ↓
    YOLO线程：独立、较低频率处理“最新一帧”，完成后只更新检测结果
        ↓
    控制线程：读取连续稳定的检测结果，执行转向 / 前进后退 / 抓取

这样 YOLO 推理、底盘 wait_for_completed()、机械臂动作都不会阻塞视频显示。

默认：
- RoboMaster 相机：360p = 640x360
- 显示目标：30 FPS
- YOLO：320x320
- YOLO 更新：最多 6 FPS
- yaw_sign = -1（已按实机验证修正）
- 靠近目标：d = 220 ± 15 mm
- 颜色：不交换 R/B
"""

import argparse
import json
import math
import os
import sys
import threading
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODEL_DEFAULT = "/home/adam/team21_ex3_ws/src/robo_ex3/bottle_tennisball_best.pt"
CALIB_DEFAULT = os.path.join(HERE, "calib.json")

if not os.environ.get("DISPLAY"):
    os.environ["DISPLAY"] = ":0"
    try:
        for name in sorted(os.listdir("/tmp/.X11-unix")):
            if name.startswith("X"):
                os.environ["DISPLAY"] = ":" + name[1:]
                break
    except OSError:
        pass

if not os.environ.get("XAUTHORITY"):
    for cand in (
        "/run/user/1000/gdm/Xauthority",
        os.path.expanduser("~/.Xauthority"),
    ):
        if os.path.exists(cand):
            os.environ["XAUTHORITY"] = cand
            break

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from robomaster import robot

from real import pick_place_lua_params as exp2


CLASS_NAMES = ("bottle", "tennis_ball")

RES_SIZE = {
    "360p": (640, 360),
    "540p": (960, 540),
    "720p": (1280, 720),
}

GRASP_R_MIN = 195.0
GRASP_R_MAX = 250.0
GRASP_R_TARGET = 220.0

NAV_ESTIMATE_MIN = 145.0
NAV_ESTIMATE_MAX = 500.0
YAW_TOL_DEG = 4.0
MAX_TURN_STEP_DEG = 25.0
MAX_MOVE_STEP_M = 0.08
MAX_SERVO_STEPS = 8

CHASSIS_XY_SPEED = 0.15
CHASSIS_Z_SPEED = 20

EXTRA_FORWARD = 30.0
COARSE_FORWARD = 60.0
COARSE_DOWN = -240.0
FINAL_FORWARD = 6.0
FINAL_DOWN = -24.0
TEST_LIFT = 25.0
MAIN_LIFT = 60.0
PRE_Y = 120.0
TOTAL_REL_FORWARD = EXTRA_FORWARD + COARSE_FORWARD + FINAL_FORWARD

OPEN_POWER = 35
RETURN_DOWN = -(TEST_LIFT + MAIN_LIFT)
RETURN_LIFT = 70.0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def frame_to_bgr(frame):
    # 当前 PyAV 解码帧可直接供 OpenCV / Ultralytics 使用。
    return np.ascontiguousarray(frame)


def load_calib(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    H = np.asarray(data["H"], dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError("calib.json 的 H 不是 3x3")

    calib_res = data.get("resolution", "360p")
    if calib_res not in RES_SIZE:
        raise ValueError("不支持的标定分辨率: " + str(calib_res))

    return H, data, calib_res


def pixel_to_ground(H, u, v):
    p = np.array([[[float(u), float(v)]]], dtype=np.float32)
    q = cv2.perspectiveTransform(p, H)[0, 0]
    return float(q[0]), float(q[1])


def detect_one_frame(
    model,
    frame_bgr,
    H,
    calib_res,
    conf,
    target,
    imgsz,
    use_cuda,
):
    h, w = frame_bgr.shape[:2]
    calib_w, calib_h = RES_SIZE[calib_res]

    result = model.predict(
        source=frame_bgr,
        imgsz=imgsz,
        conf=conf,
        verbose=False,
        device=0 if use_cuda else "cpu",
        half=use_cuda,
    )[0]

    detections = []

    if result.boxes is not None:
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

            # homography 使用物体底边中心，而不是悬空的 bbox 几何中心
            gx = cx
            gy = y2

            u_cal = gx * float(calib_w) / float(w)
            v_cal = gy * float(calib_h) / float(h)

            d, l = pixel_to_ground(H, u_cal, v_cal)
            r = math.hypot(d, l)
            yaw = math.degrees(math.atan2(l, d))

            detections.append({
                "class": name,
                "conf": score,
                "box": (x1, y1, x2, y2),
                "center": (cx, cy),
                "ground_px": (gx, gy),
                "d": d,
                "l": l,
                "r": r,
                "yaw": yaw,
            })

    detections.sort(key=lambda x: (-x["conf"], x["r"]))
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
    """
    三线程视觉：
    1) capture：只负责持续取最新帧
    2) inference：只负责 YOLO，永远处理最新帧，不堆积旧帧
    3) display：固定刷新视频，使用最近一次检测结果画 overlay
    """

    def __init__(
        self,
        camera,
        model,
        H,
        calib_res,
        target,
        conf,
        imgsz,
        infer_fps,
        display_fps,
        window_name,
    ):
        self.camera = camera
        self.model = model
        self.H = H
        self.calib_res = calib_res
        self.target = target
        self.conf = conf
        self.imgsz = imgsz
        self.infer_fps = max(0.5, float(infer_fps))
        self.display_fps = max(5.0, float(display_fps))
        self.window_name = window_name

        self.use_cuda = torch.cuda.is_available()

        self.stop_event = threading.Event()
        self.user_quit = False

        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.frame_seq = 0
        self.frame_time = 0.0

        self.det_lock = threading.Lock()
        self.latest_best = None
        self.latest_detections = []
        self.det_seq = 0
        self.det_frame_seq = 0
        self.det_time = 0.0

        self.status_lock = threading.Lock()
        self.status = "starting..."
        self.stable_n = 0
        self.stable_need = 0

        self.capture_meter = FPSMeter()
        self.infer_meter = FPSMeter()
        self.display_meter = FPSMeter()

        self.threads = []

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
            t.join(timeout=1.5)

    def is_stopped(self):
        return self.stop_event.is_set()

    def set_status(self, text, stable_n=None, stable_need=None):
        with self.status_lock:
            self.status = str(text)
            if stable_n is not None:
                self.stable_n = int(stable_n)
            if stable_need is not None:
                self.stable_need = int(stable_need)

    def current_det_seq(self):
        with self.det_lock:
            return self.det_seq

    def detection_snapshot(self):
        with self.det_lock:
            best = None if self.latest_best is None else dict(self.latest_best)
            detections = [dict(x) for x in self.latest_detections]
            return (
                self.det_seq,
                self.det_frame_seq,
                self.det_time,
                best,
                detections,
            )

    def _capture_loop(self):
        while not self.stop_event.is_set():
            try:
                frame = self.camera.read_cv2_image(
                    strategy="newest",
                    timeout=1,
                )
            except TypeError:
                frame = self.camera.read_cv2_image(strategy="newest")
            except Exception as e:
                print("camera capture warning:", repr(e))
                time.sleep(0.05)
                continue

            if frame is None:
                continue

            bgr = frame_to_bgr(frame)

            with self.frame_lock:
                self.latest_frame = bgr
                self.frame_seq += 1
                self.frame_time = time.monotonic()

            self.capture_meter.tick()

    def _get_latest_frame_copy(self):
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

            frame, frame_seq = self._get_latest_frame_copy()
            if frame is None or frame_seq == last_frame_seq:
                time.sleep(0.005)
                continue

            # 核心：永远跳过积压帧，只推理“此刻最新帧”
            last_frame_seq = frame_seq
            last_run = time.monotonic()

            try:
                best, detections = detect_one_frame(
                    model=self.model,
                    frame_bgr=frame,
                    H=self.H,
                    calib_res=self.calib_res,
                    conf=self.conf,
                    target=self.target,
                    imgsz=self.imgsz,
                    use_cuda=self.use_cuda,
                )
            except Exception as e:
                print("YOLO inference warning:", repr(e))
                time.sleep(0.05)
                continue

            with self.det_lock:
                self.latest_best = best
                self.latest_detections = detections
                self.det_seq += 1
                self.det_frame_seq = frame_seq
                self.det_time = time.monotonic()

            self.infer_meter.tick()

    def _display_loop(self):
        period = 1.0 / self.display_fps

        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 960, 540)

            while not self.stop_event.is_set():
                loop_start = time.monotonic()

                frame, frame_seq = self._get_latest_frame_copy()
                if frame is not None:
                    (
                        det_seq,
                        det_frame_seq,
                        det_time,
                        best,
                        detections,
                    ) = self.detection_snapshot()

                    with self.status_lock:
                        status = self.status
                        stable_n = self.stable_n
                        stable_need = self.stable_need

                    shown = self._draw_overlay(
                        frame,
                        detections,
                        best,
                        status,
                        stable_n,
                        stable_need,
                        frame_seq,
                        det_frame_seq,
                        det_time,
                    )

                    cv2.imshow(self.window_name, shown)
                    self.display_meter.tick()

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
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
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass

    def _draw_overlay(
        self,
        frame,
        detections,
        best,
        status,
        stable_n,
        stable_need,
        frame_seq,
        det_frame_seq,
        det_time,
    ):
        shown = frame.copy()
        h, w = shown.shape[:2]

        cv2.drawMarker(
            shown,
            (w // 2, h // 2),
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )

        # YOLO 结果会比当前视频帧稍旧，这是有意设计：
        # 视频优先保持流畅，检测框在下一次推理完成后更新。
        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det["box"]]
            cx, cy = [int(round(v)) for v in det["center"]]
            gx, gy = [int(round(v)) for v in det["ground_px"]]

            color = (0, 255, 0) if best and det == best else (0, 180, 255)

            cv2.rectangle(shown, (x1, y1), (x2, y2), color, 2)
            cv2.circle(shown, (cx, cy), 5, (255, 0, 0), -1)
            cv2.circle(shown, (gx, gy), 6, (0, 0, 255), -1)

            label = f"{det['class']} {det['conf']:.2f}"
            geo = (
                f"d={det['d']:.0f} l={det['l']:.0f} "
                f"r={det['r']:.0f} yaw={det['yaw']:.1f}"
            )

            cv2.putText(
                shown,
                label,
                (x1, max(24, y1 - 28)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
            )
            cv2.putText(
                shown,
                geo,
                (x1, max(45, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                color,
                1,
            )

        cv2.rectangle(shown, (0, 0), (w, 72), (0, 0, 0), -1)

        cap_fps = self.capture_meter.value()
        inf_fps = self.infer_meter.value()
        disp_fps = self.display_meter.value()

        det_age_ms = 0.0
        if det_time > 0:
            det_age_ms = (time.monotonic() - det_time) * 1000.0

        line1 = (
            f"{status} | VIDEO {cap_fps:.1f}fps "
            f"DISPLAY {disp_fps:.1f}fps YOLO {inf_fps:.1f}fps"
        )
        line2 = (
            f"stable {stable_n}/{stable_need} | "
            f"det age {det_age_ms:.0f}ms | "
            f"video frame {frame_seq} / det frame {det_frame_seq}"
        )

        cv2.putText(
            shown,
            line1,
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            shown,
            line2,
            (10, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        return shown


def locate_stable(vision, stable_frames, timeout, status):
    """
    只消费“新的 YOLO 结果”，不参与视频读取和显示。
    调用时记录当前 det_seq，因此底盘移动前的旧检测不会被下一轮控制使用。
    """
    vision.set_status(status, 0, stable_frames)

    history = []
    track_class = None
    last_dl = None

    start = time.monotonic()
    last_seen_seq = vision.current_det_seq()

    while time.monotonic() - start < timeout:
        if vision.is_stopped():
            raise KeyboardInterrupt

        det_seq, _, _, best, _ = vision.detection_snapshot()

        if det_seq <= last_seen_seq:
            time.sleep(0.01)
            continue

        last_seen_seq = det_seq

        if best is None:
            history = []
            track_class = None
            last_dl = None
            vision.set_status(status, 0, stable_frames)
            continue

        dl = np.array([best["d"], best["l"]], dtype=float)

        same = (
            track_class == best["class"]
            and last_dl is not None
            and np.linalg.norm(dl - last_dl) < 45.0
        )

        if same:
            history.append(best)
        else:
            history = [best]
            track_class = best["class"]

        last_dl = dl
        history = history[-stable_frames:]
        vision.set_status(status, len(history), stable_frames)

        if len(history) >= stable_frames:
            cls = history[-1]["class"]
            d = float(np.median([x["d"] for x in history]))
            l = float(np.median([x["l"] for x in history]))
            r = math.hypot(d, l)
            yaw = math.degrees(math.atan2(l, d))
            confidence = float(np.median([x["conf"] for x in history]))

            return {
                "class": cls,
                "conf": confidence,
                "d": d,
                "l": l,
                "r": r,
                "yaw": yaw,
            }

    return None


def visual_servo_to_grasp_zone(
    arm,
    chassis,
    vision,
    args,
):
    for step_i in range(1, args.max_servo_steps + 1):
        vision.set_status(
            f"SERVO {step_i}: arm recenter",
            0,
            args.stable_frames,
        )

        # 即使这里阻塞，视频线程仍继续显示。
        arm.recenter().wait_for_completed()
        time.sleep(0.4)

        print(f"\n=== VISUAL SERVO {step_i}/{args.max_servo_steps} ===")

        loc = locate_stable(
            vision=vision,
            stable_frames=args.stable_frames,
            timeout=args.target_timeout,
            status=f"SERVO {step_i}: locating {args.target}",
        )

        if loc is None:
            print("没有稳定识别到目标。")
            return None

        print(
            "定位:",
            loc["class"],
            f"conf={loc['conf']:.3f}",
            f"d={loc['d']:.1f}mm",
            f"l={loc['l']:.1f}mm",
            f"r={loc['r']:.1f}mm",
            f"yaw={loc['yaw']:.1f}deg",
        )

        if not (NAV_ESTIMATE_MIN <= loc["r"] <= NAV_ESTIMATE_MAX):
            print(
                f"ERROR: r={loc['r']:.1f}mm 超出自动导航可信区 "
                f"[{NAV_ESTIMATE_MIN:.0f},{NAV_ESTIMATE_MAX:.0f}]mm。"
            )
            return None

        if abs(loc["yaw"]) > args.yaw_tol:
            turn = clamp(
                args.yaw_sign * loc["yaw"],
                -args.max_turn_step,
                args.max_turn_step,
            )

            vision.set_status(
                f"TURN {turn:+.1f} deg",
                0,
                args.stable_frames,
            )

            print(
                f"目标位于机器人{'右' if loc['l'] > 0 else '左'}侧，"
                f"yaw={loc['yaw']:+.1f}deg，"
                f"执行转向 {turn:+.1f}deg。"
            )

            if args.dry_run:
                return loc

            chassis.move(
                x=0,
                y=0,
                z=float(turn),
                z_speed=CHASSIS_Z_SPEED,
            ).wait_for_completed()

            # 不需要停视频；只等车体机械振动稳定一点
            time.sleep(0.25)
            continue

        distance_error = loc["d"] - args.grasp_target

        if abs(distance_error) > args.distance_tol:
            dx_m = distance_error / 1000.0
            dx_m = clamp(
                dx_m,
                -args.max_move_step,
                args.max_move_step,
            )

            if 0 < abs(dx_m) < 0.02:
                dx_m = 0.02 if dx_m > 0 else -0.02

            direction = "前进" if dx_m > 0 else "后退"

            vision.set_status(
                f"{direction} {abs(dx_m)*1000:.0f} mm",
                0,
                args.stable_frames,
            )

            print(
                f"距离校正: d={loc['d']:.1f}mm，"
                f"目标={args.grasp_target:.0f}mm -> "
                f"{direction} {abs(dx_m)*1000:.0f}mm"
            )

            if args.dry_run:
                return loc

            chassis.move(
                x=float(dx_m),
                y=0,
                z=0,
                xy_speed=CHASSIS_XY_SPEED,
            ).wait_for_completed()

            time.sleep(0.25)
            continue

        if not (args.grasp_min <= loc["r"] <= args.grasp_max):
            print(
                f"ERROR: r={loc['r']:.1f}mm 不在安全抓取带 "
                f"[{args.grasp_min:.0f},{args.grasp_max:.0f}]mm。"
            )
            return None

        vision.set_status(
            "READY TO GRASP",
            args.stable_frames,
            args.stable_frames,
        )

        print(
            f"[OK] 已对准并靠近目标: "
            f"d={loc['d']:.1f}mm r={loc['r']:.1f}mm "
            f"yaw={loc['yaw']:.1f}deg"
        )
        return loc

    print("ERROR: 达到最大视觉修正次数。")
    return None


def configure_exp2_gripper(class_name):
    exp2.GRIP_POWER = 30
    exp2.GRIP_CLOSED_FAST = 2.0
    exp2.GRIP_STABLE_TIME = 2.5
    exp2.GRIP_STATUS_TIMEOUT = 8.0

    if class_name == "bottle":
        print("WARN: bottle 仍沿用网球夹爪判定参数。")


def led_idle(ep):
    try:
        exp2.robot_signal_idle(ep)
    except Exception:
        pass


def led_success(ep):
    try:
        exp2.robot_signal_success(ep)
    except Exception:
        pass


def led_error(ep):
    try:
        exp2.robot_signal_error(ep)
    except Exception:
        pass


def arm_delta(arm, dx, dy, label, vision=None):
    if vision is not None:
        vision.set_status(f"ARM {label}", 0, 0)

    print(f"  arm delta {label}: x={dx:+.1f} y={dy:+.1f}")
    arm.move(x=float(dx), y=float(dy)).wait_for_completed()
    time.sleep(0.4)


def grasp_target(ep, arm, gripper, loc, vision):
    cls = loc["class"]
    r = loc["r"]

    configure_exp2_gripper(cls)

    pre_x = r - TOTAL_REL_FORWARD

    print("\n========================================")
    print("START GRASP")
    print("========================================")
    print(
        f"class={cls} r={r:.1f}mm "
        f"arm.moveto(x={pre_x:.1f}, y={PRE_Y:.1f})"
    )

    vision.set_status("GRIPPER OPEN", 0, 0)
    gripper.open(power=OPEN_POWER)
    time.sleep(0.7)

    vision.set_status("ARM PRE-GRASP", 0, 0)
    arm.moveto(x=float(pre_x), y=float(PRE_Y)).wait_for_completed()
    time.sleep(0.4)

    arm_delta(arm, EXTRA_FORWARD, 0, "forward", vision)
    arm_delta(arm, COARSE_FORWARD, COARSE_DOWN, "coarse down", vision)
    arm_delta(arm, FINAL_FORWARD, FINAL_DOWN, "final approach", vision)

    vision.set_status("GRIPPER CLOSE", 0, 0)
    print("关闭夹爪并判断...")
    ok, detail = exp2.close_and_judge(gripper)
    print("grasp result:", ok, detail)

    if not ok:
        led_error(ep)
        vision.set_status("GRASP FAILED", 0, 0)
        try:
            exp2.safe_home(arm, gripper)
        except Exception:
            gripper.open(power=OPEN_POWER)
            arm.recenter().wait_for_completed()
        return False

    led_success(ep)
    vision.set_status("GRASP SUCCESS / LIFT", 0, 0)

    arm_delta(arm, 0, TEST_LIFT, "test lift", vision)
    arm_delta(arm, 0, MAIN_LIFT, "main lift", vision)

    print("抓取成功，物体已抬起。")
    return True


def put_back_after_demo(arm, gripper, vision):
    vision.set_status("PUT BACK", 0, 0)
    arm_delta(arm, 0, RETURN_DOWN, "return down", vision)
    gripper.open(power=OPEN_POWER)
    time.sleep(0.8)
    arm_delta(arm, 0, RETURN_LIFT, "lift away", vision)
    arm.recenter().wait_for_completed()


def cleanup_with_timeout(ep, camera_started, vision, gripper, subscribed):
    if vision is not None:
        vision.stop()

    def _cleanup():
        if subscribed and gripper is not None:
            try:
                gripper.unsub_status()
            except Exception:
                pass

        if camera_started:
            try:
                ep.camera.stop_video_stream()
            except Exception as e:
                print("stop_video_stream warning:", e)

        try:
            ep.close()
        except Exception as e:
            print("ep.close warning:", e)

    t = threading.Thread(target=_cleanup, daemon=True)
    t.start()
    t.join(5)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model", default=MODEL_DEFAULT)
    p.add_argument("--calib", default=CALIB_DEFAULT)

    p.add_argument(
        "--target",
        choices=["auto", "bottle", "tennis_ball"],
        default="auto",
    )

    p.add_argument(
        "--camera-res",
        choices=["360p", "540p", "720p"],
        default="360p",
    )

    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--conf", type=float, default=0.35)

    # 关键：视频和 YOLO 不再同频
    p.add_argument(
        "--display-fps",
        type=float,
        default=30.0,
        help="视频显示目标FPS，默认30",
    )
    p.add_argument(
        "--infer-fps",
        type=float,
        default=6.0,
        help="YOLO最多每秒更新次数，默认6；视频仍独立30FPS",
    )

    p.add_argument("--stable-frames", type=int, default=3)
    p.add_argument("--target-timeout", type=float, default=15.0)

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--nav-only", action="store_true")

    p.add_argument(
        "--yaw-sign",
        type=float,
        choices=[-1.0, 1.0],
        default=-1.0,
    )

    p.add_argument("--yaw-tol", type=float, default=YAW_TOL_DEG)
    p.add_argument("--max-turn-step", type=float, default=MAX_TURN_STEP_DEG)
    p.add_argument("--max-move-step", type=float, default=MAX_MOVE_STEP_M)
    p.add_argument("--max-servo-steps", type=int, default=MAX_SERVO_STEPS)

    p.add_argument("--grasp-min", type=float, default=GRASP_R_MIN)
    p.add_argument("--grasp-max", type=float, default=GRASP_R_MAX)
    p.add_argument("--grasp-target", type=float, default=GRASP_R_TARGET)

    p.add_argument(
        "--distance-tol",
        type=float,
        default=15.0,
    )

    p.add_argument("--keep-object", action="store_true")

    return p.parse_args()


def run_continuous_dry_run(vision, args):
    """
    定位测试持续运行，不再识别一次就退出。
    q / ESC / Ctrl+C 才结束。
    """
    print("\n持续定位测试已启动，按 q / ESC 退出。")

    while not vision.is_stopped():
        loc = locate_stable(
            vision=vision,
            stable_frames=args.stable_frames,
            timeout=2.0,
            status=f"DRY RUN: {args.target}",
        )

        if loc is not None:
            print(
                f"[LOC] {loc['class']} conf={loc['conf']:.2f} "
                f"d={loc['d']:.1f} l={loc['l']:.1f} "
                f"r={loc['r']:.1f} yaw={loc['yaw']:.1f}"
            )

        time.sleep(0.05)


def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        print("ERROR: 模型不存在:", args.model)
        return 2

    if not os.path.isfile(args.calib):
        print("ERROR: calib.json 不存在:", args.calib)
        return 2

    H, calib, calib_res = load_calib(args.calib)

    use_cuda = torch.cuda.is_available()

    print("==================================================")
    print("EXP3 ASYNC VIDEO PRIORITY")
    print("==================================================")
    print("camera:", args.camera_res)
    print("display target:", args.display_fps, "FPS")
    print("YOLO imgsz:", args.imgsz)
    print("YOLO max update:", args.infer_fps, "FPS")
    print("CUDA:", use_cuda)
    if use_cuda:
        print("GPU:", torch.cuda.get_device_name(0))
    print("yaw_sign:", args.yaw_sign)
    print(
        "approach:",
        f"{args.grasp_target:.0f}±{args.distance_tol:.0f}mm"
    )
    print("fit:", calib.get("fit", {}))
    print("==================================================")

    if not use_cuda:
        # CPU推理时限制线程，给视频解码/显示留资源
        try:
            torch.set_num_threads(2)
        except Exception:
            pass

    print("加载 YOLO...")
    model = YOLO(args.model)

    # 预热模型，避免第一帧识别时长时间卡住显示
    print("YOLO warmup...")
    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    model.predict(
        dummy,
        imgsz=args.imgsz,
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
    gripper = None
    vision = None

    try:
        print("连接 RoboMaster...")
        ep.initialize(conn_type="ap")
        initialized = True
        print("RoboMaster SDK OK")

        arm = ep.robotic_arm
        gripper = ep.gripper
        chassis = ep.chassis
        camera = ep.camera

        gripper.sub_status(
            freq=5,
            callback=exp2.on_gripper_status,
        )
        subscribed = True

        led_idle(ep)

        print("机械臂 recenter...")
        arm.recenter().wait_for_completed()
        time.sleep(0.5)

        gripper.open(power=OPEN_POWER)
        time.sleep(0.5)

        print("启动视频流:", args.camera_res)
        camera.start_video_stream(
            display=False,
            resolution=args.camera_res,
        )
        camera_started = True

        vision = AsyncVision(
            camera=camera,
            model=model,
            H=H,
            calib_res=calib_res,
            target=args.target,
            conf=args.conf,
            imgsz=args.imgsz,
            infer_fps=args.infer_fps,
            display_fps=args.display_fps,
            window_name="EXP3 Async Live",
        )
        vision.start()

        # 等第一帧
        start_wait = time.monotonic()
        while vision.frame_seq == 0:
            if time.monotonic() - start_wait > 5:
                raise RuntimeError("5秒内没有收到相机画面")
            time.sleep(0.02)

        if args.dry_run:
            run_continuous_dry_run(vision, args)
            return 0

        loc = visual_servo_to_grasp_zone(
            arm=arm,
            chassis=chassis,
            vision=vision,
            args=args,
        )

        if loc is None:
            led_error(ep)
            vision.set_status("TASK FAILED", 0, 0)
            print("任务失败：无法把目标带入抓取区。")
            return 4

        if args.nav_only:
            led_success(ep)
            vision.set_status("NAV COMPLETE - q to exit", 0, 0)
            print(
                f"NAV 完成: {loc['class']} "
                f"d={loc['d']:.1f} l={loc['l']:.1f} "
                f"r={loc['r']:.1f} yaw={loc['yaw']:.1f}"
            )
            # 保持视频继续显示几秒，方便观察最终状态
            end = time.monotonic() + 3.0
            while time.monotonic() < end and not vision.is_stopped():
                time.sleep(0.05)
            return 0

        ok = grasp_target(
            ep=ep,
            arm=arm,
            gripper=gripper,
            loc=loc,
            vision=vision,
        )

        if not ok:
            return 5

        if args.keep_object:
            vision.set_status("KEEP OBJECT - q to exit", 0, 0)
            end = time.monotonic() + 5.0
            while time.monotonic() < end and not vision.is_stopped():
                time.sleep(0.05)
        else:
            time.sleep(1.0)
            put_back_after_demo(arm, gripper, vision)
            led_idle(ep)

        vision.set_status("DONE", 0, 0)
        print("任务完成。")
        time.sleep(1.0)
        return 0

    except KeyboardInterrupt:
        print("\n用户中止。")
        try:
            if initialized and gripper is not None:
                exp2.safe_home(ep.robotic_arm, gripper)
        except Exception:
            pass
        return 130

    except Exception as e:
        print("\nERROR:", repr(e))
        try:
            if initialized:
                led_error(ep)
        except Exception:
            pass
        raise

    finally:
        if initialized:
            cleanup_with_timeout(
                ep=ep,
                camera_started=camera_started,
                vision=vision,
                gripper=gripper,
                subscribed=subscribed,
            )


if __name__ == "__main__":
    raise SystemExit(main())
