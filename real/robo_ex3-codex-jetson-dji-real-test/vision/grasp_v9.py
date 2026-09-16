#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RoboMaster EP 瀹為獙涓夛細
鍩轰簬瀛︿範瑙嗚浣嶇疆鐨勫鐩爣浼樺厛鎶撳彇 + ALIGN_PREGRASP 绋冲畾鐗?
鐩稿涓婁竴鐗堜富瑕佷慨鏀癸細
1. 缃戠悆浣跨敤 YOLO imgsz=320銆乧onf=0.35
2. 姘寸摱浣跨敤 YOLO imgsz=640銆乧onf=0.25
3. auto 妯″紡浣跨敤 imgsz=640銆乧onf=0.25
4. 绋冲畾璇嗗埆鐢扁€滆繛缁?甯р€濇敼涓衡€滄渶杩?娆＄粨鏋滀腑鍚岀被鑷冲皯鍑虹幇3娆♀€?5. 杩滆窛绂婚樁娈垫満姊拌噦淇濇寔 recenter 瑙傚療濮挎€侊紝涓嶄細涓€璇嗗埆鍒扮洰鏍囧氨浣庡ご
6. 鐩爣杩涘叆杩戣窛绂诲垏鎹㈠尯鍚庯紝鎵嶇Щ鍔ㄥ埌瀵瑰簲绫诲埆鐨勫涔犳姄鍙栧Э鎬?7. 缃戠悆鐢?bbox width 鍒ゆ柇杩滆繎
8. 姘寸摱鐢?bbox height + bottom_y 鑱斿悎鍒ゆ柇杩滆繎锛屽苟涓ユ牸澶嶇幇鎴愬姛鎶撳彇瑙嗚鐘舵€?9. 缁х画淇濈暀 bottom_x 鎺у埗宸﹀彸杞悜
10. 鎶撳彇濮挎€佷笅鑻ョ洰鏍囦涪澶憋紝浼氳嚜鍔ㄦ姮鍥炶瀵熷Э鎬佺户缁悳绱?闈犺繎

娉ㄦ剰锛?- 鏈▼搴忎細绉诲姩搴曠洏鍜屾満姊拌噦銆?- 绗竴娆″厛浣跨敤 nav-only 閫夐」娴嬭瘯銆?- 姘寸摱澶圭埅鍒ゅ畾鍙傛暟浠嶆殏鏃舵部鐢ㄧ綉鐞冨弬鏁般€?"""

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


MODEL_DEFAULT = os.path.join(ROOT, "bottle_tennisball_best.pt")

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

# 缃戠悆 bbox 灏哄姣斾緥瀹瑰樊
SIZE_TOL_RATIO = 0.08

# 姘寸摱鎴愬姛鎶撳彇瑙嗚鐘舵€侊紙鏉ヨ嚜瀹炴満5娆℃垚鍔熸牱鏈級
BOTTLE_MIN_HEIGHT = 272.0
BOTTLE_MAX_HEIGHT = 290.0
BOTTLE_MIN_BOTTOM_Y = 350.0
BOTTLE_X_TOL_PX = 12.0

# 姘寸摱鍒嗘闈犺繎闃堝€?BOTTLE_FAR_H = 240.0
BOTTLE_MID_H = 260.0
BOTTLE_NEAR_H = 270.0

# 缃戠悆棰濆妫€鏌?bottom_y
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
# 涓ら樁娈佃瑙夌姸鎬佹満
# 闃舵1锛歄BSERVE/APPROACH锛屾満姊拌噦淇濇寔 recenter锛屾憚鍍忓ご鐪嬭繙澶?# 闃舵2锛歅REGRASP锛岀洰鏍囪冻澶熻繎鍚庢墠鍒囨崲鍒板涔犳姄鍙栧Э鎬?# ==========================================================

OBSERVE_CENTER_X = 320.0
OBSERVE_X_TOL_PX = 28.0

# 杩涘叆鎶撳彇濮挎€佺殑瑙嗚澶у皬闃堝€笺€?# 杩欐槸鈥滃垏鎹㈤槇鍊尖€濓紝涓嶆槸鏈€缁堟姄鍙栭槇鍊笺€?# 鏈€缁堟姄鍙栦粛浣跨敤瀛︿範 profile 鐨勪弗鏍兼潯浠躲€?BALL_PREGRASP_WIDTH = 60.0
BOTTLE_PREGRASP_HEIGHT = 170.0

OBSERVE_FAR_MOVE_M = 0.08
OBSERVE_MID_MOVE_M = 0.06
OBSERVE_NEAR_MOVE_M = 0.04

OBSERVE_MAX_STEPS = 80
PREGRASP_RECOVERY_LIMIT = 2

# ==========================================================
# 浣庡ご鍚庣殑瀹炴椂 CLOSE_APPROACH
# ==========================================================

# 浣庡ご鍚庝笉鍐嶄緷璧?seq / after_seq 闂ㄦ銆?# 鍙褰撳墠鍚岀被妫€娴嬫瓒冲鏂伴矞锛屽氨绔嬪嵆鍙備笌鎺у埗銆?MAX_DET_AGE_S = 1.0

# 杩戣窛绂诲惊鐜懆鏈?CLOSE_LOOP_PERIOD_S = 0.10

# 杩炵画澶氫箙鐪嬩笉鍒版湁鏁堢洰鏍囨墠鍒ゅ畾鏈疆杩戣窛璺熻釜澶辫触
CLOSE_LOST_TIMEOUT_S = 2.5

# 杩戣窛绂婚樁娈垫渶澶ф帶鍒舵鏁?CLOSE_MAX_STEPS = 30

# 鏈烘鑷傚姩浣滆秴鏃讹細
# RoboMaster SDK 鐨?wait_for_completed() 榛樿 timeout=None锛?# 濡傛灉鏈烘鑷傚凡缁忕墿鐞嗗埌浣嶄絾瀹屾垚ACK娌℃湁鍥炴潵锛岀▼搴忎細姘镐箙鍗″湪 PREGRASP / MOVE ARM銆?ARM_ACTION_TIMEOUT_S = 3.0
ARM_SETTLE_AFTER_TIMEOUT_S = 0.35

# ==========================================================
# 杩滆窛绂昏繛缁€熷害鎺у埗
# ==========================================================

# 杩滆窛绂婚樁娈典笉鍐嶄娇鐢ㄤ竴娈垫 chassis.move()锛屾敼鐢?drive_speed()
# 杩欐牱搴曠洏涓嶄細鍙嶅鈥滃惎鍔?鍒硅溅-鍚姩鈥濄€?APPROACH_FAST_MPS = 0.16
APPROACH_MID_MPS = 0.12
APPROACH_SLOW_MPS = 0.07

# 杈硅蛋杈硅浆
TURN_FAST_DPS = 18.0
TURN_MID_DPS = 10.0
TURN_SLOW_DPS = 6.0

# 閫熷害骞虫粦锛氭瘡涓帶鍒跺懆鏈熸渶澶у彉鍖栭噺
MAX_DV_MPS = 0.04
MAX_DZ_DPS = 5.0

# 杩炵画鎺у埗鍛ㄦ湡
CONTROL_PERIOD_S = 0.12

# 棰勬姄鍙栭槇鍊奸檮杩戞彁鍓嶅噺閫?PREGRASP_SLOW_RATIO = 0.82

# 棰勬姄鍙栧墠蹇呴』鍏堟í鍚戝鍑嗭紝闃叉鐩爣杩樺湪鐢婚潰杈圭紭灏变綆澶淬€?PREGRASP_ALIGN_ENTER_RATIO = 0.85
PREGRASP_ALIGN_X_TOL_PX = 30.0
PREGRASP_ALIGN_REQUIRED_HITS = 2
PREGRASP_CREEP_MPS = 0.04
OBSERVE_LOST_TIMEOUT_S = 2.0


# ==========================================================
# 澶氱洰鏍囦紭鍏堟姄鍙?+ 鍒嗙被鏀剧疆
# ==========================================================

# 鍚屼竴鐢婚潰閲屼紭鍏堟姄鏈€杩戠墿浣擄細
# 鍥哄畾瑙傚療濮挎€佷笅锛岀墿浣撲笌妗岄潰鎺ヨЕ鐐?bottom_y 瓒婂ぇ锛岄€氬父瓒婇潬杩戞満鍣ㄤ汉銆?# bbox 闈㈢Н鍙敤浜?bottom_y 鎺ヨ繎鏃剁殑娆＄骇鎺掑簭銆?NEAREST_BOTTOM_Y_WEIGHT = 1.0

# 榛樿鏈€澶氬垎鎷?6 涓墿浣擄紙瀹為獙瑕佹眰鑷冲皯 6 涓洰鏍囷級
DEFAULT_MAX_ITEMS = 6

# 杩炵画澶氬皯杞湅涓嶅埌鐩爣鍚庣粨鏉熶换鍔?EMPTY_RETRY_LIMIT = 3

# 鍒嗙被鏂瑰悜锛?# 瀹炴満涓?z>0 涓哄乏杞紝z<0 涓哄彸杞€?TENNIS_DROP_TURN_DEG = 90.0
BOTTLE_DROP_TURN_DEG = -90.0

# 杞悜鍒嗙被鍖哄悗锛屽悜鍓嶈蛋涓€鐐癸紝鏀句笅鍚庡啀閫€鍥炴潵
DROP_FORWARD_M = 0.18
DROP_XY_SPEED = 0.18
DROP_Z_SPEED = 25

# 鎶撹捣鍚庡凡缁忔姮鍗囩害 85mm銆?# 鏀剧疆鍓嶄笅闄嶄竴鐐癸紝閬垮厤浠庡お楂樺鐩存帴鎵斾笅銆?DROP_LOWER_MM = 55.0
DROP_OPEN_POWER = 35
DROP_OPEN_TIME = 1.0

# ==========================================================
# RETURN_SEARCH + SCAN
# ==========================================================
# 姣忔姄瀹屽苟鍒嗙被涓€涓洰鏍囧悗锛屼笉鐩存帴鍦ㄥ綋鍓嶉潬鍓嶄綅缃户缁壘涓嬩竴浠躲€?# 鍏堝悗閫€鎭㈠鏇村瑙嗛噹锛屽啀鍋氬乏/涓?鍙虫壂鎻忋€?RETURN_BACK_M = 0.25
RETURN_BACK_SPEED = 0.18

# 鎵弿瑙掑害锛坈hassis.move锛?# 褰撳墠瀹炴満 move()锛氬乏杞负姝ｏ紝鍙宠浆涓鸿礋銆?SCAN_LEFT_DEG = 25.0
SCAN_RIGHT_DEG = -25.0
SCAN_Z_SPEED = 20

SCAN_SETTLE_S = 0.45
SCAN_DETECT_TIMEOUT_S = 1.8
SCAN_WINDOW = 5
SCAN_HITS = 2

# ==========================================================
# 杞悜绗﹀彿锛歊oboMaster SDK 涓や釜鎺ュ彛鐨?z 鏂瑰悜绾﹀畾鍦ㄥ綋鍓嶅疄鏈鸿〃鐜颁笉鍚?# ==========================================================
#
# 杩滆窛绂昏繛缁帶鍒朵娇鐢?chassis.drive_speed(z=...)
# 瀹炴満楠岃瘉锛?#   x_error < 0锛堢洰鏍囧湪鐢婚潰宸﹁竟锛?-> z < 0 鑳芥纭悜宸︿慨姝?# 鍥犳鐩存帴浣跨敤 x_error 绗﹀彿銆?TURN_SIGN_DRIVE = 1.0
#
# 杩戣窛绂荤鏁ｄ慨姝ｄ娇鐢?chassis.move(z=...)
# 瀹炴満楠岃瘉锛?#   x_error < 0锛堢洰鏍囧湪鐢婚潰宸﹁竟锛?-> chassis.move 闇€瑕?z > 0 鎵嶈兘鍚戝乏淇
# 鍥犳蹇呴』鍙嶅彿銆?TURN_SIGN_MOVE = -1.0

OPEN_POWER = 35
TEST_LIFT = 25.0
MAIN_LIFT = 60.0


def to_signed32(v):
    """
    褰撳墠 RoboMaster SDK/鏉跨鐜涓紝鏈烘鑷傝礋 y 鍙兘琚樉绀轰负 uint32銆?    渚嬪 4294967212 = 2^32 - 84锛屽簲瑙ｉ噴涓?-84銆?    """
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
            raise ValueError(f"profile 缂哄皯瀛楁: {key}")

    profile["arm_x_mm"] = to_signed32(profile["arm_x_mm"])
    profile["arm_y_mm"] = to_signed32(profile["arm_y_mm"])

    return profile


def infer_config(target):
    """
    鏍规嵁褰撳墠妫€娴嬬洰鏍囬€夋嫨鎺ㄧ悊閰嶇疆銆?
    tennis_ball:
        imgsz=320, conf=0.35

    bottle:
        imgsz=640, conf=0.25

    auto:
        涓轰簡鍏奸【閫忔槑鐡跺瓙锛岀洿鎺ョ敤 640 / 0.25
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

    # 澶氱洰鏍囦紭鍏堢骇锛?    # 1) bottom_y 瓒婂ぇ -> 鍦ㄥ浐瀹氳瀵熷Э鎬佷笅閫氬父瓒婅繎
    # 2) 濡傛灉 bottom_y 寰堟帴杩戯紝鍒?bbox 闈㈢Н鏇村ぇ鐨勪紭鍏?    # 3) 鏈€鍚庣敤缃俊搴︽墦鐮村钩灞€
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
    """FPS缁熻鍋氱嚎绋嬩繚鎶ゃ€?""
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

            # 褰撳墠 PyAV 瑙ｇ爜閾捐矾宸茬粡楠岃瘉鍙洿鎺ヤ綔涓?BGR 浣跨敤
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
    浠庡綋鍓嶇敾闈㈢殑鎵€鏈夋娴嬫涓€夋嫨鈥滃綋鍓嶈鎶撶殑鐩爣鈥濄€?
    棣栨锛?        浼樺厛 bottom_y 鏈€澶э紝涔熷氨鏄瑙変笂鏈€闈犺繎鏈哄櫒浜?鐢婚潰搴曢儴鐨勭墿浣撱€?
    宸茬粡寮€濮嬭窡韪細
        浼樺厛缁х画璺熻釜涓庝笂涓€甯т綅缃浉杩戠殑鍚屼竴鐗╀綋锛岄伩鍏嶅涓悓绫绘涔嬮棿璺虫潵璺冲幓銆?        濡傛灉鎵句笉鍒扮浉杩戠洰鏍囷紝鍐嶉€€鍥炴渶杩戠洰鏍囪鍒欍€?    """
    if not detections:
        return None

    # 宸茬粡閿佸畾涓€涓洰鏍囨椂锛屽厛鍋氱畝鍗曚綅缃繛缁窡韪?    if previous is not None:
        same_class = [
            d for d in detections
            if d["class"] == previous["class"]
        ]

        if same_class:
            ranked = []
            for d in same_class:
                dx = abs(d["bottom_x"] - previous["bottom_x"])
                dy = abs(d["bottom_y"] - previous["bottom_y"])

                # 浼樺厛鍍忕礌浣嶇疆杩炵画锛涘悓鏃惰交寰亸鍚?bottom_y 鏇村ぇ鐨勭洰鏍?                continuity_cost = dx + 0.7 * dy - 0.08 * d["bottom_y"]
                ranked.append((continuity_cost, d))

            ranked.sort(key=lambda x: x[0])

            # 鐩爣绉诲姩涓昏鏉ヨ嚜鏈哄櫒浜鸿嚜韬繍鍔紝閫氬父涓嶄細鐬棿璺冲お杩溿€?            best_cost, best_det = ranked[0]

            if (
                abs(best_det["bottom_x"] - previous["bottom_x"]) <= 140
                and abs(best_det["bottom_y"] - previous["bottom_y"]) <= 120
            ):
                return best_det

    # 棣栨閫夋嫨/璺熻釜涓㈠け锛氱洿鎺ラ€夋渶杩戠洰鏍?    return max(
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
    澶氱洰鏍囩ǔ瀹氶€夋嫨銆?
    涓庢棫鐗堜笉鍚岋細
    - 涓嶅啀鍙湅鈥滄渶楂樼疆淇″害妗嗏€?    - 姣忎竴甯у厛浠庢墍鏈夋娴嬫涓€夋嫨鏈€杩戝€欓€?    - 涓€鏃﹀紑濮嬭窡韪紝灏介噺淇濇寔鍚屼竴涓墿浣擄紝閬垮厤澶氫釜鍚岀被鐩爣闂磋烦妗?    - 鏈€杩?window_size 娆￠噷锛岃嚦灏?required_hits 娆¤窡韪垚鍔熷嵆杩斿洖涓綅鏁颁綅缃?    """
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
    鐩存帴璇诲彇褰撳墠鏈€鏂版娴嬬粨鏋溿€?
    涓嶅啀瑕佹眰 seq 蹇呴』閫掑锛屼篃涓嶅啀绛夊緟 after_seq銆?    鍙繚鐣欎袱涓潯浠讹細
    1. 绫诲埆蹇呴』鍖归厤锛?    2. 妫€娴嬬粨鏋滃勾榫?<= MAX_DET_AGE_S銆?
    杩欐牱鍙互閬垮厤鈥滅敾闈㈠凡鏈夌豢鑹叉锛屼絾鎺у埗绾跨▼鍥犱负绛夋柊 seq 鑰屼笉鍔ㄤ綔鈥濄€?    """
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

    # 浣庡ご浠ュ悗锛岀洰鏍囧湪鍒囨崲鍓嶅凡缁忓畬鎴愭í鍚戝鍑嗐€?    # 鍥犳鑻ュ悓绫绘湁澶氫釜妗嗭紝浼樺厛閫夋渶鎺ヨ繎瀛︿範涓績 ref_x 鐨勯偅涓紝
    # 閬垮厤閲嶆柊璺冲埌鏃佽竟鍙︿竴涓悓绫荤墿浣撱€?    if ref_x is not None:
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
    鏈烘鑷備綆澶村悗鐨勫疄鏃惰繎璺濈闂幆銆?
    鍏抽敭鍙樺寲锛?    - 涓嶇瓑寰呮柊鐨?seq銆?    - 涓嶄娇鐢?after_seq / last_seq銆?    - 姣忎釜寰幆鐩存帴璇诲彇褰撳墠鏈€鏂版娴嬫銆?    - 鍙妫€娴嬫骞撮緞 <= MAX_DET_AGE_S锛屽氨绔嬪嵆璁＄畻骞舵墽琛屽姩浣溿€?    - 濡傛灉鐭椂闂村唴鏆傛椂娌℃湁鏈夋晥妗嗭紝浠呯瓑寰咃紱瓒呰繃 CLOSE_LOST_TIMEOUT_S 鎵嶆仮澶嶃€?    """
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
                f"[CLOSE] 鏆傛棤鏂伴矞 {cls} 妫€娴嬶紝"
                f"绛夊緟涓?{lost_time:.1f}s"
            )

            # 娌℃湁鏂伴矞鐩爣鏃剁‘淇濆簳鐩樺仠姝?            try:
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
                    "[CLOSE] 鐩爣涓㈠け瓒呰繃闄愬埗锛岄€€鍑鸿繎璺濈闂幆銆?
                )
                return None

            time.sleep(CLOSE_LOOP_PERIOD_S)
            continue

        # 褰撳墠鏈夋柊椴滄
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
        # 妯悜淇
        # -----------------------------
        x_tol = (
            BOTTLE_X_TOL_PX
            if cls == "bottle"
            else args.x_tol
        )

        if abs(x_error) > x_tol:
            z = turn_step_from_x_error(x_error)
            direction = "鍙? if x_error > 0 else "宸?

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
        # 姘寸摱绮剧‘杩戣窛閫昏緫
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
                print("[READY] 姘寸摱宸茶揪鍒版垚鍔熸姄鍙栬瑙夌姸鎬?)
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

            direction = "鍓嶈繘" if dx > 0 else "鍚庨€€"

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
        # 缃戠悆绮剧‘杩戣窛閫昏緫
        # -----------------------------
        ball_ready = (
            abs(x_error) <= args.x_tol
            and abs(size_ratio - 1.0) <= SIZE_TOL_RATIO
            and abs(y_error) <= args.ball_y_tol
        )

        if ball_ready:
            print()
            print("==============================================")
            print("[READY] 缃戠悆宸茶揪鍒版垚鍔熸姄鍙栬瑙夌姸鎬?)
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

        direction = "鍓嶈繘" if dx > 0 else "鍚庨€€"

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

    print("CLOSE_APPROACH 杈惧埌鏈€澶ф帶鍒舵鏁帮紝鍋滄銆?)
    return None



def _wait_arm_action(action, label):
    """
    RoboMaster Action.wait_for_completed(timeout) 鏀寔瓒呮椂杩斿洖銆?    杩欓噷缁濅笉鍏佽鏈烘鑷?ACK 鎶婃暣涓瑙夌姸鎬佹満姘镐箙鍗′綇銆?    """
    try:
        ok = action.wait_for_completed(
            timeout=ARM_ACTION_TIMEOUT_S
        )
    except TypeError:
        # 鏋佺鍏煎锛氬鏋滄煇涓?SDK 鐗堟湰涓嶆帴鍙楀叧閿瓧鍙傛暟
        ok = action.wait_for_completed(
            ARM_ACTION_TIMEOUT_S
        )

    if ok:
        print(f"[ARM] {label} completed")
        return True

    print(
        f"[ARM] WARNING: {label} wait_for_completed "
        f"瓒呰繃 {ARM_ACTION_TIMEOUT_S:.1f}s銆?
    )
    print(
        "[ARM] 濡傛灉鏈烘鑷傚凡缁忕墿鐞嗗埌浣嶏紝缁х画杩涘叆瑙嗚闂幆锛?
        "涓嶅啀姘镐箙闃诲銆?
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

    # 缁欑浉鏈虹敾闈竴鐐圭ǔ瀹氭椂闂达紝浣嗕笉鍐嶆棤闄愮瓑ACK
    time.sleep(0.35)


def move_arm_to_observe_pose(arm):
    """
    杩滆窛绂昏瀵熷Э鎬併€?    recenter 鍚屾牱鍔犲叆瓒呮椂锛岄伩鍏嶆仮澶嶈瀵熷Э鎬佹椂鍗℃銆?    """
    print("鏈烘鑷?-> 杩滆窛绂昏瀵熷Э鎬?(recenter)")

    action = arm.recenter()

    _wait_arm_action(
        action,
        "recenter",
    )

    time.sleep(0.35)



def observe_size(loc):
    """
    瑙傚療濮挎€佷笅鐢ㄤ簬鍒ゆ柇杩滆繎鐨勫昂瀵革細
    缃戠悆鐢?bbox width锛屾按鐡剁敤 bbox height銆?    """
    if loc["class"] == "bottle":
        return float(loc["height"]), "height"
    return float(loc["width"]), "width"


def pregrasp_threshold(class_name):
    if class_name == "bottle":
        return BOTTLE_PREGRASP_HEIGHT
    return BALL_PREGRASP_WIDTH


def observe_move_step(class_name, size_value):
    """
    杩滆窛绂昏瀵熼樁娈电殑鍓嶈繘姝ラ暱銆?    杩欓噷涓嶈拷姹傛渶缁堢簿纭窛绂伙紝鍙礋璐ｆ妸鐩爣閫佽繘 PREGRASP 鑼冨洿銆?    """
    threshold = pregrasp_threshold(class_name)
    ratio = size_value / threshold

    if ratio < 0.55:
        return OBSERVE_FAR_MOVE_M
    if ratio < 0.78:
        return OBSERVE_MID_MOVE_M
    return OBSERVE_NEAR_MOVE_M



def clamp_step(current, target, max_step):
    """
    闄愬埗姣忎釜鍛ㄦ湡鐨勯€熷害鍙樺寲閲忥紝閬垮厤绐佺劧鍔犻€?鎬ュ仠銆?    """
    delta = target - current

    if delta > max_step:
        delta = max_step
    elif delta < -max_step:
        delta = -max_step

    return current + delta


def stop_chassis_smooth(chassis, current_x=0.0, current_z=0.0):
    """
    骞虫粦鍑忛€熷埌 0銆?    """
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
    鏍规嵁鐩爣鐩稿鈥滆繘鍏?PREGRASP 闃堝€尖€濈殑鎺ヨ繎绋嬪害璁剧疆鍓嶈繘閫熷害銆?    鐩爣瓒婃帴杩戦槇鍊硷紝閫熷害瓒婁綆銆?    """
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
    鏍规嵁妯悜鍍忕礌璇樊鐢熸垚杩炵画杞悜瑙掗€熷害銆?    """
    a = abs(x_error)

    if a <= OBSERVE_X_TOL_PX:
        return 0.0

    if a > 100:
        mag = TURN_FAST_DPS
    elif a > 55:
        mag = TURN_MID_DPS
    else:
        mag = TURN_SLOW_DPS

    # drive_speed() 鐨?z 绗﹀彿鎸夊疄鏈洪獙璇侊細
    # 宸﹁竟 x_error<0 -> z<0
    # 鍙宠竟 x_error>0 -> z>0
    return TURN_SIGN_DRIVE * math.copysign(mag, x_error)



def approach_in_observe_pose(
    arm,
    chassis,
    vision,
    profiles,
    args,
):
    """
    杩滆窛绂昏繛缁潬杩?+ ALIGN_PREGRASP銆?
    涓荤嚎绋嬫槸鍞竴鐨勫簳鐩?鏈烘鑷傛帶鍒惰€咃紱AsyncVision 涓変釜绾跨▼鍙礋璐?    鐩告満璇诲彇銆乊OLO鎺ㄧ悊鍜屾樉绀猴紝涓嶄細鍙戦€佷换浣曡繍鍔ㄥ懡浠わ紝鍥犳杩欓噷娌℃湁
    涓庤瑙夌嚎绋嬩簤鎶?chassis/arm 鐨勬帶鍒跺啿绐併€?
    鐘舵€侊細
        APPROACH
            -> 鐩爣鎺ヨ繎鍒囨崲闃堝€肩殑 85%
        ALIGN_PREGRASP
            -> 鍋滄楂橀€熷墠杩涳紝鍏堟妸鐩爣杞埌鐢婚潰涓ぎ
            -> 鑻ュ凡缁忓眳涓絾璺濈杩樺樊涓€鐐癸紝鍙互 0.04m/s 鎱㈡參鍚戝墠
            -> 鈥滆窛绂昏揪鍒伴槇鍊?+ |x_error|<=30px鈥?杩炵画2涓柊YOLO缁撴灉
        PREGRASP
            -> 骞虫粦鍋滆溅
            -> 鎵嶅厑璁告満姊拌噦浣庡ご

    杩欐牱缁濅笉浼氬啀鍑虹幇 err=-200~-300px 浣嗗洜涓?bbox 澶熷ぇ灏辩洿鎺ヤ綆澶淬€?    """
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

            # 鍙湪鏈夋柊鐨刌OLO缁撴灉鏃舵洿鏂扮洰鏍囩姸鎬侊紱杩愬姩閫熷害鍛戒护鏈韩鍙寔缁€?            if seq == last_det_seq:
                if time.monotonic() - last_seen_time > OBSERVE_LOST_TIMEOUT_S:
                    print("[APPROACH] 杩炵画瓒呰繃2绉掓病鏈夋柊鐨勬湁鏁堢洰鏍囷紝鍋滄骞舵仮澶嶃€?)
                    stop_chassis_smooth(chassis, current_x, current_z)
                    return None, None, None
                time.sleep(0.02)
                continue

            last_det_seq = seq

            # 妫€娴嬬粨鏋滆繃鏃э紝涓嶅弬涓庢帶鍒躲€?            if det_time <= 0 or (time.monotonic() - det_time) > MAX_DET_AGE_S:
                continue

            # 鍒濇浠庡叏閮ㄧ洰鏍囬€夋渶杩戯紱閿佸畾鍚庢寔缁窡韪悓涓€绫诲埆銆佺浉閭讳綅缃殑妗嗐€?            if locked_class is None:
                candidate = _priority_pick(detections, previous=None)
            else:
                same = [d for d in detections if d["class"] == locked_class]
                candidate = _priority_pick(same, previous=previous_det)

            if candidate is None:
                if time.monotonic() - last_seen_time > OBSERVE_LOST_TIMEOUT_S:
                    print("[APPROACH] 閿佸畾鐩爣涓㈠け瓒呰繃2绉掞紝鍋滄骞舵仮澶嶃€?)
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
                    f"[SMOOTH] 閿佸畾鏈€杩戠洰鏍? {locked_class}锛?
                    "鍚庣画鎸佺画璺熻釜璇ョ墿浣?
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
            # 杩涘叆 ALIGN_PREGRASP锛氭帴杩戦槇鍊煎悗鍏堝鍑嗭紝涓嶅噯鐩存帴浣庡ご
            # --------------------------------------------------
            if (
                not align_mode
                and size_ratio_to_switch >= PREGRASP_ALIGN_ENTER_RATIO
            ):
                align_mode = True
                align_hits = 0

                print(
                    "[ALIGN_PREGRASP] 宸叉帴杩戦鎶撳彇鍖哄煙锛?
                    "鍏堝仠姝㈠墠鍐插苟妯悜瀵瑰噯銆?
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

                # 鑻ョ洰鏍囧洜涓鸿浆鍚?妫€娴嬫尝鍔ㄦ槑鏄惧張鍙樿繙锛岄€€鍑篈LIGN閲嶆柊姝ｅ父闈犺繎銆?                if size_ratio_to_switch < 0.72:
                    print("[ALIGN_PREGRASP] 鐩爣鍙堟槑鏄惧彉杩滐紝杩斿洖 APPROACH銆?)
                    align_mode = False
                    align_hits = 0
                    continue

                # 鍏堟í鍚戝鍑嗐€傚亸宸ぇ鏃剁粷涓嶅悜鍓嶃€?                if abs(x_error) > PREGRASP_ALIGN_X_TOL_PX:
                    align_hits = 0
                    target_x = 0.0
                    target_z = desired_turn_speed(x_error)
                    current_x = 0.0
                    current_z = clamp_step(
                        current_z,
                        target_z,
                        MAX_DZ_DPS,
                    )

                    direction = "鍙? if x_error > 0 else "宸?
                    print(
                        f"  ALIGN: 鐩爣鍋弡direction} {x_error:+.1f}px锛?
                        f"鍘熷湴杞悜 z={current_z:+.1f}deg/s锛屼笉鍓嶈繘"
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

                # 妯悜宸茬粡鍩烘湰瀵瑰噯銆?                current_z = clamp_step(
                    current_z,
                    0.0,
                    MAX_DZ_DPS,
                )

                if size_value < switch_threshold:
                    # 杩樺樊涓€鐐硅窛绂伙細鍙厑璁告瀬鎱㈢洿琛岋紝涓嶅啀楂橀€熷啿涓婂幓銆?                    align_hits = 0
                    current_x = clamp_step(
                        current_x,
                        PREGRASP_CREEP_MPS,
                        MAX_DV_MPS,
                    )
                    print(
                        f"  ALIGN: 宸插眳涓絾璺濈杩樺樊涓€鐐癸紝"
                        f"浠呮參閫熷墠杩?x={current_x:.2f}m/s"
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

                # 鍚屾椂婊¤冻锛氳窛绂诲杩?+ 妯悜灞呬腑銆?                align_hits += 1
                current_x = 0.0

                print(
                    f"  ALIGN READY {align_hits}/{PREGRASP_ALIGN_REQUIRED_HITS}: "
                    f"|x_error|={abs(x_error):.1f}px <= {PREGRASP_ALIGN_X_TOL_PX:.1f}, "
                    f"{size_name}={size_value:.1f} >= {switch_threshold:.1f}"
                )

                # 瑕佹眰杩炵画涓や釜鈥滄柊鐨刌OLO缁撴灉鈥濋兘婊¤冻锛岄槻姝㈠崟甯box鎶栧姩璇Е鍙戙€?                if align_hits < PREGRASP_ALIGN_REQUIRED_HITS:
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
                print("[PREGRASP] 璺濈鍜屾í鍚戜綅缃潎宸茬‘璁?)
                print(
                    f"class={locked_class}, x_error={x_error:+.1f}px, "
                    f"{size_name}={size_value:.1f}"
                )
                print("鐜板湪鎵嶅厑璁稿仠杞﹀苟鍒囨崲瀛︿範鎶撳彇濮挎€併€?)
                print("==============================================")

                stop_chassis_smooth(
                    chassis,
                    current_x=current_x,
                    current_z=current_z,
                )
                return locked_class, profile, loc

            # --------------------------------------------------
            # 姝ｅ父杩滆窛绂?APPROACH
            # --------------------------------------------------
            vision.set_status(
                f"SMOOTH APPROACH {step}/{args.observe_max_steps}"
            )

            target_x = desired_approach_speed(
                locked_class,
                size_value,
            )

            abs_x_err = abs(x_error)

            # 澶ц搴﹀亸绂绘椂鍏堣浆锛屼笉鍏佽涓€杈逛弗閲嶅亸鑸竴杈圭户缁墠鍐层€?            if abs_x_err > 120:
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

        print("杩炵画瑙傚療/闈犺繎闃舵杈惧埌鏈€澶ф帶鍒舵鏁般€?)
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
    鐘舵€佹満锛?
        OBSERVE / SMOOTH_APPROACH
            鈫?        PREGRASP
            鈫?        鏈烘鑷備笅闄?            鈫?        鐩存帴杩涘叆 CLOSE_APPROACH
            鈫?        瀹炴椂璇诲彇 latest detection
            鈫?        TURN / FORWARD / BACKWARD / READY

    涓嶅啀浣跨敤 seq 闂ㄦ銆?    """
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
            "杩涘叆瀛︿範鎶撳彇濮挎€?,
        )

        print()
        print(
            "[PREGRASP] arm.moveto 闃诲宸查噴鏀俱€?
        )
        print(
            "[PREGRASP] 鏈烘鑷傚凡涓嬮檷/鍒颁綅鎴栧凡瓒呮椂鏀捐锛?
            "鐩存帴杩涘叆瀹炴椂 CLOSE_APPROACH銆?
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
                "[RECOVERY] CLOSE_APPROACH 鏈畬鎴愶紝"
                "鎶洖瑙傚療濮挎€侀噸鏂版悳绱€?
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

    # 娉ㄦ剰锛氳繖閲岀粰 chassis.move(z=...) 浣跨敤銆?    # 褰撳墠瀹炴満涓?move() 涓?drive_speed() 鐨勮浆鍚戠鍙风浉鍙嶏細
    # 宸﹁竟 x_error<0 -> z>0
    # 鍙宠竟 x_error>0 -> z<0
    return (
        TURN_SIGN_MOVE
        * math.copysign(
            mag,
            x_error,
        )
    )


def move_step_from_size_ratio(ratio):
    """
    ratio = 褰撳墠瑙嗚灏哄 / 瀛︿範瑙嗚灏哄

    ratio < 1 -> 鐩爣姣斿涔犵姸鎬佸皬 -> 鐩爣鏇磋繙 -> 鍓嶈繘
    ratio > 1 -> 鐩爣姣斿涔犵姸鎬佸ぇ -> 鐩爣鏇磋繎 -> 鍚庨€€
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
            print("娌℃湁绋冲畾璇嗗埆鍒扮洰鏍囷紝鍋滄銆?)
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

        # 鍏堟í鍚戝鍑?        if abs(x_error) > x_tol:
            z = turn_step_from_x_error(x_error)
            direction = "鍙? if x_error > 0 else "宸?
            print(f"  鐩爣鍦ㄧ敾闈direction}渚?-> 杞悜 {z:+.1f}deg")

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

        # 姘寸摱锛氫弗鏍煎鐜版垚鍔熸姄鍙栬瑙夌姸鎬?        if cls == "bottle":
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
                direction = "鍓嶈繘" if dx > 0 else "鍚庨€€"
                print(
                    f"  姘寸摱绮剧‘璺濈鏍℃: h={h:.1f}px, bottom_y={bottom_y:.1f}px "
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
                print("  姘寸摱灏氭湭婊¤冻鏈€缁堟姄鍙栨潯浠讹紝缁х画瑙嗚闂幆銆?)
                time.sleep(0.15)
                continue

            print()
            print("==============================================")
            print("[OK] 姘寸摱宸蹭弗鏍煎鐜版垚鍔熸姄鍙栬瑙夌姸鎬?)
            print(
                f"x_error={x_error:+.1f}px, "
                f"height={h:.1f}px, "
                f"bottom_y={bottom_y:.1f}px"
            )
            print("==============================================")
            vision.set_status("BOTTLE READY TO GRASP")
            return loc

        # 缃戠悆淇濇寔涓婁竴鐗堥€昏緫
        dx = move_step_from_size_ratio(size_ratio)

        if dx != 0.0:
            direction = "鍓嶈繘" if dx > 0 else "鍚庨€€"
            print(
                f"  璺濈瑙嗚鏍℃: {size_name} ratio={size_ratio:.3f} "
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
            direction = "鍓嶈繘" if dx > 0 else "鍚庨€€"
            print(f"  缃戠悆 Y 璇樊浠嶄负 {y_error:+.1f}px -> {direction} {abs(dx)*1000:.0f}mm")

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
        print("[OK] 缃戠悆宸插埌瀛︿範寰楀埌鐨勬垚鍔熸姄鍙栬瑙変綅缃?)
        print(
            f"x_error={x_error:+.1f}px, "
            f"width_ratio={size_ratio:.3f}, "
            f"y_error={y_error:+.1f}px"
        )
        print("==============================================")
        vision.set_status("READY TO GRASP")
        return loc

    print("杈惧埌鏈€澶ц瑙夐棴鐜鏁帮紝鍋滄銆?)
    return None

def configure_gripper(class_name):
    exp2.GRIP_POWER = 30
    exp2.GRIP_CLOSED_FAST = 2.0
    exp2.GRIP_STABLE_TIME = 2.5
    exp2.GRIP_STATUS_TIMEOUT = 8.0

    if class_name == "bottle":
        print(
            "WARN: bottle 鏆傛椂娌跨敤缃戠悆澶圭埅鍒ゅ畾鍙傛暟銆?
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
        "鍏抽棴澶圭埅骞跺垽鏂?.."
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
        "鎶撳彇鎴愬姛锛屽紑濮嬫姮璧枫€?
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
        "鐗╀綋宸叉姮璧枫€?
    )

    return True




def _scan_detect_current_view(
    vision,
    timeout=SCAN_DETECT_TIMEOUT_S,
):
    """
    褰撳墠鏈濆悜涓嬪揩閫熷垽鏂槸鍚︽湁鐩爣銆?    鎵弿闃舵鍙姹傝緝瀹芥澗鐨?2/5 鍛戒腑锛?    鐩殑鏄喅瀹氣€滀笅涓€杞悳绱㈠簲璇ユ湞鍝釜鏂瑰悜鈥濓紝涓嶆槸鐩存帴鎶撱€?    """
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
    姣忔鍒嗙被鏀剧疆鍚庢墽琛岋細

        1. 鏈烘鑷傚洖楂樹綅瑙傚療濮挎€?        2. 鍚戝悗閫€鍥哄畾璺濈锛屾仮澶嶈緝瀹借閲?        3. 涓棿瑙嗚妫€娴?        4. 宸﹁浆绾?5掳妫€娴?        5. 鍙虫壂绾?0掳鍒板彸渚ф娴?        6. 鏍规嵁涓変釜瑙嗚涓€滆瑙変笂鏈€杩戔€濈殑鐩爣锛?           鏈€鍚庢妸杞︽湞鍚戦偅涓瑙?        7. 涓嬩竴杞粠杩欎釜鏂瑰悜閲嶆柊鍋氭渶杩戠洰鏍囬€夋嫨

    娉ㄦ剰锛?    杩欓噷涓嶆槸涓ユ牸鍥炴斁鍘熻矾寰勶紝鑰屾槸鈥滄仮澶嶆悳绱㈣閲?+ 涓诲姩閲嶈娴嬧€濄€?    杩欐瘮鎶撳畬鍚庡仠鍦ㄩ潬鍓嶄綅缃洿鎺ョ户缁瘑鍒洿绋炽€?    """
    print()
    print("============================================================")
    print("[RETURN_SEARCH] 鎭㈠鎼滅储浣嶇疆骞舵壂鎻忓墿浣欑洰鏍?)
    print("============================================================")

    vision.set_status("RETURN_SEARCH / OBSERVE POSE")

    move_arm_to_observe_pose(arm)

    vision.set_target("auto")
    vision.set_profile(None)

    # 1) 鍚庨€€锛屾墿澶ц閲?    vision.set_status("RETURN_SEARCH / BACK UP")

    print(
        f"[RETURN_SEARCH] 鍚庨€€ {RETURN_BACK_M*100:.0f}cm锛?
        "鎭㈠鏇村畬鏁寸殑鐩爣瑙嗛噹"
    )

    chassis.move(
        x=-RETURN_BACK_M,
        y=0,
        z=0,
        xy_speed=RETURN_BACK_SPEED,
    ).wait_for_completed()

    time.sleep(0.35)

    candidates = []

    # 2) 涓棿瑙嗚
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

    # 3) 宸︿晶瑙嗚
    vision.set_status("SCAN / LEFT")
    print(f"[SCAN] 宸﹁浆 {SCAN_LEFT_DEG:.0f}deg")

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

    # 4) 浠庡乏渚ф壂鍒板彸渚э細鎬诲叡鍙宠浆50搴?    vision.set_status("SCAN / RIGHT")
    sweep_to_right = SCAN_RIGHT_DEG - SCAN_LEFT_DEG

    print(
        f"[SCAN] 浠庡乏渚ф壂鍒板彸渚?{sweep_to_right:+.0f}deg"
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
        # 娌℃壘鍒扮洰鏍囷紝鍥炴绛夊緟涓嬩竴杞悳绱?        print("[SCAN] 宸?涓?鍙抽兘娌″彂鐜扮ǔ瀹氱洰鏍囷紝鍥炴銆?)

        chassis.move(
            x=0,
            y=0,
            z=-SCAN_RIGHT_DEG,
            z_speed=SCAN_Z_SPEED,
        ).wait_for_completed()

        time.sleep(0.30)

        vision.set_status("SCAN COMPLETE / NO TARGET")
        return None

    # 5) 涓変釜瑙嗚閲岄€夋嫨鈥滆瑙変笂鏈€杩戔€濈殑鐩爣
    # 涓讳紭鍏堬細bottom_y锛涙浼樺厛锛氱洰鏍囪瑙夐潰绉€?    best_view, best_angle, best_loc = max(
        candidates,
        key=lambda item: (
            item[2]["bottom_y"],
            item[2]["width"] * item[2]["height"],
            item[2]["conf"],
        ),
    )

    print()
    print(
        f"[SCAN] 涓嬩竴杞帹鑽愭柟鍚? {best_view}, "
        f"target={best_loc['class']}, "
        f"bottom_y={best_loc['bottom_y']:.1f}"
    )

    # 褰撳墠杞﹀ご鍦?RIGHT(-25掳)銆?    # 杞埌鎵€閫夎瑙掔殑缁濆鐩稿瑙掑害銆?    current_angle = SCAN_RIGHT_DEG
    correction = best_angle - current_angle

    if abs(correction) > 0.5:
        print(
            f"[SCAN] 璋冩暣鍒?{best_view} 瑙嗚: "
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
    tennis_ball -> 宸︿晶鍒嗙被鍖?    bottle      -> 鍙充晶鍒嗙被鍖?    """
    if class_name == "tennis_ball":
        return TENNIS_DROP_TURN_DEG, "LEFT / 缃戠悆鍖?

    return BOTTLE_DROP_TURN_DEG, "RIGHT / 姘寸摱鍖?


def place_to_sort_side(
    arm,
    chassis,
    gripper,
    class_name,
    vision,
):
    """
    鎶撳彇鎴愬姛鍚庣殑鍒嗙被鍔ㄤ綔锛?
        鎶撹捣
          鈫?        缃戠悆宸﹁浆90掳 / 姘寸摱鍙宠浆90掳
          鈫?        鍓嶈繘鍒板垎绫绘斁缃尯
          鈫?        鏈烘鑷備笅闄嶄竴鐐?          鈫?        鏉惧紑澶圭埅
          鈫?        鏈烘鑷傛姮璧?          鈫?        鍚庨€€鍥炲師浣嶇疆
          鈫?        杞洖鍘熸潵鐨勭洰鏍囧尯鍩熸柟鍚?
    杩欐牱姣忔鍒嗘嫞鍚庢満鍣ㄤ汉浠嶈兘缁х画闈㈠鍓╀綑鐩爣銆?    """
    turn_deg, zone_name = sort_direction(class_name)

    print()
    print("==============================================")
    print(f"[SORT] {class_name} -> {zone_name}")
    print("==============================================")

    vision.set_status(
        f"SORT {class_name} -> {zone_name}"
    )

    # 1. 杞悜瀵瑰簲鍒嗙被鍖?    chassis.move(
        x=0,
        y=0,
        z=float(turn_deg),
        z_speed=DROP_Z_SPEED,
    ).wait_for_completed()

    time.sleep(0.35)

    # 2. 鍚戝垎绫诲尯鍓嶈繘涓€鐐癸紝閬垮厤鐗╀綋钀藉湪鏈哄櫒浜烘鏃佽竟
    chassis.move(
        x=DROP_FORWARD_M,
        y=0,
        z=0,
        xy_speed=DROP_XY_SPEED,
    ).wait_for_completed()

    time.sleep(0.30)

    # 3. 鎶婄墿浣撶◢寰斁浣?    try:
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

    # 4. 鏉惧紑澶圭埅瀹屾垚鍒嗙被鏀剧疆
    print(f"[SORT] 鏀剧疆 {class_name}")
    gripper.open(power=DROP_OPEN_POWER)
    time.sleep(DROP_OPEN_TIME)

    # 5. 鏈烘鑷傞噸鏂版姮楂橈紝闃叉鍚庨€€鏃剁鍒板凡鏀剧疆鐗╀綋
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

    # 6. 閫€鍥炴斁缃墠鐨勪綅缃?    chassis.move(
        x=-DROP_FORWARD_M,
        y=0,
        z=0,
        xy_speed=DROP_XY_SPEED,
    ).wait_for_completed()

    time.sleep(0.30)

    # 7. 杞洖鍘熸潵闈㈠鐩爣鍖哄煙鐨勬柟鍚?    chassis.move(
        x=0,
        y=0,
        z=float(-turn_deg),
        z_speed=DROP_Z_SPEED,
    ).wait_for_completed()

    time.sleep(0.35)

    # 8. 澶圭埅鎵撳紑锛涚湡姝ｇ殑鈥滄仮澶嶈閲?+ 鎵弿鈥濇斁鍒?RETURN_SEARCH 闃舵
    gripper.open(power=OPEN_POWER)
    time.sleep(0.25)

    vision.set_target("auto")
    vision.set_profile(None)
    vision.set_status("SORT DONE / RETURN SEARCH")

    print("[SORT] 鍒嗙被鏀剧疆鍔ㄤ綔瀹屾垚锛屽噯澶囨仮澶嶆悳绱㈣閲庛€?)



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
        help="杩炵画鎶撳彇澶氫釜鐩爣骞舵寜绫诲埆宸﹀彸鍒嗘嫞",
    )

    p.add_argument(
        "--max-items",
        type=int,
        default=DEFAULT_MAX_ITEMS,
        help="鏈疆鏈€澶氬垎鎷ｅ灏戜釜鐗╀綋",
    )

    p.add_argument(
        "--empty-retries",
        type=int,
        default=EMPTY_RETRY_LIMIT,
        help="杩炵画澶氬皯杞瘑鍒笉鍒扮洰鏍囧悗缁撴潫鍒嗘嫞",
    )

    return p.parse_args()


def main():
    args = parse_args()

    if (
        args.stable_hits
        > args.stable_window
    ):
        print(
            "ERROR: stable-hits 涓嶈兘澶т簬 stable-window"
        )
        return 2

    if not os.path.isfile(
        args.model
    ):
        print(
            "ERROR: 妯″瀷涓嶅瓨鍦?",
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
                "WARN: profile 涓嶅瓨鍦?",
                path,
            )

    if (
        args.target != "auto"
        and args.target not in profiles
    ):
        print(
            "ERROR: 鎸囧畾绫诲埆娌℃湁瀵瑰簲 profile銆?
        )
        return 2

    use_cuda = torch.cuda.is_available()

    print()
    print(
        "鍔犺浇 YOLO..."
    )

    model = YOLO(
        args.model
    )

    # 棰勭儹 640锛屽吋瀹?bottle 妯″紡
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
            "杩炴帴 RoboMaster..."
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
            "鍚姩瑙嗛娴?.."
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

        # 绛夌涓€甯?        start_wait = time.monotonic()

        while vision.frame_seq == 0:
            if (
                time.monotonic()
                - start_wait
                > 5.0
            ):
                raise RuntimeError(
                    "5绉掑唴娌℃湁鏀跺埌瑙嗛"
                )

            time.sleep(0.02)

        # ----------------------------------------------------
        # 澶氱洰鏍囨ā寮忥細
        #
        # 鐩告満鐢婚潰閲屽彲浠ュ悓鏃跺嚭鐜板涓綉鐞?姘寸摱銆?        # 姣忚疆浼樺厛閫夋嫨 bottom_y 鏈€澶э紙瑙嗚涓婃渶杩戯級鐨勭洰鏍囥€?        #
        # 鍗曟鎶撳彇锛?        #   OBSERVE -> APPROACH -> PREGRASP -> PRECISE -> GRASP
        #
        # 鍒嗙被妯″紡锛?        #   鎶撳彇鎴愬姛
        #       鈫?        #   tennis_ball 宸︿晶鏀剧疆
        #   bottle      鍙充晶鏀剧疆
        #       鈫?        #   杞洖鐩爣鍖?        #       鈫?        #   鍐嶆浠庡墿浣欐涓€夋渶杩戠洰鏍?        # ----------------------------------------------------

        if not args.sort_mode:
            # 淇濈暀鍗曠洰鏍囨祴璇曟ā寮?            cls, profile = pregrasp_with_recovery(
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
            print("涓ら樁娈靛畾浣嶅畬鎴?)
            print("鐩爣绫诲埆:", cls)
            print("==============================================")

            if args.nav_only or args.dry_run:
                try:
                    exp2.robot_signal_success(ep)
                except Exception:
                    pass

                vision.set_status("NAV COMPLETE")
                print("浠呭鑸祴璇曞畬鎴愶紝涓嶆墽琛屾姄鍙栥€?)
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
                print("淇濇寔澶规寔 5 绉掋€?)
                vision.set_status("KEEP OBJECT")
                time.sleep(5.0)

            vision.set_status("DONE")
            print("浠诲姟瀹屾垚銆?)
            time.sleep(1.0)
            return 0

        # ====================================================
        # 杩炵画鑷姩鍒嗘嫞妯″紡
        # ====================================================
        sorted_count = 0
        empty_count = 0
        tennis_count = 0
        bottle_count = 0

        # 澶氱墿浣撳垎鎷ｅ缁堝厑璁稿悓鏃惰瘑鍒袱绫?        args.target = "auto"
        vision.set_target("auto")
        vision.set_profile(None)

        print()
        print("============================================================")
        print("寮€濮嬪鐩爣鑷姩鍒嗘嫞")
        print("浼樺厛绾э細bottom_y 鏈€澶х殑鐩爣浼樺厛锛堟渶杩戜紭鍏堬級")
        print("缃戠悆 -> 宸︿晶锛涙按鐡?-> 鍙充晶")
        print(f"鏈€澶氬鐞?{args.max_items} 涓洰鏍?)
        print("============================================================")

        while sorted_count < args.max_items:
            vision.set_target("auto")
            vision.set_profile(None)
            vision.set_status(
                f"SEARCH NEXT {sorted_count + 1}/{args.max_items}"
            )

            print()
            print(
                f"========== 绗?{sorted_count + 1} 涓洰鏍?=========="
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
                    f"[SEARCH] 鏈疆娌℃湁鎵惧埌鍙姄鐩爣 "
                    f"({empty_count}/{args.empty_retries})"
                )

                # 鍥為珮浣嶅啀鐪嬩竴娆★紝閬垮厤鍥犱负涓婁竴杞Э鎬佹畫鐣欏鑷磋鍒や负绌?                move_arm_to_observe_pose(arm)
                vision.set_target("auto")
                vision.set_profile(None)

                if empty_count >= args.empty_retries:
                    print()
                    print("杩炵画澶氳疆娌℃湁鐩爣锛岃涓哄垎鎷ｅ尯鍩熷凡缁忓鐞嗗畬鎴愩€?)
                    break

                time.sleep(0.6)
                continue

            empty_count = 0

            print(
                f"[PRIORITY] 褰撳墠鏈€杩戠洰鏍? {cls}"
            )

            if args.nav_only or args.dry_run:
                print(
                    "sort-mode + nav-only锛氬凡瀹屾垚鏈€杩戠洰鏍囬€夋嫨鍜岄潬杩戯紝"
                    "涓嶆墽琛屾姄鍙?鍒嗙被銆?
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
                    "[GRASP] 鏈鎶撳彇澶辫触锛屾姮鍥炶瀵熷Э鎬佸悗缁х画瀵绘壘銆?
                )

                move_arm_to_observe_pose(arm)
                gripper.open(power=OPEN_POWER)
                vision.set_target("auto")
                vision.set_profile(None)
                time.sleep(0.5)
                continue

            # 鎶撳彇鎴愬姛锛屾墽琛屽乏鍙冲垎绫?            place_to_sort_side(
                arm=arm,
                chassis=chassis,
                gripper=gripper,
                class_name=cls,
                vision=vision,
            )

            # 鍏抽敭鏂板锛?            # 鏀惧畬浠ュ悗涓嶈鐩存帴鍦ㄥ綋鍓嶉潬鍓嶄綅缃壘涓嬩竴涓€?            # 鍏堝悗閫€鎭㈠瑙嗛噹锛屽啀鍋氬乏/涓?鍙虫壂鎻忥紝
            # 鏈€缁堣杞︽湞鍚戜笅涓€鎵圭洰鏍囨渶鏈夊笇鏈涚殑鏂瑰悜銆?            return_search_and_scan(
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
                f"[PROGRESS] 宸插垎鎷?{sorted_count}/{args.max_items} | "
                f"tennis_ball={tennis_count} | bottle={bottle_count}"
            )

            # RETURN_SEARCH 宸插畬鎴愬悗閫€涓庡乏/涓?鍙抽噸瑙傛祴銆?            # 绛夌敾闈㈢ǔ瀹氾紝鍐嶉噸鏂伴€夋嫨褰撳墠鏈€杩戠洰鏍囥€?            time.sleep(0.6)

        try:
            exp2.robot_signal_success(ep)
        except Exception:
            pass

        vision.set_status("SORT COMPLETE")

        print()
        print("============================================================")
        print("鑷姩鍒嗘嫞缁撴潫")
        print(f"鎬绘暟        : {sorted_count}")
        print(f"tennis_ball : {tennis_count}")
        print(f"bottle      : {bottle_count}")
        print("============================================================")

        time.sleep(2.0)
        return 0

    except KeyboardInterrupt:
        print(
            "\n鐢ㄦ埛涓銆?
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

