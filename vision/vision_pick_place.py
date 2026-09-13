#!/usr/bin/env python3
"""视觉检测联动定点抓取: 检测 → 定位修正 → 分类抓取 → 转向 B 点放置。

原理: 物体摆在固定 A 点附近任意位置 (±几厘米)。机械臂固定到标定参考位
(相机装在机械臂上, 臂动=视角动 → 复现参考位=复现标定视角),
检测框底边中心 → calibrate.py 的 homography → 桌面坐标 (d 前方 mm,
l 横向 mm) → 底盘微转 yaw=atan2(l,d) 把物体转到正前方 → 盲抓已验证的
轨迹整体平移 (arm.moveto(x=r-96, y=120) 后三个相对 move 与盲抓逐字
相同) → 按类别参数 close_and_judge 判定 → 抬升 → 绝对转向 180° 到
B 区 → 放下。5 轮 ≥80% 验收 (与盲抓同标)。

机器人热点只接受一个 SDK 连接, 所以相机流与机械臂控制在同一个进程、
同一个 ep 对象里 —— 这是本脚本一体化存在的原因, 不能拆两个 ROS2 进程。

用法 (先跑 vision/calibrate.py 生成 ~/Team21/vision/calib.json):
    cd ~/Team21/colcon_ws/src/robomaster_pick_place_sim
    ~/Team21/Team21/bin/python3 vision/vision_pick_place.py --dry-run
    ~/Team21/Team21/bin/python3 vision/vision_pick_place.py --sequence bottle,tennis_ball
    ~/Team21/Team21/bin/python3 vision/vision_pick_place.py --target tennis_ball --runs 5

--dry-run: 只检测定位 (打印 d/l/r/yaw), 会把机械臂放回参考位但不抓取,
不动底盘/夹爪, 用于标定后先验证定位精度 (步骤 2)。
--keep-yaw: 兼容"实验员每轮不把机器人手动转回初始朝向"的协议
(默认模式假设每轮实验员会把机器人转回初始朝向, 无累计误差, 优先)。

日志: ~/Team21/logs/vision_pick_<时间戳>.txt
"""
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback

# ---------- sys.path 手术: 必须在 import numpy/cv2 之前 ----------
# 与 vision_detector.py 相同: 防止 ROS 的 numpy 1.x 顶掉 cv2 5.x 需要的
# numpy 2.x (2026-09-13 踩过)。
_demoted = [p for p in sys.path if "/opt/ros" in p or "/ros2_ws/" in p]
for p in _demoted:
    sys.path.remove(p)
sys.path.extend(_demoted)

import cv2          # noqa: E402
import numpy as np  # noqa: E402

LOG_DIR = os.path.expanduser("~/Team21/logs")
LOG_LINES = []
_LIVE_PATH = None
_LIVE_FILE = None


class _Tee:
    def __init__(self, stream):
        self.stream = stream

    def write(self, text):
        self.stream.write(text)
        if text.strip():
            LOG_LINES.append(text.rstrip("\n"))
        # 实时落盘; 必须写原始 text (print 会把分隔符拆成单独 write)
        if _LIVE_FILE is not None:
            try:
                _LIVE_FILE.write(text)
                _LIVE_FILE.flush()
            except Exception:
                pass
        return len(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _open_live_log():
    global _LIVE_PATH, _LIVE_FILE
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        _LIVE_PATH = os.path.join(LOG_DIR, f"vision_pick_{stamp}.txt")
        _LIVE_FILE = open(_LIVE_PATH, "w")
    except Exception:
        _LIVE_FILE = None


def save_log():
    try:
        if _LIVE_PATH:
            print(f"日志已保存: {_LIVE_PATH}")
            return
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOG_DIR, f"vision_pick_{stamp}.txt")
        with open(path, "w") as f:
            f.write("\n".join(LOG_LINES) + "\n")
        print(f"日志已保存: {path}")
    except Exception as e:
        print(f"保存日志失败: {e}")


def setup_display():
    """SSH 里跑也要能弹窗到板子的显示器上 (与 vision_detector.py 相同:
    不 pop LD_LIBRARY_PATH, rclpy 的原生库还要靠它)。"""
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
        for cand in ("/run/user/1000/gdm/Xauthority",
                     os.path.expanduser("~/.Xauthority")):
            if os.path.exists(cand):
                os.environ["XAUTHORITY"] = cand
                break
    print("DISPLAY:", os.environ.get("DISPLAY"),
          "XAUTHORITY:", os.environ.get("XAUTHORITY", "(无)"))


def prepare_ros_paths():
    """让 rclpy 可用: 补 sys.path + LD_LIBRARY_PATH (幂等, 与
    vision_detector.py 相同)。numpy/cv2 已导入并缓存, 不受影响。"""
    ros_site = [
        "/opt/ros/humble/lib/python3.10/site-packages",
        "/opt/ros/humble/local/lib/python3.10/dist-packages",
    ]
    for p in ros_site:
        if p not in sys.path:
            sys.path.append(p)
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    ros_libs = "/opt/ros/humble/lib:/opt/ros/humble/lib/aarch64-linux-gnu"
    if ros_libs not in ld:
        os.environ["LD_LIBRARY_PATH"] = ros_libs + (":" + ld if ld else "")


def current_wifi_ssid():
    try:
        out = subprocess.run(
            ["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        for line in out.splitlines():
            if line.startswith("yes:"):
                return line.split(":", 1)[1]
    except Exception:
        return "?"
    return ""


def cleanup_with_timeout(ep):
    """SDK 的 stop_video_stream/close 可能卡死, 放守护线程里限时 5 秒。"""
    def _clean():
        try:
            ep.camera.stop_video_stream()
        except Exception as e:
            print("stop_video_stream warning:", e)
        try:
            ep.close()
        except Exception as e:
            print("close warning:", e)

    t = threading.Thread(target=_clean, daemon=True)
    t.start()
    t.join(5)
    if t.is_alive():
        print("WARN: SDK 清理超时 (>5s), 直接退出 (守护线程会被回收)")


# ---------- TensorRT 推理 (复制自 vision_detector.py, TRT 10 API) ----------

CLASS_NAMES = ["bottle", "tennis_ball"]   # 与队友模型 data.yaml 顺序一致
CONF_THRESH = 0.5
NMS_THRESH = 0.45
COLORS = {"bottle": (0, 128, 255), "tennis_ball": (0, 255, 128)}


class TrtYOLO(object):
    """TensorRT YOLOv8 推理封装 (引擎输入 640x640)"""

    def __init__(self, engine_path, img_size):
        import tensorrt as trt
        import pycuda.autoinit  # noqa: F401  初始化 CUDA 上下文
        import pycuda.driver as cuda
        self._cuda = cuda
        self.img_size = img_size
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self._alloc_buf()

    def _alloc_buf(self):
        import tensorrt as trt
        import numpy as np
        host, device = {}, {}
        for name in ("images", "output0"):
            shape = self.engine.get_tensor_shape(name)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host[name] = np.empty(size, dtype=dtype)
            device[name] = self._cuda.mem_alloc(host[name].nbytes)
        self.input_host = host["images"]
        self.output_host = host["output0"]
        self.input_device = device["images"]
        self.output_device = device["output0"]

    @staticmethod
    def letterbox(img, new_size, color=(114, 114, 114)):
        h, w = img.shape[:2]
        r = min(float(new_size) / h, float(new_size) / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img, (nw, nh))
        pad_h, pad_w = (new_size - nh) / 2.0, (new_size - nw) / 2.0
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
        return padded, r, left, top

    def infer(self, bgr_frame):
        """输入 BGR 帧, 返回 (output, ratio, pad_x, pad_y)"""
        import numpy as np
        img, ratio, pad_x, pad_y = self.letterbox(bgr_frame, self.img_size)
        blob = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        np.copyto(self.input_host.reshape(-1), blob.reshape(-1))
        self._cuda.memcpy_htod_async(self.input_device, self.input_host,
                                     self.stream)
        self.context.set_tensor_address("images", int(self.input_device))
        self.context.set_tensor_address("output0", int(self.output_device))
        self.context.execute_async_v3(self.stream.handle)
        self._cuda.memcpy_dtoh_async(self.output_host, self.output_device,
                                     self.stream)
        self.stream.synchronize()
        output = self.output_host.reshape(4 + len(CLASS_NAMES), -1)
        return output, ratio, pad_x, pad_y

    @staticmethod
    def postprocess(output, ratio, pad_x, pad_y, img_w, img_h):
        import numpy as np
        boxes = output[:4].T          # (N, 4) cx cy w h (640 坐标系)
        scores = output[4:].T
        cls_ids = scores.argmax(axis=1)
        confs = scores.max(axis=1)
        keep = confs > CONF_THRESH
        boxes, cls_ids, confs = boxes[keep], cls_ids[keep], confs[keep]
        if len(boxes) == 0:
            return []

        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - pad_x) / ratio
        xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - pad_y) / ratio
        xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2 - pad_x) / ratio
        xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2 - pad_y) / ratio
        xyxy[:, 0] = np.clip(xyxy[:, 0], 0, img_w)
        xyxy[:, 1] = np.clip(xyxy[:, 1], 0, img_h)
        xyxy[:, 2] = np.clip(xyxy[:, 2], 0, img_w)
        xyxy[:, 3] = np.clip(xyxy[:, 3], 0, img_h)

        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        order = confs.argsort()[::-1]
        picked = []
        while len(order) > 0:
            i = order[0]
            picked.append(i)
            if len(order) == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(xyxy[i, 0], xyxy[rest, 0])
            yy1 = np.maximum(xyxy[i, 1], xyxy[rest, 1])
            xx2 = np.minimum(xyxy[i, 2], xyxy[rest, 2])
            yy2 = np.minimum(xyxy[i, 3], xyxy[rest, 3])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
            order = rest[iou < NMS_THRESH]

        dets = []
        for i in picked:
            x1, y1, x2, y2 = xyxy[i].astype(int)
            dets.append({
                "class": CLASS_NAMES[cls_ids[i]],
                "conf": round(float(confs[i]), 3),
                "box": [int(x1), int(y1), int(x2), int(y2)],
            })
        return dets


def _iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class TemporalSmoother(object):
    """时间平滑: 新目标需连续 2 帧确认, 已确认目标可容忍 3 帧丢失。"""

    def __init__(self, min_hits=2, max_miss=3, iou_thr=0.3):
        self.min_hits = min_hits
        self.max_miss = max_miss
        self.iou_thr = iou_thr
        self.tracks = []

    def update(self, dets):
        matched_dets = set()
        for t in self.tracks:
            best_i, best_d = None, None
            for i, d in enumerate(dets):
                if i in matched_dets or d["class"] != t["class"]:
                    continue
                if _iou(d["box"], t["box"]) >= self.iou_thr:
                    if best_d is None or d["conf"] > best_d["conf"]:
                        best_i, best_d = i, d
            if best_d is not None:
                matched_dets.add(best_i)
                t["box"], t["conf"] = best_d["box"], best_d["conf"]
                t["hits"] += 1
                t["miss"] = 0
            else:
                t["miss"] += 1
        self.tracks = [t for t in self.tracks if t["miss"] <= self.max_miss]
        for i, d in enumerate(dets):
            if i not in matched_dets:
                self.tracks.append({"class": d["class"], "conf": d["conf"],
                                    "box": d["box"], "hits": 1, "miss": 0})
        out = []
        for t in self.tracks:
            if t["miss"] == 0 and t["hits"] >= self.min_hits:
                out.append({"class": t["class"], "conf": t["conf"],
                            "box": t["box"]})
        return out


# ---------- 抓取参数 (分类策略) ----------
# tennis_ball: 实机扫参实值 (2026-09-11 gripper_sweep_ball_*.txt 结论:
# power=30 时 空夹 2.5s 到底 vs 夹球 1.5s 被挡, 差距最大)。
# bottle: 占位 = 网球值, M3 扫参 (gripper_status_sweep.py bottle_empty /
# bottle) 后更新 power/closed_fast/stable_time; 几何参数 (coarse_down/
# final_down/grasp_arm_y) 在 M4 单轮试抓后按"瓶身被夹住且不推倒"微调。
GRASP_PARAMS = {
    "tennis_ball": {
        "power": 30, "closed_fast": 2.0, "stable_time": 2.5, "timeout": 8.0,
        "grasp_arm_y": 120,              # moveto 起始高度 (与盲抓同)
        "extra_forward": 30, "coarse_forward": 60, "final_forward": 6,
        "coarse_down": -240, "final_down": -24,
        "test_lift": 25, "main_lift": 60,
        "release_down": -80, "final_lift": 70,
    },
    "bottle": {
        # 扫参实值 2026-09-13 (判读规则 2: 所有功率夹瓶也闭合, 取两遍
        # first_closed 时间差最大者): power=20 时空夹 2.5s vs 夹瓶 1.5s
        # (Δ=1.0s 最大), closed_fast 取中值 2.0s。日志:
        # ~/Desktop/gripper_sweep_bottle_empty_20260913_063403.txt
        # ~/Desktop/gripper_sweep_bottle_20260913_063527.txt
        # 注意 power=20 夹持力低于网球档 (30), 搬运是否够力留待步骤 4 单轮
        # 真抓确认; 若转运中脱手, 优先试 power=30 (Δ=0.5s, closed_fast 1.75)。
        "power": 20, "closed_fast": 2.0, "stable_time": 2.5, "timeout": 8.0,
        "grasp_arm_y": 120,
        "extra_forward": 30, "coarse_forward": 60, "final_forward": 6,
        "coarse_down": -240, "final_down": -24,
        "test_lift": 25, "main_lift": 60,
        "release_down": -80, "final_lift": 70,
    },
}

GRASP_OFFSET_X_MM = 96     # 盲抓轨迹净前移 (30+60+6), 平移基准
R_VALID_MM = (191.0, 266.0)   # r=hypot(d,l) 有效带; 下限 = 近排 196 内缩 5mm
                              # (再近相机照不到), 上限 = 远排 266
VERIFY_L_RESID_MM = 20.0   # 转向后残余横向偏移阈值 (mm)
MAX_YAW_FIX_TURNS = 2      # 转向修正最多次数
YAW_CORR_LIMIT_DEG = 15.0  # 单次修正角限幅
YAW_SIGN = +1              # 实测确定: 物体在 l>0 处, 底盘 z=YAW_SIGN*角度
                           # 转向应使物体进入画面中心 (验证协议步骤 0)
TURN_B_BASE = -180.0       # 与盲抓一致: z=-180 到 B 区; B 转命令
                           # = TURN_B_BASE - YAW_SIGN*yaw_exec, 绝对 180° 方向
LOCATE_TIMEOUT_S = 10.0    # 单轮定位超时
OPEN_POWER = 35

RUN_COUNT_DEFAULT = 5
INIT_SETTLE_TIME = 1.0
OPEN_TIME = 1.5
CHASSIS_SETTLE_TIME = 1.2
FORWARD_MOVE_TIME = 1.0
FORWARD_SETTLE_TIME = 1.0
COARSE_MOVE_TIME = 1.0
COARSE_SETTLE_TIME = 1.0
FINAL_MOVE_TIME = 1.5
GRASP_HEIGHT_PAUSE_TIME = 1.2
TEST_LIFT_TIME = 1.5
TEST_SETTLE_TIME = 1.0
MAIN_LIFT_TIME = 1.5
BEFORE_TURN_TIME = 1.5
AFTER_TURN_TIME = 1.5
LOWER_TIME = 1.5
GROUND_PAUSE_TIME = 1.2
RELEASE_TIME = 1.5
FINAL_LIFT_TIME = 1.5
TURN_SPEED_DPS = 20
ARM_POSE_TOL_MM = 12.0               # sub_position 校验机械臂参考位容差

gripper_status = {"value": "unknown", "ts": 0.0}
status_hook = None    # ROS2 状态发布 (report_status 时同步到话题)


class _Quit(Exception):
    """窗口按 q 请求退出。"""


# ---------- 标定数据 ----------

def load_calib(path):
    """读 calib.json, 返回含 H(np.float32) 的 dict; 不完整则报错。"""
    if not os.path.exists(path):
        print(f"ERROR: 标定文件不存在: {path}")
        print("       先跑 vision/calibrate.py 生成它 (标定与运行必须用同一"
              "个机械臂参考位)。")
        sys.exit(1)
    with open(path) as f:
        doc = json.load(f)
    if not doc.get("complete") or doc.get("H") is None:
        print(f"ERROR: {path} 是不完整的标定 (partial), 先完成 calibrate.py。")
        sys.exit(1)
    doc["H"] = np.float32(doc["H"])
    print(f"标定已加载: {path} (arm_pose={doc.get('arm_pose', 'recenter')}, "
          f"fit={doc.get('fit')})")
    return doc


def uv_to_dl(H, u, v):
    p = cv2.perspectiveTransform(np.float32([[[u, v]]]), H)[0][0]
    return float(p[0]), float(p[1])


def bottom_center(det):
    """检测框底边中心 (与标定时同口径)。"""
    x1, y1, x2, y2 = det["box"]
    return (x1 + x2) / 2.0, float(y2)


# ---------- 抓取基础 (复制自 pick_place_lua_params.py, close_and_judge
#            四阈值参数化; 网球传的正是原常量, 行为等价) ----------

def report_status(text):
    print(text)
    if status_hook is not None:
        try:
            status_hook(text)
        except Exception:
            pass


def step(attempt, number, label, sec=0.0):
    print("\n========================================")
    print(f"RUN {attempt} | STEP {number}: {label}")
    print("========================================")
    if sec > 0:
        time.sleep(sec)


def on_gripper_status(status):
    if isinstance(status, (list, tuple)) and status:
        status = status[0]
    gripper_status["value"] = status
    gripper_status["ts"] = time.time()


def close_and_judge(gripper, power, closed_fast, stable_time, timeout):
    """闭合夹爪并判定是否夹到物体, 返回 (success, detail)。

    判据来自实机功率-状态扫描数据:
      1. status 到达 "closed" 用时 <= closed_fast -> 物体挡在行程中间,
         爪子提前进入闭合区 -> 夹到物体 (实测: 夹网球 1.5s vs 空夹 2.5s)
      2. status 到达 "closed" 用时 > closed_fast -> 走满全程到机械限位
         -> 未夹到
      3. status 稳定在 "normal" 持续 stable_time 且从未 closed -> 刚性
         物体把爪子停在中间 -> 夹到 (水瓶预期走这个分支)
      4. 超时 / 无推送 -> 无法判定 -> 失败

    注意: 不要在中途 pause() —— 空夹时爪子还没走到限位就被冻结在中间,
    状态停在 normal 会被误判为夹到物体。
    """
    gripper_status["value"] = "unknown"
    gripper_status["ts"] = 0.0
    gripper.close(power=power)

    t0 = time.time()
    deadline = t0 + timeout
    last = None
    stable_since = None

    while True:
        now = time.time()
        v = str(gripper_status["value"]).strip().lower()
        ts = gripper_status.get("ts", 0.0)

        if v == "closed":
            elapsed = now - t0
            if elapsed <= closed_fast:
                return True, (
                    f"快速闭合({elapsed:.1f}s <= {closed_fast}s), "
                    f"判定夹到物体"
                )
            return False, (
                f"缓慢闭合({elapsed:.1f}s > {closed_fast}s), "
                f"走满行程, 未夹到物体"
            )

        if v != last:
            last = v
            stable_since = now
        elif (v == "normal" and stable_since is not None
              and now - stable_since >= stable_time
              and ts > 0 and now - ts < 1.0):
            return True, (
                f"夹爪稳定停在中间位置 normal({stable_time}s), "
                f"判定夹到物体"
            )

        if now >= deadline:
            return False, (
                f"超时({timeout}s)未得到稳定判定, "
                f"最后状态 {gripper_status['value']}"
            )

        time.sleep(0.2)


def move_arm_delta(arm, dx_mm, dy_mm, label, wait_time):
    print(f"{label}: arm.move x={dx_mm} mm, y={dy_mm} mm")
    arm.move(x=dx_mm, y=dy_mm).wait_for_completed()
    time.sleep(wait_time)


def safe_home(arm, gripper):
    print("\n========================================")
    print("SAFE HOME: open gripper and recenter arm")
    print("========================================")
    try:
        gripper.open(power=OPEN_POWER)
        time.sleep(1.0)
    except Exception as e:
        print("Open gripper warning:", e)
    try:
        arm.recenter().wait_for_completed()
        time.sleep(2.0)
    except Exception as e:
        print("Arm recenter warning:", e)


def robot_signal_error(ep):
    try:
        from robomaster import led
        ep.led.set_led(comp=led.COMP_ALL, r=255, g=0, b=0,
                       effect=led.EFFECT_FLASH, freq=5)
    except Exception as e:
        print("Set error LED warning:", e)
    try:
        from robomaster import robot
        ep.play_sound(robot.SOUND_ID_ATTACK).wait_for_completed(timeout=3)
    except Exception as e:
        print("Play error sound warning:", e)


def robot_signal_success(ep):
    try:
        from robomaster import led
        ep.led.set_led(comp=led.COMP_ALL, r=0, g=255, b=0,
                       effect=led.EFFECT_ON)
    except Exception as e:
        print("Set success LED warning:", e)
    try:
        from robomaster import robot
        ep.play_sound(robot.SOUND_ID_RECOGNIZED).wait_for_completed(timeout=3)
    except Exception as e:
        print("Play success sound warning:", e)


def robot_signal_idle(ep):
    try:
        from robomaster import led
        ep.led.set_led(comp=led.COMP_ALL, effect=led.EFFECT_OFF)
    except Exception as e:
        print("Set idle LED warning:", e)


# ---------- 定位 ----------

def get_smoothed_dets(ep, detector, smoother, win_name=None):
    """取一帧 → 推理 → 平滑。返回 (dets, bgr); 取不到帧返回 (None, None)。"""
    img = None
    for _ in range(5):
        try:
            img = ep.camera.read_cv2_image(timeout=2, strategy="newest")
        except Exception:
            continue
        if img is not None:
            break
    if img is None:
        return None, None
    # SDK 解码出的是 RGB → BGR, 必须 ascontiguousarray (cv2 5.x, 2026-09-13)
    bgr = np.ascontiguousarray(img[:, :, ::-1])
    output, ratio, px, py = detector.infer(bgr)
    dets = TrtYOLO.postprocess(output, ratio, px, py,
                               bgr.shape[1], bgr.shape[0])
    dets = smoother.update(dets)
    if win_name is not None:
        draw_frame(bgr, dets, None)
        cv2.imshow(win_name, bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise _Quit()
    return dets, bgr


def draw_frame(bgr, dets, fps):
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        color = COLORS.get(d["class"], (0, 255, 0))
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        label = "{} {:.2f}".format(d["class"], d["conf"])
        cv2.putText(bgr, label, (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if fps is not None:
        cv2.putText(bgr, "FPS: {:.1f}".format(fps), (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


def select_target(dets, img_h, target):
    """按目标类别选一个检测; auto 模式取 conf 最高且框底边在画面下 2/3。"""
    cands = [d for d in dets
             if target == "auto" or d["class"] == target]
    if target == "auto":
        cands = [d for d in cands if d["box"][3] > img_h * 2 / 3]
    if not cands:
        return None
    return max(cands, key=lambda d: d["conf"])


def wait_for_target(ep, detector, smoother, target, timeout_s, win_name=None):
    """等到目标类别的平滑检测连续 3 帧稳定 (IoU>=0.5), 返回 det 或 None。"""
    t0 = time.time()
    last_box, stable = None, 0
    last_print = 0.0
    while time.time() - t0 < timeout_s:
        dets, bgr = get_smoothed_dets(ep, detector, smoother, win_name)
        img_h = bgr.shape[0] if bgr is not None else 360
        det = select_target(dets or [], img_h, target)
        if det is not None:
            if last_box is not None and _iou(last_box, det["box"]) >= 0.5:
                stable += 1
            else:
                stable = 1
            last_box = det["box"]
            now = time.time()
            if now - last_print >= 2.0:
                print(f"检测到 {det['class']} conf={det['conf']} "
                      f"稳定 {stable}/3 ...")
                last_print = now
            if stable >= 3:
                return det
        else:
            last_box, stable = None, 0
    return None


# ---------- 动作 ----------

def wait_arm_pose(ep, target_xy, tol=ARM_POSE_TOL_MM, timeout=8.0):
    """sub_position 订阅实际臂坐标, 等到 (x,y) 到位 (或超时)。返回 (ok, actual)。"""
    tx, ty = target_xy
    got = {"xy": None, "done": False}

    def _cb(*args):
        try:
            info = args[0] if args else None
            if not isinstance(info, (list, tuple)) or len(info) < 2:
                return
            x, y = float(info[0]), float(info[1])
            got["xy"] = (x, y)
            got["done"] = (abs(x - tx) <= tol and abs(y - ty) <= tol)
        except Exception:
            pass

    try:
        ep.robotic_arm.sub_position(freq=5, callback=_cb)
    except Exception as e:
        print("sub_position warning:", e)
        return True, None
    t0 = time.time()
    while time.time() - t0 < timeout:
        if got["done"]:
            break
        time.sleep(0.1)
    try:
        ep.robotic_arm.unsub_position()
    except Exception:
        pass
    ok = got["done"]
    print(f"机械臂实际位置={got['xy']} (目标 {target_xy}, 容差 {tol}mm) "
          f"{'OK' if ok else '未到位'}")
    return ok, got["xy"]


def set_arm_pose(ep, arm_pose):
    """每轮开始复现标定机械臂参考位 (相机随臂): recenter → 可选 moveto → 校验。"""
    try:
        ep.robotic_arm.recenter().wait_for_completed(timeout=8)
        time.sleep(2.0)
    except Exception as e:
        print("Arm recenter warning:", e)
    if arm_pose == "recenter" or not arm_pose:
        print("机械臂参考位 = recenter (与标定时一致)。")
        return True, None
    print(f"机械臂 moveto 参考位 {arm_pose} ...")
    try:
        ep.robotic_arm.moveto(x=arm_pose[0],
                              y=arm_pose[1]).wait_for_completed(timeout=8)
    except Exception as e:
        print(f"arm.moveto 异常: {type(e).__name__}: {e}")
    return wait_arm_pose(ep, arm_pose)


def turn_chassis(chassis, z_deg, label):
    if abs(z_deg) < 0.5:
        print(f"{label}: 角度过小 ({z_deg:+.2f}°), 跳过。")
        return
    print(f"{label}: chassis.move z={z_deg:+.1f}° (speed {TURN_SPEED_DPS})")
    chassis.move(x=0, y=0, z=z_deg,
                 z_speed=TURN_SPEED_DPS).wait_for_completed()
    time.sleep(CHASSIS_SETTLE_TIME)


def turn_to_object(chassis, yaw_deg, calib, ep, detector, smoother,
                   target, win_name):
    """转 YAW_SIGN*yaw_deg 使物体到正前方; 二次检测修正残余横向偏移。
    返回实际总转角 yaw_exec ((d,l) 坐标系度数, 含修正)。"""
    yaw_exec = yaw_deg
    remaining = yaw_deg
    for k in range(MAX_YAW_FIX_TURNS + 1):
        turn_chassis(chassis, YAW_SIGN * remaining,
                     f"turn toward object (修正第 {k} 次)")
        if k == MAX_YAW_FIX_TURNS:
            break
        det = wait_for_target(ep, detector, smoother, target, 5.0, win_name)
        if det is None:
            print("二次检测未找到目标, 跳过转向修正。")
            break
        u, v = bottom_center(det)
        d2, l2 = uv_to_dl(calib["H"], u, v)
        if abs(l2) <= VERIFY_L_RESID_MM:
            print(f"转向后残余横向 l={l2:+.1f}mm, 在容差内。")
            break
        corr = math.degrees(math.atan2(l2, d2))
        corr = max(-YAW_CORR_LIMIT_DEG, min(YAW_CORR_LIMIT_DEG, corr))
        print(f"残余横向 l={l2:+.1f}mm → 修正 {corr:+.1f}°")
        remaining = corr
        yaw_exec += corr
    return yaw_exec


def grasp_sequence(arm, gripper, p, r):
    """盲抓轨迹整体平移: moveto(x=r-96, y=120) 后三个相对 move 与盲抓
    逐字相同 → 爪心最终落在 x=r, y=-144 (物体上)。返回 close 判定。"""
    start_x = r - GRASP_OFFSET_X_MM
    print(f"grasp: arm.moveto x={start_x:.1f} mm, y={p['grasp_arm_y']} mm "
          f"(r={r:.1f}, 爪心终点 x={r:.1f})")
    arm.moveto(x=start_x, y=p["grasp_arm_y"]).wait_for_completed()
    time.sleep(INIT_SETTLE_TIME)

    move_arm_delta(arm, p["extra_forward"], 0, "forward alignment",
                   FORWARD_MOVE_TIME)
    time.sleep(FORWARD_SETTLE_TIME)
    move_arm_delta(arm, p["coarse_forward"], p["coarse_down"],
                   "lean forward and down", COARSE_MOVE_TIME)
    time.sleep(COARSE_SETTLE_TIME)
    move_arm_delta(arm, p["final_forward"], p["final_down"],
                   "final forward and down", FINAL_MOVE_TIME)
    time.sleep(GRASP_HEIGHT_PAUSE_TIME)

    return close_and_judge(gripper, p["power"], p["closed_fast"],
                           p["stable_time"], p["timeout"])


def place_at_b(arm, gripper, chassis, p, turn_z_deg):
    """抬升 → 绝对转向 B → 放下 → 抬走 (盲抓步骤 11-20 同构)。"""
    move_arm_delta(arm, 0, p["test_lift"], "small test lift", TEST_LIFT_TIME)
    time.sleep(TEST_SETTLE_TIME)
    move_arm_delta(arm, 0, p["main_lift"], "main lift", MAIN_LIFT_TIME)
    time.sleep(BEFORE_TURN_TIME)

    turn_chassis(chassis, turn_z_deg, "absolute turn to B point (180°)")
    time.sleep(AFTER_TURN_TIME)

    move_arm_delta(arm, 0, p["release_down"], "lower object", LOWER_TIME)
    time.sleep(GROUND_PAUSE_TIME)
    print("release gripper")
    gripper.open(power=OPEN_POWER)
    time.sleep(RELEASE_TIME)
    move_arm_delta(arm, 0, p["final_lift"], "lift away", FINAL_LIFT_TIME)


# ---------- 单轮状态机 ----------

def run_once(attempt, ep, arm, gripper, chassis, detector, smoother,
             calib, target, win_name, yaw_state, locate_timeout):
    """视觉抓取一轮。成功返回 (True, record), 失败返回 (False, detail)。"""
    arm_pose = calib.get("arm_pose", "recenter")
    step(attempt, "0A", f"ARM TO CALIB REF POSE ({arm_pose})")
    set_arm_pose(ep, arm_pose)

    step(attempt, "0B", "OPEN GRIPPER")
    gripper.open(power=OPEN_POWER)
    time.sleep(OPEN_TIME)

    step(attempt, 1, f"LOCATE TARGET ({target})")
    det = wait_for_target(ep, detector, smoother, target, locate_timeout,
                          win_name)
    if det is None:
        detail = f"定位超时 ({locate_timeout:.0f}s 内未检测到稳定目标)"
        report_status(f"ERROR: RUN {attempt} {detail}, 停止循环")
        robot_signal_error(ep)
        safe_home(arm, gripper)
        return False, detail

    cls = det["class"]
    u, v = bottom_center(det)
    d, l = uv_to_dl(calib["H"], u, v)
    r = math.hypot(d, l)
    yaw_i = math.degrees(math.atan2(l, d))
    print(f"定位: {cls} conf={det['conf']} (u={u:.1f}, v={v:.1f}) → "
          f"d={d:.1f} l={l:+.1f} r={r:.1f} yaw={yaw_i:+.1f}°")
    if not (R_VALID_MM[0] <= r <= R_VALID_MM[1]):
        detail = (f"物体距离 r={r:.0f}mm 超出有效带 {R_VALID_MM}, "
                  f"请重摆 (中心 216)")
        report_status(f"ERROR: RUN {attempt} {detail}, 停止循环")
        robot_signal_error(ep)
        safe_home(arm, gripper)
        return False, detail

    p = GRASP_PARAMS[cls]
    print(f"抓取参数: {cls} power={p['power']} closed_fast={p['closed_fast']}s "
          f"stable={p['stable_time']}s")

    step(attempt, 2, "TURN TOWARD OBJECT (yaw correction)")
    yaw_exec = turn_to_object(chassis, yaw_i, calib, ep, detector, smoother,
                              target, win_name)
    print(f"对齐完成: yaw 目标 {yaw_i:+.1f}°, 实际转 {yaw_exec:+.1f}°")

    step(attempt, 3, "APPROACH & DESCEND (r-96 平移轨迹)")
    ok, detail = grasp_sequence(arm, gripper, p, r)

    step(attempt, 4, "JUDGE GRASP BY CLOSE TIME")
    print(f"gripper close judgment: ok={ok}, detail={detail}")
    if not ok:
        detail = f"抓取失败 - {detail}"
        report_status(f"ERROR: RUN {attempt} {detail}, 停止循环")
        robot_signal_error(ep)
        safe_home(arm, gripper)
        return False, detail
    report_status(f"SUCCESS: RUN {attempt} {cls} {detail}")
    robot_signal_success(ep)

    step(attempt, 5, "TURN TO B POINT + PLACE")
    yaw_state["total"] += yaw_exec
    ref = yaw_state["total"] if yaw_state["keep"] else yaw_exec
    turn_z = TURN_B_BASE - YAW_SIGN * ref
    print(f"B 转向: yaw_exec={yaw_exec:+.1f}° "
          f"(累计 {yaw_state['total']:+.1f}°) → turn_z={turn_z:+.1f}°")
    place_at_b(arm, gripper, chassis, p, turn_z)

    step(attempt, 6, "DONE - RUN COMPLETE")
    record = {"cls": cls, "conf": det["conf"], "d": round(d, 1),
              "l": round(l, 1), "r": round(r, 1),
              "yaw": round(yaw_exec, 1), "detail": detail}
    report_status(f"SUCCESS: RUN {attempt} 抓取-放置完成 "
                  f"({cls}, d={d:.0f}, l={l:+.0f}, yaw={yaw_exec:+.1f}°)")
    robot_signal_idle(ep)
    return True, record


# ---------- 入口 ----------

def _run():
    import argparse
    ap = argparse.ArgumentParser(
        description="视觉检测联动定点抓取 (先跑 vision/calibrate.py 标定)")
    ap.add_argument("--target", choices=["bottle", "tennis_ball", "auto"],
                    default="auto",
                    help="每轮目标类别 (默认 auto: 任一类取 conf 最高)")
    ap.add_argument("--sequence", default=None,
                    help='显式指定每轮类别, 如 "bottle,tennis_ball,bottle" '
                         "(覆盖 --target, 长度不足则最后一类循环)")
    ap.add_argument("--runs", type=int, default=RUN_COUNT_DEFAULT,
                    help="抓取轮数 (默认 5, 验收 ≥80%)")
    ap.add_argument("--calib", default=os.path.expanduser(
                        "~/Team21/vision/calib.json"),
                    help="标定文件 (calibrate.py 产物)")
    ap.add_argument("--engine",
                    default=os.path.expanduser(
                        "~/Team21/vision/yolov8s_fp16.engine"),
                    help="TensorRT 引擎路径")
    ap.add_argument("--conf", type=float, default=CONF_THRESH,
                    help="置信度阈值 (默认 0.5)")
    ap.add_argument("--resolution", default="360p",
                    help="SDK 视频流分辨率: 360p/540p/720p")
    ap.add_argument("--target-timeout", type=float, default=LOCATE_TIMEOUT_S,
                    help="单轮定位超时秒数 (默认 10)")
    ap.add_argument("--retry-pause", type=float, default=2.0,
                    help="轮间准备时间秒数 (默认 2)")
    ap.add_argument("--dry-run", action="store_true",
                    help="只检测定位 (打印 d/l/r/yaw); 会把机械臂放回参考位"
                         "但不抓取, 不动底盘/夹爪")
    ap.add_argument("--arm-pose", default=None,
                    help='覆盖 calib.json 的机械臂参考位, 如 "100,220" '
                         "(mm; 标定时 --arm-pose 用了哪个这里就用哪个)")
    ap.add_argument("--keep-yaw", action="store_true",
                    help="兼容'实验员不把机器人转回初始朝向'的协议 "
                         "(默认模式假设每轮转回, 无累计误差)")
    ap.add_argument("--no-ros", action="store_true",
                    help="不启动 ROS2 状态话题")
    ap.add_argument("--no-display", action="store_true",
                    help="不弹窗口 (无显示器时)")
    args = ap.parse_args()

    setup_display()

    # 1) 弹窗自检 (先弹窗口后连机器人, 与 vision_detector.py 一致)
    win_name = None
    if not args.no_display:
        win_name = "Vision Pick-Place (bottle/tennis_ball)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        black = np.zeros((360, 640, 3), np.uint8)
        cv2.putText(black, "waiting robot...", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow(win_name, black)
        cv2.waitKey(1)
        print(f"窗口 {win_name!r} 已创建。")

    # 2) 加载 TensorRT 引擎
    print(f"加载引擎: {args.engine}")
    if not os.path.exists(args.engine):
        print(f"ERROR: 引擎不存在: {args.engine}")
        sys.exit(1)
    try:
        detector = TrtYOLO(args.engine, 640)
    except ImportError as e:
        print(f"ERROR: 导入 tensorrt/pycuda 失败: {e}")
        sys.exit(1)
    globals()["CONF_THRESH"] = args.conf
    print(f"引擎加载 OK, conf={args.conf}")

    # 3) 标定数据
    calib = load_calib(args.calib)
    if args.arm_pose:
        calib["arm_pose"] = [float(x) for x in args.arm_pose.split(",")]
        print(f"--arm-pose 覆盖机械臂参考位: {calib['arm_pose']}")

    # 4) ROS2 状态话题 (尽力而为, --no-ros 或环境缺 ROS 都继续跑)
    node = None
    if not args.no_ros:
        prepare_ros_paths()
        try:
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import String
            rclpy.init()
            node = Node("vision_pick_place")
            pub = node.create_publisher(String, "vision_pick_place/status", 10)

            def _publish(text):
                s = String()
                s.data = text
                pub.publish(s)

            globals()["status_hook"] = _publish
            print("ROS2 节点 vision_pick_place 已启动, "
                  "发布话题 vision_pick_place/status")
        except Exception as e:
            print(f"WARN: ROS2 启动失败 ({type(e).__name__}: {e}), "
                  "继续跑, 只是不发状态话题。")

    # 5) 预检: 板子必须已连机器人热点
    ssid = current_wifi_ssid()
    print(f"current wifi: {ssid!r}")
    if not ssid.startswith("RMEP"):
        print("ERROR: 板子当前不在机器人热点上, 中止。")
        print("       先开机机器人, 然后执行: nmcli connection up RMEP-21bdc0")
        sys.exit(1)

    # 6) 连机器人 + 开视频流
    from robomaster import robot
    ep = robot.Robot()
    initialized = False
    for attempt in range(1, 4):
        try:
            initialized = ep.initialize(conn_type="ap")
        except Exception as e:
            print(f"初始化第 {attempt} 次异常: {type(e).__name__}: {e}")
        if initialized:
            break
        if attempt < 3:
            print(f"初始化第 {attempt} 次失败, 5 秒后重试 ...")
            time.sleep(5)
    if not initialized:
        print("ERROR: 连不上机器人 (重试 3 次失败)。")
        sys.exit(1)

    print(f"开启视频流 ({args.resolution}) ...")
    ok = ep.camera.start_video_stream(display=False, resolution=args.resolution)
    if not ok:
        print("ERROR: 视频流开启失败 (检查手机 App 是否还连着机器人)")
        cleanup_with_timeout(ep)
        sys.exit(1)
    print("视频流已开启。")

    arm = ep.robotic_arm
    gripper = ep.gripper
    chassis = ep.chassis
    smoother = TemporalSmoother()
    arm_pose = calib.get("arm_pose", "recenter")

    try:
        gripper.sub_status(freq=5, callback=on_gripper_status)
        time.sleep(0.5)
    except Exception as e:
        print("Gripper status subscribe warning:", e)

    # 7) dry-run: 只定位不动作
    if args.dry_run:
        print("\n==== DRY RUN: 只检测定位, 不动机械臂/底盘/夹爪 ====")
        print(f"机械臂回标定参考位 (arm_pose={arm_pose}) ...")
        set_arm_pose(ep, arm_pose)
        print("把物体摆到 A 区任意位置; 每定位一次会打印 (d, l, r, yaw),")
        print("拿开物体后可以放下一处; q (窗口) 或 Ctrl+C 退出。")
        try:
            while True:
                det = wait_for_target(ep, detector, smoother, "auto",
                                      args.target_timeout, win_name)
                if det is None:
                    print(f"{args.target_timeout:.0f}s 内未检测到目标, 继续等待 ...")
                    continue
                u, v = bottom_center(det)
                d, l = uv_to_dl(calib["H"], u, v)
                r = math.hypot(d, l)
                yaw = math.degrees(math.atan2(l, d))
                in_band = R_VALID_MM[0] <= r <= R_VALID_MM[1]
                print(f"定位: {det['class']} conf={det['conf']} "
                      f"(u={u:.1f}, v={v:.1f}) → d={d:.1f} l={l:+.1f} "
                      f"r={r:.1f} yaw={yaw:+.1f}° "
                      f"[有效带{'内' if in_band else '外! 需重摆'}]")
                t0 = time.time()
                cleared = False
                while time.time() - t0 < 60:
                    dets, _ = get_smoothed_dets(ep, detector, smoother,
                                                win_name)
                    if not dets:
                        cleared = True
                        break
                    time.sleep(0.3)
                if cleared:
                    print("已清空, 可以放下一处。")
                else:
                    print("60s 未清空, 继续定位 (可能又检测到同一物体)。")
        except _Quit:
            print("用户按 q 退出 dry-run。")
        except KeyboardInterrupt:
            print("Interrupted by user.")
        _shutdown(node)
        cleanup_with_timeout(ep)
        print("dry-run done.")
        return

    # 8) 每轮目标类别
    if args.sequence:
        seq = [s.strip() for s in args.sequence.split(",") if s.strip()]
        for s in seq:
            if s not in GRASP_PARAMS:
                print(f"ERROR: --sequence 里有未知类别 {s!r} "
                      f"(可选 {sorted(GRASP_PARAMS)})")
                sys.exit(1)
        print(f"--sequence: {seq}")
    else:
        seq = None
        print(f"--target: {args.target}")
    runs = max(1, args.runs)

    def target_for(i):
        if seq:
            return seq[min(i, len(seq) - 1)]
        return args.target

    # 9) 主循环
    yaw_state = {"total": 0.0, "keep": args.keep_yaw}
    success_count = 0
    fail_count = 0
    records = []
    aborted = False
    print("\n========================================")
    print(f"准备开始: {runs} 轮, 目标 {seq or args.target}")
    protocol = ("--keep-yaw (每轮不转回机器人)" if args.keep_yaw
                else "每轮开始实验员把机器人手动转回初始朝向")
    print(f"协议: {protocol}")
    print("========================================")

    try:
        for attempt in range(1, runs + 1):
            print(f"\n>>> 第 {attempt}/{runs} 轮开始, 请确认物体已摆好 "
                  "(并已把机器人转回初始朝向, 除非 --keep-yaw)。")
            target = target_for(attempt - 1)
            try:
                ok, res = run_once(attempt, ep, arm, gripper, chassis,
                                   detector, smoother, calib, target,
                                   win_name, yaw_state,
                                   args.target_timeout)
            except _Quit:
                print("用户按 q 中止循环。")
                safe_home(arm, gripper)
                aborted = True
                break
            if not ok:
                fail_count += 1
                records.append(f"RUN {attempt}: FAILED - {res}")
                print("Loop stopped because grasp/locate failed.")
                break
            success_count += 1
            records.append(
                f"RUN {attempt}/{runs} | target={res['cls']} | "
                f"d={res['d']:.1f} l={res['l']:+.1f} yaw={res['yaw']:+.1f}° "
                f"| power={GRASP_PARAMS[res['cls']]['power']} "
                f"| 判定={res['detail']}"
            )
            if attempt < runs:
                print(f"第 {attempt} 轮成功。下一轮: 把物体摆回 A 区, "
                      f"{args.retry_pause:.0f}s 后开始。")
                time.sleep(args.retry_pause)

        print("\n========================================")
        print("FINAL RESULT")
        print("========================================")
        for line in records:
            print(line)
        print(f"success: {success_count}")
        print(f"failed: {fail_count}")
        print(f"planned runs: {runs}")
        if aborted:
            print("(用户中止)")

        passed = success_count >= max(1, int(0.8 * runs))
        report_status(
            f"验收结果: {'通过' if passed else '未通过'} "
            f"({success_count}/{runs})"
        )
        if fail_count > 0 or aborted:
            report_status(f"ERROR: 循环已停止 (success={success_count}, "
                          f"failed={fail_count}, planned={runs})")
            robot_signal_error(ep)
        else:
            report_status(f"SUCCESS: 全部 {success_count}/{runs} 次抓取成功")
            robot_signal_success(ep)

    except KeyboardInterrupt:
        print("Interrupted by user.")
        safe_home(arm, gripper)
    except _Quit:
        print("用户按 q 中止。")
        safe_home(arm, gripper)
    except Exception as e:
        print("Unexpected error:", e)
        traceback.print_exc()
        try:
            safe_home(arm, gripper)
        except Exception:
            pass
        robot_signal_error(ep)

    try:
        gripper.unsub_status()
    except Exception:
        pass
    _shutdown(node)
    cleanup_with_timeout(ep)
    print("vision pick-place done.")


def _shutdown(node):
    if node is None:
        return
    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        import rclpy
        rclpy.shutdown()
    except Exception:
        pass


def main():
    _open_live_log()
    sys.stdout = _Tee(sys.stdout)
    sys.stderr = _Tee(sys.stderr)
    code = 0
    try:
        _run()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 0
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        save_log()
        if _LIVE_FILE is not None:
            try:
                _LIVE_FILE.close()
            except Exception:
                pass
        # SDK 的非守护线程会让解释器退出卡死, 直接硬退出
        os._exit(code)


if __name__ == "__main__":
    main()
