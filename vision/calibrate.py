#!/usr/bin/env python3
"""视觉标定工具: 桌面网格 + 网球 → 像素(u,v) ↔ 桌面平面(d,l) 的 homography。

用途: 给 vision_pick_place.py 提供"像素 → 机械臂坐标"映射。
方法: 桌面 A 区贴 3×3 网格标记 (d=[196,216,266]mm, l=[-100,0,100]mm;
      最近排原定 166 被机械臂挡出视野, 实机挪到 196),
      机械臂固定到参考位 (相机装在机械臂上, 臂动=视角动; 默认 recenter,
      可 --arm-pose "x,y" 抬臂换视角),
      逐点把网球放到标记上 → 取检测框底边中心像素 (u,v) →
      cv2.findHomography(RANSAC) 拟合像素→桌面 (d,l) 映射 → 保存 calib.json。

用法 (板子已连机器人热点, 手机 App 断开机器人):
    cd ~/Team21/colcon_ws/src/robomaster_pick_place_sim
    ~/Team21/Team21/bin/python3 vision/calibrate.py
    ~/Team21/Team21/bin/python3 vision/calibrate.py --arm-pose "100,220"
    ~/Team21/Team21/bin/python3 vision/calibrate.py --resume calib_partial_xxx.json

窗口按键 (默认, 人站在桌边放球即可, 不用回键盘):
    回车/空格 = 采样当前点   r = 重采当前点   u = 撤销上一步 (采点或跳过)
    s = 跳过当前点   c = 拟合并打印误差表   g <数字> 回车 = 跳到第 i 点
    q = 结束: 有效点>=4 拟合后存 calib.json, 否则存 calib_partial_<ts>.json
--cli-input: 终端整行命令模式 (无显示器时), 命令同上 (如 "g 3")。

重要: vision_pick_place.py 定位时必须把机械臂放回同一个参考位 (arm_pose 存在
calib.json 里, 抓取脚本自动复现)。相机随臂, 参考位不同 = 标定作废, 需重标。

日志: ~/Team21/logs/calibrate_<时间戳>.txt
"""
import json
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
        _LIVE_PATH = os.path.join(LOG_DIR, f"calibrate_{stamp}.txt")
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
        path = os.path.join(LOG_DIR, f"calibrate_{stamp}.txt")
        with open(path, "w") as f:
            f.write("\n".join(LOG_LINES) + "\n")
        print(f"日志已保存: {path}")
    except Exception as e:
        print(f"保存日志失败: {e}")


def setup_display():
    """SSH 里跑也要能弹窗到板子的显示器上 (与 vision_detector.py 相同,
    但标定不 import rclpy, 不需要 LD_LIBRARY_PATH)。"""
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


# ---------- 标定参数 ----------

GRID_D_MM = [196, 216, 266]        # 以盲抓目标 216 为中心; 最近排 166
                                   # 被收起的机械臂挡出视野 (实机确认),
                                   # 挪到 196; 远排 266
GRID_L_MM = [-100, 0, 100]         # 3x3 = 9 点 (左右 ±10cm); l 正方向与
                                   # YAW_SIGN 一起在实机上确认
                                   # (vision_pick_place.py)
N_COLS = len(GRID_L_MM)
N_POINTS = len(GRID_D_MM) * N_COLS
SAMPLE_FRAMES = 10                   # 每点采样帧数 (取中位数)
SAMPLE_TIMEOUT_S = 30.0              # 单点采样总超时
STD_WARN_PX = 3.0                    # 采样像素标准差告警线 (球在动?)
RANSAC_THRESH_MM = 5.0               # findHomography RANSAC 阈值
FIT_RMS_LIMIT_MM = 8.0               # 拟合 RMS 通过阈值 (>=6 点)
FIT_MAX_LIMIT_MM = 15.0              # 单点最大重投影误差阈值
FIT_RMS_TIGHT_MM = 5.0               # 4~5 点时收紧的 RMS 阈值 (无冗余)
ARM_POSE_TOL_MM = 12.0               # sub_position 校验机械臂参考位的容差

KEY_ENTER = 13
KEY_SPACE = 32
KEY_ESC = 27
KEY_BACKSPACE = 8
KEY_Q = ord("q")
KEY_R = ord("r")
KEY_U = ord("u")
KEY_S = ord("s")
KEY_C = ord("c")
KEY_G = ord("g")


def grid_point(i):
    """网格点索引 → (d_mm, l_mm)。i = 0..N_POINTS-1, 按行 (d 固定, l 变化)。"""
    return GRID_D_MM[i // N_COLS], GRID_L_MM[i % N_COLS]


def bottom_center(det):
    """检测框底边中心: 与抓取时定位取点同口径 (系统误差被 homography 吸收)。"""
    x1, y1, x2, y2 = det["box"]
    return (x1 + x2) / 2.0, float(y2)


def pick_ball_det(dets):
    """标定时桌面只放网球; 多框时取 conf 最高且框底边在画面下 2/3 的。"""
    best = None
    for d in dets:
        if d["class"] != "tennis_ball":
            continue
        if best is None or d["conf"] > best["conf"]:
            best = d
    return best


# ---------- 拟合与保存 ----------

def fit_points(valid_points):
    """像素(u,v) → 桌面(d,l) homography。返回 (H, errs_mm, inlier_mask,
    rms_mm, max_mm); 点数不足返回 None。"""
    if len(valid_points) < 4:
        return None
    src = np.float32([[p["u"], p["v"]] for p in valid_points])
    dst = np.float32([[p["d_mm"], p["l_mm"]] for p in valid_points])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC,
                                 ransacReprojThreshold=RANSAC_THRESH_MM)
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    errs = np.linalg.norm(proj - dst, axis=1)
    if mask is None:
        mask = np.ones(len(valid_points), bool)
    else:
        mask = mask.ravel().astype(bool)
    inl = errs[mask]
    rms = float(np.sqrt(np.mean(inl ** 2))) if len(inl) else 0.0
    return H, errs, mask, rms, float(errs.max())


def uv_to_dl(H, u, v):
    p = cv2.perspectiveTransform(np.float32([[[u, v]]]), H)[0][0]
    return float(p[0]), float(p[1])


def report_fit(points):
    """打印拟合质量报告; 返回 (ok, H, fit_info)。"""
    valid = [p for p in points if p and not p.get("skipped")
             and p.get("u") is not None]
    print("\n---- 拟合报告 ----")
    print(f"有效点: {len(valid)} / {N_POINTS}")
    res = fit_points(valid)
    if res is None:
        print(f"ERROR: 有效点 <4 ({len(valid)}), 无法拟合, 继续采点。")
        return False, None, None
    H, errs, mask, rms, mx = res
    for p, e, inl in zip(valid, errs, mask):
        print(f"  点{p['i']} (d={p['d_mm']}, l={p['l_mm']}): "
              f"(u={p['u']:.1f}, v={p['v']:.1f}) 误差 {e:.1f}mm"
              + ("" if inl else "  <-- RANSAC 外点"))
    limit = FIT_RMS_LIMIT_MM if len(valid) >= 6 else FIT_RMS_TIGHT_MM
    ok = rms <= limit and mx <= FIT_MAX_LIMIT_MM and mask.all()
    print(f"RMS={rms:.1f}mm (限 {limit:.1f}), 最大={mx:.1f}mm "
          f"(限 {FIT_MAX_LIMIT_MM:.1f}), "
          f"外点={int((~mask).sum())} → {'通过' if ok else '不通过'}")
    if not ok:
        worst = sorted(range(len(valid)), key=lambda i: -errs[i])[:2]
        print("建议重采: " + ", ".join(f"点{valid[i]['i']}"
              f"(误差{errs[i]:.1f}mm)" for i in worst)
              + " (窗口按 g <点号> 跳过去重采)")
    return ok, H, {"rms_mm": round(rms, 2), "max_mm": round(mx, 2),
                   "used": len(valid), "outliers": []}


def save_calib(path, points, H, arm_pose, resolution, engine, complete):
    """写 calib.json (或 partial)。points 里只存已采/已跳过的点。"""
    filled = [p for p in points if p]
    valid = [p for p in filled if not p.get("skipped")
             and p.get("u") is not None]
    fit = None
    if H is not None:
        res = fit_points(valid)
        if res is not None:
            _, errs, mask, rms, mx = res
            fit = {"rms_mm": round(rms, 2), "max_mm": round(mx, 2),
                   "used": len(valid), "outliers": []}
    doc = {
        "version": 1,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "complete": complete,
        "arm_pose": arm_pose,       # "recenter" 或 [x, y]; 相机随臂,
                                    # 复现视角全靠这个参考位
        "resolution": resolution,
        "engine": engine,
        "grid": {"d_mm": GRID_D_MM, "l_mm": GRID_L_MM},
        "points": filled,
        "H": H.tolist() if H is not None else None,
        "fit": fit,
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    return doc


def load_partial(path):
    """--resume: 读 partial/full calib, 把已采点填回 points 数组。"""
    with open(path) as f:
        doc = json.load(f)
    points = [None] * N_POINTS
    for p in doc.get("points", []):
        i = int(p.get("i", -1))
        if 0 <= i < N_POINTS:
            p = dict(p)
            p.setdefault("skipped", False)
            p["from_resume"] = True
            points[i] = p
    return points, doc


# ---------- 交互状态机 ----------

def draw_ui(bgr, dets, points, cur, state, extra, sampling_progress):
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        color = COLORS.get(d["class"], (0, 255, 0))
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(bgr, "{} {:.2f}".format(d["class"], d["conf"]),
                    (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for p in points:
        if not p or p.get("u") is None:
            continue
        if p.get("skipped"):
            cv2.drawMarker(bgr, (int(p["u"]), int(p["v"])), (0, 0, 255),
                           cv2.MARKER_TILTED_CROSS, 14, 2)
        else:
            cv2.circle(bgr, (int(p["u"]), int(p["v"])), 6, (0, 255, 0), -1)
            cv2.putText(bgr, str(p["i"]), (int(p["u"]) + 8, int(p["v"]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    d_cur, l_cur = grid_point(cur)
    lines = [f"POINT {cur} (d={d_cur}mm, l={l_cur}mm) | {state}"]
    if sampling_progress is not None:
        n, tot = sampling_progress
        lines.append(f"采样中 {n}/{tot}")
    if extra:
        lines.append(extra)
    lines.append("Enter采样 r重采 u撤销 s跳过 c拟合 g<号>跳点 q保存退出")
    for k, line in enumerate(lines):
        cv2.putText(bgr, line, (10, 28 + k * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


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


def _run():
    import argparse
    ap = argparse.ArgumentParser(description="桌面网格 → homography 标定工具")
    ap.add_argument("--arm-pose", default=None,
                    help="机械臂参考位 \"x,y\" (mm, recenter 后 moveto 到该位; "
                         "相机随臂, 抬臂可换视角看到更近的网格)。"
                         "默认只 recenter。")
    ap.add_argument("--engine",
                    default=os.path.expanduser(
                        "~/Team21/vision/yolov8s_fp16.engine"),
                    help="TensorRT 引擎路径")
    ap.add_argument("--conf", type=float, default=CONF_THRESH,
                    help="置信度阈值 (默认 0.5)")
    ap.add_argument("--resolution", default="360p",
                    help="SDK 视频流分辨率: 360p/540p/720p")
    ap.add_argument("--out", default=os.path.expanduser(
                        "~/Team21/vision/calib.json"),
                    help="标定结果输出路径")
    ap.add_argument("--resume", default=None,
                    help="从 partial/full calib.json 继续 (跳过已采点)")
    ap.add_argument("--grid-d", default=None,
                    help="覆盖网格 d 排 (近→远, mm), 逗号分隔, "
                         "如 \"190,216,266\"; 最近排被机械臂挡出视野时用")
    ap.add_argument("--cli-input", action="store_true",
                    help="终端整行命令模式 (无显示器时用)")
    args = ap.parse_args()
    if args.grid_d:
        global GRID_D_MM
        GRID_D_MM = [float(x) for x in args.grid_d.split(",")]
    arm_pose = ("recenter" if args.arm_pose is None
                else [float(x) for x in args.arm_pose.split(",")])

    setup_display()

    # 1) 弹窗自检 (先弹窗口后连机器人, 与 vision_detector.py 一致)
    if not args.cli_input:
        win_name = "Calibrate (grid -> homography)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        black = np.zeros((360, 640, 3), np.uint8)
        cv2.putText(black, "waiting robot...", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow(win_name, black)
        cv2.waitKey(1)
        print(f"窗口 {win_name!r} 已创建。")
    else:
        win_name = None

    # 2) 加载 TensorRT 引擎
    print(f"加载引擎: {args.engine}")
    if not os.path.exists(args.engine):
        print(f"ERROR: 引擎不存在: {args.engine}")
        print("       先在板子上跑: build_engine.py bottle_tennisball_best.onnx "
              "yolov8s_fp16.engine")
        sys.exit(1)
    try:
        detector = TrtYOLO(args.engine, 640)
    except ImportError as e:
        print(f"ERROR: 导入 tensorrt/pycuda 失败: {e}")
        sys.exit(1)
    globals()["CONF_THRESH"] = args.conf
    print(f"引擎加载 OK, conf={args.conf}")

    # 3) 标定数据状态
    points = [None] * N_POINTS
    if args.resume:
        if os.path.exists(args.resume):
            points, _doc = load_partial(args.resume)
            print(f"--resume: 从 {args.resume} 恢复 {sum(p is not None for p in points)} 个点")
        else:
            print(f"WARN: --resume 文件不存在: {args.resume}, 从头开始。")
    history = []          # 采点/跳过的顺序 (u 撤销用)
    cur = 0               # 当前点 (第一个没数据的点)
    for i in range(N_POINTS):
        if points[i] is None:
            cur = i
            break
    state = "idle"        # idle / sampling / confirm / goto
    samples = []          # 采样中的 (u, v) 缓存
    sample_deadline = 0.0
    sample_med = None
    sample_std = 0.0
    goto_buf = ""
    last_status = ""

    def advance():
        nonlocal cur
        for i in range(cur + 1, N_POINTS):
            if points[i] is None:
                cur = i
                return
        cur = min(cur, 8)

    def finish_sample():
        nonlocal state, sample_med, sample_std, sample_deadline
        if not samples:
            return
        arr = np.array(samples)
        sample_med = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
        sample_std = float(max(np.std(arr[:, 0]), np.std(arr[:, 1])))
        print(f"采样完成: (u, v) = ({sample_med[0]:.1f}, {sample_med[1]:.1f}), "
              f"std={sample_std:.2f}px ({len(samples)} 帧)"
              + ("  <-- 球在动?" if sample_std > STD_WARN_PX else ""))
        state = "confirm"

    def cmd_sample():
        nonlocal state, samples, sample_deadline, sample_med
        d, l = grid_point(cur)
        print(f"\n把网球放到标记 {cur} (d={d}mm, l={l}mm), 采样 {SAMPLE_FRAMES} 帧 ...")
        state = "sampling"
        samples = []
        sample_deadline = time.time() + SAMPLE_TIMEOUT_S
        sample_med = None

    def cmd_accept():
        nonlocal state, cur
        if sample_med is None:
            return
        d, l = grid_point(cur)
        points[cur] = {"i": cur, "d_mm": d, "l_mm": l,
                       "u": round(sample_med[0], 2),
                       "v": round(sample_med[1], 2),
                       "n_samples": len(samples),
                       "std_px": round(sample_std, 2),
                       "skipped": False}
        history.append(cur)
        print(f"已记录点 {cur}: (u={sample_med[0]:.1f}, v={sample_med[1]:.1f})")
        state = "idle"
        advance()

    def cmd_skip():
        nonlocal state, cur
        d, l = grid_point(cur)
        points[cur] = {"i": cur, "d_mm": d, "l_mm": l, "u": None, "v": None,
                       "n_samples": 0, "std_px": None, "skipped": True}
        history.append(cur)
        print(f"点 {cur} 已标记跳过 (不参与拟合), 可用 g {cur} 回头补。")
        state = "idle"
        advance()

    def cmd_undo():
        nonlocal state, cur
        if not history:
            print("没有可撤销的操作。")
            return
        i = history.pop()
        points[i] = None
        cur = i
        print(f"已撤销点 {i}, 回到该点。")
        state = "idle"

    def cmd_fit():
        report_fit(points)

    def cmd_goto(i):
        nonlocal state, cur, goto_buf
        if 0 <= i < N_POINTS:
            cur = i
            d, l = grid_point(i)
            print(f"跳到点 {i} (d={d}mm, l={l}mm)。")
        else:
            print(f"ERROR: 点号范围 0-{N_POINTS - 1}, 收到 {i!r}。")
        goto_buf = ""
        state = "idle"

    def process_key(key):
        """窗口按键 → 命令。返回 False 表示退出。"""
        nonlocal state, goto_buf, sample_deadline
        if state == "goto":
            if key == KEY_ENTER or key == KEY_SPACE:
                cmd_goto(int(goto_buf) if goto_buf.isdigit() else -1)
            elif key == KEY_BACKSPACE:
                goto_buf = goto_buf[:-1]
            elif KEY_ESC == key or key == KEY_Q:
                goto_buf = ""
                state = "idle"
            elif 48 <= key <= 57:
                goto_buf += chr(key)
            return True
        if key in (KEY_ENTER, KEY_SPACE):
            if state == "confirm":
                cmd_accept()
            elif state == "idle":
                cmd_sample()
            # sampling 中的回车: 采样已超时则重试
            elif state == "sampling" and time.time() > sample_deadline:
                cmd_sample()
        elif key == KEY_R:
            if state != "sampling":
                cmd_sample()
        elif key == KEY_U:
            if state != "sampling":
                cmd_undo()
        elif key == KEY_S:
            if state == "confirm":
                cmd_skip()
            elif state == "idle":
                cmd_skip()
        elif key == KEY_C:
            if state != "sampling":
                cmd_fit()
        elif key == KEY_G:
            if state != "sampling":
                state = "goto"
                goto_buf = ""
        elif key == KEY_Q:
            return False
        return True

    # 4) 预检: 板子必须已连机器人热点
    ssid = current_wifi_ssid()
    print(f"current wifi: {ssid!r}")
    if not ssid.startswith("RMEP"):
        print("ERROR: 板子当前不在机器人热点上, 中止。")
        print("       先开机机器人, 然后执行: nmcli connection up RMEP-21bdc0")
        sys.exit(1)

    # 5) 连机器人 + 开视频流
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

    # 机械臂回到参考位。相机装在机械臂上 (臂动=视角动), 标定视角由臂位唯一
    # 决定: 默认 recenter, 可 --arm-pose "x,y" 抬臂换视角。
    try:
        ep.robotic_arm.recenter().wait_for_completed(timeout=8)
        time.sleep(2.0)
        print("机械臂已 recenter。")
    except Exception as e:
        print("Arm recenter warning:", e)
    if arm_pose == "recenter":
        print("机械臂参考位 = recenter (相机视角固定, 与抓取脚本一致)。")
    else:
        print(f"机械臂 moveto 参考位 {arm_pose} ...")
        try:
            ep.robotic_arm.moveto(x=arm_pose[0],
                                  y=arm_pose[1]).wait_for_completed(timeout=8)
        except Exception as e:
            print(f"arm.moveto 异常: {type(e).__name__}: {e}")
        pose_ok, actual_xy = wait_arm_pose(ep, arm_pose)
        if not pose_ok:
            print(f"WARN: 机械臂未到位 (实际 {actual_xy}, 目标 {arm_pose})。"
                  "若视野覆盖不了网格, q 退出后换 --arm-pose 重试。")

    print(f"开启视频流 ({args.resolution}) ...")
    ok = ep.camera.start_video_stream(display=False, resolution=args.resolution)
    if not ok:
        print("ERROR: 视频流开启失败 (检查手机 App 是否还连着机器人)")
        cleanup_with_timeout(ep)
        sys.exit(1)
    print("视频流已开启。按窗口上的按键开始标定 (q=保存退出)。")
    print(f"网格: d={GRID_D_MM}mm × l={GRID_L_MM}mm, 共 {N_POINTS} 点, 当前从点 {cur} 开始。")

    # 6) 主循环: 取帧 → 推理 → 平滑 → 采样/按键
    smoother = TemporalSmoother()
    last_heartbeat = time.time()
    running = True
    want_quit = False
    try:
        while running:
            try:
                img = ep.camera.read_cv2_image(timeout=2, strategy="newest")
            except Exception as e:
                now = time.time()
                if now - last_heartbeat >= 5:
                    print(f"read_cv2_image 异常: {type(e).__name__}: {e}")
                    last_heartbeat = now
                if not args.cli_input:
                    cv2.waitKey(1)
                continue
            if img is None:
                now = time.time()
                if now - last_heartbeat >= 5:
                    print("暂无帧 ...")
                    last_heartbeat = now
                if not args.cli_input:
                    cv2.waitKey(1)
                continue

            # SDK 解码出的是 RGB, 推理/显示用 BGR。
            # 必须 ascontiguousarray: 负步长非连续视图会让 cv2 5.x putText
            # 报 "Layout incompatible" (2026-09-13 踩过)。
            bgr = np.ascontiguousarray(img[:, :, ::-1])
            output, ratio, px, py = detector.infer(bgr)
            dets = TrtYOLO.postprocess(output, ratio, px, py,
                                       bgr.shape[1], bgr.shape[0])
            dets = smoother.update(dets)

            # 采样状态: 只收网球框底边中心, 取平滑后的检测
            if state == "sampling":
                det = pick_ball_det(dets)
                if det is not None:
                    samples.append(bottom_center(det))
                    if len(samples) >= SAMPLE_FRAMES:
                        finish_sample()
                    elif len(samples) == 1:
                        print("检测到网球, 采集中 ...")
                elif time.time() > sample_deadline:
                    print(f"ERROR: {SAMPLE_TIMEOUT_S:.0f}s 内没检测到网球。"
                          "检查球是否在标记上/有无遮挡; 回车重试, s 跳过。")
                    state = "idle"

            progress = ((len(samples), SAMPLE_FRAMES)
                        if state == "sampling" else None)
            extra = last_status
            if state == "goto":
                extra = f"跳点: g {goto_buf}_ (数字后回车)"
            if args.cli_input:
                if state == "sampling" and len(samples) and len(samples) % 4 == 0:
                    print(f"采样中 {len(samples)}/{SAMPLE_FRAMES} "
                          f"(u={samples[-1][0]:.1f}, v={samples[-1][1]:.1f})")
                if state == "idle":
                    d, l = grid_point(cur)
                    line = input(f"[点{cur} d={d} l={l}] 命令: ").strip().lower()
                    parts = line.split()
                    cmd = parts[0] if parts else ""
                    if cmd in ("", "enter", "e"):
                        cmd_sample()
                    elif cmd == "r":
                        cmd_sample()
                    elif cmd == "u":
                        cmd_undo()
                    elif cmd == "s":
                        cmd_skip()
                    elif cmd == "c":
                        cmd_fit()
                    elif cmd == "g" and len(parts) > 1 and parts[1].isdigit():
                        cmd_goto(int(parts[1]))
                    elif cmd == "q":
                        want_quit = True
                        running = False
                    else:
                        print("命令: 回车/enter 采样, r 重采, u 撤销, s 跳过, "
                              "c 拟合, g <号> 跳点, q 保存退出")
                elif state == "confirm":
                    line = input("[确认] 回车=记录, r=重采, s=跳过: ").strip().lower()
                    if line in ("", "enter", "e"):
                        cmd_accept()
                    elif line == "r":
                        cmd_sample()
                    elif line == "s":
                        cmd_skip()
                if state != "idle" and state != "confirm":
                    continue
            else:
                if state == "confirm" and sample_med is not None:
                    cv2.circle(bgr, (int(sample_med[0]), int(sample_med[1])),
                               8, (255, 255, 255), 2)
                draw_ui(bgr, dets, points, cur, state, extra, progress)
                cv2.imshow(win_name, bgr)
                key = cv2.waitKey(1) & 0xFF
                if key != 255 and not process_key(key):
                    want_quit = True
                    running = False
    except KeyboardInterrupt:
        want_quit = True

    # 7) 收尾: 拟合 + 保存
    valid = [p for p in points if p and not p.get("skipped")
             and p.get("u") is not None]
    res = fit_points(valid)
    H = res[0] if res is not None else None
    if want_quit or H is not None:
        if len(valid) >= 4 and H is not None:
            report_fit(points)
            save_calib(args.out, points, H, arm_pose, args.resolution,
                       args.engine, complete=True)
            print(f"标定已保存: {args.out}")
            print("vision_pick_place.py 定位时会自动把机械臂放回参考位 "
                  f"(arm_pose={arm_pose}), 相机视角与标定一致。")
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(args.out)
            partial = f"{base}_partial_{stamp}{ext}"
            save_calib(partial, points, None, arm_pose, args.resolution,
                       args.engine, complete=False)
            print(f"有效点不足 4 个, 只存半成品: {partial}")
            print(f"下次: --resume {partial} 继续。")

    if not args.cli_input:
        cv2.destroyAllWindows()
    cleanup_with_timeout(ep)
    print("calibrate done.")


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
