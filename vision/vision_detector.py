#!/usr/bin/env python3
"""实验三视觉检测节点: 机器人云台相机 + TensorRT YOLOv8s (bottle/tennis_ball)。

取帧: robomaster SDK (read_cv2_image, 与 camera_viewer.py 同一条已验证链路)
推理: TensorRT 引擎 (队友的 bottle_tennisball_best.pt 导出 ONNX 后板子上构建,
      复用 exp1 的 build_engine.py + TrtYOLO/后处理/时间平滑)
输出: cv2 实时窗口 (检测框+置信度+FPS) + ROS2 话题 detections (JSON String)
      {"objects": [{"class": "bottle", "conf": 0.92, "box": [x1,y1,x2,y2]}], "fps": 28.0}

用法 (板子已连机器人热点, 手机 App 断开机器人):
    cd ~/Team21/colcon_ws/src/robomaster_pick_place_sim
    ~/Team21/Team21/bin/python3 vision/vision_detector.py
    ~/Team21/Team21/bin/python3 vision/vision_detector.py --conf 0.35 --resolution 720p
    默认引擎 FP16 (~30fps); 换 FP32: --engine ~/Team21/vision/yolov8s_fp32.engine

不用 source ROS: 脚本自己把 /opt/ros/humble 加进 sys.path/LD_LIBRARY_PATH
(rclpy 的原生库需要 LD_LIBRARY_PATH)。numpy/cv2 在加 ROS 路径之前先导入,
避免 ROS 的 numpy 1.x 顶掉 cv2 5.x 需要的 numpy 2.x (2026-09-13 踩过)。

日志: ~/Team21/logs/vision_detector_<时间戳>.txt
"""
import os
import subprocess
import sys
import threading
import time
import traceback

# ---------- sys.path 手术: 必须在 import numpy/cv2 之前 ----------
# 用户 shell 可能 source 过 ROS (activate_team21.sh): ROS site-packages 里的
# numpy 1.x 排在 venv 前面, 会把 cv2 5.x 依赖的 numpy 2.x 顶掉 → cv2 一 import
# 就崩 (2026-09-13 踩过)。把 ROS 路径挪到 sys.path 末尾, 保证下面 import 到
# venv 自己的 numpy; rclpy 在末尾仍能找到, 只是排序靠后而已。
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
        _LIVE_PATH = os.path.join(LOG_DIR, f"vision_detector_{stamp}.txt")
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
        path = os.path.join(LOG_DIR, f"vision_detector_{stamp}.txt")
        with open(path, "w") as f:
            f.write("\n".join(LOG_LINES) + "\n")
        print(f"日志已保存: {path}")
    except Exception as e:
        print(f"保存日志失败: {e}")


def setup_display():
    """SSH 里跑也要能弹窗到板子的显示器上。

    注: 与 camera_viewer 不同, 这里不能 pop LD_LIBRARY_PATH ——
    rclpy 的原生库 (rmw/rcl) 必须靠它找到 /opt/ros/humble/lib。
    cv2 Qt 走自己 bundle 的绝对路径, 不受影响 (2026-09-13 已验证)。
    """
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
    """让 rclpy 可用: 补 sys.path + LD_LIBRARY_PATH (幂等)。

    ROS 路径的排序问题已在模块顶部处理过 (挪到末尾), 这里只负责
    "没 source 过 ROS" 的情况: 把路径补进去。numpy/cv2 已导入并缓存,
    不受影响。
    """
    ros_site = [
        "/opt/ros/humble/lib/python3.10/site-packages",
        "/opt/ros/humble/local/lib/python3.10/dist-packages",
    ]
    for p in ros_site:
        if p not in sys.path:
            sys.path.append(p)      # 手动补上
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


# ---------- TensorRT 推理 (移植自 exp1 detector_pkg, TRT 10 API) ----------

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


def draw_frame(bgr, dets, fps):
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        color = COLORS.get(d["class"], (0, 255, 0))
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        label = "{} {:.2f}".format(d["class"], d["conf"])
        cv2.putText(bgr, label, (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(bgr, "FPS: {:.1f}".format(fps), (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


def _run():
    import argparse
    ap = argparse.ArgumentParser(description="SDK 相机 + TensorRT 检测节点")
    ap.add_argument("--engine",
                    default=os.path.expanduser(
                        "~/Team21/vision/yolov8s_fp16.engine"),
                    help="TensorRT 引擎路径 (2026-09-13 实机 A/B 后定 FP16 "
                         "为默认; 要更稳的置信度可换 yolov8s_fp32.engine)")
    ap.add_argument("--conf", type=float, default=CONF_THRESH,
                    help="置信度阈值 (默认 0.5, 效果不稳可降到 0.35)")
    ap.add_argument("--resolution", default="360p",
                    help="SDK 视频流分辨率: 360p/540p/720p")
    ap.add_argument("--no-display", action="store_true",
                    help="不弹窗口 (无显示器时)")
    args = ap.parse_args()

    setup_display()

    # 1) 弹窗自检 (先弹窗口后连机器人, 与 camera_viewer.py 一致)
    #    numpy/cv2 已在模块顶部导入 (sys.path 手术之后, 见文件头)
    win_name = "Vision Detector (bottle/tennis_ball)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    black = np.zeros((360, 640, 3), np.uint8)
    cv2.putText(black, "waiting robot...", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imshow(win_name, black)
    cv2.waitKey(1)
    print(f"窗口 {win_name!r} 已创建 —— 显示器上应该已经能看到它。")

    # 3) 加载 TensorRT 引擎 (tensorrt/pycuda 在系统 site-packages)
    print(f"加载引擎: {args.engine}")
    if not os.path.exists(args.engine):
        print(f"ERROR: 引擎不存在: {args.engine}")
        print("       先在板子上跑: build_engine.py bottle_tennisball_best.onnx "
              "yolov8s_fp32.engine --fp32")
        sys.exit(1)
    try:
        detector = TrtYOLO(args.engine, 640)
    except ImportError as e:
        print(f"ERROR: 导入 tensorrt/pycuda 失败: {e}")
        print("       team venv 需要能看到系统 site-packages (板子上检查 "
              "pyvenv.cfg 的 include-system-site-packages)。")
        sys.exit(1)
    globals()["CONF_THRESH"] = args.conf
    print(f"引擎加载 OK, conf={args.conf}")

    # 4) ROS2 (脚本自己补路径, 不用 source)
    prepare_ros_paths()
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
    except ImportError as e:
        print(f"ERROR: rclpy 导入失败: {e} (检查 /opt/ros/humble 是否安装)")
        sys.exit(1)
    rclpy.init()
    node = Node("vision_detector")
    pub = node.create_publisher(String, "detections", 10)
    print("ROS2 节点 vision_detector 已启动, 发布话题 detections")

    def publish(msg_dict):
        s = String()
        import json
        s.data = json.dumps(msg_dict)
        pub.publish(s)

    # 5) 预检: 板子必须已连机器人热点
    ssid = current_wifi_ssid()
    print(f"current wifi: {ssid!r}")
    if not ssid.startswith("RMEP"):
        print("ERROR: 板子当前不在机器人热点上, 中止。")
        print("       先开机机器人, 然后执行: nmcli connection up RMEP-21bdc0")
        sys.exit(1)

    # 6) 连机器人 + 开视频流 (复用 camera_viewer 的成熟流程)
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
    print("视频流已开启, 等帧中 ...")

    # 7) 主循环: 取帧 → 推理 → 平滑 → 画框 → 发布
    from collections import deque
    fps_q = deque(maxlen=30)
    smoother = TemporalSmoother()
    frames = 0
    last_frames = 0
    first_saved = False
    t0 = time.time()
    last_heartbeat = t0
    running = True
    try:
        while running and rclpy.ok():
            try:
                img = ep.camera.read_cv2_image(timeout=2, strategy="newest")
            except Exception as e:
                now = time.time()
                if now - last_heartbeat >= 5:
                    print(f"[t={now-t0:.0f}s] read_cv2_image 异常: "
                          f"{type(e).__name__}: {e}")
                    last_heartbeat = now
                cv2.waitKey(1)
                continue
            if img is None:
                now = time.time()
                if now - last_heartbeat >= 5:
                    print(f"[t={now-t0:.0f}s] 暂无帧 ...")
                    last_heartbeat = now
                cv2.waitKey(1)
                continue

            # SDK 解码出的是 RGB, 推理/显示用 BGR。
            # 必须 ascontiguousarray: [:, :, ::-1] 是负步长非连续视图,
            # cv2 5.x 的 putText 会报 "Layout incompatible" (2026-09-13 踩过)。
            bgr = np.ascontiguousarray(img[:, :, ::-1])
            t_frame = time.time()
            t_infer = time.time()
            output, ratio, px, py = detector.infer(bgr)
            dets = TrtYOLO.postprocess(output, ratio, px, py,
                                       bgr.shape[1], bgr.shape[0])
            dets = smoother.update(dets)
            infer_ms = (time.time() - t_infer) * 1000

            # 整帧耗时 (取帧+推理+画框+发布), 毫秒 → fps 要乘 1000
            # (之前 1.0/ms 忘了换算, FPS 永远显示 0.00, 2026-09-13 踩过)
            loop_ms = (time.time() - t_frame) * 1000
            fps_q.append(1000.0 / max(1e-9, loop_ms))
            avg_fps = sum(fps_q) / len(fps_q)
            publish({"objects": dets, "fps": round(avg_fps, 1)})

            frames += 1
            now = time.time()
            if now - t0 >= 2.0:
                names = ",".join(f"{d['class']}:{d['conf']}" for d in dets)
                # 真实帧率 = 本窗口帧数 / 窗口时长 (之前拿累计帧数除, 数字
                # 一直在涨不是 fps, 2026-09-13 踩过)
                print(f"fps={(frames-last_frames)/(now-t0):.1f} "
                      f"推理={infer_ms:.0f}ms 检测={names or '(无)'}")
                t0 = now
                last_frames = frames

            if not first_saved and dets:
                first_saved = True
                png = os.path.join(LOG_DIR, "vision_frame0.png")
                cv2.imwrite(png, bgr)
                print(f"首帧含检测 {png}")
            elif not first_saved and frames >= 30:
                first_saved = True
                png = os.path.join(LOG_DIR, "vision_frame0.png")
                cv2.imwrite(png, bgr)
                print(f"首帧 (30 帧内无检测目标) {png}")

            if not args.no_display:
                draw_frame(bgr, dets, avg_fps)
                cv2.imshow(win_name, bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("按 q 退出。")
                    running = False
    except KeyboardInterrupt:
        print("Interrupted by user.")

    cv2.destroyAllWindows()
    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass
    cleanup_with_timeout(ep)
    print("vision detector done.")


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
