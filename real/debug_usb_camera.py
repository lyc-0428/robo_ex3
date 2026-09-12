#!/usr/bin/env python3
"""USB (UVC) 摄像头检测: 枚举 /dev/video*, 抓一帧看是不是真画面。

机械臂上的相机是 USB-C 接口, 如果它是标准 UVC 摄像头,
直接插到板子 USB 口就能被 cv2 打开, 完全绕开机器人视频子系统。

用法:
    cd ~/Team21/colcon_ws/src/robomaster_pick_place_sim
    python3 real/debug_usb_camera.py

日志: ~/Team21/logs/usb_camera_debug_<时间戳>.txt
抓帧: ~/Team21/logs/usb_camera_frame0.png
"""
import glob
import os
import subprocess
import sys
import time
import traceback

LOG_DIR = os.path.expanduser("~/Team21/logs")
LOG_LINES = []


class _Tee:
    def __init__(self, stream):
        self.stream = stream

    def write(self, text):
        self.stream.write(text)
        if text.strip():
            LOG_LINES.append(text.rstrip("\n"))
        return len(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def save_log():
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOG_DIR, f"usb_camera_debug_{stamp}.txt")
        with open(path, "w") as f:
            f.write("\n".join(LOG_LINES) + "\n")
        print(f"调试日志已保存: {path}")
    except Exception as e:
        print(f"保存调试日志失败: {e}")


def _run():
    import cv2
    import numpy as np

    # 从 SSH 里跑也要能弹窗到板子的显示器上
    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"
        try:
            for name in sorted(os.listdir("/tmp/.X11-unix")):
                if name.startswith("X"):
                    os.environ["DISPLAY"] = ":" + name[1:]
                    break
        except OSError:
            pass
    print("DISPLAY:", os.environ.get("DISPLAY"))
    print("cv2 version:", cv2.__version__)

    devices = sorted(glob.glob("/dev/video*"))
    print("/dev/video* 设备:", devices or "(无)")

    # v4l2 列表 (如果装了 v4l-utils)
    try:
        out = subprocess.run(["v4l2-ctl", "--list-devices"],
                             capture_output=True, text=True, timeout=5)
        print("v4l2-ctl --list-devices:\n" + out.stdout + out.stderr)
    except FileNotFoundError:
        print("(v4l2-ctl 未安装, 跳过)")
    except Exception as e:
        print("v4l2-ctl 失败:", e)

    # 逐个尝试 cv2.VideoCapture
    found = False
    for idx in range(len(devices)):
        print(f"--- 尝试 /dev/video{idx} ---")
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"video{idx} 打不开")
            cap.release()
            continue
        print(f"video{idx} 打开成功")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        time.sleep(1.0)  # 等自动曝光
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"video{idx} 抓帧失败")
            cap.release()
            continue
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_v, std_v = float(gray.mean()), float(gray.std())
        print(f"video{idx} 抓帧 OK: {w}x{h}, 亮度均值 {mean_v:.1f}, "
              f"标准差 {std_v:.1f} (std 大 = 有真实画面)")
        png_path = os.path.join(LOG_DIR, "usb_camera_frame0.png")
        cv2.imwrite(png_path, frame)
        print(f"帧已存 {png_path}")
        cv2.imshow(f"USB Camera /dev/video{idx}", frame)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        print(f"video{idx} 已在显示器上显示 3 秒")
        cap.release()
        found = True
        break

    if not found:
        print("WARN: 没有任何 USB 摄像头可用")
        print("      (先确认相机已插到板子 USB 口, 相机供电正常)")
    print("usb camera probe done")


def main():
    sys.stdout = _Tee(sys.stdout)
    sys.stderr = _Tee(sys.stderr)
    try:
        _run()
    finally:
        save_log()


if __name__ == "__main__":
    main()
