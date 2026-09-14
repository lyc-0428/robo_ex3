#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RoboMaster EP 实时摄像头查看器（无 WiFi 名称检查版）

用法：
    python3 real/camera_viewer.py
    python3 real/camera_viewer.py 360p
    python3 real/camera_viewer.py 720p

说明：
- 不再调用 nmcli，也不检查 Current WiFi / RMEP 名称。
- 直接通过 RoboMaster SDK:
      ep.initialize(conn_type="ap")
  判断是否真正连接到机器人。
- Jetson 显示器实时显示 RoboMaster 相机画面。
- q / ESC / Ctrl+C 退出。
"""

import os
import sys
import threading
import time

# 尽量让 SSH 终端启动时也能把 OpenCV 窗口显示到 Jetson 本地显示器
if not os.environ.get("DISPLAY"):
    os.environ["DISPLAY"] = ":0"
    try:
        xs = sorted(os.listdir("/tmp/.X11-unix"))
        for name in xs:
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
from robomaster import robot


WINDOW_NAME = "RoboMaster LiveView"


def cleanup_with_timeout(ep, camera_started):
    """避免 SDK 的 stop/close 偶尔卡住导致程序退不出去。"""

    def _cleanup():
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

    if t.is_alive():
        print("WARN: SDK 清理超过 5 秒，直接结束程序。")


def main():
    resolution = "720p"

    if len(sys.argv) >= 2:
        resolution = sys.argv[1].strip().lower()

    if resolution not in ("360p", "540p", "720p"):
        print("分辨率只支持: 360p / 540p / 720p")
        return 2

    print("==========================================")
    print("RoboMaster Camera Viewer")
    print("==========================================")
    print("WiFi SSID check: DISABLED")
    print("实际连接由 RoboMaster SDK initialize() 判断")
    print("resolution:", resolution)
    print(
        "DISPLAY:",
        os.environ.get("DISPLAY"),
        "XAUTHORITY:",
        os.environ.get("XAUTHORITY", "(无)"),
    )
    print()
    print("注意：手机 RoboMaster App 请断开机器人。")
    print("机器人摄像头同一时间只能被一路程序占用。")
    print("q / ESC / Ctrl+C 退出")
    print("==========================================")

    ep = robot.Robot()
    initialized = False
    camera_started = False

    try:
        print("\n[1] 正在连接 RoboMaster ...")

        result = ep.initialize(conn_type="ap")

        # 有的 SDK 版本 initialize() 成功后返回 None，
        # 所以不能只用 if not result 判断失败。
        initialized = True

        print("[OK] RoboMaster SDK 初始化完成")
        if result is not None:
            print("initialize result:", result)

        print("\n[2] 正在启动视频流 ...")

        start_result = ep.camera.start_video_stream(
            display=False,
            resolution=resolution,
        )
        camera_started = True

        print("[OK] 视频流启动命令已发送")
        if start_result is not None:
            print("start_video_stream result:", start_result)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        print("\n[3] 正在等待相机画面 ...")

        first_frame = True
        empty_count = 0

        while True:
            try:
                frame = ep.camera.read_cv2_image(
                    strategy="newest",
                    timeout=3,
                )
            except TypeError:
                # 兼容某些 SDK 版本没有 timeout 参数
                frame = ep.camera.read_cv2_image(strategy="newest")

            if frame is None:
                empty_count += 1

                if empty_count == 1 or empty_count % 10 == 0:
                    print(f"暂未收到画面 ({empty_count}) ...")

                # 一直收不到帧时也让窗口事件有机会处理
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    break

                continue

            empty_count = 0

            if first_frame:
                h, w = frame.shape[:2]
                print(f"[OK] 收到第一帧: {w}x{h}")
                print(f"[OK] Jetson 显示窗口: {WINDOW_NAME}")
                first_frame = False

            # RoboMaster SDK 的 read_cv2_image 已返回可供 OpenCV 显示的图像
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，退出。")

    except Exception as e:
        print("\nERROR:", repr(e))
        print()
        print("如果这里是视频解码相关错误，请检查：")
        print("  python3 -c \"import av; print(av.__version__)\"")
        print("以及 ~/.local/lib/python3.10/site-packages/libmedia_codec.py")
        return 1

    finally:
        cv2.destroyAllWindows()

        if initialized:
            cleanup_with_timeout(ep, camera_started)

    print("Camera viewer finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
