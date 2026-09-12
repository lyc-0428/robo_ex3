#!/usr/bin/env python3
"""机器人云台摄像头实时显示 (板子接显示器, 弹窗看画面)。

依赖:
  - 板子上 pip3 install --user av  (PyAV, H264 解码)
  - libmedia_codec.py 桩已换成真解码器 (见 real/README.md)

用法 (板子上, 已连机器人热点, 已接显示器):
    cd ~/Team21/colcon_ws/src/robomaster_pick_place_sim
    python3 real/camera_viewer.py           # 默认 720p
    python3 real/camera_viewer.py 360p      # 卡顿就用低分辨率
"""
import os
import subprocess
import sys
import time

from robomaster import robot


def current_wifi_ssid():
    """返回板子当前连接的 WiFi SSID; 读不到返回 ""。"""
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


def main():
    # 从 SSH 里跑也要能弹窗到板子的显示器上:
    # 自动探测 X 显示号 (GNOME 会话可能开在 :0 或 :1)
    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"
        try:
            for name in sorted(os.listdir("/tmp/.X11-unix")):
                if name.startswith("X"):
                    os.environ["DISPLAY"] = ":" + name[1:]
                    break
        except OSError:
            pass

    resolution = sys.argv[1] if len(sys.argv) > 1 else "720p"
    if resolution not in ("360p", "540p", "720p"):
        print(f"ERROR: 不支持的分辨率 {resolution!r}, 可选 360p/540p/720p")
        sys.exit(1)

    # 预检: 板子必须已连机器人热点 (同主脚本)
    ssid = current_wifi_ssid()
    print(f"current wifi: {ssid!r}")
    if not ssid.startswith("RMEP"):
        print("ERROR: 板子当前不在机器人热点上, 中止。")
        print("       先开机机器人, 然后执行: nmcli connection up RMEP-21bbc5")
        sys.exit(1)

    ep = robot.Robot()
    try:
        initialized = ep.initialize(conn_type="ap")
    except Exception as e:
        print(f"ERROR: SDK 初始化异常, 中止: {e}")
        sys.exit(1)

    if not initialized:
        print("ERROR: 连不上机器人, 检查机器人是否开机, 中止")
        sys.exit(1)

    print(f"开启视频流 ({resolution}) ...")
    ok = ep.camera.start_video_stream(display=True, resolution=resolution)
    if not ok:
        print("ERROR: 视频流开启失败 (检查机器人摄像头是否被 APP 占用)")
        ep.close()
        sys.exit(1)

    print("视频流已开启, 显示器上应弹出 RoboMaster LiveView 窗口。")
    print("按 Ctrl+C 退出。")

    try:
        # SDK 的解码/显示线程在后台跑, 主线程保持机器人连接即可
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user.")

    try:
        ep.camera.stop_video_stream()
    except Exception as e:
        print("stop_video_stream warning:", e)

    ep.close()
    print("camera viewer done.")


if __name__ == "__main__":
    main()
