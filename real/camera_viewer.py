#!/usr/bin/env python3
"""机器人云台摄像头实时显示 (板子接显示器, 弹窗看画面)。

v2: 不依赖 SDK 的后台显示线程 (display=True 换板后不弹窗, 且 stop/close
    会卡死), 改为主线程自己 read_cv2_image + imshow, 帧率实时打印, 日志落盘。

依赖 (Team21 venv 已装好):
  - av 17.1.0 + libmedia_codec.py 真解码器 (PyAV pts 递增)
  - opencv-python 5.x

用法 (板子已连机器人热点, 已接显示器, 手机 App 必须断开机器人):
    cd ~/Team21/colcon_ws/src/robomaster_pick_place_sim
    ~/Team21/Team21/bin/python3 real/camera_viewer.py          # 默认 720p
    ~/Team21/Team21/bin/python3 real/camera_viewer.py 360p     # 卡顿就用低分辨率

日志: ~/Team21/logs/camera_viewer_<时间戳>.txt
"""
import os
import subprocess
import sys
import threading
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
        path = os.path.join(LOG_DIR, f"camera_viewer_{stamp}.txt")
        with open(path, "w") as f:
            f.write("\n".join(LOG_LINES) + "\n")
        print(f"日志已保存: {path}")
    except Exception as e:
        print(f"保存日志失败: {e}")


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


def setup_display():
    """SSH 里跑也要能弹窗到板子的显示器上。"""
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


def _run():
    setup_display()

    resolution = sys.argv[1] if len(sys.argv) > 1 else "720p"
    if resolution not in ("360p", "540p", "720p"):
        print(f"ERROR: 不支持的分辨率 {resolution!r}, 可选 360p/540p/720p")
        sys.exit(1)

    # 预检: 板子必须已连机器人热点
    ssid = current_wifi_ssid()
    print(f"current wifi: {ssid!r}")
    if not ssid.startswith("RMEP"):
        print("ERROR: 板子当前不在机器人热点上, 中止。")
        print("       先开机机器人, 然后执行: nmcli connection up RMEP-21bbc5")
        sys.exit(1)

    from robomaster import robot
    import cv2

    ep = robot.Robot()
    try:
        initialized = ep.initialize(conn_type="ap")
    except Exception as e:
        print(f"ERROR: SDK 初始化异常, 中止: {type(e).__name__}: {e}")
        sys.exit(1)
    if not initialized:
        print("ERROR: 连不上机器人, 检查机器人是否开机, 中止")
        sys.exit(1)

    print(f"开启视频流 ({resolution}) ...")
    ok = ep.camera.start_video_stream(display=False, resolution=resolution)
    if not ok:
        print("ERROR: 视频流开启失败 (检查手机 App 是否还连着机器人)")
        cleanup_with_timeout(ep)
        sys.exit(1)

    print("视频流已开启。")
    win_name = "RoboMaster LiveView"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    print(f"窗口 {win_name!r} 已创建, 等帧中 ...")

    frames = 0
    first_saved = False
    t0 = time.time()
    last_heartbeat = t0
    try:
        while True:
            try:
                img = ep.camera.read_cv2_image(timeout=2, strategy="newest")
            except Exception as e:
                print(f"read_cv2_image 异常: {type(e).__name__}: {e}")
                continue
            if img is None:
                # 5 秒没帧才打一次心跳, 避免刷屏
                now = time.time()
                if now - last_heartbeat >= 5:
                    print(f"[t={now-t0:.0f}s] 暂无帧 ...")
                    last_heartbeat = now
                cv2.waitKey(1)
                continue

            frames += 1
            cv2.imshow(win_name, img)
            cv2.waitKey(1)

            if not first_saved:
                first_saved = True
                png = os.path.join(LOG_DIR, "viewer_frame0.png")
                cv2.imwrite(png, img)
                print(f"首帧 {img.shape[1]}x{img.shape[0]} 已存 {png}")

            now = time.time()
            if now - t0 >= 2.0:
                fps = frames / (now - t0)
                print(f"fps={fps:.1f}  累计 {frames} 帧  ({img.shape[1]}x{img.shape[0]})")
                t0 = now
    except KeyboardInterrupt:
        print("Interrupted by user.")

    cv2.destroyAllWindows()
    cleanup_with_timeout(ep)
    print("camera viewer done.")


def main():
    sys.stdout = _Tee(sys.stdout)
    sys.stderr = _Tee(sys.stderr)
    try:
        _run()
    except BaseException:
        # 先把堆栈打进日志再抛出去, 否则 traceback 打印发生在 save_log 之后
        traceback.print_exc()
        save_log()
        raise
    save_log()


if __name__ == "__main__":
    main()
