#!/usr/bin/env python3
"""直播链路诊断: 一次跑完定位 "视频流已开启但显示器没窗口" 的原因。

监测点:
  A. 40921 视频端口是否真的收到字节   (socket 打点)
  B. H264 解码器被调用次数 / 出帧数    (libmedia_codec 打点)
  C. read_cv2_image 能否取到帧
  D. cv2.imshow 能否弹窗               (SSH 实测: 可以, 与 XAUTHORITY 无关)
  E. 若 A/B/C 全空 -> 自动退回明文 SDK 探针: command; + stream on; + 裸收 40921

用法 (板子已连机器人热点, 手机 RoboMaster App 必须先断开机器人!):
    cd ~/Team21/colcon_ws/src/robomaster_pick_place_sim
    python3 real/debug_camera_live.py

全程约 60 秒, 自动退出。日志: ~/Team21/logs/camera_live_debug_<时间戳>.txt
"""
import os
import socket
import sys
import threading
import time
import traceback

LOG_DIR = os.path.expanduser("~/Team21/logs")
LOG_LINES = []

# 打点计数器 (模块级, 各 hook 往里写)
VIDEO = {"connect_40921": 0, "bytes": 0, "recv_calls": 0, "first": b""}
DEC = {"calls": 0, "bytes_in": 0, "frames_out": 0, "errors": []}


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
        path = os.path.join(LOG_DIR, f"camera_live_debug_{stamp}.txt")
        with open(path, "w") as f:
            f.write("\n".join(LOG_LINES) + "\n")
        print(f"调试日志已保存: {path}")
    except Exception as e:
        print(f"保存调试日志失败: {e}")


# ---------- 打点 hook ----------
# 注意: 只能给 socket.socket 的"方法"打补丁, 绝不能替换 socket.socket 类本身 ——
# ssl.py 里有 class SSLSocket(socket.socket), 类被换成函数会直接 TypeError 崩掉。
# (2026-09-13 踩过这个坑, 见 git 历史)

_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex
_orig_recv = socket.socket.recv


def _spy_connect(self, address):
    try:
        if address and len(address) >= 2 and address[1] == 40921:
            VIDEO["connect_40921"] += 1
    except Exception:
        pass
    return _orig_connect(self, address)


def _spy_connect_ex(self, address):
    try:
        if address and len(address) >= 2 and address[1] == 40921:
            VIDEO["connect_40921"] += 1
    except Exception:
        pass
    return _orig_connect_ex(self, address)


def _spy_recv(self, *a, **k):
    data = _orig_recv(self, *a, **k)
    if data:
        VIDEO["recv_calls"] += 1
        VIDEO["bytes"] += len(data)
        if not VIDEO["first"]:
            VIDEO["first"] = bytes(data[:64])
    return data


def install_socket_spy():
    socket.socket.connect = _spy_connect
    socket.socket.connect_ex = _spy_connect_ex
    socket.socket.recv = _spy_recv


def install_decoder_spy():
    import libmedia_codec
    orig = libmedia_codec.H264Decoder.decode

    def counted_decode(self, data):
        DEC["calls"] += 1
        DEC["bytes_in"] += len(data)
        try:
            res = orig(self, data)
        except Exception as e:
            DEC["errors"].append(f"{type(e).__name__}: {e}")
            raise
        DEC["frames_out"] += len(res)
        return res

    libmedia_codec.H264Decoder.decode = counted_decode


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


def setup_display():
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


def h264_decode_to_png(data, tag):
    """PyAV 解码一段 H264, 第一帧存 PNG (pts 递增技巧, 见 libmedia_codec_real.py)。"""
    import av
    nal = data.find(b"\x00\x00\x00\x01")
    if nal < 0:
        nal = data.find(b"\x00\x00\x01")
    if nal < 0:
        print(f"{tag}: 数据里没有 NAL 起始码, 前 64 字节: {data[:64].hex()}")
        return 0
    codec = av.CodecContext.create("h264", "r")
    codec.open()
    pts = 0
    frames = 0
    for packet in codec.parse(data[nal:]):
        packet.pts = pts
        packet.dts = pts
        pts += 1
        for frame in codec.decode(packet):
            if frames == 0:
                h, w = frame.height, frame.width
                png_path = os.path.join(LOG_DIR, f"live_frame0_{w}x{h}.png")
                frame.to_image().save(png_path)
                print(f"{tag}: 解码出图 {w}x{h}, 已存 {png_path}")
            frames += 1
    print(f"{tag}: 共解码 {frames} 帧")
    return frames


def text_sdk_fallback():
    """明文 SDK 探针: 绕过 robomaster 库, 裸 socket 走一遍。"""
    print("--- 回退: 明文 SDK 探针 ---")
    try:
        ctrl = socket.create_connection(("192.168.2.1", 40923), timeout=5)
    except OSError as e:
        print(f"40923 连接失败: {e}")
        return
    ctrl.settimeout(1.5)

    def send(cmd):
        msg = cmd if cmd.endswith(";") else cmd + ";"
        print(f">>> {msg}")
        try:
            ctrl.sendall(msg.encode())
        except OSError as e:
            print(f"send 失败: {e}")
            return
        try:
            reply = ctrl.recv(4096).decode("utf-8", errors="replace")
            print(f"<<< {reply!r}")
        except socket.timeout:
            print("<<< (超时无回复)")

    send("command;")
    send("stream on;")
    time.sleep(1.0)
    send("stream on;")

    video = None
    try:
        video = socket.create_connection(("192.168.2.1", 40921), timeout=3)
    except OSError as e:
        print(f"40921 连接失败: {e}")
    if video is not None:
        print("40921 已连接, 收 8 秒...")
        video.settimeout(1.0)
        data = b""
        end = time.time() + 8.0
        while time.time() < end:
            try:
                chunk = video.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                print("(40921 被对端关闭)")
                break
            data += chunk
            if len(data) > 4 * 1024 * 1024:
                break
        video.close()
        print(f"裸收 {len(data)} 字节, 前 64 字节: "
              f"{data[:64].hex() if data else '(空)'}")
        if data:
            h264_decode_to_png(data, "text_sdk")
    send("stream off;")
    send("quit;")
    ctrl.close()


def _run():
    start = time.time()
    setup_display()

    # 先装打点, 再建 SDK 连接 (socket hook 必须在 SDK 建连接前生效)
    install_socket_spy()
    install_decoder_spy()

    from robomaster import robot
    import cv2
    import numpy as np

    # WiFi 预检 (只提醒, 不中止)
    try:
        import subprocess
        out = subprocess.run(
            ["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        ssid = next((l.split(":", 1)[1] for l in out.splitlines()
                     if l.startswith("yes:")), "")
        print(f"current wifi: {ssid!r}")
        if not ssid.startswith("RMEP"):
            print("WARN: 当前不在机器人热点, 连不上机器人属正常")
    except Exception as e:
        print("WiFi 检查失败:", e)

    ep = robot.Robot()
    try:
        initialized = ep.initialize(conn_type="ap")
    except Exception as e:
        print(f"ERROR: SDK 初始化异常 (机器人没开机 / 热点没连): "
              f"{type(e).__name__}: {e}")
        sys.exit(1)
    if not initialized:
        print("ERROR: SDK 初始化失败 (机器人没开机 / 热点没连)")
        sys.exit(1)

    ok = ep.camera.start_video_stream(display=False, resolution="360p")
    print(f"start_video_stream(display=False) -> {ok}")
    if not ok:
        print("ERROR: 流开启失败 (检查手机 App 是否还连着机器人)")
        ep.close()
        sys.exit(1)

    # C. 轮询取帧 30 秒
    print("轮询 read_cv2_image 30 秒 ...")
    end = time.time() + 30
    frames = 0
    first_img = None
    while time.time() < end:
        try:
            img = ep.camera.read_cv2_image(timeout=5, strategy="newest")
        except Exception as e:
            print(f"read_cv2_image 异常: {type(e).__name__}: {e}")
            break
        if img is None:
            print(f"[t={time.time()-start:.0f}s] 没取到帧 "
                  f"(40921 已收 {VIDEO['bytes']} 字节, "
                  f"解码调用 {DEC['calls']} 次, 出帧 {DEC['frames_out']})")
            continue
        frames += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"[t={time.time()-start:.0f}s] 取到第 {frames} 帧: {img.shape}, "
              f"亮度均值 {gray.mean():.1f} std {gray.std():.1f}")
        if first_img is None:
            first_img = img
            cv2.imwrite(os.path.join(LOG_DIR, "live_frame0.png"), img)
            print("首帧已存 ~/Team21/logs/live_frame0.png")

    cleanup_with_timeout(ep)

    # D. imshow 自测 (主线程, 异常会直接打出来)
    disp_img = first_img if first_img is not None \
        else np.zeros((360, 640, 3), np.uint8)
    try:
        cv2.imshow("DIAG_WINDOW", disp_img)
        cv2.waitKey(2500)
        cv2.destroyAllWindows()
        print("imshow 自测: OK (显示器上应出现过 DIAG_WINDOW 窗口)")
    except Exception as e:
        print(f"imshow 自测: FAILED {type(e).__name__}: {e}")

    # E. 回退探针
    if VIDEO["bytes"] == 0 and frames == 0:
        text_sdk_fallback()
    else:
        print("40921 已有字节流入, 跳过明文 SDK 回退。")

    print("---- VERDICT ----")
    print(f"40921 连接次数: {VIDEO['connect_40921']}, "
          f"收到字节: {VIDEO['bytes']}, recv 调用: {VIDEO['recv_calls']}")
    if VIDEO["first"]:
        print(f"40921 首批字节: {VIDEO['first'].hex()}")
    print(f"解码器: 调用 {DEC['calls']} 次, 入 {DEC['bytes_in']} 字节, "
          f"出 {DEC['frames_out']} 帧, 异常 {len(DEC['errors'])}")
    for e in DEC["errors"][:5]:
        print("  解码异常:", e)
    print(f"read_cv2_image 取到帧数: {frames}")
    print("debug camera live done")


def main():
    sys.stdout = _Tee(sys.stdout)
    sys.stderr = _Tee(sys.stderr)
    try:
        _run()
    except BaseException:
        # 先把堆栈打进日志再抛出去, 否则 traceback 打印发生在 save_log 之后,
        # 日志文件里会丢掉真正的报错内容 (2026-09-13 踩过)。
        traceback.print_exc()
        save_log()
        raise
    save_log()


if __name__ == "__main__":
    main()
