#!/usr/bin/env python3
"""明文 SDK (text SDK) 探针 v2: 一轮测完机器人状态 + 相机/视频/音频子系统。

上一轮结果: command; -> ok;  stream on; -> fail;  (机器人明确拒绝开视频流)。
v2 增加: 机器人模式/电量查询、相机曝光命令、stream on 重试、audio on,
用来判断是整个音视频编码管线挂了, 还是只有视频。

用法 (板子已连机器人热点; 建议先重启机器人再跑):
    cd ~/Team21/colcon_ws/src/robomaster_pick_place_sim
    python3 real/debug_text_sdk.py

日志: ~/Team21/logs/text_sdk_debug_<时间戳>.txt
出图: ~/Team21/logs/text_sdk_frame0_<宽>x<高>.png
"""
import os
import socket
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
        path = os.path.join(LOG_DIR, f"text_sdk_debug_{stamp}.txt")
        with open(path, "w") as f:
            f.write("\n".join(LOG_LINES) + "\n")
        print(f"调试日志已保存: {path}")
    except Exception as e:
        print(f"保存调试日志失败: {e}")


def send_cmd(sock, cmd, wait=1.5):
    """发一条明文命令 (自动补分号), 收并打印回复。"""
    msg = cmd if cmd.endswith(";") else cmd + ";"
    print(f">>> send: {msg}")
    try:
        sock.sendall(msg.encode("utf-8"))
    except OSError as e:
        print(f"send 失败: {type(e).__name__} {e}")
        return ""
    replies = []
    sock.settimeout(wait)
    deadline = time.time() + wait
    while time.time() < deadline:
        try:
            buf = sock.recv(4096)
        except socket.timeout:
            break
        if not buf:
            print("(控制连接被对端关闭)")
            break
        text = buf.decode("utf-8", errors="replace")
        replies.append(text)
        print(f"<<< recv: {text!r}")
        time.sleep(0.3)
        sock.settimeout(0.3)
        deadline = min(deadline, time.time() + 0.3)
    return "".join(replies)


def try_decode(data, tag):
    """H264 解码尝试, 出图存 PNG (复用 PyAV pts 递增技巧)。"""
    import av
    nal = data.find(b"\x00\x00\x00\x01")
    if nal < 0:
        nal = data.find(b"\x00\x00\x01")
    if nal < 0:
        print(f"{tag}: 数据里没有 H264 NAL 起始码, 前 64 字节: {data[:64].hex()}")
        return False
    codec = av.CodecContext.create("h264", "r")
    codec.open()
    pts = 0
    frames = 0
    try:
        for packet in codec.parse(data[nal:]):
            packet.pts = pts
            packet.dts = pts
            pts += 1
            for frame in codec.decode(packet):
                if frames == 0:
                    h, w = frame.height, frame.width
                    png_path = os.path.join(LOG_DIR, f"text_sdk_frame0_{w}x{h}.png")
                    frame.to_image().save(png_path)
                    print(f"{tag}: 解码出图 {w}x{h}, 已存 {png_path}")
                frames += 1
        print(f"{tag}: 共解码 {frames} 帧")
        return frames > 0
    except Exception:
        print(f"{tag}: H264 解码异常:")
        traceback.print_exc()
        return False


def _run():
    host = "192.168.2.1"
    ctrl_port = 40923
    video_port = 40921

    ctrl = socket.create_connection((host, ctrl_port), timeout=5)
    print(f"控制连接 OK: {host}:{ctrl_port}")

    # 0. 使能 SDK 模式
    send_cmd(ctrl, "command;", wait=2.0)

    # 1. 机器人状态
    send_cmd(ctrl, "robot mode ?;", wait=1.5)
    send_cmd(ctrl, "robot battery ?;", wait=1.5)

    # 2. 相机子系统: 曝光命令有没有反应
    send_cmd(ctrl, "camera exposure small;", wait=1.5)

    # 3. 视频流: 连试 3 次
    stream_replies = []
    for i in range(3):
        stream_replies.append(send_cmd(ctrl, "stream on;", wait=1.5))
        time.sleep(1.0)
    stream_ok = any("ok" in r for r in stream_replies)
    print(f"stream on 三次回复: {stream_replies!r}")

    # 4. 音频流 (判断是不是整个 A/V 管线都挂了)
    send_cmd(ctrl, "audio on;", wait=1.5)

    # 5. 视频端口
    video = None
    deadline = time.time() + 3.0
    while time.time() < deadline:
        try:
            video = socket.create_connection((host, video_port), timeout=2)
            break
        except OSError as e:
            print(f"{video_port} 还没开 ({type(e).__name__}), 重试...")
            time.sleep(0.5)

    if video is not None:
        print(f"视频流连接 OK: {host}:{video_port}, 收 8 秒...")
        video.settimeout(1.0)
        data = b""
        end = time.time() + 8.0
        while time.time() < end:
            try:
                chunk = video.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                print("(视频流被对端关闭)")
                break
            data += chunk
            if len(data) > 4 * 1024 * 1024:
                break
        video.close()
        print(f"共收 {len(data)} 字节")
        if data:
            print(f"前 64 字节 hex: {data[:64].hex()}")
            try_decode(data, "video")
    else:
        print(f"ERROR: {video_port} 始终连不上")

    # 6. 收尾
    send_cmd(ctrl, "audio off;", wait=1.0)
    send_cmd(ctrl, "stream off;", wait=1.0)
    send_cmd(ctrl, "quit;", wait=1.0)
    ctrl.close()
    print("text sdk probe v2 done")


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
