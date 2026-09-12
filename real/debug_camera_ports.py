#!/usr/bin/env python3
"""40923/40925 端口嗅探: 弄清这两个总是开着的端口在吐什么数据。

端口扫描发现: 192.168.2.1 上 40923/40925 在没发任何流控命令时就开着,
而 SDK 只定义了 40921(云台视频)/40922(音频)。用户说机器人上有两个
摄像头 (一个跟机械臂在一起, 另一个外连) —— 这两个端口很可能就是
第二个摄像头的流。

本脚本逐个连接, 被动收 6 秒, 存原始字节, 并尝试按 H264 解码出图。
若 2 秒内无数据, 再发 HTTP GET 试探 (万一是 web 服务)。

用法 (板子已连机器人热点):
    cd ~/colcon_ws/src/robomaster_pick_place_sim
    python3 real/debug_camera_ports.py

日志: ~/Desktop/port_sniff_debug_<时间戳>.txt
原始数据: ~/Desktop/port_<端口>_dump.bin
"""
import logging
import os
import socket
import sys
import time
import traceback

LOG_DIR = os.path.expanduser("~/Desktop")
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
        path = os.path.join(LOG_DIR, f"port_sniff_debug_{stamp}.txt")
        with open(path, "w") as f:
            f.write("\n".join(LOG_LINES) + "\n")
        print(f"调试日志已保存: {path}")
    except Exception as e:
        print(f"保存调试日志失败: {e}")


def try_decode_h264(data, tag):
    """从字节流里切 NAL 单元喂 PyAV, 成功出图就存 PNG。"""
    import av
    nal = data.find(b"\x00\x00\x00\x01")
    if nal < 0:
        nal = data.find(b"\x00\x00\x01")
    if nal < 0:
        print(f"{tag}: 没有找到 H264 NAL 起始码")
        return False
    codec = av.CodecContext.create("h264", "r")
    codec.open()
    pts = 0
    frame_count = 0
    try:
        for packet in codec.parse(data[nal:]):
            packet.pts = pts
            packet.dts = pts
            pts += 1
            for frame in codec.decode(packet):
                img = frame.to_ndarray(format="rgb24")
                h, w, _ = img.shape
                if frame_count == 0:
                    png_path = os.path.join(
                        LOG_DIR, f"port_{tag}_frame0_{w}x{h}.png")
                    frame.to_image().save(png_path)
                    print(f"{tag}: H264 解码出图 {w}x{h}, 已存 {png_path}")
                frame_count += 1
        print(f"{tag}: H264 解码共 {frame_count} 帧")
        return frame_count > 0
    except Exception:
        print(f"{tag}: H264 解码异常:")
        traceback.print_exc()
        return False


def sniff_port(ip, port):
    print(f"=== sniff {ip}:{port} ===")
    data = b""
    try:
        s = socket.create_connection((ip, port), timeout=5)
        s.settimeout(1.0)
        print(f"{port}: 连接成功, 被动收 6 秒...")
        deadline = time.time() + 6.0
        while time.time() < deadline:
            try:
                chunk = s.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                print(f"{port}: 对端关闭连接 (共收 {len(data)} 字节)")
                break
            data += chunk
        if len(data) == 0:
            # 可能是 HTTP 之类的问答式服务, 试探一下
            try:
                s.sendall(b"GET / HTTP/1.0\r\n\r\n")
                print(f"{port}: 无被动数据, 已发 HTTP GET 试探")
                deadline = time.time() + 4.0
                while time.time() < deadline:
                    try:
                        chunk = s.recv(65536)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    data += chunk
            except Exception as e:
                print(f"{port}: HTTP 试探失败:", type(e).__name__, e)
        s.close()
    except Exception as e:
        print(f"{port}: 连接失败:", type(e).__name__, e)
        return

    print(f"{port}: 共收 {len(data)} 字节")
    if data:
        dump_path = os.path.join(LOG_DIR, f"port_{port}_dump.bin")
        with open(dump_path, "wb") as f:
            f.write(data)
        print(f"{port}: 原始数据已存 {dump_path}")
        print(f"{port}: 前 64 字节 hex: {data[:64].hex()}")
        printable = "".join(chr(b) if 32 <= b < 127 else "." for b in data[:64])
        print(f"{port}: 前 64 字节 ascii: {printable}")
        if data[:2] == b"\xff\xd8":
            print(f"{port}: JPEG 魔数! FFD8 开头")
            jpg_path = os.path.join(LOG_DIR, f"port_{port}_frame0.jpg")
            with open(jpg_path, "wb") as f:
                f.write(data[:200000])
            print(f"{port}: 前 200KB 已存为 {jpg_path} (可打开看)")
        else:
            try_decode_h264(data, str(port))
    else:
        print(f"{port}: 没有收到任何数据")


def _run():
    from robomaster import robot, logger

    logger.setLevel(logging.INFO)
    ep = robot.Robot()
    ok = ep.initialize(conn_type="ap")
    print("initialize:", ok)
    if not ok:
        print("初始化失败, 退出")
        sys.exit(1)
    try:
        sniff_port("192.168.2.1", 40923)
        sniff_port("192.168.2.1", 40925)
    finally:
        ep.close()
    print("port sniff done")


def main():
    sys.stdout = _Tee(sys.stdout)
    sys.stderr = _Tee(sys.stderr)
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s",
                        stream=sys.stderr)
    try:
        _run()
    finally:
        save_log()


if __name__ == "__main__":
    main()
