#!/usr/bin/env python3
"""机器人网段扫描: 找出网络上所有设备 + 各自开放的端口。

机身 192.168.2.1 上只有 40923/40925 开着但静默; 第二个摄像头
(智能中控?) 是独立设备, 有自己的 IP。本脚本扫 192.168.2.2-2.60
的所有主机, 对每个活跃主机探测常见端口, 并抓取开放端口的
banner (前几秒吐出的数据), 判断哪个设备/端口是视频流。

用法 (板子已连机器人热点):
    cd ~/colcon_ws/src/robomaster_pick_place_sim
    python3 real/debug_camera_scan_net.py

日志: ~/Desktop/net_scan_debug_<时间戳>.txt
"""
import concurrent.futures
import logging
import os
import socket
import sys
import time
import traceback

LOG_DIR = os.path.expanduser("~/Desktop")
LOG_LINES = []

PROBE_PORTS = [21, 22, 23, 80, 443, 8000, 8080, 8888,
               40921, 40922, 40923, 40924, 40925, 40926, 40927]


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
        path = os.path.join(LOG_DIR, f"net_scan_debug_{stamp}.txt")
        with open(path, "w") as f:
            f.write("\n".join(LOG_LINES) + "\n")
        print(f"调试日志已保存: {path}")
    except Exception as e:
        print(f"保存调试日志失败: {e}")


def tcp_probe(ip, port, timeout=0.25):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        if s.connect_ex((ip, port)) == 0:
            return port
        return None
    finally:
        s.close()


def scan_host(ip):
    open_ports = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        for r in ex.map(lambda p: tcp_probe(ip, p), PROBE_PORTS):
            if r:
                open_ports.append(r)
    return open_ports


def sniff_banner(ip, port, seconds=2.0):
    """连接后被动收几秒, 返回收到的字节 (可能为空)。"""
    data = b""
    try:
        s = socket.create_connection((ip, port), timeout=3)
        s.settimeout(0.5)
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                chunk = s.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                break
            data += chunk
            if len(data) > 4096:
                break
        s.close()
    except Exception:
        pass
    return data


def _run():
    live = []
    print("扫描 192.168.2.2 - 192.168.2.60 ...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as ex:
        def check(i):
            ip = f"192.168.2.{i}"
            r = tcp_probe(ip, 21, timeout=0.3)
            if r is None:
                r = tcp_probe(ip, 40921, timeout=0.3)
            return ip if r is not None else None
        for ip in ex.map(check, range(2, 61)):
            if ip:
                live.append(ip)
                print(f"活跃主机: {ip}")
    print(f"共 {len(live)} 台活跃设备")

    for ip in live:
        ports = scan_host(ip)
        print(f"{ip} 开放端口: {ports}")
        for port in ports:
            banner = sniff_banner(ip, port)
            if banner:
                printable = "".join(chr(b) if 32 <= b < 127 else "."
                                   for b in banner[:80])
                print(f"  {ip}:{port} banner ({len(banner)} 字节): {printable}")
                if banner[:2] == b"\xff\xd8":
                    print(f"  {ip}:{port} <<< JPEG 视频流!")
                elif banner[:4] in (b"\x00\x00\x00\x01", b"\x00\x00\x01"):
                    print(f"  {ip}:{port} <<< H264 NAL 视频流!")
    print("net scan done")


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
