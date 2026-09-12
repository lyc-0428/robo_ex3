#!/usr/bin/env python3
"""相机视频流探针 v3: 端口扫描 + 双 conn_type 尝试。

背景: 机器人对所有流控命令要么不答, 要么回 0xd1 ack(retcode 0),
但 40921 端口始终不开; 云台/视觉管线正常, 说明相机硬件没问题。
本脚本在发完流控命令后扫描机器人 TCP 端口 (40900-40930 + 40000-41000),
并用 conn_type=0 (WiFi) 和 conn_type=1 (RNDIS) 各试一遍。

用法 (板子已连机器人热点, 本地终端):
    cd ~/colcon_ws/src/robomaster_pick_place_sim
    python3 real/debug_camera_raw.py [360p|540p|720p]

日志存 ~/Desktop/camera_raw_debug_<时间戳>.txt。
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
        path = os.path.join(LOG_DIR, f"camera_raw_debug_{stamp}.txt")
        with open(path, "w") as f:
            f.write("\n".join(LOG_LINES) + "\n")
        print(f"调试日志已保存: {path}")
    except Exception as e:
        print(f"保存调试日志失败: {e}")


def main():
    sys.stdout = _Tee(sys.stdout)
    sys.stderr = _Tee(sys.stderr)
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s",
                        stream=sys.stderr)
    try:
        _run()
    finally:
        save_log()


def scan_ports(ip, lo, hi, timeout=0.2):
    """并行 TCP connect 扫描, 返回开放端口列表。"""
    def check(p):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            return p if s.connect_ex((ip, p)) == 0 else None
        finally:
            s.close()

    open_ports = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as ex:
        for r in ex.map(check, range(lo, hi + 1)):
            if r:
                open_ports.append(r)
    return sorted(open_ports)


def _run():
    from robomaster import robot, protocol, logger

    logger.setLevel(logging.INFO)

    # 0xd1/0xd2 调试类: 打印 ack 原始载荷 (带 cmdset/cmdid 属性避免别的
    # 模块分发时崩掉)
    class _ProtoD1Debug:
        cmdset = 0x3f
        cmdid = 0xd1

        def __init__(self):
            self._retcode = None

        def pack_req(self):
            return b"\x01"

        def unpack_req(self, buf, offset=0):
            print("0xd1 REQ payload:", bytes(buf).hex())
            return True

        def unpack_resp(self, buf, offset=0):
            self._retcode = buf[offset]
            print("0xd1 ACK payload:", bytes(buf).hex(), "retcode:", self._retcode)
            return True

    class _ProtoD2Debug:
        cmdset = 0x3f
        cmdid = 0xd2

        def __init__(self):
            self._retcode = None

        def pack_req(self):
            return b"\x01"

        def unpack_req(self, buf, offset=0):
            print("0xd2 REQ payload:", bytes(buf).hex())
            return True

        def unpack_resp(self, buf, offset=0):
            self._retcode = buf[offset]
            print("0xd2 ACK payload:", bytes(buf).hex(), "retcode:", self._retcode)
            return True

    protocol.registered_protos[protocol.make_proto_cls_key(0x3f, 0xd1)] = _ProtoD1Debug
    protocol.registered_protos[protocol.make_proto_cls_key(0x3f, 0xd2)] = _ProtoD2Debug

    resolution = sys.argv[1] if len(sys.argv) > 1 else "360p"
    res_code = {"720p": 0, "360p": 1, "540p": 2}.get(resolution, 1)
    print("resolution:", resolution, "code:", res_code)

    ep = robot.Robot()
    ok = ep.initialize(conn_type="ap")
    print("initialize:", ok)
    if not ok:
        print("初始化失败, 退出")
        sys.exit(1)

    try:
        print("firmware version:", ep.get_version())
    except Exception:
        traceback.print_exc()

    print("baseline open ports 40000-41000:",
          scan_ports("192.168.2.1", 40000, 41000))

    client = ep.client
    target = ep.camera._host

    def send_stream_ctrl(ctrl, state, conn_type):
        proto = protocol.ProtoStreamCtrl()
        proto._ctrl = ctrl
        proto._conn_type = conn_type
        proto._state = state
        proto._resolution = res_code
        msg = protocol.Msg(client.hostbyte, target, proto)
        client.send_msg(msg)
        print(f"sent stream ctrl: ctrl={ctrl} state={state} conn_type={conn_type} res={resolution}")

    def try_sequence(conn_type):
        print(f"--- try conn_type={conn_type} ---")
        send_stream_ctrl(1, 1, conn_type)
        time.sleep(1.0)
        send_stream_ctrl(2, 1, conn_type)
        time.sleep(4.0)
        ports = scan_ports("192.168.2.1", 40900, 40930)
        print(f"open ports 40900-40930: {ports}")
        try:
            s = socket.create_connection(("192.168.2.1", 40921), timeout=5)
            print("40921 CONNECT OK  <<< 视频流端口已开!")
            s.close()
            return True
        except Exception as e:
            print("40921 connect failed:", type(e).__name__, e)
        return False

    opened = try_sequence(0)  # WiFi
    if not opened:
        try_sequence(1)  # RNDIS

    print("open ports 40000-41000 after attempts:",
          scan_ports("192.168.2.1", 40000, 41000))

    try:
        send_stream_ctrl(2, 0, 0)
        time.sleep(1.0)
        send_stream_ctrl(1, 0, 0)
        time.sleep(1.0)
    except Exception:
        traceback.print_exc()
    ep.close()
    print("raw debug done")


if __name__ == "__main__":
    main()
