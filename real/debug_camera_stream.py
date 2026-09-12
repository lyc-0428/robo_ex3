#!/usr/bin/env python3
"""相机视频流开启失败定位脚本: 逐步执行 SDK 启动链路, 记录每步结果。

用法 (板子已连机器人热点, 板子本地终端里跑):
    cd ~/colcon_ws/src/robomaster_pick_place_sim
    python3 real/debug_camera_stream.py

完整输出 (含 SDK 内部日志) 同时保存到 ~/Desktop/camera_debug_<时间戳>.txt,
跑完把板子切回校园网, 该文件可直接被 Mac 读回分析。
"""
import logging
import os
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
        path = os.path.join(LOG_DIR, f"camera_debug_{stamp}.txt")
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


def _run():
    from robomaster import robot, logger, protocol

    # 机器人推送的 cmdset:0x3f/cmdid:0x29 事件 SDK 没注册,
    # 视频流控制失败前后它总会出现 —— 注册一个调试类把原始字节打出来
    class _ProtoDebug29(protocol.ProtoData):
        _cmdset = 0x3f
        _cmdid = 0x29

        def __init__(self):
            self._raw = b""

        def pack_req(self):
            return b""

        def unpack_req(self, buf, offset=0):
            self._raw = bytes(buf)
            print("0x29 event payload:", self._raw.hex())
            return True

        def unpack_resp(self, buf, offset=0):
            self._raw = bytes(buf)
            print("0x29 resp payload:", self._raw.hex())
            return True

    protocol.registered_protos[protocol.make_proto_cls_key(0x3f, 0x29)] = _ProtoDebug29

    logger.setLevel(logging.INFO)

    # SSH 里跑也能弹窗 (本地终端跑则已自带 DISPLAY)
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

    ep = robot.Robot()
    try:
        ok = ep.initialize(conn_type="ap")
        print("initialize:", ok)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    if not ok:
        print("初始化失败, 退出")
        sys.exit(1)

    cam = ep.camera
    print("stream addr:", cam.video_stream_addr,
          "proto:", ep.conf.video_stream_proto)

    try:
        print("firmware version:", ep.get_version())
    except Exception:
        print("get_version EXC:")
        traceback.print_exc()

    resolution = sys.argv[1] if len(sys.argv) > 1 else "720p"
    print("resolution:", resolution)

    # 第 0 步: 先尝试干净退出 SDK 流模式 (上次异常退出可能把机器人卡住)
    try:
        print("step0 _stream_sdk(0) ->", cam._stream_sdk(0, resolution))
        time.sleep(1.0)
    except Exception:
        print("step0 EXC:")
        traceback.print_exc()

    # 第 1/2 步: SDK 流控制命令 (start_video_stream 内部链路)
    for name, fn in [
        ("step1 _stream_sdk(1)", lambda: cam._stream_sdk(1, resolution)),
        ("step2 _video_stream(1)", lambda: cam._video_stream(1, resolution)),
    ]:
        try:
            print(name, "->", fn())
        except Exception:
            print(name, "EXC:")
            traceback.print_exc()
        time.sleep(0.5)

    # 第 3 步: LiveView TCP 连接 (先不开显示, 只看流本身)
    cam._video_enable = True
    try:
        r3 = cam._liveview.start_video_stream(
            False, cam.video_stream_addr, ep.conf.video_stream_proto)
        print("step3 liveview connect (display=False):", r3)
    except Exception:
        print("step3 EXC:")
        traceback.print_exc()

    # 等解码帧 (看 TCP 流 + PyAV 解码是否出帧)
    for i in range(8):
        n = getattr(cam._liveview, "_video_frame_count", 0)
        print("t+%ds frames decoded: %d" % (i, n))
        if n > 0:
            break
        time.sleep(1)

    # 显示测试: 主线程弹窗 3 秒
    try:
        import cv2
        print("cv2 version:", cv2.__version__)
        try:
            frame = cam._liveview.read_video_frame(timeout=5)
        except Exception as e:
            print("read frame failed:", type(e).__name__, e)
            frame = None
        print("got frame:", None if frame is None else frame.shape)
        if frame is not None:
            cv2.imshow("RoboMaster LiveView", frame)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
            print("window shown OK")
        else:
            print("WARN: 没有拿到帧, 跳过显示测试")
    except Exception:
        print("display EXC:")
        traceback.print_exc()

    try:
        cam._liveview.stop_video_stream()
    except Exception:
        pass
    try:
        cam._stream_sdk(0)
    except Exception:
        pass
    ep.close()
    print("debug done")


if __name__ == "__main__":
    main()
