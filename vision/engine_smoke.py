#!/usr/bin/env python3
"""引擎冒烟测试: 不连机器人, 验证 TRT 引擎 + pycuda + 后处理整条链。

用法 (板子上, 校园网即可, 不需要机器人):
    cd ~/Team21/vision
    ~/Team21/Team21/bin/python3 engine_smoke.py yolov8s_fp32.engine

黑图输入 5 次 (首推理含 lazy 初始化), 打印 output 形状/耗时, 正常输出
"SMOKE OK" 即说明检测节点的推理部分没问题, 只差机器人连上后跑真帧。
"""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/home/nvidia/Team21/colcon_ws/src/robomaster_pick_place_sim/vision")
from vision_detector import TrtYOLO, CLASS_NAMES  # noqa: E402

engine_path = sys.argv[1] if len(sys.argv) > 1 else "yolov8s_fp32.engine"
detector = TrtYOLO(engine_path, 640)
print("引擎加载 OK, 类名:", CLASS_NAMES)

img = np.zeros((360, 640, 3), np.uint8)
for i in range(5):
    t = time.time()
    out, r, px, py = detector.infer(img)
    dets = TrtYOLO.postprocess(out, r, px, py, 640, 360)
    ms = (time.time() - t) * 1000
    print(f"第 {i+1} 次: output {out.shape}, {ms:.1f}ms, dets={dets}")
    if ms > 100:
        print("WARN: 推理 >100ms (首帧初始化可忽略)")

print("SMOKE OK")
