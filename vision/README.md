# vision —— 实验三视觉检测 (bottle / tennis_ball)

> **视觉联动定点抓取 (实验升级)**: 见 [pick_place_README.md](pick_place_README.md)
> — calibrate.py 标定 + vision_pick_place.py 一体化抓取, 含进度/移交清单。

机器人云台相机 (robomaster SDK 取帧) + TensorRT YOLOv8s 推理 + cv2 实时窗口
+ ROS2 话题 `detections` (JSON String)。

模型: 队友训练的 `bottle_tennisball_best.pt` (YOLOv8s, imgsz 640, 类名
`{0: bottle, 1: tennis_ball}`), 位于本仓库 main 分支根目录。

## 文件

| 文件 | 作用 |
|---|---|
| `vision_detector.py` | 检测节点: SDK 取帧 → TRT 推理 → 时间平滑 → 画框/发布。板子上直接 `python3` 跑, 不用 colcon |
| `build_engine.py` | ONNX → TensorRT 引擎 (exp1 移植, 替代坏掉的 trtexec; `--fp32` 可选) |
| `engine_smoke.py` | 引擎冒烟测试: 不连机器人, 验证引擎+pycuda+后处理全链路 |

## 板子部署流程 (Jetson Orin NX, Team21)

```bash
# 1. Mac 上导出 ONNX (exp1 venv 有 ultralytics):
#    m = YOLO('bottle_tennisball_best.pt')
#    m.export(format='onnx', opset=12, simplify=True, imgsz=640)

# 2. 传到板子 ~/Team21/vision/ 后构建引擎 (约 5 分钟, 板子专属, 换板必须重建):
#    验收用 FP16 (2026-09-13 实机 A/B: FP16 更快且置信度无明显下降);
#    FP32 一并构建留作高置信度备份:
~/Team21/Team21/bin/python3 build_engine.py bottle_tennisball_best.onnx yolov8s_fp16.engine
~/Team21/Team21/bin/python3 build_engine.py --fp32 bottle_tennisball_best.onnx yolov8s_fp32.engine

# 3. 冒烟测试 (不用连机器人):
~/Team21/Team21/bin/python3 engine_smoke.py yolov8s_fp32.engine

# 4. 真机运行 (板子连机器人热点 RMEP-21bdc0, 手机 App 断开机器人):
cd ~/Team21/colcon_ws/src/robomaster_pick_place_sim
~/Team21/Team21/bin/python3 vision/vision_detector.py            # 默认 FP16 引擎, conf 0.5, 360p
~/Team21/Team21/bin/python3 vision/vision_detector.py --conf 0.35 --resolution 720p
```

日志落盘 `~/Team21/logs/vision_detector_<时间戳>.txt`; 按 `q` 退出。

## 踩过的坑 (2026-09-13, 新板 Orin NX / TRT 10.3)

- **pycuda 必须对着 numpy 2 编译**: pip 默认隔离构建用 numpy 1.x 编 wheel,
  运行时 numpy 2.2.6 直接 `SystemError: _ARRAY_API not found`。修复:
  `pip install -U pybind11 setuptools wheel && pip install --no-build-isolation
  --force-reinstall pycuda` (构建隔离会再装 numpy 1.x)。装完把 setuptools 降回
  `<80` (venv 里 colcon-core 0.21.0 要求)。
- **ROS numpy 1.x 顶掉 cv2 5.x 的 numpy 2.x**: 脚本顶部在 `import numpy/cv2`
  之前把 /opt/ros 路径挪到 sys.path 末尾; rclpy 之后仍能找到, 不用 source ROS
  也能跑 (脚本自己补路径 + LD_LIBRARY_PATH)。
- **cv2 5.x putText 拒绝负步长数组**: SDK 帧是 RGB, `img[:, :, ::-1]` 转 BGR
  得到非连续视图, putText 报 "Layout of the output array img is incompatible"。
  必须 `np.ascontiguousarray(...)`。
- **SDK 帧是 RGB 不是 BGR** (libmedia_codec rgb24 直接 reshape)。

## 性能 (2026-09-13 实测, Orin NX 25W 默认功率档)

- FP32 引擎: 推理 56~84ms (平均 ~70ms, 约 14fps), 置信度稳定 —
  真机 21 分钟连续跑: bottle 0.83~0.94, tennis_ball 0.89~0.93, 多目标同框正常。
- FP16 引擎: 冒烟实测稳态 **29.7ms** (约 2.4 倍速, 推理已超相机 30fps);
  **2026-09-13 实机 A/B 确认 FP16 效果更好, 定为默认验收引擎**。
  FP32 保留作高置信度备份 (`--engine` 切换)。
- 板子功率档 25W (默认); 切 40W MAXN 还能再快 20-30% 但需 sudo+重启,
  小组实验用不上 (2026-09-13 与用户确认不动系统设置)。
- 360p 即可: 模型输入固定 640x640, 提相机分辨率只增加解码负担, 不提升检测。
