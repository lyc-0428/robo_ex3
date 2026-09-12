#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNX -> TensorRT FP16 引擎（在 Jetson 上运行）

用法: python3 build_engine.py best.onnx yolov8n_fp16.engine

替代 trtexec：本机 trtexec 10.7 构建成功后写引擎文件时报
"Saving engine to file failed"，改用 tensorrt Python API 直接序列化。
构建约 7 分钟（Orin Nano 逐层性能剖析），输出固定 640x640。
"""

import sys

import tensorrt as trt


def main():
    args = sys.argv[1:]
    fp32 = "--fp32" in args
    args = [a for a in args if a != "--fp32"]
    onnx_path, engine_path = args[0], args[1]
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            sys.exit(1)

    print("网络输入:")
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print("  ", t.name, t.shape, t.dtype)
    print("网络输出:")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print("  ", t.name, t.shape, t.dtype)

    config = builder.create_builder_config()
    if not fp32:
        config.set_flag(trt.BuilderFlag.FP16)
    print("精度:", "FP32" if fp32 else "FP16")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("构建失败", file=sys.stderr)
        sys.exit(1)
    with open(engine_path, "wb") as f:
        f.write(bytes(serialized))
    print("已保存引擎: {} ({} bytes)".format(engine_path, serialized.nbytes))


if __name__ == "__main__":
    main()
