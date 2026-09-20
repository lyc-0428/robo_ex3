#!/usr/bin/env python3
"""Run the repository YOLO model on the Gazebo RGB camera."""

from __future__ import annotations

import json
import os
import signal
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)
import torch
from ultralytics import YOLO

from vision_common import BOTTLE, TENNIS, validate_model_names


DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "model" / "best.pt"


def image_message_to_bgr(message):
    """Convert a ROS Image without the NumPy-sensitive cv_bridge extension."""
    encoding = str(message.encoding).lower()
    channels_by_encoding = {
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
        "8uc1": 1,
        "8uc3": 3,
        "8uc4": 4,
    }
    channels = channels_by_encoding.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported camera encoding: {message.encoding}")
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    row_bytes = width * channels
    if height <= 0 or width <= 0 or step < row_bytes:
        raise ValueError(
            f"invalid image geometry: {width}x{height}, step={step}, encoding={encoding}"
        )
    raw = np.frombuffer(message.data, dtype=np.uint8)
    required = height * step
    if raw.size < required:
        raise ValueError(f"short image buffer: got {raw.size} bytes, expected {required}")
    pixels = raw[:required].reshape(height, step)[:, :row_bytes]
    if channels == 1:
        mono = pixels.reshape(height, width)
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
    pixels = pixels.reshape(height, width, channels)
    if encoding in {"rgb8", "rgba8"}:
        return pixels[:, :, :3][:, :, ::-1].copy()
    return pixels[:, :, :3].copy()


def bgr_to_image_message(frame, header):
    """Build a bgr8 ROS Image without calling cv_bridge."""
    image = np.ascontiguousarray(frame, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR image, got shape {image.shape}")
    message = Image()
    message.header = header
    message.height = int(image.shape[0])
    message.width = int(image.shape[1])
    message.encoding = "bgr8"
    message.is_bigendian = 0
    message.step = int(image.shape[1] * 3)
    message.data = image.tobytes()
    return message


def request_stop(_signum, _frame):
    """Exit without running the unstable Jetson CUDA interpreter teardown."""
    os._exit(0)


class YoloBottleTennisDetector(Node):
    def __init__(self):
        super().__init__("bottle_tennis_yolo_detector")
        self.declare_parameter("model_path", str(DEFAULT_MODEL))
        self.declare_parameter("bottle_confidence", 0.40)
        self.declare_parameter("tennis_confidence", 0.50)
        self.declare_parameter("image_size", 640)
        self.declare_parameter("device", "0")
        self.declare_parameter("detection_log_path", "")

        self.model_path = Path(
            str(self.get_parameter("model_path").value)
        ).expanduser().resolve()
        self.thresholds = {
            BOTTLE: float(self.get_parameter("bottle_confidence").value),
            TENNIS: float(self.get_parameter("tennis_confidence").value),
        }
        self.image_size = int(self.get_parameter("image_size").value)
        self.device = str(self.get_parameter("device").value)
        detection_log_path = str(
            self.get_parameter("detection_log_path").value
        ).strip()
        self._detection_log = None
        if detection_log_path:
            log_path = Path(detection_log_path).expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._detection_log = log_path.open(
                "a", encoding="utf-8", buffering=1
            )
        if not self.model_path.is_file():
            raise RuntimeError(f"YOLO weight not found: {self.model_path}")
        if self.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; Jetson GPU inference is required. "
                "Use device:=cpu only for explicit diagnostics."
            )

        self.get_logger().info(f"Loading YOLO weight: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.class_mapping = validate_model_names(self.model.names)
        self.get_logger().info(f"Model classes: {self.model.names}")
        self.get_logger().info(f"Task class mapping: {self.class_mapping}")

        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, camera_qos
        )
        self._detection_enabled = False
        control_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            Bool,
            "/vision_sorting/enable_detection",
            self.enable_callback,
            control_qos,
        )
        self.image_publisher = self.create_publisher(
            Image, "/detections/image", 10
        )
        self.detection_publisher = self.create_publisher(
            String, "/detections", 10
        )
        self.structured_detection_publisher = self.create_publisher(
            Detection2DArray, "/detections_2d", 10
        )
        self.frame_count = 0
        self.smoothed_fps = 0.0
        self._reported_first_frame = False
        self._consecutive_failures = 0
        self.get_logger().info(
            "YOLO detector ready in paused state; awaiting acquire/align phase"
        )

    def enable_callback(self, message):
        enabled = bool(message.data)
        if enabled == self._detection_enabled:
            return
        self._detection_enabled = enabled
        self.get_logger().info(
            f"YOLO {'ENABLED' if enabled else 'PAUSED'} BY SORTING STATE"
        )

    def image_callback(self, message):
        if not self._detection_enabled:
            return
        start = time.perf_counter()
        try:
            frame = image_message_to_bgr(message)
            if not self._reported_first_frame:
                self._reported_first_frame = True
                self.get_logger().info(
                    "CAMERA FRAME READY | "
                    f"shape={frame.shape[1]}x{frame.shape[0]} | "
                    f"min={int(frame.min())} max={int(frame.max())} "
                    f"mean={float(frame.mean()):.1f}"
                )
            result = self.model.predict(
                source=frame,
                imgsz=self.image_size,
                conf=min(self.thresholds.values()),
                device=self.device,
                verbose=False,
            )[0]

            detections = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                for index, (box, score, class_number) in enumerate(
                    zip(boxes, scores, classes)
                ):
                    class_id = int(class_number)
                    task_class = self.class_mapping.get(class_id)
                    if task_class is None or float(score) < self.thresholds[task_class]:
                        continue
                    x1, y1, x2, y2 = [float(value) for value in box]
                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": task_class,
                            "raw_class_name": str(result.names[class_id]),
                            "confidence": round(float(score), 4),
                            "bbox": {
                                "x1": round(x1, 1),
                                "y1": round(y1, 1),
                                "x2": round(x2, 1),
                                "y2": round(y2, 1),
                            },
                        }
                    )
            elapsed = time.perf_counter() - start
            current_fps = 1.0 / elapsed if elapsed > 0.0 else 0.0
            self.smoothed_fps = (
                current_fps
                if self.smoothed_fps == 0.0
                else 0.9 * self.smoothed_fps + 0.1 * current_fps
            )
            annotated = frame.copy()
            for detection in detections:
                box = detection["bbox"]
                x1, y1, x2, y2 = (
                    int(round(box[key])) for key in ("x1", "y1", "x2", "y2")
                )
                label = (
                    f"{detection['class_name']} {detection['confidence']:.2f}"
                )
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
            cv2.putText(
                annotated,
                f"FPS {self.smoothed_fps:.1f} | objects {len(detections)}",
                (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            output = String()
            output.data = json.dumps(
                {
                    "frame": self.frame_count,
                    "fps": round(self.smoothed_fps, 2),
                    "count": len(detections),
                    "detections": detections,
                },
                ensure_ascii=False,
            )
            self.detection_publisher.publish(output)
            if self._detection_log is not None:
                self._detection_log.write(output.data + "\n")
            structured = Detection2DArray()
            structured.header = message.header
            for item in detections:
                box = item["bbox"]
                detection_2d = Detection2D()
                detection_2d.header = message.header
                detection_2d.bbox.center.position.x = (
                    float(box["x1"]) + float(box["x2"])
                ) * 0.5
                detection_2d.bbox.center.position.y = (
                    float(box["y1"]) + float(box["y2"])
                ) * 0.5
                detection_2d.bbox.size_x = float(box["x2"]) - float(box["x1"])
                detection_2d.bbox.size_y = float(box["y2"]) - float(box["y1"])
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = item["class_name"]
                hypothesis.hypothesis.score = float(item["confidence"])
                detection_2d.results.append(hypothesis)
                structured.detections.append(detection_2d)
            self.structured_detection_publisher.publish(structured)
            image = bgr_to_image_message(annotated, message.header)
            self.image_publisher.publish(image)
            self.frame_count += 1
            self._consecutive_failures = 0
        except Exception as error:
            self._consecutive_failures += 1
            if self._consecutive_failures == 1:
                self.get_logger().error(
                    "YOLO inference failed:\n" + traceback.format_exc()
                )
            if self._consecutive_failures >= 3:
                raise RuntimeError(
                    "YOLO failed on three consecutive camera frames: "
                    f"{type(error).__name__}: {error!r}"
                ) from error

    def close_detection_log(self):
        if self._detection_log is not None:
            self._detection_log.close()
            self._detection_log = None


def main(args=None):
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = YoloBottleTennisDetector()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.10)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close_detection_log()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
