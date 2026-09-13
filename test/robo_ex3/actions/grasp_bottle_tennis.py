#!/usr/bin/env python3
"""Vision-guided six-object sorting with measured-pose chassis feedback."""

from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
import shutil
import time

import rclpy
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Bool, String
from sorting_identity import SortingIdentity
from pose_feedback import (
    Pose2D,
    PoseController,
    PoseControllerConfig,
    standoff_pose,
    wrap_angle,
)
from task_state_machine import SortingState, SortingStateMachine

from grasp_cube import GraspActionError, GraspCubeAction, POSITION_JOINTS
from vision_common import (
    BOTTLE,
    TENNIS,
    canonical_class,
    horizontal_angle_radians,
    stable_center_detection,
)


class GraspBottleTennisAction(GraspCubeAction):
    """Sort six objects using camera classes and measured chassis pose."""

    def __init__(self):
        super().__init__()
        self.run_count = 6
        self.spawn_test_cube = False
        self.expected_counts = {BOTTLE: 3, TENNIS: 3}

        # Preserve the two user-calibrated grasp modes.
        self.bottle_arm_2_delta = self.arm_2_delta
        self.tennis_arm_2_delta = -0.78
        self._use_tennis_grasp = False

        # The outer slots need a longer detector window on the Jetson.
        self.declare_parameter("detection_timeout_seconds", 15.0)
        self.declare_parameter("stable_window_frames", 5)
        self.declare_parameter("stable_required_frames", 4)
        self.declare_parameter("stable_center_spread_px", 35.0)
        self.declare_parameter("alignment_tolerance_px", 12.0)
        # Jetson live calibration: use half the original angular gain.  The
        # camera image moves opposite to the chassis yaw command, so visual
        # corrections invert the geometric image angle below.
        self.declare_parameter("alignment_kp", 0.02)
        self.declare_parameter("alignment_min_speed", 0.4)
        self.declare_parameter("alignment_max_speed", 3.0)
        self.declare_parameter("alignment_smoothing", 0.35)
        self.declare_parameter("aligned_required_frames", 3)
        self.declare_parameter("max_grasp_retries", 2)
        self.declare_parameter("controller_ready_file", "")
        self.declare_parameter("scene_ready_file", "")
        self.declare_parameter("scene_ready_timeout_seconds", 60.0)
        self.declare_parameter("pose_motion_timeout_seconds", 60.0)
        self.declare_parameter("pose_control_rate_hz", 40.0)
        self.declare_parameter("position_tolerance_m", 0.010)
        self.declare_parameter("position_reacquire_tolerance_m", 0.025)
        self.declare_parameter("yaw_tolerance_degrees", 1.0)
        self.declare_parameter("pose_stable_samples", 4)
        self.declare_parameter("pose_linear_kp", 8.0)
        self.declare_parameter("pose_angular_kp", 4.0)
        self.declare_parameter("pose_max_linear_wheel_speed", 2.0)
        self.declare_parameter("pose_max_angular_wheel_speed", 1.8)
        self.declare_parameter("bottle_grasp_standoff_m", 0.350)
        self.declare_parameter("tennis_grasp_standoff_m", 0.300)
        self.declare_parameter("placement_yaw_degrees", 90.0)
        self.declare_parameter("placement_distance_m", 0.450)
        self.declare_parameter("scene_mode", "nominal")
        self.declare_parameter("total_slot_count", 6)
        self.declare_parameter("minimum_success_count", 5)
        self.declare_parameter("class_limit_per_category", 3)

        self.detection_timeout = float(
            self.get_parameter("detection_timeout_seconds").value
        )
        self.stable_window_frames = int(
            self.get_parameter("stable_window_frames").value
        )
        self.stable_required_frames = int(
            self.get_parameter("stable_required_frames").value
        )
        self.stable_center_spread_px = float(
            self.get_parameter("stable_center_spread_px").value
        )
        self.alignment_tolerance_px = float(
            self.get_parameter("alignment_tolerance_px").value
        )
        self.alignment_kp = float(
            self.get_parameter("alignment_kp").value
        )
        self.alignment_min_speed = float(
            self.get_parameter("alignment_min_speed").value
        )
        self.alignment_max_speed = float(
            self.get_parameter("alignment_max_speed").value
        )
        self.alignment_smoothing = float(
            self.get_parameter("alignment_smoothing").value
        )
        self.aligned_required_frames = int(
            self.get_parameter("aligned_required_frames").value
        )
        self.max_grasp_retries = int(
            self.get_parameter("max_grasp_retries").value
        )
        self.controller_ready_file = str(
            self.get_parameter("controller_ready_file").value
        )
        self.scene_ready_file = str(self.get_parameter("scene_ready_file").value)
        self.scene_ready_timeout = float(
            self.get_parameter("scene_ready_timeout_seconds").value
        )
        self.pose_motion_timeout = float(
            self.get_parameter("pose_motion_timeout_seconds").value
        )
        pose_control_rate = float(
            self.get_parameter("pose_control_rate_hz").value
        )
        self.pose_control_period = 1.0 / pose_control_rate
        self.bottle_grasp_standoff = float(
            self.get_parameter("bottle_grasp_standoff_m").value
        )
        self.tennis_grasp_standoff = float(
            self.get_parameter("tennis_grasp_standoff_m").value
        )
        placement_yaw_degrees = float(
            self.get_parameter("placement_yaw_degrees").value
        )
        if not math.isclose(abs(placement_yaw_degrees), 90.0, abs_tol=1.0e-6):
            raise GraspActionError("placement_yaw_degrees must be exactly 90")
        self.placement_yaw = math.pi * 0.5
        self.placement_distance = float(
            self.get_parameter("placement_distance_m").value
        )
        self.run_count = int(self.get_parameter("total_slot_count").value)
        self.minimum_success_count = int(
            self.get_parameter("minimum_success_count").value
        )
        class_limit = int(
            self.get_parameter("class_limit_per_category").value
        )
        self.scene_mode = str(self.get_parameter("scene_mode").value).strip()
        if self.scene_mode not in {"nominal", "unknown", "empty", "tennis_only"}:
            raise GraspActionError(f"unsupported scene_mode: {self.scene_mode}")
        if self.scene_mode == "tennis_only":
            # Placement-debug mode: no bottle entities and no bottle-class
            # detections are accepted. Six balls exercise repeated return,
            # exact turn, distant placement, and return-home behavior.
            self.run_count = 6
            self.minimum_success_count = 5
            self.expected_counts = {BOTTLE: 0, TENNIS: 6}
            self.enabled_classes = {TENNIS}
            self.require_two_categories = False
        else:
            self.expected_counts = {BOTTLE: class_limit, TENNIS: class_limit}
            self.enabled_classes = {BOTTLE, TENNIS}
            self.require_two_categories = True
        self.pose_controller = PoseController(PoseControllerConfig(
            position_tolerance=float(
                self.get_parameter("position_tolerance_m").value
            ),
            position_reacquire_tolerance=float(
                self.get_parameter("position_reacquire_tolerance_m").value
            ),
            yaw_tolerance=math.radians(float(
                self.get_parameter("yaw_tolerance_degrees").value
            )),
            required_stable_samples=int(
                self.get_parameter("pose_stable_samples").value
            ),
            linear_kp=float(self.get_parameter("pose_linear_kp").value),
            angular_kp=float(self.get_parameter("pose_angular_kp").value),
            max_linear_wheel_speed=float(
                self.get_parameter("pose_max_linear_wheel_speed").value
            ),
            max_angular_wheel_speed=float(
                self.get_parameter("pose_max_angular_wheel_speed").value
            ),
        ))

        if not self.stable_window_frames >= self.stable_required_frames >= 1:
            raise GraspActionError("stable window must be >= required frames >= 1")
        if min(
            self.detection_timeout,
            self.stable_center_spread_px,
            self.alignment_tolerance_px,
        ) <= 0.0:
            raise GraspActionError("vision timing and alignment values must be positive")
        if min(
            self.pose_motion_timeout,
            pose_control_rate,
            self.bottle_grasp_standoff,
            self.tennis_grasp_standoff,
            self.placement_yaw,
            self.placement_distance,
        ) <= 0.0:
            raise GraspActionError("pose-feedback parameters must be positive")
        if self.max_grasp_retries < 0:
            raise GraspActionError("max_grasp_retries cannot be negative")
        if not 1 <= self.minimum_success_count <= self.run_count:
            raise GraspActionError(
                "minimum_success_count must be between one and total_slot_count"
            )
        if class_limit < 1:
            raise GraspActionError("class_limit_per_category must be positive")

        self.camera_width = None
        self.camera_focal_x = None
        self.camera_principal_x = None
        self.detection_frames = deque(maxlen=self.stable_window_frames)
        self.last_detection_frame_number = -1
        self.create_subscription(
            CameraInfo,
            "/camera/camera_info",
            self._camera_info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(String, "/detections", self._detection_callback, 10)
        vision_control_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.vision_enable_publisher = self.create_publisher(
            Bool,
            "/vision_sorting/enable_detection",
            vision_control_qos,
        )

        self.category_counts = {BOTTLE: 0, TENNIS: 0}
        self._current_class = None
        self._active_initial_positions = None
        self._ignore_detections = True
        self._identity = None
        self._locked_object_id = None
        self._locked_object_origin = None
        self._cycle_pose = None
        self._aligned_yaw = None
        self._placement_ready = False
        self.camera_focal_y = None
        self.camera_principal_y = None
        self.task_state = SortingStateMachine(self._publish_status)

    @staticmethod
    def _write_ready_file(path_text):
        if path_text:
            path = Path(path_text)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("ready\n", encoding="utf-8")

    def _set_vision_enabled(self, enabled, reason):
        """Gate both YOLO inference and controller-side detection intake."""
        enabled = bool(enabled)
        self._ignore_detections = not enabled
        self.detection_frames.clear()
        message = Bool()
        message.data = enabled
        for _ in range(3):
            self.vision_enable_publisher.publish(message)
        self._publish_status(
            f"VISION {'ENABLED' if enabled else 'PAUSED'} | reason={reason}"
        )

    def _wait_for_scene_ready(self):
        if not self.scene_ready_file:
            return
        deadline = time.monotonic() + self.scene_ready_timeout
        scene_path = Path(self.scene_ready_file)
        while rclpy.ok() and time.monotonic() < deadline:
            if scene_path.is_file():
                self._publish_status("SORTING SCENE READY FOR VISION")
                return
            rclpy.spin_once(self, timeout_sec=0.10)
        raise GraspActionError(
            f"sorting scene was not ready within {self.scene_ready_timeout:.1f} s"
        )

    def _camera_info_callback(self, message):
        if message.width <= 0 or len(message.k) < 9 or message.k[0] <= 0.0:
            return
        self.camera_width = int(message.width)
        self.camera_focal_x = float(message.k[0])
        self.camera_principal_x = float(message.k[2])
        self.camera_focal_y = float(message.k[4])
        self.camera_principal_y = float(message.k[5])

    def _detection_callback(self, message):
        if self._ignore_detections:
            return
        try:
            payload = json.loads(message.data)
            frame_number = int(payload["frame"])
            if frame_number <= self.last_detection_frame_number:
                return
            clean = []
            for detection in payload.get("detections", []):
                task_class = canonical_class(detection.get("class_name", ""))
                bbox = detection.get("bbox", {})
                if task_class not in self.enabled_classes or not all(
                    key in bbox for key in ("x1", "y1", "x2", "y2")
                ):
                    continue
                item = dict(detection)
                item["class_name"] = task_class
                clean.append(item)
            if self._identity is not None:
                clean = self._identity.associate(
                    clean, self.latest_joint_state,
                    (self.camera_focal_x, self.camera_focal_y,
                     self.camera_principal_x, self.camera_principal_y),
                    locked=self._locked_object_id,
                    counts=self.category_counts,
                    class_limits=self.expected_counts,
                )
            self.detection_frames.append(clean)
            self.last_detection_frame_number = frame_number
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().warning(f"ignored malformed /detections message: {error}")

    def _wait_for_camera_info(self):
        deadline = time.monotonic() + self.detection_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.10)
            if self.camera_focal_x is not None:
                self._publish_status(
                    "CAMERA INFO READY | "
                    f"width={self.camera_width} fx={self.camera_focal_x:.2f} "
                    f"cx={self.camera_principal_x:.2f}"
                )
                return
        raise GraspActionError(
            "no /camera/camera_info received; check the Gazebo sensor and bridge"
        )

    def _wait_for_detector_stream(self):
        """Wait for one YOLO result frame after Gazebo has been unpaused."""
        deadline = time.monotonic() + self.detection_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.10)
            if self.last_detection_frame_number >= 0:
                self._publish_status(
                    "DETECTOR STREAM READY | "
                    f"frame={self.last_detection_frame_number}"
                )
                return
        raise GraspActionError(
            "no /detections frames received after Gazebo started running; "
            "check CAMERA FRAME READY and YOLO inference errors"
        )

    def _wait_stable_detection(self, preferred_class=None):
        self.detection_frames.clear()
        deadline = time.monotonic() + self.detection_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.10)
            detection = stable_center_detection(
                list(self.detection_frames),
                self.camera_principal_x,
                required_votes=self.stable_required_frames,
                preferred_class=preferred_class,
                max_center_spread_px=self.stable_center_spread_px,
                selection="leftmost",
            )
            if detection is not None:
                self._publish_status(
                    "VISION LOCK | "
                    f"id={detection.get('object_id', 'unassociated')} | "
                    f"class={detection['class_name']} | "
                    f"confidence={detection['confidence']:.3f} | "
                    f"stable={detection['stable_votes']}/{self.stable_window_frames} | "
                    "tracking=leftmost"
                )
                return detection
        class_text = preferred_class or "bottle/tennis"
        raise GraspActionError(
            f"no stable {class_text} detection within {self.detection_timeout:.1f} s"
        )

    def _align_to_visual_target(self, held_positions):
        """Continuously rotate the chassis while visually centering the target."""
        detection = self._wait_stable_detection()
        target_class = detection["class_name"]
        self._locked_object_id = detection['object_id']
        if self._locked_object_origin is None:
            self._locked_object_origin = self._identity.position(self._locked_object_id)
        alignment_yaw_before = self._identity.yaw()

        # The image angle is diagnostic.  Forward travel is no longer chosen
        # from a time/step table; the measured object and chassis coordinates
        # determine a standoff pose after visual locking.
        initial_target_angle = abs(
            horizontal_angle_radians(
                detection,
                self.camera_focal_x,
                self.camera_principal_x,
            )
        )

        robot_position = self._identity.position('robomaster_ep_core')
        target_position = self._identity.position(self._locked_object_id)
        yaw = self._identity.yaw()
        if robot_position is not None and target_position is not None and yaw is not None:
            bearing = math.atan2(target_position[1]-robot_position[1],
                                 target_position[0]-robot_position[0]) - yaw
            initial_target_angle = abs(math.atan2(math.sin(bearing), math.cos(bearing)))

        self._publish_status(
            "TARGET GEOMETRY LOCKED | "
            f"class={target_class} | "
            f"initial_angle={math.degrees(initial_target_angle):.1f} deg | "
            "approach=measured_pose_standoff"
        )

        def center_x(item):
            box = item["bbox"]
            return (float(box["x1"]) + float(box["x2"])) * 0.5

        last_center_x = center_x(detection)
        filtered_center_x = last_center_x
        current_turn_speed = 0.0
        aligned_frames = 0
        last_frame_number = self.last_detection_frame_number
        log_counter = 0

        self._publish_status(
            f"CONTINUOUS VISUAL ALIGN START | class={target_class}"
        )

        deadline = time.monotonic() + self.detection_timeout

        try:
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.01)

                # Hold the arm continuously while chassis is rotating.
                self._publish(held_positions, repeat=1)

                # Wait for a fresh YOLO frame.
                if self.last_detection_frame_number == last_frame_number:
                    wheel_command = [
                        -current_turn_speed,
                        current_turn_speed,
                        -current_turn_speed,
                        current_turn_speed,
                    ]
                    self._publish_to(
                        self.wheel_publisher,
                        wheel_command,
                        repeat=1,
                    )
                    continue

                last_frame_number = self.last_detection_frame_number

                if not self.detection_frames:
                    continue

                latest = self.detection_frames[-1]

                candidates = [
                    item
                    for item in latest
                    if canonical_class(item.get("class_name", ""))
                    == target_class
                ]

                # If detection is temporarily lost, smoothly slow down.
                if not candidates:
                    aligned_frames = 0
                    current_turn_speed = 0.0

                    if abs(current_turn_speed) < 0.10:
                        current_turn_speed = 0.0

                    wheel_command = [
                        -current_turn_speed,
                        current_turn_speed,
                        -current_turn_speed,
                        current_turn_speed,
                    ]

                    self._publish_to(
                        self.wheel_publisher,
                        wheel_command,
                        repeat=1,
                    )
                    continue

                # Track the same object rather than jumping to another
                # object of the same class.
                detection = min(
                    candidates,
                    key=lambda item: abs(center_x(item) - last_center_x),
                )

                measured_center_x = center_x(detection)
                last_center_x = measured_center_x

                # Low-pass filtering of YOLO bbox center.
                filtered_center_x = (
                    0.70 * filtered_center_x
                    + 0.30 * measured_center_x
                )

                error_px = filtered_center_x - self.camera_principal_x

                if abs(error_px) <= self.alignment_tolerance_px:
                    aligned_frames += 1
                    desired_turn_speed = 0.0
                else:
                    aligned_frames = 0

                    speed = self.alignment_kp * abs(error_px)
                    speed = max(
                        self.alignment_min_speed,
                        min(speed, self.alignment_max_speed),
                    )

                    # Image-right error requires clockwise chassis motion.
                    desired_turn_speed = (
                        -speed if error_px > 0.0 else speed
                    )

                # Smooth acceleration / deceleration.
                alpha = self.alignment_smoothing
                current_turn_speed += alpha * (
                    desired_turn_speed - current_turn_speed
                )

                if abs(current_turn_speed) < 0.05:
                    current_turn_speed = 0.0

                wheel_command = [
                    -current_turn_speed,
                    current_turn_speed,
                    -current_turn_speed,
                    current_turn_speed,
                ]

                self._publish_to(
                    self.wheel_publisher,
                    wheel_command,
                    repeat=1,
                )

                log_counter += 1
                if log_counter % 5 == 0:
                    self.get_logger().info(
                        "CONTINUOUS ALIGN | "
                        f"class={target_class} | "
                        f"error_px={error_px:.1f} | "
                        f"wheel={current_turn_speed:.2f} | "
                        f"stable={aligned_frames}/"
                        f"{self.aligned_required_frames}"
                    )

                if (
                    aligned_frames >= self.aligned_required_frames
                    and abs(current_turn_speed) <= 0.30
                ):
                    self._publish_to(
                        self.wheel_publisher,
                        [0.0] * 4,
                        repeat=10,
                    )
                    self._publish(held_positions, repeat=10)

                    self._publish_status(
                        "TARGET ALIGNED | "
                        f"class={target_class} | "
                        f"error_px={error_px:.1f} | feedback=vision"
                    )

                    return target_class

        finally:
            self._publish_to(
                self.wheel_publisher,
                [0.0] * 4,
                repeat=10,
            )
            self._publish(held_positions, repeat=10)
            self._wait_sim_duration(self.post_turn_settle_duration)
            self._aligned_yaw = self._identity.yaw()
            if self._aligned_yaw is not None and alignment_yaw_before is not None:
                yaw_delta = math.atan2(
                    math.sin(self._aligned_yaw - alignment_yaw_before),
                    math.cos(self._aligned_yaw - alignment_yaw_before),
                )
                self._publish_status(
                    "VISUAL ALIGNMENT WORLD POSE | "
                    f"yaw_delta_deg={math.degrees(yaw_delta):.2f}"
                )

        raise GraspActionError(
            f"unable to continuously center {target_class} "
            f"within {self.alignment_tolerance_px:.1f} px"
        )

    def _arm_pose(self, initial, fraction):
        pose = super()._arm_pose(initial, fraction)
        index = {name: i for i, name in enumerate(POSITION_JOINTS)}
        # The bottle sum is zero, so its experiment2 wrist does not change.
        pitch_delta = self.arm_extend_delta + self.arm_2_delta
        pose[index["endpoint_bracket_joint"]] -= pitch_delta * fraction
        return pose

    @staticmethod
    def _vertical_clearance_pose(source, initial):
        index = {name: i for i, name in enumerate(POSITION_JOINTS)}
        pose = list(source)
        pose[index["arm_1_joint"]] = initial[index["arm_1_joint"]] + 1.020
        pose[index["arm_2_joint"]] = initial[index["arm_2_joint"]] - 0.568
        pose[index["endpoint_bracket_joint"]] = (
            initial[index["endpoint_bracket_joint"]] - 0.452
        )
        return pose

    def _move(self, label, start, target):
        if label == "下降到放置位" and not self._placement_ready:
            raise GraspActionError(
                "refusing to lower: retreat, exact 90 degree turn, and "
                "placement travel have not all been verified"
            )
        if (
            label == "释放后抬臂离开"
            and self._use_tennis_grasp
            and self._active_initial_positions is not None
        ):
            initial = self._active_initial_positions
            index = {name: i for i, name in enumerate(POSITION_JOINTS)}
            clearance = self._vertical_clearance_pose(start, initial)
            retracted = self._gripper_pose(initial, initial, index, opened=True)
            super()._move("释放后先垂直抬高 5 cm", start, clearance)
            super()._move("抬高后向后收回机械臂", clearance, retracted)
            target[:] = retracted
            return
        super()._move(label, start, target)

    def _safe_home_pose(self, initial, current, index):
        if not self._use_tennis_grasp:
            return super()._safe_home_pose(initial, current, index)
        current_open = self._gripper_pose(current, initial, index, opened=True)
        self._move_gripper("安全张开夹爪", current, current_open)
        clearance = self._vertical_clearance_pose(current_open, initial)
        super()._move("异常回收：先垂直抬高", current_open, clearance)
        home_open = self._gripper_pose(initial, initial, index, opened=True)
        super()._move("异常回收：抬高后向后收臂", clearance, home_open)
        self._settle(home_open)
        self._publish_status("SAFE HOME COMPLETE: lifted first, then retracted")
        return home_open

    def _drive_to_pose(
        self,
        held_positions,
        target,
        label,
        allow_reverse=False,
        control_mode="pose",
    ):
        """Run one explicit feedback phase: position, yaw, or full pose."""
        if not isinstance(target, Pose2D):
            raise GraspActionError(f"{label}: target is not a Pose2D")
        if control_mode not in {"position", "yaw", "pose"}:
            raise GraspActionError(f"{label}: invalid control mode {control_mode}")
        starting_pose = self._identity.pose2d()
        if starting_pose is None:
            raise GraspActionError(f"{label}: measured chassis pose is unavailable")
        self._publish_status(
            f"POSE TARGET | label={label} | mode={control_mode} | "
            f"from=({starting_pose.x:.4f},{starting_pose.y:.4f},"
            f"{math.degrees(starting_pose.yaw):.2f}deg) | "
            f"to=({target.x:.4f},{target.y:.4f},"
            f"{math.degrees(target.yaw):.2f}deg)"
        )
        self.pose_controller.reset()
        deadline = time.monotonic() + self.pose_motion_timeout
        last_valid_pose = time.monotonic()
        previous_phase = None
        log_counter = 0
        try:
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=self.pose_control_period)
                current = self._identity.pose2d()
                if current is None:
                    self._publish_to(self.wheel_publisher, [0.0] * 4, repeat=1)
                    if time.monotonic() - last_valid_pose > 2.0:
                        raise GraspActionError(
                            f"{label}: measured chassis pose became stale"
                        )
                    continue
                last_valid_pose = time.monotonic()
                if control_mode == "position":
                    output = self.pose_controller.compute_position(
                        current,
                        target,
                        allow_reverse=allow_reverse,
                    )
                elif control_mode == "yaw":
                    output = self.pose_controller.compute_yaw(current, target)
                else:
                    output = self.pose_controller.compute(
                        current,
                        target,
                        allow_reverse=allow_reverse,
                    )
                self._publish(held_positions, repeat=1)
                self._publish_to(
                    self.wheel_publisher, list(output.wheels), repeat=1
                )
                log_counter += 1
                if output.phase != previous_phase or log_counter % 20 == 0:
                    self.get_logger().info(
                        f"POSE FEEDBACK | label={label} | phase={output.phase} | "
                        f"position_error={output.position_error:.4f} m | "
                        f"yaw_error={math.degrees(output.yaw_error):.2f} deg"
                    )
                    previous_phase = output.phase
                if output.reached:
                    self._publish_status(
                        f"POSE REACHED | label={label} | "
                        f"position_error={output.position_error:.4f} m | "
                        f"yaw_error={math.degrees(output.yaw_error):.2f} deg"
                    )
                    return current
        finally:
            self._publish_to(self.wheel_publisher, [0.0] * 4, repeat=10)
            self._publish(held_positions, repeat=10)
        raise GraspActionError(
            f"{label}: pose target did not converge within "
            f"{self.pose_motion_timeout:.1f} s"
        )

    def _approach_locked_object(self, held_positions):
        self.task_state.transition(SortingState.APPROACH)
        current = self._identity.pose2d()
        target_position = self._identity.position(self._locked_object_id)
        if current is None or target_position is None:
            raise GraspActionError("cannot compute grasp standoff from stale pose")
        standoff = (
            self.tennis_grasp_standoff
            if self._current_class == TENNIS
            else self.bottle_grasp_standoff
        )
        target = standoff_pose(
            current,
            float(target_position[0]),
            float(target_position[1]),
            standoff,
        )
        self._drive_to_pose(
            held_positions,
            target,
            f"APPROACH POSITION {self._locked_object_id} AT {standoff:.3f} M",
            control_mode="position",
        )
        self._drive_to_pose(
            held_positions,
            target,
            f"FACE LOCKED OBJECT {self._locked_object_id}",
            control_mode="yaw",
        )
        measured = self._identity.pose2d()
        object_after = self._identity.position(self._locked_object_id)
        if measured is None or object_after is None:
            raise GraspActionError("pose feedback lost after grasp approach")
        remaining = math.hypot(
            float(object_after[0]) - measured.x,
            float(object_after[1]) - measured.y,
        )
        if abs(remaining - standoff) > 0.025:
            raise GraspActionError(
                f"grasp standoff verification failed: {remaining:.3f} m"
            )
        self._publish_status(
            f"GRASP STANDOFF VERIFIED | distance={remaining:.3f} m"
        )

    def _rotate_chassis(self, held_positions):
        self._placement_ready = False
        if self._current_class not in (BOTTLE, TENNIS):
            raise GraspActionError("placement requested without a vision class")
        if self._cycle_pose is None:
            raise GraspActionError("placement requested without a fresh cycle pose")
        cycle_pose = self._cycle_pose

        # First carry the grasped object backwards along the clear approach
        # corridor.  Do not sweep the arm through the remaining objects.
        self.task_state.transition(SortingState.RETREAT)
        self._publish_status("RETREAT BEFORE CLASSIFICATION | position phase")
        self._drive_to_pose(
            held_positions,
            cycle_pose,
            "RETREAT TO CYCLE ORIGIN POSITION WITH OBJECT",
            allow_reverse=True,
            control_mode="position",
        )
        self._drive_to_pose(
            held_positions,
            cycle_pose,
            "RESTORE CYCLE HEADING WITH OBJECT",
            control_mode="yaw",
        )

        # World +yaw is left.  Bottle marker is on robot-right (-y), tennis
        # marker is on robot-left (+y).
        offset = -self.placement_yaw if self._current_class == BOTTLE else self.placement_yaw
        target_yaw = wrap_angle(cycle_pose.yaw + offset)
        zone = (
            "RIGHT BOTTLE ZONE"
            if self._current_class == BOTTLE
            else "LEFT TENNIS ZONE"
        )
        self.task_state.transition(SortingState.PLACE, zone)
        self._drive_to_pose(
            held_positions,
            Pose2D(cycle_pose.x, cycle_pose.y, target_yaw),
            f"TURN EXACTLY 90 DEG TO {zone}",
            control_mode="yaw",
        )
        measured = self._identity.pose2d()
        if measured is None:
            raise GraspActionError("pose feedback lost after exact 90 degree turn")
        actual_offset = wrap_angle(measured.yaw - cycle_pose.yaw)
        if abs(wrap_angle(actual_offset - offset)) > self.pose_controller.config.yaw_tolerance:
            raise GraspActionError(
                "classification turn did not settle at exactly 90 degrees"
            )
        self._publish_status(
            "EXACT CLASSIFICATION TURN VERIFIED | "
            f"delta={math.degrees(actual_offset):.2f} deg"
        )

        # Use the measured post-turn position as the start of the straight
        # placement leg. In-place skid steering can translate a few cm, and
        # forcing x/y back to the pre-turn point would undo the 90-degree turn.
        placement_pose = Pose2D(
            measured.x + self.placement_distance * math.cos(target_yaw),
            measured.y + self.placement_distance * math.sin(target_yaw),
            target_yaw,
        )
        self._drive_to_pose(
            held_positions,
            placement_pose,
            f"DRIVE {self.placement_distance:.3f} M TO {zone}",
            control_mode="position",
        )
        placement_measured = self._identity.pose2d()
        if placement_measured is None:
            raise GraspActionError("pose feedback lost at placement zone")
        self._drive_to_pose(
            held_positions,
            Pose2D(placement_measured.x, placement_measured.y, target_yaw),
            f"SET EXACT 90 DEG HEADING AT {zone}",
            control_mode="yaw",
        )
        self._placement_ready = True
        self._publish_status(
            "PLACEMENT MOTION VERIFIED | lowering and release now permitted"
        )

    def _run_once(self, attempt, initial, current, index):
        self._active_initial_positions = list(initial)
        self._placement_ready = False
        self.task_state.transition(SortingState.GRASP)
        result = super()._run_once(attempt, initial, current, index)
        self.task_state.transition(SortingState.VERIFY)
        return result

    def _prepare_observation_pose(self, initial, current, index):
        observation = self._gripper_pose(initial, initial, index, opened=True)
        if any(abs(a - b) > 1.0e-6 for a, b in zip(current, observation)):
            self._move("机械臂回到固定摄像观察姿态", current, observation)
            self._settle(observation)
        return observation

    def _record_unrecognized_or_empty(self, processed_count):
        """Handle a full detection timeout without inventing a class label."""
        active = self._identity.active_task_objects()
        if active:
            # The entity name is class-neutral.  Pose data establishes only
            # occupancy; the category remains unknown because YOLO supplied no
            # stable label.  Exclude one slot and continue safely.
            robot = self._identity.pose2d()
            if robot is None:
                raise GraspActionError(
                    "cannot handle unrecognized object with stale pose feedback"
                )

            def scan_order(name):
                position = self._identity.position(name)
                if position is None:
                    return float("inf")
                bearing = math.atan2(
                    float(position[1]) - robot.y,
                    float(position[0]) - robot.x,
                ) - robot.yaw
                # Camera-left first, matching the normal detector policy.
                return -math.atan2(math.sin(bearing), math.cos(bearing))

            object_id = min(active, key=scan_order)
            self._identity.completed.add(object_id)
            self._publish_status(
                "EXCEPTION UNRECOGNIZED | "
                f"slot={object_id} | action=skip | chassis=stopped"
            )
            return 1, 1, 0

        empty_slots = max(0, self.run_count - int(processed_count))
        if empty_slots:
            self._publish_status(
                "EXCEPTION EMPTY GRID | "
                f"count={empty_slots} | action=skip | chassis=stopped"
            )
        return empty_slots, 0, empty_slots

    def _restore_world_heading(self, held_positions, target_yaw):
        if target_yaw is None:
            return
        current = self._identity.pose2d()
        if current is None:
            raise GraspActionError('world heading feedback lost during restoration')
        self._drive_to_pose(
            held_positions,
            Pose2D(current.x, current.y, float(target_yaw)),
            'RESTORE WORLD HEADING',
            control_mode="yaw",
        )

    def _restore_cycle_offsets(self, held_positions):
        if self.task_state.state != SortingState.SAFE_STOP:
            self.task_state.transition(SortingState.RETURN_HOME)
        self._set_vision_enabled(False, "returning to cycle center")
        self._publish_status("RESTORE CYCLE CENTER | separate position/yaw phases")
        try:
            if self._cycle_pose is None:
                raise GraspActionError("cycle start pose was not captured")
            self._drive_to_pose(
                held_positions,
                self._cycle_pose,
                "RETURN TO CYCLE X Y POSITION",
                allow_reverse=True,
                control_mode="position",
            )
            self._drive_to_pose(
                held_positions,
                self._cycle_pose,
                "RETURN TO CYCLE YAW",
                control_mode="yaw",
            )
        finally:
            self._cycle_pose = None
            self.detection_frames.clear()

    def run_action(self):
        if shutil.which("ros2") is None or shutil.which("ign") is None:
            raise GraspActionError("ros2 or ign not found; source the Team21 environment")

        self.get_logger().info(
            f"VISION SORTING | mode={self.scene_mode} | "
            f"objects={self.run_count} | "
            "bottle=right | tennis=left | chassis=world-pose feedback"
        )
        self._ensure_broadcaster()
        initial = self._wait_for_joint_state()
        self._ensure_hold_controller(initial)
        self._ensure_wheel_controller()

        self._publish(initial, repeat=10)
        self._publish_to(self.wheel_publisher, [0.0] * 4, repeat=10)
        previous_sim_time = self.latest_sim_time
        self._publish_status("PREP: START CONTINUOUS GAZEBO EXECUTION")
        self._set_world_paused(False)
        self._wait_for_sim_time(after=previous_sim_time)
        self._write_ready_file(self.controller_ready_file)
        self._publish_status("ROBOT CONTROLLERS READY | continuous dynamic mode")
        self.task_state.transition(SortingState.WAIT_SCENE)

        index = {name: i for i, name in enumerate(POSITION_JOINTS)}
        current = self._gripper_pose(initial, initial, index, opened=True)
        self._move_gripper("开始分类前张开夹爪", initial, current)
        self._settle(current)
        self._wait_for_scene_ready()
        self._wait_for_camera_info()
        self._identity = SortingIdentity(
            Path(__file__).resolve().parents[1]
            / 'src/robomaster_pick_place_sim/urdf/robomaster_ep_static.urdf'
        )
        deadline = time.monotonic() + 15.0
        while not self._identity.fresh() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not self._identity.fresh():
            raise GraspActionError('Gazebo object identity poses unavailable; refusing untracked sorting')
        self._publish_status('OBJECT IDENTITY READY | completed entities excluded from all detections')
        self._set_vision_enabled(True, "initial detector handshake")
        self._wait_for_detector_stream()
        self._set_vision_enabled(False, "waiting for first acquire phase")
        self.task_state.transition(SortingState.ACQUIRE)

        success_count = 0
        unknown_count = 0
        empty_count = 0
        processed_count = 0
        try:
            while processed_count < self.run_count:
                completed = False
                for retry in range(self.max_grasp_retries + 1):
                    current = self._prepare_observation_pose(initial, current, index)
                    self._cycle_pose = self._identity.pose2d()
                    if self._cycle_pose is None:
                        raise GraspActionError(
                            "cannot start cycle without a fresh chassis pose"
                        )
                    self._set_vision_enabled(
                        True,
                        f"acquire and align cycle {processed_count + 1}",
                    )
                    self.task_state.transition(SortingState.ALIGN)
                    try:
                        self._current_class = self._align_to_visual_target(current)
                    except GraspActionError as error:
                        self._set_vision_enabled(False, "acquire phase ended")
                        if not str(error).startswith("no stable"):
                            raise
                        consumed, unknown, empty = (
                            self._record_unrecognized_or_empty(processed_count)
                        )
                        if consumed <= 0:
                            raise
                        processed_count += consumed
                        unknown_count += unknown
                        empty_count += empty
                        self._cycle_pose = None
                        self.task_state.transition(SortingState.RECORD)
                        completed = True
                        break
                    self._set_vision_enabled(
                        False,
                        f"locked {self._locked_object_id}; motion phases own control",
                    )
                    self._use_tennis_grasp = self._current_class == TENNIS
                    self.arm_2_delta = (
                        self.tennis_arm_2_delta
                        if self._use_tennis_grasp
                        else self.bottle_arm_2_delta
                    )
                    mode = (
                        "LOW LEVEL HORIZONTAL TENNIS GRASP"
                        if self._use_tennis_grasp
                        else "ORIGINAL EXPERIMENT2 BOTTLE GRASP"
                    )
                    self._publish_status(
                        f"OBJECT {success_count + 1}/{self.run_count} | retry={retry}/"
                        f"{self.max_grasp_retries} | class={self._current_class} | {mode}"
                    )
                    self._approach_locked_object(current)

                    ok, current = self._run_once(
                        success_count + 1, initial, current, index
                    )
                    current = list(self.last_position_command or current)
                    if ok:
                        placed = self._identity.position(self._locked_object_id)
                        origin = self._locked_object_origin
                        moved = (placed is not None and origin is not None and
                                 math.hypot(placed[0]-origin[0], placed[1]-origin[1]) >= 0.08)
                        if not moved:
                            ok = False
                            self._publish_status('PLACEMENT NOT VERIFIED | target did not leave source; no count increment')
                        else:
                            self._identity.completed.add(self._locked_object_id)
                            self._publish_status(f'OBJECT EXCLUDED | id={self._locked_object_id} | class={self._current_class}')
                    self._restore_cycle_offsets(current)
                    self.task_state.transition(SortingState.RECORD)
                    if ok:
                        self.category_counts[self._current_class] += 1
                        success_count += 1
                        processed_count += 1
                        completed = True
                        self._locked_object_id = None
                        self._locked_object_origin = None
                        self.detection_frames.clear()
                        self._publish_status(
                            "SORTED | "
                            f"processed={processed_count}/{self.run_count} | "
                            f"sorted={success_count} | bottles="
                            f"{self.category_counts[BOTTLE]}/"
                            f"{self.expected_counts[BOTTLE]} | tennis="
                            f"{self.category_counts[TENNIS]}/"
                            f"{self.expected_counts[TENNIS]}"
                        )
                        break
                    self._publish_status(
                        f"EMPTY GRASP | retry {retry + 1}/"
                        f"{self.max_grasp_retries + 1}"
                    )
                    if retry < self.max_grasp_retries:
                        self.task_state.transition(
                            SortingState.ACQUIRE, "retry same locked target"
                        )
                if not completed:
                    raise GraspActionError(
                        "same target failed after two retries; refusing to guess or continue"
                    )
                if processed_count < self.run_count:
                    self.task_state.transition(SortingState.ACQUIRE)

            recognized_categories = sum(
                count > 0 for count in self.category_counts.values()
            )
            if success_count < self.minimum_success_count:
                raise GraspActionError(
                    f"only {success_count}/{self.run_count} objects were sorted; "
                    f"minimum is {self.minimum_success_count}"
                )
            if self.require_two_categories and recognized_categories < 2:
                raise GraspActionError(
                    f"fewer than two categories were sorted: {self.category_counts}"
                )
            self._publish_status(
                "VISION SORTING COMPLETE | "
                f"sorted={success_count}/{self.run_count} | "
                f"bottles={self.category_counts[BOTTLE]} right | "
                f"tennis={self.category_counts[TENNIS]} left | "
                f"unknown={unknown_count} | empty={empty_count} | "
                "acceptance target reached | Gazebo pausing"
            )
            self._set_vision_enabled(False, "sorting complete")
            self.task_state.transition(SortingState.DONE)
        except Exception as task_error:
            self._set_vision_enabled(False, "task failure and safe recovery")
            self.task_state.fail(task_error)
            current = list(self.last_position_command or current)
            try:
                current = self._safe_home_pose(initial, current, index)
            except Exception as recovery_error:
                self.get_logger().warning(f"arm recovery incomplete: {recovery_error}")
            try:
                self._restore_cycle_offsets(current)
            except Exception as recovery_error:
                self.get_logger().warning(f"chassis recovery incomplete: {recovery_error}")
            raise


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    exit_code = 0
    try:
        node = GraspBottleTennisAction()
        node.run_action()
    except KeyboardInterrupt:
        exit_code = 130
    except Exception as error:
        exit_code = 1
        if node is not None:
            node.get_logger().error(str(error))
        else:
            print(f"vision sorting failed to start: {error}")
    finally:
        if node is not None:
            if node._identity is not None:
                node._identity.close()
            node.safe_stop()
            node.close_log_file()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
