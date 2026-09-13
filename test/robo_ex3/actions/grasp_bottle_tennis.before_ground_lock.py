#!/usr/bin/env python3
"""Vision-guided four-object sorting with the two calibrated EP1 grasps."""

from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
import shutil
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from sorting_identity import SortingIdentity

from grasp_cube import GraspActionError, GraspCubeAction, POSITION_JOINTS
from vision_common import (
    BOTTLE,
    TENNIS,
    angle_to_steps,
    canonical_class,
    horizontal_angle_radians,
    placement_step,
    stable_center_detection,
)


class GraspBottleTennisAction(GraspCubeAction):
    """Sort four objects using camera classification, never spawn metadata."""

    def __init__(self):
        super().__init__()
        self.run_count = 4
        self.spawn_test_cube = False

        # Preserve the two user-calibrated grasp modes.
        self.bottle_arm_2_delta = self.arm_2_delta
        self.tennis_arm_2_delta = -0.78
        self._use_tennis_grasp = False
        self.turn_steps = 2500

        # The outer ±30-degree slots need about 11-12 seconds at the current
        # filtered YOLO frame rate and smooth wheel acceleration.
        self.declare_parameter("detection_timeout_seconds", 15.0)
        self.declare_parameter("stable_window_frames", 5)
        self.declare_parameter("stable_required_frames", 4)
        self.declare_parameter("stable_center_spread_px", 35.0)
        self.declare_parameter("alignment_tolerance_px", 12.0)
        # Jetson live calibration: use half the original angular gain.  The
        # camera image moves opposite to the chassis yaw command, so visual
        # corrections invert the geometric image angle below.
        self.declare_parameter("alignment_steps_per_radian", 1000.0)
        self.declare_parameter("max_alignment_corrections", 8)
        self.declare_parameter("alignment_kp", 0.02)
        self.declare_parameter("alignment_min_speed", 0.4)
        self.declare_parameter("alignment_max_speed", 3.0)
        self.declare_parameter("alignment_smoothing", 0.35)
        self.declare_parameter("aligned_required_frames", 3)
        self.declare_parameter("max_grasp_retries", 2)
        self.declare_parameter("minimum_wheel_command_seconds", 0.10)
        self.declare_parameter("forward_approach_steps", 333)
        self.declare_parameter("bottle_place_steps", [2350, 2500, 2650])
        self.declare_parameter("tennis_place_steps", [2350, 2500, 2650])
        self.declare_parameter("controller_ready_file", "")
        self.declare_parameter("scene_ready_file", "")
        self.declare_parameter("scene_ready_timeout_seconds", 60.0)

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
        self.alignment_steps_per_radian = float(
            self.get_parameter("alignment_steps_per_radian").value
        )
        self.max_alignment_corrections = int(
            self.get_parameter("max_alignment_corrections").value
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
        self.minimum_wheel_command_duration = float(
            self.get_parameter("minimum_wheel_command_seconds").value
        )
        self.forward_approach_steps = int(
            self.get_parameter("forward_approach_steps").value
        )
        self.bottle_place_steps = list(
            self.get_parameter("bottle_place_steps").value
        )
        self.tennis_place_steps = list(
            self.get_parameter("tennis_place_steps").value
        )
        self.controller_ready_file = str(
            self.get_parameter("controller_ready_file").value
        )
        self.scene_ready_file = str(self.get_parameter("scene_ready_file").value)
        self.scene_ready_timeout = float(
            self.get_parameter("scene_ready_timeout_seconds").value
        )

        if not self.stable_window_frames >= self.stable_required_frames >= 1:
            raise GraspActionError("stable window must be >= required frames >= 1")
        if min(
            self.detection_timeout,
            self.stable_center_spread_px,
            self.alignment_tolerance_px,
            self.alignment_steps_per_radian,
            self.minimum_wheel_command_duration,
        ) <= 0.0:
            raise GraspActionError("vision timing and alignment values must be positive")
        if min(
            self.max_alignment_corrections,
            self.max_grasp_retries + 1,
            self.forward_approach_steps,
        ) < 1:
            raise GraspActionError("vision retries and forward steps are invalid")
        for sequence in (self.bottle_place_steps, self.tennis_place_steps):
            placement_step(sequence, 0)

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

        self.category_counts = {BOTTLE: 0, TENNIS: 0}
        self._current_class = None
        self._alignment_steps = 0
        self._placement_relative_steps = 0
        self._forward_offset_active = False
        self._active_initial_positions = None
        self._ignore_detections = False
        self._cycle_wheel_positions = None
        self._identity = None
        self._locked_object_id = None
        self._locked_object_origin = None
        self._cycle_yaw = None
        self._aligned_yaw = None
        self.camera_focal_y = None
        self.camera_principal_y = None

    @staticmethod
    def _write_ready_file(path_text):
        if path_text:
            path = Path(path_text)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("ready\n", encoding="utf-8")

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
                if task_class is None or not all(
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
                    locked=self._locked_object_id, counts=self.category_counts,
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

    def _wheel_motion_profile(self, steps):
        """Preserve calibrated travel while respecting the 100 Hz control loop."""
        nominal_duration = (
            abs(int(steps)) * self.physics_step_seconds / self.speed_scale
        )
        duration = max(nominal_duration, self.minimum_wheel_command_duration)
        command_speed = self.wheel_speed * nominal_duration / duration
        return command_speed, duration

    def _wheel_positions(self):
        if self.latest_joint_state is None:
            return None
        values = dict(
            zip(self.latest_joint_state.name, self.latest_joint_state.position)
        )
        names = (
            "front_left_wheel_joint",
            "front_right_wheel_joint",
            "rear_left_wheel_joint",
            "rear_right_wheel_joint",
        )
        if not all(name in values for name in names):
            return None
        return tuple(float(values[name]) for name in names)

    def _turn_steps_from_wheel_positions(self, before, after):
        """Convert measured counter-rotating wheel travel to signed turn steps."""
        if before is None or after is None:
            return None
        deltas = tuple(end - start for start, end in zip(before, after))
        signed_wheel_delta = (-deltas[0] + deltas[1] - deltas[2] + deltas[3]) * 0.25
        radians_per_step = (
            self.wheel_speed * self.physics_step_seconds / self.speed_scale
        )
        if radians_per_step <= 0.0:
            return None
        return int(round(signed_wheel_delta / radians_per_step))

    def _rotate_by_steps(self, held_positions, signed_steps, label):
        signed_steps = int(signed_steps)
        if signed_steps == 0:
            return
        direction = 1.0 if signed_steps > 0 else -1.0
        command_speed, duration = self._wheel_motion_profile(signed_steps)
        wheel_command = [
            -direction * command_speed,
            direction * command_speed,
            -direction * command_speed,
            direction * command_speed,
        ]
        self._publish_status(
            f"{label} | signed_steps={signed_steps} | "
            f"wheel_speed={command_speed:.3f} | sim_duration={duration:.3f}s"
        )
        positions_before = self._wheel_positions()
        try:
            def keep_turning(ratio):
                if ratio >= 1.0:
                    return
                self._publish(held_positions, repeat=1)
                self._publish_to(self.wheel_publisher, wheel_command, repeat=1)

            self._wait_sim_duration(duration, callback=keep_turning)
        finally:
            self._publish_to(self.wheel_publisher, [0.0] * 4, repeat=10)
            self._publish(held_positions, repeat=10)
        self._wait_sim_duration(self.post_turn_settle_duration)
        positions_after = self._wheel_positions()
        if positions_before is not None and positions_after is not None:
            deltas = tuple(
                after - before
                for before, after in zip(positions_before, positions_after)
            )
            self._publish_status(
                "WHEEL MOTION FEEDBACK | joint_delta="
                + ",".join(f"{value:.4f}" for value in deltas)
            )

    def _align_to_visual_target(self, held_positions):
        """Continuously rotate the chassis while visually centering the target."""
        detection = self._wait_stable_detection()
        target_class = detection["class_name"]
        self._locked_object_id = detection['object_id']
        if self._locked_object_origin is None:
            self._locked_object_origin = self._identity.position(self._locked_object_id)
        alignment_positions_before = self._wheel_positions()

        # 根据目标最初相对摄像头的角度决定最终向前插入距离。
        # 中间槽位约 ±10° -> 1 cm；外侧槽位约 ±30° -> 2 cm。
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

        if math.degrees(initial_target_angle) < 20.0:
            self.final_insert_steps = 67      # 约 1 cm
            insert_distance = "1 cm (middle slot)"
        else:
            self.final_insert_steps = 133     # 约 2 cm
            insert_distance = "2 cm (outer slot)"

        self._publish_status(
            "FINAL INSERT SELECT | "
            f"class={target_class} | "
            f"initial_angle={math.degrees(initial_target_angle):.1f} deg | "
            f"distance={insert_distance} | "
            f"steps={self.final_insert_steps}"
        )

        def center_x(item):
            box = item["bbox"]
            return (float(box["x1"]) + float(box["x2"])) * 0.5

        last_center_x = center_x(detection)
        filtered_center_x = last_center_x
        current_turn_speed = 0.0
        aligned_frames = 0
        last_frame_number = self.last_detection_frame_number
        last_sim_time = self.latest_sim_time
        log_counter = 0

        self._alignment_steps = 0

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

                    # Keep the same steering direction as the old:
                    # steps = -angle_to_steps(...)
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

                # Convert continuous rotation back into the old calibrated
                # signed-step representation so placement/restoration works.
                current_sim_time = self.latest_sim_time

                if (
                    last_sim_time is not None
                    and current_sim_time is not None
                    and current_sim_time > last_sim_time
                    and abs(current_turn_speed) > 0.01
                ):
                    dt = current_sim_time - last_sim_time

                    step_rate = (
                        current_turn_speed
                        / self.wheel_speed
                        * self.speed_scale
                        / self.physics_step_seconds
                    )

                    self._alignment_steps += int(
                        round(step_rate * dt)
                    )

                last_sim_time = current_sim_time

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
                        f"error_px={error_px:.1f} | "
                        f"total_steps={self._alignment_steps}"
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
            measured_steps = self._turn_steps_from_wheel_positions(
                alignment_positions_before,
                self._wheel_positions(),
            )
            if measured_steps is not None:
                self._alignment_steps = measured_steps
                self._publish_status(
                    "VISUAL ALIGNMENT ODOMETRY | "
                    f"measured_steps={self._alignment_steps}"
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

    def _move_chassis_linear(self, held_positions, forward):
        direction = 1.0 if forward else -1.0
        command_speed, duration = self._wheel_motion_profile(
            self.forward_approach_steps
        )
        wheel_command = [direction * command_speed] * 4
        label = "FORWARD 5 CM" if forward else "RESTORE BACKWARD 5 CM"
        self._publish_status(f"TENNIS CHASSIS {label}")
        try:
            def keep_moving(ratio):
                if ratio >= 1.0:
                    return
                self._publish(held_positions, repeat=1)
                self._publish_to(self.wheel_publisher, wheel_command, repeat=1)

            self._wait_sim_duration(duration, callback=keep_moving)
        finally:
            self._publish_to(self.wheel_publisher, [0.0] * 4, repeat=10)
            self._publish(held_positions, repeat=10)
        self._wait_sim_duration(self.post_turn_settle_duration)

    def _rotate_chassis(self, held_positions):
        if self._current_class not in (BOTTLE, TENNIS):
            raise GraspActionError("placement requested without a vision class")
        sequence = (
            self.bottle_place_steps
            if self._current_class == BOTTLE
            else self.tennis_place_steps
        )
        magnitude = placement_step(
            sequence, self.category_counts[self._current_class]
        )
        # Positive is robot-right and negative is robot-left.
        # Bottle -> robot right; Tennis -> robot left.
        desired_absolute = -magnitude if self._current_class == BOTTLE else magnitude
        self._placement_relative_steps = desired_absolute - self._alignment_steps
        zone = (
            "RIGHT BOTTLE ZONE"
            if self._current_class == BOTTLE
            else "LEFT TENNIS ZONE"
        )
        self._rotate_by_steps(
            held_positions,
            self._placement_relative_steps,
            f"TURN TO {zone}",
        )

    def _run_once(self, attempt, initial, current, index):
        self._active_initial_positions = list(initial)
        return super()._run_once(attempt, initial, current, index)

    def _prepare_observation_pose(self, initial, current, index):
        observation = self._gripper_pose(initial, initial, index, opened=True)
        if any(abs(a - b) > 1.0e-6 for a, b in zip(current, observation)):
            self._move("机械臂回到固定摄像观察姿态", current, observation)
            self._settle(observation)
        return observation

    def _restore_world_heading(self, held_positions, target_yaw):
        if target_yaw is None:
            return
        for correction in range(40):
            yaw = self._identity.yaw()
            if yaw is None:
                raise GraspActionError('world heading feedback lost during restoration')
            error = math.atan2(math.sin(target_yaw-yaw), math.cos(target_yaw-yaw))
            if abs(error) <= math.radians(1.0):
                self._publish_status(f'WORLD HEADING RESTORED | error_deg={math.degrees(error):.2f}')
                return
            steps = int(round(max(-250, min(250, error*1600))))
            self._rotate_by_steps(held_positions, steps,
                                  f'WORLD HEADING RESTORE | error_deg={math.degrees(error):.2f}')
        raise GraspActionError('actual world heading did not converge; no new target selected')

    def _restore_cycle_offsets(self, held_positions):
        self._ignore_detections = True
        self.detection_frames.clear()
        self._publish_status("RESTORE CYCLE OFFSETS | vision detections ignored")
        try:
            if self._placement_relative_steps:
                self._rotate_by_steps(
                    held_positions,
                    -self._placement_relative_steps,
                    "RESTORE HEADING AFTER PLACEMENT",
                )
                self._placement_relative_steps = 0
            if self._forward_offset_active:
                # Undo translation along the same actual heading used on approach.
                # Encoder reversal alone can leave the chassis facing the zone.
                self._restore_world_heading(held_positions, self._aligned_yaw)
                self._move_chassis_linear(held_positions, forward=False)
                self._forward_offset_active = False
            if self._alignment_steps:
                self._rotate_by_steps(
                    held_positions,
                    -self._alignment_steps,
                    "RESTORE CAMERA OBSERVATION HEADING",
                )
                self._alignment_steps = 0
            for correction in range(2):
                residual_steps = self._turn_steps_from_wheel_positions(
                    self._cycle_wheel_positions,
                    self._wheel_positions(),
                )
                if residual_steps is None or abs(residual_steps) <= 1:
                    break
                self._rotate_by_steps(
                    held_positions,
                    -residual_steps,
                    f"RESTORE ENCODER HEADING RESIDUAL {correction + 1}/2",
                )
            final_residual = self._turn_steps_from_wheel_positions(
                self._cycle_wheel_positions,
                self._wheel_positions(),
            )
            if final_residual is not None:
                self._publish_status(
                    "RESTORE HEADING COMPLETE | "
                    f"encoder_residual_steps={final_residual}"
                )
            self._restore_world_heading(held_positions, self._cycle_yaw)
        finally:
            self._cycle_wheel_positions = None
            self.detection_frames.clear()
            self._ignore_detections = False

    def run_action(self):
        if shutil.which("ros2") is None or shutil.which("ign") is None:
            raise GraspActionError("ros2 or ign not found; source the Team21 environment")

        self.get_logger().info(
            "VISION SORTING | four neutral objects | bottle=right | tennis=left"
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
        self._wait_for_detector_stream()

        success_count = 0
        try:
            while success_count < self.run_count:
                completed = False
                for retry in range(self.max_grasp_retries + 1):
                    current = self._prepare_observation_pose(initial, current, index)
                    self._cycle_wheel_positions = self._wheel_positions()
                    self._cycle_yaw = self._identity.yaw()
                    self._current_class = self._align_to_visual_target(current)
                    self._use_tennis_grasp = self._current_class == TENNIS
                    self.arm_2_delta = (
                        self.tennis_arm_2_delta
                        if self._use_tennis_grasp
                        else self.bottle_arm_2_delta
                    )
                    self._placement_relative_steps = 0
                    mode = (
                        "LOW LEVEL HORIZONTAL TENNIS GRASP"
                        if self._use_tennis_grasp
                        else "ORIGINAL EXPERIMENT2 BOTTLE GRASP"
                    )
                    self._publish_status(
                        f"OBJECT {success_count + 1}/{self.run_count} | retry={retry}/"
                        f"{self.max_grasp_retries} | class={self._current_class} | {mode}"
                    )
                    if self._use_tennis_grasp:
                        self._move_chassis_linear(current, forward=True)
                        self._forward_offset_active = True

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
                    if ok:
                        self.category_counts[self._current_class] += 1
                        success_count += 1
                        completed = True
                        self._locked_object_id = None
                        self._locked_object_origin = None
                        self.detection_frames.clear()
                        self._publish_status(
                            "SORTED | "
                            f"total={success_count}/{self.run_count} | bottles="
                            f"{self.category_counts[BOTTLE]}/2 | tennis="
                            f"{self.category_counts[TENNIS]}/2"
                        )
                        break
                    self._publish_status(
                        f"EMPTY GRASP | retry {retry + 1}/"
                        f"{self.max_grasp_retries + 1}"
                    )
                if not completed:
                    raise GraspActionError(
                        "same target failed after two retries; refusing to guess or continue"
                    )

            if self.category_counts != {BOTTLE: 2, TENNIS: 2}:
                raise GraspActionError(
                    f"unexpected final class counts: {self.category_counts}"
                )
            self._publish_status(
                "VISION SORTING COMPLETE | bottles=2 right | tennis=2 left | Gazebo pausing"
            )
        except Exception:
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
