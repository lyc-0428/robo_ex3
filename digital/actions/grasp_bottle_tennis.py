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
    wrap_angle,
)
from task_state_machine import SortingState, SortingStateMachine
from placement_zone import (
    in_classification_half,
    lateral_offset,
    placement_distance_for_attempt,
)
from gripper_staging import (
    is_tip_first_closing,
    tennis_closed_pose,
    tennis_open_pose,
    tip_first_pose,
)

from grasp_cube import GraspActionError, GraspCubeAction, POSITION_JOINTS
from vision_common import (
    BOTTLE,
    TENNIS,
    canonical_class,
    horizontal_angle_radians,
    stable_center_detection,
)


class MissedGraspError(GraspActionError):
    """The locked object did not rise with the gripper."""


class RetryableAcquisitionError(GraspActionError):
    """Vision or approach geometry requires a return-home reacquisition."""


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
        self.declare_parameter("aligned_required_frames", 1)
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
        self.declare_parameter("minimum_grasp_approach_m", 0.040)
        self.declare_parameter("grasp_approach_arrival_tolerance_m", 0.012)
        self.declare_parameter("grasp_lift_minimum_m", 0.012)
        self.declare_parameter("grasp_lift_verification_timeout_seconds", 1.50)
        self.declare_parameter("cycle_return_tolerance_m", 0.050)
        # Curl the two front pads before moving the gripper roots.  This makes
        # a shallow retaining lip in front of a round ball, then closes the
        # main jaws slowly so the ball cannot be squeezed straight forward.
        self.declare_parameter("tennis_tip_preclose_offset_rad", 0.28)
        self.declare_parameter("tennis_tip_preclose_duration_seconds", 0.90)
        self.declare_parameter("tennis_tip_guard_pause_seconds", 0.35)
        self.declare_parameter("tennis_root_close_duration_seconds", 3.20)
        self.declare_parameter("placement_yaw_degrees", 90.0)
        self.declare_parameter("placement_distance_m", 0.850)
        self.declare_parameter("placement_distance_decrement_m", 0.120)
        self.declare_parameter("placement_minimum_distance_m", 0.250)
        self.declare_parameter("placement_arrival_tolerance_m", 0.050)
        self.declare_parameter("classification_zone_margin_m", 0.200)
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
        self.minimum_grasp_approach = float(
            self.get_parameter("minimum_grasp_approach_m").value
        )
        self.grasp_approach_arrival_tolerance = float(
            self.get_parameter("grasp_approach_arrival_tolerance_m").value
        )
        self.grasp_lift_minimum = float(
            self.get_parameter("grasp_lift_minimum_m").value
        )
        self.grasp_lift_verification_timeout = float(
            self.get_parameter("grasp_lift_verification_timeout_seconds").value
        )
        self.cycle_return_tolerance = float(
            self.get_parameter("cycle_return_tolerance_m").value
        )
        self.tennis_tip_preclose_offset = float(
            self.get_parameter("tennis_tip_preclose_offset_rad").value
        )
        self.tennis_tip_preclose_duration = float(
            self.get_parameter("tennis_tip_preclose_duration_seconds").value
        ) / self.speed_scale
        self.tennis_tip_guard_pause = float(
            self.get_parameter("tennis_tip_guard_pause_seconds").value
        ) / self.speed_scale
        self.tennis_root_close_duration = float(
            self.get_parameter("tennis_root_close_duration_seconds").value
        ) / self.speed_scale
        placement_yaw_degrees = float(
            self.get_parameter("placement_yaw_degrees").value
        )
        if not math.isclose(abs(placement_yaw_degrees), 90.0, abs_tol=1.0e-6):
            raise GraspActionError("placement_yaw_degrees must be exactly 90")
        self.placement_yaw = math.pi * 0.5
        self.placement_distance = float(
            self.get_parameter("placement_distance_m").value
        )
        self.placement_distance_decrement = float(
            self.get_parameter("placement_distance_decrement_m").value
        )
        self.placement_minimum_distance = float(
            self.get_parameter("placement_minimum_distance_m").value
        )
        self.placement_arrival_tolerance = float(
            self.get_parameter("placement_arrival_tolerance_m").value
        )
        self.classification_zone_margin = float(
            self.get_parameter("classification_zone_margin_m").value
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
            self.minimum_grasp_approach,
            self.grasp_approach_arrival_tolerance,
            self.grasp_lift_minimum,
            self.grasp_lift_verification_timeout,
            self.cycle_return_tolerance,
            self.tennis_tip_preclose_duration,
            self.tennis_tip_guard_pause,
            self.tennis_root_close_duration,
            self.placement_yaw,
            self.placement_distance,
            self.placement_arrival_tolerance,
            self.classification_zone_margin,
        ) <= 0.0:
            raise GraspActionError("pose-feedback parameters must be positive")
        if not 0.0 <= self.placement_distance_decrement < self.placement_distance:
            raise GraspActionError(
                "placement_distance_decrement_m must be nonnegative and below start"
            )
        if not 0.0 < self.placement_minimum_distance <= self.placement_distance:
            raise GraspActionError(
                "placement_minimum_distance_m must be positive and at most start"
            )
        if not 0.0 <= self.tennis_tip_preclose_offset <= 0.60:
            raise GraspActionError(
                "tennis_tip_preclose_offset_rad must be between 0 and 0.60"
            )
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
        self._sorting_home_pose = None
        self._aligned_yaw = None
        self._placement_ready = False
        self._placement_departure_pose = None
        self._placement_heading = None
        self._placement_return_pending = False
        self._approach_departure_pose = None
        self._approach_heading = None
        self._approach_return_pending = False
        self._active_sort_attempt = 1
        self._last_attempt_failure = None
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

    @staticmethod
    def _is_hsv_only_detection(detection):
        """Return True only for a bbox created entirely by HSV.

        HSV may promote an existing YOLO bbox to bottle.  Such a bbox is
        still geometrically useful because its coordinates came from YOLO.
        Only raw_class_name == hsv_only (or class_id == -1) is forbidden
        from final pixel-level alignment.
        """
        if detection is None:
            return False

        if str(detection.get("raw_class_name", "")) == "hsv_only":
            return True

        try:
            return int(detection.get("class_id", -999)) == -1
        except (TypeError, ValueError):
            return False

    def _wait_stable_detection(
        self,
        preferred_class=None,
        require_yolo=False,
        timeout_seconds=None,
    ):
        """Wait for a stable detection.

        require_yolo=False:
            ACQUIRE may use YOLO or independent HSV.

        require_yolo=True:
            pure HSV-only boxes are removed.  This mode is used only for
            final camera-centre refinement, where bbox geometry must come
            from YOLO rather than the transparent bottle colour contour.
        """

        self.detection_frames.clear()

        timeout = (
            self.detection_timeout
            if timeout_seconds is None
            else float(timeout_seconds)
        )

        deadline = time.monotonic() + timeout

        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(
                self,
                timeout_sec=0.10,
            )

            frames = list(
                self.detection_frames
            )

            if require_yolo:
                frames = [
                    [
                        item
                        for item in frame
                        if not self._is_hsv_only_detection(
                            item
                        )
                    ]
                    for frame in frames
                ]

            detection = stable_center_detection(
                frames,
                self.camera_principal_x,
                required_votes=self.stable_required_frames,
                preferred_class=preferred_class,
                max_center_spread_px=self.stable_center_spread_px,
                selection="leftmost",
            )

            if detection is not None:

                source = (
                    "HSV_ONLY"
                    if self._is_hsv_only_detection(
                        detection
                    )
                    else "YOLO"
                )

                self._publish_status(
                    "VISION LOCK | "
                    f"id={detection.get('object_id', 'unassociated')} | "
                    f"class={detection['class_name']} | "
                    f"source={source} | "
                    f"confidence={detection['confidence']:.3f} | "
                    f"stable={detection['stable_votes']}/"
                    f"{self.stable_window_frames} | "
                    "tracking=leftmost"
                )

                return detection

        class_text = (
            preferred_class
            or "bottle/tennis"
        )

        source_text = (
            " YOLO"
            if require_yolo
            else ""
        )

        raise RetryableAcquisitionError(
            f"no stable{source_text} {class_text} detection "
            f"within {timeout:.1f} s"
        )

    def _align_to_visual_target(self, held_positions):
        """Continuously rotate the chassis while visually centering the target."""
        detection = self._wait_stable_detection()
        target_class = detection["class_name"]

        # HSV-only is permitted for discovery / identity association,
        # but never for final pixel-level grasp alignment.
        initial_hsv_only = self._is_hsv_only_detection(
            detection
        )
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
        coarse_target_yaw = None
        if robot_position is not None and target_position is not None and yaw is not None:
            coarse_target_yaw = math.atan2(
                target_position[1] - robot_position[1],
                target_position[0] - robot_position[0],
            )
            bearing_error = wrap_angle(coarse_target_yaw - yaw)
            initial_target_angle = abs(bearing_error)

        if initial_hsv_only and coarse_target_yaw is None:
            raise RetryableAcquisitionError(
                "HSV-only bottle was detected, but its associated "
                "Gazebo object has no usable world pose; refusing "
                "to use HSV contour geometry for grasp alignment"
            )

        self._publish_status(
            "TARGET GEOMETRY LOCKED | "
            f"class={target_class} | "
            f"initial_angle={math.degrees(initial_target_angle):.1f} deg | "
            "approach=measured_pose_standoff"
        )

        # YOLO selects and locks the physical entity.  Gazebo world feedback
        # then performs the large, deterministic part of the turn; YOLO is
        # deliberately kept enabled and performs the final camera correction.
        # This avoids asking a low-FPS Jetson detector to finish a 55-degree
        # outer-slot rotation inside one short alignment timeout.
        if coarse_target_yaw is not None:
            current_pose = self._identity.pose2d()
            if current_pose is None:
                raise RetryableAcquisitionError(
                    "pose feedback lost before coarse target alignment"
                )
            self._publish_status(
                "COARSE WORLD-BEARING ALIGN | "
                f"id={self._locked_object_id} | "
                f"target_yaw={math.degrees(coarse_target_yaw):.2f} deg | "
                "fine_stage=live_vision"
            )
            self._drive_to_pose(
                held_positions,
                Pose2D(current_pose.x, current_pose.y, coarse_target_yaw),
                "COARSE ALIGN TO LOCKED OBJECT COORDINATE",
                control_mode="yaw",
            )
            # Discard pre-turn boxes and establish a new stable observation at
            # the coarse heading before applying pixel-level corrections.
            if (
                target_class == BOTTLE
                and initial_hsv_only
            ):
                try:
                    detection = self._wait_stable_detection(
                        preferred_class=BOTTLE,
                        require_yolo=True,
                        timeout_seconds=min(
                            4.0,
                            self.detection_timeout,
                        ),
                    )

                    self._publish_status(
                        "HSV->YOLO HANDOFF SUCCESS | "
                        f"id={self._locked_object_id} | "
                        "YOLO bbox will perform final pixel alignment"
                    )

                except RetryableAcquisitionError:

                    current_pose = (
                        self._identity.pose2d()
                    )

                    if current_pose is None:
                        raise RetryableAcquisitionError(
                            "HSV bottle was world-aligned but "
                            "chassis pose feedback was lost"
                        )

                    self._aligned_yaw = (
                        current_pose.yaw
                    )

                    self._publish_status(
                        "HSV->YOLO HANDOFF UNAVAILABLE | "
                        f"id={self._locked_object_id} | "
                        "HSV bbox is NOT used for final alignment | "
                        "continue with locked Gazebo world pose"
                    )

                    return target_class

            else:
                detection = self._wait_stable_detection(
                    preferred_class=target_class
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
                    if canonical_class(
                        item.get(
                            "class_name",
                            "",
                        )
                    )
                    == target_class
                ]

                # HSV-only boxes are useful for finding / classifying
                # bottles, but their contour centres are not trusted for
                # the final +/- pixel alignment used before grasping.
                if target_class == BOTTLE:
                    candidates = [
                        item
                        for item in candidates
                        if not self._is_hsv_only_detection(
                            item
                        )
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

        if (
            target_class == BOTTLE
            and self._locked_object_id is not None
        ):
            current_pose = self._identity.pose2d()

            if current_pose is not None:
                self._aligned_yaw = (
                    current_pose.yaw
                )

                self._publish_status(
                    "YOLO FINAL ALIGN UNAVAILABLE | "
                    f"id={self._locked_object_id} | "
                    "reject HSV contour steering | "
                    "continue with associated Gazebo world position"
                )

                return target_class

        raise RetryableAcquisitionError(
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

    def _gripper_pose(self, pose, initial, index, opened):
        """Keep ordinary opening, but retain a curled front pad when closed."""
        result = super()._gripper_pose(pose, initial, index, opened)
        if self._use_tennis_grasp:
            if opened:
                result = tennis_open_pose(result, initial, index)
            else:
                result = tennis_closed_pose(
                    result,
                    initial,
                    index,
                    self.tennis_tip_preclose_offset,
                )
        return result

    def _move_gripper(self, label, start, target):
        """For tennis only: front pads first, root joints second."""
        index = {name: i for i, name in enumerate(POSITION_JOINTS)}
        staged_close = (
            self._use_tennis_grasp
            and is_tip_first_closing(start, target, index)
        )
        if not staged_close:
            super()._move_gripper(label, start, target)
            return

        guarded = tip_first_pose(start, target, index)
        self._publish_status(
            "TENNIS GRIP PHASE 1/2 | curl front pads first | "
            f"offset={self.tennis_tip_preclose_offset:.3f} rad"
        )
        self._move_interpolated(
            "网球抓取阶段 1：指尖先内收形成挡球唇",
            start,
            guarded,
            self.tennis_tip_preclose_duration,
        )
        self._publish_status(
            "TENNIS TIP GUARD READY | roots remain open; settling before clamp"
        )
        self._settle(guarded, self.tennis_tip_guard_pause)

        self._publish_status(
            "TENNIS GRIP PHASE 2/2 | close roots slowly behind guarded ball"
        )
        self._move_interpolated(
            "网球抓取阶段 2：根部缓慢合拢夹紧",
            guarded,
            target,
            self.tennis_root_close_duration,
        )

    def _is_fully_closed(self, closed_pose, index):
        # A tennis ball may be retained by the curled joint_5 pads while the
        # two root joints still reach their nominal command.  Its measured
        # world-height change after the test lift is the authoritative check.
        if self._use_tennis_grasp:
            return False
        return super()._is_fully_closed(closed_pose, index)

    def _verify_locked_object_lift(self):
        """Require the locked entity to rise with the test-lift motion."""
        if self._locked_object_id is None or self._locked_object_origin is None:
            raise GraspActionError("test lift has no locked object pose")
        origin_z = float(self._locked_object_origin[2])
        deadline = time.monotonic() + self.grasp_lift_verification_timeout
        best_lift = float("-inf")
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            position = self._identity.position(self._locked_object_id)
            if position is None:
                continue
            lift = float(position[2]) - origin_z
            best_lift = max(best_lift, lift)
            if lift >= self.grasp_lift_minimum:
                self._publish_status(
                    "REAL GRASP VERIFIED | "
                    f"id={self._locked_object_id} | lift={lift:.3f} m"
                )
                return True
        measured = "unavailable" if not math.isfinite(best_lift) else f"{best_lift:.3f} m"
        self._publish_status(
            "MISSED GRASP DETECTED | "
            f"id={self._locked_object_id} | lift={measured} | "
            f"required={self.grasp_lift_minimum:.3f} m"
        )
        return False

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
        if (
            label == "小幅试抬"
            and self._current_class in (BOTTLE, TENNIS)
            and not self._verify_locked_object_lift()
        ):
            raise MissedGraspError("locked object did not rise during test lift")

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
        travel_yaw=None,
        arrival_tolerance=None,
    ):
        """Run one explicit feedback phase: position, yaw, or full pose."""
        if not isinstance(target, Pose2D):
            raise GraspActionError(f"{label}: target is not a Pose2D")
        if control_mode not in {"position", "yaw", "pose", "straight"}:
            raise GraspActionError(f"{label}: invalid control mode {control_mode}")
        if control_mode == "straight":
            if travel_yaw is None or arrival_tolerance is None:
                raise GraspActionError(
                    f"{label}: straight mode needs travel_yaw and arrival_tolerance"
                )
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
                        arrival_tolerance=arrival_tolerance,
                    )
                elif control_mode == "yaw":
                    output = self.pose_controller.compute_yaw(current, target)
                elif control_mode == "straight":
                    output = self.pose_controller.compute_straight(
                        current,
                        target,
                        travel_yaw,
                        arrival_tolerance,
                    )
                else:
                    output = self.pose_controller.compute(
                        current,
                        target,
                        allow_reverse=allow_reverse,
                    )
                if control_mode == "yaw":
                    # A yaw phase must be a true skid-steer pivot: left and
                    # right wheels are equal and opposite, with zero forward
                    # component. Refuse any future controller regression that
                    # would drive an arc while carrying an object.
                    wheels = output.wheels
                    in_place = (
                        math.isclose(wheels[0], wheels[2], abs_tol=1.0e-9)
                        and math.isclose(wheels[1], wheels[3], abs_tol=1.0e-9)
                        and math.isclose(wheels[0], -wheels[1], abs_tol=1.0e-9)
                    )
                    if not in_place:
                        raise GraspActionError(
                            f"{label}: yaw controller generated non-pivot wheels"
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
        # The camera has just centred this object.  Freeze that measured world
        # heading as the grasp corridor instead of recomputing a slightly
        # different bearing from noisy entity x/y samples.  Only progress on
        # this one axis is controlled, so the base cannot chase the endpoint
        # alternately to the left and right.
        locked_heading = current.yaw
        object_distance = math.hypot(
            float(target_position[0]) - current.x,
            float(target_position[1]) - current.y,
        )
        travel_distance = max(0.0, object_distance - standoff)
        if travel_distance < self.minimum_grasp_approach:
            self._publish_status(
                "UNSAFE SHORT APPROACH REJECTED | "
                f"object_distance={object_distance:.3f} m | "
                f"standoff={standoff:.3f} m | "
                f"travel={travel_distance:.3f} m | action=return_home_reacquire"
            )
            raise RetryableAcquisitionError(
                "locked target is already inside the minimum approach corridor"
            )
        target = Pose2D(
            current.x + travel_distance * math.cos(locked_heading),
            current.y + travel_distance * math.sin(locked_heading),
            locked_heading,
        )
        self._approach_departure_pose = current
        self._approach_heading = locked_heading
        self._approach_return_pending = True
        self._publish_status(
            "STRAIGHT GRASP CORRIDOR LOCKED | "
            f"yaw={math.degrees(locked_heading):.2f} deg | "
            f"travel={travel_distance:.3f} m | steering=heading_hold_only"
        )
        self._drive_to_pose(
            held_positions,
            target,
            f"APPROACH POSITION {self._locked_object_id} AT {standoff:.3f} M",
            control_mode="straight",
            travel_yaw=locked_heading,
            arrival_tolerance=self.grasp_approach_arrival_tolerance,
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

    def _reverse_approach_corridor(self, held_positions):
        """Back out along the visually aligned approach without turning."""
        if not self._approach_return_pending:
            return
        if self._approach_departure_pose is None or self._approach_heading is None:
            raise GraspActionError("approach return requested without corridor pose")
        self._publish_status(
            "REVERSE GRASP CORRIDOR | fixed heading back to pre-approach pose"
        )
        self._drive_to_pose(
            held_positions,
            self._approach_departure_pose,
            "REVERSE STRAIGHT FROM GRASP",
            control_mode="straight",
            travel_yaw=self._approach_heading,
            arrival_tolerance=self.grasp_approach_arrival_tolerance,
        )
        self._approach_return_pending = False

    def _return_to_cycle_center_staged(
        self,
        held_positions,
        target,
        label_prefix,
    ):
        """Return to cycle x/y without continuously chasing endpoint bearing.

        The ordinary position controller recomputes the target bearing every
        frame.  Near the origin, skid-steer pivot drift changes that bearing
        faster than the loaded chassis can settle, producing a spiral.

        Here every correction pass is finite:
            fixed yaw -> fixed straight motion -> measure again.

        At most three correction passes are allowed, so this routine can never
        spin indefinitely.
        """

        max_passes = 3

        for pass_index in range(1, max_passes + 1):

            current = self._identity.pose2d()

            if current is None:
                raise GraspActionError(
                    f"{label_prefix}: chassis pose unavailable"
                )

            dx = target.x - current.x
            dy = target.y - current.y
            distance = math.hypot(dx, dy)

            if distance <= self.cycle_return_tolerance:
                self._publish_status(
                    f"{label_prefix} CENTER REACHED | "
                    f"pass={pass_index - 1} | "
                    f"error={distance:.4f} m"
                )
                return current

            # World direction from current measured position to the fixed
            # cycle center.
            bearing = math.atan2(dy, dx)

            # Choose forward or reverse according to whichever requires
            # less chassis rotation.
            forward_heading = bearing
            reverse_heading = wrap_angle(
                bearing + math.pi
            )

            forward_error = abs(
                wrap_angle(
                    forward_heading - current.yaw
                )
            )

            reverse_error = abs(
                wrap_angle(
                    reverse_heading - current.yaw
                )
            )

            if reverse_error < forward_error:
                travel_yaw = reverse_heading
                travel_mode = "reverse"
            else:
                travel_yaw = forward_heading
                travel_mode = "forward"

            self._publish_status(
                f"{label_prefix} STAGED RETURN | "
                f"pass={pass_index}/{max_passes} | "
                f"distance={distance:.4f} m | "
                f"travel={travel_mode} | "
                f"heading={math.degrees(travel_yaw):.2f} deg"
            )

            # Phase A: one explicit pivot.
            self._drive_to_pose(
                held_positions,
                Pose2D(
                    current.x,
                    current.y,
                    travel_yaw,
                ),
                f"{label_prefix} PASS {pass_index} FIXED TURN",
                control_mode="yaw",
            )

            # Phase B: freeze that heading and translate.
            #
            # compute_straight never recomputes atan2(target-current)
            # every frame, so endpoint noise cannot create an orbit.
            self._drive_to_pose(
                held_positions,
                target,
                f"{label_prefix} PASS {pass_index} FIXED STRAIGHT",
                control_mode="straight",
                travel_yaw=travel_yaw,
                arrival_tolerance=self.cycle_return_tolerance,
            )

        # Finite failure instead of an infinite spiral.
        current = self._identity.pose2d()

        if current is None:
            raise GraspActionError(
                f"{label_prefix}: pose lost after staged return"
            )

        final_error = math.hypot(
            target.x - current.x,
            target.y - current.y,
        )

        if final_error > self.cycle_return_tolerance:
            raise GraspActionError(
                f"{label_prefix}: staged cycle-center return "
                f"did not converge after {max_passes} passes; "
                f"error={final_error:.4f} m"
            )

        return current

    def _rotate_chassis(self, held_positions):
        self._placement_ready = False
        self._placement_departure_pose = None
        self._placement_heading = None
        self._placement_return_pending = False
        if self._current_class not in (BOTTLE, TENNIS):
            raise GraspActionError("placement requested without a vision class")
        if self._cycle_pose is None:
            raise GraspActionError("placement requested without a fresh cycle pose")
        cycle_pose = self._cycle_pose

        # First carry the grasped object backwards along the clear approach
        # corridor.  Do not sweep the arm through the remaining objects.
        self.task_state.transition(SortingState.RETREAT)
        self._publish_status("RETREAT BEFORE CLASSIFICATION | straight reverse")
        self._reverse_approach_corridor(held_positions)
        self._drive_to_pose(
            held_positions,
            cycle_pose,
            "CONFIRM RETREAT WITHIN CYCLE CENTER REGION",
            allow_reverse=True,
            control_mode="position",
            arrival_tolerance=self.cycle_return_tolerance,
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
        turn_direction = (
            "CLOCKWISE/RIGHT" if self._current_class == BOTTLE
            else "COUNTERCLOCKWISE/LEFT"
        )
        self._publish_status(
            "CLASS-SPECIFIC IN-PLACE TURN | "
            f"class={self._current_class} | direction={turn_direction} | "
            "wheel_contract=left_equals_negative_right"
        )
        self._publish_status(
            "DIRECT CLASSIFICATION TURN | center region confirmed; "
            "skip redundant intermediate home-yaw turn"
        )
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
        placement_distance = placement_distance_for_attempt(
            self.placement_distance,
            self.placement_distance_decrement,
            self.placement_minimum_distance,
            self._active_sort_attempt,
        )
        self._publish_status(
            "PLACEMENT DISTANCE SCHEDULE | "
            f"attempt={self._active_sort_attempt}/{self.run_count} | "
            f"distance={placement_distance:.3f} m | "
            f"start={self.placement_distance:.3f} m | "
            f"decrement={self.placement_distance_decrement:.3f} m"
        )
        placement_pose = Pose2D(
            measured.x + placement_distance * math.cos(target_yaw),
            measured.y + placement_distance * math.sin(target_yaw),
            target_yaw,
        )
        self._placement_departure_pose = measured
        self._placement_heading = target_yaw
        self._placement_return_pending = True
        self._drive_to_pose(
            held_positions,
            placement_pose,
            f"DRIVE {placement_distance:.3f} M TO {zone}",
            control_mode="straight",
            travel_yaw=target_yaw,
            arrival_tolerance=self.placement_arrival_tolerance,
        )
        self._placement_ready = True
        self._publish_status(
            "PLACEMENT CORRIDOR ARRIVAL VERIFIED | stop, lower, and release; "
            "no endpoint turn"
        )

    def _run_once(self, attempt, initial, current, index):
        self._active_initial_positions = list(initial)
        self._active_sort_attempt = int(attempt)
        self._last_attempt_failure = None
        self._placement_ready = False
        self._placement_departure_pose = None
        self._placement_heading = None
        self._placement_return_pending = False
        self.task_state.transition(SortingState.GRASP)
        try:
            result = super()._run_once(attempt, initial, current, index)
        except MissedGraspError as error:
            self._last_attempt_failure = str(error)
            held = list(self.last_position_command or current)
            self._publish_status(
                "RECOVER MISSED GRASP | open, retract arm, return origin, retry round"
            )
            self._move_chassis_final_insert(held, forward=False)
            result = False, self._safe_home_pose(initial, held, index)
        if not result[0] and self._last_attempt_failure is None:
            self._last_attempt_failure = "gripper closure reported an empty grasp"
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

    def _restore_pre_grasp_heading(self, held_positions):
        """Before a verified grasp, stop and restore yaw only.

        Visual-alignment or approach retries must not invoke the full
        return-home position controller.  The chassis keeps its current x/y
        and only pivots back to the fixed sorting-home heading before the next
        acquisition.
        """
        if self._sorting_home_pose is None:
            raise GraspActionError(
                "pre-grasp heading recovery requested without sorting home pose"
            )

        current = self._identity.pose2d()
        if current is None:
            raise GraspActionError(
                "pre-grasp heading recovery lost chassis pose feedback"
            )

        self._publish_status(
            "PRE-GRASP RETRY RECOVERY | stop wheels; restore home yaw only; "
            "x/y return disabled until grasp is verified"
        )

        self._publish_to(
            self.wheel_publisher,
            [0.0] * 4,
            repeat=10,
        )

        target = Pose2D(
            current.x,
            current.y,
            self._sorting_home_pose.yaw,
        )

        self._drive_to_pose(
            held_positions,
            target,
            "PRE-GRASP RESTORE HOME YAW ONLY",
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
            self._reverse_approach_corridor(held_positions)
            if (
                self._placement_return_pending
                and self._placement_departure_pose is not None
                and self._placement_heading is not None
            ):
                self._publish_status(
                    "DIRECT REVERSE AFTER RELEASE | retrace placement corridor"
                )
                self._drive_to_pose(
                    held_positions,
                    self._placement_departure_pose,
                    "REVERSE STRAIGHT FROM PLACEMENT ZONE",
                    control_mode="straight",
                    travel_yaw=self._placement_heading,
                    arrival_tolerance=self.cycle_return_tolerance,
                )
                self._placement_return_pending = False
            self._drive_to_pose(
                held_positions,
                self._cycle_pose,
                "CONFIRM RETURN WITHIN CYCLE CENTER REGION",
                allow_reverse=True,
                control_mode="position",
                arrival_tolerance=self.cycle_return_tolerance,
            )
            self._drive_to_pose(
                held_positions,
                self._cycle_pose,
                "RETURN TO CYCLE YAW",
                control_mode="yaw",
            )
            # Yaw-only skid steering may translate the chassis.  Close the
            # loop on x, y and yaw together before permitting another image
            # acquisition, so every cycle uses the same world-frame origin.
            # Position and yaw were already verified separately.
            # Accept the relaxed home region instead of chasing exact (0, 0, 0).
            current_pose = self._identity.pose2d()
            if current_pose is not None:
                home_error = math.hypot(
                    self._cycle_pose.x - current_pose.x,
                    self._cycle_pose.y - current_pose.y,
                )
                yaw_error = abs(
                    wrap_angle(
                        self._cycle_pose.yaw - current_pose.yaw
                    )
                )
                self._publish_status(
                    "RELAXED HOME ACCEPTED | "
                    f"position_error={home_error:.4f} m | "
                    f"yaw_error={math.degrees(yaw_error):.2f} deg | "
                    f"position_tolerance={self.cycle_return_tolerance:.3f} m"
                )

        finally:
            self._cycle_pose = None
            self._placement_departure_pose = None
            self._placement_heading = None
            self._placement_return_pending = False
            self._approach_departure_pose = None
            self._approach_heading = None
            self._approach_return_pending = False
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
        self._sorting_home_pose = self._identity.pose2d()
        if self._sorting_home_pose is None:
            raise GraspActionError('cannot capture fixed world-frame sorting origin')
        self._publish_status(
            'FIXED SORTING HOME CAPTURED | '
            f'x={self._sorting_home_pose.x:.4f} | '
            f'y={self._sorting_home_pose.y:.4f} | '
            f'yaw={math.degrees(self._sorting_home_pose.yaw):.2f} deg'
        )
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
                    # Never redefine home from the accumulated pose error of
                    # the preceding round.  All six rounds share this one
                    # immutable world-frame origin.
                    self._cycle_pose = self._sorting_home_pose
                    self._set_vision_enabled(
                        True,
                        f"acquire and align cycle {processed_count + 1}",
                    )
                    self.task_state.transition(SortingState.ALIGN)
                    try:
                        self._current_class = self._align_to_visual_target(current)
                    except RetryableAcquisitionError as error:
                        self._set_vision_enabled(False, "acquire phase ended")
                        if self.scene_mode in {"unknown", "empty"} and str(
                            error
                        ).startswith("no stable"):
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
                        current = list(self.last_position_command or current)
                        self._publish_status(
                            "VISION ALIGN RETRY | "
                            f"reason={error} | action=return_fixed_home_reacquire | "
                            "processed_count_unchanged=true"
                        )
                        self._restore_pre_grasp_heading(current)
                        self._locked_object_id = None
                        self._locked_object_origin = None
                        self._current_class = None
                        if retry < self.max_grasp_retries:
                            self.task_state.transition(
                                SortingState.ACQUIRE,
                                "retry visual alignment from fixed home",
                            )
                            continue
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
                    try:
                        self._approach_locked_object(current)
                    except RetryableAcquisitionError as error:
                        self._set_vision_enabled(False, "unsafe approach rejected")
                        current = list(self.last_position_command or current)
                        self._publish_status(
                            "APPROACH RETRY | "
                            f"reason={error} | action=return_fixed_home_reacquire | "
                            "processed_count_unchanged=true"
                        )
                        self._restore_pre_grasp_heading(current)
                        self._locked_object_id = None
                        self._locked_object_origin = None
                        self._current_class = None
                        if retry < self.max_grasp_retries:
                            self.task_state.transition(
                                SortingState.ACQUIRE,
                                "retry short approach from fixed home",
                            )
                            continue
                        break

                    ok, current = self._run_once(
                        success_count + 1, initial, current, index
                    )
                    current = list(self.last_position_command or current)
                    if ok:
                        placed = self._identity.position(self._locked_object_id)
                        origin = self._locked_object_origin
                        moved = (placed is not None and origin is not None and
                                 math.hypot(placed[0]-origin[0], placed[1]-origin[1]) >= 0.08)
                        in_zone = False
                        zone_offset = None
                        if placed is not None and self._cycle_pose is not None:
                            zone_offset = lateral_offset(
                                self._cycle_pose.x,
                                self._cycle_pose.y,
                                self._cycle_pose.yaw,
                                placed[0],
                                placed[1],
                            )
                            in_zone = in_classification_half(
                                self._cycle_pose.x,
                                self._cycle_pose.y,
                                self._cycle_pose.yaw,
                                placed[0],
                                placed[1],
                                left_half=self._current_class == TENNIS,
                                margin=self.classification_zone_margin,
                            )
                        if not moved:
                            ok = False
                            self._last_attempt_failure = (
                                "placement verification found no source displacement"
                            )
                            self._publish_status('PLACEMENT NOT VERIFIED | target did not leave source; no count increment')
                        elif not in_zone:
                            ok = False
                            self._last_attempt_failure = (
                                "placement verification found object outside class half"
                            )
                            measured_text = (
                                "unavailable"
                                if zone_offset is None
                                else f"{zone_offset:.3f} m"
                            )
                            self._publish_status(
                                "PLACEMENT WRONG HALF | "
                                f"class={self._current_class} | "
                                f"left_axis_offset={measured_text} | "
                                f"required_margin={self.classification_zone_margin:.3f} m"
                            )
                        else:
                            self._identity.completed.add(self._locked_object_id)
                            self._publish_status(
                                f"CLASS HALF VERIFIED | class={self._current_class} | "
                                f"left_axis_offset={zone_offset:.3f} m | "
                                f"margin={self.classification_zone_margin:.3f} m"
                            )
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
                    failure_reason = (
                        self._last_attempt_failure or "attempt did not verify"
                    )
                    self._locked_object_id = None
                    self._locked_object_origin = None
                    self._current_class = None
                    self._use_tennis_grasp = False
                    self.detection_frames.clear()
                    self._publish_status(
                        "ROUND RETRY | "
                        f"reason={failure_reason} | "
                        f"attempt={retry + 1}/{self.max_grasp_retries + 1} | "
                        "returned_to_origin=true | reacquire=true | "
                        "success_count_unchanged=true"
                    )
                    if retry < self.max_grasp_retries:
                        self.task_state.transition(
                            SortingState.ACQUIRE,
                            "retry round with fresh visual acquisition",
                        )
                if not completed:
                    raise GraspActionError(
                        "round failed after configured retries; refusing to guess or continue"
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
