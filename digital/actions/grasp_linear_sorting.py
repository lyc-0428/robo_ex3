#!/usr/bin/env python3
"""Fixed-heading lateral scan, physical grasp/retraction, and verified sorting.

VelocityControl supplies simulated holonomic base motion. No entity is
teleported or attached to the gripper; lifting and carrying remain contacts.
"""
from collections import deque
import json
import math
from pathlib import Path
import time

import rclpy
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Twist

from grasp_bottle_tennis import GraspBottleTennisAction, MissedGraspError
from grasp_cube import GraspActionError, GraspCubeAction, POSITION_JOINTS
from sorting_identity import SortingIdentity
from linear_arm_kinematics import LinearArmKinematics, ARM_JOINTS, ArmReachError
from linear_sorting_core import (
    ROW_X, LEFT_END, RIGHT_END, EMPTY_GRID_CELL, EMPTY_GRID_Y, VerifiedLedger,
    drop_y, fixed_grid_cell, rail_command, select_visible,
)
from vision_common import BOTTLE, TENNIS, TENNIS_SPAWN_HEIGHT


class AcquisitionMiss(GraspActionError):
    """A target can be retried on the return scan without counting it."""


class LinearSorter(GraspBottleTennisAction):
    def __init__(self):
        super().__init__()
        defaults = dict(linear_scan_speed_mps=.050, linear_transport_speed_mps=.60,
                        linear_empty_return_speed_mps=.90, linear_acceleration_mps2=1.00,
                        linear_move_timeout_seconds=600.0,
                        linear_watch_seconds=3.0, linear_max_passes=0,
                        linear_position_tolerance_m=.010)
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.scan_speed = float(self.get_parameter("linear_scan_speed_mps").value)
        self.transport_speed = float(self.get_parameter("linear_transport_speed_mps").value)
        self.empty_return_speed = float(self.get_parameter("linear_empty_return_speed_mps").value)
        self.acceleration = float(self.get_parameter("linear_acceleration_mps2").value)
        self.move_timeout = float(self.get_parameter("linear_move_timeout_seconds").value)
        self.watch_seconds = float(self.get_parameter("linear_watch_seconds").value)
        self.max_passes = int(self.get_parameter("linear_max_passes").value)
        self.rail_tolerance = float(self.get_parameter("linear_position_tolerance_m").value)
        if not (0 < self.scan_speed <= .06 and 0 < self.transport_speed <= .65
                and 0 < self.empty_return_speed <= .95
                and 0 < self.acceleration <= 1.10 and self.move_timeout > 0
                and self.watch_seconds > 0 and 0 < self.rail_tolerance <= .01
                and self.max_passes >= 0):
            raise GraspActionError("invalid linear sorting parameters")
        self.velocity = self.create_publisher(Twist, "/linear_sort/cmd_vel", 10)
        self._last_frame_time = 0.0
        self._raw_count = 0
        self._last_vision_report = 0.0
        self._carried = None
        self._carried_relative = None
        self._ledger = VerifiedLedger()
        self._pass_attempted = set()
        self._pass_watched = set()
        self._pass_empty_cells = set()
        self._scan_direction = -1
        self._command_y = 0.0
        self._initial = None
        self._current_grid = None
        self._index = {name: i for i, name in enumerate(POSITION_JOINTS)}
        urdf = Path(__file__).resolve().parents[1] / "src/robomaster_pick_place_sim/urdf/robomaster_ep_static.urdf"
        self._urdf_path = urdf
        self._kinematics = LinearArmKinematics(urdf, POSITION_JOINTS)

    def _detection_callback(self, message):
        before = self.last_detection_frame_number
        super()._detection_callback(message)
        if self.last_detection_frame_number != before:
            self._last_frame_time = time.monotonic()
            try:
                self._raw_count = len(json.loads(message.data).get("detections", []))
            except (ValueError, TypeError):
                self._raw_count = 0

    def _zero_base(self):
        self.velocity.publish(Twist())
        self._command_y = 0.0

    def safe_stop(self):
        if hasattr(self, "velocity"):
            for _ in range(5):
                self._zero_base()
        super().safe_stop()

    def _pose(self):
        pose = self._identity.pose2d()
        if pose is None:
            self._zero_base()
            raise GraspActionError("Gazebo base pose feedback is stale")
        return pose

    def _positions(self):
        result = {}
        for name in self._identity.active_task_objects():
            pos = self._identity.position(name)
            if pos is not None:
                result[name] = pos
        return result

    def _visible_target(self):
        if time.monotonic() - self._last_frame_time > 1.5:
            return None
        positions = {name: pos for name, pos in self._positions().items()
                     if abs(pos[0]-ROW_X) < .06 and abs(pos[1]) < 1.0}
        return select_visible(list(self.detection_frames), positions,
                              self._ledger.completed, self._pass_attempted,
                              self._pose().y, self._scan_direction)

    def _check_carry(self):
        if self._carried is None:
            return
        pos = self._identity.position(self._carried)
        robot = self._pose()
        if pos is None:
            raise GraspActionError("held object pose feedback lost")
        relative = (float(pos[0])-robot.x, float(pos[1])-robot.y, float(pos[2]))
        if max(abs(a-b) for a, b in zip(relative, self._carried_relative)) > .045:
            self._zero_base()
            raise GraspActionError("object slipped during transport; count unchanged")
        for name, other in self._positions().items():
            if name != self._carried and abs(other[1]) < 1.0:
                # A measured bottle load clears the source row by 6.8 cm.
                # Six centimetres leaves physical clearance while avoiding a
                # false abort on the former 7 cm numerical threshold.
                horizontal_clearance = .06
                # A bottle raised above 8 cm clears the source-row bodies
                # vertically, so its short, reachable retraction is enough.
                if self._current_class == BOTTLE and float(pos[2]) >= .08:
                    horizontal_clearance = .03
                if float(other[0])-float(pos[0]) < horizontal_clearance:
                    raise GraspActionError(f"carried object has not cleared source row: load_x={pos[0]:.3f}, other={name}, other_x={other[0]:.3f}; base stopped")

    def _move_y(self, target_y, held, scan=False, arrival_tolerance=None, speed=None):
        """Only intentional world-y travel; no rotate-to-target manoeuvres."""
        lateral_tolerance = (self.rail_tolerance if arrival_tolerance is None
                             else float(arrival_tolerance))
        if not 0 < lateral_tolerance <= .03:
            raise GraspActionError("invalid lateral arrival tolerance")
        max_speed = (self.scan_speed if scan else
                     (self.transport_speed if speed is None else float(speed)))
        if not 0 < max_speed <= .95:
            raise GraspActionError("invalid lateral speed")
        deadline = time.monotonic() + self.move_timeout
        stable = 0
        previous_sim = self.latest_sim_time
        last_progress = time.monotonic()
        last_y = self._pose().y
        last_log = 0.0
        try:
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=.02)
                self._publish(held, repeat=1)
                pose = self._pose()
                if scan:
                    candidate = self._visible_target()
                    if candidate is not None:
                        self._zero_base()
                        return candidate
                    if (EMPTY_GRID_CELL not in self._pass_empty_cells
                            and abs(pose.y-EMPTY_GRID_Y) < .018):
                        self._pass_empty_cells.add(EMPTY_GRID_CELL)
                        self._publish_status(
                            f"EMPTY GRID | cell={EMPTY_GRID_CELL} | action=continue "
                            f"{'right' if self._scan_direction < 0 else 'left'}"
                        )
                    # Pause at each occupied row location once per pass. Its
                    # class remains unknown until YOLO supplies a detection.
                    for name, pos in self._positions().items():
                        if (name not in self._pass_watched and name not in self._pass_attempted
                                and abs(pos[0]-ROW_X) < .06 and abs(pos[1]-pose.y) < .025):
                            self._pass_watched.add(name)
                            self._zero_base()
                            candidate = self._watch(held, self.watch_seconds)
                            if candidate is not None:
                                return candidate
                    if time.monotonic()-self._last_frame_time > 2.5:
                        self._zero_base()
                        if time.monotonic()-self._last_frame_time > 20:
                            raise GraspActionError("detector stream stopped; base stopped")
                        continue
                self._check_carry()
                sim_now = self.latest_sim_time
                if sim_now is None or previous_sim is None:
                    self._zero_base()
                    previous_sim = sim_now
                    continue
                dt = sim_now-previous_sim
                previous_sim = sim_now
                if dt <= 0:
                    continue
                vx, vy, omega = rail_command(pose.x, pose.y, pose.yaw, target_y,
                                             max_speed)
                if (abs(target_y-pose.y) <= lateral_tolerance
                        and abs(pose.x) <= self.rail_tolerance
                        and abs(pose.yaw) <= math.radians(1)):
                    self._zero_base()
                    stable += 1
                    if stable >= 5:
                        return None
                    continue
                stable = 0
                if abs(target_y-pose.y) > lateral_tolerance and abs(vy) < .018:
                    vy = math.copysign(.018, target_y-pose.y)
                limit = self.acceleration * min(dt, .1)
                self._command_y += max(-limit, min(limit, vy-self._command_y))
                cmd = Twist()
                cmd.linear.x, cmd.linear.y, cmd.angular.z = vx, self._command_y, omega
                self.velocity.publish(cmd)
                now = time.monotonic()
                if abs(pose.y-last_y) > .002:
                    last_progress, last_y = now, pose.y
                elif now-last_progress > 25:
                    raise GraspActionError("no lateral progress: check VelocityControl/bridge")
                if now-last_log > 3:
                    self._publish_status(f"{'SCAN' if scan else 'LATERAL'} | y={pose.y:.3f} -> {target_y:.3f} | sorted={len(self._ledger.completed)}/6")
                    last_log = now
            raise GraspActionError("lateral motion timeout")
        finally:
            self._zero_base()

    def _watch(self, held, seconds):
        deadline = time.monotonic()+seconds
        while rclpy.ok() and time.monotonic() < deadline:
            self._zero_base()
            self._publish(held, repeat=1)
            rclpy.spin_once(self, timeout_sec=.05)
            candidate = self._visible_target()
            if candidate is not None:
                return candidate
        self._publish_status(f"VISION WAIT | raw={self._raw_count} | associated={len(self.detection_frames[-1]) if self.detection_frames else 0} | missed targets remain for return scan")
        return None

    def _align_laterally(self, detection, held):
        self._locked_object_id = detection["object_id"]
        pos = self._identity.position(self._locked_object_id)
        if pos is None:
            raise AcquisitionMiss("object pose expired before alignment")
        self._locked_object_origin = pos.copy()
        self._current_class = detection["class_name"]
        self._current_grid, grid_y = fixed_grid_cell(float(pos[1]))
        box = detection["bbox"]
        center_x = .5 * (float(box["x1"]) + float(box["x2"]))
        center_y = .5 * (float(box["y1"]) + float(box["y2"]))
        self._publish_status(
            f"FIXED GRID TARGET | cell={self._current_grid} | cell_y={grid_y:.2f} "
            f"| bbox_center=({center_x:.1f},{center_y:.1f}) | class={self._current_class} | destination="
            f"{'left tennis area' if self._current_class == TENNIS else 'right bottle area'}"
        )
        self._use_tennis_grasp = self._current_class == TENNIS
        self.arm_2_delta = self.tennis_arm_2_delta if self._use_tennis_grasp else self.bottle_arm_2_delta
        # Small tennis balls need millimetre alignment. The minimum lateral
        # command above overcomes the previous low-speed contact dead zone.
        self._move_y(float(pos[1]), held, arrival_tolerance=.004)
        if not self._use_tennis_grasp:
            # select_visible already required two stable, same-ID bottle
            # detections.  The tall bottle can disappear behind the gripper
            # after exact lateral alignment, so do not demand a redundant
            # post-alignment frame before starting the grasp.
            actual = self._identity.position(self._locked_object_id)
            if actual is None or abs(actual[1]-self._pose().y) > .008:
                raise AcquisitionMiss("bottle moved during lateral alignment")
            self._publish_status(
                f"LINEAR TARGET LOCK | id={self._locked_object_id} "
                f"| cell={self._current_grid} | class={self._current_class} "
                f"| confidence={detection['confidence']:.3f} | stable pre-alignment detection"
            )
            return
        self.detection_frames.clear()
        deadline = time.monotonic()+10
        consecutive = 0
        last_frame = self.last_detection_frame_number
        try:
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=.05)
                self._publish(held, repeat=1)
                self._zero_base()
                if self.last_detection_frame_number == last_frame:
                    continue
                last_frame = self.last_detection_frame_number
                detections = [d for d in (self.detection_frames[-1] if self.detection_frames else [])
                              if d.get("object_id") == self._locked_object_id
                              and d["class_name"] == self._current_class]
                if not detections:
                    consecutive = 0
                    continue
                item = detections[0]
                error = .5*(item["bbox"]["x1"]+item["bbox"]["x2"])-self.camera_principal_x
                # Poses establish the exact lateral centre. Vision confirms the
                # entity is still present; do not turn the body for camera bias.
                actual = self._identity.position(self._locked_object_id)
                if actual is None or abs(actual[1]-self._pose().y) > .006:
                    raise AcquisitionMiss("object moved during alignment")
                if abs(error) <= 20:
                    consecutive += 1
                    if consecutive >= 2:
                        self._publish_status(f"LINEAR TARGET LOCK | id={self._locked_object_id} | cell={self._current_grid} | class={self._current_class} | confidence={item['confidence']:.3f}")
                        return
                else:
                    consecutive = 0
            raise AcquisitionMiss("target not confirmed at lateral centre; retry on return pass")
        finally:
            self._zero_base()

    def _carry_pose(self, source):
        # Preserve every finger command, including the tennis joint_5 lip.
        folded = list(source)
        for name in ARM_JOINTS:
            folded[self._index[name]] = self._initial[self._index[name]]
        return self._kinematics.shift(folded, dx=-.05, dz=-.03)

    def _move_arm(self, label, start, target, duration=None):
        self._zero_base()
        if duration is None:
            GraspCubeAction._move(self, label, start, target)
        else:
            self._move_interpolated(label, start, target, duration)
        # Commands finishing does not imply that the simulated joints arrived.
        deadline = time.monotonic() + 20.0
        stable = 0
        last_stamp = None
        while rclpy.ok() and time.monotonic() < deadline:
            self._zero_base()
            self._publish(target, repeat=1)
            rclpy.spin_once(self, timeout_sec=.05)
            message = self.latest_joint_state
            if message is None:
                continue
            stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
            if stamp == last_stamp:
                continue
            last_stamp = stamp
            measured = dict(zip(message.name, message.position))
            error = max(abs(measured.get(n, math.inf)-target[self._index[n]])
                        for n in ARM_JOINTS)
            stable = stable+1 if error <= .006 else 0
            if stable >= 3:
                self._publish_status(f"ARM REACHED | {label} | joint_error={error:.4f} rad")
                return list(target)
        raise GraspActionError(f"arm did not reach commanded pose: {label}; base remains stopped")

    def _open_and_fold(self, current):
        opened = self._gripper_pose(current, self._initial, self._index, opened=True)
        self._move_gripper("张开夹爪", current, opened)
        # First lift with fixed pitch, then fold behind the row.
        try:
            lifted = self._kinematics.shift(opened, dz=.04)
        except ValueError:
            lifted = opened
        current = self._move_arm("空爪抬离物体", opened, lifted)
        return self._move_arm("空爪收回后才能横移", current, self._carry_pose(current))

    def _target_grasp_pose(self):
        pose = super()._arm_pose(self._initial, 1.0)
        object_position = self._identity.position(self._locked_object_id)
        if object_position is None:
            raise AcquisitionMiss("object position expired")
        original_standoff = self.tennis_grasp_standoff if self._use_tennis_grasp else self.bottle_grasp_standoff
        insertion = .05*self.wheel_speed*self.final_insert_steps*self.physics_step_seconds/self.speed_scale
        object_x = float(object_position[0])
        row_tolerance = .012 if self._use_tennis_grasp else .050
        if abs(object_x-ROW_X) > row_tolerance:
            raise AcquisitionMiss("target pushed out of the fixed arm-reach row")
        shift_x = object_x-(original_standoff-insertion)
        # The fixed row replaces the old forward base approach/insertion.
        # Tennis at row=.28005 preserves its calibrated final tool pose.
        return self._kinematics.shift(pose, dx=shift_x)

    def _shift_held(self, source, dx=0.0, dz=0.0, label="持物移动"):
        # After a verified grasp, use large, quick waypoints.  Each waypoint
        # still checks whether the load moved relative to the tool.
        steps = max(1, math.ceil(max(abs(dx), abs(dz))/.08))
        # Solve all waypoints before moving so reachability errors cannot
        # interrupt a half-completed withdrawal.
        poses = [self._kinematics.shift(source, dx=dx*i/steps, dz=dz*i/steps)
                 for i in range(1, steps+1)]
        def relative():
            position = self._identity.position(self._locked_object_id)
            if position is None:
                raise MissedGraspError("held object feedback missing")
            message = self.latest_joint_state
            measured = dict(zip(message.name, message.position))
            actual = [measured.get(n, source[i]) for i,n in enumerate(POSITION_JOINTS)]
            tool = self._kinematics.fk(actual)
            robot = self._pose()
            return (float(position[0])-robot.x-tool[0],
                    float(position[1])-robot.y, float(position[2])-tool[1])
        reference = relative()
        current = list(source)
        for i, target in enumerate(poses):
            current = self._move_arm(f"{label} {i+1}/{steps}", current, target,
                                     duration=.25)
            self._settle(current, .06)
            offset = relative()
            drift = max(abs(a-b) for a,b in zip(offset, reference))
            allowed_drift = .020 if self._use_tennis_grasp else .030
            if drift > allowed_drift:
                self._zero_base()
                raise MissedGraspError(f"load slipped relative to tool by {drift:.3f} m; no transport/count")
        return current

    def _retract_bottle_reachable(self, source):
        """Retract a lifted bottle without aborting on one unreachable IK pose."""
        attempts = ((-.080, -.015), (-.070, -.015),
                    (-.060, -.020), (-.050, -.020), (-.040, -.020))
        last_error = None
        for dx, dz in attempts:
            try:
                current = self._shift_held(
                    source, dx=dx, dz=dz, label="快速斜向安全收臂")
                self._publish_status(
                    f"BOTTLE RETRACT SAFE | dx={dx:.3f} | dz={dz:.3f}"
                )
                return current
            except ArmReachError as error:
                last_error = error
                self._publish_status(
                    f"BOTTLE RETRACT POSE UNREACHABLE | dx={dx:.3f} "
                    f"| dz={dz:.3f} | trying shorter safe pose"
                )
        raise GraspActionError(
            f"no safe bottle retract pose found after verified grasp: {last_error}"
        )

    def _grasp_and_place(self, current):
        self._set_vision_enabled(False, "physical grasp and transport")
        grasp = self._gripper_pose(self._target_grasp_pose(), self._initial, self._index, opened=True)
        # Stable tennis sequence: lower first, insert while open, then close.
        if self._use_tennis_grasp:
            preinsert = self._kinematics.shift(grasp, dx=-.01995)
        else:
            # Bottles were being contacted by the front edge of the gripper.
            # Stop 1.5 cm earlier and 8 mm lower before closing.
            preinsert = self._kinematics.shift(grasp, dx=-.015, dz=-.008)
        # Two direct approach moves are sufficient: above the object, then to
        # the calibrated grasp height.  The former intermediate descent added
        # time without improving the verified grasp result.
        hover = self._kinematics.shift(preinsert, dz=.08)
        current = self._move_arm("快速伸臂到目标上方", current, hover, duration=.38)
        current = self._move_arm("快速下降到标定抓取高度", current, preinsert,
                                 duration=.38)
        if self._use_tennis_grasp:
            current = self._move_arm("张爪水平前插约2厘米", current, grasp,
                                     duration=.30)
        self._settle(current, .10)
        closed = self._gripper_pose(current, self._initial, self._index, opened=False)
        # Inherited stable tennis two-stage closure; bottle uses original close.
        if self._use_tennis_grasp:
            self._move_gripper("缓慢闭合夹爪", current, closed)
        else:
            self._move_interpolated("水瓶闭爪", current, closed, .60)
        self._settle(closed, .40 if not self._use_tennis_grasp else self.gripper_settle_duration)
        if self._is_fully_closed(closed, self._index):
            raise MissedGraspError("empty closure")
        if self._use_tennis_grasp:
            test_lift = list(closed)
            test_lift[self._index["arm_1_joint"]] = (
                self._initial[self._index["arm_1_joint"]] + self.arm_test_lift_delta)
        else:
            test_lift = self._kinematics.shift(closed, dz=.025)
        current = self._move_arm("小幅试抬", closed, test_lift)
        if not self._verify_locked_object_lift():
            raise MissedGraspError("object did not rise with gripper")
        # Bottles need higher vertical clearance before withdrawing.  One short
        # withdrawal then replaces the former second retract waypoint that
        # could stall with a bottle held between the fingers.
        lift_distance = .04 if self._use_tennis_grasp else .08
        current = self._shift_held(current, dz=lift_distance, label="快速持物抬高")
        if self._use_tennis_grasp:
            current = self._shift_held(current, dx=-.105, label="快速收臂")
        else:
            current = self._retract_bottle_reachable(current)
        self._settle(current, .10)
        pos = self._identity.position(self._locked_object_id)
        robot = self._pose()
        if pos is None or pos[2]-self._locked_object_origin[2] < .015:
            raise MissedGraspError("object lost during retraction")
        self._carried = self._locked_object_id
        self._carried_relative = (float(pos[0])-robot.x, float(pos[1])-robot.y, float(pos[2]))
        self._check_carry()
        target_y = drop_y(self._current_class, self.category_counts[self._current_class])
        self._publish_status(f"CARRY RETRACTED | id={self._carried} | destination_y={target_y:.2f}")
        self._move_y(target_y, current, arrival_tolerance=.02)
        self._carried = None
        # At the destination keep the arm retracted: lower, release, and lift
        # vertically.  No forward reach is needed inside the wide drop zone.
        carry_tool = self._kinematics.fk(current)
        grasp_tool = self._kinematics.fk(closed)
        current = self._shift_held(current, dz=float(grasp_tool[1]-carry_tool[1]),
                                   label="仅竖直下放放置")
        self._settle(current, .08)
        opened = self._gripper_pose(current, self._initial, self._index, opened=True)
        self._move_interpolated("快速释放物品", current, opened, .35)
        self._settle(opened, .20)
        released_up = self._kinematics.shift(opened, dz=.08)
        current = self._move_arm("释放后仅竖直抬起", opened, released_up,
                                 duration=.25)
        if self._current_class == BOTTLE:
            # Pull the empty gripper back before the high-speed return so it
            # cannot clip a bottle that has just been released.
            rear_clearance = self._kinematics.shift(current, dx=-.02)
            current = self._move_arm("放瓶后向后收臂", current, rear_clearance,
                                     duration=.25)
        released = self._verify_ground_release(target_y)
        self._ledger.record(self._locked_object_id, self._current_class,
                            self._locked_object_origin, released, target_y)
        self._identity.completed.add(self._locked_object_id)
        self.category_counts = dict(self._ledger.counts)
        self._publish_status(f"SORTED | total={len(self._ledger.completed)}/6 | bottles={self.category_counts[BOTTLE]}/3 | tennis={self.category_counts[TENNIS]}/3")
        return current

    def _verify_ground_release(self, target_y):
        deadline = time.monotonic()+8
        previous = None
        stable = 0
        last_update = None
        while rclpy.ok() and time.monotonic() < deadline:
            self._zero_base()
            rclpy.spin_once(self, timeout_sec=.1)
            pos = self._identity.position(self._locked_object_id)
            if pos is None or self._identity.updated == last_update:
                continue
            last_update = self._identity.updated
            ground_z = TENNIS_SPAWN_HEIGHT if self._current_class == TENNIS else 0
            on_ground = abs(pos[2]-ground_z) < .02 and abs(pos[1]-target_y) < .10
            quiet = previous is not None and max(abs(pos-previous)) < .004
            stable = stable+1 if on_ground and quiet else 0
            previous = pos.copy()
            if stable >= 5:
                return pos
        raise GraspActionError("release not verified on ground; no success count")

    def run_action(self):
        self._ensure_broadcaster()
        initial = self._wait_for_joint_state()
        self._initial = list(initial)
        self._active_initial_positions = list(initial)
        self._ensure_hold_controller(initial)
        self._ensure_wheel_controller()
        self._publish_to(self.wheel_publisher, [0.0]*4, repeat=10)
        self._zero_base()
        self._set_world_paused(False)
        self._wait_for_sim_time()
        current = self._open_and_fold(initial)
        self._write_ready_file(self.controller_ready_file)
        self._wait_for_scene_ready()
        self._wait_for_camera_info()
        self._identity = SortingIdentity(self._urdf_path)
        deadline = time.monotonic()+20
        while not self._identity.fresh() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=.1)
        if not self._identity.fresh():
            raise GraspActionError("Gazebo identity feedback unavailable")
        self._move_y(LEFT_END, current)
        passes = 0
        while rclpy.ok() and not self._ledger.done:
            self._pass_attempted.clear()
            self._pass_watched.clear()
            self._pass_empty_cells.clear()
            self._publish_status(f"SCAN PASS {passes+1} | direction={'right' if self._scan_direction < 0 else 'left'} | sorted={len(self._ledger.completed)}/6")
            end = RIGHT_END if self._scan_direction < 0 else LEFT_END
            while rclpy.ok() and not self._ledger.done:
                self._locked_object_id = None
                self._locked_object_origin = None
                self._set_vision_enabled(True, "lateral scan")
                self._last_frame_time = time.monotonic()
                detection = self._watch(current, .7)
                if detection is None:
                    detection = self._move_y(end, current, scan=True)
                if detection is None:
                    break
                checkpoint = self._pose().y
                name = detection["object_id"]
                sorted_this_target = False
                # Retry this same entity once before proceeding to another.
                for attempt in range(2):
                    try:
                        if attempt:
                            self._set_vision_enabled(True, "retry same object")
                        self._align_laterally(detection, current)
                        current = self._grasp_and_place(current)
                        sorted_this_target = True
                        break
                    except (AcquisitionMiss, MissedGraspError) as error:
                        self._zero_base()
                        self._carried = None
                        self._publish_status(f"GRASP FAILED | id={name} | attempt={attempt+1}/2 | {error} | count unchanged")
                        current = self._open_and_fold(list(self.last_position_command or current))
                        if attempt == 1:
                            self._pass_attempted.add(name)
                            self._publish_status(f"RETRY NEXT PASS | id={name} | two attempts failed")
                if self._ledger.done:
                    break
                self._set_vision_enabled(False, "resume lateral scan")
                if sorted_this_target:
                    # Continue just beyond the completed row instead of
                    # retracing its approach.  The next slow scan starts in
                    # front of the next possible object.
                    resume_y = float(self._locked_object_origin[1]) + .22*self._scan_direction
                    resume_y = min(LEFT_END, max(RIGHT_END, resume_y))
                    self._publish_status(
                        f"FAST RESUME | completed_y={self._locked_object_origin[1]:.2f} "
                        f"| next_scan_y={resume_y:.2f}"
                    )
                    self._move_y(resume_y, current, speed=self.empty_return_speed,
                                 arrival_tolerance=.02)
                else:
                    self._move_y(checkpoint, current, speed=self.empty_return_speed,
                                 arrival_tolerance=.02)
            passes += 1
            if self._ledger.done:
                break
            if self.max_passes and passes >= self.max_passes:
                raise GraspActionError(f"pass limit reached; only {len(self._ledger.completed)}/6 verified")
            self._scan_direction *= -1
            self._publish_status(f"RESCAN | remaining={6-len(self._ledger.completed)} | no false completion")
        if not self._ledger.done:
            raise GraspActionError("sorting interrupted before six verified releases")
        self._set_vision_enabled(False, "six verified releases")
        self._zero_base()
        self._publish_status("LINEAR SORTING COMPLETE | tennis=3 | bottle=3 | total=6/6")


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = None
    code = 1
    try:
        node = LinearSorter()
        node.run_action()
        code = 0
    except KeyboardInterrupt:
        code = 130
    except Exception as error:
        if node is not None:
            node.get_logger().error(f"LINEAR SORTING STOPPED | {error}")
        else:
            print(f"LINEAR SORTING START FAILED | {error}", flush=True)
    finally:
        if node is not None:
            node.safe_stop()
            if node._identity is not None:
                node._identity.close()
            node.close_log_file()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
