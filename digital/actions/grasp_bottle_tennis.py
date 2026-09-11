#!/usr/bin/env python3
"""Run two EP1 pickup modes: legacy bottle, then low-level tennis ball."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import rclpy
from rclpy.signals import SignalHandlerOptions

from grasp_cube import GraspActionError, GraspCubeAction, POSITION_JOINTS


WORLD_REMOVE_SERVICE = "/world/pick_place/remove"
REMOTE_MODEL_ROOT = (
    "/home/nvidia/robomaster_ep_static_gripper_fixed_20260910_122551/src"
)


class GraspBottleTennisAction(GraspCubeAction):
    def __init__(self):
        super().__init__()
        self.run_count = 2
        self.spawn_test_cube = False
        self.place_points = [self.place_points[0], self.place_points[0]]

        # The bottle keeps the original experiment2 arm motion.  Only the
        # tennis-ball round switches to the lower, level pickup configuration.
        self.bottle_arm_2_delta = self.arm_2_delta
        self.tennis_arm_2_delta = -0.78
        self._use_tennis_grasp = False
        self.turn_steps = 2500

        self.declare_parameter("model_root", REMOTE_MODEL_ROOT)
        self.declare_parameter("bottle_name", "task_water_bottle")
        self.declare_parameter("tennis_name", "task_tennis_ball")
        self.declare_parameter("bottle_mass", 0.20)
        self.declare_parameter("tennis_mass", 0.057)
        self.declare_parameter("object_friction", 5.0)
        # 333 calibrated 1 ms wheel steps at 3 rad/s and a 0.05 m wheel
        # radius move the chassis forward by approximately 0.05 m.
        self.declare_parameter("forward_approach_steps", 333)

        self.model_root = Path(
            str(self.get_parameter("model_root").value)
        ).expanduser()
        self.bottle_name = str(self.get_parameter("bottle_name").value)
        self.tennis_name = str(self.get_parameter("tennis_name").value)
        self.bottle_mass = float(self.get_parameter("bottle_mass").value)
        self.tennis_mass = float(self.get_parameter("tennis_mass").value)
        self.object_friction = float(
            self.get_parameter("object_friction").value
        )
        self.forward_approach_steps = int(
            self.get_parameter("forward_approach_steps").value
        )

        if min(
            self.bottle_mass,
            self.tennis_mass,
            self.object_friction,
        ) <= 0.0:
            raise GraspActionError(
                "bottle_mass、tennis_mass 和 object_friction 必须大于零"
            )
        if self.forward_approach_steps < 1:
            raise GraspActionError("forward_approach_steps 必须大于零")

    def _arm_pose(self, initial, fraction):
        pose = super()._arm_pose(initial, fraction)
        index = {name: i for i, name in enumerate(POSITION_JOINTS)}

        # arm_1 and arm_2 now produce 0.60 rad of downward gripper pitch at
        # full extension.  Counter-rotate the wrist by the same amount so the
        # gripper stays level throughout the approach without changing height.
        pitch_delta = self.arm_extend_delta + self.arm_2_delta
        pose[index["endpoint_bracket_joint"]] -= pitch_delta * fraction
        return pose

    @staticmethod
    def _vertical_clearance_pose(source, initial):
        index = {name: i for i, name in enumerate(POSITION_JOINTS)}
        pose = list(source)

        # This pose is an IK solution approximately 50 mm directly above the
        # pickup pose: x changes by less than 0.1 mm and the gripper stays level.
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
            and hasattr(self, "_active_initial_positions")
        ):
            initial = self._active_initial_positions
            index = {name: i for i, name in enumerate(POSITION_JOINTS)}
            clearance = self._vertical_clearance_pose(start, initial)
            retracted = self._gripper_pose(
                initial, initial, index, opened=True
            )
            super()._move("释放后先垂直抬高 5 cm", start, clearance)
            super()._move("抬高后向后收回机械臂", clearance, retracted)

            # The base routine publishes and returns this list after _move().
            # Keep its bookkeeping consistent with the substituted trajectory.
            target[:] = retracted
            return
        super()._move(label, start, target)

    def _safe_home_pose(self, initial, current, index):
        if not self._use_tennis_grasp:
            return super()._safe_home_pose(initial, current, index)

        current_open = self._gripper_pose(
            current, initial, index, opened=True
        )
        self._move_gripper("安全张开夹爪", current, current_open)
        clearance = self._vertical_clearance_pose(current_open, initial)
        super()._move("异常回收：先垂直抬高", current_open, clearance)
        home_open = self._gripper_pose(
            initial, initial, index, opened=True
        )
        super()._move("异常回收：抬高后向后收臂", clearance, home_open)
        self._settle(home_open)
        self._publish_status(
            "SAFE HOME COMPLETE: lifted first, then retracted"
        )
        return home_open

    def _run_once(self, attempt, initial, current, index):
        self._active_initial_positions = list(initial)
        return super()._run_once(attempt, initial, current, index)

    def _resource_uri(self, relative_path):
        path = (self.model_root / relative_path).resolve()
        if not path.is_file():
            raise GraspActionError(f"找不到模型资源：{path}")
        return path.as_uri()

    def _bottle_sdf(self):
        visual_scale = 0.65
        radius = 0.05445 * visual_scale
        length = 0.26044 * visual_scale
        mass = self.bottle_mass
        ixy = mass * (3.0 * radius**2 + length**2) / 12.0
        izz = 0.5 * mass * radius**2
        mesh_uri = self._resource_uri(
            "water_bottle/meshes/WaterBottle_fortress.obj"
        )
        return f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{self.bottle_name}">
    <static>false</static>
    <link name="bottle_link">
      <inertial>
        <mass>{mass:.9f}</mass>
        <inertia>
          <ixx>{ixy:.12f}</ixx><iyy>{ixy:.12f}</iyy>
          <izz>{izz:.12f}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="bottle_collision">
        <geometry>
          <cylinder><radius>{radius}</radius><length>{length}</length></cylinder>
        </geometry>
        <surface>
          <friction>
            <ode><mu>{self.object_friction}</mu><mu2>{self.object_friction}</mu2></ode>
          </friction>
          <contact><ode><kp>100000</kp><kd>10</kd></ode></contact>
        </surface>
      </collision>
      <visual name="bottle_visual">
        <geometry>
          <mesh><uri>{mesh_uri}</uri><scale>{visual_scale} {visual_scale} {visual_scale}</scale></mesh>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>
"""

    def _tennis_sdf(self):
        radius = 0.0335
        mass = self.tennis_mass
        inertia = 0.4 * mass * radius**2
        mesh_uri = self._resource_uri("056_tennis_ball/textured.obj")
        return f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{self.tennis_name}">
    <static>false</static>
    <link name="tennis_link">
      <gravity>false</gravity>
      <inertial>
        <mass>{mass:.9f}</mass>
        <inertia>
          <ixx>{inertia:.12f}</ixx><iyy>{inertia:.12f}</iyy>
          <izz>{inertia:.12f}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="tennis_collision">
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
        <surface>
          <friction>
            <ode><mu>{self.object_friction}</mu><mu2>{self.object_friction}</mu2></ode>
          </friction>
          <contact><ode><kp>100000</kp><kd>10</kd></ode></contact>
        </surface>
      </collision>
      <visual name="tennis_visual">
        <pose>-0.0082115 0.044278 -0.0331315 0 0 0</pose>
        <geometry><mesh><uri>{mesh_uri}</uri></mesh></geometry>
      </visual>
    </link>
  </model>
</sdf>
"""

    def _spawn_task_object(self, label, name, sdf_text, pose):
        self._remove_task_object(name, required=False)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".sdf", delete=False
        ) as stream:
            stream.write(sdf_text)
            sdf_path = stream.name
        try:
            self._run(
                [
                    "ros2",
                    "run",
                    "ros_gz_sim",
                    "create",
                    "-world",
                    "pick_place",
                    "-name",
                    name,
                    "-file",
                    sdf_path,
                    "-x",
                    str(pose[0]),
                    "-y",
                    str(pose[1]),
                    "-z",
                    str(pose[2]),
                ],
                timeout=30.0,
            )
        finally:
            Path(sdf_path).unlink(missing_ok=True)
        self._publish_status(
            f"TASK OBJECT READY | {label} | name={name} | "
            f"position=({pose[0]:.6f},{pose[1]:.6f},{pose[2]:.3f})"
        )

    def _remove_task_object(self, name, required=True):
        result = self._run(
            [
                "ign",
                "service",
                "-s",
                WORLD_REMOVE_SERVICE,
                "--reqtype",
                "ignition.msgs.Entity",
                "--reptype",
                "ignition.msgs.Boolean",
                "--timeout",
                str(self.world_service_timeout_ms),
                "--req",
                f'name: "{name}"',
            ],
            check=False,
            timeout=self.world_service_timeout_ms / 1000.0 + 5.0,
            log=False,
        )
        removed = (
            result.returncode == 0
            and "data: true" in result.stdout.lower()
        )
        if required and not removed:
            raise GraspActionError(
                f"无法移除上一轮物体 {name}：{result.stdout.strip()}"
            )
        return removed

    def _move_chassis_forward(self, held_positions):
        wheel_command = [self.wheel_speed] * 4
        duration = (
            self.forward_approach_steps
            * self.physics_step_seconds
            / self.speed_scale
        )
        self._publish_status(
            "PREP: MOVE CHASSIS FORWARD ABOUT 5 CM FOR EXTRA REACH"
        )
        try:
            def keep_moving(ratio):
                if ratio >= 1.0:
                    return
                self._publish(held_positions, repeat=1)
                self._publish_to(
                    self.wheel_publisher, wheel_command, repeat=1
                )

            self._wait_sim_duration(duration, callback=keep_moving)
        finally:
            self._publish_to(self.wheel_publisher, [0.0] * 4, repeat=10)
            self._publish(held_positions, repeat=10)
        self._wait_sim_duration(self.post_turn_settle_duration)

    def _restore_chassis_heading(self, held_positions):
        reverse_wheel_command = [
            self.wheel_speed,
            -self.wheel_speed,
            self.wheel_speed,
            -self.wheel_speed,
        ]
        turn_duration = (
            self.turn_steps
            * self.physics_step_seconds
            / self.speed_scale
        )
        self._publish_status(
            "BETWEEN ROUNDS: RESTORE INITIAL CHASSIS HEADING"
        )
        try:
            def keep_turning(ratio):
                if ratio >= 1.0:
                    return
                self._publish(held_positions, repeat=1)
                self._publish_to(
                    self.wheel_publisher,
                    reverse_wheel_command,
                    repeat=1,
                )

            self._wait_sim_duration(turn_duration, callback=keep_turning)
        finally:
            self._publish_to(self.wheel_publisher, [0.0] * 4, repeat=10)
            self._publish(held_positions, repeat=10)
        self._wait_sim_duration(self.post_turn_settle_duration)

    def run_action(self):
        if shutil.which("ros2") is None or shutil.which("ign") is None:
            raise GraspActionError("找不到 ros2 或 ign，请先 source ROS 2 环境")

        self.get_logger().info(
            "TWO-OBJECT ACTION | round 1=legacy water-bottle grasp | "
            "round 2=low-level tennis-ball grasp"
        )
        self.get_logger().info("准备 1/2：激活只读关节状态")
        self._ensure_broadcaster()
        initial = self._wait_for_joint_state()

        self.get_logger().info("准备 2/2：原位锁定全部位置关节并启用车轮控制")
        self._ensure_hold_controller(initial)
        self._ensure_wheel_controller()

        self._publish(initial, repeat=10)
        self._publish_to(self.wheel_publisher, [0.0] * 4, repeat=10)
        previous_sim_time = self.latest_sim_time
        self._publish_status("PREP: START CONTINUOUS GAZEBO EXECUTION")
        self._set_world_paused(False)
        self._wait_for_sim_time(after=previous_sim_time)

        index = {name: i for i, name in enumerate(POSITION_JOINTS)}
        self._publish_status("PREP: OPEN GRIPPER BEFORE ALL STEPS")
        startup_open = self._gripper_pose(
            initial, initial, index, opened=True
        )
        self._move_gripper("开始所有步骤前先张开夹爪", initial, startup_open)
        self._settle(startup_open)

        tasks = (
            (
                "WATER BOTTLE",
                self.bottle_name,
                self._bottle_sdf,
                (self.cube_x, self.cube_y, 0.085),
            ),
            (
                "TENNIS BALL",
                self.tennis_name,
                self._tennis_sdf,
                (self.cube_x, self.cube_y, 0.061),
            ),
        )

        current = list(startup_open)
        success_count = 0
        failed_count = 0
        previous_name = None

        for attempt, (label, name, sdf_factory, pose) in enumerate(
            tasks, start=1
        ):
            self._use_tennis_grasp = label == "TENNIS BALL"
            self.arm_2_delta = (
                self.tennis_arm_2_delta
                if self._use_tennis_grasp
                else self.bottle_arm_2_delta
            )
            grasp_mode = (
                "LOWER LEVEL TENNIS GRASP"
                if self._use_tennis_grasp
                else "ORIGINAL EXPERIMENT2 BOTTLE GRASP"
            )
            self._publish_status(
                f"ROUND {attempt}/2 GRASP MODE | {grasp_mode}"
            )

            if previous_name is not None:
                self._publish_status(
                    f"PREP ROUND {attempt}: REMOVE PREVIOUS {previous_name}"
                )
                self._remove_task_object(previous_name)
                self._settle(current)
                self._restore_chassis_heading(current)

            self._publish_status(f"PREP ROUND {attempt}: CREATE {label}")
            self._spawn_task_object(label, name, sdf_factory(), pose)
            self._settle(current)

            # Only the lower tennis pose loses forward arm reach.  Apply the
            # compensating chassis approach after round-one heading recovery,
            # immediately before the tennis pickup.
            if self._use_tennis_grasp:
                self._move_chassis_forward(current)

            self._publish_status(
                f"ROUND {attempt}/2 START | target={label}"
            )
            ok, current = self._run_once(
                attempt, initial, current, index
            )
            if not ok:
                failed_count += 1
                break
            success_count += 1
            previous_name = name

            if attempt == 1 and self.cycle_pause_seconds > 0.0:
                self._publish_status(
                    "ROUND 1 BOTTLE COMPLETE | preparing tennis ball"
                )
                self._wait_sim_duration(self.cycle_pause_seconds)

        self._publish_status(
            f"FINAL RESULT | success: {success_count} | failed: "
            f"{failed_count} | planned runs: 2"
        )
        if success_count == 2:
            self._publish_status(
                "TWO-OBJECT TASK COMPLETE | bottle then tennis ball"
            )


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
            print(f"两物体动作启动失败：{error}")
    finally:
        if node is not None:
            node.safe_stop()
            node.close_log_file()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
