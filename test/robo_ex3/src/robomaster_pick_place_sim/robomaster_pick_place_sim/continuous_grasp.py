#!/usr/bin/env python3
"""在不修改模型文件的前提下执行五次 ROS 2 搬运任务。

适用模型: robomaster_ep_static_gripper_fixed_20260910_122551

安全策略:
1. 仅在加载控制器时保持 Gazebo 暂停并执行少量启动单步。
2. 控制器锁定全部关节后解除暂停，按 /joint_states 时间戳连续执行。
3. 只按动作需要改变主机械臂、夹爪根关节和四个轮关节。
4. 结束或异常时先停止车轮、保持关节，再重新暂停 Gazebo。
5. 临时控制器参数写到 /tmp；不读写 URDF、SDF 或模型目录。
"""

from __future__ import annotations

from datetime import datetime
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String


WORLD_CONTROL_SERVICE = "/world/pick_place/control"
ACTION_VERSION = "2026-09-11-task-points-v15"
BROADCASTER = "joint_state_broadcaster"
HOLD_CONTROLLER = "robomaster_position_hold_controller"
HOLD_TOPIC = f"/{HOLD_CONTROLLER}/commands"
WHEEL_CONTROLLER = "robomaster_wheel_velocity_controller"
WHEEL_TOPIC = f"/{WHEEL_CONTROLLER}/commands"
FORWARD_CONTROLLER_TYPE = (
    "forward_command_controller/ForwardCommandController"
)

# 顺序必须与发送给 ForwardCommandController 的数组完全一致。
POSITION_JOINTS = (
    "arm_1_joint",
    "arm_2_joint",
    "endpoint_bracket_joint",
    "rod_1_joint",
    "rod_2_joint",
    "rod_3_joint",
    "rod_joint",
    "triangle_joint",
    "gripper_m_joint",
    "left_gripper_joint_1",
    "left_gripper_joint_2",
    "left_gripper_joint_4",
    "left_gripper_joint_5",
    "left_gripper_joint_6",
    "left_gripper_joint_7",
    "right_gripper_joint_1",
    "right_gripper_joint_2",
    "right_gripper_joint_4",
    "right_gripper_joint_5",
    "right_gripper_joint_6",
    "right_gripper_joint_7",
)

LEFT_ROOT = "left_gripper_joint_1"
RIGHT_ROOT = "right_gripper_joint_1"
ARM_1 = "arm_1_joint"
ARM_2 = "arm_2_joint"

VELOCITY_JOINTS = (
    "front_left_wheel_joint",
    "front_right_wheel_joint",
    "rear_left_wheel_joint",
    "rear_right_wheel_joint",
)

# 以世界坐标系 x 轴为初始取物方向。每次向右转约 30 度后，物体仍在
# 距原点 0.35 m 的圆弧上；y 轴向左为正，因此右转后的 y 为负。
DEFAULT_PLACE_POINTS = (
    (0.303109, -0.175000, 0.061),
    (0.175000, -0.303109, 0.061),
    (0.000000, -0.350000, 0.061),
    (-0.175000, -0.303109, 0.061),
    (-0.303109, -0.175000, 0.061),
)


class GraspActionError(RuntimeError):
    pass


class FileMirroringLogger:
    """把节点日志同时写到 rclpy 控制台和普通文本文件。"""

    def __init__(self, ros_logger, stream):
        self._ros_logger = ros_logger
        self._stream = stream

    def __getattr__(self, name):
        return getattr(self._ros_logger, name)

    def _emit(self, method, level, message):
        text = str(message)
        getattr(self._ros_logger, method)(text)
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        self._stream.write(f"{timestamp} [{level}] {text}\n")
        self._stream.flush()

    def debug(self, message):
        self._emit("debug", "DEBUG", message)

    def info(self, message):
        self._emit("info", "INFO", message)

    def warning(self, message):
        self._emit("warning", "WARN", message)

    def error(self, message):
        self._emit("error", "ERROR", message)

    def fatal(self, message):
        self._emit("fatal", "FATAL", message)


class GraspCubeAction(Node):
    def __init__(self):
        super().__init__("grasp_cube_action")

        self._file_logger = None
        self._log_stream = None
        self.declare_parameter("log_dir", "/home/nvidia/ros2_actions/logs")
        self.log_dir = str(self.get_parameter("log_dir").value)
        self._open_log_file()

        # 相对当前标定姿态的对称转角。正的 open_offset 增大开口；
        # 正的 close_offset 从标定姿态向内闭合。
        # 对照实机完全张开照片：根连杆应接近 80 度。夹爪根关节是
        # continuous，因此用 0.70 rad 作为本动作的软件最大开度。
        self.declare_parameter("open_offset", 0.70)
        # 闭合时回到已经调好的标定角，不再额外向内压，避免穿模。
        self.declare_parameter("close_offset", 0.00)
        self.declare_parameter("run_count", 5)
        self.declare_parameter("spawn_test_cube", True)
        self.declare_parameter("cube_name", "grasp_test_cube")
        self.declare_parameter("cube_length", 0.05)
        self.declare_parameter("cube_width", 0.05)
        self.declare_parameter("cube_height", 0.12)
        self.declare_parameter("cube_mass", 0.20)
        self.declare_parameter("cube_friction", 5.0)
        # 柱体前后位置参数；运行时可用 --ros-args -p cube_x:=VALUE 覆盖。
        self.declare_parameter("cube_x", 0.35)
        self.declare_parameter("cube_y", 0.0)
        # 12 cm 高柱体的中心位于 6 cm 处，另加 1 mm 离地余量。
        self.declare_parameter("cube_z", 0.061)
        self.declare_parameter("turn_angle_degrees", -30.0)
        for number, point in enumerate(DEFAULT_PLACE_POINTS, start=1):
            self.declare_parameter(f"place_point_{number}", list(point))
        self.declare_parameter("safe_height", 0.45)
        # 正式动作按仿真时间连续插值，不再为每一帧调用 Gazebo 单步服务。
        # 1.0 为原实测速度；默认 4.0，可在 1.0～5.0 之间调整。
        self.declare_parameter("speed_scale", 4.0)
        self.declare_parameter("command_rate_hz", 100.0)
        self.declare_parameter("motion_duration_seconds", 1.5)
        self.declare_parameter("gripper_duration_seconds", 2.0)
        self.declare_parameter("settle_duration_seconds", 0.35)
        self.declare_parameter("gripper_settle_seconds", 0.75)
        self.declare_parameter("cycle_pause_seconds", 0.5)
        # arm_1_joint 的模型上限为 1.384 rad；1.38 rad 是本动作的
        # 最低位置，仅保留约 0.004 rad 防止浮点误差越过硬限位。
        self.declare_parameter("arm_extend_delta", 1.38)
        self.declare_parameter("arm_initial_fraction", 0.45)
        self.declare_parameter("arm_alignment_fraction", 0.58)
        self.declare_parameter("arm_coarse_fraction", 0.92)
        self.declare_parameter("arm_test_lift_delta", 1.22)
        self.declare_parameter("arm_lift_delta", 0.95)
        self.declare_parameter("arm_2_delta", -1.38)
        self.declare_parameter("wheel_speed", 3.0)
        # 实测 1050 个 0.001 s 仿真步约为 180 度；连续模式换算为 1.05 s。
        self.declare_parameter("turn_steps", 1050)
        self.declare_parameter("physics_step_seconds", 0.001)
        self.declare_parameter("post_turn_settle_seconds", 0.30)
        self.declare_parameter("world_service_timeout_ms", 15000)
        self.declare_parameter("step_retries", 2)
        self.declare_parameter("stop_on_empty_grasp", True)
        self.declare_parameter("grasp_position_tolerance", 0.02)
        self.declare_parameter("controller_timeout_seconds", 20.0)

        self.open_offset = float(self.get_parameter("open_offset").value)
        self.close_offset = float(self.get_parameter("close_offset").value)
        self.run_count = int(self.get_parameter("run_count").value)
        self.spawn_test_cube = bool(
            self.get_parameter("spawn_test_cube").value
        )
        self.cube_name = str(self.get_parameter("cube_name").value)
        self.cube_length = float(self.get_parameter("cube_length").value)
        self.cube_width = float(self.get_parameter("cube_width").value)
        self.cube_height = float(self.get_parameter("cube_height").value)
        self.cube_mass = float(self.get_parameter("cube_mass").value)
        self.cube_friction = float(
            self.get_parameter("cube_friction").value
        )
        self.cube_x = float(self.get_parameter("cube_x").value)
        self.cube_y = float(self.get_parameter("cube_y").value)
        self.cube_z = float(self.get_parameter("cube_z").value)
        self.turn_angle_degrees = float(
            self.get_parameter("turn_angle_degrees").value
        )
        self.place_points = []
        for number in range(1, len(DEFAULT_PLACE_POINTS) + 1):
            point = list(self.get_parameter(f"place_point_{number}").value)
            if len(point) != 3:
                raise GraspActionError(
                    f"place_point_{number} 必须包含 [x, y, z] 三个数值"
                )
            self.place_points.append(tuple(float(value) for value in point))
        self.safe_height = float(self.get_parameter("safe_height").value)
        self.speed_scale = float(self.get_parameter("speed_scale").value)
        if not 1.0 <= self.speed_scale <= 5.0:
            raise GraspActionError("speed_scale 必须在 1.0～5.0 之间")
        self.command_rate_hz = float(
            self.get_parameter("command_rate_hz").value
        )
        self.motion_duration = float(
            self.get_parameter("motion_duration_seconds").value
        ) / self.speed_scale
        self.gripper_duration = float(
            self.get_parameter("gripper_duration_seconds").value
        ) / self.speed_scale
        self.settle_duration = float(
            self.get_parameter("settle_duration_seconds").value
        ) / self.speed_scale
        self.gripper_settle_duration = float(
            self.get_parameter("gripper_settle_seconds").value
        ) / self.speed_scale
        self.cycle_pause_seconds = float(
            self.get_parameter("cycle_pause_seconds").value
        ) / self.speed_scale
        self.arm_extend_delta = float(
            self.get_parameter("arm_extend_delta").value
        )
        self.arm_initial_fraction = float(
            self.get_parameter("arm_initial_fraction").value
        )
        self.arm_alignment_fraction = float(
            self.get_parameter("arm_alignment_fraction").value
        )
        self.arm_coarse_fraction = float(
            self.get_parameter("arm_coarse_fraction").value
        )
        self.arm_test_lift_delta = float(
            self.get_parameter("arm_test_lift_delta").value
        )
        self.arm_lift_delta = float(self.get_parameter("arm_lift_delta").value)
        self.arm_2_delta = float(self.get_parameter("arm_2_delta").value)
        self.wheel_speed = (
            float(self.get_parameter("wheel_speed").value)
            * self.speed_scale
        )
        self.turn_steps = int(self.get_parameter("turn_steps").value)
        self.physics_step_seconds = float(
            self.get_parameter("physics_step_seconds").value
        )
        self.post_turn_settle_duration = float(
            self.get_parameter("post_turn_settle_seconds").value
        ) / self.speed_scale
        self.world_service_timeout_ms = int(
            self.get_parameter("world_service_timeout_ms").value
        )
        self.step_retries = int(self.get_parameter("step_retries").value)
        self.stop_on_empty_grasp = bool(
            self.get_parameter("stop_on_empty_grasp").value
        )
        self.grasp_position_tolerance = float(
            self.get_parameter("grasp_position_tolerance").value
        )
        self.controller_timeout = float(
            self.get_parameter("controller_timeout_seconds").value
        )
        if min(self.open_offset, self.close_offset) < 0.0:
            raise GraspActionError("open_offset 和 close_offset 不能为负数")
        if self.run_count < 1:
            raise GraspActionError("run_count 必须大于零")
        if min(
            self.cube_length,
            self.cube_width,
            self.cube_height,
            self.cube_mass,
            self.cube_friction,
        ) <= 0.0:
            raise GraspActionError("柱体长、宽、高、质量和摩擦系数必须大于零")
        if not 0.0 < abs(self.turn_angle_degrees) <= 180.0:
            raise GraspActionError("turn_angle_degrees 必须在 -180～180 度内且不能为零")
        if not 0.0 < self.safe_height < 0.5:
            raise GraspActionError("safe_height 必须大于 0 且小于 0.5 m")
        if re.fullmatch(r"[A-Za-z0-9_-]+", self.cube_name) is None:
            raise GraspActionError(
                "cube_name 只能包含字母、数字、下划线和连字符"
            )
        if min(
            self.command_rate_hz,
            self.motion_duration,
            self.gripper_duration,
            self.settle_duration,
            self.gripper_settle_duration,
            self.physics_step_seconds,
            self.post_turn_settle_duration,
        ) <= 0.0:
            raise GraspActionError("连续动作频率、持续时间和仿真步长必须大于零")
        fractions = (
            self.arm_initial_fraction,
            self.arm_alignment_fraction,
            self.arm_coarse_fraction,
        )
        if not (0.0 < fractions[0] < fractions[1] < fractions[2] < 1.0):
            raise GraspActionError(
                "机械臂分段比例必须满足 0 < initial < alignment < coarse < 1"
            )
        if not (
            self.arm_extend_delta
            >= self.arm_test_lift_delta
            >= self.arm_lift_delta
        ):
            raise GraspActionError(
                "机械臂参数必须满足 extend >= test_lift >= main_lift"
            )
        if self.turn_steps < 1:
            raise GraspActionError("turn_steps 必须大于零")
        if (
            self.world_service_timeout_ms < 1000
            or self.step_retries < 1
        ):
            raise GraspActionError(
                "world service timeout 至少为 1000，"
                "step_retries 至少为 1"
            )

        self.latest_joint_state = None
        self.latest_sim_time = None
        self.last_position_command = None
        self.world_running = False
        self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            qos_profile_sensor_data,
        )
        self.command_publisher = self.create_publisher(
            Float64MultiArray, HOLD_TOPIC, 10
        )
        self.wheel_publisher = self.create_publisher(
            Float64MultiArray, WHEEL_TOPIC, 10
        )
        self.status_publisher = self.create_publisher(
            String, "/real_pick_place/status", 10
        )

        self.get_logger().info(f"日志文件：{self.log_path}")

    def get_logger(self):
        ros_logger = super().get_logger()
        return getattr(self, "_file_logger", None) or ros_logger

    def _open_log_file(self):
        try:
            directory = Path(self.log_dir).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            filename = (
                "continuous_grasp_"
                + datetime.now().strftime("%Y%m%d_%H%M%S")
                + f"_{os.getpid()}.log"
            )
            self.log_path = str(directory / filename)
            self._log_stream = open(
                self.log_path, "a", encoding="utf-8", buffering=1
            )
            self._file_logger = FileMirroringLogger(
                super().get_logger(), self._log_stream
            )
        except OSError as error:
            self.log_path = "<日志文件创建失败>"
            super().get_logger().warning(f"无法创建日志文件：{error}")

    def close_log_file(self):
        if self._log_stream is not None:
            self._log_stream.flush()
            self._log_stream.close()
            self._log_stream = None
            self._file_logger = None

    def _joint_state_callback(self, message):
        self.latest_joint_state = message
        self.latest_sim_time = (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1.0e-9
        )

    def _run(self, command, check=True, timeout=30.0, log=True):
        if log:
            self.get_logger().info("执行: " + " ".join(command))
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        if log and result.stdout.strip():
            self.get_logger().info(result.stdout.strip())
        if check and result.returncode != 0:
            raise GraspActionError(
                f"命令失败（返回码 {result.returncode}）: {' '.join(command)}"
            )
        return result

    def _controller_states(self):
        result = self._run(
            ["ros2", "control", "list_controllers"], timeout=10.0
        )
        states = {}
        for line in result.stdout.splitlines():
            # ros2 control 会按终端配置插入 ANSI 颜色码；若不清除，
            # 控制器名字可能变成 "\x1b[...mjoint_state_broadcaster"，
            # 从而被误判成尚未加载。
            clean_line = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line)
            fields = clean_line.split()
            if len(fields) >= 3:
                state = fields[-1].lower()
                if state in {"active", "inactive", "unconfigured", "finalized"}:
                    states[fields[0]] = state
        return states

    def _step(self, count, allow_retry=True, timeout_ms=None):
        """仅用于暂停状态下加载控制器的启动单步。"""
        attempts = self.step_retries if allow_retry else 1
        request_timeout_ms = (
            self.world_service_timeout_ms
            if timeout_ms is None
            else int(timeout_ms)
        )
        command = [
            "ign",
            "service",
            "-s",
            WORLD_CONTROL_SERVICE,
            "--reqtype",
            "ignition.msgs.WorldControl",
            "--reptype",
            "ignition.msgs.Boolean",
            "--timeout",
            str(request_timeout_ms),
            "--req",
            f"multi_step: {int(count)}",
        ]
        last_output = ""
        for attempt in range(1, attempts + 1):
            try:
                result = self._run(
                    command,
                    check=False,
                    timeout=request_timeout_ms / 1000.0 + 5.0,
                    log=False,
                )
                last_output = result.stdout.strip()
                if result.returncode == 0 and "data: true" in result.stdout.lower():
                    return
            except subprocess.TimeoutExpired:
                last_output = "ign service 进程等待超时"

            if attempt < attempts:
                self.get_logger().warning(
                    f"Gazebo 单步请求失败，第 {attempt}/{attempts} 次；正在重试"
                )

        raise GraspActionError(
            f"Gazebo 单步服务连续 {attempts} 次未成功：{last_output}"
        )

    def _set_world_paused(self, paused):
        command = [
            "ign",
            "service",
            "-s",
            WORLD_CONTROL_SERVICE,
            "--reqtype",
            "ignition.msgs.WorldControl",
            "--reptype",
            "ignition.msgs.Boolean",
            "--timeout",
            str(self.world_service_timeout_ms),
            "--req",
            f"pause: {'true' if paused else 'false'}",
        ]
        try:
            result = self._run(
                command,
                check=False,
                timeout=self.world_service_timeout_ms / 1000.0 + 5.0,
                log=False,
            )
        except subprocess.TimeoutExpired as error:
            raise GraspActionError(
                "暂停 Gazebo 超时" if paused else "解除 Gazebo 暂停超时"
            ) from error
        if result.returncode != 0 or "data: true" not in result.stdout.lower():
            raise GraspActionError(
                ("暂停" if paused else "解除暂停")
                + f" Gazebo 失败：{result.stdout.strip()}"
            )
        self.world_running = not paused
        self.get_logger().info(
            "Gazebo 已暂停" if paused else "Gazebo 已进入连续运行模式"
        )

    def _wait_for_sim_time(self, after=None, timeout_seconds=10.0):
        deadline = time.monotonic() + timeout_seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.10)
            current = self.latest_sim_time
            if current is None:
                continue
            if after is None or current > after + 1.0e-9:
                return current
        raise GraspActionError(
            "连续模式下 /joint_states 时间戳未更新，"
            "请确认 Gazebo 已解除暂停且 broadcaster 为 active"
        )

    def _wait_sim_duration(self, duration_seconds, callback=None):
        duration = float(duration_seconds)
        if duration <= 0.0:
            if callback is not None:
                callback(1.0)
            return

        start = self._wait_for_sim_time(after=self.latest_sim_time)
        deadline = time.monotonic() + max(30.0, duration * 50.0)
        next_command_time = start
        command_period = 1.0 / self.command_rate_hz
        if callback is not None:
            callback(0.0)

        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.05, command_period))
            current = self.latest_sim_time
            if current is None:
                continue
            elapsed = max(0.0, current - start)
            if elapsed >= duration:
                if callback is not None:
                    callback(1.0)
                return
            if callback is not None and current >= next_command_time:
                callback(elapsed / duration)
                next_command_time = current + command_period

        if not rclpy.ok():
            raise KeyboardInterrupt
        raise GraspActionError(
            f"等待 {duration:.2f} 秒仿真时间超时，"
            "/joint_states 时间戳可能已经停止"
        )

    def _activate_with_steps(self, controller):
        process = subprocess.Popen(
            ["ros2", "control", "set_controller_state", controller, "active"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + self.controller_timeout
        output = ""
        while process.poll() is None and time.monotonic() < deadline:
            self._step(10)
            rclpy.spin_once(self, timeout_sec=0.05)
        if process.poll() is None:
            process.terminate()
            raise GraspActionError(f"激活 {controller} 超时")
        output = process.communicate(timeout=2.0)[0] or ""
        if output.strip():
            self.get_logger().info(output.strip())
        if process.returncode != 0:
            raise GraspActionError(f"无法激活控制器 {controller}")

    def _ensure_broadcaster(self):
        states = self._controller_states()
        if BROADCASTER not in states:
            self._run(
                [
                    "ros2",
                    "run",
                    "controller_manager",
                    "spawner",
                    BROADCASTER,
                    "--controller-manager",
                    "/controller_manager",
                    "--inactive",
                ],
                timeout=30.0,
            )
            states = self._controller_states()
        if states.get(BROADCASTER) != "active":
            self._activate_with_steps(BROADCASTER)

    def _wait_for_joint_state(self):
        required = set(POSITION_JOINTS)
        deadline = time.monotonic() + self.controller_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            self._step(10)
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.latest_joint_state is None:
                continue
            if required.issubset(self.latest_joint_state.name):
                values = dict(
                    zip(
                        self.latest_joint_state.name,
                        self.latest_joint_state.position,
                    )
                )
                return [float(values[name]) for name in POSITION_JOINTS]
        raise GraspActionError("没有收到包含全部位置关节的 /joint_states")

    def _controller_parameter_file(self, controller, joints, interface):
        joint_lines = "\n".join(f"      - {name}" for name in joints)
        return f"""controller_manager:
  ros__parameters:
    {controller}:
      type: {FORWARD_CONTROLLER_TYPE}

{controller}:
  ros__parameters:
    joints:
{joint_lines}
    interface_name: {interface}
"""

    def _register_runtime_controller_type(self, controller):
        """兼容 Humble：在 load_controller 前显式注册插件类型。"""
        available = self._run(
            ["ros2", "control", "list_controller_types"], timeout=10.0
        )
        clean_output = re.sub(
            r"\x1b\[[0-?]*[ -/]*[@-~]", "", available.stdout
        )
        if FORWARD_CONTROLLER_TYPE not in clean_output:
            raise GraspActionError(
                "系统没有提供 " + FORWARD_CONTROLLER_TYPE
                + "；请把 ros2 control list_controller_types 的输出发给我"
            )

        result = self._run(
            [
                "ros2",
                "param",
                "set",
                "/controller_manager",
                f"{controller}.type",
                FORWARD_CONTROLLER_TYPE,
            ],
            timeout=10.0,
        )
        if "successful" not in result.stdout.lower():
            raise GraspActionError(
                f"controller_manager 未接受 {controller} 的 type 参数"
            )

    def _remove_orphaned_controller(self, controller):
        """清除上次失败后可能残留的已加载/未配置控制器。"""
        result = self._run(
            ["ros2", "control", "unload_controller", controller],
            check=False,
            timeout=10.0,
        )
        if result.returncode == 0:
            self.get_logger().info(f"已清除残留控制器 {controller}")
            return

        # Humble 对“控制器不存在”只返回通用的 Error unloading controllers，
        # 无法据此和真正的卸载错误区分。这里本来就是可选的残留清理，
        # 后续 spawner 会给出权威结果，因此失败时记录并继续。
        self.get_logger().info(
            f"没有可清除的 {controller} 残留实例，继续创建新控制器"
        )

    def _spawn_active_with_steps(
        self, controller, parameter_path, initial_command, publisher
    ):
        """让 spawner 在保持初始命令的同时一次完成加载和激活。"""
        command = [
            "ros2",
            "run",
            "controller_manager",
            "spawner",
            controller,
            "--controller-manager",
            "/controller_manager",
            "--param-file",
            parameter_path,
        ]
        self.get_logger().info("执行并单步激活: " + " ".join(command))
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + self.controller_timeout
        while process.poll() is None and time.monotonic() < deadline:
            # 控制器话题一出现就会收到真实初始值；激活等待期间持续推进
            # 少量仿真步，使 controller_manager 的切换周期能够完成。
            self._publish_to(publisher, initial_command, repeat=3)
            self._step(10)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
            raise GraspActionError(f"加载并激活 {controller} 超时")
        output = process.communicate(timeout=2.0)[0] or ""
        if output.strip():
            self.get_logger().info(output.strip())
        if process.returncode != 0:
            raise GraspActionError(
                f"无法加载并激活控制器 {controller}；spawner 输出：{output.strip()}"
            )

    def _ensure_runtime_controller(
        self, controller, joints, interface, initial_command, publisher
    ):
        states = self._controller_states()
        if controller not in states:
            # Humble 可能不在 list_controllers 中显示加载失败后的残留实例，
            # 但下一次 spawner 会报告 "Controller already loaded"。
            self._remove_orphaned_controller(controller)
            self._register_runtime_controller_type(controller)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".yaml", delete=False
            ) as stream:
                stream.write(
                    self._controller_parameter_file(controller, joints, interface)
                )
                parameter_path = stream.name
            try:
                self._spawn_active_with_steps(
                    controller,
                    parameter_path,
                    initial_command,
                    publisher,
                )
            finally:
                Path(parameter_path).unlink(missing_ok=True)

        states = self._controller_states()
        if controller not in states:
            raise GraspActionError(
                f"{controller} 的 spawner 已结束，但控制器未保留在 controller_manager 中"
            )
        # 对于运行前已经存在但处于 inactive 的控制器，仍采用单独激活流程。
        self._publish_to(publisher, initial_command, repeat=10)
        if states.get(controller) != "active":
            self._activate_with_steps(controller)
        self._publish_to(publisher, initial_command, repeat=10)
        self._step(20)

    def _ensure_hold_controller(self, initial_positions):
        self._ensure_runtime_controller(
            HOLD_CONTROLLER,
            POSITION_JOINTS,
            "position",
            initial_positions,
            self.command_publisher,
        )

    def _ensure_wheel_controller(self):
        self._ensure_runtime_controller(
            WHEEL_CONTROLLER,
            VELOCITY_JOINTS,
            "velocity",
            [0.0] * len(VELOCITY_JOINTS),
            self.wheel_publisher,
        )

    def _publish_to(self, publisher, values, repeat=3):
        message = Float64MultiArray(data=[float(value) for value in values])
        for _ in range(repeat):
            publisher.publish(message)
            rclpy.spin_once(self, timeout_sec=0.03)

    def _publish(self, positions, repeat=3):
        self.last_position_command = list(positions)
        self._publish_to(self.command_publisher, positions, repeat)

    def safe_stop(self):
        """停止车轮、保持机械臂，然后重新暂停连续运行的 Gazebo。"""
        if not rclpy.ok():
            return
        try:
            self._publish_to(self.wheel_publisher, [0.0] * 4, repeat=10)
            if self.last_position_command is not None:
                self._publish(self.last_position_command, repeat=10)
            if self.world_running:
                self._wait_sim_duration(0.15)
        except Exception as error:
            self.get_logger().warning(f"零速和关节保持确认未完全执行：{error}")
        finally:
            if self.world_running and shutil.which("ign") is not None:
                try:
                    self._set_world_paused(True)
                except Exception as error:
                    self.get_logger().warning(f"Gazebo 最终暂停失败：{error}")

    def _move_interpolated(self, label, start, target, duration_seconds):
        self.get_logger().info(label)
        def publish_at(ratio):
            # 三次平滑插值让起点和终点速度均为零，减少夹取冲击。
            blend = ratio * ratio * (3.0 - 2.0 * ratio)
            command = [
                begin + (end - begin) * blend
                for begin, end in zip(start, target)
            ]
            self._publish(command, repeat=1)

        self._wait_sim_duration(duration_seconds, callback=publish_at)
        self._publish(target, repeat=3)

    def _move(self, label, start, target):
        self._move_interpolated(
            label,
            start,
            target,
            self.motion_duration,
        )

    def _move_gripper(self, label, start, target):
        self._move_interpolated(
            label,
            start,
            target,
            self.gripper_duration,
        )

    def _publish_status(self, text):
        message = String()
        message.data = str(text)
        self.status_publisher.publish(message)
        self.get_logger().info(str(text))
        rclpy.spin_once(self, timeout_sec=0.05)

    def _cube_sdf(self):
        ixx = self.cube_mass * (
            self.cube_width**2 + self.cube_height**2
        ) / 12.0
        iyy = self.cube_mass * (
            self.cube_length**2 + self.cube_height**2
        ) / 12.0
        izz = self.cube_mass * (
            self.cube_length**2 + self.cube_width**2
        ) / 12.0
        dimensions = (
            f"{self.cube_length} {self.cube_width} {self.cube_height}"
        )
        return f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{self.cube_name}">
    <static>false</static>
    <link name="cube_link">
      <inertial>
        <mass>{self.cube_mass:.9f}</mass>
        <inertia>
          <ixx>{ixx:.12f}</ixx>
          <iyy>{iyy:.12f}</iyy>
          <izz>{izz:.12f}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="cube_collision">
        <geometry><box><size>{dimensions}</size></box></geometry>
        <surface>
          <friction><ode><mu>{self.cube_friction}</mu><mu2>{self.cube_friction}</mu2></ode></friction>
          <contact><ode><kp>100000</kp><kd>10</kd></ode></contact>
        </surface>
      </collision>
      <visual name="cube_visual">
        <geometry><box><size>{dimensions}</size></box></geometry>
        <material>
          <ambient>0.85 0.12 0.05 1</ambient>
          <diffuse>0.95 0.18 0.06 1</diffuse>
          <specular>0.20 0.20 0.20 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""

    def _spawn_cube(self):
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".sdf", delete=False
        ) as stream:
            stream.write(self._cube_sdf())
            sdf_path = stream.name
        try:
            self._run(
                [
                    "ros2",
                    "run",
                    "ros_gz_sim",
                    "create",
                    "-name",
                    self.cube_name,
                    "-file",
                    sdf_path,
                    "-x",
                    str(self.cube_x),
                    "-y",
                    str(self.cube_y),
                    "-z",
                    str(self.cube_z),
                ],
                timeout=30.0,
            )
        finally:
            Path(sdf_path).unlink(missing_ok=True)

        self._publish_status(
            f"TEST BLOCK CREATED | name={self.cube_name} | "
            f"size=({self.cube_length:.3f}, {self.cube_width:.3f}, "
            f"{self.cube_height:.3f}) m | "
            f"friction={self.cube_friction:.2f} | "
            f"position=({self.cube_x:.3f}, {self.cube_y:.3f}, "
            f"{self.cube_z:.3f})"
        )

    def _report_step(self, attempt, number, label):
        self._publish_status(
            f"RUN {attempt}/{self.run_count} | STEP {number}: {label}"
        )

    def _settle(self, held_positions, duration_seconds=None):
        self._publish(held_positions, repeat=5)
        duration = (
            self.settle_duration
            if duration_seconds is None
            else float(duration_seconds)
        )
        self._wait_sim_duration(duration)

    def _arm_pose(self, initial, fraction):
        index = {name: i for i, name in enumerate(POSITION_JOINTS)}
        pose = list(initial)
        pose[index[ARM_1]] += self.arm_extend_delta * fraction
        pose[index[ARM_2]] += self.arm_2_delta * fraction
        return pose

    @staticmethod
    def _copy_gripper_pose(pose, source, index):
        result = list(pose)
        result[index[LEFT_ROOT]] = source[index[LEFT_ROOT]]
        result[index[RIGHT_ROOT]] = source[index[RIGHT_ROOT]]
        return result

    def _gripper_pose(self, pose, initial, index, opened):
        result = list(pose)
        if opened:
            result[index[LEFT_ROOT]] = (
                initial[index[LEFT_ROOT]] + self.open_offset
            )
            result[index[RIGHT_ROOT]] = (
                initial[index[RIGHT_ROOT]] - self.open_offset
            )
        else:
            result[index[LEFT_ROOT]] = (
                initial[index[LEFT_ROOT]] - self.close_offset
            )
            result[index[RIGHT_ROOT]] = (
                initial[index[RIGHT_ROOT]] + self.close_offset
            )
        return result

    def _is_fully_closed(self, closed_pose, index):
        """仿真空抓检测：两侧都到达闭合目标时判定没有夹到物体。"""
        rclpy.spin_once(self, timeout_sec=0.2)
        if self.latest_joint_state is None:
            return False
        values = dict(
            zip(self.latest_joint_state.name, self.latest_joint_state.position)
        )
        if LEFT_ROOT not in values or RIGHT_ROOT not in values:
            return False
        return (
            abs(values[LEFT_ROOT] - closed_pose[index[LEFT_ROOT]])
            <= self.grasp_position_tolerance
            and abs(values[RIGHT_ROOT] - closed_pose[index[RIGHT_ROOT]])
            <= self.grasp_position_tolerance
        )

    def _safe_home_pose(self, initial, current, index):
        current_open = self._gripper_pose(current, initial, index, opened=True)
        self._move_gripper("安全张开夹爪", current, current_open)
        home_open = self._gripper_pose(initial, initial, index, opened=True)
        self._move("安全回中机械臂", current_open, home_open)
        self._settle(home_open)
        self._publish_status("SAFE HOME COMPLETE: gripper open, arm initialized")
        return home_open

    def _rotate_chassis(self, held_positions):
        # Mecanum / skid-steer 原地旋转：左轮反转，右轮正转。
        wheel_command = [
            -self.wheel_speed,
            self.wheel_speed,
            -self.wheel_speed,
            self.wheel_speed,
        ]
        turn_duration = (
            self.turn_steps
            * self.physics_step_seconds
            / self.speed_scale
        )
        self.get_logger().info(
            f"小车开始连续原地旋转：轮速 {self.wheel_speed:.2f} rad/s，"
            f"仿真时间 {turn_duration:.3f} s（原标定 {self.turn_steps} 步）"
        )
        try:
            def keep_turning(ratio):
                if ratio >= 1.0:
                    return
                self._publish(held_positions, repeat=1)
                self._publish_to(
                    self.wheel_publisher, wheel_command, repeat=1
                )

            self._wait_sim_duration(turn_duration, callback=keep_turning)
        finally:
            # 无论转向成功、超时还是 Ctrl+C，都先写入零轮速。
            self._publish_to(self.wheel_publisher, [0.0] * 4, repeat=10)
            self._publish(held_positions, repeat=10)
        self._wait_sim_duration(self.post_turn_settle_duration)
        self.get_logger().info("小车旋转结束，车轮已停止")

    def _run_once(self, attempt, initial, current, index):
        # 对应实机脚本：0A 回中，0B 到初始前伸姿态，然后张开夹爪。
        self._report_step(attempt, "0A", "RECENTER ARM")
        recentered = self._copy_gripper_pose(initial, current, index)
        self._move("机械臂回到标定中心", current, recentered)

        self._report_step(attempt, "0B", "MOVE TO INITIAL ARM POSE")
        initial_forward = self._arm_pose(initial, self.arm_initial_fraction)
        initial_forward = self._copy_gripper_pose(
            initial_forward, recentered, index
        )
        self._move("机械臂伸到初始前方姿态", recentered, initial_forward)

        self._report_step(attempt, 1, "OPEN GRIPPER")
        initial_open = self._gripper_pose(
            initial_forward, initial, index, opened=True
        )
        self._move_gripper("张开夹爪", initial_forward, initial_open)

        self._report_step(attempt, 2, "CHASSIS SETTLE")
        self._settle(initial_open)

        self._report_step(attempt, 3, "FIRST FORWARD ALIGNMENT")
        aligned_open = self._gripper_pose(
            self._arm_pose(initial, self.arm_alignment_fraction),
            initial,
            index,
            opened=True,
        )
        self._move("第一次向前对准", initial_open, aligned_open)

        self._report_step(attempt, 4, "FORWARD SETTLE")
        self._settle(aligned_open)

        self._report_step(attempt, 5, "FORWARD-LEANING COARSE DESCENT")
        coarse_open = self._gripper_pose(
            self._arm_pose(initial, self.arm_coarse_fraction),
            initial,
            index,
            opened=True,
        )
        self._move("机械臂向前并粗略下降", aligned_open, coarse_open)

        self._report_step(attempt, 6, "PAUSE BEFORE FINAL APPROACH")
        self._settle(coarse_open)

        self._report_step(attempt, 7, "FINAL FORWARD-LEANING DESCENT")
        grasp_open = self._gripper_pose(
            self._arm_pose(initial, 1.0), initial, index, opened=True
        )
        self._move("最后向前下降到抓取位", coarse_open, grasp_open)

        self._report_step(attempt, 8, "AT GRASP POSITION")
        self._settle(grasp_open)

        self._report_step(attempt, 9, "CLOSE GRIPPER")
        grasp_closed = self._gripper_pose(
            grasp_open, initial, index, opened=False
        )
        self._move_gripper("缓慢闭合夹爪", grasp_open, grasp_closed)
        self._settle(grasp_closed, self.gripper_settle_duration)

        self._report_step(attempt, 10, "CHECK GRIPPER CLOSED ANGLE / STATUS")
        if self.stop_on_empty_grasp and self._is_fully_closed(
            grasp_closed, index
        ):
            self._publish_status("抓取失败：夹爪完全闭合，停止搬运循环")
            return False, self._safe_home_pose(
                initial, grasp_closed, index
            )
        if self.stop_on_empty_grasp:
            self._publish_status("抓取判断：夹爪未完全闭合，继续搬运")
        else:
            self._publish_status("仿真夹爪状态检测未启用，按抓取成功继续")

        self._report_step(attempt, 11, "TEST LIFT")
        test_lift_closed = list(grasp_closed)
        test_lift_closed[index[ARM_1]] = (
            initial[index[ARM_1]] + self.arm_test_lift_delta
        )
        self._move("小幅试抬", grasp_closed, test_lift_closed)

        self._report_step(attempt, 12, "CHECK REAL GRASP")
        self._settle(test_lift_closed)
        if self.stop_on_empty_grasp and self._is_fully_closed(
            test_lift_closed, index
        ):
            self._publish_status("抓取失败：试抬后夹爪完全闭合，执行安全回位")
            return False, self._safe_home_pose(
                initial, test_lift_closed, index
            )

        self._report_step(attempt, 13, "MAIN LIFT")
        lifted_closed = list(test_lift_closed)
        lifted_closed[index[ARM_1]] = (
            initial[index[ARM_1]] + self.arm_lift_delta
        )
        self._move("主抬升", test_lift_closed, lifted_closed)

        self._report_step(attempt, 14, "SETTLE BEFORE TURN")
        self._settle(lifted_closed)

        self._report_step(attempt, 15, "RIGHT TURN TO PLACE AREA")
        self._rotate_chassis(lifted_closed)

        self._report_step(attempt, 16, "LOWER CUBE AT B POINT")
        configured_place = self.place_points[
            (attempt - 1) % len(self.place_points)
        ]
        self.get_logger().info(
            f"第 {attempt} 轮配置放置点（world）："
            f"({configured_place[0]:.6f}, {configured_place[1]:.6f}, "
            f"{configured_place[2]:.3f})"
        )
        self._move("下降到放置位", lifted_closed, grasp_closed)

        self._report_step(attempt, 17, "GROUND SETTLE")
        self._settle(grasp_closed)

        self._report_step(attempt, 18, "RELEASE CUBE")
        released = self._gripper_pose(
            grasp_closed, initial, index, opened=True
        )
        self._move_gripper("释放物品", grasp_closed, released)

        self._report_step(attempt, 19, "LIFT ARM AWAY")
        lifted_open = list(released)
        lifted_open[index[ARM_1]] = (
            initial[index[ARM_1]] + self.arm_lift_delta
        )
        self._move("释放后抬臂离开", released, lifted_open)

        self._report_step(attempt, 20, "DONE - KEEP FINAL POSE")
        self._publish(lifted_open, repeat=10)
        self._publish_to(self.wheel_publisher, [0.0] * 4, repeat=10)
        self._settle(lifted_open)
        return True, lifted_open

    def run_action(self):
        if shutil.which("ros2") is None or shutil.which("ign") is None:
            raise GraspActionError("找不到 ros2 或 ign，请先 source ROS 2 环境")

        self.get_logger().info(
            f"ACTION VERSION: {ACTION_VERSION} | "
            f"speed={self.speed_scale:.1f}x | "
            f"continuous={self.command_rate_hz:.0f} Hz | "
            f"motion={self.motion_duration:.2f} s | "
            f"gripper={self.gripper_duration:.2f} s | "
            f"open={self.open_offset:.2f} rad | "
            f"lowest={self.arm_extend_delta:.2f} rad | "
            f"block={self.cube_length:.2f}x{self.cube_width:.2f}x"
            f"{self.cube_height:.2f} m/{self.cube_mass:.2f} kg/"
            f"mu={self.cube_friction:.1f} | "
            f"turn={self.turn_steps * self.physics_step_seconds / self.speed_scale:.3f} s"
        )
        places = "; ".join(
            f"P{number}=({point[0]:.6f},{point[1]:.6f},{point[2]:.3f})"
            for number, point in enumerate(self.place_points, start=1)
        )
        self.get_logger().info(
            f"空间配置 | PICK=({self.cube_x:.3f},{self.cube_y:.3f},"
            f"{self.cube_z:.3f}) | TURN={self.turn_angle_degrees:.1f} deg | "
            f"SAFE_HEIGHT={self.safe_height:.3f} m | {places}"
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

        if self.spawn_test_cube:
            self._publish_status("PREP: CREATE SLENDER TEST BLOCK")
            self._spawn_cube()
            self._settle(startup_open)

        current = list(startup_open)
        success_count = 0
        failed_count = 0

        self._publish_status(
            f"Starting simulated pick-place sequence: {self.run_count} runs"
        )
        for attempt in range(1, self.run_count + 1):
            ok, current = self._run_once(attempt, initial, current, index)
            if not ok:
                failed_count += 1
                break
            success_count += 1
            if attempt < self.run_count and self.cycle_pause_seconds > 0.0:
                self._publish_status(
                    f"Run {attempt} success. Next run starts in "
                    f"{self.cycle_pause_seconds:.1f} seconds"
                )
                self._wait_sim_duration(self.cycle_pause_seconds)

        self._publish_status(
            f"FINAL RESULT | success: {success_count} | failed: "
            f"{failed_count} | planned runs: {self.run_count}"
        )
        if success_count == self.run_count:
            self._publish_status(
                "Five-run transport task finished; chassis keeps final heading, "
                "Gazebo will pause after the safety stop"
            )


def main(args=None):
    # 不让 rclpy 在 SIGINT 到来时抢先关闭 context；这样 Ctrl+C 后仍能
    # 先发布零轮速和最后的机械臂保持命令，再由 finally 正常 shutdown。
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    exit_code = 0
    try:
        node = GraspCubeAction()
        node.run_action()
    except KeyboardInterrupt:
        exit_code = 130
    except Exception as error:
        exit_code = 1
        if node is not None:
            node.get_logger().error(str(error))
        else:
            print(f"夹爪动作启动失败：{error}")
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
