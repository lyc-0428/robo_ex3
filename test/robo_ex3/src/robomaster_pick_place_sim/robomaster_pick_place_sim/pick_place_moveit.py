#!/usr/bin/env python3
"""RoboMaster EP 定点抓取节点 (原生 MoveIt 2 接口, 不依赖 moveit_commander)

流程: 回零 -> 取物点A上方(预抓取) -> 沿手指方向接近 -> 夹取 ->
      抬升 -> 放置点B上方(预放置) -> 沿手指方向接近 -> 释放 -> 回零

- 手臂运动: 经 move_group 的原生接口规划执行
    * 位置IK/关节目标:  /move_action (MoveGroup Action)
    * 笛卡尔直线:      /compute_cartesian_path + /execute_trajectory 动作
    * 末端位姿:        /compute_fk 服务
    * 急停:            向 /trajectory_execution/event 发 stop 事件
- 夹爪通过 /gripper_controller/commands 话题直接控制
- 每次规划/执行失败(不可达、无逆解、关节超限)都记录错误、急停、
  返回安全位置; 日志/轨迹/结果保存到 log_dir
"""
import csv
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import yaml

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from controller_manager_msgs.srv import ListControllers
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (AllowedCollisionEntry, AttachedCollisionObject,
                             CollisionObject, Constraints,
                             JointConstraint, MoveItErrorCodes,
                             PlanningScene, RobotState)
from moveit_msgs.srv import GetCartesianPath, GetPositionFK
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Float64MultiArray, String

from robomaster_pick_place_sim.robot_kinematics import solve_ik

# 与 robomaster_ep_moveit_config/config/joint_limits.yaml 保持一致
JOINT_LIMITS = {
    'base_yaw_joint': (-3.14, 3.14),
    'arm_lift_joint': (-0.8, 1.0),
    'wrist_pitch_joint': (-1.5, 1.5),
    'left_finger_joint': (0.0, 0.045),
    'right_finger_joint': (0.0, 0.045),
}

DEFAULT_PARAMS = {
    'num_cycles': 5,
    'gripper_open': 0.040,
    'gripper_closed': 0.006,
    'max_velocity_scale': 0.5,
    'max_acceleration_scale': 0.5,
    # 预设点由解析逆解+FK 校验 (见 run() 预检查):
    #   pick/place  z=0.35 时手指近水平 (倾角 <2°), 手指箱底距桌面 3.1cm;
    #   pre_pick/pre_place 与对应抓取点同 lift 角, 仅腕差 (纯腕部接近/撤离运动)。
    'pick_point': {'x': 0.60, 'y': 0.0, 'z': 0.35},
    'place_point': {'x': 0.55, 'y': 0.25, 'z': 0.35},
    'pre_pick_point': {'x': 0.596, 'y': 0.0, 'z': 0.399},
    'pre_place_point': {'x': 0.546, 'y': 0.248, 'z': 0.395},
    'attach_object': True,
    'object_size': 0.07,
    'table': {'center': [0.78, 0.0, 0.15], 'size': [0.80, 0.70, 0.30]},
    'log_dir': 'results',
    'abort_on_error': False,
}

EE_LINK = 'gripper_base_link'
GROUP = 'arm'

# 解析逆解运动学常量/求解器见 robomaster_pick_place_sim/robot_kinematics.py
# (与 urdf/robomaster_ep_gazebo.urdf 一致; 改动 URDF 关节 origin 需同步,
#  run() 预检查会用 /compute_fk 服务校验, 漂移会被 FK 误差检查拦住)


class TaskError(Exception):
    """一次抓取循环中的可恢复错误."""


class PickPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_place_node')
        self.declare_parameter('params_file', '')
        # 注: use_sim_time 由 launch 文件统一传入, 这里不再 declare (否则重复声明抛异常)

        self.p = self._load_params()

        # ---------- 日志与结果文件 ----------
        self.log_dir = os.path.abspath(self.p['log_dir'])
        os.makedirs(self.log_dir, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(self.log_dir, f'run_{stamp}.log')
        self.csv_path = os.path.join(self.log_dir, f'run_{stamp}_trajectory.csv')
        self.json_path = os.path.join(self.log_dir, f'run_{stamp}_results.json')
        self.cycle = 0
        self.successes = 0
        self.failures = 0
        self.cycle_records = []
        self.csv_f = open(self.csv_path, 'w', newline='')
        self.csv_w = csv.writer(self.csv_f)
        self.csv_w.writerow(['timestamp', 'cycle', 'step', 'base_yaw', 'arm_lift',
                             'wrist_pitch', 'left_finger', 'right_finger',
                             'ee_x', 'ee_y', 'ee_z', 'success'])

        # ---------- 发布器 / 订阅 ----------
        self.grip_pub = self.create_publisher(Float64MultiArray, '/gripper_controller/commands', 10)
        self.scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        self.status_pub = self.create_publisher(String, '/pick_place/status', 10)
        self.stop_pub = self.create_publisher(String, '/trajectory_execution/event', 10)
        self._last_js = None
        self.create_subscription(JointState, '/joint_states', self._js_cb, 10)

        # ---------- MoveIt 原生接口 ----------
        self.mg_cli = ActionClient(self, MoveGroup, '/move_action')
        self.fk_cli = self.create_client(GetPositionFK, '/compute_fk')
        self.cart_cli = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        # humble 的 MoveGroupExecuteService capability 是坏的 (类被移除),
        # 轨迹执行走 MoveGroupExecuteTrajectoryAction 动作接口
        self.exec_cli = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')

        # ---------- 等待控制系统就绪 ----------
        self._wait_for_system()

        # ---------- 规划场景: 桌面 + 目标物体 ----------
        self._publish_scene()
        time.sleep(1.5)  # 等 move_group 的场景监视器更新
        self.log('规划场景已添加: 桌面 + 目标物体')

    # ================= 参数与日志 =================

    def _load_params(self):
        params = dict(DEFAULT_PARAMS)
        path = self.get_parameter('params_file').value
        if not path:
            path = os.path.join(
                get_package_share_directory('robomaster_pick_place_sim'),
                'config', 'pick_place_params.yaml')
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            params.update(data.get('pick_place_node', {}).get('ros__parameters', {}))
            self.get_logger().info(f'参数文件: {path}')
        else:
            self.get_logger().warn(f'参数文件不存在, 使用默认值: {path}')
        return params

    def log(self, msg, level='info'):
        line = f'[{datetime.now().strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        with open(self.log_path, 'a') as f:
            f.write(line + '\n')
        getattr(self.get_logger(), level)(msg)

    def status(self, msg):
        self.log(f'状态: {msg}')
        self.status_pub.publish(String(data=msg))

    # ================= 系统就绪等待 =================

    def _js_cb(self, msg):
        self._last_js = msg

    def _wait_joint_states(self, timeout):
        # 必须收到包含全部 5 个关节的完整消息 (move_group 收到空 JointState 时
        # 会用默认状态规划, 轨迹起点错位; 内容校验可彻底排除这种情况)
        need = set(JOINT_LIMITS.keys())
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._last_js is not None and need.issubset(self._last_js.name):
                return self._last_js
            rclpy.spin_once(self, timeout_sec=0.2)
        raise TaskError('/joint_states 超时: 未收到完整的 5 个关节状态')

    def _current_joint_states(self):
        """取最新几帧关节状态(含夹爪真实位置)."""
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.05)
        return self._last_js

    def _wait_for_system(self):
        # 1. 等 joint_state_broadcaster 发布关节状态
        self.log('等待 /joint_states ...')
        self._wait_joint_states(timeout=60.0)

        # 2. 等 arm_controller 的 FollowJointTrajectory Action 服务
        self.log('等待 /arm_controller/follow_joint_trajectory ...')
        act_cli = ActionClient(self, FollowJointTrajectory,
                               '/arm_controller/follow_joint_trajectory')
        if not act_cli.wait_for_server(timeout_sec=60.0):
            raise TaskError('arm_controller Action 服务超时未出现')

        # 3. 等三个控制器都进入 active 状态
        self.log('等待控制器进入 active ...')
        cli = self.create_client(ListControllers, '/controller_manager/list_controllers')
        if not cli.wait_for_service(timeout_sec=60.0):
            raise TaskError('controller_manager 服务超时未出现')
        deadline = time.time() + 90.0
        while time.time() < deadline:
            fut = cli.call_async(ListControllers.Request())
            rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
            if fut.done() and fut.result() is not None:
                states = {c.name: c.state for c in fut.result().controller}
                want = ['joint_state_broadcaster', 'arm_controller', 'gripper_controller']
                if all(states.get(w) == 'active' for w in want):
                    self.log('所有控制器已激活')
                    break
            time.sleep(2.0)
        else:
            raise TaskError('控制器激活超时')

        # 4. 等 move_group 的规划/求解/执行接口
        self.log('等待 move_group 接口 ...')
        if not self.mg_cli.wait_for_server(timeout_sec=60.0):
            raise TaskError('/move_action Action 服务超时未出现')
        for svc in (self.fk_cli, self.cart_cli):
            if not svc.wait_for_service(timeout_sec=60.0):
                raise TaskError(f'{svc.srv_name} 服务超时未出现')
        if not self.exec_cli.wait_for_server(timeout_sec=60.0):
            raise TaskError('/execute_trajectory Action 服务超时未出现')
        self.log('move_group 接口就绪')

    # ================= 规划场景 (桌面 + 物体 + attach) =================

    def _box_obj(self, name, center, size, op=CollisionObject.ADD):
        obj = CollisionObject()
        obj.id = name
        obj.header.frame_id = 'world'
        obj.header.stamp = self.get_clock().now().to_msg()
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = list(size)
        obj.primitives = [prim]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = center
        pose.orientation.w = 1.0
        obj.primitive_poses = [pose]
        obj.operation = op
        return obj

    def _object_center(self):
        """物体中心: 桌面顶面 (z=0.30) 上方 5mm, 避免 attach 后与桌面
        恰好接触 (0.30 vs 0.30) 因浮点误差被 FCL 判为穿透。"""
        pp = self.p['pick_point']
        return [pp['x'], pp['y'], 0.34]

    def _publish_scene(self):
        ps = PlanningScene()
        ps.is_diff = True
        t = self.p['table']
        ps.world.collision_objects = [
            self._box_obj('table', t['center'], t['size']),
            self._box_obj('target_object', self._object_center(),
                          [self.p['object_size']] * 3),
        ]
        # 夹爪与目标物体之间允许接触: 手指需要框住物体侧面才能夹取,
        # 否则 OMPL 碰撞检测会把抓取位姿判为无效。
        # humble 的 AllowedCollisionMatrix 字段是 entry_values (不是 entry);
        # enabled[j] 表示允许 entry_names[i] 与 entry_names[j] 碰撞, 长度需一致,
        # 且 target_object 也要列进 entry_names 才能被豁免。
        acm = ps.allowed_collision_matrix
        acm.entry_names = [EE_LINK, 'left_finger_link', 'right_finger_link',
                           'target_object']
        acm.entry_values = [AllowedCollisionEntry(enabled=[True] * 4)
                            for _ in acm.entry_names]
        self.scene_pub.publish(ps)

    def _attach_object(self, attach):
        """把目标物体附加/解除到夹爪基座 (仅影响 MoveIt 规划场景)."""
        ps = PlanningScene()
        ps.is_diff = True
        ps.robot_state.is_diff = True
        att = AttachedCollisionObject()
        att.link_name = EE_LINK
        att.touch_links = [EE_LINK, 'left_finger_link', 'right_finger_link']
        att.object = self._box_obj(
            'target_object', self._object_center(),
            [self.p['object_size']] * 3,
            CollisionObject.ADD if attach else CollisionObject.REMOVE)
        ps.robot_state.attached_collision_objects = [att]
        self.scene_pub.publish(ps)
        time.sleep(1.0)  # 等场景监视器更新
        self.log('物体已附加到夹爪' if attach else '物体已从夹爪解除')

    # ================= MoveIt 原生调用 =================

    def _joint_constraint(self, joints):
        """关节目标约束 (无路径约束 -> OMPL 自由空间规划, 稳健可靠).

        注: 位置+姿态路径约束对 3 自由度臂不可行 — 约束流形退化为孤立点,
        OMPL 约束采样 (KDL IK 采样器) 几乎必然失败, 表现为
        "Motion planning start tree could not be initialized" (error 99999)。
        """
        c = Constraints()
        c.joint_constraints = []
        for name, value in zip(('base_yaw_joint', 'arm_lift_joint',
                                'wrist_pitch_joint'), joints):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = value
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        return c

    def _home_joints(self):
        """从 SRDF 读取 home 位姿 [yaw, lift, wrist]."""
        srdf = os.path.join(
            get_package_share_directory('robomaster_ep_moveit_config'),
            'config', 'robomaster_ep.srdf')
        tree = ET.parse(srdf)
        for gs in tree.getroot().findall('group_state'):
            if gs.get('name') == 'home':
                vals = {j.get('name'): float(j.get('value'))
                        for j in gs.findall('joint')}
                return [vals['base_yaw_joint'], vals['arm_lift_joint'],
                        vals['wrist_pitch_joint']]
        raise TaskError('SRDF 中找不到 home 位姿')

    def _home_constraint(self):
        c = Constraints()
        c.joint_constraints = []
        for name, value in zip(('base_yaw_joint', 'arm_lift_joint',
                                'wrist_pitch_joint'), self._home_joints()):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = value
            jc.tolerance_above = 0.001
            jc.tolerance_below = 0.001
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        return c

    def _mg_call(self, constraints, plan_only, label, timeout=180.0,
                 start_joints=None):
        """通过 /move_action 规划(或规划并执行), 返回 MoveGroup.Result.

        start_joints: 显式规划起点 [yaw, lift, wrist] (预检查预演用, 机器人
        并未真正运动到各中间点, 不能依赖 move_group 的当前监视状态);
        为 None 时用空 diff 表示"从当前状态规划" (实际执行用)。
        """
        goal = MoveGroup.Goal()
        goal.request.group_name = GROUP
        goal.request.goal_constraints = [constraints]
        goal.request.allowed_planning_time = 3.0
        goal.request.num_planning_attempts = 10
        goal.request.max_velocity_scaling_factor = float(self.p['max_velocity_scale'])
        goal.request.max_acceleration_scaling_factor = float(self.p['max_acceleration_scale'])
        goal.planning_options.plan_only = plan_only
        goal.planning_options.planning_scene_diff.is_diff = True
        if start_joints is not None:
            rs = RobotState()
            rs.joint_state.name = ['base_yaw_joint', 'arm_lift_joint',
                                   'wrist_pitch_joint']
            rs.joint_state.position = [float(v) for v in start_joints]
            goal.request.start_state = rs
        else:
            goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        fut = self.mg_cli.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        if not fut.done() or fut.result() is None or not fut.result().accepted:
            raise TaskError(f'{label}: MoveGroup 动作未接受')
        res_fut = fut.result().get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=timeout)
        if not res_fut.done() or res_fut.result() is None:
            raise TaskError(f'{label}: MoveGroup 规划/执行超时')
        return res_fut.result().result

    def _fk_pose(self):
        """当前末端位姿 (空 robot_state 表示用 move_group 的当前状态)."""
        return self._fk_for_joints(None)

    def _fk_for_joints(self, joints):
        """给定臂关节值 [yaw, lift, wrist] 求末端位姿; joints=None 用当前状态.
        (预检查用它校验解析逆解与 URDF 运动学一致, 防止常量漂移)."""
        req = GetPositionFK.Request()
        req.header.frame_id = 'world'
        req.header.stamp = self.get_clock().now().to_msg()
        req.fk_link_names = [EE_LINK]
        req.robot_state = RobotState()
        if joints is not None:
            req.robot_state.joint_state.name = ['base_yaw_joint',
                                                'arm_lift_joint',
                                                'wrist_pitch_joint']
            req.robot_state.joint_state.position = [float(v) for v in joints]
        fut = self.fk_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        if not fut.done() or fut.result() is None:
            raise TaskError('FK 服务调用失败')
        r = fut.result()
        if r.error_code.val != MoveItErrorCodes.SUCCESS or not r.pose_stamped:
            raise TaskError('FK 求解失败')
        return r.pose_stamped[0].pose

    def _execute_trajectory(self, traj, label):
        """通过 /execute_trajectory Action 执行轨迹 (humble 无可用服务版)."""
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = traj
        fut = self.exec_cli.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        if not fut.done() or fut.result() is None or not fut.result().accepted:
            raise TaskError(f'{label}: 轨迹执行动作未接受')
        res_fut = fut.result().get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=180.0)
        if not res_fut.done() or res_fut.result() is None:
            raise TaskError(f'{label}: 轨迹执行超时')
        r = res_fut.result().result
        if r.error_code.val != MoveItErrorCodes.SUCCESS:
            raise TaskError(f'{label}: 轨迹执行失败 (error code={r.error_code.val})')

    def _stop(self):
        """急停: 通知 trajectory_execution_manager 停止当前轨迹."""
        self.stop_pub.publish(String(data='stop'))

    # ================= 运动原语 =================

    def _check_limits(self, label):
        js = self._current_joint_states()
        if js is None:
            return
        for name, value in zip(js.name, js.position):
            if name in JOINT_LIMITS:
                lo, hi = JOINT_LIMITS[name]
                if not lo - 0.01 <= value <= hi + 0.01:
                    raise TaskError(f'{label}: 关节 {name} 超限 '
                                    f'({value:.3f} 超出 [{lo}, {hi}])')

    def go_to_pos(self, pos, label, theta=0.0):
        """解析逆解 -> 关节目标自由空间规划 -> 轨迹执行 (theta 仅作兼容保留).

        手指方向由逆解的 psi2 约束保证: 近似水平、指向径向, 从侧面夹取物体。
        """
        sol = solve_ik(pos)
        if sol is None:
            raise TaskError(
                f'{label}: 目标 {pos} 无逆解 (不可达/关节超限/手指姿态不可行), '
                f'请用 calibrate_pose 重新标定')
        yaw, lift, wrist, tilt = sol
        res = self._mg_call(self._joint_constraint([yaw, lift, wrist]), False, label)
        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            raise TaskError(f'{label}: 规划失败 (error code={res.error_code.val}), '
                            f'目标 {pos} 逆解 ({yaw:.3f}, {lift:.3f}, {wrist:.3f}) '
                            f'与障碍碰撞或路径不存在')
        self._check_limits(label)
        self._record(label, True)
        self.log(f'{label}: 到达 ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) '
                 f'[joints {yaw:.3f}/{lift:.3f}/{wrist:.3f}, '
                 f'手指倾角 {math.degrees(tilt):.1f}°]')

    def grip(self, value, label):
        self.grip_pub.publish(Float64MultiArray(data=[float(value), float(value)]))
        time.sleep(1.5)
        self._record(label, True)
        self.log(f'{label}: 夹爪指令 {value:.3f} m')

    def go_home(self):
        res = self._mg_call(self._home_constraint(), False, '回零')
        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            raise TaskError(f'回零: 规划失败 (error code={res.error_code.val})')
        self._record('home', True)
        self.log('回零完成')

    # ================= 记录 =================

    def _record(self, label, success):
        row = [datetime.now().strftime('%H:%M:%S.%f')[:-3], self.cycle, label,
               '', '', '', '', '', '', '', '', int(success)]
        js = self._current_joint_states()
        if js is not None:
            vals = dict(zip(js.name, js.position))
            row[3] = round(vals.get('base_yaw_joint', float('nan')), 4)
            row[4] = round(vals.get('arm_lift_joint', float('nan')), 4)
            row[5] = round(vals.get('wrist_pitch_joint', float('nan')), 4)
            row[6] = round(vals.get('left_finger_joint', float('nan')), 4)
            row[7] = round(vals.get('right_finger_joint', float('nan')), 4)
        try:
            pose = self._fk_pose()
            row[8] = round(pose.position.x, 4)
            row[9] = round(pose.position.y, 4)
            row[10] = round(pose.position.z, 4)
        except Exception:
            pass
        self.csv_w.writerow(row)
        self.csv_f.flush()

    # ================= 抓取循环 =================

    def _do_cycle(self, i):
        self.cycle = i
        pp = self.p['pick_point']
        bp = self.p['place_point']
        ppre = self.p['pre_pick_point']
        bpre = self.p['pre_place_point']
        # 手指水平指向径向: 目标点方位角 = 期望的 gripper_base +X 方向
        th_pp = math.atan2(pp['y'], pp['x'])
        th_bp = math.atan2(bp['y'], bp['x'])
        th_ppre = math.atan2(ppre['y'], ppre['x'])
        th_bpre = math.atan2(bpre['y'], bpre['x'])
        steps = []

        self.status(f'开始第 {i}/{self.p["num_cycles"]} 次抓取')

        self.go_home()
        steps.append('home')

        # --- 取物点 A: 预抓取点(上方可达) -> 抓取点(物体中心, 手指水平径向) ---
        self.go_to_pos([ppre['x'], ppre['y'], ppre['z']], f'cycle{i}_pre_pick', th_ppre)
        steps.append(f'cycle{i}_pre_pick')
        self.go_to_pos([pp['x'], pp['y'], pp['z']], f'cycle{i}_pick', th_pp)
        steps.append(f'cycle{i}_pick')

        # --- 夹取 + attach ---
        self.grip(self.p['gripper_closed'], f'cycle{i}_grip_close')
        steps.append(f'cycle{i}_grip_close')
        if self.p['attach_object']:
            self._attach_object(True)

        # --- 抬离桌面: 回预抓取点 (物体随夹爪, 与夹爪之间无碰撞) ---
        self.go_to_pos([ppre['x'], ppre['y'], ppre['z']], f'cycle{i}_lift', th_ppre)
        steps.append(f'cycle{i}_lift')

        # --- 移向放置点 B: 预放置点 -> 放置点 ---
        self.go_to_pos([bpre['x'], bpre['y'], bpre['z']], f'cycle{i}_pre_place', th_bpre)
        steps.append(f'cycle{i}_pre_place')
        self.go_to_pos([bp['x'], bp['y'], bp['z']], f'cycle{i}_place', th_bp)
        steps.append(f'cycle{i}_place')

        # --- 释放 + detach + 撤回 ---
        if self.p['attach_object']:
            self._attach_object(False)
        self.grip(self.p['gripper_open'], f'cycle{i}_release')
        steps.append(f'cycle{i}_release')
        self.go_to_pos([bpre['x'], bpre['y'], bpre['z']], f'cycle{i}_withdraw', th_bpre)
        steps.append(f'cycle{i}_withdraw')

        self.go_home()
        steps.append('home')
        self.status(f'第 {i}/{self.p["num_cycles"]} 次抓取成功')
        return steps

    def _stop_safe(self, reason):
        """失败后的安全处理: 急停 -> 尽力回零 -> 记录错误."""
        self.status(f'错误: {reason}, 急停')
        self._stop()
        self._record(f'stop:{reason}', False)
        try:
            self.go_home()
        except Exception:
            self.log('警告: 回零也失败, 机械臂已停止', 'error')

    def run(self):
        # 启动前预检查, 分两层:
        # 1) 解析逆解 + FK 一致性校验 (逆解-URDF 漂移、不可达、限位、手指姿态)
        # 2) 整个循环全部 7 段关节轨迹的 plan-only 预演 (起点显式给定),
        #    碰撞/无路径在正式开始前全部暴露
        home = self._home_joints()
        sols = {}
        for name in ('pick_point', 'place_point', 'pre_pick_point',
                     'pre_place_point'):
            pos = self.p[name]
            xyz = [pos['x'], pos['y'], pos['z']]
            sol = solve_ik(xyz)
            if sol is None:
                self.log(f'错误: {name} {xyz} 无逆解 (不可达/关节超限/'
                         f'手指姿态不可行), 任务中止, 请用 calibrate_pose 重新标定',
                         'error')
                self._stop_safe(f'{name} 不可达')
                return 1
            pose = self._fk_for_joints(sol[:3])
            err = math.hypot(pose.position.x - xyz[0],
                             pose.position.y - xyz[1],
                             pose.position.z - xyz[2])
            if err > 0.005:
                self.log(f'错误: {name} 逆解 FK 误差 {err * 1000:.1f}mm, '
                         f'解析模型与 URDF 不一致', 'error')
                self._stop_safe('FK 校验失败')
                return 1
            sols[name] = sol
            self.log(f'可达性检查通过: {name} {xyz} -> joints '
                     f'({sol[0]:.3f}, {sol[1]:.3f}, {sol[2]:.3f}), '
                     f'手指倾角 {math.degrees(sol[3]):.1f}°')

        cycle_seq = [('home', home),
                     ('pre_pick_point', sols['pre_pick_point']),
                     ('pick_point', sols['pick_point']),
                     ('pre_pick_point', sols['pre_pick_point']),
                     ('pre_place_point', sols['pre_place_point']),
                     ('place_point', sols['place_point']),
                     ('pre_place_point', sols['pre_place_point']),
                     ('home', home)]
        prev = cycle_seq[0]
        for name, js in cycle_seq[1:]:
            res = self._mg_call(self._joint_constraint(js[:3]), True,
                                f'{name} 路径预演', start_joints=prev[1][:3])
            if res.error_code.val != MoveItErrorCodes.SUCCESS:
                self.log(f'错误: {prev[0]} -> {name} 规划失败 '
                         f'(error code={res.error_code.val}), 路径碰撞或不存在, '
                         f'任务中止, 请用 calibrate_pose 重新标定', 'error')
                self._stop_safe(f'{prev[0]}->{name} 规划失败')
                return 1
            prev = (name, js)
        self.log('整循环路径预演通过 (7/7 段)')

        self.status('开始连续抓取实验')
        for i in range(1, self.p['num_cycles'] + 1):
            t0 = time.time()
            try:
                steps = self._do_cycle(i)
                self.successes += 1
                self.cycle_records.append(
                    {'cycle': i, 'success': True, 'steps': steps,
                     'duration_s': round(time.time() - t0, 1), 'error': None})
            except TaskError as e:
                self.failures += 1
                self.log(f'第 {i} 次抓取失败: {e}', 'error')
                self.cycle_records.append(
                    {'cycle': i, 'success': False, 'steps': [],
                     'duration_s': round(time.time() - t0, 1), 'error': str(e)})
                self._stop_safe(str(e))
                if self.p['abort_on_error']:
                    self.status('检测到错误且 abort_on_error=true, 任务终止')
                    break
        return 0

    def finish(self):
        self.status(f'实验结束: 成功 {self.successes}/{self.p["num_cycles"]}, '
                    f'失败 {self.failures}')
        passed = self.successes >= max(1, int(0.8 * self.p['num_cycles']))
        result = {
            'experiment': 'robomaster_ep_pick_place',
            'finish_time': datetime.now().isoformat(),
            'num_cycles': self.p['num_cycles'],
            'successes': self.successes,
            'failures': self.failures,
            'passed': bool(passed),
            'cycles': self.cycle_records,
            'params': self.p,
            'log_file': self.log_path,
            'trajectory_file': self.csv_path,
        }
        with open(self.json_path, 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        self.csv_f.close()
        self.log(f'验收结果: {"通过" if passed else "未通过"} '
                 f'({self.successes}/{self.p["num_cycles"]} >= 80%)')
        self.log(f'结果已保存: {self.json_path}')
        self.log(f'轨迹已保存: {self.csv_path}')


def main():
    rclpy.init()
    node = PickPlaceNode()
    code = 0
    try:
        code = node.run()
    except TaskError as e:
        node.log(f'致命错误: {e}', 'error')
        code = 1
    except KeyboardInterrupt:
        node.log('用户中断, 执行安全停止', 'warn')
        try:
            node._stop_safe('用户中断')
        except Exception:
            pass
        code = 130
    finally:
        try:
            node.finish()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
