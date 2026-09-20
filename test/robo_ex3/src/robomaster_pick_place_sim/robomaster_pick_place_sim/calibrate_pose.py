#!/usr/bin/env python3
"""A/B 点标定工具: 让机械臂运动到指定 XYZ, 打印实际位姿与关节角.

用于确定可取的取物点/放置点坐标, 然后写入 pick_place_params.yaml。
不依赖 moveit_commander, 直接使用 MoveIt 原生接口
(/move_action Action + /compute_fk 服务)。

用法:
  ros2 run robomaster_pick_place_sim calibrate_pose
  输入 "x y z" 回车 -> 规划并执行到该位置, 打印末端实际位姿
  输入空行 -> 打印当前位姿与关节角
  输入 q 退出
"""
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (Constraints, JointConstraint, MoveItErrorCodes,
                             RobotState)
from moveit_msgs.srv import GetPositionFK
from sensor_msgs.msg import JointState

from robomaster_pick_place_sim.robot_kinematics import solve_ik

EE_LINK = 'gripper_base_link'
ARM_JOINTS = ['base_yaw_joint', 'arm_lift_joint', 'wrist_pitch_joint']


class CalibrateNode(Node):
    def __init__(self):
        super().__init__('calibrate_pose')
        self.mg_cli = ActionClient(self, MoveGroup, '/move_action')
        self.fk_cli = self.create_client(GetPositionFK, '/compute_fk')
        self._last_js = None
        self.create_subscription(JointState, '/joint_states', self._js_cb, 10)

        if not self.mg_cli.wait_for_server(timeout_sec=60.0):
            raise RuntimeError('/move_action 超时未出现')
        if not self.fk_cli.wait_for_service(timeout_sec=60.0):
            raise RuntimeError('/compute_fk 超时未出现')
        print('已连接 move_group。输入 "x y z" 移动, 空行显示当前位姿, q 退出')

    def _js_cb(self, msg):
        self._last_js = msg

    def _go(self, x, y, z):
        """解析逆解 + 关节目标规划并执行到 (x, y, z), 返回 (ok, error_code)."""
        sol = solve_ik([x, y, z])
        if sol is None:
            return False, -1
        c = Constraints()
        c.joint_constraints = []
        for name, value in zip(ARM_JOINTS, sol[:3]):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = value
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        goal = MoveGroup.Goal()
        goal.request.group_name = 'arm'
        goal.request.goal_constraints = [c]
        goal.request.allowed_planning_time = 3.0
        goal.request.num_planning_attempts = 10
        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3
        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        fut = self.mg_cli.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        if not fut.done() or fut.result() is None or not fut.result().accepted:
            return False, -1
        res_fut = fut.result().get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=120.0)
        if not res_fut.done() or res_fut.result() is None:
            return False, -2
        r = res_fut.result().result
        return r.error_code.val == MoveItErrorCodes.SUCCESS, r.error_code.val

    def _pose(self):
        req = GetPositionFK.Request()
        req.header.frame_id = 'world'
        req.header.stamp = self.get_clock().now().to_msg()
        req.fk_link_names = [EE_LINK]
        req.robot_state = RobotState()
        fut = self.fk_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        if not fut.done() or fut.result() is None:
            return None
        r = fut.result()
        if r.error_code.val != MoveItErrorCodes.SUCCESS or not r.pose_stamped:
            return None
        return r.pose_stamped[0].pose

    def show(self):
        pose = self._pose()
        if pose is None:
            print('无法获取当前位姿')
            return
        q = pose.orientation
        fx = 1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        fy = 2.0 * (q.x * q.y + q.w * q.z)
        fz = 2.0 * (q.x * q.z - q.w * q.y)
        print(f'末端位置: ({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})')
        print(f'手指方向: ({fx:.3f}, {fy:.3f}, {fz:.3f})')
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.05)
        js = self._last_js
        if js is not None:
            vals = dict(zip(js.name, js.position))
            print(f'关节角:   {[round(vals.get(j, float("nan")), 3) for j in ARM_JOINTS]}')
        print()


def main():
    rclpy.init()
    node = CalibrateNode()
    try:
        while True:
            try:
                line = input()
            except EOFError:
                break
            except KeyboardInterrupt:
                break
            line = line.strip()
            if line in ('q', 'quit', 'exit'):
                break
            if not line:
                node.show()
                continue
            try:
                x, y, z = map(float, line.split())
            except ValueError:
                print('格式: x y z')
                continue
            ok, code = node._go(x, y, z)
            if not ok:
                print(f'不可达/无逆解: ({x}, {y}, {z})  error code={code}')
                continue
            node.show()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
