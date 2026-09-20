#!/usr/bin/env python3
"""控制器冒烟测试: 不经 MoveIt, 直接通过 JTC 话题让手臂走几个安全位置.

用于首次联调时确认 Gazebo + ros2_control 控制链路正常:
关节状态广播 -> JTC 位置控制 -> 夹爪前向指令。

用法: ros2 run robomaster_pick_place_sim test_joints
注意: 关节空间直控不做碰撞检测, 本脚本只使用安全角度。
"""
import time

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINTS = ['base_yaw_joint', 'arm_lift_joint', 'wrist_pitch_joint']


class TestJoints(Node):
    def __init__(self):
        super().__init__('test_joints')
        self.arm_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.grip_pub = self.create_publisher(
            Float64MultiArray, '/gripper_controller/commands', 10)

    def arm(self, name, yaw, lift, wrist, sec=3.0):
        msg = JointTrajectory()
        msg.joint_names = ARM_JOINTS
        p = JointTrajectoryPoint()
        p.positions = [float(yaw), float(lift), float(wrist)]
        p.time_from_start = Duration(sec=sec)
        msg.points = [p]
        self.get_logger().info(f'{name}: yaw={yaw}, lift={lift}, wrist={wrist}')
        self.arm_pub.publish(msg)
        time.sleep(sec + 1.0)

    def grip(self, name, v):
        self.get_logger().info(f'{name}: gripper={v}')
        self.grip_pub.publish(Float64MultiArray(data=[float(v), float(v)]))
        time.sleep(1.5)


def main():
    rclpy.init()
    n = TestJoints()
    time.sleep(2.0)

    n.grip('open', 0.040)
    n.arm('home', 0.0, 0.2, 0.0, 3)
    n.arm('yaw left', -0.8, 0.2, 0.0, 3)
    n.arm('yaw right', 0.8, 0.2, 0.0, 3)
    n.arm('raise arm', 0.0, 0.8, -0.3, 3)
    n.grip('close', 0.006)
    n.grip('open', 0.040)
    n.arm('home', 0.0, 0.2, 0.0, 3)

    n.get_logger().info('冒烟测试完成')
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
