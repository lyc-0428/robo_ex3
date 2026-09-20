#!/usr/bin/env python3
"""RoboMaster EP 机械臂解析运动学 (与 urdf/robomaster_ep_gazebo.urdf 一致).

臂为 3 自由度平面臂:
    base_yaw_joint    轴 +Z @ z=0.14        -> 决定臂平面方位角 yaw
    arm_lift_joint    轴 +Y @ (0.18,0,0.02) -> 肩在 base 系 (0.18, 0, 0.16)
    wrist_pitch_joint 轴 +Y @ (0.28,0,0)
    wrist_to_gripper  固定 (0.22,0,0)       -> gripper_base_link 原点 = 末端

平面内 2 连杆 L1=0.28 (lift->wrist), L2=0.22 (wrist->末端)。
绕 +Y 正向旋转使 +X 下俯, 故手指 (+X) 方向角 psi2 = -(lift+wrist)。
改动 URDF 关节 origin 时须同步本文件常量; pick_place_moveit 的 run() 预检查
会用 move_group 的 /compute_fk 服务校验逆解, 常量漂移会被 FK 误差检查拦住。

为什么不用 MoveIt 的 KDL IK + OMPL 路径约束:
3 自由度臂上位置+姿态路径约束的可行流形退化为孤立点, OMPL 约束采样
(KDL IK 采样器, 且每次以同一参考状态作种子) 几乎必然采不出合法样本,
表现为 "Motion planning start tree could not be initialized" (error 99999)。
解析逆解 + 关节目标自由空间规划彻底规避该问题。
"""
import math

# 与 URDF 关节 origin 一一对应
ARM_SHOULDER = (0.18, 0.16)   # lift 轴在 base 系的位置 (x, z)
ARM_L1 = 0.28                 # lift 轴 -> wrist 轴
ARM_L2 = 0.22                 # wrist 轴 -> gripper_base_link 原点

# 手指 (+X) 偏离水平面的最大允许角度 [rad]: 超过则无法从侧面水平夹取物体
MAX_FINGER_TILT = 0.5

JOINT_LIMITS = {
    'base_yaw_joint': (-3.14, 3.14),
    'arm_lift_joint': (-0.8, 1.0),
    'wrist_pitch_joint': (-1.5, 1.5),
}


def solve_ik(xyz):
    """目标末端位置 -> (yaw, lift, wrist, finger_tilt) 或 None.

    须同时满足: 可达 (|L1-L2| <= d <= L1+L2)、关节限位、
    手指近似水平指向前方 (|tilt| <= MAX_FINGER_TILT)。
    多解时取手指最接近水平的解 (finger_tilt 即 psi2)。
    """
    x, y, z = xyz
    r = math.hypot(x, y)
    th = math.atan2(y, x)
    dx, dz = r - ARM_SHOULDER[0], z - ARM_SHOULDER[1]
    d = math.hypot(dx, dz)
    if d > ARM_L1 + ARM_L2 + 1e-9 or d < abs(ARM_L1 - ARM_L2) - 1e-9:
        return None
    cos_g = (d * d - ARM_L1 * ARM_L1 - ARM_L2 * ARM_L2) / (2 * ARM_L1 * ARM_L2)
    g = math.acos(max(-1.0, min(1.0, cos_g)))
    best = None
    for sg in (g, -g):
        # 平面内: 上臂方向角 psi1, 手指(前臂)方向角 psi2
        psi1 = math.atan2(dz, dx) + math.atan2(ARM_L2 * math.sin(sg),
                                               ARM_L1 + ARM_L2 * math.cos(sg))
        psi2 = math.atan2(dz, dx) - math.atan2(ARM_L1 * math.sin(sg),
                                               ARM_L2 + ARM_L1 * math.cos(sg))
        yaw = th
        lift = -psi1
        wrist = -psi2 - lift
        if abs(psi2) > MAX_FINGER_TILT:
            continue
        if not (JOINT_LIMITS['arm_lift_joint'][0] <= lift <= JOINT_LIMITS['arm_lift_joint'][1]
                and JOINT_LIMITS['wrist_pitch_joint'][0] <= wrist <= JOINT_LIMITS['wrist_pitch_joint'][1]):
            continue
        if abs(yaw) > JOINT_LIMITS['base_yaw_joint'][1]:
            continue
        if best is None or abs(psi2) < best[0]:
            best = (abs(psi2), yaw, lift, wrist, psi2)
    if best is None:
        return None
    return (best[1], best[2], best[3], best[4])


def forward_kinematics(yaw, lift, wrist):
    """关节值 -> 末端位置 (x, y, z) (与 URDF 的 FK 等价, 供离线校验)."""
    c1, s1 = math.cos(lift), math.sin(lift)
    c2, s2 = math.cos(lift + wrist), math.sin(lift + wrist)
    x = ARM_SHOULDER[0] + ARM_L1 * c1 + ARM_L2 * c2
    z = ARM_SHOULDER[1] - ARM_L1 * s1 - ARM_L2 * s2
    return (x * math.cos(yaw), x * math.sin(yaw), z)
