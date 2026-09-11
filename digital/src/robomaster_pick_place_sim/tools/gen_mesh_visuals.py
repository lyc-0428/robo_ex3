#!/usr/bin/env python3
"""视觉升级工具: 计算官方 robomaster_ros mesh 移植到本仓库简化 URDF 的变换。

用法:  python3 tools/gen_mesh_visuals.py
输出:  每块 mesh 在本仓库各 link 帧下的 origin/rpy/scale (splicing 参考)
依赖:  纯标准库 (numpy 可选, 缺失时用内置矩阵运算)
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

MESH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'meshes')
COLLADA_NS = '{http://www.collada.org/2005/11/COLLADASchema}'

# ---------- 基础线性代数 ----------
def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def norm(a):
    return math.sqrt(dot(a, a))

def unit(a):
    n = norm(a)
    return tuple(x/n for x in a)

def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))

def add(a, b):
    return tuple(x+y for x, y in zip(a, b))

def mul(a, s):
    return tuple(x*s for x in a)

def rot_about(u, th):
    """绕单位向量 u 旋转 th 的旋转矩阵 (行主序 3x3 tuple)."""
    c, s = math.cos(th), math.sin(th)
    x, y, z = u
    return (
        (c+x*x*(1-c),    x*y*(1-c)-z*s,  x*z*(1-c)+y*s),
        (y*x*(1-c)+z*s,  c+y*y*(1-c),    y*z*(1-c)-x*s),
        (z*x*(1-c)-y*s,  z*y*(1-c)+x*s,  c+z*z*(1-c)),
    )

def mat_vec(R, v):
    return tuple(dot(row, v) for row in R)

def mat_mat(A, B):
    BT = tuple(zip(*B))
    return tuple(tuple(dot(row, col) for col in BT) for row in A)

def det3(R):
    (a,b,c),(d,e,f),(g,h,i) = R
    return a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)

def rpy_from_R(R):
    """R = Rz(y)Ry(p)Rx(r), 返回 (r,p,y) 弧度. p=±90° 时取 y=0."""
    r31, r32, r33 = R[2]
    if abs(abs(r31)-1.0) < 1e-9:
        p = -math.pi/2 if r31 > 0 else math.pi/2
        r = 0.0
        y = math.atan2(-R[0][1], R[1][1])
    else:
        p = math.asin(-r31)
        r = math.atan2(r32, r33)
        y = math.atan2(R[1][0], R[0][0])
    return r, p, y

# ---------- DAE 包围盒 ----------
def dae_bbox(mesh):
    path = os.path.join(MESH_DIR, mesh)
    tree = ET.parse(path)
    pts = []
    for fa in tree.iter(COLLADA_NS + 'float_array'):
        if 'positions' in fa.get('id', ''):
            pts = [float(x) for x in fa.text.split()]
            break
    if not pts:
        raise RuntimeError(f'{mesh}: no positions found')
    n = len(pts) // 3
    mn = [min(pts[i::3]) for i in range(3)]
    mx = [max(pts[i::3]) for i in range(3)]
    return tuple(mn), tuple(mx)

# ---------- 跨段求解: s*R*a+t=A, s*R*b+t=B, roll 使 |c'| 的 y 最小 ----------
def solve_span(a, b, A, B, c=None, keep_y_min=True, roll_step=math.pi/720):
    s = norm(sub(B, A)) / norm(sub(b, a))
    u_src = unit(sub(b, a))
    u_dst = unit(sub(B, A))
    # 基础旋转: u_src -> u_dst (Rodrigues), 共线时取单位阵
    d = dot(u_src, u_dst)
    if d > 0.999999:
        R0 = ((1,0,0),(0,1,0),(0,0,1))
    elif d < -0.999999:
        # 反向: 绕任意垂直轴转 180°
        perp = unit(cross(u_src, (1,0,0))) if abs(u_src[0]) < 0.9 else unit(cross(u_src, (0,1,0)))
        R0 = rot_about(perp, math.pi)
    else:
        axis = unit(cross(u_src, u_dst))
        th = math.acos(max(-1, min(1, d)))
        R0 = rot_about(axis, th)
    if c is None:
        t = sub(A, mul(mat_vec(R0, a), s))
        return s, R0, t
    best = None
    for i in range(1440):
        th = i * roll_step
        R = mat_mat(rot_about(u_dst, th), R0)
        cp = mat_vec(R, c)
        val = abs(cp[1]) if keep_y_min else abs(cp[2])
        if best is None or val < best[0]:
            best = (val, R)
    _, R = best
    t = sub(A, mul(mat_vec(R, a), s))
    return s, R, t

def fmt(s, R, t):
    r, p, y = rpy_from_R(R)
    return (f'origin xyz="{t[0]:.4f} {t[1]:.4f} {t[2]:.4f}" '
            f'rpy="{r:.4f} {p:.4f} {y:.4f}" scale="{s:.4f} {s:.4f} {s:.4f}"')

def show(title, *items):
    print(f'===== {title} =====')
    for name, line in items:
        print(f'  <visual name="{name}"> {line}  -> {os.path.join(MESH_DIR, name + ".dae")}')

# ============ 1. 底盘 base_link (官方 chassis_base_link -> 本仓库 base_link, 平移 (0,0,0.0246)) ============
DZ_BASE = 0.0246  # 官方 chassis_base 高于官方 base(地面) 0.03465; 本仓库 base 高 0.01
base_parts = [
    ('chassis',         (-0.0059039,  0.0004154,  0.0527955), (0,0,0), 1.0),
    ('armor',           (-0.0064553, -0.078141,   0.0292877), (0,0,0), 1.0),
    ('battery',         (-0.130336,  -0.0018702,  0.0174177), (0,0,0), 1.0),
    ('battery_pin_left',(-0.1158907,  0.066115,   0.0525243), (0,0,0), 1.0),
    ('battery_pin_right',(-0.1158907,-0.0658844,  0.0525243), (0,0,0), 1.0),
    ('controller',      (-0.1142288, -0.0006716,  0.052675),  (0,0,0), 1.0),
    ('controller_cover',(-0.0787757, -0.002522,   0.0720827), (0,0,0), 1.0),
]
out = []
for name, xyz, rpy, sc in base_parts:
    x, y, z = xyz
    out.append((name, f'origin xyz="{x:.4f} {y:.4f} {z + DZ_BASE:.4f}" '
                     f'rpy="{rpy[0]} {rpy[1]} {rpy[2]}" scale="{sc} {sc} {sc}"'))
show('base_link <- 官方 chassis_base_link', *out)

# 轮子: 官方 base_link 帧 (±0.1, ±0.1, 0.01535) -> 本仓库 base_link 帧
out = []
for name, x, y in [('front_left',0.1,0.1),('front_right',0.1,-0.1),
                   ('rear_left',-0.1,0.1),('rear_right',-0.1,-0.1)]:
    out.append((f'{name}_wheel',
                f'origin xyz="{x:.4f} {y:.4f} {0.01535-0.01:.4f}" rpy="0 0 0" scale="1 1 1"'))
show('base_link <- 官方车轮', *out)

# ============ 2. 云台 yaw_link (官方 gimbal_link -> 本仓库 yaw_link, 平移 (0,0,-0.032)) ============
DZ_GIM = -0.032  # 官方云台偏航轴 z=0.118(世界), 本仓库 yaw 轴 z=0.15(世界)
out = [
    ('gimbal',          f'origin xyz="-0.0000 -0.0010 {0.03369+DZ_GIM:.4f}" rpy="0 0 0" scale="1 1 1"'),
    ('gimbal_left_armor', f'origin xyz="-0.0047  0.0642 {0.07122+DZ_GIM:.4f}" rpy="0 0 0" scale="1 1 1"'),
    ('gimbal_right_armor',f'origin xyz="-0.0053 -0.0628 {0.07314+DZ_GIM:.4f}" rpy="0 0 0" scale="1 1 1"'),
    ('gimbal_side',     f'origin xyz="-0.0003  0.0003 {0.08755+DZ_GIM:.4f}" rpy="0 0 0" scale="1 1 1"'),
]
show('yaw_link <- 官方 gimbal_link', *out)

# 水弹枪 blaster: blaster_link 帧 = gimbal_link + (0,0,0.08755) -> yaw_link (0,0,0.08755-0.032)
DZ_BL = 0.08755 + DZ_GIM
out = [
    ('blaster_gun',    f'origin xyz="0.0390  0.0033 {-0.01071+DZ_BL:.4f}" rpy="0 0 0" scale="1 1 1"'),
    ('blaster_magazin',f'origin xyz="-0.0203  0.0000 { 0.02866+DZ_BL:.4f}" rpy="0 0 0" scale="1 1 1"'),
    ('blaster_top',    f'origin xyz="0.0014  0.0008 { 0.01703+DZ_BL:.4f}" rpy="0 0 0" scale="1 1 1"'),
    ('speaker',        f'origin xyz="-0.0130  0.0069 {-0.06117+DZ_BL:.4f}" rpy="0 0 0" scale="1 1 1"'),
]
show('yaw_link <- 官方 blaster/speaker', *out)

# 炮塔底座 gimbal_base: 官方 gimbal_base_link = 官方 base_link + (-0.002,0,0.118) -> yaw_link
DZ_GB = 0.118 - 0.01 - 0.14
out = [('gimbal_base', f'origin xyz="{-0.002:.4f} 0.0 {DZ_GB:.4f}" rpy="0 0 0" scale="1 1 1"')]
show('yaw_link <- 官方 gimbal_base_link', *out)

# 支臂底座 mast: 官方 arm_base_link 帧, 官方 arm_1_joint 在其帧内 (0.0104,-0.0256,0.0307)
#   本仓库 lift 轴在 yaw 帧 (0.18,0,0.02) -> arm_base 帧平移
T_AB = (0.18 - 0.0103961, 0.0 - (-0.0255713), 0.02 - 0.030741)
out = [
    ('arm_base',    f'origin xyz="{-0.0009+T_AB[0]:.4f} {-0.0030+T_AB[1]:.4f} {0.0+T_AB[2]:.4f}" rpy="0 0 0" scale="1 1 1"'),
    ('left_servo',  f'origin xyz="{-0.0135+T_AB[0]:.4f} {0.0+T_AB[1]:.4f} {-0.0061+T_AB[2]:.4f}" rpy="0 0 0" scale="1 1 1"'),
    ('right_servo', f'origin xyz="{-0.0135+T_AB[0]:.4f} {0.0+T_AB[1]:.4f} {-0.0061+T_AB[2]:.4f}" rpy="0 0 0" scale="1 1 1"'),
]
show('yaw_link <- 官方 arm_base (mast)', *out)

# ============ 3. 上臂 upper_arm_link (官方 arm_1_link 帧 -> 本仓库上臂帧) ============
# 锚点 (官方 arm_1 帧):  lift 轴 (0,0,0); arm_2 轴 (0.00187, 0.04387, 0.12102)
# 目标 (本仓库 upper_arm 帧): (0,0,0) -> (0.28,0,0)
arm1_vis = (0.0073108, 0.0184816, 0.0437162)
a1 = mul(arm1_vis, -1.0)
b1 = sub((0.0018704, 0.0438659, 0.1210238), arm1_vis)
c1 = sub((-0.0239031, 0.0255713, -0.0368083), arm1_vis)  # 圆柱视觉原点 (第3参考点)
s1, R1, t1 = solve_span(a1, b1, (0,0,0), (0.28,0,0), c=c1)
print('arm_1:  span |src|=%.4f -> |dst|=0.28, s=%.4f' % (norm(sub(b1,a1)), s1))
print('  arm_2 轴落点: %s (应为 (0.28,0,0))' % (tuple(round(x,4) for x in add(mul(mat_vec(R1,b1),s1),t1)),))
print('  圆柱参考点 y: %.4f' % mat_vec(R1,c1)[1])
show('upper_arm_link <- 官方 arm_1_link', *[(n, fmt(s1,R1,t1)) for n in ('arm_1','arm_1_cylinder')])

# ============ 4. 前臂 wrist_link (官方 arm_2_link 帧 -> 本仓库腕帧) ============
arm2_vis = (0.0256122, -0.020134, -0.001308)
a2 = mul(arm2_vis, -1.0)
b2 = sub((0.1058557, 0.006752, -0.0561093), arm2_vis)  # bracket 轴在 arm_2 帧
c2 = sub((0.0, 0.0, 0.0), arm2_vis)
s2, R2, t2 = solve_span(a2, b2, (0,0,0), (0.19,0,0), c=c2)
print('arm_2:  |src|=%.4f -> |dst|=0.19, s=%.4f' % (norm(sub(b2,a2)), s2))
print('  bracket 轴落点: %s (应为 (0.19,0,0))' % (tuple(round(x,4) for x in add(mul(mat_vec(R2,b2),s2),t2)),))
show('wrist_link <- 官方 arm_2_link', *[(n, fmt(s2,R2,t2)) for n in ('arm_2','arm_2_bar_1','arm_2_bar_2')])

# ============ 5. 腕托 gripper_base_link (官方 endpoint_bracket_link + gripper_link + 手指) ============
# bracket: 官方帧 bracket 轴 (0,0,0), end_point (0.01101,-0.02703,-0.01038)
#   本仓库 gripper_base 帧: wrist 轴在 (-0.22,0,0); arm_2 视觉末端在 (-0.22+0.19,0,0)=(-0.03,0,0)
br_vis = (0.0071657, -0.0245846, 0.0008407)
a_br = mul(br_vis, -1.0)
b_br = sub((0.0110125, -0.0270264, -0.0103816), br_vis)
s_br, R_br, t_br = solve_span(a_br, b_br, (-0.03,0,0), (0,0,0), c=None)
print('bracket: |src|=%.4f -> 0.03, s=%.4f' % (norm(sub(b_br,a_br)), s_br))
show('gripper_base_link <- 官方 endpoint_bracket', ('endpoint_bracket', fmt(s_br,R_br,t_br)))

# 手掌 gripper_base mesh: 官方 gripper_link 帧 = end_point + (0.00028,-0.02703,0.00018)
#   本仓库: 手掌应填满 gripper_base 框 (x 0.06 / y 0.13 / z 0.04), 用包围盒定标
for m in ('gripper_base', 'endpoint_bracket', 'gripper_left_1', 'gripper_left_2',
          'gripper_left_5', 'gripper_left_7', 'arm_base', 'rod', 'rod_1', 'rod_2',
          'rod_3', 'triangle', 'wheel'):
    mn, mx = dae_bbox(m + '.dae')
    sz = sub(mx, mn)
    print(f'bbox {m}: min={tuple(round(v,4) for v in mn)} max={tuple(round(v,4) for v in mx)} size={tuple(round(v,4) for v in sz)}')
