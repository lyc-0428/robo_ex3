# 机械臂定点抓取与放置系统

## ROS 2 Humble · Gazebo Fortress · MoveIt 2 · ros2_control · 仿真开发与真机验证

**机器人集成小组项目Ⅰ｜机械臂定点抓取实验**

项目仓库：

```text
https://github.com/X-u-e-B-a-o/robomaster-pick-place-sim
```

---

## 0. README 说明与实验数据口径

本 README 根据课程《机械臂定点抓取》实验要求、仓库 `main` / `real-ep-arm` 等分支中的程序与配置，以及当前已保存的实验材料进行整理。

本文严格区分三类证据：

1. **程序实现**：代码已经实现的功能，例如 5 次循环、IK/FK 检查、路径预规划、错误处理和日志保存。
2. **静态可验证结果**：可由当前参数和运动学模型直接复算的结果，例如 A/B 点是否存在解析逆解、关节角是否位于限制范围内。
3. **实际运行结果**：必须来自某一次真实运行保存的 `run_*_results.json`、轨迹 CSV、完整日志或连续实验视频。

当前仓库和已上传材料能够确认完整的软件实现与静态运动学验证，但**未发现一组可独立核验的“仿真 5 次正式抓取结果文件”以及“真机 5 次正式抓取结果文件”**。因此本文不会把“程序设置了 5 次循环”写成“实际 5/5 成功”，也不会虚构 Trial 成功记录。第 16、26 节将五次验收表按当前证据状态完整填写。

---

# 1. 实验目标

本实验要求实现机械臂固定位置抓取与放置：目标物位于固定取物点 **A**，机械臂完成抓取后将目标物移动到固定放置区域 **B**。实验重点是机械臂控制与系统集成，不考查视觉定位。

完整动作流程为：

```text
Home
  ↓
Pre-Pick
  ↓
Pick A
  ↓
Close Gripper
  ↓
Lift
  ↓
Pre-Place
  ↓
Place B
  ↓
Release
  ↓
Withdraw
  ↓
Home
```

课程验收要求为：

```text
连续抓取 5 次
至少成功 4 次
```

因此：

\[
N_{\mathrm{success}}\geq 4,\qquad N_{\mathrm{trial}}=5
\]

\[
\mathrm{SuccessRate}
=
\frac{N_{\mathrm{success}}}{5}\times100\%
\geq80\%
\]

此外，系统需要满足：

- 不发生明显桌面碰撞、自碰撞或目标物危险碰撞；
- 不超过机械臂关节限位；
- 不可达、无逆解或规划失败时安全停止；
- 保存机械臂轨迹、执行结果和错误日志；
- 真机阶段复用仿真阶段的任务逻辑，仅替换设备、通信、运动学和位置参数。

---

# 2. 项目总体设计

本项目采用分层设计，将任务层、运动学层、规划层和设备层分离。

```text
Task Parameters
A / B / Home / Safe Height / Gripper
                 │
                 ▼
        pick_place_moveit
        Unified Task Logic
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
Analytical IK          MoveIt 2
robot_kinematics       FK / OMPL
       │                   │
       └─────────┬─────────┘
                 ▼
          ROS 2 Interface
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
Gazebo Simulation      Hardware Adapter
ros2_control           EP / other robot
      │                     │
      ▼                     ▼
Simulated Robot        Physical Robot

                 │
                 ▼
        Log / CSV / JSON
```

该架构的核心思想是：

> **Task Logic Reuse + Device Layer Replacement**

即仿真和真机尽量共用同一任务状态机，而不是切换设备后重写整个抓取程序。

---

# 3. Git 分支与版本关系

仓库中存在多个开发阶段。

## 3.1 `main`

主要包含：

- ROS 2 仿真 package；
- Gazebo world；
- RoboMaster EP/Core URDF 与 mesh；
- MoveIt 2 配置；
- 参数文件；
- `lpx/` 后期更新实现；
- 静态模型快照。

## 3.2 `real-ep-arm`

该分支用于 RoboMaster EP 真机适配，包含：

```text
robomaster_ep_driver/
├── config/
│   ├── ep_driver_params.yaml
│   └── pick_place_params_real.yaml
├── launch/
│   └── pick_place_real.launch.py
└── robomaster_ep_driver/
    ├── ep_arm_driver.py
    └── calibrate_real.py

docs/
└── real_machine_runbook.md
```

其设计目标是把 RoboMaster SDK 封装为与仿真阶段相近的 ROS 2 控制接口，使高层 `pick_place_moveit` 任务节点能够复用。

## 3.3 `lpx/`

`main/lpx/` 保存了较完整的后期实现，包括：

```text
pick_place_moveit.py
robot_kinematics.py
calibrate_pose.py
test_joints.py
robomaster_ep_driver/
```

最终提交前建议将已经验证的最终实现统一到一个 canonical package，避免根目录和 `lpx/` 同时存在行为不同的版本。

---

# 4. 推荐项目结构

```text
robomaster-pick-place-sim/
│
├── README.md
├── package.xml
├── setup.py
├── setup.cfg
│
├── config/
│   ├── controllers.yaml
│   └── pick_place_params.yaml
│
├── launch/
│   ├── pick_place_gazebo.launch.py
│   └── pick_place_moveit.launch.py
│
├── worlds/
│   └── pick_place.sdf
│
├── urdf/
│   └── robomaster_ep_gazebo.urdf
│
├── meshes/
│   └── ...
│
├── robomaster_pick_place_sim/
│   ├── __init__.py
│   ├── pick_place_moveit.py
│   ├── robot_kinematics.py
│   ├── calibrate_pose.py
│   └── test_joints.py
│
├── robomaster_ep_moveit_config/
│   ├── config/
│   ├── launch/
│   └── rviz/
│
├── robomaster_ep_driver/
│   ├── config/
│   │   ├── ep_driver_params.yaml
│   │   └── pick_place_params_real.yaml
│   ├── launch/
│   │   └── pick_place_real.launch.py
│   └── robomaster_ep_driver/
│       ├── ep_arm_driver.py
│       └── calibrate_real.py
│
├── docs/
│   └── real_machine_runbook.md
│
└── experiment_results/
    ├── simulation/
    └── real_robot/
```

---

# 5. 软件环境

仿真开发环境：

```text
Ubuntu 22.04
ROS 2 Humble
Ignition Gazebo / Gazebo Fortress
MoveIt 2
OMPL
ros2_control
ros2_controllers
gz_ros2_control
Python 3
```

常用依赖：

```bash
sudo apt update

sudo apt install \
    ros-humble-moveit \
    ros-humble-ros-gz \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-controller-manager
```

设置 Gazebo：

```bash
export GZ_VERSION=fortress
```

---

# 6. 构建项目

```bash
mkdir -p ~/colcon_ws/src
cd ~/colcon_ws/src

git clone https://github.com/X-u-e-B-a-o/robomaster-pick-place-sim.git
```

如需使用完整真机适配版本：

```bash
cd robomaster-pick-place-sim
git checkout real-ep-arm
```

安装依赖并编译：

```bash
cd ~/colcon_ws

source /opt/ros/humble/setup.bash

rosdep install \
    --from-paths src \
    --ignore-src \
    -r \
    -y

colcon build --symlink-install

source install/setup.bash
```

检查 ROS 2 Python 节点：

```bash
ros2 pkg executables robomaster_pick_place_sim
```

更新版本应至少包含：

```text
pick_place_moveit
calibrate_pose
test_joints
```

---

# 7. Gazebo 仿真场景

仿真环境包括：

- 地面；
- 实验桌；
- RoboMaster EP/Core 机械臂；
- 夹爪；
- 固定取物点 A；
- 固定放置区域 B；
- 标准目标物。

当前基础场景中的目标物采用约 7 cm 立方体：

| 参数 | 数值 |
|---|---:|
| 尺寸 | 0.07 × 0.07 × 0.07 m |
| 质量 | 约 0.04 kg |
| 摩擦系数 | 约 1.5 |
| 初始位置 | A 点 |

本任务不使用视觉定位，目标位置由参数配置预先给出。

---

# 8. ros2_control

仿真主要使用：

```text
joint_state_broadcaster
arm_controller
gripper_controller
```

机械臂控制关节：

```text
base_yaw_joint
arm_lift_joint
wrist_pitch_joint
```

MoveIt 轨迹接口：

```text
/arm_controller/follow_joint_trajectory
```

夹爪命令接口：

```text
/gripper_controller/commands
```

状态反馈：

```text
/joint_states
```

主任务节点在运动开始前等待完整 Joint State，避免 MoveIt 使用未初始化的关节状态。

---

# 9. 核心程序

## 9.1 `pick_place_moveit.py`

负责：

- 参数加载；
- Controller/Action/Service 就绪检查；
- Joint State 检查；
- PlanningScene 初始化；
- 解析 IK；
- MoveIt FK 验证；
- 整体路径预规划；
- 机械臂轨迹执行；
- 夹爪开合；
- Collision Object Attach/Detach；
- Joint Limit 检查；
- 异常停止与安全恢复；
- 五次 Trial 统计；
- Log/CSV/JSON 持久化。

## 9.2 `robot_kinematics.py`

仿真机械臂采用三自由度解析模型：

```text
base_yaw_joint
      ↓
arm_lift_joint
      ↓
wrist_pitch_joint
      ↓
gripper_base_link
```

模型参数约为：

```text
Shoulder = (0.18, 0.16) m
L1 = 0.28 m
L2 = 0.22 m
```

## 9.3 `calibrate_pose.py`

用于输入 `(x, y, z)` 并检查：

```text
Cartesian Target
→ Analytical IK
→ MoveIt Plan
→ Move
→ FK
→ Joint Angles / End-Effector Pose
```

## 9.4 `test_joints.py`

用于控制器 smoke test。该脚本绕过完整 MoveIt 碰撞规划，只应在调试阶段使用。

---

# 10. 解析逆运动学

对于目标：

\[
P=(x,y,z)
\]

水平径向距离为：

\[
r=\sqrt{x^2+y^2}
\]

底座偏航角为：

\[
\theta=\operatorname{atan2}(y,x)
\]

相对肩关节：

\[
d_x=r-x_s
\]

\[
d_z=z-z_s
\]

\[
d=\sqrt{d_x^2+d_z^2}
\]

可达条件：

\[
|L_1-L_2|\le d\le L_1+L_2
\]

程序同时检查：

- 是否存在解析解；
- 关节是否超限；
- 夹爪方向是否有效；
- 多解情况下的优选解。

---

# 11. MoveIt FK 交叉验证

正式实验之前，系统执行：

```text
Desired XYZ
   ↓
Analytical IK
   ↓
Joint Angles
   ↓
MoveIt /compute_fk
   ↓
Recovered XYZ
   ↓
Error Check
```

当前代码采用约 **5 mm** 的位置容差。

如果解析运动学与 URDF/MoveIt 几何不一致，则在机械臂真正运动之前终止任务。

---

# 12. 关节限位

当前仿真软件限制：

| Joint | Minimum | Maximum |
|---|---:|---:|
| `base_yaw_joint` | -3.14 rad | 3.14 rad |
| `arm_lift_joint` | -0.80 rad | 1.00 rad |
| `wrist_pitch_joint` | -1.50 rad | 1.50 rad |
| `left_finger_joint` | 0 m | 0.045 m |
| `right_finger_joint` | 0 m | 0.045 m |

---

# 13. 仿真任务参数

当前后期配置使用的主要参数为：

| 参数 | 数值 |
|---|---|
| `num_cycles` | 5 |
| `gripper_open` | 0.040 m |
| `gripper_closed` | 0.006 m |
| `max_velocity_scale` | 0.5 |
| `max_acceleration_scale` | 0.5 |
| `pick_point` | (0.600, 0.000, 0.350) |
| `place_point` | (0.550, 0.250, 0.350) |
| `pre_pick_point` | (0.596, 0.000, 0.399) |
| `pre_place_point` | (0.546, 0.248, 0.395) |
| `object_size` | 0.070 m |
| `attach_object` | true |
| `log_dir` | results |

---

# 14. 仿真关键点静态运动学结果

根据当前参数和仓库中的解析运动学模型，四个关键点可得到以下静态解：

| 目标点 | XYZ / m | Yaw / rad | Lift / rad | Wrist / rad | 解析 IK | 关节限位 |
|---|---|---:|---:|---:|---|---|
| Pick A | (0.600, 0.000, 0.350) | 0.0000 | -0.7748 | 0.8015 | Pass | Pass |
| Place B | (0.550, 0.250, 0.350) | 0.4266 | -0.7537 | 0.7611 | Pass | Pass |
| Pre-Pick | (0.596, 0.000, 0.399) | 0.0000 | -0.7736 | 0.5752 | Pass | Pass |
| Pre-Place | (0.546, 0.248, 0.395) | 0.4264 | -0.7548 | 0.5573 | Pass | Pass |

因此，**当前四个预设关键点在解析运动学模型下均可达，并且关节解落在软件限制范围内**。

注意：该表属于**静态运动学验证结果**，不是物理抓取成功率。

---

# 15. 完整路径预检查与任务状态机

正式开始五次循环前，代码会首先进行完整路径 `plan_only`：

```text
Home
→ Pre-Pick
→ Pick
→ Pre-Pick
→ Pre-Place
→ Place
→ Pre-Place
→ Home
```

共检查 7 段路径。

单次正式 Trial：

```text
Home
→ Pre-Pick
→ Pick
→ Close Gripper
→ Attach Object
→ Lift
→ Pre-Place
→ Place
→ Detach Object
→ Release
→ Withdraw
→ Home
```

任一关键步骤发生 `TaskError`，该 Trial 应被判定为失败，并进入停止/恢复逻辑。

---

# 16. 仿真实验结果

## 16.1 软件与静态验收结果

| 验收项 | 当前结果 |
|---|---|
| 单 Launch 完整启动架构 | 已实现 |
| 5 次循环逻辑 | 已实现 |
| ≥4/5 自动 Pass 判定 | 已实现 |
| A 点解析 IK | Pass |
| B 点解析 IK | Pass |
| Pre-Pick 解析 IK | Pass |
| Pre-Place 解析 IK | Pass |
| 四个关键点关节限位 | Pass |
| MoveIt FK 交叉验证 | 已实现 |
| 7 段 plan-only 预检查 | 已实现 |
| PlanningScene 桌面/目标物 | 已实现 |
| Object Attach/Detach | 已实现 |
| 关节超限安全处理 | 已实现 |
| 不可达目标处理 | 已实现 |
| 错误日志 | 已实现 |
| 轨迹 CSV | 已实现 |
| 结果 JSON | 已实现 |

## 16.2 连续 5 次仿真验收表

当前已上传材料及可读取仓库内容中，**没有找到一份正式运行生成的五次 Trial 结果 JSON/CSV/log**。因此，五次 Trial 的实际结果不能从代码配置反推。

| Trial | 任务配置 | 关键点静态可达性 | 实际抓取运行证据 | 实际 Result |
|---:|---|---|---|---|
| 1 | 已配置 | Pass | 未发现对应正式 Trial 运行记录 | **Not verified** |
| 2 | 已配置 | Pass | 未发现对应正式 Trial 运行记录 | **Not verified** |
| 3 | 已配置 | Pass | 未发现对应正式 Trial 运行记录 | **Not verified** |
| 4 | 已配置 | Pass | 未发现对应正式 Trial 运行记录 | **Not verified** |
| 5 | 已配置 | Pass | 未发现对应正式 Trial 运行记录 | **Not verified** |

当前可以得出的正式结论是：

```text
Static kinematic feasibility: PASS
Joint-limit feasibility: PASS
Five-trial execution framework: IMPLEMENTED
Actual 5-trial success count: NOT AUDITABLE FROM CURRENT SAVED RESULTS
Course simulation acceptance (≥4/5): CANNOT BE CLAIMED FROM CURRENT EVIDENCE
```

这里的 `Not verified` 不是程序失败，而是表示当前提交材料中缺少可独立审计的对应运行结果。

---

# 17. 仿真启动方法

```bash
source /opt/ros/humble/setup.bash
source ~/colcon_ws/install/setup.bash

export GZ_VERSION=fortress

ros2 launch robomaster_pick_place_sim pick_place_moveit.launch.py
```

带 RViz：

```bash
ros2 launch robomaster_pick_place_sim \
    pick_place_moveit.launch.py \
    use_rviz:=true
```

分步调试：

```bash
ros2 launch robomaster_pick_place_sim pick_place_gazebo.launch.py
```

另一终端：

```bash
ros2 run robomaster_pick_place_sim test_joints
```

检查控制器：

```bash
ros2 control list_controllers
```

---

# 18. 仿真输出

程序设计为每次运行生成：

```text
results/
├── run_<timestamp>.log
├── run_<timestamp>_trajectory.csv
└── run_<timestamp>_results.json
```

典型轨迹字段：

```text
timestamp
cycle
step
base_yaw
arm_lift
wrist_pitch
left_finger
right_finger
ee_x
ee_y
ee_z
success
```

结果 JSON 应记录：

```text
num_cycles
successes
failures
passed
cycles
duration_s
error
params
log_file
trajectory_file
```

验收逻辑：

```text
passed = successes >= 0.8 * num_cycles
```

对于 5 次循环，即至少 4 次成功。

---

# 19. 异常与安全处理

| 异常 | 检测方式 | 系统响应 |
|---|---|---|
| 目标不可达 | Analytical IK | 启动前中止 |
| 无合法逆解 | IK + Joint Limit | 输出错误并停止 |
| IK/URDF 不一致 | MoveIt FK | 中止预检 |
| 路径发生碰撞 | MoveIt Planning | 不执行 |
| 无合法路径 | MoveIt Error Code | Trial Fail |
| Joint 超限 | Joint State Check | Stop |
| Controller 未就绪 | Startup Check | 不开始 |
| Joint State 不完整 | Startup Check | 不开始 |
| 执行失败 | Action Result | Stop + Record |
| 用户中断 | Ctrl+C / Exception | Stop + Recovery |

任务层恢复流程：

```text
Error
 ↓
Stop Current Trajectory
 ↓
Record Failure
 ↓
Attempt Home
 ↓
If Home Fails
 ↓
Remain Stopped
```

---

# 20. 真机：仓库中的 RoboMaster EP 适配方案

`real-ep-arm` 分支实现了一个 RoboMaster EP ROS 2 兼容层。

对上层提供/模拟的接口包括：

```text
/joint_states
/arm_controller/follow_joint_trajectory
/gripper_controller/commands
/controller_manager/list_controllers
/ep_arm/freeze
```

因此高层任务仍可使用：

```text
pick_place_moveit
```

而设备执行端由：

```text
ep_arm_driver
→ RoboMaster SDK
→ Physical RoboMaster EP
```

完成。

---

# 21. 真机运动映射

RoboMaster EP 实际机械臂的运动结构不同于仿真中的 3-DOF 模型，因此真机驱动将 MoveIt 轨迹转换为二维末端坐标。

仿真 FK：

\[
x_s =
x_0+
L_1\cos q_1+
L_2\cos(q_1+q_2)
\]

\[
z_s =
z_0-
L_1\sin q_1-
L_2\sin(q_1+q_2)
\]

标定后映射：

\[
x_r=k_xx_s+b_x
\]

\[
y_r=k_zz_s+b_z
\]

最后通过 RoboMaster SDK：

```text
robotic_arm.moveto(x, y)
```

向真实机械臂发送运动目标。

---

# 22. 真机双锚点标定

仓库默认映射示例为：

| 点 | 仿真平面坐标 | 真机坐标 |
|---|---|---|
| Anchor A | (0.60, 0.32) m | (60, 40) mm |
| Anchor B | (0.64, 0.35) m | (170, 60) mm |
| Home | — | (32, 114) mm |

对应示例映射：

\[
x_r = 2750x_s-1590
\]

\[
y_r \approx 666.67z_s-173.33
\]

这些值仅是仓库中的默认标定关系。正式真机实验应使用现场标定得到的 `anchor1_real`、`anchor2_real` 和 `home_real`。

标定工具：

```bash
ros2 run robomaster_ep_driver calibrate_real
```

典型交互命令：

```text
pos
moveto <x> <y>
move <dx> <dy>
open
close
anchor1
anchor2
home
show
save
quit
```

---

# 23. RoboMaster EP 真机参数

仓库后期真机参数主要为：

| 参数 | 数值 |
|---|---|
| `num_cycles` | 5 |
| `gripper_open` | 0.040 m |
| `gripper_closed` | 0.006 m |
| `max_velocity_scale` | 0.2 |
| `max_acceleration_scale` | 0.2 |
| `pick_point` | (0.60, 0.00, 0.32) |
| `place_point` | (0.64, 0.00, 0.35) |
| `pre_pick_point` | (0.60, 0.00, 0.38) |
| `pre_place_point` | (0.62, 0.00, 0.38) |
| `home_joints` | [0.0, -0.751, 0.385] |
| `object_size` | 0.07 m |
| `grip_settle_s` | 2.5 s |
| `log_dir` | results_real |

默认真实工作空间保护：

```text
x ∈ [0, 220] mm
y ∈ [0, 150] mm
```

轨迹 waypoint 发送前先完成坐标转换和边界检查。

---

# 24. 真机 Dry-Run 与启动

正式连接真机前应先运行：

```bash
ros2 launch robomaster_ep_driver \
    pick_place_real.launch.py \
    dry_run:=true
```

Dry-Run 用于检查：

```text
ROS 2 interfaces
MoveIt planning
IK / FK
Task state machine
simulation-to-real coordinate mapping
workspace limits
```

但不应把 Dry-Run 当成物理抓取成功。

正式运行：

```bash
ros2 launch robomaster_ep_driver pick_place_real.launch.py
```

软件冻结：

```bash
ros2 service call \
    /ep_arm/freeze \
    std_srvs/srv/SetBool \
    "{data: true}"
```

解除：

```bash
ros2 service call \
    /ep_arm/freeze \
    std_srvs/srv/SetBool \
    "{data: false}"
```

软件冻结不能替代硬件断电/急停。

---

# 25. 真机安全要求

第一次真实运动必须低速进行，并建议双人操作：

```text
Operator A:
运行程序、观察日志

Operator B:
观察机械臂和工作空间
负责急停
```

启动前检查：

- 底座固定；
- 电源稳定；
- USB/串口/网络通信正常；
- 夹爪安装牢固；
- 急停/断电装置有效；
- 工作空间无人员身体或无关障碍物；
- A/B 已标记；
- Home 已验证；
- 安全高度已确认；
- 目标物质量符合要求；
- 关节与工作空间限制有效。

---

# 26. 真机实验结果

## 26.1 真机软件准备状态

| 验收项 | 当前证据状态 |
|---|---|
| 真机设备适配层 | 已实现 RoboMaster EP adapter |
| FollowJointTrajectory 兼容接口 | 已实现 |
| `/joint_states` | 已实现 |
| 夹爪命令桥接 | 已实现 |
| Controller 状态兼容 | 已实现 |
| 真机 A/B 参数 | 已配置 |
| Home 参数 | 已配置 |
| 双锚点标定工具 | 已实现 |
| Dry-Run | 已实现 |
| 工作空间边界保护 | 已实现 |
| Freeze | 已实现 |
| 低速参数 | 已配置 |
| 5 次循环逻辑 | 已实现 |
| `results_real` 日志结构 | 已实现 |

## 26.2 连续 5 次真机验收表

当前已上传材料中，可以看到 Jetson/Ubuntu ROS 2 工作空间的真机环境调试痕迹，但**未发现能够逐 Trial 核验的五次真实抓取结果 JSON、CSV、完整执行日志或对应连续验收视频数据**。

因此表格按实际证据状态填写如下：

| Trial | Home | Pick A | Grasp | Transport | Place B | Safe Return | 实际 Result |
|---:|---|---|---|---|---|---|---|
| 1 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | **Not verified** |
| 2 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | **Not verified** |
| 3 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | **Not verified** |
| 4 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | **Not verified** |
| 5 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | 无可核验 Trial 记录 | **Not verified** |

当前真机结论：

```text
Real-hardware control architecture: IMPLEMENTED
Real-hardware parameterization: IMPLEMENTED
Dry-run / workspace safety logic: IMPLEMENTED
Actual physical five-trial result: NOT AUDITABLE FROM CURRENT SAVED RESULTS
Course real-robot acceptance (≥4/5): CANNOT BE CLAIMED FROM CURRENT EVIDENCE
```

---

# 27. 课程要求的真机平台与仓库真机分支的差异

课程指导书给出的真机平台为：

```text
Jetson Orin
+
mechArm 270
```

而仓库中能够完整确认的专门真机适配分支为：

```text
real-ep-arm
→ RoboMaster EP
```

因此二者不能在实验报告中混写。

如果 RoboMaster EP 是经课程教师批准的替代平台，应明确注明平台替换。

如果课程严格要求 mechArm 270，则 `real-ep-arm` 应描述为：

> **Sim-to-Real 接口复用和硬件适配方法验证**

而不能直接作为 “mechArm 270 五次真机验收结果”。

现有上传的一张 Jetson/Ubuntu 终端记录中出现过 `mycobot_interfaces` 工作空间配置，说明项目曾进行 mechArm/myCobot ROS 2 环境调试；但该记录同时显示对应 `local_setup.bash` 未找到，因此它属于环境调试证据，不等同于完成真实五次抓取验收。

---

# 28. 仿真与真机结果文件

仿真：

```text
results/
├── run_<timestamp>.log
├── run_<timestamp>_trajectory.csv
└── run_<timestamp>_results.json
```

RoboMaster EP 真机：

```text
results_real/
├── run_<timestamp>.log
├── run_<timestamp>_trajectory.csv
└── run_<timestamp>_results.json
```

正式提交建议将最终验收结果从临时输出目录复制到版本控制目录：

```text
experiment_results/
├── simulation/
│   ├── final_run.log
│   ├── final_run_trajectory.csv
│   └── final_run_results.json
│
└── real_robot/
    ├── final_run.log
    ├── final_run_trajectory.csv
    └── final_run_results.json
```

这样可以保证 README 中的每一个实验数字都能回溯到具体运行记录。

---

# 29. 当前仓库需要统一的关键问题

## 29.1 B 点显示位置与任务参数

部分后期任务参数为：

```text
place_point = (0.55, 0.25, 0.35)
```

但部分 Gazebo world 中绿色 B Marker 使用不同平面位置。

正式提交前应确保：

```text
Gazebo B marker
=
pick_place_params.yaml place_point
=
MoveIt task target
```

否则视觉上的 B 区域与程序实际放置目标不一致。

## 29.2 桌面几何

部分版本中：

```text
MoveIt PlanningScene table
```

与：

```text
Gazebo SDF table
```

使用了不同的中心位置和尺寸。

正式版必须统一，否则可能出现 Gazebo 实际碰撞与 MoveIt 碰撞模型不一致。

## 29.3 `setup.py`

根目录历史版本与 `lpx/setup.py` 的节点注册曾存在差异。

最终版本应确保至少正确注册：

```text
pick_place_moveit
calibrate_pose
test_joints
```

## 29.4 多套实现

提交前应明确：

```text
canonical branch
canonical config
canonical launch
canonical result files
```

避免老师无法判断根目录、`lpx/` 或其他分支哪一套是最终实验。

---

# 30. 实验验收判定原则

## 仿真

课程通过条件：

```text
5 trials
successes >= 4
no dangerous collision
no joint-limit violation
unreachable target handled safely
trajectory/result/error logs saved
```

## 真机

课程通过条件：

```text
5 trials
successes >= 4
object placed in designated region
no collision / limit / dangerous motion
failure → safe stop or safe return
same high-level task logic reused
all operators complete safety checks
```

只有真实运行数据满足上述条件，才应在 README 中写：

```text
PASS
```

---

# 31. 推荐正式实验记录格式

当最终结果文件提交后，推荐保留以下字段：

```text
run_id
git_commit
branch
parameter_file
robot_platform
trial
start_time
duration_s
home_success
pick_success
grasp_success
transport_success
place_success
return_success
collision
joint_limit_violation
error
result
```

这比只写 “5/5” 更具有可复现性和审计价值。

---

# 32. 最终提交前复现流程

建议在干净工作空间重新执行：

```bash
cd ~/colcon_ws

rm -rf build install log

source /opt/ros/humble/setup.bash

rosdep install \
    --from-paths src \
    --ignore-src \
    -r \
    -y

colcon build --symlink-install

source install/setup.bash
```

随后依次完成：

```text
1. Controller Test
2. MoveIt Test
3. Pose Calibration
4. Simulation Preflight
5. Five-Trial Simulation
6. Abnormal-Condition Test
7. Save Simulation Result Files
8. Hardware Calibration
9. Real Dry-Run
10. Low-Speed Real Trial
11. Five-Trial Real Experiment
12. Emergency-Stop Test
13. Save Real Result Files
14. Commit Final Evidence
```

---

# 33. 提交材料对应关系

| 课程要求 | 本仓库对应内容 |
|---|---|
| ROS 2 程序 | `robomaster_pick_place_sim` / driver package |
| Launch | Gazebo / MoveIt / real-hardware launch |
| 参数文件 | `pick_place_params*.yaml` |
| 机械臂模型 | URDF + meshes |
| 仿真场景 | SDF world |
| 取物点 A | `pick_point` |
| 放置点 B | `place_point` |
| 安全高度 | `pre_pick_point` / `pre_place_point` |
| 轨迹记录 | trajectory CSV |
| 执行结果 | results JSON |
| 错误日志 | `.log` |
| 仿真视频 | 正式五次连续实验 |
| 真机视频 | 正式五次连续实验 |
| 异常记录 | unreachable / planning / emergency test |
| 简要实验报告 | 项目实验报告 |
| 项目代码 | GitHub Repository |

---

# 34. 项目总结

本项目已经建立了一套较完整的机械臂固定点抓取软件体系，包括：

```text
ROS 2 Humble
Gazebo Fortress
MoveIt 2
OMPL
ros2_control
Analytical IK
MoveIt FK Cross-Validation
PlanningScene
Preflight Planning
Pick-and-Place State Machine
Gripper Control
Attach / Detach
Failure Handling
Log / CSV / JSON
Hardware Adapter
Sim-to-Real Interface Reuse
```

当前代码能够明确证明的是：

- 仿真 A/B 与安全中间点具有合法解析 IK；
- 关键关节角位于软件限制范围内；
- 完整五次抓取验收逻辑已经实现；
- 轨迹、结果与错误日志保存机制已经实现；
- 真机设备适配、标定、Dry-Run 与安全边界机制已经建立。

当前材料**不能严谨证明**的是：

- 某一次正式仿真实验已经取得 4/5 或 5/5；
- 某一次正式真机实验已经取得 4/5 或 5/5。

因此，本 README 不通过推测填造实验成功数据。正式实验结果应以提交的 `run_*_results.json`、对应轨迹、日志和连续视频为最终依据。

---

## Repository

```text
https://github.com/X-u-e-B-a-o/robomaster-pick-place-sim
```

