# RoboMaster EP 定点抓取仿真 (Gazebo + MoveIt 2)

机器人集成小组项目 Ⅰ | 机械臂定点抓取(小组实验)

用 Ignition Gazebo (Fortress) 做物理仿真、MoveIt 2 做运动规划与执行,
实现 RoboMaster EP 机械臂从固定取物点 A 抓取 7cm 立方体、放到固定放置点 B,
连续抓取 5 次至少成功 4 次。

## 仓库结构

```
robomaster_pick_place_sim/            # 仿真包 (世界 + 模型 + 任务节点)
├── urdf/robomaster_ep_gazebo.urdf    #   机械臂 URDF (含 gz_ros2_control 插件)
├── worlds/pick_place.sdf             #   Gazebo 世界: 地面/桌面/A点红色标记/B点绿色标记/目标物
├── config/controllers.yaml           #   ros2_control 控制器 (JTC 手臂 + 前向指令夹爪)
├── config/pick_place_params.yaml     #   抓取任务参数 (A/B 坐标、开合度、循环次数等)
├── launch/pick_place_gazebo.launch.py     #   基础仿真 (仅 Gazebo+控制器, 联调用)
├── launch/pick_place_moveit.launch.py     #   一键启动完整仿真+MoveIt+抓取任务
└── robomaster_pick_place_sim/        #   Python 节点
    ├── pick_place_moveit.py          #   核心抓取节点 (MoveIt 规划+执行+日志)
    ├── calibrate_pose.py             #   A/B 点交互式标定工具
    └── test_joints.py                #   控制器冒烟测试 (不经 MoveIt)
robomaster_ep_moveit_config/          # MoveIt 配置包
├── config/robomaster_ep.srdf         #   规划组/末端执行器/自碰撞矩阵/home 位姿
├── config/{kinematics,joint_limits,ompl_planning,moveit_controllers}.yaml
├── launch/move_group.launch.py       #   move_group 规划器
├── launch/moveit_rviz.launch.py      #   RViz 可视化
└── rviz/moveit.rviz                  #   RViz 界面配置
```

## 环境要求

- Ubuntu 22.04 + ROS 2 Humble
- Ignition Gazebo Fortress (`ign gazebo`), `ros_gz_sim`
- MoveIt 2 (`ros-humble-moveit`), 任务节点直接用 MoveIt 原生接口
  (/move_action Action、/compute_fk、/compute_cartesian_path、/execute_trajectory),
  不依赖 moveit_commander
- `ros2_control` + `ros2_controllers` + `gz_ros2_control` (源码或 apt 均可)

```bash
sudo apt install ros-humble-moveit ros-humble-ros-gz ros-humble-ros2-control \
                 ros-humble-ros2-controllers ros-humble-controller-manager
# gz_ros2_control 若未安装可源码编译到同一 workspace
```

## 构建

```bash
cd ~/colcon_ws/src
git clone https://github.com/X-u-e-B-a-o/robomaster-pick-place-sim.git
cd ~/colcon_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## 一键启动(验收用)

```bash
source /opt/ros/humble/setup.bash
source ~/colcon_ws/install/setup.bash
export GZ_VERSION=fortress
ros2 launch robomaster_pick_place_sim pick_place_moveit.launch.py
```

启动后会自动: 打开 Gazebo 世界 → 生成机械臂 → 激活 3 个控制器 →
启动 move_group → 启动抓取节点, 连续执行 5 次 A→B 抓放, 最后输出验收结果。

需要 RViz 时加参数: `ros2 launch robomaster_pick_place_sim pick_place_moveit.launch.py use_rviz:=true`

## 分步调试流程

### 第 1 步: 控制器冒烟测试

```bash
ros2 launch robomaster_pick_place_sim pick_place_gazebo.launch.py   # 终端 1
ros2 run robomaster_pick_place_sim test_joints                      # 终端 2
```

手臂应依次完成: 回零 → 左右偏航 → 抬臂 → 夹爪开合 → 回零。
确认 `ros2 control list_controllers` 三个控制器都是 `active`。

### 第 2 步: MoveIt 验证

```bash
ros2 launch robomaster_ep_moveit_config moveit_rviz.launch.py       # RViz 中应看到模型
```

(在已启动 Gazebo 仿真后) 另开终端:

```bash
ros2 launch robomaster_ep_moveit_config move_group.launch.py
```

RViz 里 MotionPlanning 面板拖动目标球, 点 Plan/Execute 应能规划执行。
不拖动时勾选 Planning 后直接 Plan 也行, 只要能出轨迹说明 MoveIt 通。

### 第 3 步: 标定 A/B 点

```bash
ros2 run robomaster_pick_place_sim calibrate_pose
```

输入 `x y z` 回车, 机械臂移动到该位置并打印实际末端位姿、手指方向和关节角。
依次探测取物点、放置点的坐标, 确认:

- 打印出的手指方向约等于从预抓取点指向物体的方向(用于直线接近);
- 关节角离限位有足够余量;
- 报"不可达/无逆解"的位置不能用(会带 error code)。

标定完成后把坐标写入 `config/pick_place_params.yaml` 的
`pick_point` / `place_point`, 重新 `colcon build --packages-select robomaster_pick_place_sim`。
也可不改文件, 直接 `params_file:=/path/to/你的.yaml` 覆盖。

### 第 4 步: 一键抓取 + 验收

回到"一键启动", 观察 Gazebo 中物体从红色 A 点被搬到绿色 B 点 5 次。

## 工作空间说明(调参必读)

机械臂是 2 个俯仰关节 + 底盘偏航, 末端可到达的范围是以
(0, 0, 0.16) 为中心的一个"甜甜圈"环带:

- 末端离肩中心太近(< 约 0.37 m)或太远(> 0.50 m)都不可达;
- 加上偏航后, 同一高度上可达半径约为 0.15 ~ 0.68 m;
- A、B 两点都必须在桌面范围内且离桌面边缘留 2 cm 以上余量。

当前世界参数:

| 元素 | 数值 |
| --- | --- |
| 桌面 | 0.90 × 0.70 × 0.30, 中心 (0.75, 0, 0.15), 顶面 z = 0.30 |
| 取物点 A | (0.60, 0), 红色标记垫, 目标物中心 z = 0.337 |
| 放置点 B | (0.35, 0.30), 绿色标记垫 |
| 目标物 | 7 cm 立方体, 40 g, μ = 1.5 |

改动桌面/A/B 位置时, `worlds/pick_place.sdf` 与
`config/pick_place_params.yaml`(含 `table` 碰撞场景参数)要同步修改。

## 参数说明 (pick_place_params.yaml)

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| num_cycles | 5 | 连续抓取次数(验收 5 次 ≥4 次成功) |
| pick_point / place_point | 见上表 | A/B 点坐标(物体中心) |
| gripper_open / gripper_closed | 0.040 / 0.006 m | 手指开度指令, 闭合值对应夹住 7cm 物体 |
| approach_dist | 0.12 m | 预抓取点沿手指方向距物体中心的距离 |
| lift_dist | 0.12 m | 抓取/放置后的安全抬升高度 |
| finger_center_offset | 0.06 m | 手指中心相对末端连杆沿 +X 的偏移 |
| attach_object | true | 夹紧后把物体附加到夹爪(MoveIt 规划场景) |
| max_velocity_scale / max_acceleration_scale | 0.5 | 运动速度/加速度比例 |
| abort_on_error | false | true 时任一失败即终止全部实验 |

## 输出文件 (results/)

每次运行生成三个文件(目录已加入 .gitignore, 不会误提交):

- `run_<时间>.log` — 每一步动作、状态与错误日志
- `run_<时间>_trajectory.csv` — 每一步后的关节角与末端位姿轨迹
- `run_<时间>_results.json` — 成功/失败次数、是否通过验收、每次循环记录

实验报告直接引用这些文件即可。

## 异常处理(对照验收要求)

| 异常 | 处理 |
| --- | --- |
| A/B 点不可达/无逆解 | 启动前预检查, 输出 error code 后中止任务 |
| 运动执行失败 | 急停 → 尽力回零 → 记录错误后继续下一轮(或 abort_on_error 终止) |
| 关节超限 | 每次运动后与限位表比对, 超限即报错并安全停止 |
| 用户 Ctrl+C | 急停、回零、保存结果后退出 |

## 各步骤对应 git 提交

每个开发步骤都有独立提交, 便于回溯:

1. `修复:` 控制器配置路径参数化 + spawner 加载
2. `新增:` 底盘偏航关节(工作空间从一条弧线扩展为圆环区域)
3. `新增:` 仿真世界桌面 + A/B 标记 + 目标物
4. `新增:` MoveIt 配置包
5. `新增:` 一体化 launch
6. `新增:` MoveIt 抓取节点与辅助脚本
7. 文档与 .gitignore
