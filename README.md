# 实验 3：桌面物体自动分类整理

本仓库实现了 RoboMaster EP 小车对网球和水瓶的视觉识别、自动抓取与分类放置，包含 Gazebo 仿真系统和 Jetson 真机程序。

当前正式真机版本位于 `real/minified`，版本标识为：

```text
V10 RESTORED-MOTION-COUNT-ONLY
BUILD: RESTORED-MOTION-COUNT-ONLY
```

> 安全提示：真机程序会直接控制底盘、机械臂和夹爪。首次运行应架空车轮或清空运动区域，并准备随时按 `Q`、`Esc` 或 `Ctrl+C` 停止。

## 1. 实验任务

课程要求机器人自动完成“识别、判断、抓取、分类、异常处理”任务链：

- 自动识别不少于 2 类物体；
- 在 4 至 6 个固定取物位置中判断物体类别；
- 测试 6 个物体，至少正确整理 5 个；
- 正常任务中不人工指定类别、抓取位置或放置区域；
- 跳过空位置和未识别物体；
- 抓取失败、目标不可达或传感器异常时安全恢复或停止；
- 保存识别、抓取、放置和状态日志；
- 仿真由一个 Launch 文件启动完整系统。

本项目使用的类别和分类方向为：

| 模型类别 | 中文名称 | 真机分类方向 |
| --- | --- | --- |
| `tennis_ball` | 网球 | 左侧 |
| `bottle` | 水瓶 | 右侧 |

## 2. 系统组成

### 2.1 仿真系统

仿真代码位于 `digital/`，使用 ROS 2 Humble、Gazebo Fortress、`ros2_control` 和 YOLO。单个 Launch 文件会启动：

1. RoboMaster EP、机械臂、夹爪和相机模型；
2. Gazebo 与 ROS 图像桥接；
3. 六物体随机场景；
4. YOLO 检测节点；
5. 抓取与分类状态机；
6. 检测画面和运行日志。

主状态机为：

```text
INITIALIZE -> WAIT_SCENE -> ACQUIRE -> ALIGN -> APPROACH
           -> GRASP -> RETREAT -> PLACE -> VERIFY
           -> RETURN_HOME -> RECORD -> ACQUIRE / DONE
```

任一关键动作失败时进入 `SAFE_STOP`。仿真提供 `nominal`、`unknown` 和 `empty` 场景，分别用于正常六物体任务、未识别物体和空场景测试。

### 2.2 真机系统

真机程序位于 `real/minified/`，通过 RoboMaster Python SDK 直接控制 EP。V10 的主要流程如下：

```text
连接小车并启动 640 x 360 视频
  -> YOLO 同时识别网球和水瓶
  -> 在画面中央 ROI 中选择 bottom_y 最大的最近目标
  -> 观察姿态靠近并对准
  -> 切换到对应类别的标定抓取姿态
  -> 近距离闭环修正并夹取、抬升
  -> 回到运输姿态，后退 0.15 m 并恢复航向
  -> 网球向左平移，水瓶向右平移
  -> 黑色胶带持续消失后停止平移
  -> 前进 0.15 m、降低机械臂、释放物体
  -> 后退 0.15 m
  -> 从放置区反方向开始横向搜索下一个目标
  -> 完成一次往返搜索且无可执行目标后结束
```

真机视觉只保留画面中间 50% 区域，左右各 25% 被屏蔽，减少一次同时看到多个目标的概率。如果仍有多个检测框，则选择视觉上最近的目标。

黑色胶带使用灰度二值化检测，运行时会额外显示 `Black Marker Binary` 黑白窗口。当前主要参数为：

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `BLACK_GRAY_THRESHOLD` | `60` | 灰度小于阈值的像素作为黑色候选 |
| `BLACK_MIN_AREA` / `BLACK_MAX_AREA` | `250` / `6500` | 黑色标记面积范围 |
| `ROI_LEFT_RATIO` / `ROI_RIGHT_RATIO` | `0.25` / `0.75` | 中央视觉区域 |
| `STABLE_WINDOW` / `STABLE_HITS` | `5` / `3` | 稳定识别判定 |
| `CONF_BALL` | `0.35` | 网球检测阈值 |
| `CONF_BOTTLE` | `0.25` | 水瓶检测阈值 |
| `POST_GRASP_BACK_M` | `0.15 m` | 抓取后的安全后退距离 |
| `POST_DROP_BACK_M` | `0.15 m` | 放置后的安全后退距离 |

网球和水瓶的最终抓取位置分别来自 `grasp_profile_tennis_ball_arm.json` 和 `grasp_profile_bottle_arm.json`。加载时程序会把 SDK 返回的无符号坐标转换为有符号坐标。

## 3. 目录结构

```text
robo_ex3-main/
├── README.md                          # 本说明
├── ACTIVE_VERSION.md                  # 当前正式版本说明
├── bottle_tennisball_best.pt          # YOLO 权重副本
├── digital/                           # ROS 2 / Gazebo 仿真
│   ├── actions/                       # 检测、抓取、状态机和场景逻辑
│   ├── model/best.pt                  # 仿真运行时权重
│   ├── models/                        # Gazebo 物体模型
│   ├── src/robomaster_pick_place_sim/ # ROS 2 package、Launch、URDF、参数
│   ├── tests/                         # 纯 Python 静态测试
│   ├── activate_team21.sh             # 环境加载脚本
│   └── run_vision_sorting_improved.sh # 仿真主入口
├── real/minified/                     # 当前真机 V10
│   ├── vision/grasp_v10.py            # 视觉、导航、抓取和分拣主程序
│   ├── real/pick_place_lua_params.py  # 实验二夹爪与机械臂逻辑
│   ├── bottle_tennisball_best.pt      # 真机 YOLO 权重
│   ├── grasp_profile_*.json           # 两类物体的抓取标定
│   ├── run_grasp_v10.sh               # Jetson 运行脚本
│   └── install_mac.sh                 # Mac 到 Jetson 的部署脚本
├── model/                             # 模型与物体资源
├── test/                              # 历史测试快照
└── try/                               # 开发过程与调试版本
```

正式运行应优先使用 `digital/` 和 `real/minified/`。`test/`、`try/`、`new_lpx/` 等目录用于保留开发过程，不是当前入口。

仓库中的四份 `bottle_tennisball_best.pt` 内容一致，SHA-256 为：

```text
9f1e151909ce6fe1dc835cf362c8297aa20ce1686e29c54a7403f54e6ff1d603
```

## 4. 环境要求

### 4.1 仿真

- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Fortress / Ignition Gazebo
- `ros_gz_bridge` 或 `ros_ign_bridge`
- `ros2_control`、`gz_ros2_control`
- Python 3、PyTorch、Ultralytics、OpenCV、NumPy `< 2`
- Jetson 本地图形桌面，用于 Gazebo 和检测窗口

缺少常用 ROS 依赖时可安装：

```bash
sudo apt update
sudo apt install ros-humble-ros-gz-bridge ros-humble-cv-bridge \
  ros-humble-rqt-image-view
```

不要用普通 PyTorch 覆盖 Jetson 的 NVIDIA PyTorch 构建。

### 4.2 真机

- Jetson 上可用的 Python 3
- RoboMaster Python SDK
- `torch`
- `ultralytics`
- `opencv-python`
- `numpy`
- Jetson 与 RoboMaster EP 处于可通信网络

运行前可以检查依赖：

```bash
python3 -c "import cv2, numpy, torch, ultralytics, robomaster; print('dependencies OK')"
```

## 5. 运行仿真

进入 `digital` 目录并加载环境：

```bash
cd /path/to/robo_ex3-main/digital
source ./activate_team21.sh
colcon build --symlink-install --packages-select robomaster_pick_place_sim
```

运行正常六物体场景：

```bash
bash ./run_vision_sorting_improved.sh 21 nominal
```

第一个参数是随机种子。异常场景可分别运行：

```bash
bash ./run_vision_sorting_improved.sh 21 unknown
bash ./run_vision_sorting_improved.sh 21 empty
```

脚本会打开 Gazebo 和检测窗口。按 `Ctrl+C` 结束。主要输出位于：

```text
digital/logs/detections.jsonl
digital/logs/continuous_grasp_<时间戳>_<进程号>.log
```

仿真参数集中在：

```text
digital/src/robomaster_pick_place_sim/config/vision_sorting_improved.yaml
```

默认验收参数为 `total_slot_count: 6`、`minimum_success_count: 5`、每类最多 3 个。

## 6. 部署与运行真机

### 6.1 从 Mac 上传到 Jetson

在 Mac 终端进入真机目录，将 `<JETSON_IP>` 替换为 Jetson 当前 IP：

```bash
cd /path/to/robo_ex3-main/real/minified
JETSON_HOST=adam@<JETSON_IP> bash ./install_mac.sh
```

脚本会：

1. 上传 `real/minified` 中的完整 V10 文件；
2. 在 Jetson 上执行 Python 语法检查；
3. 把旧目录改名为带时间戳的备份；
4. 将新版本安装到 `/home/adam/Team21/yjh/real_grid_sorting_v10`。

### 6.2 在 Jetson 上运行

Jetson 连接 RoboMaster EP 的 Wi-Fi 后，在 Jetson 本机终端执行：

```bash
cd /home/adam/Team21/yjh/real_grid_sorting_v10
bash ./run_grasp_v10.sh
```

启动时应看到：

```text
BUILD: RESTORED-MOTION-COUNT-ONLY
```

运行脚本会先检查 `vision/grasp_v10.py` 的语法，再用以下正式参数启动：

```text
--target auto --sort-mode --max-items 6
```

真机日志自动保存在：

```text
/home/adam/Team21/yjh/real_grid_sorting_v10/logs/grasp_v10_<时间戳>.log
```

### 6.3 低风险检查

只验证识别、目标选择和靠近，不闭合夹爪或分类放置：

```bash
cd /home/adam/Team21/yjh/real_grid_sorting_v10
python3 ./vision/grasp_v10.py --target auto --sort-mode --nav-only
```

查看所有可调参数：

```bash
python3 ./vision/grasp_v10.py --help
```

## 7. 异常与停止策略

- 多目标同时出现：优先选择 `bottom_y` 最大的最近目标；
- 抓取失败：打开夹爪、回到观察姿态并后退，再继续搜索；
- 黑色标记在放置前不可见：先尝试重新找到标记，失败则不开始横移；
- 横移时标记图像过期：立即停车，不释放仍夹持的物体；
- 未确认到胶带边界：停车并保持夹持，避免在错误位置放置；
- 搜索完成一次完整往返且无目标：正常结束任务；
- 用户停止：按检测窗口中的 `Q`、`Esc`，或在终端按 `Ctrl+C`。

## 8. 测试

仓库中的仿真纯 Python 测试覆盖检测稳定性、类别身份、未知目标、空场景、夹爪阶段动作、位姿闭环、分类区域、启动同步和网球专项模式。

```bash
python3 -m unittest discover -s digital/tests -p 'test_*.py' -v
```

当前源码静态测试结果：

```text
Ran 68 tests
OK
```

这些测试不替代 Gazebo 完整运行和真机实验。

## 9. 结果口径与当前限制

- 仿真控制器按 6 个物体、至少 5 个成功进行结果判定，并记录检测和状态日志。
- 真机 V10 的 `sorted_count` 表示完成的抓放动作次数，不代表已确认的独立物体数量。
- 真机放置后没有独立的视觉验收，因此 `PLACEMENT-EVENT-n` 是动作记录，不能单独证明物体已正确落入分类区。
- 真机任务以“完整往返搜索后无可执行目标”为结束条件，`--max-items 6` 不作为强制退出条件。
- 仓库未提交运行生成的 `logs/`、演示视频或正式验收统计；最终实验成绩应以现场视频、人工核验和对应运行日志为准。
- 仿真具备单 Launch 入口；当前真机正式入口是 RoboMaster SDK 脚本，不是 ROS 2 Launch。

## 10. 维护说明

- 当前正式版本以 `ACTIVE_VERSION.md` 和 `real/minified/VERSION.txt` 为准。
- 修改真机参数前先备份可工作的 `real/minified`，并优先使用 `--nav-only` 检查视觉与方向。
- 模型、抓取 profile 和主程序应作为同一版本一起部署，避免标定与权重不匹配。
- 不要把 `test/` 或 `try/` 中的历史脚本覆盖到正式目录。

## License

本仓库采用 MIT License，详见 `LICENSE`。
