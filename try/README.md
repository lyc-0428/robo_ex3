# 实验 3 改进探索工作区

本目录保存对 `experiment3` 的改进探索、推荐实现和验收材料。推荐覆盖层已于 2026-09-13 应用到本地 `digital` 工程；没有创建提交，也没有向 GitHub push。远端同步结果、基线问题、候选方案、推荐覆盖层和测试仍完整保留在这里。

## 结论

推荐组合是：任务物体使用凸多面体碰撞体，六个物体以 0.47 m 半径宽弧随机摆放，仿真底盘使用 Gazebo 世界坐标闭环控制，视觉只负责类别和目标锁定，抓取前接近、分类转向以及每轮回中都以实测 `(x, y, yaw)` 达标为停止条件。这样同时解决碰撞难夹、物体过近、回中漂移和等时转动不一致四个问题。

推荐覆盖层还补上了实验要求中原工程缺失的内容：6 个测试物体、至少成功 5 个的门槛、空网格和未识别物体场景、`vision_msgs/Detection2DArray`、显式状态机、识别 JSONL 日志以及一个完整 Launch 入口。

## 目录

- `docs/实验3改进方案与代码审计.md`：需求映射、原代码问题、四类候选方案和推荐理由。
- `docs/验收与测试清单.md`：在 Jetson 和 Gazebo 上执行的名义与异常验收步骤。
- `docs/基线测试记录.md`：同步状态与原工程测试结果。
- `recommended_overlay/digital`：已应用到本地 `digital` 目录的推荐实现副本，可用于复核或重新应用。
- `alternatives`：碰撞体、布局、定位和航向控制的其他可运行算法方案。
- `tests`：不依赖 ROS/Gazebo 的单元测试和静态合同测试。
- `sync_record.json`：远端提交与未 push 证明记录。
- `_qa/requirements`：实验要求文档的内部逐页核对材料。

## 本机验证

在 PowerShell 中运行：

```powershell
cd D:\robotics_project\experiment3\try
.\run_tests.ps1
```

当前测试覆盖布局间距、类别随机化、碰撞体 XML、质量与惯性、闭环姿态收敛、异常场景、单 Launch 构成和状态机文件。

## 查看或重新应用覆盖层

当前本地 `digital` 已应用覆盖层。默认命令仍只显示目标，不写入工程：

```powershell
.\apply_overlay.ps1
```

确认后才显式应用：

```powershell
.\apply_overlay.ps1 -Apply
```

Linux/Jetson 对应命令：

```bash
bash apply_overlay.sh
bash apply_overlay.sh /home/nvidia/Team21/lyc/digital --apply
```

应用覆盖层只复制本地文件，不会执行 `git add`、`git commit` 或 `git push`。

## Jetson 运行

覆盖并重新构建 ROS 2 包后，一个 Launch 文件启动 Gazebo、相机桥接、YOLO、场景生成、抓放控制和状态机：

```bash
cd /home/nvidia/Team21/lyc/digital
colcon build --symlink-install --packages-select robomaster_pick_place_sim
source install/setup.bash
bash run_vision_sorting_improved.sh 21 nominal
```

异常测试：

```bash
bash run_vision_sorting_improved.sh 21 unknown
bash run_vision_sorting_improved.sh 21 empty
```

也可直接运行：

```bash
ros2 launch robomaster_pick_place_sim vision_sorting_improved.launch.py \
  digital_root:=/home/nvidia/Team21/lyc/digital seed:=21 scenario:=nominal
```

当前 Windows 工作区没有 ROS 2 Humble、Ignition Gazebo Fortress 和 Jetson CUDA 环境，因此这里完成的是代码级、XML 级和纯控制器闭环验证；最终 Gazebo 动态验收需按清单在 Jetson 上完成。
