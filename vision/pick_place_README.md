# 视觉检测联动定点抓取 —— 进度与移交文档 (2026-09-13 晚)

> 实验升级点: 物体在固定 A 点附近任意摆放 (±5cm), 视觉测出实际位置 →
> 底盘微转 + 机械臂 moveto 修正 → 按类别参数抓取 → 转向 B 点放置。
> 定位误差被视觉吸收, 比盲抓 (固定轨迹) 更稳。

## 当前进度

| 步骤 | 状态 | 说明 |
|---|---|---|
| 标定 (步骤 1) | ✅ 通过 | `calib.json` 已生成在板子 `~/Team21/vision/`, 9/9 点, **RMS 1.12mm, max 10.5mm** |
| dry-run 定位精度 (步骤 2) | ✅ 通过 | 两轮 dry-run: 首轮 5 点与卷尺对账**最大误差 8mm**; 二轮 7 点定位 (含 bottle 检测), 2 例有效带外均被正确拦截 (<15mm 判据) |
| yaw 符号 (步骤 0 前半) | ✅ 确认 | 球在 l=+100 处打印 yaw 为正 |
| 水瓶扫参 (步骤 3) | ⬜ **下一步** | `gripper_status_sweep.py` 两遍, 判读后填 `GRASP_PARAMS["bottle"]` |
| 单轮真抓联调 (步骤 4) | ⬜ | bottle/tennis_ball 各 1 次; **首次真抓顺带确认底盘转向方向** (见下) |
| 5 轮验收 (步骤 5) | ⬜ | ≥4/5 成功 (与盲抓验收同标) |

## 接手前必读: 这台机器的三个硬件事实

1. **相机装在机械臂上, 臂动 = 视角动; 云台 pitch/yaw 指令无效** (2026-09-13
   实机确认, `--pitch -25` 完全没反应)。所以标定与定位的相机视角由
   **机械臂参考位**唯一决定:
   - 默认参考位 = `recenter()` (臂收回)。`calib.json` 里存了 `arm_pose`,
     抓取脚本每轮自动 recenter 后再定位, 与标定视角一致。
   - 可选自定义参考位 `--arm-pose "100,220"` (抬臂换视角)。**换参考位
     = 旧标定作废, 必须重标**, 且抓取时标定用哪个、抓取就用哪个
     (抓取脚本自动读 calib.json, 也可 `--arm-pose` 覆盖)。
2. **机器人热点只接受一个 SDK 连接** → 相机流与机械臂必须在**同一个进程**里
   (这是 `vision_pick_place.py` 一体化存在的原因, 不能拆 ROS2 多进程)。
   手机 RoboMaster App 必须断开机器人。
3. **有效抓取带 R_VALID_MM = (191, 266)mm**: 物体到车头前沿距离 r≈216±50;
   再近 (r<191) 相机照不到 (收起的臂挡住近处视野), 再远够不到。

## 板子环境 (Jetson Orin NX, 借的 Team21 板)

- **IP 会变 (DHCP)**: 现在 `10.140.244.120`。连不上就先 ping, 再问
  `ip addr`。板子与你的电脑必须在同一网段。
- venv: `~/Team21/Team21/bin/python3` (3.10.12, 已装 robomaster/cv2/numpy/
  tensorrt; **不要装到系统环境** — 板子是借的, 所有东西放 `~/Team21/`)。
- 仓库: `~/Team21/colcon_ws/src/robomaster_pick_place_sim` (本仓库,
  分支 `codex/jetson-dji-real-test`, push 到 ex3)。
- 日志: `~/Team21/logs/`。标定日志 `calibrate_*.txt`, 抓取日志
  `vision_pick_*.txt`, 扫参日志 `gripper_sweep_*.txt`。
- **工作流**: 在自己电脑上改代码 → scp 到板子 → 板子切到机器人热点
  (RMEP-21bdc0, `nmcli connection up RMEP-21bdc0`) 跑测试 → 切回校园网
  看日志/传文件。机器人只能连一台设备。

## 文件

| 文件 | 作用 |
|---|---|
| `vision/calibrate.py` | 网格标定工具: 逐点放球采样 → homography → `calib.json` (窗口按键: 回车采样 / r 重采 / u 撤销 / s 跳过 / c 拟合检查 / g<数字> 跳点 / q 保存退出) |
| `vision/vision_pick_place.py` | 一体化抓取主脚本 (检测 → 定位 → 转向 → 抓取 → 放 B 区), 参数见下 |
| `vision/vision_detector.py` | 纯检测节点 (已验收, FP16 引擎默认 ~30fps) |
| `real/gripper_status_sweep.py` | 夹爪功率扫参 (已验收的盲抓脚本, 直接复用于水瓶) |
| `real/pick_place_lua_params.py` | 盲抓备份 (已验收 5 轮 ≥80%, 不要动) |
| `~/Team21/vision/calib.json` | 标定产物 (板子本地生成, 不提交 git) |

## 抓取脚本用法

```bash
cd ~/Team21/colcon_ws/src/robomaster_pick_place_sim

# dry-run: 只定位不动臂 (定位精度复测用)
~/Team21/Team21/bin/python3 vision/vision_pick_place.py --dry-run

# 正式: 5 轮, 每轮类别按顺序循环
~/Team21/Team21/bin/python3 vision/vision_pick_place.py \
    --sequence bottle,tennis_ball --runs 5

# 常用参数: --target {bottle,tennis_ball,auto}  --runs N
#   --target-timeout 10 (单轮定位超时)  --no-ros  --no-display
#   --keep-yaw (每轮不把机器人转回初始朝向的协议; 默认假设转回)
```

**朝向协议 (与实验员对齐)**: 每轮抓完、机器人转向 B 区放完后,
**手动把机器人转回初始朝向** (车头朝 A 区网格) 再放下一轮物体
→ 默认模式, 无需 --keep-yaw。

`GRASP_PARAMS` (vision_pick_place.py 内) 每类一套参数:
`power` (夹持功率) / `closed_fast` (快速闭合判定阈值 s) /
`stable_time` / `timeout` / `grasp_arm_y` + 相对 move 轨迹 / 抬升放置参数。
tennis_ball 已是扫参实值; **bottle 目前占位 = 网球值, 待步骤 3 更新**。

## 下一步操作清单

### 步骤 3: 水瓶扫参 (现在就能做, 不需要相机)

```bash
cd ~/Team21/colcon_ws/src/robomaster_pick_place_sim
~/Team21/Team21/bin/python3 real/gripper_status_sweep.py bottle_empty
~/Team21/Team21/bin/python3 real/gripper_status_sweep.py bottle
```

- `bottle_empty`: 夹爪空夹, 各功率记录夹爪闭合用时
- `bottle`: 水瓶塞进夹爪中间、**用手扶住瓶身**, 同样扫一遍

判读规则:
1. 若存在功率 P: empty 遍闭合 ≤2.5s 且 bottle 遍 8s 从未闭合 →
   刚性瓶走 normal-stable 判据: 选 P, `stable_time` = empty 遍闭合时间 +0.5s;
2. 若所有功率 bottle 也闭合 → 取两遍闭合时间差最大的功率,
   `closed_fast` 取两遍中值 (与网球同型判据);
3. 完全无法区分 → 报告, 与组员讨论。

把判读结论 + 日志路径注释写进 `GRASP_PARAMS["bottle"]`。

### 步骤 4: 单轮真抓联调 (顺序建议)

1. **第一轮先放中线 l≈0** (yaw≈0, 转向无关紧要): 验证
   locate → turn → grasp → judge → 放 B 区整链, 关注爪落点。
2. **第二轮放 l=+50** (中线右侧): 看机器人转向是否朝物体 —
   **这是 YAW_SIGN 的最后确认**。若机器人朝反方向转, 改
   `vision_pick_place.py` 里 `YAW_SIGN = +1` → `-1` 重新部署。
3. bottle 与 tennis_ball 各成功 1 次后微调 bottle 几何参数
   (下降深度/夹持高度, 目前占位=网球值)。

### 步骤 5: 五轮验收

`--sequence bottle,tennis_ball,... --runs 5`, 判据 ≥4/5 成功
(失败轮: 报错灯 + safe_home + 中止)。通过后更新本 README 的进度表。

## 待观察项 (步骤 4 真抓时评估)

- **有效带下限可能偏保守**: dry-run 实测相机能看到 d≈178mm 的球 (v≈288 处),
  但 R_VALID_MM 下限 191 会拦下。若实验想支持更近的摆放, 先真抓验证
  r=186 时爪落点没问题, 再把下限放宽到 ~181 (vision_pick_place.py 一处)。

## 踩过的坑 (勿重蹈)

- **云台 pitch 无效**: 相机随臂, 用参考位协议, 别再用 gimbal 指令。
- **最近网格排 166mm 看不见**: 收起的机械臂挡住近处视野, 网格最近排
  定在 196mm (贴的标记要一致; 换网格必须重标)。
- **脚本必须整文件复制自 vision_detector.py 的关键块**: sys.path 手术
  (ROS numpy 1.x 顶掉 cv2 5.x)、RGB→BGR `np.ascontiguousarray`
  (cv2 putText 拒绝负步长)、守护线程清理 + `os._exit` (stop_video_stream
  会卡死)。
- **Python 3.10**: f-string 里不能有多行表达式 (SyntaxError)。
- **部署后先在板子上 `python3 -m py_compile` 验证**再切热点跑。
