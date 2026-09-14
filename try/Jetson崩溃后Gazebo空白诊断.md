# Jetson 崩溃后 Gazebo 空白诊断

## 日志结论

这次不是世界或机器人文件丢失：两次重试都出现了
`ros_gz_sim: OK creation of entity`。真正的异常证据是：

- 首次运行的 Gazebo 以 `exit code -9` 结束，说明进程被外部强制杀死；
- 后续启动提示 `Found additional publishers on /stats` 和 `/clock`；
- 动作节点能看到旧控制器名称，却收不到完整 `/joint_states`；
- GUI 的 Entity List 为空。

这表示异常退出后残留了另一个 Gazebo server。新 GUI、模型创建服务、ROS
控制器连接到了不同的 server/world，因此界面和控制数据彼此不一致。

仅凭应用日志不能断定 `SIGKILL` 是否由 OOM、GPU/桌面崩溃或人工终止造成。
可在 Jetson 上检查：

```bash
journalctl -k -b | grep -Ei 'oom|out of memory|killed process|nvrm|gpu|xid'
journalctl -k -b -1 | grep -Ei 'oom|out of memory|killed process|nvrm|gpu|xid'
```

## 启动恢复改动

`run_vision_sorting_improved.sh` 现在会：

1. 只查找当前用户、Team21 Gazebo 分区下的旧 Gazebo/ROS 进程；
2. 依次发送 INT、TERM、KILL，清除确实没有退出的残留进程；
3. 删除旧的本实验 ready 文件并刷新 ROS 2 daemon；
4. 为本次运行创建唯一分区 `team21_lyc_<PID>`；
5. 用独立进程组启动整套 launch，退出时回收整个进程组；
6. 默认不打开额外的 `rqt_image_view`，减少 Jetson GPU/RAM 压力，但 Gazebo
   GUI 仍在 Jetson 本地屏幕显示。

普通调试：

```bash
bash run_tennis_placement_debug.sh 21
```

确实需要额外检测画面窗口时：

```bash
ROBO_EX3_SHOW_IMAGE=true bash run_tennis_placement_debug.sh 21
```

正常启动开头应看到：

```text
Stopping ... stale Team21 Gazebo/ROS processes.
Using isolated Gazebo partition team21_lyc_...; rqt image=false.
```

不应再看到 `Found additional publishers on /clock`。
