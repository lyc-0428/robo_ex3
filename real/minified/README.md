# 真机 V10：恢复动作，仅修正计数退出

恢复水瓶严格判定改动前的定向搜索版本。抓取、识别、移动、放置和搜索参数均采用恢复基准。

唯一任务逻辑变化：不再把抓放六次当作六个不同物体并退出。每次抓放记录为 PLACEMENT-EVENT-n，继续原有搜索，直至完整往返搜索返回 EMPTY（无可执行目标）。异常停止及手动停止保持有效。

当前没有可靠物体唯一身份或放置区视觉验收，因此仅报告动作次数，不宣称六个独立物体均放置成功。搜索无目标也不等于验证全部物体到位。

Mac 连接校园网，上传：
```bash
JETSON_HOST=adam@10.140.247.161 bash "$HOME/Desktop/real_grid_sorting_v10/install_mac.sh"
```

Jetson 连接小车 Wi-Fi，在 Jetson 本机终端直接启动：
```bash
cd /home/adam/Team21/yjh/real_grid_sorting_v10
bash ./run_grasp_v10.sh
```

启动显示 BUILD: RESTORED-MOTION-COUNT-ONLY。窗口 Q、Esc 或终端 Ctrl+C 停止。此次未运行真机测试。
