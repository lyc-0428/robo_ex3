"""真机 (RoboMaster EP + Jetson) 抓取实验代码包。

- pick_place_lua_params.py: 真机抓取主脚本 (含判定/报错/日志保存)
- real_pick_place_ros2_node.py: ROS2 包装节点 (发布 /real_pick_place/status)
- camera_viewer.py: 云台摄像头实时显示 (板子接显示器)
- libmedia_codec_real.py: PyAV 版 H264 解码器 (替换板子上的空壳)
- gripper_status_sweep.py: 夹爪功率-状态扫描探针
- gripper_status_probe.py: 夹爪状态观察探针
"""
