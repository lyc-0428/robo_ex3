# 备选方案代码索引

- `collision_strategies.py`：立方体、正六棱柱和八棱柱碰撞体，以及惯性和接触参数生成。
- `layout_strategies.py`：宽弧、双排错位和双半径扇区三种六物体布局。
- `localization_strategies.py`：轮编码器加 IMU、绝对定位纠偏和回中判定。
- `heading_strategies.py`：yaw PID 与视觉伺服小角度对准。
- `pose_source_adapters.py`：Gazebo、ROS Odometry 和 RoboMaster SDK 位姿源接入示例。

这些文件用于比较方案和移植到实机；推荐仿真实现已经整合在 `../recommended_overlay`。

