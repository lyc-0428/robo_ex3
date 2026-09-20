import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory("robomaster_pick_place_sim")
    default_params = os.path.join(
        package_share, "config", "continuous_grasp.yaml"
    )
    default_points = os.path.join(
        package_share, "config", "task_points.yaml"
    )
    static_launch = os.path.join(
        package_share, "launch", "static_model.launch.py"
    )

    params_file = LaunchConfiguration("params_file")
    points_file = LaunchConfiguration("points_file")
    log_dir = LaunchConfiguration("log_dir")
    speed_scale = LaunchConfiguration("speed_scale")
    start_simulator = LaunchConfiguration("start_simulator")

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="ROS 2 parameter file for the five-run grasp task",
        ),
        DeclareLaunchArgument(
            "points_file",
            default_value=default_points,
            description="Pick, five place points, and safe-height parameters",
        ),
        DeclareLaunchArgument(
            "log_dir",
            default_value="/home/nvidia/ros2_actions/logs",
            description="Directory for console and ROS 2 launch logs",
        ),
        DeclareLaunchArgument(
            "speed_scale",
            default_value="4.0",
            description="Overall motion speed multiplier from 1.0 to 5.0",
        ),
        DeclareLaunchArgument(
            "start_simulator",
            default_value="true",
            description="Start the verified paused Gazebo model in this launch",
        ),
        SetEnvironmentVariable("ROS_LOG_DIR", log_dir),
        # 与模型包原有 run_static.sh 保持一致。SSH 登录时通常没有
        # DISPLAY / XAUTHORITY；缺少这些环境变量会使 Gazebo 服务端退出，
        # ros_gz_sim create 随后只会不断等待 world 名称。
        SetEnvironmentVariable("GZ_VERSION", "fortress"),
        SetEnvironmentVariable(
            "DISPLAY",
            EnvironmentVariable("DISPLAY", default_value=":1"),
        ),
        SetEnvironmentVariable(
            "XAUTHORITY",
            EnvironmentVariable(
                "XAUTHORITY",
                default_value="/home/nvidia/.Xauthority",
            ),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(static_launch),
            condition=IfCondition(start_simulator),
        ),
        # static_model.launch.py 在 3 秒时生成机器人；再留出时间等待
        # gz_ros2_control 和 controller_manager 服务就绪。
        TimerAction(
            period=8.0,
            actions=[
                Node(
                    package="robomaster_pick_place_sim",
                    executable="continuous_grasp",
                    name="grasp_cube_action",
                    parameters=[
                        params_file,
                        points_file,
                        {
                            "log_dir": log_dir,
                            "speed_scale": ParameterValue(
                                speed_scale,
                                value_type=float,
                            ),
                        },
                    ],
                    output="both",
                    emulate_tty=True,
                )
            ],
        ),
    ])
