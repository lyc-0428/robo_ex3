from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg = get_package_share_directory('robomaster_pick_place_sim')
    urdf = '/home/nvidia/colcon_ws/src/robomaster_pick_place_sim/urdf/robomaster_ep_full.urdf'
    world = os.path.join(pkg, 'worlds', 'pick_place.sdf')
    robot_description = {'robot_description': ParameterValue(
        Command(['xacro ', urdf]), value_type=str),
        'use_sim_time': True}

    return LaunchDescription([
        ExecuteProcess(cmd=['ign', 'gazebo', '-v', '1', world, '--force-version', '6'], output='screen'),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[robot_description], output='screen'),
        # 等 Gazebo 起来后再把机器人从 robot_description topic 生成到世界里
        TimerAction(period=3.0, actions=[
            Node(package='ros_gz_sim', executable='create',
                 arguments=['-name', 'robomaster_ep_core', '-topic', 'robot_description',
                            '-x', '0', '-y', '0', '-z', '0'],
                 output='screen')
        ]),
        # spawner 会等 controller_manager 服务可用后再加载并激活控制器
        TimerAction(period=6.0, actions=[
            Node(package='controller_manager', executable='spawner',
                 arguments=['joint_state_broadcaster', 'arm_controller', 'gripper_controller', '--inactive'],
                 parameters=[{'use_sim_time': True}], output='screen')
        ]),
    ])
