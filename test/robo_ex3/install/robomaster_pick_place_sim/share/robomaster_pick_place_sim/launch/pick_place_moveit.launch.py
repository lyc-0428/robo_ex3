import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('robomaster_pick_place_sim')
    moveit_pkg = get_package_share_directory('robomaster_ep_moveit_config')
    urdf = os.path.join(pkg, 'urdf', 'robomaster_ep_gazebo.urdf')
    world = os.path.join(pkg, 'worlds', 'pick_place.sdf')
    params_file = os.path.join(pkg, 'config', 'pick_place_params.yaml')

    use_rviz = LaunchConfiguration('use_rviz', default='false')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    params_file_arg = LaunchConfiguration('params_file', default=params_file)

    # 1. Gazebo 仿真世界
    gazebo = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', '-v', '1', world, '--force-version', '6'],
        output='screen')

    # 2. 机器人状态发布 (TF + robot_description)
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': ParameterValue(
                         Command(['xacro ', urdf]), value_type=str),
                     'use_sim_time': use_sim_time}],
        output='screen')

    # 3. 等 Gazebo 起来后生成机器人
    spawn = TimerAction(period=3.0, actions=[
        Node(package='ros_gz_sim', executable='create',
             arguments=['-name', 'robomaster_ep_core', '-topic', 'robot_description',
                        '-x', '0', '-y', '0', '-z', '0'],
             output='screen')])

    # 4. 加载并激活 ros2_control 控制器
    spawner = TimerAction(period=6.0, actions=[
        Node(package='controller_manager', executable='spawner',
             arguments=['joint_state_broadcaster', 'arm_controller', 'gripper_controller', '--activate'],
             parameters=[{'use_sim_time': use_sim_time}],
             output='screen')])

    # 5. MoveIt move_group (规划与轨迹执行)
    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(moveit_pkg, 'launch', 'move_group.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items())

    # 6. 定点抓取任务节点 (等控制器就绪后自动开始)
    pick_place = TimerAction(period=12.0, actions=[
        Node(package='robomaster_pick_place_sim',
             executable='pick_place_moveit',
             parameters=[{'use_sim_time': use_sim_time, 'params_file': params_file_arg}],
             output='screen')])

    # 7. 可选 RViz (use_rviz:=true 时启动)
    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(moveit_pkg, 'launch', 'moveit_rviz.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
        condition=IfCondition(use_rviz))

    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='false',
                              description='是否同时启动 RViz (需图形界面)'),
        DeclareLaunchArgument('params_file', default_value=params_file,
                              description='抓取参数文件路径, 可覆盖默认值'),
        gazebo,
        rsp,
        spawn,
        spawner,
        move_group,
        pick_place,
        rviz,
    ])
