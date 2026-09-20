import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    moveit_config_share = get_package_share_directory('robomaster_ep_moveit_config')
    sim_share = get_package_share_directory('robomaster_pick_place_sim')
    urdf_path = os.path.join(sim_share, 'urdf', 'robomaster_ep_gazebo.urdf')
    rviz_config = os.path.join(moveit_config_share, 'rviz', 'moveit.rviz')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', urdf_path]),
                     'use_sim_time': use_sim_time}],
        output='screen',
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    return LaunchDescription([robot_state_publisher, rviz])
