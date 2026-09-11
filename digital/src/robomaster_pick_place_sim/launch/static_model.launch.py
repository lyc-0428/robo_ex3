from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory("robomaster_pick_place_sim")
    urdf_file = os.path.join(pkg, "urdf", "robomaster_ep_static.urdf")
    controllers = os.path.join(pkg, "config", "controllers.yaml")
    world = os.path.join(pkg, "worlds", "pick_place.sdf")

    with open(urdf_file, "r", encoding="utf-8") as stream:
        robot_xml = stream.read()

    robot_xml = robot_xml.replace(
        "__CONTROLLERS_FILE__",
        controllers,
    )

    resource_paths = [
        os.path.dirname(pkg),
        os.path.join(pkg, "meshes"),
    ]

    previous = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    if previous:
        resource_paths.append(previous)

    resource_path = os.pathsep.join(resource_paths)

    return LaunchDescription([
        # 不带 -r：Gazebo 保持暂停，避免物理引擎使模型散架。
        ExecuteProcess(
            cmd=[
                "ign", "gazebo",
                "-v", "2",
                world,
                "--force-version", "6",
            ],
            additional_env={
                "GZ_SIM_RESOURCE_PATH": resource_path,
                "IGN_GAZEBO_RESOURCE_PATH": resource_path,
            },
            output="screen",
        ),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{
                "robot_description": robot_xml,
                "use_sim_time": True,
            }],
            output="screen",
        ),

        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package="ros_gz_sim",
                    executable="create",
                    arguments=[
                        "-name", "robomaster_ep_core",
                        "-topic", "robot_description",
                        "-x", "0",
                        "-y", "0",
                        "-z", "0.02",
                    ],
                    output="screen",
                )
            ],
        ),
    ])
