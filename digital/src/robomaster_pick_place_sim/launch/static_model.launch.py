from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import math
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
    camera_defaults = {
        "__CAMERA_WIDTH__": "640",
        "__CAMERA_HEIGHT__": "480",
        "__CAMERA_FPS__": "30",
        "__CAMERA_HFOV__": "1.5707963267948966",
        "__CAMERA_NEAR__": "0.05",
        "__CAMERA_FAR__": "5.0",
        # camera_link has +1.6898531 rad pitch. The default below produces a
        # net +0.959931 rad (55 deg downward) Gazebo +X viewing direction.
        "__CAMERA_SENSOR_PITCH__": "-0.729922011",
    }
    for token, default in camera_defaults.items():
        environment_name = "TEAM21_" + token.strip("_")
        robot_xml = robot_xml.replace(
            token,
            os.environ.get(environment_name, default),
        )

    # Move only the virtual sensor origin in the robot's global X-Z plane.
    # Convert the desired forward / upward offset into camera_link coordinates;
    # the camera mesh and every calibrated robot joint remain unchanged.
    camera_mount_pitch = 1.6898531
    camera_forward_offset = float(
        os.environ.get("TEAM21_CAMERA_FORWARD_OFFSET", "0.05")
    )
    camera_height_offset = float(
        os.environ.get("TEAM21_CAMERA_HEIGHT_OFFSET", "0.20")
    )
    if not 0.0 <= camera_forward_offset <= 0.20:
        raise ValueError("CAMERA_FORWARD_OFFSET must be between 0.0 and 0.20 m")
    if not 0.0 <= camera_height_offset <= 0.40:
        raise ValueError("CAMERA_HEIGHT_OFFSET must be between 0.0 and 0.40 m")
    sensor_x = (
        0.002
        + math.cos(camera_mount_pitch) * camera_forward_offset
        - math.sin(camera_mount_pitch) * camera_height_offset
    )
    sensor_z = (
        0.004
        + math.sin(camera_mount_pitch) * camera_forward_offset
        + math.cos(camera_mount_pitch) * camera_height_offset
    )
    robot_xml = robot_xml.replace("__CAMERA_SENSOR_X__", f"{sensor_x:.9f}")
    robot_xml = robot_xml.replace("__CAMERA_SENSOR_Z__", f"{sensor_z:.9f}")

    resource_paths = [
        os.path.dirname(pkg),
        os.path.join(pkg, "meshes"),
    ]

    previous = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    if previous:
        resource_paths.append(previous)

    resource_path = os.pathsep.join(resource_paths)

    return LaunchDescription([
        # 连续动态模式。控制器初始化使用 Gazebo 的正常 update loop，
        # 不再通过暂停状态下的 multi_step 推动物理引擎。
        ExecuteProcess(
            cmd=[
                "ign", "gazebo",
                "-r",
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
            period=5.0,
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
