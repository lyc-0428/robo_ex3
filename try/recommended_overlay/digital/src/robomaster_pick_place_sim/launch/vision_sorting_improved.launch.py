"""One-launch entry point for the complete six-object simulation task."""

from pathlib import Path
import os

from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def first_available_package(*names):
    for name in names:
        try:
            get_package_share_directory(name)
            return name
        except PackageNotFoundError:
            continue
    raise RuntimeError(
        "No ROS-Gazebo bridge package is installed; install "
        "ros-humble-ros-gz-bridge (or ros-humble-ros-ign-bridge)."
    )


def generate_launch_description():
    package_share = get_package_share_directory("robomaster_pick_place_sim")
    static_launch = str(Path(package_share) / "launch" / "static_model.launch.py")
    params_file = str(
        Path(package_share) / "config" / "vision_sorting_improved.yaml"
    )

    digital_root = LaunchConfiguration("digital_root")
    model_path = LaunchConfiguration("model_path")
    seed = LaunchConfiguration("seed")
    show_image = LaunchConfiguration("show_image")
    headless = LaunchConfiguration("headless")
    scenario = LaunchConfiguration("scenario")
    bottle_confidence = LaunchConfiguration("bottle_confidence")
    tennis_confidence = LaunchConfiguration("tennis_confidence")

    # A unique pair avoids stale readiness from a previous launch.
    controller_ready = f"/tmp/robo_ex3_controller_ready_{os.getpid()}"
    scene_ready = f"/tmp/robo_ex3_scene_ready_{os.getpid()}"
    Path(controller_ready).unlink(missing_ok=True)
    Path(scene_ready).unlink(missing_ok=True)

    actions = PathJoinSubstitution([digital_root, "actions"])
    models = PathJoinSubstitution([digital_root, "models"])
    logs = PathJoinSubstitution([digital_root, "logs"])
    bridge_package = first_available_package("ros_gz_bridge", "ros_ign_bridge")

    return LaunchDescription([
        DeclareLaunchArgument(
            "digital_root",
            default_value="/home/adam/Team21/lyc/robo_ex3",
            description="Absolute path to the experiment3 digital directory",
        ),
        DeclareLaunchArgument(
            "model_path",
            default_value=PathJoinSubstitution([digital_root, "model", "best.pt"]),
        ),
        DeclareLaunchArgument("seed", default_value="21"),
        DeclareLaunchArgument(
            "headless",
            default_value="false" if os.environ.get("DISPLAY") else "true",
        ),
        DeclareLaunchArgument(
            "show_image",
            default_value="true" if os.environ.get("DISPLAY") else "false",
        ),
        DeclareLaunchArgument(
            "scenario",
            default_value="nominal",
            description="nominal, unknown, or empty",
        ),
        DeclareLaunchArgument("bottle_confidence", default_value="0.40"),
        DeclareLaunchArgument("tennis_confidence", default_value="0.50"),

        # Gazebo, robot_state_publisher, the camera sensor in the URDF, and the
        # simulated robot entity.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(static_launch),
            launch_arguments={"headless": headless}.items(),
        ),

        # Camera transport and YOLO classification.
        TimerAction(period=6.0, actions=[
            Node(
                package=bridge_package,
                executable="parameter_bridge",
                arguments=[
                    "/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image",
                    "/camera/image_raw/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
                    "/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
                    "/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
                    "--ros-args",
                    "-r", "/camera/image_raw/camera_info:=/camera/camera_info",
                    "-r", "/camera_info:=/camera/camera_info",
                ],
                output="screen",
            ),
            ExecuteProcess(
                cmd=[
                    "python3", PathJoinSubstitution([actions, "yolo_detector.py"]),
                    "--ros-args",
                    "-p", ["model_path:=", model_path],
                    "-p", ["bottle_confidence:=", bottle_confidence],
                    "-p", ["tennis_confidence:=", tennis_confidence],
                    "-p", [
                        "detection_log_path:=",
                        PathJoinSubstitution([logs, "detections.jsonl"]),
                    ],
                ],
                output="screen",
            ),
        ]),

        # Task controller and scene generator start together.  The generator
        # blocks on controller_ready, then creates six objects and signals
        # scene_ready, so no fixed startup sleep controls correctness.
        TimerAction(period=7.0, actions=[
            ExecuteProcess(
                cmd=[
                    "python3", PathJoinSubstitution([actions, "grasp_bottle_tennis.py"]),
                    "--ros-args", "--params-file", params_file,
                    "-p", ["scene_mode:=", scenario],
                    "-p", f"controller_ready_file:={controller_ready}",
                    "-p", f"scene_ready_file:={scene_ready}",
                    "-p", ["log_dir:=", logs],
                ],
                output="screen",
            ),
            ExecuteProcess(
                cmd=[
                    "python3", PathJoinSubstitution([actions, "spawn_sorting_scene.py"]),
                    "--seed", seed,
                    "--model-root", models,
                    "--slot-radius", "0.470",
                    "--slot-angles", "-40", "-24", "-8", "8", "24", "40",
                    "--minimum-separation", "0.120",
                    "--scenario", scenario,
                    "--controller-ready-file", controller_ready,
                    "--scene-ready-file", scene_ready,
                    "--ready-timeout-seconds", "90",
                ],
                output="screen",
            ),
        ]),

        TimerAction(period=10.0, actions=[
            Node(
                package="rqt_image_view",
                executable="rqt_image_view",
                arguments=["/detections/image"],
                condition=IfCondition(show_image),
                output="screen",
            ),
        ]),
    ])
