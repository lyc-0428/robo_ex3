"""Self-contained source launch; never resolves an old installed task launch."""
import math
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def add_fixed_linear_grid(world):
    """Draw seven black, collision-free cells; G4 is intentionally empty."""
    centers = (.90, .60, .30, 0.0, -.30, -.60, -.90)
    for index, center_y in enumerate(centers, start=1):
        color = "0 0 0 1"
        model = ET.fromstring(f"""
        <model name="grid_G{index}">
          <static>true</static>
          <pose>0.28005 {center_y} 0.002 0 0 0</pose>
          <link name="border">
            <visual name="top"><pose>0 0.146 0 0 0 0</pose><geometry><box><size>0.50 0.008 0.002</size></box></geometry><material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual>
            <visual name="bottom"><pose>0 -0.146 0 0 0 0</pose><geometry><box><size>0.50 0.008 0.002</size></box></geometry><material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual>
            <visual name="left"><pose>-0.246 0 0 0 0 0</pose><geometry><box><size>0.008 0.30 0.002</size></box></geometry><material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual>
            <visual name="right"><pose>0.246 0 0 0 0 0</pose><geometry><box><size>0.008 0.30 0.002</size></box></geometry><material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual>
          </link>
        </model>
        """)
        world.append(model)


def package(*names):
    for name in names:
        try:
            get_package_share_directory(name)
            return name
        except PackageNotFoundError:
            pass
    raise RuntimeError(f"Required ROS package missing: {names}")


def launch_setup(context):
    root = Path(LaunchConfiguration("digital_root").perform(context)).resolve()
    src = root / "src/robomaster_pick_place_sim"
    actions = root / "actions"
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="robo_linear_"))
    ready = temporary / "controller.ready"
    scene = temporary / "scene.ready"
    result_file = root / "logs/linear_exit_code.txt"
    result_file.unlink(missing_ok=True)
    xml = (src / "urdf/robomaster_ep_static.urdf").read_text()
    xml = xml.replace("package://robomaster_pick_place_sim/", src.as_uri()+"/")
    xml = xml.replace("__CONTROLLERS_FILE__", str(src / "config/controllers.yaml"))
    camera = {"WIDTH":"640", "HEIGHT":"480", "FPS":"30", "HFOV":"1.5707963267948966",
              "NEAR":"0.05", "FAR":"5.0", "SENSOR_PITCH":"-0.729922011"}
    for key, value in camera.items():
        xml = xml.replace(f"__CAMERA_{key}__", os.environ.get(f"TEAM21_CAMERA_{key}", value))
    pitch = 1.6898531
    forward = float(os.environ.get("TEAM21_CAMERA_FORWARD_OFFSET", ".05"))
    height = float(os.environ.get("TEAM21_CAMERA_HEIGHT_OFFSET", ".20"))
    xml = xml.replace("__CAMERA_SENSOR_X__", str(.002+math.cos(pitch)*forward-math.sin(pitch)*height))
    xml = xml.replace("__CAMERA_SENSOR_Z__", str(.004+math.sin(pitch)*forward+math.cos(pitch)*height))
    xml = xml.replace("</robot>", '''
  <gazebo>
    <plugin filename="ignition-gazebo-velocity-control-system"
            name="ignition::gazebo::systems::VelocityControl">
      <topic>/linear_sort/cmd_vel</topic>
      <initial_linear>0 0 0</initial_linear>
      <initial_angular>0 0 0</initial_angular>
    </plugin>
  </gazebo>
</robot>''')
    robot_path = temporary / "robot.urdf"
    robot_path.write_text(xml)
    world_tree = ET.parse(src / "worlds/pick_place.sdf")
    world = world_tree.getroot().find("world")
    for model in world.findall("model"):
        name = model.get("name", "")
        if name == "ground":
            for size in model.findall(".//geometry/box/size"):
                size.text = "8 8 0.02"
        elif name.endswith("zone_marker"):
            side = 1 if name.startswith("tennis") else -1
            model.find("pose").text = f"0.28005 {side*1.6} 0.001 0 0 0"
            for size in model.findall(".//geometry/box/size"):
                size.text = ".50 .90 .002"
    add_fixed_linear_grid(world)
    world_path = temporary / "linear_world.sdf"
    world_tree.write(world_path, encoding="utf-8", xml_declaration=True)
    resources = os.pathsep.join([str(root/"models/tennis_debug"), str(root/"models"),
                                str(src.parent), str(src/"meshes"),
                                os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")])
    gazebo_env = dict(IGN_GAZEBO_RESOURCE_PATH=resources, GZ_SIM_RESOURCE_PATH=resources)
    command = ["ign", "gazebo", "-r", "-v", "2", str(world_path), "--force-version", "6"]
    if LaunchConfiguration("headless").perform(context).lower() == "true":
        command.extend(["-s", "--headless-rendering"])
    gazebo = ExecuteProcess(cmd=command, additional_env=gazebo_env, output="screen")
    bridge_package = package("ros_gz_bridge", "ros_ign_bridge")
    create_package = package("ros_gz_sim", "ros_ign_gazebo")
    bridge = Node(package=bridge_package, executable="parameter_bridge", output="screen",
                  arguments=[
                      "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
                      "/linear_sort/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
                      "/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image",
                      "/camera/image_raw/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
                      "/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
                      "/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
                      "--ros-args", "-r", "/camera/image_raw/camera_info:=/camera/camera_info",
                      "-r", "/camera_info:=/camera/camera_info"])
    robot = Node(package=create_package, executable="create", output="screen",
                 arguments=["-world","pick_place","-name","robomaster_ep_core",
                            "-file",str(robot_path),"-x","0","-y",".90","-z",".02"])
    action = ExecuteProcess(output="screen", cmd=[
        "python3", str(actions/"grasp_linear_sorting.py"), "--ros-args",
        "--params-file", str(root/"config/linear_sorting.yaml"),
        "-p", f"controller_ready_file:={ready}", "-p", f"scene_ready_file:={scene}",
        "-p", f"log_dir:={logs}"])
    detector = ExecuteProcess(output="screen", cmd=[
        "python3", str(actions/"yolo_detector.py"), "--ros-args",
        "-p", f"model_path:={root/'model/best.pt'}",
        "-p", "bottle_confidence:=0.40", "-p", "tennis_confidence:=0.50",
        "-p", f"detection_log_path:={logs/'detections.jsonl'}"])
    spawner = ExecuteProcess(output="screen", cmd=[
        "python3", str(actions/"spawn_linear_scene.py"),
        "--seed", LaunchConfiguration("seed").perform(context),
        "--model-root", str(root/"models"),
        "--controller-ready-file", str(ready), "--scene-ready-file", str(scene)])

    def finished(event, _context):
        result_file.write_text(str(event.returncode)+"\n")
        return [EmitEvent(event=Shutdown(reason=f"linear action exit {event.returncode}"))]

    def failed_spawn(event, _context):
        if event.returncode:
            result_file.write_text("1\n")
            return [EmitEvent(event=Shutdown(reason="linear scene creation failed"))]
        return []

    nodes = [gazebo, Node(package="robot_state_publisher", executable="robot_state_publisher",
                           parameters=[{"robot_description":xml,"use_sim_time":True}], output="screen"),
             bridge, TimerAction(period=5.0, actions=[robot]),
             TimerAction(period=7.0, actions=[action, detector, spawner]),
             RegisterEventHandler(OnProcessExit(target_action=action, on_exit=finished)),
             RegisterEventHandler(OnProcessExit(target_action=spawner, on_exit=failed_spawn)),
             RegisterEventHandler(OnProcessExit(target_action=robot, on_exit=failed_spawn))]
    if LaunchConfiguration("show_image").perform(context).lower() == "true":
        nodes.append(TimerAction(period=10.0, actions=[Node(package="rqt_image_view", executable="rqt_image_view",
                                                        arguments=["/detections/image"], output="screen")]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("digital_root", default_value=str(Path(__file__).resolve().parents[1])),
        DeclareLaunchArgument("seed", default_value="21"),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("show_image", default_value="true"),
        OpaqueFunction(function=launch_setup),
    ])
