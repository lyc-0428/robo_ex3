# Vision sorting simulation

The sorting task uses the RGB sensor mounted on the existing `camera_link` and
the repository-trained `model/bottle_tennisball_best.pt` weight. The runtime
copy is `model/best.pt`.

## Jetson dependencies

The existing Team21 virtual environment must be able to import `torch`,
`ultralytics`, `cv2`, and the ROS `cv_bridge` module. ROS 2 must provide
`ros_gz_bridge` and `rqt_image_view`. On the Jetson, install missing ROS
packages with:

```bash
sudo apt update
sudo apt install ros-humble-ros-gz-bridge ros-humble-cv-bridge \
  ros-humble-rqt-image-view
```

Activate the existing environment before installing a missing Ultralytics
package. Keep NumPy below 2 for ROS Humble `cv_bridge` compatibility, and do
not replace the NVIDIA Jetson PyTorch build.

```bash
source /home/adam/Team21/lyc/robo_ex3/activate_team21.sh
python -m pip install 'numpy<2' 'ultralytics<9'
```

## Build and run

```bash
cd /home/adam/Team21/lyc/robo_ex3
source activate_team21.sh
colcon build --symlink-install --packages-select robomaster_pick_place_sim
bash run_vision_sorting_improved.sh 21 nominal
```

For the current placement-only debugging phase, use the dedicated entry
point below. It is now a frozen, self-contained copy of the earlier 58-test
snapshot: 5.4 cm tennis collision, 93% visual scale, 0.65-to-0.30 m placement
schedule, 1.20 x 0.80 m markers, and the pre-coarse-alignment turn behavior.
It spawns six tennis balls, rejects bottle-class detections, and does not
require both categories for completion:

```bash
bash run_tennis_placement_debug.sh 21
```

Its controller modules live under `actions/tennis_debug`, and it has separate
launch, parameter, world, model, log, and transport-partition resources. Future
nominal bottle changes therefore do not alter the tennis debug behavior.

The improved scripts open Gazebo and `rqt_image_view` on the Jetson's local
screen. Press Ctrl+C in the launch terminal to close them. The full six-object
experiment remains available through the explicit `nominal` command above.
The nominal run shows annotated detections by default; set
`ROBO_EX3_SHOW_IMAGE=false` only when GPU/RAM pressure requires it.

The current nominal scene uses pale-blue bottles. Bottle detections have a
`0.40` final threshold; a bottle ROI containing at least 18% pale-blue pixels
receives a bounded `+0.50` color-evidence bonus. Detection logs preserve the
raw score, pale-blue fraction, applied bonus, and adjusted score. Successful
placement distances are `0.85, 0.73, 0.61, 0.49, 0.37, 0.25 m`. Every round
returns to the one world-frame home pose captured at startup before starting
the next visual acquisition.

Gazebo starts directly in continuous dynamic mode. The action process loads
and activates the joint-state, calibrated-position-hold, and wheel controllers
through the normal controller-manager update loop; it never advances a paused
world with `multi_step`. The task objects are spawned only after the hold
and wheel controllers report ready, so their creation cannot overlap robot
initialization.

Runtime settings can be overridden through environment variables, including
`MODEL_PATH`, `SLOT_RADIUS`, `BOTTLE_CONFIDENCE`, `TENNIS_CONFIDENCE`,
`YOLO_DEVICE`, `CAMERA_WIDTH`, `CAMERA_HEIGHT`, `CAMERA_FPS`, `CAMERA_HFOV`,
`CAMERA_NEAR`, `CAMERA_FAR`, `CAMERA_SENSOR_PITCH`,
`CAMERA_FORWARD_OFFSET`, and `CAMERA_HEIGHT_OFFSET`. The default sensor pitch
is `-0.729922011`, which aims the mounted camera forward and about 55 degrees
downward at the object arc. The virtual sensor origin is moved `0.05` m
forward and `0.20` m upward by default to see over the chassis and gripper,
without changing the camera mesh or grasp calibration. Set
`VIEW_TOPIC=/camera/image_raw` to make
the automatically opened window show the unprocessed camera stream instead
of the default `/detections/image` stream.

This copy uses `ROS_DOMAIN_ID=121` and `IGN_PARTITION=team21_lyc` by default,
so another Team21 member can run a world with the same name without sharing
models, ROS topics, or Ignition services.

## Inspect the camera

The camera is created from the URDF under
`robomaster_ep_core/arm_2_link/camera_link` with sensor name
`sorting_rgb_camera`; it is not a separate top-level model. In Gazebo
Fortress, open the plugin menu, add **Image Display**, and select
`/camera/image_raw` to inspect the simulator image before YOLO processing.

The same raw image and its rate can be checked from a second Jetson terminal:

```bash
cd /home/adam/Team21/lyc/robo_ex3
source activate_team21.sh
ros2 topic hz /camera/image_raw
ros2 run rqt_image_view rqt_image_view /camera/image_raw
```

The launch terminal prints `CAMERA FRAME READY` for the first received frame.
Its `min`, `max`, and `mean` values make a genuinely black render distinct
from a YOLO frame containing no detections. Image frames begin as soon as the
dynamic Gazebo sensor and bridge are ready. Task objects are withheld until
the terminal prints `Robot hold and wheel controllers are active in continuous
mode.`
