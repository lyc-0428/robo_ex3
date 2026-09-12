#!/usr/bin/env bash
set -Eeo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SORTING_SEED="${1:-${SORTING_SEED:-$(date +%s)}}"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/model/best.pt}"
SLOT_RADIUS="${SLOT_RADIUS:-0.350}"
BOTTLE_CONFIDENCE="${BOTTLE_CONFIDENCE:-0.40}"
TENNIS_CONFIDENCE="${TENNIS_CONFIDENCE:-0.50}"
YOLO_DEVICE="${YOLO_DEVICE:-0}"
VIEW_TOPIC="${VIEW_TOPIC:-/detections/image}"
CONTROLLER_READY_FILE="/tmp/team21_lyc_controller_ready_$$"
SCENE_READY_FILE="/tmp/team21_lyc_scene_ready_$$"

export TEAM21_CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
export TEAM21_CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
export TEAM21_CAMERA_FPS="${CAMERA_FPS:-30}"
export TEAM21_CAMERA_HFOV="${CAMERA_HFOV:-1.5707963267948966}"
export TEAM21_CAMERA_NEAR="${CAMERA_NEAR:-0.05}"
export TEAM21_CAMERA_FAR="${CAMERA_FAR:-5.0}"
export TEAM21_CAMERA_SENSOR_PITCH="${CAMERA_SENSOR_PITCH:--0.729922011}"
export TEAM21_CAMERA_FORWARD_OFFSET="${CAMERA_FORWARD_OFFSET:-0.05}"
export TEAM21_CAMERA_HEIGHT_OFFSET="${CAMERA_HEIGHT_OFFSET:-0.20}"

export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/home/nvidia/.Xauthority}"
export QT_X11_NO_MITSHM=1
export GZ_VERSION=fortress
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-121}"
export IGN_PARTITION="${IGN_PARTITION:-team21_lyc}"
export GZ_PARTITION="${GZ_PARTITION:-team21_lyc}"

if [[ -f "${SCRIPT_DIR}/activate_team21.sh" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/activate_team21.sh"
else
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/install/setup.bash"
fi

# ROS 2 setup scripts read a few optional variables before defining them, so
# nounset must only be enabled after the complete environment has been sourced.
set -u

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Missing YOLO weight: ${MODEL_PATH}" >&2
  exit 2
fi

for command in ros2 ign python3 pgrep setsid; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Required command not found: ${command}" >&2
    exit 2
  fi
done

# ROS Humble's cv_bridge on the Jetson was built against the NumPy 1.x ABI.
# Detect an incompatible virtual environment before starting Gazebo and the
# controller processes, rather than crashing on the first camera frame.
NUMPY_VERSION="$(python3 -c 'import numpy; print(numpy.__version__)')"
NUMPY_MAJOR="${NUMPY_VERSION%%.*}"
if [[ ! "${NUMPY_MAJOR}" =~ ^[0-9]+$ ]] || (( NUMPY_MAJOR >= 2 )); then
  echo "Incompatible NumPy ${NUMPY_VERSION}: ROS Humble cv_bridge requires NumPy 1.x." >&2
  echo "Repair once with: python3 -m pip install 'numpy<2'" >&2
  exit 2
fi

for package in ros_gz_bridge cv_bridge rqt_image_view; do
  if ! ros2 pkg prefix "${package}" >/dev/null 2>&1; then
    echo "Required ROS package not found: ${package}" >&2
    exit 2
  fi
done

if ! python3 -c 'import cv2, cv_bridge, torch, ultralytics' >/dev/null 2>&1; then
  echo "Python vision dependencies are missing in the active Team21 environment" >&2
  echo "Required imports: cv2, cv_bridge, torch, ultralytics" >&2
  exit 2
fi

mkdir -p "${SCRIPT_DIR}/logs"
declare -a CHILD_PIDS=()

cleanup_stale_runtime() {
  local pid
  local -a candidates=()
  local -a stale_pids=()
  local pattern
  pattern='ros2 launch robomaster_pick_place_sim static_model.launch.py|robot_state_publisher|ros_gz_bridge.*parameter_bridge|/parameter_bridge|ign gazebo (server|gui)|actions/yolo_detector.py|actions/grasp_bottle_tennis.py|rqt_image_view'
  mapfile -t candidates < <(pgrep -u "$(id -u)" -f "${pattern}" || true)
  for pid in "${candidates[@]}"; do
    [[ "${pid}" == "$$" || "${pid}" == "${PPID}" ]] && continue
    if tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null \
      | grep -qx "IGN_PARTITION=${IGN_PARTITION}"; then
      stale_pids+=("${pid}")
    fi
  done
  if (( ${#stale_pids[@]} == 0 )); then
    return
  fi

  echo "Stopping ${#stale_pids[@]} stale ${IGN_PARTITION} runtime processes."
  kill -INT "${stale_pids[@]}" 2>/dev/null || true
  sleep 2
  for pid in "${stale_pids[@]}"; do
    kill -0 "${pid}" 2>/dev/null && kill -TERM "${pid}" 2>/dev/null || true
  done
  sleep 1
  for pid in "${stale_pids[@]}"; do
    kill -0 "${pid}" 2>/dev/null && kill -KILL "${pid}" 2>/dev/null || true
  done
  ros2 daemon stop >/dev/null 2>&1 || true
}

cleanup() {
  local pid
  trap - EXIT INT TERM
  for pid in "${CHILD_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -INT -- "-${pid}" 2>/dev/null || kill -INT "${pid}" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "${CHILD_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  rm -f -- "${CONTROLLER_READY_FILE}" "${SCENE_READY_FILE}"
}
trap cleanup EXIT INT TERM

cleanup_stale_runtime

echo "Starting vision sorting | seed=${SORTING_SEED} | model=${MODEL_PATH}"
setsid ros2 launch robomaster_pick_place_sim static_model.launch.py &
SIM_PID=$!
CHILD_PIDS+=("${SIM_PID}")

deadline=$((SECONDS + 30))
until ign service -l 2>/dev/null | grep -q '^/world/pick_place/create$'; do
  if (( SECONDS >= deadline )); then
    echo "Gazebo world service did not appear within 30 seconds" >&2
    exit 3
  fi
  sleep 1
done

# Do not populate the scene until the robot and gz_ros2_control plugin exist.
# Mesh loading on Jetson can take considerably longer than the launch timer.
controller_deadline=$((SECONDS + 75))
robot_retry_at=$((SECONDS + 20))
robot_retry_done=false
until ros2 service list 2>/dev/null | grep -q '^/controller_manager/list_controllers$'; do
  if ! kill -0 "${SIM_PID}" 2>/dev/null; then
    echo "Gazebo launch exited before the robot became ready" >&2
    exit 3
  fi
  if [[ "${robot_retry_done}" == false ]] && (( SECONDS >= robot_retry_at )); then
    robot_retry_done=true
    echo "Controller manager is not ready; retrying robot entity creation once."
    ros2 run ros_gz_sim create \
      -world pick_place \
      -name robomaster_ep_core \
      -topic robot_description \
      -x 0 -y 0 -z 0.02 || true
  fi
  if (( SECONDS >= controller_deadline )); then
    echo "Robot/controller_manager did not become ready within 75 seconds." >&2
    echo "Inspect the ros_gz_sim create output above; task objects were not spawned." >&2
    exit 3
  fi
  sleep 1
done
echo "Robot and controller_manager are ready."

setsid ros2 run ros_gz_bridge parameter_bridge \
  '/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image' \
  '/camera/image_raw/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo' \
  '/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo' \
  '/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo' \
  --ros-args \
  -r /camera/image_raw/camera_info:=/camera/camera_info \
  -r /camera_info:=/camera/camera_info &
BRIDGE_PID=$!
CHILD_PIDS+=("${BRIDGE_PID}")

setsid python3 "${SCRIPT_DIR}/actions/yolo_detector.py" --ros-args \
  -p model_path:="${MODEL_PATH}" \
  -p bottle_confidence:="${BOTTLE_CONFIDENCE}" \
  -p tennis_confidence:="${TENNIS_CONFIDENCE}" \
  -p "device:='${YOLO_DEVICE}'" &
DETECTOR_PID=$!
CHILD_PIDS+=("${DETECTOR_PID}")

setsid python3 "${SCRIPT_DIR}/actions/grasp_bottle_tennis.py" --ros-args \
  -p log_dir:="${SCRIPT_DIR}/logs" \
  -p controller_ready_file:="${CONTROLLER_READY_FILE}" \
  -p scene_ready_file:="${SCENE_READY_FILE}" &
ACTION_PID=$!
CHILD_PIDS+=("${ACTION_PID}")

# The robot starts first. No task object is created until all three controllers
# are active and the calibrated joints are being held in continuous mode.
ready_deadline=$((SECONDS + 60))
until [[ -f "${CONTROLLER_READY_FILE}" ]]; do
  if ! kill -0 "${ACTION_PID}" 2>/dev/null; then
    wait "${ACTION_PID}" || ACTION_RC=$?
    echo "Robot controller initialization failed with exit code ${ACTION_RC:-1}." >&2
    exit "${ACTION_RC:-1}"
  fi
  if ! kill -0 "${DETECTOR_PID}" 2>/dev/null; then
    echo "YOLO detector exited during startup; inspect the model classes above" >&2
    exit 4
  fi
  if (( SECONDS >= ready_deadline )); then
    echo "Robot controllers did not enter continuous mode within 60 seconds." >&2
    exit 3
  fi
  sleep 0.2
done
echo "Robot hold and wheel controllers are active in continuous mode."

setsid ros2 run rqt_image_view rqt_image_view "${VIEW_TOPIC}" &
VIEW_PID=$!
CHILD_PIDS+=("${VIEW_PID}")

python3 "${SCRIPT_DIR}/actions/spawn_sorting_scene.py" \
  --seed "${SORTING_SEED}" \
  --model-root "${SCRIPT_DIR}/models" \
  --slot-radius "${SLOT_RADIUS}"
touch "${SCENE_READY_FILE}"

set +e
wait "${ACTION_PID}"
ACTION_RC=$?
set -e

if (( ACTION_RC != 0 )); then
  echo "Vision sorting stopped safely with exit code ${ACTION_RC}" >&2
  exit "${ACTION_RC}"
fi

echo "Sorting complete. Gazebo is paused and remains open for inspection."
echo "Press Ctrl+C in this terminal to close Gazebo and the detection window."
wait "${SIM_PID}"
