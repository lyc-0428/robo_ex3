#!/usr/bin/env bash
set -eo pipefail

DIGITAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEED="${1:-21}"
SCENARIO="tennis_only"

# Always render on the Jetson's own desktop.  Do not use an SSH-forwarded
# DISPLAY such as localhost:10.0.  ROBO_EX3_DISPLAY may override discovery.
if [[ -n "${ROBO_EX3_DISPLAY:-}" ]]; then
  DISPLAY="${ROBO_EX3_DISPLAY}"
elif [[ -z "${DISPLAY:-}" || "${DISPLAY}" == localhost:* || "${DISPLAY}" == 127.0.0.1:* ]]; then
  DISPLAY=""
  for socket in /tmp/.X11-unix/X0 /tmp/.X11-unix/X1 /tmp/.X11-unix/X2; do
    if [[ -S "${socket}" ]]; then
      DISPLAY=":${socket##*X}"
      break
    fi
  done
fi
if [[ -z "${DISPLAY:-}" ]]; then
  echo "No local desktop display found under /tmp/.X11-unix." >&2
  echo "Log in to the adam-desktop graphical session, then run again." >&2
  exit 2
fi
export DISPLAY

if [[ -z "${XAUTHORITY:-}" ]]; then
  for authority in "/run/user/$(id -u)/gdm/Xauthority" "${HOME}/.Xauthority"; do
    if [[ -f "${authority}" ]]; then
      XAUTHORITY="${authority}"
      export XAUTHORITY
      break
    fi
  done
fi
if [[ -d "/run/user/$(id -u)" ]]; then
  export XDG_RUNTIME_DIR="/run/user/$(id -u)"
fi

echo "Using Jetson local display ${DISPLAY} (XAUTHORITY=${XAUTHORITY:-auto})."

if [[ -f "${DIGITAL_ROOT}/activate_team21.sh" ]]; then
  # shellcheck disable=SC1091
  source "${DIGITAL_ROOT}/activate_team21.sh"
else
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  # shellcheck disable=SC1091
  source "${DIGITAL_ROOT}/install/setup.bash"
fi

# Resolve the frozen tennis mesh copy before the nominal model directory.
export GZ_SIM_RESOURCE_PATH="${DIGITAL_ROOT}/models/tennis_debug:${GZ_SIM_RESOURCE_PATH:-}"
export IGN_GAZEBO_RESOURCE_PATH="${DIGITAL_ROOT}/models/tennis_debug:${IGN_GAZEBO_RESOURCE_PATH:-}"
export ROS_DOMAIN_ID="${ROBO_EX3_TENNIS_ROS_DOMAIN_ID:-122}"

for command in pgrep setsid ros2 ign; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Required command not found: ${command}" >&2
    exit 2
  fi
done

# A GUI or Gazebo server can survive if the desktop session or launch parent
# is killed by the kernel.  Remove only processes owned by this user and
# carrying this experiment's transport partition.  This prevents a new GUI
# from attaching to an empty stale server while ROS talks to another world.
cleanup_stale_runtime() {
  local pid environment
  local -a candidates=()
  local -a stale_pids=()
  local pattern
  pattern='ros2 launch robomaster_pick_place_sim vision_sorting_tennis_debug.launch.py|ign gazebo|ignition-gazebo|robot_state_publisher|ros_gz_bridge.*parameter_bridge|ros_ign_bridge.*parameter_bridge|actions/tennis_debug/yolo_detector.py|actions/tennis_debug/grasp_bottle_tennis.py|actions/tennis_debug/spawn_sorting_scene.py|rqt_image_view'
  mapfile -t candidates < <(pgrep -u "$(id -u)" -f "${pattern}" || true)
  for pid in "${candidates[@]}"; do
    [[ "${pid}" == "$$" || "${pid}" == "${PPID}" ]] && continue
    [[ -r "/proc/${pid}/environ" ]] || continue
    environment="$(tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null || true)"
    if grep -Eq '^IGN_PARTITION=team21_tennis_debug(_[0-9]+)?$' <<< "${environment}" \
      || grep -Eq '^GZ_PARTITION=team21_tennis_debug(_[0-9]+)?$' <<< "${environment}"; then
      stale_pids+=("${pid}")
    fi
  done
  if (( ${#stale_pids[@]} > 0 )); then
    echo "Stopping ${#stale_pids[@]} stale Team21 Gazebo/ROS processes."
    kill -INT "${stale_pids[@]}" 2>/dev/null || true
    sleep 2
    for pid in "${stale_pids[@]}"; do
      kill -0 "${pid}" 2>/dev/null && kill -TERM "${pid}" 2>/dev/null || true
    done
    sleep 1
    for pid in "${stale_pids[@]}"; do
      kill -0 "${pid}" 2>/dev/null && kill -KILL "${pid}" 2>/dev/null || true
    done
  fi
  rm -f -- /tmp/robo_ex3_controller_ready_[0-9]* \
    /tmp/robo_ex3_scene_ready_[0-9]*
  ros2 daemon stop >/dev/null 2>&1 || true
}

cleanup_stale_runtime

# Even if an unkillable transport process remains, this run has an isolated
# partition, so Gazebo GUI, server, create service, and pose topics cannot
# split across different worlds.
RUN_PARTITION="${ROBO_EX3_TENNIS_PARTITION:-team21_tennis_debug_$$}"
export IGN_PARTITION="${RUN_PARTITION}"
export GZ_PARTITION="${RUN_PARTITION}"

# Frozen tennis snapshot keeps the extra image window disabled by default,
# matching the requested 58/58 version and reducing Jetson load.
SHOW_IMAGE="${ROBO_EX3_SHOW_IMAGE:-false}"
if [[ "${SHOW_IMAGE}" != "true" && "${SHOW_IMAGE}" != "false" ]]; then
  echo "ROBO_EX3_SHOW_IMAGE must be true or false" >&2
  exit 2
fi
echo "Using isolated Gazebo partition ${RUN_PARTITION}; rqt image=${SHOW_IMAGE}."

LAUNCH_PID=""
cleanup_current_run() {
  local pid
  trap - EXIT INT TERM
  if [[ -n "${LAUNCH_PID}" ]] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    kill -INT -- "-${LAUNCH_PID}" 2>/dev/null \
      || kill -INT "${LAUNCH_PID}" 2>/dev/null || true
    sleep 2
    kill -TERM -- "-${LAUNCH_PID}" 2>/dev/null \
      || kill -TERM "${LAUNCH_PID}" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-${LAUNCH_PID}" 2>/dev/null \
      || kill -KILL "${LAUNCH_PID}" 2>/dev/null || true
    wait "${LAUNCH_PID}" 2>/dev/null || true
  fi
  cleanup_stale_runtime
}
handle_signal() {
  exit 130
}
trap cleanup_current_run EXIT
trap handle_signal INT TERM

setsid ros2 launch robomaster_pick_place_sim vision_sorting_tennis_debug.launch.py \
  digital_root:="${DIGITAL_ROOT}" seed:="${SEED}" scenario:="${SCENARIO}" \
  headless:=false show_image:="${SHOW_IMAGE}" &
LAUNCH_PID=$!
wait "${LAUNCH_PID}"
