#!/usr/bin/env bash
set -eo pipefail

DIGITAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEED="${1:-21}"
SCENARIO="${2:-nominal}"

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

exec ros2 launch robomaster_pick_place_sim vision_sorting_improved.launch.py \
  digital_root:="${DIGITAL_ROOT}" seed:="${SEED}" scenario:="${SCENARIO}" \
  headless:=false show_image:=true
