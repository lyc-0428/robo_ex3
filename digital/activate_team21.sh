#!/usr/bin/env bash

DIGITAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Please load this environment with: source ${DIGITAL_ROOT}/activate_team21.sh"
  exit 1
fi

export ROBO_EX3_DIGITAL_ROOT="${ROBO_EX3_DIGITAL_ROOT:-${DIGITAL_ROOT}}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-121}"
export IGN_PARTITION="${IGN_PARTITION:-team21_lyc}"
export GZ_PARTITION="${GZ_PARTITION:-team21_lyc}"

# ROS 2 Humble setup scripts read some variables before defining them, so they
# cannot be sourced while Bash nounset is active. Preserve the caller's option.
_ROBO_EX3_RESTORE_NOUNSET=0
if [[ "$-" == *u* ]]; then
  _ROBO_EX3_RESTORE_NOUNSET=1
  set +u
fi

source /opt/ros/humble/setup.bash
if [[ -f "${HOME}/ros2_ws/install/setup.bash" ]]; then
  source "${HOME}/ros2_ws/install/setup.bash"
fi
if [[ -f "${HOME}/gazebo_ros2_ws/install/setup.bash" ]]; then
  source "${HOME}/gazebo_ros2_ws/install/setup.bash"
fi
if [[ -n "${ROBO_EX3_VENV:-}" && -f "${ROBO_EX3_VENV}/bin/activate" ]]; then
  source "${ROBO_EX3_VENV}/bin/activate"
fi
if [[ -f "${DIGITAL_ROOT}/install/setup.bash" ]]; then
  source "${DIGITAL_ROOT}/install/setup.bash"
fi

if [[ "${_ROBO_EX3_RESTORE_NOUNSET}" == "1" ]]; then
  set -u
fi
unset _ROBO_EX3_RESTORE_NOUNSET

# Do not append the shared Team21 model directory: identically named models
# from another member must never shadow this workspace's calibrated assets.
export GZ_SIM_RESOURCE_PATH="${DIGITAL_ROOT}/models"
export IGN_GAZEBO_RESOURCE_PATH="${DIGITAL_ROOT}/models"

cd "${DIGITAL_ROOT}"
