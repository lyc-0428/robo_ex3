#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Please load this environment with: source /home/nvidia/Team21/lyc/digital/activate_team21.sh"
  exit 1
fi

export TEAM21_ROOT="/home/nvidia/Team21"
export LYC_ROOT="${TEAM21_ROOT}/lyc"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-121}"
export IGN_PARTITION="${IGN_PARTITION:-team21_lyc}"
export GZ_PARTITION="${GZ_PARTITION:-team21_lyc}"

source /opt/ros/humble/setup.bash
source /home/nvidia/ros2_ws/install/setup.bash
source "${TEAM21_ROOT}/Team21/bin/activate"
source "${LYC_ROOT}/digital/install/setup.bash"

# Do not append the shared Team21 model directory: identically named models
# from another member must never shadow this workspace's calibrated assets.
export GZ_SIM_RESOURCE_PATH="${LYC_ROOT}/digital/models"
export IGN_GAZEBO_RESOURCE_PATH="${LYC_ROOT}/digital/models"

cd "${LYC_ROOT}/digital"
