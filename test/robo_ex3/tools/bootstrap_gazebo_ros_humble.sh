#!/usr/bin/env bash
set -eo pipefail

WORKSPACE="${1:-${HOME}/gazebo_ros2_ws}"
SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROS_GZ_REPOSITORY="${WORKSPACE}/src/ros_gz"
CONTROL_REPOSITORY="${WORKSPACE}/src/gz_ros2_control"
ROS_GZ_ARCHIVE="${SCRIPT_ROOT}/ros_gz-0.244.25.tar.gz"
CONTROL_ARCHIVE="${SCRIPT_ROOT}/gz_ros2_control-humble.tar.gz"

source /opt/ros/humble/setup.bash
export GZ_VERSION=fortress

mkdir -p "${WORKSPACE}/src"

printf '%s  %s\n' \
  '422ff9936a2a50e3e9157993cc241ca6560cdbe67677e889d7be0e88a56fbac5' \
  "${ROS_GZ_ARCHIVE}" | sha256sum --check
printf '%s  %s\n' \
  'e185e8d89073ef84bbdd3a04dc0b20273a47878a231ffe3c4d3230e66e48e1ac' \
  "${CONTROL_ARCHIVE}" | sha256sum --check

extract_once() {
  local archive="$1"
  local destination="$2"
  local required_file="$3"
  if [[ -f "${destination}/${required_file}" ]]; then
    echo "Using existing complete source tree ${destination}"
    return
  fi
  if [[ -e "${destination}" ]]; then
    local backup="${destination}.incomplete.$(date +%Y%m%d_%H%M%S)"
    mv "${destination}" "${backup}"
    echo "Moved incomplete source tree to ${backup}"
  fi
  mkdir -p "${destination}"
  tar -xzf "${archive}" --strip-components=1 -C "${destination}"
}

extract_once \
  "${ROS_GZ_ARCHIVE}" \
  "${ROS_GZ_REPOSITORY}" \
  "ros_gz_sim/package.xml"
extract_once \
  "${CONTROL_ARCHIVE}" \
  "${CONTROL_REPOSITORY}" \
  "gz_ros2_control/package.xml"

# These are the non-ROS build dependencies not already supplied by the
# installed Fortress development packages and ROS 2 control installation.
sudo apt-get install -y build-essential cmake libgflags-dev libignition-plugin-dev

if command -v rosdep >/dev/null 2>&1; then
  rosdep install \
    --from-paths \
      "${ROS_GZ_REPOSITORY}/ros_gz_sim" \
      "${CONTROL_REPOSITORY}/gz_ros2_control" \
    --ignore-src \
    --rosdistro humble \
    -r -y
fi

cd "${WORKSPACE}"
colcon build \
  --symlink-install \
  --packages-select ros_gz_sim gz_ros2_control \
  --cmake-args -DCMAKE_BUILD_TYPE=Release

set +u
source "${WORKSPACE}/install/setup.bash"
set -u

ros2 pkg prefix ros_gz_sim
ros2 pkg prefix gz_ros2_control
test -f "${WORKSPACE}/install/gz_ros2_control/lib/libgz_ros2_control-system.so"

echo "Gazebo ROS 2 dependencies are ready in ${WORKSPACE}."
