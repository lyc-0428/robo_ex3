#!/usr/bin/env bash
set -eo pipefail
LINEAR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
# Reuse the working Python / gz_ros2_control environment, not its old launch.
for activate in "${ROBO_EX3_ENV_SCRIPT:-/home/nvidia/Team21/lyc/digital/activate_team21.sh}" "${LINEAR_ROOT}/activate_team21.sh"; do
  if [[ -f "$activate" ]]; then
    source "$activate"
    break
  fi
done
set -u
export ROS_DOMAIN_ID="${ROBO_EX3_LINEAR_DOMAIN_ID:-123}"
export IGN_PARTITION="team21_linear_$$"
export GZ_PARTITION="$IGN_PARTITION"
export QT_X11_NO_MITSHM=1
if [[ -n "${ROBO_EX3_DISPLAY:-}" ]]; then
  export DISPLAY="$ROBO_EX3_DISPLAY"
elif [[ -z "${DISPLAY:-}" || "$DISPLAY" == localhost:* ]]; then
  for socket in /tmp/.X11-unix/X0 /tmp/.X11-unix/X1 /tmp/.X11-unix/X2; do
    if [[ -S "$socket" ]]; then export DISPLAY=":${socket##*X}"; break; fi
  done
fi
if [[ -z "${DISPLAY:-}" ]]; then
  echo "Jetson 没有图形桌面，请先登录 Jetson 桌面。" >&2
  exit 2
fi
if [[ -z "${XAUTHORITY:-}" ]]; then
  for authority in "/run/user/$(id -u)/gdm/Xauthority" "$HOME/.Xauthority"; do
    if [[ -f "$authority" ]]; then export XAUTHORITY="$authority"; break; fi
  done
fi
for command in ros2 ign python3 setsid; do
  command -v "$command" >/dev/null || { echo "缺少命令：$command"; exit 2; }
done
python3 -c 'import rclpy,numpy,cv2,torch,ultralytics,vision_msgs'
[[ -f "$LINEAR_ROOT/model/best.pt" ]] || { echo "缺少 model/best.pt"; exit 2; }
mkdir -p "$LINEAR_ROOT/logs"
LAUNCH_PID=""
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$LAUNCH_PID" ]] && kill -0 "$LAUNCH_PID" 2>/dev/null; then
    kill -INT -- "-$LAUNCH_PID" 2>/dev/null || true
    sleep 2
    kill -TERM -- "-$LAUNCH_PID" 2>/dev/null || true
    wait "$LAUNCH_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP
echo "运行目录：$LINEAR_ROOT"
echo "直线横移：G1-G7 黑色固定网格，G4 留空；3 网球 + 3 水瓶，验证到 6 后结束。"
set +e
setsid ros2 launch "$LINEAR_ROOT/launch/linear_sorting.launch.py" \
  digital_root:="$LINEAR_ROOT" seed:="${1:-21}" &
LAUNCH_PID=$!
wait "$LAUNCH_PID"
LAUNCH_RC=$?
set -e
if [[ -f "$LINEAR_ROOT/logs/linear_exit_code.txt" ]]; then
  read -r ACTION_RC < "$LINEAR_ROOT/logs/linear_exit_code.txt"
  echo "任务退出码：$ACTION_RC；日志：$LINEAR_ROOT/logs"
  exit "$ACTION_RC"
fi
echo "启动流程结束，未收到任务完成状态；日志：$LINEAR_ROOT/logs" >&2
if [[ "$LAUNCH_RC" == 0 ]]; then exit 1; else exit "$LAUNCH_RC"; fi
