param(
    [string]$Target = "adam@10.140.163.196",
    [string]$OutputPath = (Join-Path $PSScriptRoot "remote_environment_diagnostic.txt")
)

$ErrorActionPreference = "Stop"

$RemoteCommand = @"
set -o pipefail
echo '=== identity ==='
id
printf 'shell='; getent passwd "`$(id -un)" | cut -d: -f7
printf 'display=%s\n' "`${DISPLAY:-<unset>}"
printf 'xauthority=%s\n' "`${XAUTHORITY:-<unset>}"

echo '=== command locations ==='
for command_name in ros2 colcon ign gz gazebo; do
  printf '%-10s ' "`$command_name"
  command -v "`$command_name" || true
done
ign gazebo --versions 2>&1 | head -n 20 || true

echo '=== environment before overlays ==='
printf 'ROS_DISTRO=%s\n' "`${ROS_DISTRO:-<unset>}"
printf 'AMENT_PREFIX_PATH=%s\n' "`${AMENT_PREFIX_PATH:-<unset>}"
printf 'COLCON_PREFIX_PATH=%s\n' "`${COLCON_PREFIX_PATH:-<unset>}"

echo '=== setup files under adam home ==='
find /home/adam -maxdepth 8 -type f -name setup.bash -print 2>/dev/null | sort

echo '=== package markers anywhere under adam home ==='
find /home/adam -maxdepth 12 -type f \( \
  -path '*/share/ament_index/resource_index/packages/ros_gz_sim' -o \
  -path '*/share/ament_index/resource_index/packages/ros_ign_gazebo' -o \
  -path '*/share/ament_index/resource_index/packages/ros_gz_bridge' -o \
  -path '*/share/ament_index/resource_index/packages/ros_ign_bridge' -o \
  -path '*/share/ament_index/resource_index/packages/gz_ros2_control' -o \
  -path '*/share/ament_index/resource_index/packages/ign_ros2_control' \
\) -print 2>/dev/null | sort

echo '=== packages after /opt/ros/humble ==='
source /opt/ros/humble/setup.bash
ros2 pkg list | grep -E '(^ros_(gz|ign)|^(gz|ign)_ros2_control|gazebo|controller_manager|ros2_control)' | sort || true
printf 'AMENT_PREFIX_PATH=%s\n' "`${AMENT_PREFIX_PATH:-<unset>}"

echo '=== installed apt packages ==='
dpkg-query -W -f='`$`{binary:Package\}\t`$`{Version\}\n' 2>/dev/null | \
  grep -E '^(ros-humble-(ros-(gz|ign)|gz-|ign-|gazebo|.*ros2-control)|libignition|ignition-|gz-)' | sort || true

echo '=== apt candidates ==='
apt-cache search '^ros-humble-(ros-(gz|ign)|gz-|ign-)' | sort || true

echo '=== simulator/control libraries ==='
find /opt/ros /usr/lib /usr/local /home/adam -maxdepth 12 -type f \( \
  -name 'libgz_ros2_control-system.so' -o \
  -name 'libign_ros2_control-system.so' -o \
  -name 'create' \
\) -print 2>/dev/null | sort
"@
$RemoteCommand = $RemoteCommand.Replace("`r`n", "`n").Trim()

Write-Host "Enter the $Target password once. Collecting read-only environment data..."
& ssh -o BatchMode=no $Target $RemoteCommand 2>&1 | Tee-Object -FilePath $OutputPath
if ($LASTEXITCODE -ne 0) {
    throw "Remote environment diagnosis failed"
}
Write-Host "Diagnostic saved to $OutputPath"
