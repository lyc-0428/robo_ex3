param(
    [string]$Target = "adam@10.140.163.196",
    [string]$OutputPath = (Join-Path $PSScriptRoot "remote_workspace_diagnostic.txt")
)

$ErrorActionPreference = "Stop"

$RemoteCommand = @"
set -o pipefail
echo '=== local graphical session ==='
ls -la /tmp/.X11-unix 2>&1 || true
for authority in /home/adam/.Xauthority /run/user/1000/gdm/Xauthority; do
  if [ -e "`$authority" ]; then ls -l "`$authority"; fi
done
ps -ef | grep -E '[X]org|[X]wayland|[g]nome-session|[w]eston' || true
loginctl list-sessions 2>&1 || true

echo '=== classic and ignition commands ==='
for command_name in gazebo gzserver gzclient ign gz; do
  printf '%-10s ' "`$command_name"
  command -v "`$command_name" || true
done
gazebo --version 2>&1 | head -n 5 || true
ign gazebo --versions 2>&1 | head -n 10 || true

echo '=== package index in every workspace ==='
for workspace in \
  /home/adam/robot_pick_ws \
  /home/adam/robot_ws \
  /home/adam/ros2_ws \
  /home/adam/Team9/lh/ros2_ws \
  /home/adam/Team9/xh \
  /home/adam/Team9/yer/jetson2026/ros2_ws \
  /home/adam/Team9/yer/robot_pick_ws; do
  echo "--- `$workspace"
  find "`$workspace/install" -type f -path '*/share/ament_index/resource_index/packages/*' \
    -printf '%f\n' 2>/dev/null | \
    grep -E 'gazebo|(^|_)(gz|ign)(_|$)|ros2_control|controller_manager' | sort -u || true
done

echo '=== source package names and paths ==='
find /home/adam/robot_pick_ws /home/adam/robot_ws /home/adam/ros2_ws \
  /home/adam/Team9 -maxdepth 9 -type f -name package.xml -print 2>/dev/null | sort | \
while IFS= read -r package_file; do
  package_name=`$(sed -n 's:.*<name>\([^<]*\)</name>.*:\1:p' "`$package_file" | head -n 1)
  case "`$package_name" in
    *gazebo*|*gz*|*ign*|*ros2_control*|*controller_manager*)
      printf '%s\t%s\n' "`$package_name" "`$package_file" ;;
  esac
done

echo '=== installed relevant Debian packages ==='
dpkg -l 2>/dev/null | grep -E 'ros-humble-(gazebo|ros-gz|ros-ign|gz-|ign-|.*ros2-control)|gazebo|ignition-gazebo' || true

echo '=== apt classic Gazebo candidates ==='
apt-cache search '^ros-humble-gazebo' | sort || true

echo '=== relevant plugin libraries ==='
find /opt/ros /usr/lib /usr/local /home/adam -maxdepth 12 -type f \
  \( -name '*gazebo*ros*control*.so' -o -name '*gz*ros2*control*.so' -o \
     -name '*ign*ros2*control*.so' -o -name 'spawn_entity.py' \) \
  -print 2>/dev/null | sort
"@
$RemoteCommand = $RemoteCommand.Replace("`r`n", "`n").Trim()

Write-Host "Enter the $Target password once. Inspecting existing workspaces..."
& ssh -o BatchMode=no $Target $RemoteCommand 2>&1 | Tee-Object -FilePath $OutputPath
if ($LASTEXITCODE -ne 0) {
    throw "Remote workspace diagnosis failed"
}
Write-Host "Diagnostic saved to $OutputPath"
