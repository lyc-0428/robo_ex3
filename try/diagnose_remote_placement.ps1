param(
    [string]$Target = "adam@10.140.163.196",
    [string]$RemoteRoot = "/home/adam/Team21/lyc/robo_ex3",
    [string]$OutputPath = (Join-Path $PSScriptRoot "remote_placement_diagnostic.txt")
)

$ErrorActionPreference = "Stop"

$RemoteCommand = @'
set -o pipefail
root="__REMOTE_ROOT__"

echo '=== host and project ==='
date --iso-8601=seconds
hostname
printf 'project='; readlink -f "$root"

echo '=== relevant running processes ==='
ps -eo pid,lstart,args | grep -E 'grasp_bottle_tennis|vision_sorting_improved|spawn_sorting_scene' | grep -v grep || true

echo '=== deployed file metadata and hashes ==='
for path in \
  "$root/actions/grasp_bottle_tennis.py" \
  "$root/actions/pose_feedback.py" \
  "$root/actions/spawn_sorting_scene.py" \
  "$root/src/robomaster_pick_place_sim/launch/vision_sorting_improved.launch.py" \
  "$root/src/robomaster_pick_place_sim/config/vision_sorting_improved.yaml" \
  "$root/install/robomaster_pick_place_sim/share/robomaster_pick_place_sim/launch/vision_sorting_improved.launch.py" \
  "$root/install/robomaster_pick_place_sim/share/robomaster_pick_place_sim/config/vision_sorting_improved.yaml"
do
  if [ -f "$path" ]; then
    stat -c '%y %s %n' "$path"
    sha256sum "$path"
  else
    echo "MISSING $path"
  fi
done

echo '=== deployed placement implementation ==='
grep -n -E 'RETREAT BEFORE|RETREAT TO CYCLE|TURN EXACTLY|EXACT CLASSIFICATION|DRIVE .*zone|placement_distance|def _rotate_chassis' \
  "$root/actions/grasp_bottle_tennis.py" || true

echo '=== deployed pose-controller yaw branch ==='
grep -n -E 'target_yaw_error|turn_to_final_yaw|required_stable_samples|output.reached' \
  "$root/actions/pose_feedback.py" "$root/actions/grasp_bottle_tennis.py" || true

echo '=== newest controller logs ==='
find "$root/logs" -maxdepth 1 -type f -name 'continuous_grasp_*.log' \
  -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 8 || true
latest="$(find "$root/logs" -maxdepth 1 -type f -name 'continuous_grasp_*.log' \
  -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [ -n "$latest" ] && [ -f "$latest" ]; then
  echo "LATEST_LOG=$latest"
  echo '--- placement/state evidence from newest log ---'
  grep -E 'STATE|RETREAT|POSE FEEDBACK|POSE REACHED|TURN EXACTLY|EXACT CLASSIFICATION|DRIVE [0-9]|RELEASE|释放|FAILED|SAFE_STOP|ERROR|Traceback' "$latest" | tail -n 500 || true
  echo '--- final 160 lines from newest log ---'
  tail -n 160 "$latest"
else
  echo 'NO_CONTROLLER_LOG_FOUND'
fi
'@

$RemoteCommand = $RemoteCommand.Replace("__REMOTE_ROOT__", $RemoteRoot)
$RemoteCommand = $RemoteCommand.Replace("`r`n", "`n").Trim()

Write-Host "Enter the $Target password once. Collecting read-only placement evidence..."
$ErrorActionPreference = "Continue"
& ssh -o BatchMode=no -o LogLevel=ERROR $Target $RemoteCommand 2>&1 |
    Tee-Object -FilePath $OutputPath
$ErrorActionPreference = "Stop"
if ($LASTEXITCODE -ne 0) {
    throw "Remote placement diagnosis failed"
}
Write-Host "Diagnostic saved to $OutputPath"
