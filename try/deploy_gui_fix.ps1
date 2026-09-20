param(
    [string]$Target = "adam@10.140.163.196",
    [string]$RemoteRoot = "/home/adam/Team21/lyc/robo_ex3"
)

$ErrorActionPreference = "Stop"
$DigitalRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\digital")).Path
$RelativeFiles = @(
    "run_vision_sorting_improved.sh",
    "src/robomaster_pick_place_sim/launch/static_model.launch.py",
    "src/robomaster_pick_place_sim/launch/vision_sorting_improved.launch.py"
)
$ArchiveName = "robo_ex3_gui_fix_$([guid]::NewGuid().ToString('N')).tar.gz"
$ArchivePath = Join-Path ([System.IO.Path]::GetTempPath()) $ArchiveName
$RemoteArchive = "/tmp/$ArchiveName"

try {
    & tar -C $DigitalRoot -czf $ArchivePath @RelativeFiles
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to create GUI fix archive"
    }

    Write-Host "Enter the remote password to upload the GUI fix."
    & scp -q -o BatchMode=no -- $ArchivePath "${Target}:$RemoteArchive"
    if ($LASTEXITCODE -ne 0) {
        throw "GUI fix upload failed"
    }

$RemoteCommand = @"
set -e
tar -xzf '$RemoteArchive' -C '$RemoteRoot'
rm -f '$RemoteArchive'
rm -f '$RemoteRoot/src/robomaster_pick_place_sim/launch/activate_team21.sh'
rm -f '$RemoteRoot/src/robomaster_pick_place_sim/launch/run_vision_sorting_improved.sh'
source /opt/ros/humble/setup.bash
if [ -f /home/adam/ros2_ws/install/setup.bash ]; then
  source /home/adam/ros2_ws/install/setup.bash
fi
source /home/adam/gazebo_ros2_ws/install/setup.bash
cd '$RemoteRoot'
colcon build --symlink-install --packages-select robomaster_pick_place_sim
source install/setup.bash
grep -q 'headless:=false show_image:=true' run_vision_sorting_improved.sh
grep -q 'Run the Gazebo server without Qt GUI' install/robomaster_pick_place_sim/share/robomaster_pick_place_sim/launch/static_model.launch.py
echo 'GUI patch installed and verified.'
"@
    $RemoteCommand = $RemoteCommand.Replace("`r`n", "`n").Trim()

    Write-Host "Enter the remote password once more to rebuild and verify."
    & ssh -o BatchMode=no $Target $RemoteCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Remote GUI patch rebuild failed"
    }
} finally {
    if (Test-Path -LiteralPath $ArchivePath) {
        Remove-Item -LiteralPath $ArchivePath -Force
    }
}

Write-Host "GUI fix deployed. Run: cd $RemoteRoot && bash run_vision_sorting_improved.sh 21 nominal"
