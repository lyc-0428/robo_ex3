param(
    [string]$Target = "adam@10.140.246.150",
    [string]$RemoteRoot = "/home/adam/Team21/lyc/robo_ex3"
)

$ErrorActionPreference = "Stop"
$LocalRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$Files = @(
    "activate_team21.sh",
    "README_vision_sorting.md",
    "run_vision_sorting.sh",
    "run_vision_sorting_improved.sh",
    "run_tennis_placement_debug.sh",
    "actions",
    "model",
    "models",
    "src",
    "tests",
    "tools"
)

foreach ($RelativePath in $Files) {
    $LocalPath = Join-Path $LocalRoot $RelativePath
    if (-not (Test-Path -LiteralPath $LocalPath)) {
        throw "Missing deployment file: $LocalPath"
    }
}

$ArchiveName = "team21_vision_sorting_$([guid]::NewGuid().ToString('N')).tar.gz"
$ArchivePath = Join-Path ([System.IO.Path]::GetTempPath()) $ArchiveName
$RemoteArchive = "/tmp/$ArchiveName"
$SshOptions = @(
    "-o", "BatchMode=no",
    "-o", "PreferredAuthentications=password,keyboard-interactive",
    "-o", "PubkeyAuthentication=no",
    "-o", "NumberOfPasswordPrompts=1",
    "-o", "ConnectTimeout=15"
)

try {
    $TarArguments = @("-C", $LocalRoot, "-czf", $ArchivePath) + $Files
    & tar @TarArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to create deployment archive"
    }

    Write-Host "Uploading one deployment archive to $Target"
    Write-Host "Enter the remote account password when prompted."
    & scp @SshOptions -- $ArchivePath "${Target}:$RemoteArchive"
    if ($LASTEXITCODE -ne 0) {
        throw "Archive upload failed"
    }

$RemoteCommand = @"
set -e
mkdir -p '$RemoteRoot'
tar -xzf '$RemoteArchive' -C '$RemoteRoot'
cd '$RemoteRoot'
source /opt/ros/humble/setup.bash
if [ -f /home/adam/ros2_ws/install/setup.bash ]; then
  source /home/adam/ros2_ws/install/setup.bash
fi
if [ -f /home/adam/gazebo_ros2_ws/install/setup.bash ]; then
  source /home/adam/gazebo_ros2_ws/install/setup.bash
fi
python3 -m unittest discover -s tests -v
colcon build --symlink-install --packages-select robomaster_pick_place_sim
sha256sum model/best.pt
chmod +x activate_team21.sh run_vision_sorting.sh run_vision_sorting_improved.sh run_tennis_placement_debug.sh actions/*.py
"@
$RemoteCommand = $RemoteCommand.Replace("`r`n", "`n").Trim()

    Write-Host "Enter the remote account password once more for build and verification."
    & ssh @SshOptions -- $Target $RemoteCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Remote extraction, tests, or build failed"
    }
} finally {
    if (Test-Path -LiteralPath $ArchivePath) {
        Remove-Item -LiteralPath $ArchivePath -Force
    }
}

Write-Host "Deployment and build completed on $Target"
Write-Host "Placement debug: cd $RemoteRoot && bash run_tennis_placement_debug.sh 21"
Write-Host "Full experiment: cd $RemoteRoot && bash run_vision_sorting_improved.sh 21 nominal"
