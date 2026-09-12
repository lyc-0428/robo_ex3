param(
    [string]$Target = "nvidia@10.140.244.120",
    [string]$RemoteRoot = "/home/nvidia/Team21/lyc/digital"
)

$ErrorActionPreference = "Stop"
$LocalRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$Files = @(
    "activate_team21.sh",
    "README_vision_sorting.md",
    "run_vision_sorting.sh",
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

try {
    $TarArguments = @("-C", $LocalRoot, "-czf", $ArchivePath) + $Files
    & tar @TarArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to create deployment archive"
    }

    Write-Host "Uploading one deployment archive to $Target"
    Write-Host "Enter the nvidia account password when prompted."
    & scp -q -o BatchMode=no -- $ArchivePath "${Target}:$RemoteArchive"
    if ($LASTEXITCODE -ne 0) {
        throw "Archive upload failed"
    }

$RemoteCommand = @"
set -e
mkdir -p '$RemoteRoot'
tar -xzf '$RemoteArchive' -C '$RemoteRoot'
cd '$RemoteRoot'
source /opt/ros/humble/setup.bash
source /home/nvidia/ros2_ws/install/setup.bash
source /home/nvidia/Team21/Team21/bin/activate
python3 -m unittest discover -s tests -v
colcon build --symlink-install --packages-select robomaster_pick_place_sim
sha256sum model/best.pt
chmod +x activate_team21.sh run_vision_sorting.sh actions/*.py
"@

    Write-Host "Enter the nvidia account password once more for build and verification."
    & ssh -o BatchMode=no $Target $RemoteCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Remote extraction, tests, or build failed"
    }
} finally {
    if (Test-Path -LiteralPath $ArchivePath) {
        Remove-Item -LiteralPath $ArchivePath -Force
    }
}

Write-Host "Deployment and build completed on $Target"
Write-Host "Run: cd $RemoteRoot && bash run_vision_sorting.sh"
