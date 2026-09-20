param(
    [switch]$Apply,
    [string]$DigitalRoot = (Join-Path $PSScriptRoot "..\digital")
)

$ErrorActionPreference = "Stop"
$source = (Resolve-Path (Join-Path $PSScriptRoot "recommended_overlay\digital")).Path
$target = [System.IO.Path]::GetFullPath($DigitalRoot)
$experiment = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))

if (-not $target.StartsWith($experiment, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to write outside experiment3: $target"
}

Write-Host "Overlay source: $source"
Write-Host "Digital target: $target"
if (-not $Apply) {
    Write-Host "Dry run only. Re-run with -Apply to copy the reviewed overlay."
    exit 0
}

Copy-Item -Path (Join-Path $source "*") -Destination $target -Recurse -Force
Write-Host "Overlay copied. No git commit or push was performed."

