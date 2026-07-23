<#
.SYNOPSIS
    Downloads the ONNX Runtime shared library (onnxruntime.dll) that `ort` loads
    at runtime, and places it at runtime/onnxruntime.dll in the project root.

.DESCRIPTION
    The project uses `ort` in load-dynamic mode, so ONNX Runtime is NOT statically
    linked or committed to the repo. This script fetches the exact version `ort`
    2.0.0-rc.12 targets (ONNX Runtime 1.24.2, C API v24) for Windows x64 (CPU).

    .cargo/config.toml sets ORT_DYLIB_PATH to runtime/onnxruntime.dll automatically,
    so once this script has run, `cargo test`, `cargo run`, and `cargo tauri dev`
    find the DLL with no further setup.

    Run once per clone:  pwsh scripts/fetch-onnxruntime.ps1
#>
[CmdletBinding()]
param(
    # ONNX Runtime version. Must be >= 1.24 to satisfy ort 2.0.0-rc.12 (API v24).
    [string]$Version = "1.24.2"
)

$ErrorActionPreference = "Stop"

# Resolve project root as the parent of this script's directory.
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeDir  = Join-Path $projectRoot "runtime"
$dllPath     = Join-Path $runtimeDir "onnxruntime.dll"

if (Test-Path $dllPath) {
    $existing = (Get-Item $dllPath).VersionInfo.ProductVersion
    Write-Host "onnxruntime.dll already present (version $existing) at $dllPath"
    Write-Host "Delete it and re-run to force a refresh."
    return
}

$assetName = "onnxruntime-win-x64-$Version"
$url  = "https://github.com/microsoft/onnxruntime/releases/download/v$Version/$assetName.zip"
$tmp  = Join-Path ([System.IO.Path]::GetTempPath()) "$assetName.zip"
$tmpX = Join-Path ([System.IO.Path]::GetTempPath()) $assetName

Write-Host "Downloading ONNX Runtime $Version (win-x64 CPU)..."
$ProgressPreference = "SilentlyContinue"
Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing

Write-Host "Extracting onnxruntime.dll..."
if (Test-Path $tmpX) { Remove-Item $tmpX -Recurse -Force }
Expand-Archive -Path $tmp -DestinationPath $tmpX -Force

$srcDll = Join-Path $tmpX "$assetName\lib\onnxruntime.dll"
if (-not (Test-Path $srcDll)) { throw "onnxruntime.dll not found in archive at $srcDll" }

New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
Copy-Item $srcDll -Destination $dllPath -Force

# Clean up temp files.
Remove-Item $tmp -Force -ErrorAction SilentlyContinue
Remove-Item $tmpX -Recurse -Force -ErrorAction SilentlyContinue

$ver = (Get-Item $dllPath).VersionInfo.ProductVersion
Write-Host "Done. Placed onnxruntime.dll (version $ver) at $dllPath"
Write-Host "Verify with:  cargo test -p tauri-app onnxruntime_dylib_loads -- --nocapture"
