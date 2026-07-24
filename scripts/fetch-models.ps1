<#
.SYNOPSIS
    Downloads the face-detection and face-landmark ONNX models into models/.

.DESCRIPTION
    Both models are commercial-safe (see docs/MODELS.md) and are fetched, not
    committed (models/ is gitignored):

      * YuNet face detector           — MIT       (OpenCV Zoo, git-LFS media URL)
      * MediaPipe FaceMesh landmarker — Apache-2.0 (Google weights, ONNX build
                                        from the PINTO model zoo, Apache-2.0)

    Run once per clone:  pwsh scripts/fetch-models.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$projectRoot = Split-Path -Parent $PSScriptRoot
$modelsDir = Join-Path $projectRoot "models"
New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null

# --- 1. YuNet detector (MIT) --------------------------------------------------
$yunet = Join-Path $modelsDir "face_detection_yunet_2023mar.onnx"
if (Test-Path $yunet) {
    Write-Host "YuNet already present."
} else {
    Write-Host "Downloading YuNet detector (MIT)..."
    # Note the media.githubusercontent.com host: OpenCV Zoo uses git-LFS, so the
    # plain raw.githubusercontent.com URL returns a 131-byte pointer, not the model.
    Invoke-WebRequest -UseBasicParsing `
        -Uri "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" `
        -OutFile $yunet
    Write-Host ("  -> {0} ({1:N0} KB)" -f (Split-Path $yunet -Leaf), ((Get-Item $yunet).Length / 1KB))
}

# --- 2. MediaPipe FaceMesh landmarker (Apache-2.0) ----------------------------
$landmarker = Join-Path $modelsDir "face_landmarker.onnx"
if (Test-Path $landmarker) {
    Write-Host "FaceMesh landmarker already present."
} else {
    Write-Host "Downloading MediaPipe FaceMesh (Apache-2.0) via PINTO model zoo..."
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "pinto_facemesh"
    if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null

    $outer = Join-Path $tmp "032_FaceMesh.tar.gz"
    Invoke-WebRequest -UseBasicParsing `
        -Uri "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/032_FaceMesh/032_FaceMesh.tar.gz" `
        -OutFile $outer
    tar -xzf $outer -C $tmp

    # The ONNX build lives in a nested per-format archive.
    $inner = Join-Path $tmp "032_FaceMesh\20_new_onnx_postprocess_N-batch\resources_post.tar.gz"
    $innerOut = Join-Path $tmp "onnx"
    New-Item -ItemType Directory -Force -Path $innerOut | Out-Null
    tar -xzf $inner -C $innerOut

    $src = Join-Path $innerOut "face_mesh_192x192.onnx"   # input [1,3,192,192], out 468*3 landmarks
    if (-not (Test-Path $src)) { throw "face_mesh_192x192.onnx not found in archive at $src" }
    Copy-Item $src -Destination $landmarker -Force
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host ("  -> {0} ({1:N0} KB)" -f (Split-Path $landmarker -Leaf), ((Get-Item $landmarker).Length / 1KB))
}

Write-Host ""
Write-Host "Models ready in $modelsDir. Validate the pipeline with a face photo:"
Write-Host '  $env:CULLING_TEST_FACE_IMAGE = "C:\path\to\face.jpg"'
Write-Host "  cargo test -p tauri-app detect_real_face -- --ignored --nocapture"
