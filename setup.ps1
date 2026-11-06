#winget install --id=astral-sh.uv -e
$env:INSTALLER_DOWNLOAD_URL='https://wheelnext.astral.sh'; 
irm https://astral.sh/uv/install.ps1 | iex

$env:UV_LINK_MODE='copy';

$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User") 
uv python install 3.12
uv python pin 3.12
uv sync


# Video-Depth-Anything model code (must match VIDEO_DEPTH_ANYTHING_REPO_DIR in constants.py)
$depthAnythingDir = "vendor\Video-Depth-Anything"

if (-Not (Test-Path -Path $depthAnythingDir -PathType Container)) {
    Write-Host "Downloading Video-Depth-Anything repository..."
    New-Item -Path "vendor" -ItemType Directory -Force | Out-Null
    git clone https://github.com/DepthAnything/Video-Depth-Anything.git $depthAnythingDir
}

$modelDir = "models\Video-Depth-Anything-Large"
if (-Not (Test-Path -Path $modelDir -PathType Container)) {
    New-Item -Path $modelDir -ItemType Directory
}

$modelFile="models/Video-Depth-Anything-Large/video_depth_anything_vitl.pth"

if (-not (Test-Path -Path $modelFile -PathType Leaf)) {
    Write-Host "Downloading Video-Depth-Anything-Large model (~1.3GB, this may take a while)..."
    Invoke-WebRequest -Uri "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth" -OutFile $modelFile
}

Write-Host ""
Write-Host "More models (optional; run without arguments to see what's installed):"
Write-Host "  .\scripts\download_models.ps1 recommended  # models used by the CLI and web UI defaults"
Write-Host "  .\scripts\download_models.ps1 all          # every Video-Depth-Anything and DA3 model"
