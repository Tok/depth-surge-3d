#winget install --id=astral-sh.uv -e
$env:INSTALLER_DOWNLOAD_URL='https://wheelnext.astral.sh'; 
irm https://astral.sh/uv/install.ps1 | iex

$env:UV_LINK_MODE='copy';

$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User") 
uv python install 3.12
uv python pin 3.12
uv sync


$depthAnythingDir = "video_depth_anything_repo"

if (-Not (Test-Path -Path $depthAnythingDir -PathType Container)) {
    Write-Host "Downloading Video-Depth-Anything repository..."  
    git clone https://github.com/DepthAnything/Video-Depth-Anything.git video_depth_anything_repo
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