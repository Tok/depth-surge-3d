$gpu = Get-WmiObject Win32_VideoController | Select-Object -ExpandProperty Name

python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

if ($gpu -match "NVIDIA") {
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
} else {
    pip install torch torchvision
}


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