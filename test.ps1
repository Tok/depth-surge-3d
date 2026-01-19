# Colors from CSS (exact match with templates/index.html)
# #39ff14 - accent-lime (RGB: 57, 255, 20)
# #00d9ff - info/cyan (RGB: 0, 217, 255)

function Write-Lime {
    param([string]$Text)
    Write-Host $Text -ForegroundColor ([System.ConsoleColor]::Green)
}

function Write-Cyan {
    param([string]$Text)
    Write-Host $Text -ForegroundColor ([System.ConsoleColor]::Cyan)
}

# Quick test script to verify setup
Write-Cyan "Testing Depth Surge 3D setup..."

# Check if virtual environment exists and activate it
if (Test-Path ".venv") {
    Write-Lime "Activating virtual environment (.venv)..."
    & .\.venv\Scripts\Activate.ps1
} elseif (Test-Path "venv") {
    Write-Lime "Activating virtual environment (venv)..."
    & .\venv\Scripts\Activate.ps1
} else {
    Write-Host "Error: Virtual environment not found. Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}

# Test basic imports
Write-Host ""
Write-Cyan "Testing Python imports..."
python -c @"
import torch
import cv2
import numpy as np
print('\033[38;2;57;255;20m✓\033[0m PyTorch version:', torch.__version__)
print('\033[38;2;57;255;20m✓\033[0m OpenCV version:', cv2.__version__)
print('\033[38;2;57;255;20m✓\033[0m NumPy version:', np.__version__)
print('\033[38;2;57;255;20m✓\033[0m CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('\033[38;2;57;255;20m✓\033[0m CUDA device:', torch.cuda.get_device_name(0))
"@

# Test if video_depth_anything module loads
Write-Host ""
Write-Cyan "Testing Video-Depth-Anything module..."
python -c @"
import sys
from pathlib import Path

# Add the vendored Video-Depth-Anything repo to path
repo_path = Path('vendor/Video-Depth-Anything')
if repo_path.exists():
    sys.path.insert(0, str(repo_path))

try:
    from video_depth_anything import VideoDepthAnything
    print('\033[38;2;57;255;20m✓\033[0m Video-Depth-Anything module imported successfully')
except Exception as e:
    print(f'\033[38;2;255;0;0m✗\033[0m Error importing Video-Depth-Anything: {e}')
"@

# Test if model file exists
Write-Host ""
$modelPath = "models\Video-Depth-Anything-Large\video_depth_anything_vitl.pth"
if (Test-Path $modelPath) {
    Write-Lime "✓ Video-Depth-Anything-Large model file found"
    $modelSize = (Get-Item $modelPath).Length / 1MB
    Write-Host "  Model size: $([math]::Round($modelSize, 2)) MB"
} else {
    Write-Host "✗ Model file not found at expected location" -ForegroundColor Yellow
    Write-Host "  Run .\scripts\download_models.ps1 large to download"
}

# Test ffmpeg
Write-Host ""
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Write-Lime "✓ FFmpeg is available"
    ffmpeg -version 2>$null | Select-Object -First 1
} else {
    Write-Host "✗ FFmpeg not found" -ForegroundColor Red
}

Write-Host ""
Write-Lime "Setup test complete!"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  Web UI:  .\run_ui.ps1"
Write-Host "  CLI:     python depth_surge_3d.py input_video.mp4"
