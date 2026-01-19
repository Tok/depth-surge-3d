#!/bin/bash

# Quick test script to verify setup
echo "Testing Depth Surge 3D setup..."

# Check if virtual environment exists and activate it
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

# Test basic imports
echo "Testing Python imports..."
python -c "
import torch
import cv2
import numpy as np
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ OpenCV version: {cv2.__version__}')
print(f'✓ NumPy version: {np.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA device: {torch.cuda.get_device_name(0)}')
"

# Test if video_depth_anything module loads
echo "Testing Video-Depth-Anything module..."
python -c "
import sys
from pathlib import Path

# Add the vendored Video-Depth-Anything repo to path
repo_path = Path('vendor/Video-Depth-Anything')
if repo_path.exists():
    sys.path.insert(0, str(repo_path))

try:
    from video_depth_anything import VideoDepthAnything
    print('✓ Video-Depth-Anything module imported successfully')
except Exception as e:
    print(f'✗ Error importing Video-Depth-Anything: {e}')
"

# Test if model file exists
if [ -f "models/Video-Depth-Anything-Large/video_depth_anything_vitl.pth" ]; then
    echo "✓ Video-Depth-Anything-Large model file found"
    echo "  Model size: $(du -h 'models/Video-Depth-Anything-Large/video_depth_anything_vitl.pth' | cut -f1)"
else
    echo "✗ Model file not found at expected location"
    echo "  Run ./scripts/download_models.sh large to download"
fi

# Test ffmpeg
if command -v ffmpeg &> /dev/null; then
    echo "✓ FFmpeg is available"
    ffmpeg -version | head -1
else
    echo "✗ FFmpeg not found"
fi

echo ""
echo "Setup test complete!"
echo ""
echo "Next steps:"
echo "  Web UI:  ./run_ui.sh"
echo "  CLI:     python depth_surge_3d.py input_video.mp4"