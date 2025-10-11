#!/bin/bash

# Quick test script to verify setup
echo "Testing Depth Surge 3D setup..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

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

# Test if depth_anything_v2 module loads
echo "Testing Depth Anything V2 module..."
python -c "
import sys
from pathlib import Path

# Add the depth_anything_v2_repo to path (same as depth_estimator.py does)
repo_path = Path('depth_anything_v2_repo')
if repo_path.exists():
    sys.path.insert(0, str(repo_path))

try:
    from depth_anything_v2.dpt import DepthAnythingV2
    print('✓ Depth Anything V2 module imported successfully')
except Exception as e:
    print(f'✗ Error importing Depth Anything V2: {e}')
"

# Test if model file exists
if [ -f "models/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth" ]; then
    echo "✓ Model file found"
else
    echo "✗ Model file not found at expected location"
fi

# Test if input video exists
if [ -f "input_video.mp4" ]; then
    echo "✓ Input video found"
    # Get video info
    python -c "
import cv2
cap = cv2.VideoCapture('input_video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps if fps > 0 else 0
cap.release()
print(f'  - Resolution: {width}x{height}')
print(f'  - FPS: {fps:.2f}')
print(f'  - Duration: {duration:.1f} seconds')
print(f'  - Frame count: {frame_count}')
"
else
    echo "✗ Input video not found (input_video.mp4)"
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
echo "If all items show ✓, you can run: ./start.sh START_TIME END_TIME"
echo "Example: ./start.sh 00:10 00:15"