#!/bin/bash

# Colors from CSS (exact match with templates/index.html)
LIME='\033[38;2;57;255;20m'    # #39ff14 - accent-lime
CYAN='\033[38;2;0;217;255m'    # #00d9ff - info/cyan
NC='\033[0m'                    # No Color

# Quick test script to verify setup
echo -e "${CYAN}Testing Depth Surge 3D setup...${NC}"

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
echo -e "\n${CYAN}Testing Python imports...${NC}"
python -c "
import torch
import cv2
import numpy as np
print('\033[38;2;57;255;20m✓\033[0m PyTorch version:', torch.__version__)
print('\033[38;2;57;255;20m✓\033[0m OpenCV version:', cv2.__version__)
print('\033[38;2;57;255;20m✓\033[0m NumPy version:', np.__version__)
print('\033[38;2;57;255;20m✓\033[0m CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('\033[38;2;57;255;20m✓\033[0m CUDA device:', torch.cuda.get_device_name(0))
"

# Test if video_depth_anything module loads
echo -e "\n${CYAN}Testing Video-Depth-Anything module...${NC}"
python -c "
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
"

# Test if model file exists
echo ""
if [ -f "models/Video-Depth-Anything-Large/video_depth_anything_vitl.pth" ]; then
    echo -e "${LIME}✓${NC} Video-Depth-Anything-Large model file found"
    echo "  Model size: $(du -h 'models/Video-Depth-Anything-Large/video_depth_anything_vitl.pth' | cut -f1)"
else
    echo -e "\033[38;2;255;193;7m✗\033[0m Model file not found at expected location"
    echo "  Run ./scripts/download_models.sh large to download"
fi

# Test ffmpeg
echo ""
if command -v ffmpeg &> /dev/null; then
    echo -e "${LIME}✓${NC} FFmpeg is available"
    ffmpeg -version | head -1
else
    echo -e "\033[38;2;255;0;0m✗\033[0m FFmpeg not found"
fi

echo ""
echo -e "${LIME}Setup test complete!${NC}"
echo ""
echo "Next steps:"
echo "  Web UI:  ./run_ui.sh"
echo "  CLI:     python depth_surge_3d.py input_video.mp4"
