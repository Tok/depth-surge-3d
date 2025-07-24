#!/bin/bash

# Simple start script for Depth Surge 3D
# Usage: ./start.sh START_TIME END_TIME
# Example: ./start.sh 1:18 1:33

INPUT_VIDEO="input_video.mp4"

# Check if input video exists
if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Error: $INPUT_VIDEO not found!"
    echo "Please place your video file as 'input_video.mp4' in this directory."
    exit 1
fi

# Check arguments
if [ $# -ne 2 ]; then
    echo "Usage: ./start.sh START_TIME END_TIME"
    echo "Example: ./start.sh 1:18 1:33"
    echo "Time format: mm:ss or hh:mm:ss"
    exit 1
fi

START_TIME=$1
END_TIME=$2

echo "Processing $INPUT_VIDEO from $START_TIME to $END_TIME..."

# Check if uv is available and working
if command -v uv &> /dev/null && uv --version &> /dev/null 2>&1; then
    echo "Using uv to run Depth Surge 3D..."
    # Test if uv run works (check for lock file issues)
    if uv run python --version &> /dev/null; then
        uv run python stereo_projector.py "$INPUT_VIDEO" -s "$START_TIME" -e "$END_TIME"
    else
        echo "uv run failed, falling back to virtual environment..."
        source .venv/bin/activate
        python stereo_projector.py "$INPUT_VIDEO" -s "$START_TIME" -e "$END_TIME"
    fi
else
    echo "Using virtual environment to run Depth Surge 3D..."
    # Try .venv first (uv style), then venv (traditional)
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "Error: No virtual environment found. Please run ./setup.sh first."
        exit 1
    fi
    
    python stereo_projector.py "$INPUT_VIDEO" -s "$START_TIME" -e "$END_TIME"
fi

echo "Processing complete!"
echo "Output video saved with audio preserved."