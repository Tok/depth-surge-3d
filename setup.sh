#!/bin/bash

echo "Setting up Depth Surge 3D..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if ffmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is required but not installed."
    echo "Please install ffmpeg first:"
    echo "  Ubuntu/Debian: sudo apt install ffmpeg"
    echo "  macOS: brew install ffmpeg"
    exit 1
fi

# Check if uv is available, if not provide installation instructions
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    # Check if installation was successful
    if ! command -v uv &> /dev/null; then
        echo "Failed to install uv automatically."
        echo "Please install uv manually:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo "  Then restart your shell or run: source ~/.bashrc"
        exit 1
    fi
fi

# Use uv to create virtual environment and install dependencies
echo "Creating virtual environment with uv..."
uv venv

echo "Installing Python dependencies with uv..."
uv pip install -e .

# Check if model exists
if [ ! -f "models/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth" ]; then
    echo "Warning: Depth Anything V2 model not found at expected location."
    echo "Expected: models/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth"
    echo "Please ensure the model is available before running Depth Surge 3D."
fi

echo "Setup complete!"
echo ""
echo "To use Depth Surge 3D:"
echo "1. Activate the virtual environment: source .venv/bin/activate"
echo "2. Run: python stereo_projector.py input_video.mp4"
echo "   Or use the installed command: depth-surge-3d input_video.mp4"
echo ""
echo "For help: python stereo_projector.py --help"
echo ""
echo "With uv, you can also run directly:"
echo "  uv run python stereo_projector.py input_video.mp4"