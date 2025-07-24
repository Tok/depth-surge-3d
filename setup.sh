#!/bin/bash

echo "🚀 Setting up Depth Surge 3D..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for Python 3
if ! command_exists python3; then
    echo "❌ Error: Python 3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python $required_version or later is required. Found: $python_version"
    exit 1
fi

echo "✅ Python $python_version found"

# Check for uv package manager
if command_exists uv; then
    echo "✅ uv package manager found - using for fast setup"
    use_uv=true
else
    echo "📦 uv not found - using pip (consider installing uv for faster setup: curl -LsSf https://astral.sh/uv/install.sh | sh)"
    use_uv=false
fi

# Setup virtual environment and dependencies
if [ "$use_uv" = true ]; then
    echo "📦 Setting up environment with uv..."
    uv sync
else
    echo "📦 Setting up environment with pip..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Download Depth-Anything-V2 repository if not present
if [ ! -d "depth_anything_v2_repo" ]; then
    echo "🔄 Downloading Depth-Anything-V2 repository..."
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git depth_anything_v2_repo
    if [ $? -eq 0 ]; then
        echo "✅ Depth-Anything-V2 repository downloaded successfully"
    else
        echo "❌ Failed to download Depth-Anything-V2 repository"
        exit 1
    fi
else
    echo "✅ Depth-Anything-V2 repository already exists"
fi

# Create models directory if not present
mkdir -p models

# Download Depth-Anything-V2-Large model if not present
model_path="models/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth"
if [ ! -f "$model_path" ]; then
    echo "🔄 Downloading Depth-Anything-V2-Large model (this may take a while)..."
    mkdir -p "models/Depth-Anything-V2-Large"
    
    # Try using curl first, then wget as fallback
    if command_exists curl; then
        curl -L "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth" \
             -o "$model_path" --progress-bar
    elif command_exists wget; then
        wget "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth" \
             -O "$model_path" --progress=bar
    else
        echo "❌ Error: Neither curl nor wget found. Please install one of them to download the model."
        echo "   You can manually download the model from:"
        echo "   https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
        echo "   and place it at: $model_path"
        exit 1
    fi
    
    if [ -f "$model_path" ]; then
        echo "✅ Model downloaded successfully"
        echo "📊 Model size: $(du -h "$model_path" | cut -f1)"
    else
        echo "❌ Failed to download model"
        exit 1
    fi
else
    echo "✅ Depth-Anything-V2-Large model already exists"
    echo "📊 Model size: $(du -h "$model_path" | cut -f1)"
fi

# Check if FFmpeg is available
if command_exists ffmpeg; then
    echo "✅ FFmpeg found"
else
    echo "⚠️  FFmpeg not found - install it for video processing:"
    if command_exists apt-get; then
        echo "   sudo apt-get install ffmpeg"
    elif command_exists brew; then
        echo "   brew install ffmpeg"
    elif command_exists dnf; then
        echo "   sudo dnf install ffmpeg"
    else
        echo "   Please install FFmpeg for your system"
    fi
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📖 Quick start:"
echo "   # Test with help command:"
echo "   python depth_surge_3d.py --help"
echo ""
echo "   # Quick test (if you have uv):"
echo "   uv run python depth_surge_3d.py input_video.mp4"
echo ""
echo "   # Or activate virtual environment first:"
if [ "$use_uv" = true ]; then
    echo "   # With uv: commands run directly with 'uv run'"
else
    echo "   source venv/bin/activate"
fi
echo "   python depth_surge_3d.py input_video.mp4"
echo ""
echo "   # Web UI:"
echo "   python app.py"
echo ""
echo "🔧 Available models:"
echo "   - Depth-Anything-V2-Large (335M params) - ✅ Downloaded"
echo ""
echo "💡 Tips:"
echo "   - Use --vr-resolution 16x9-720p for quick tests"
echo "   - Use --processing-mode batch for faster processing"
echo "   - Try custom resolutions: --vr-resolution custom:1920x1080"
echo ""
echo "📁 Output will be saved to ./output/ directory"