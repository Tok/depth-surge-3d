# Installation Guide

## Prerequisites

- Python 3.9+
- FFmpeg
- CUDA 13.0+ (required for GPU acceleration)
- CUDA-compatible GPU (optional, but strongly recommended for faster processing)
- Git
- curl or wget (for downloading models)

## Quick Setup (Recommended)

Clone the repository and run the setup script:

```bash
git clone https://github.com/tok/depth-surge-3d.git depth-surge-3d
cd depth-surge-3d
chmod +x setup.sh
./setup.sh
```

The setup script will automatically:
- Install uv package manager (if not present, falls back to pip)
- Create a virtual environment
- Install all Python dependencies
- Download Video-Depth-Anything repository
- Download Video-Depth-Anything-Large model (~1.3GB)
- Verify system requirements

## Model Management

The project includes flexible model management:

```bash
# Download specific models
./scripts/download_models.sh large          # Large model (best quality)
./scripts/download_models.sh small          # Small model (fastest)
./scripts/download_models.sh small base     # Multiple models
./scripts/download_models.sh all            # All models

# Check model status
./scripts/download_models.sh                # Shows current status

# Models are automatically downloaded if missing
python depth_surge_3d.py input.mp4  # Auto-downloads if needed
```

**Available Models:**
- **Small** (24.8M params) - Fast processing, lower quality
- **Base** (97.5M params) - Balanced performance and quality
- **Large** (335.3M params) - Best quality (default)

## Manual Installation

If you prefer manual setup or if the automatic setup fails:

### Step 1: System Dependencies

**FFmpeg** (required for video processing):
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS (with Homebrew)
brew install ffmpeg

# Windows (with Chocolatey)
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

**Python 3.9+** and **Git**:
```bash
# Ubuntu/Debian
sudo apt install python3 python3-venv git

# macOS (with Homebrew)
brew install python3 git

# Windows: Download from python.org and git-scm.com
```

### Step 2: Python Environment

```bash
# Clone the repository
git clone https://github.com/tok/depth-surge-3d.git depth-surge-3d
cd depth-surge-3d

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### Step 3: Download Video-Depth-Anything

**Option A: Automatic download (recommended)**
```bash
./scripts/download_models.sh large
```

**Option B: Manual download**
```bash
# Clone the repository
git clone https://github.com/DepthAnything/Video-Depth-Anything.git video_depth_anything_repo

# Create models directory
mkdir -p models/Video-Depth-Anything-Large

# Download the model (choose one):
# Large model (1.3GB) - best quality
wget https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth \
     -O models/Video-Depth-Anything-Large/video_depth_anything_vitl.pth

# OR Base model (390MB) - balanced
mkdir -p models/Video-Depth-Anything-Base
wget https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth \
     -O models/Video-Depth-Anything-Base/video_depth_anything_vitb.pth

# OR Small model (98MB) - fastest
mkdir -p models/Video-Depth-Anything-Small
wget https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth \
     -O models/Video-Depth-Anything-Small/video_depth_anything_vits.pth
```

### Step 4: Verify Installation

```bash
# Test the installation
python depth_surge_3d.py --help

# Check model availability
python depth_surge_3d.py --model-info

# List supported resolutions
python depth_surge_3d.py --list-resolutions
```

**Troubleshooting Manual Setup:**
- If `wget` is not available, use `curl -L -o <output> <url>` instead
- On Windows, use PowerShell or download files manually from the URLs
- Ensure all dependencies are in your PATH before running

## Testing Your Installation

Run `./test.sh` to verify your installation:
- ✓ Python dependencies
- ✓ CUDA availability
- ✓ Model files
- ✓ Input video
- ✓ FFmpeg
