# Installation Guide

## Prerequisites

- **Python 3.9, 3.10, 3.11, or 3.12** (Python 3.13+ not yet supported)
  - **Note**: Python 3.13 is not compatible due to dependency limitations (specifically `open3d` in Depth-Anything V3)
  - See [Issue #11](https://github.com/Tok/depth-surge-3d/issues/11) for tracking Python 3.13 support
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

**Python 3.9-3.12** (not 3.13+) and **Git**:
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

---

## Advanced: Python 3.13+ Support

**Status**: Python 3.13 is not officially supported yet, but advanced users can compile dependencies from source.

**Why the limitation?** The `open3d` library (dependency of Depth-Anything V3) doesn't provide pre-built wheels for Python 3.13 yet. See [Open3D Issue #7284](https://github.com/isl-org/Open3D/issues/7284) and [#7318](https://github.com/isl-org/Open3D/issues/7318) for upstream status.

### Option 1: Wait for Official Support (Recommended)

The Open3D team is actively working on Python 3.13 wheels. For most users, **we recommend using Python 3.9-3.12** until official wheels are released.

Track progress:
- [Depth Surge 3D Issue #11](https://github.com/Tok/depth-surge-3d/issues/11)
- [Open3D Python 3.13 Support](https://github.com/isl-org/Open3D/issues/7284)

### Option 2: Compile Open3D from Source (Advanced)

The latest Open3D development version supports Python 3.13 when compiled from source.

**⚠️ Warning**: This is complex and time-consuming (10-30 minutes). Only attempt if you're comfortable with:
- Command-line build tools
- Debugging compilation errors
- Managing system dependencies

**System Requirements:**
- CMake >= 3.18
- C++14 compiler (GCC 5+, Clang 7+, Visual Studio 2017+, or Xcode 8+)
- Git
- Build tools (make, ninja, or MSBuild)
- Python 3.13 development headers

**Installation Steps:**

#### Ubuntu/Debian
```bash
# Install build dependencies
sudo apt update
sudo apt install -y \
    cmake \
    build-essential \
    git \
    python3.13-dev \
    libpython3.13-dev \
    pkg-config \
    libeigen3-dev \
    libgl1-mesa-dev \
    libgomp1

# Clone Open3D repository
git clone --recursive https://github.com/isl-org/Open3D.git
cd Open3D

# Configure and build
mkdir build && cd build
cmake -DBUILD_PYTHON_MODULE=ON \
      -DPYTHON_EXECUTABLE=$(which python3.13) \
      -DCMAKE_BUILD_TYPE=Release \
      ..

# Compile (this takes 10-30 minutes)
make -j$(nproc)
make install-pip-package

# Verify installation
python3.13 -c "import open3d; print(open3d.__version__)"
```

#### macOS (with Homebrew)
```bash
# Install build dependencies
brew install cmake eigen pkg-config

# Clone and build Open3D
git clone --recursive https://github.com/isl-org/Open3D.git
cd Open3D
mkdir build && cd build

cmake -DBUILD_PYTHON_MODULE=ON \
      -DPYTHON_EXECUTABLE=$(which python3.13) \
      -DCMAKE_BUILD_TYPE=Release \
      ..

make -j$(sysctl -n hw.ncpu)
make install-pip-package
```

#### Windows (with Visual Studio)
```powershell
# Requires Visual Studio 2017+ with C++ tools
# Install CMake from https://cmake.org/download/

# Clone Open3D
git clone --recursive https://github.com/isl-org/Open3D.git
cd Open3D
mkdir build
cd build

# Configure (adjust Python path)
cmake -DBUILD_PYTHON_MODULE=ON ^
      -DPYTHON_EXECUTABLE=C:\Python313\python.exe ^
      -DCMAKE_BUILD_TYPE=Release ^
      -G "Visual Studio 17 2022" ^
      ..

# Build (takes 15-30 minutes)
cmake --build . --config Release --target install-pip-package
```

**After Compiling Open3D:**

Now install Depth Surge 3D normally:
```bash
# Clone Depth Surge 3D
git clone https://github.com/Tok/depth-surge-3d.git
cd depth-surge-3d

# Create virtual environment with Python 3.13
python3.13 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (open3d already compiled above)
pip install -r requirements.txt

# Verify
python depth_surge_3d.py --help
```

**Troubleshooting Compilation:**
- **CMake errors**: Ensure CMake >= 3.18 installed
- **Missing headers**: Install development packages for your Python version
- **Eigen3 errors**: Install libeigen3-dev (Ubuntu) or eigen (macOS)
- **Build failures**: Check [Open3D compilation docs](https://www.open3d.org/docs/latest/compilation.html)
- **CUDA support**: Add `-DBUILD_CUDA_MODULE=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` to cmake

**Still having issues?** Comment on [Issue #11](https://github.com/Tok/depth-surge-3d/issues/11) with:
- Python version (`python --version`)
- OS and version
- Error messages from compilation

### Option 3: Use Docker (Alternative)

If compilation fails, consider using Docker with Python 3.12:
```bash
# Create Dockerfile with Python 3.12
FROM python:3.12-slim
# ... rest of setup
```

---

## Additional Resources

**Official Documentation:**
- [Open3D Build from Source](https://www.open3d.org/docs/latest/compilation.html)
- [Open3D Getting Started](https://www.open3d.org/docs/latest/getting_started.html)

**Community Support:**
- [Depth Surge 3D Discussions](https://github.com/Tok/depth-surge-3d/discussions)
- [Open3D GitHub Issues](https://github.com/isl-org/Open3D/issues)
