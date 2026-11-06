# Installation Guide

## Prerequisites

- **Python 3.10, 3.11, 3.12, or 3.13** (3.12 recommended)
  - The setup scripts install Python 3.12 for you via [uv](https://docs.astral.sh/uv/)
- FFmpeg
- NVIDIA driver 580+ for GPU acceleration (the default PyTorch build uses CUDA 13)
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

Depth Surge 3D doesn't ship model weights; it downloads them from Hugging Face. The setup
script fetches Video-Depth-Anything-Large. Use the download script for the others:

```bash
./scripts/download_models.sh                       # status of all models + usage
./scripts/download_models.sh recommended           # models used by the CLI and web UI defaults
./scripts/download_models.sh vda-base da3-large    # specific models
./scripts/download_models.sh vda-metric            # a group
./scripts/download_models.sh all                   # everything
# Windows: .\scripts\download_models.ps1 with the same arguments
```

Missing models are also downloaded automatically on first use.

**Available models:**

| Name | Backend | Notes | Licence |
|---|---|---|---|
| `vda-small`, `vda-small-metric` | Video-Depth-Anything (V2) | Fastest | Apache-2.0 |
| `vda-base`, `vda-base-metric` | Video-Depth-Anything (V2) | Balanced | CC BY-NC 4.0 |
| `vda-large`, `vda-large-metric` | Video-Depth-Anything (V2) | Best V2 quality (CLI default) | CC BY-NC 4.0 |
| `da3-small`, `da3-base` | Depth Anything V3 | Low VRAM | Apache-2.0 |
| `da3-large` | Depth Anything V3 (DA3MONO-LARGE) | Monocular relative depth | Apache-2.0 |
| `da3-large-metric` | Depth Anything V3 (DA3METRIC-LARGE) | Metric depth (web UI default) | Apache-2.0 |
| `da3-giant`, `da3-giant-large` | Depth Anything V3 1.1 | Highest DA3 quality, most VRAM | CC BY-NC 4.0 |

Groups: `recommended`, `vda`, `vda-metric`, `da3`. DA3 models are stored in the Hugging Face
cache (`~/.cache/huggingface`), the Video-Depth-Anything files under `models/`.

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

**Python 3.10-3.13** and **Git** (or let `uv` install Python for you):
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

# Recommended: uv installs the exact tested versions from uv.lock
uv python install 3.12
uv sync

# Alternative: plain pip
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **pip on Python 3.13:** Depth-Anything-3 pins `numpy<2`, which has no Python 3.13 wheels.
> `uv sync` lifts that pin automatically. With pip, see the note at the top of
> `requirements.txt`.

### Step 3: Download Video-Depth-Anything

**Option A: Automatic download (recommended)**
```bash
./scripts/download_models.sh recommended
```

**Option B: Manual download**
```bash
# Clone the repository
git clone https://github.com/DepthAnything/Video-Depth-Anything.git vendor/Video-Depth-Anything

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
- ✓ Python dependencies (PyTorch, OpenCV, NumPy)
- ✓ CUDA availability and GPU detection
- ✓ Video-Depth-Anything module
- ✓ Model files
- ✓ FFmpeg

For developers: Run `./scripts/run-unit-tests.sh` for full test suite (800+ tests)

---

## Python 3.13

Python 3.13 is supported since v0.10.0. `open3d` 0.20 (a Depth-Anything-3 dependency) ships
Python 3.13 wheels, and `uv sync` overrides DA3's `numpy<2` pin so numpy 2.x is used.
See [Issue #11](https://github.com/Tok/depth-surge-3d/issues/11) for the history.

Python 3.14 is untested: Depth-Anything-3 declares `requires-python <= 3.13`.

## xformers

Depth-Anything-3 installs `xformers`, whose PyPI wheels target one specific PyTorch/CUDA build.
If its kernels can't load with your PyTorch (you may see `xFormers can't load C++/CUDA
extensions`), Depth Surge 3D automatically uses PyTorch's built-in scaled-dot-product attention
instead, which uses about the same memory. You'll see `Attention backend: PyTorch SDPA` in the
log. No action is needed.

---

## Additional Resources

**Community Support:**
- [Depth Surge 3D Discussions](https://github.com/Tok/depth-surge-3d/discussions)
- [Depth Surge 3D Issues](https://github.com/Tok/depth-surge-3d/issues)
