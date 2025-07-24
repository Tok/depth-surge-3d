# Depth Surge 3D

A powerful command-line tool and web application that converts 2D videos to immersive 3D VR format using advanced AI depth estimation with Depth Anything V2.

## Features

- Extract frames from video using FFmpeg
- Generate depth maps using Depth Anything V2
- Create left and right stereo images from depth information
- Output in VR-compatible formats (side-by-side or over-under)
- **Wide aspect ratio support** (16:9, 21:9, cinema formats)
- **Smart format recommendations** based on content aspect ratio
- **Dual processing modes**: Serial (frame-by-frame) and Batch (task-by-task with parallelization)
- **Low resolution options** for quick testing (480×480, 720×720)
- **Enhanced progress tracking** with step-by-step feedback
- Save intermediate results (depth maps, stereo frames)
- High-performance GPU acceleration support
- Intuitive web interface for easy processing
- Advanced fisheye projection and distortion correction

## Requirements

- Python 3.8+
- FFmpeg
- CUDA-compatible GPU (optional, for faster processing)

## Installation

### Prerequisites

- Python 3.8 or later
- CUDA-capable GPU (recommended) or CPU
- FFmpeg for video processing
- Git
- curl or wget (for downloading models)

### Quick Setup (Recommended)

Clone the repository and run the setup script:

```bash
git clone <repository-url> depth-surge-3d
cd depth-surge-3d
chmod +x setup.sh
./setup.sh
```

The setup script will automatically:
- Install uv package manager (if not present, falls back to pip)
- Create a virtual environment
- Install all Python dependencies
- Download Depth-Anything-V2 repository
- Download Depth-Anything-V2-Large model (~1.3GB)
- Verify system requirements

### Model Management

The project includes flexible model management:

```bash
# Download specific models
./download_models.sh large          # Large model (best quality)
./download_models.sh small          # Small model (fastest)
./download_models.sh small base     # Multiple models
./download_models.sh all            # All models

# Check model status
./download_models.sh                # Shows current status

# Models are automatically downloaded if missing
python depth_surge_3d.py input.mp4  # Auto-downloads if needed
```

**Available Models:**
- **Small** (24.8M params) - Fast processing, lower quality
- **Base** (97.5M params) - Balanced performance and quality  
- **Large** (335.3M params) - Best quality (default)

### Manual Installation

If you prefer manual setup:

```bash
# Using uv (recommended)
uv sync

# Or using pip
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download dependencies
git clone https://github.com/DepthAnything/Depth-Anything-V2.git depth_anything_v2_repo
./download_models.sh large
```

## Usage

### Basic Usage

```bash
# With uv (recommended - runs directly without manual activation)
uv run python depth_surge_3d.py input_video.mp4

# Or activate virtual environment first
source .venv/bin/activate  # for uv
# source venv/bin/activate  # for traditional venv

# Then run the script
python depth_surge_3d.py input_video.mp4

# Or use the installed command (if installed with -e flag)
stereo-projector input_video.mp4
```

### Web UI (Recommended)

Launch the web interface for an intuitive experience:

```bash
./run_ui.sh
```

Then open http://localhost:5000 in your browser. The web UI provides:
- **Drag & drop video upload** with instant preview
- **Visual time range selection** with duration display
- **Real-time progress tracking** with frame previews
- **Interactive settings panel** with proper defaults
- **Batch management** with timestamped outputs
- **Live preview** of depth maps and stereo frames

### Quick Start Script

For command-line processing with time ranges:

```bash
# Place your video as 'input_video.mp4' in the project directory
# Then run with start and end times
./start.sh 1:11 2:22

# The script automatically:
# - Uses input_video.mp4 as source
# - Extracts the specified time range
# - Preserves audio in the output (by default)
# - Upscales to minimum 1080p resolution
# - Interpolates frames to 60fps for smooth VR experience
# - Uses optimized settings for high-quality processing
```

### Advanced Usage

```bash
# With uv (all examples work with "uv run" prefix)
uv run python depth_surge_3d.py input_video.mp4 -o ./my_output -f over_under

# Or with activated virtual environment:

# Specify output directory and format
python depth_surge_3d.py input_video.mp4 -o ./my_output -f over_under

# Process only a specific time range (from 1 minute 30 seconds to 3 minutes 45 seconds)
python depth_surge_3d.py input_video.mp4 -s 01:30 -e 03:45

# Process with hours:minutes:seconds format
python depth_surge_3d.py input_video.mp4 -s 00:01:30 -e 00:03:45

# Process from start to specific time
python depth_surge_3d.py input_video.mp4 -e 02:00

# Process from specific time to end
python depth_surge_3d.py input_video.mp4 -s 01:00

# Adjust stereo parameters
python depth_surge_3d.py input_video.mp4 -b 0.15 -fl 1200

# Use CPU only (if no GPU available)
python depth_surge_3d.py input_video.mp4 --device cpu

# Don't save intermediate files
python depth_surge_3d.py input_video.mp4 --no-intermediates

# Disable audio preservation (audio is preserved by default)
python depth_surge_3d.py input_video.mp4 -s 01:30 -e 02:00 --no-audio

# Custom framerate and resolution
python depth_surge_3d.py input_video.mp4 --fps 120 --resolution 4k

# Process with original resolution (no upscaling)
python depth_surge_3d.py input_video.mp4 --resolution original --fps 30

# Combined example: high-quality 4K 60fps processing
python depth_surge_3d.py input_video.mp4 -s 01:30 -e 02:00 -f over_under --resolution 4k --fps 60 --no-intermediates
```

### Command Line Options

- `input_video`: Path to input video file
- `-o, --output`: Output directory (default: ./output)
- `-m, --model`: Path to Depth Anything V2 model file
- `-f, --format`: VR format - 'side_by_side' or 'over_under' (default: side_by_side)
- `--processing-mode`: Processing mode - 'serial' or 'batch' (default: serial)
- `--vr-resolution`: Resolution format including all aspect ratios (default: auto)
  - Square: square-480, square-720, square-1k, square-2k, square-3k, square-4k, square-5k
  - 16:9: 16x9-480p, 16x9-720p, 16x9-1080p, 16x9-1440p, 16x9-4k, 16x9-5k, 16x9-8k
  - Legacy Wide: wide-2k, wide-4k, ultrawide
  - Cinema: cinema-2k, cinema-4k (ultra-wide, recommend over-under)
  - Custom: custom:WIDTHxHEIGHT (e.g., custom:1920x1080)
- `-s, --start`: Start time in mm:ss or hh:mm:ss format (e.g., 01:30 or 00:01:30)
- `-e, --end`: End time in mm:ss or hh:mm:ss format (e.g., 03:45 or 00:03:45)
- `-b, --baseline`: Stereo baseline distance (default: 0.065 - average human IPD)
- `-fl, --focal-length`: Virtual focal length (default: 1000)
- `--fisheye-projection`: Projection model (default: stereographic)
- `--fisheye-fov`: Field of view in degrees (default: 105, range: 75-180)
- `--crop-factor`: Center crop factor (default: 1.0 - no crop)
- `--hole-fill-quality`: Hole filling quality - 'fast' or 'advanced' (default: fast)
- `--fps`: Target framerate for output video (default: 60)
- `--resolution`: Minimum resolution - '720p', '1080p', '4k', or 'original' (default: 1080p)
- `--no-audio`: Do not preserve audio from original video (audio preserved by default)
- `--no-intermediates`: Don't save intermediate depth maps and stereo frames
- `--device`: Processing device - 'cpu', 'cuda', or 'auto' (default: auto)

## Output Structure

When processing a video, the tool creates:

```
output/
├── frames/              # Extracted video frames
├── depth_maps/          # Generated depth maps
├── left_frames/         # Left stereo images
├── right_frames/        # Right stereo images
├── vr_frames/          # Combined VR frames
└── stereo_output_side_by_side.mp4  # Final VR video
```

## Wide Aspect Ratio Support

Depth Surge 3D now supports preserving more of your original content with wide aspect ratios:

### Format Recommendations
- **Ultra-wide content (>2.2:1)**: Cinema formats with **over-under** layout recommended
- **Wide content (>1.6:1)**: Wide formats, consider **over-under** for better preservation
- **Standard content**: Square formats work best with **side-by-side**

### Resolution Options
- **Square formats**: Optimized for VR headsets (1:1 aspect ratio)
- **Wide formats**: 16:9 aspect ratio, preserves more horizontal content
- **Cinema formats**: 2.39:1 ultra-wide, ideal for cinematic content

### Auto-Detection
The system automatically detects your content's aspect ratio and recommends the best format and resolution combination.

### Custom Resolutions

You can specify any resolution for your VR output:

**Command Line:**
```bash
python depth_surge_3d.py --vr-resolution custom:1920x1080 input.mp4
python depth_surge_3d.py --vr-resolution custom:2560x1600 input.mp4  # Custom aspect ratio
```

**Web UI:** Select "Custom" in the VR Resolution dropdown and enter width/height values.

**16:9 Standard Formats:**
- `16x9-480p` → 854×480 per eye (quick testing)
- `16x9-720p` → 1280×720 per eye (HD)
- `16x9-1080p` → 1920×1080 per eye (Full HD)
- `16x9-1440p` → 2560×1440 per eye (QHD)
- `16x9-4k` → 3840×2160 per eye (Ultra HD)
- `16x9-5k` → 5120×2880 per eye (5K)
- `16x9-8k` → 7680×4320 per eye (8K)

## Processing Modes

Depth Surge 3D offers two processing modes optimized for different scenarios:

### Serial Mode (Default)
- **Frame-by-frame processing**: Each frame goes through the complete pipeline before starting the next
- **Lower memory usage**: Only one frame in memory at a time
- **Predictable progress**: Clear frame-by-frame progress tracking
- **Best for**: Limited memory systems, debugging, or when consistent memory usage is important

### Batch Mode (Recommended for Performance)
- **Task-by-task processing**: Complete each processing step for all frames before moving to the next step
- **Parallelization**: Uses multiple CPU cores for significant speed improvements
- **Higher memory usage**: Multiple frames processed simultaneously
- **Step-based progress**: Tracks progress by processing steps rather than individual frames
- **Best for**: Systems with ample RAM and multi-core CPUs

### Performance Comparison
- **Serial**: Predictable, lower memory, good for limited resources
- **Batch**: 2-4x faster on multi-core systems, but requires more RAM

## VR Viewing

The generated video can be viewed with:
- VR headsets (Oculus, HTC Vive, etc.)
- Cardboard VR viewers
- 3D video players that support side-by-side or over-under formats
- Wide-screen displays (for cinema formats)

## Workflow Example

Here's a typical workflow using the start script:

1. **Setup** (one time):
   ```bash
   ./setup.sh
   ```

2. **Test setup** (optional but recommended):
   ```bash
   ./test.sh
   ```

3. **Prepare your video**:
   ```bash
   # Copy or rename your video file
   cp my_video.mp4 input_video.mp4
   ```

4. **Quick processing**:
   ```bash
   # Process a 15-second clip from 1:00 to 1:15 with audio
   ./start.sh 1:00 1:15
   ```

5. **Find your output**:
   ```bash
   # The processed video will be in:
   # ./output/stereo_output_side_by_side.mp4
   ```

The start script automatically preserves audio and uses optimized settings for quick processing.

## Testing & Troubleshooting

### Quick Setup Test
Run `./test.sh` to verify your installation:
- ✓ Python dependencies
- ✓ CUDA availability
- ✓ Model files
- ✓ Input video
- ✓ FFmpeg

### Common Issues

1. **"uv.lock parse error"**: The script automatically falls back to virtual environment mode
2. **"xFormers not available"**: This warning is normal and doesn't affect functionality
3. **Slow processing**: GPU acceleration requires CUDA-compatible hardware
4. **Out of memory**: Reduce resolution with `--resolution 720p` or use `--device cpu`

## Attribution

This project builds upon the excellent **Depth-Anything-V2** model:

- **Research Paper**: [Depth Anything V2](https://arxiv.org/abs/2406.09414) by Lihe Yang, Bingyi Kang, Zilong Huang, et al.
- **Original Repository**: [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)

> Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., & Zhao, H. (2024). *Depth Anything V2*. arXiv preprint arXiv:2406.09414.

## Technical Details

### Stereo Generation Algorithm

1. **Video Enhancement**: FFmpeg upscales video to minimum 1080p and interpolates frames to 60fps using motion compensation
2. **Frame Extraction**: Enhanced frames are extracted as high-quality PNG files
3. **Depth Estimation**: Depth Anything V2 predicts relative depth for each pixel
4. **Disparity Calculation**: Depth maps are converted to disparity using stereo parameters
5. **Stereo Pair Creation**: Both left and right images are generated by horizontal pixel shifting in opposite directions based on disparity
6. **VR Format Creation**: Left and right images are combined in side-by-side or over-under format
7. **Audio Synchronization**: Original audio is extracted for the same time range and combined with the final video

### Parameters

- **Baseline**: Distance between virtual left and right cameras (default: 0.065m - average human IPD)
- **Focal Length**: Virtual camera focal length. Affects the depth perception scale
- **Projection**: Fisheye projection model (default: stereographic for better quality)
- **Field of View**: Fisheye FOV in degrees (default: 105° for natural viewing)
- **Crop Factor**: Center crop amount (default: 1.0 - no crop to preserve content)
- **Target FPS**: Frame interpolation target (default 60fps for smooth VR)
- **Resolution**: Minimum output resolution with intelligent upscaling

### Video Enhancement Features

- **Frame Interpolation**: Uses FFmpeg's minterpolate with motion compensation for smooth 60fps+ output
- **Upscaling**: Lanczos algorithm for high-quality resolution enhancement
- **Audio Sync**: Precise audio extraction and synchronization with time ranges
- **Format Support**: Handles various input formats and resolutions

## Performance

- GPU processing is ~10x faster than CPU for depth estimation
- Frame interpolation adds ~30% processing time but significantly improves VR smoothness
- Upscaling from 720p to 1080p adds ~20% processing time
- Typical processing: ~2-4 seconds per output frame on modern GPU (including enhancement)

## Limitations

- Works best with scenes containing varied depth information
- May struggle with mirrors, glass, or very dark scenes
- Generated stereo effect is approximate, not true stereo capture
- Processing time can be significant for long/high-resolution videos

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Components and Attribution

This project builds upon the **Depth-Anything-V2** model:

- **Original Repository**: [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- **Research Paper**: [Depth Anything V2](https://arxiv.org/abs/2406.09414) (arXiv:2406.09414)
- **Model License**: CC-BY-NC-4.0 (non-commercial use)

**Citation**:
```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}
```

**Important**: The Depth-Anything-V2 models are licensed under CC-BY-NC-4.0, which permits academic and research use but restricts commercial use. Please review the [original license](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/LICENSE) for complete terms.