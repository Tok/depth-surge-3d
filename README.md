# Depth Surge 3D

**Convert 2D videos to 3D VR format using AI depth estimation.**

Depth Surge 3D is a command-line tool and web application that converts flat videos into stereoscopic 3D format for VR headsets and 3D displays. Using the **Video-Depth-Anything** neural network, it analyzes video frames with temporal consistency to predict depth information, then generates left and right eye views for stereoscopic viewing.

## How It Works

The conversion process combines computer vision with stereoscopic rendering:

1. **AI Depth Analysis**: Video-Depth-Anything processes video frames with temporal consistency to create smooth depth maps across time, estimating the 3D structure of the scene without requiring specialized camera equipment.

2. **Stereo Pair Generation**: Using the predicted depth information, the system generates separate left and right eye images by shifting pixels based on their distance from the viewer - closer objects appear more separated, distant objects less so.

3. **VR Optimization**: The stereo pairs are processed through configurable fisheye distortion and projection models to match different VR headset optics.

4. **Format Adaptation**: Final output can be rendered in standard VR formats (side-by-side or over-under) with resolution options from preview quality to 8K.

The result is 3D video that maintains the original content's motion and timing while adding depth perception suitable for VR viewing.

## Key Features

- **Temporal Consistency**: Uses Video-Depth-Anything for smooth depth across frames with 32-frame sliding windows
- **AI Depth Estimation**: Monocular depth prediction with video-aware processing
- **Multiple VR Formats**: Supports side-by-side and over-under stereoscopic formats
- **Flexible Resolutions**: Square (VR-optimized), 16:9 (standard), and custom resolutions
- **Fisheye Distortion**: Multiple projection models for VR headset compatibility
- **Auto-detection**: Optimal settings based on source content aspect ratio
- **Resume Capability**: Save settings and resume interrupted processing
- **Audio Preservation**: Maintains original audio synchronization
- **Progress Tracking**: Real-time progress with ETA estimates
- **Wide Format Support**: Cinema, ultra-wide, and standard aspect ratios
- **GPU Acceleration**: CUDA support for faster processing
- **Web Interface**: Browser-based interface alongside command-line tools
- **Intermediate Files**: Optional saving of depth maps and processing stages
- **Memory-efficient**: Chunked processing for long videos and high resolutions

## Requirements

- Python 3.9+
- FFmpeg
- CUDA-compatible GPU (optional, for faster processing)

## Based on Video-Depth-Anything

This project uses [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) for temporal-consistent depth estimation across video frames. The model processes frames in 32-frame chunks with overlap to ensure smooth depth transitions.

The setup script automatically clones the repository and downloads the model weights.

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

If you prefer manual setup or if the automatic setup fails:

#### Step 1: System Dependencies

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

#### Step 2: Python Environment

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

#### Step 3: Download Video-Depth-Anything

**Option A: Automatic download (recommended)**
```bash
./download_models.sh large
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

#### Step 4: Verify Installation

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

## Usage

### Basic Usage

```bash
# With uv (recommended - runs directly without manual activation)
uv run python depth_surge_3d.py input_video.mp4

# Or activate virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Then run the script
python depth_surge_3d.py input_video.mp4
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
- `-m, --model`: Path to Video-Depth-Anything model file (auto-downloads if missing)
- `-f, --format`: VR format - 'side_by_side' or 'over_under' (default: side_by_side)
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

### Understanding Stereo Parameters & Artifact Management

#### Baseline and Focal Length Relationship

The **baseline** (distance between virtual cameras) and **focal length** work together to control 3D depth strength:

- **Baseline (default: 0.065m)**: Physical separation between left/right viewpoints
  - **Larger baseline** = stronger 3D effect, more "pop-out"
  - **Smaller baseline** = subtler 3D effect, more comfortable viewing
  - **Range**: 0.02m (subtle) to 0.15m (very strong)

- **Focal Length (default: 1000px)**: Virtual camera lens characteristics
  - **Higher focal length** = objects appear closer, stronger depth separation
  - **Lower focal length** = objects appear farther, gentler depth transitions
  - **Range**: 500px (wide-angle feel) to 2000px (telephoto feel)

**Tuning Strategy**:
```bash
# Subtle 3D for comfortable viewing
--baseline 0.04 --focal-length 800

# Strong 3D for dramatic effect
--baseline 0.10 --focal-length 1200

# Balanced (default)
--baseline 0.065 --focal-length 1000
```

#### Managing Artifacts and "Invented Pixels"

**Common Artifacts**:
- **Stretching/warping**: Objects appear distorted at depth boundaries
- **Floating pixels**: Disconnected visual elements
- **Edge artifacts**: Jagged or broken object boundaries
- **"Invented pixels"**: AI fills gaps with estimated content that may not match reality

**Artifact Reduction Strategies**:

1. **Reduce 3D Strength** (most effective):
   ```bash
   # Conservative settings for clean results
   --baseline 0.035 --focal-length 700
   ```

2. **Adjust Hole Filling**:
   - `--hole-fill-quality fast`: Simple inpainting, fewer artifacts but visible gaps
   - `--hole-fill-quality advanced`: Better gap filling but may introduce AI "hallucinations"

3. **Crop More Aggressively**:
   ```bash
   # Remove problematic edges
   --crop-factor 0.8 --fisheye-crop-factor 0.9
   ```

**Trade-offs to Understand**:
- **Strong 3D vs. Clean Image**: More dramatic depth = more artifacts
- **Hole Filling Quality vs. Performance**: Advanced filling is 3-5x slower with marginal visual improvement
- **Edge Preservation vs. 3D Effect**: Keeping original content vs. creating convincing stereo

#### Performance vs. Quality Trade-offs

**Hole Filling Quality Impact**:
- **Fast**: ~2-4 seconds per frame, basic gap filling
- **Advanced**: ~8-15 seconds per frame, sophisticated depth-guided filling
- **Reality**: Advanced mode often provides only 10-20% visual improvement despite 3-4x processing time

**Optimization Recommendations**:
```bash
# Fast processing for testing
--hole-fill-quality fast --vr-resolution 16x9-720p

# Balanced quality/speed
--hole-fill-quality fast --vr-resolution 16x9-1080p

# Maximum quality (slow)
--hole-fill-quality advanced --vr-resolution 16x9-4k
```

**When to Reduce 3D Strength**:
- Content with many depth discontinuities (trees, hair, complex objects)
- Fast camera movement or quick subject motion
- Scenes with reflective surfaces or transparent objects
- When artifacts are more distracting than the 3D effect is beneficial

**Remember**: Subtle, clean 3D is often more enjoyable than aggressive 3D with artifacts. Start conservative and increase strength only if the content handles it well.

## Processing Settings & Resume

### Automatic Settings Recording

Every processing job automatically creates a `[batchname]-settings.json` file in the output directory containing:

- **Complete processing parameters**: All settings used for the conversion
- **Video metadata**: Source video properties and technical details  
- **Timestamps**: Creation time, completion time, and processing duration
- **Status tracking**: Current processing state and progress information
- **Output information**: Expected filenames and directory structure

### Resume Interrupted Processing

If processing is interrupted (power failure, system crash, manual stop), you can resume from where it left off:

```bash
# Resume processing from any output directory
python depth_surge_3d.py --resume ./output/my_video_1234567890/

# Check what can be resumed
python depth_surge_3d.py --resume ./output/my_video_1234567890/
```

**Resume Process:**
1. **Validates directory**: Checks for valid settings file and intermediate files
2. **Analyzes progress**: Determines how many frames were already processed
3. **Loads original settings**: Uses exact same parameters as the original run
4. **Continues processing**: Picks up from the last completed frame

**Resume Capability:**
- ✅ **Supported**: Interrupted processing (`in_progress`, `failed`, `paused`)
- ✅ **Smart detection**: Automatically finds completed intermediate stages
- ✅ **Progress analysis**: Shows exactly how much work was completed
- ❌ **Not needed**: Already completed processing (`completed`)

### Settings File Example

```json
{
  "metadata": {
    "batch_name": "sample_video_1703123456",
    "source_video": "/path/to/sample_video.mp4",
    "created_at": "2024-01-20 15:30:56",
    "processing_status": "completed",
    "processing_duration_formatted": "12m 34s"
  },
  "video_properties": {
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "frame_count": 1800
  },
  "processing_settings": {
    "vr_format": "side_by_side",
    "vr_resolution": "16x9-1080p",
    "baseline": 0.065,
    "fisheye_fov": 180
  },
  "output_info": {
    "expected_output_filename": "sample_video_side-by-side_16x9-1080p.mp4"
  }
}
```

## Attribution

This project builds upon the excellent **Video-Depth-Anything** model:

- **Original Repository**: [DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)
- **Research Paper**: Based on Depth Anything V2 architecture with temporal consistency

Video-Depth-Anything provides temporally-consistent depth estimation across video frames, essential for smooth 3D VR experiences.

## Technical Details

### Processing Pipeline

The conversion process follows a multi-stage pipeline:

1. **Video Preprocessing**: FFmpeg extracts frames and optionally enhances resolution and frame rate
2. **AI Depth Analysis**: Video-Depth-Anything neural network analyzes video frames in 32-frame chunks with temporal consistency to generate smooth depth maps showing distance of every pixel
3. **Disparity Conversion**: Depth information is converted to horizontal disparity values using stereo camera parameters
4. **Stereo Image Generation**: Both left and right eye images are created by shifting pixels based on their depth - closer objects appear more separated, distant objects less so
5. **VR Lens Simulation**: Optional fisheye distortion and projection models simulate different VR headset optics
6. **Format Assembly**: Stereo pairs are combined into standard VR formats (side-by-side or over-under)
7. **Audio Integration**: Original audio is synchronized with the converted video to maintain lip-sync

### Parameters

- **Baseline**: Distance between virtual left and right cameras (default: 0.065m - average human IPD)
- **Focal Length**: Virtual camera focal length. Affects the depth perception scale
- **Projection**: Fisheye projection model (default: stereographic)
- **Field of View**: Fisheye FOV in degrees (default: 105°)
- **Crop Factor**: Center crop amount (default: 1.0 - no crop)
- **Target FPS**: Frame interpolation target (default 60fps)
- **Resolution**: Minimum output resolution with upscaling

### Video Enhancement Features

- **Frame Interpolation**: Uses FFmpeg's minterpolate with motion compensation for smooth 60fps+ output
- **Upscaling**: Lanczos algorithm for high-quality resolution enhancement
- **Audio Sync**: Precise audio extraction and synchronization with time ranges
- **Format Support**: Handles various input formats and resolutions

## Performance

- GPU processing is ~10x faster than CPU for depth estimation
- Video model processes in 32-frame chunks for temporal consistency
- Automatic memory management for 4K/HD videos with chunked processing
- Better temporal consistency than frame-by-frame approaches
- Typical processing: ~2-4 seconds per output frame on modern GPU

## Quality Expectations & Limitations

### When It Works Well
- Videos with clear depth variation (landscapes, interiors with furniture, people in scenes)
- Good lighting conditions with visible detail
- Source resolution 1080p or higher for best results
- Content shot with steady camera movement

### Known Limitations
- **AI depth estimation**: Generated stereo effect is approximate, not true stereo capture
- **Challenging scenes**: May produce poor results with mirrors, glass, water reflections, or very dark scenes
- **Monochrome/low contrast**: Scenes with little visual variation provide insufficient depth cues
- **Fast motion**: Rapid camera movements or quick object motion may cause artifacts
- **Processing time**: Can be significant for long/high-resolution videos (2-4 seconds per output frame)

### Resolution Recommendations
- **Low resolutions** (480p, 720p): Primarily for quick testing and preview - expect reduced quality
- **1080p+**: Recommended minimum for acceptable VR viewing quality
- **4K+**: Best results for high-end VR headsets
- **Ultra-low source material**: Results will be limited by source quality regardless of output resolution

**Note**: This tool converts monocular video to pseudo-stereo using AI depth estimation. While results can be compelling for many types of content, they will never match the quality of content shot with actual stereo cameras or specialized VR equipment.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Components and Attribution

This project builds upon the **Video-Depth-Anything** model:

- **Original Repository**: [DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)
- **Based on**: Depth Anything V2 architecture with temporal consistency
- **Model License**: Check the Video-Depth-Anything repository for current license terms

**Important**: Please review the [Video-Depth-Anything license](https://github.com/DepthAnything/Video-Depth-Anything) for complete terms before commercial use.