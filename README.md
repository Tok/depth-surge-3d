# Depth Surge 3D

A powerful command-line tool and web application that converts 2D videos to immersive 3D VR format using advanced AI depth estimation with Depth Anything V2.

## Features

- Extract frames from video using FFmpeg
- Generate depth maps using Depth Anything V2
- Create left and right stereo images from depth information
- Output in VR-compatible formats (side-by-side or over-under)
- Save intermediate results (depth maps, stereo frames)
- High-performance GPU acceleration support
- Intuitive web interface for easy processing
- Advanced fisheye projection and distortion correction

## Requirements

- Python 3.8+
- FFmpeg
- CUDA-compatible GPU (optional, for faster processing)

## Installation

### Option 1: Using the setup script (recommended)
1. Clone or download this repository
2. Run the setup script (automatically installs uv if needed):
   ```bash
   ./setup.sh
   ```

### Option 2: Manual installation with uv
1. Install uv if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Create virtual environment and install dependencies:
   ```bash
   uv venv
   uv pip install -e .
   ```

### Option 3: Traditional pip installation
1. Create virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
# With uv (recommended - runs directly without manual activation)
uv run python stereo_projector.py input_video.mp4

# Or activate virtual environment first
source .venv/bin/activate  # for uv
# source venv/bin/activate  # for traditional venv

# Then run the script
python stereo_projector.py input_video.mp4

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
./start.sh 1:18 1:33

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
uv run python stereo_projector.py input_video.mp4 -o ./my_output -f over_under

# Or with activated virtual environment:

# Specify output directory and format
python stereo_projector.py input_video.mp4 -o ./my_output -f over_under

# Process only a specific time range (from 1 minute 30 seconds to 3 minutes 45 seconds)
python stereo_projector.py input_video.mp4 -s 01:30 -e 03:45

# Process with hours:minutes:seconds format
python stereo_projector.py input_video.mp4 -s 00:01:30 -e 00:03:45

# Process from start to specific time
python stereo_projector.py input_video.mp4 -e 02:00

# Process from specific time to end
python stereo_projector.py input_video.mp4 -s 01:00

# Adjust stereo parameters
python stereo_projector.py input_video.mp4 -b 0.15 -fl 1200

# Use CPU only (if no GPU available)
python stereo_projector.py input_video.mp4 --device cpu

# Don't save intermediate files
python stereo_projector.py input_video.mp4 --no-intermediates

# Disable audio preservation (audio is preserved by default)
python stereo_projector.py input_video.mp4 -s 01:30 -e 02:00 --no-audio

# Custom framerate and resolution
python stereo_projector.py input_video.mp4 --fps 120 --resolution 4k

# Process with original resolution (no upscaling)
python stereo_projector.py input_video.mp4 --resolution original --fps 30

# Combined example: high-quality 4K 60fps processing
python stereo_projector.py input_video.mp4 -s 01:30 -e 02:00 -f over_under --resolution 4k --fps 60 --no-intermediates
```

### Command Line Options

- `input_video`: Path to input video file
- `-o, --output`: Output directory (default: ./output)
- `-m, --model`: Path to Depth Anything V2 model file
- `-f, --format`: VR format - 'side_by_side' or 'over_under' (default: side_by_side)
- `-s, --start`: Start time in mm:ss or hh:mm:ss format (e.g., 01:30 or 00:01:30)
- `-e, --end`: End time in mm:ss or hh:mm:ss format (e.g., 03:45 or 00:03:45)
- `-b, --baseline`: Stereo baseline distance (default: 0.1)
- `-fl, --focal-length`: Virtual focal length (default: 1000)
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

## VR Viewing

The generated video can be viewed with:
- VR headsets (Oculus, HTC Vive, etc.)
- Cardboard VR viewers
- 3D video players that support side-by-side or over-under formats

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
   # Process a 15-second clip from 1:18 to 1:33 with audio
   ./start.sh 1:18 1:33
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

## Technical Details

### Stereo Generation Algorithm

1. **Video Enhancement**: FFmpeg upscales video to minimum 1080p and interpolates frames to 60fps using motion compensation
2. **Frame Extraction**: Enhanced frames are extracted as high-quality PNG files
3. **Depth Estimation**: Depth Anything V2 predicts relative depth for each pixel
4. **Disparity Calculation**: Depth maps are converted to disparity using stereo parameters
5. **Stereo Pair Creation**: Left image is original, right image is generated by horizontal pixel shifting based on disparity
6. **VR Format Creation**: Left and right images are combined in side-by-side or over-under format
7. **Audio Synchronization**: Original audio is extracted for the same time range and combined with the final video

### Parameters

- **Baseline**: Distance between virtual left and right cameras. Larger values create stronger 3D effect but may cause eye strain
- **Focal Length**: Virtual camera focal length. Affects the depth perception scale
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

This project uses Depth Anything V2 which has specific licensing terms. Please check the original repository for details.