# Depth Surge 3D

**Convert 2D videos to 3D VR format using AI depth estimation.**

Depth Surge 3D transforms flat videos into stereoscopic 3D for VR headsets using the **Video-Depth-Anything** neural network. It analyzes video frames with temporal consistency to predict depth, then generates left and right eye views for immersive stereoscopic viewing.

## Key Features

- **Temporal Consistency**: Uses Video-Depth-Anything for smooth depth across frames with 32-frame sliding windows
- **Multiple VR Formats**: Side-by-side and over-under stereoscopic formats
- **Flexible Resolutions**: Square (VR-optimized), 16:9 (standard), cinema, and custom resolutions up to 8K
- **Resume Capability**: Intelligent step-level resume for interrupted processing
- **Audio Preservation**: Maintains original audio synchronization
- **Web Interface**: Modern browser-based UI with real-time progress tracking
- **GPU Acceleration**: CUDA support for 10x faster processing
- **Wide Format Support**: Cinema, ultra-wide, and standard aspect ratios

## Quick Start

### Installation

```bash
git clone https://github.com/tok/depth-surge-3d.git depth-surge-3d
cd depth-surge-3d
chmod +x setup.sh
./setup.sh
```

The setup script automatically installs all dependencies, downloads the Video-Depth-Anything model (~1.3GB), and verifies your system.

**See [Installation Guide](docs/INSTALLATION.md) for detailed setup instructions.**

### Usage

**Web UI (Recommended):**
```bash
./run_ui.sh
# Opens http://localhost:5000 in your browser
```

**Command Line:**
```bash
# Basic usage
python depth_surge_3d.py input_video.mp4

# Process specific time range with custom settings
python depth_surge_3d.py input_video.mp4 -s 01:30 -e 03:45 -f over_under --resolution 4k
```

**Quick Start Script:**
```bash
# Process a clip with optimized settings
./start.sh 1:11 2:22
```

**See [Usage Guide](docs/USAGE.md) for comprehensive usage examples.**

## Requirements

- Python 3.9+
- FFmpeg
- CUDA 13.0+ (required for GPU acceleration)
- CUDA-compatible GPU (optional but strongly recommended)

## Documentation

- **[Installation Guide](docs/INSTALLATION.md)** - Detailed setup instructions and troubleshooting
- **[Usage Guide](docs/USAGE.md)** - Complete usage examples and workflows
- **[Parameters Reference](docs/PARAMETERS.md)** - All command-line options and settings explained
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and performance tips
- **[Architecture](docs/ARCHITECTURE.md)** - Technical details and processing pipeline

## Output Structure

Each processing session creates a self-contained timestamped directory:

```
output/
└── timestamp_videoname_timestamp/
    ├── original_video.mp4      # Source video
    ├── original_audio.flac     # Pre-extracted audio
    ├── frames/                 # Extracted frames
    ├── vr_frames/             # Final VR frames
    └── videoname_3D_side_by_side.mp4  # Final 3D video
```

## VR Viewing

Generated videos work with:
- VR headsets (Meta Quest, HTC Vive, etc.)
- Cardboard VR viewers
- 3D video players supporting side-by-side or over-under formats

## Performance

- **GPU Processing**: ~2-4 seconds per output frame (RTX 4070+ class)
- **CPU Processing**: ~30-60 seconds per output frame
- **Typical 1-minute clip**: ~2-4 hours on modern GPU at 60fps output

## Attribution

This project uses [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) for temporal-consistent depth estimation. The model is based on Depth Anything V2 architecture and processes frames in 32-frame chunks with overlap to ensure smooth depth transitions.

## License

MIT License - see [LICENSE](LICENSE) file for details.

**Third-Party Components**: Please review the [Video-Depth-Anything license](https://github.com/DepthAnything/Video-Depth-Anything) for terms before commercial use.

## Quality Expectations, Parallax-Glitchwave and Z-Collapse Slopcore Aesthetics

This tool converts monocular video to pseudo-stereo using AI depth estimation. Results can be compelling for many types of content but will never match true stereo cameras or specialized VR equipment.

**Best results with:**
- Clear depth variation (landscapes, interiors, people)
- Good lighting and detail
- Source resolution 1080p or higher
- Steady camera movement

**May struggle with:**
- Mirrors, glass, water reflections
- Very dark or low-contrast scenes
- Fast motion or rapid camera movements

**Artifact Embracement**: Expect algorithmic stereo divergence, synthetic depth layers, and monocular hallucinations. These AI-generated depth discontinuities create a unique aesthetic - depth-drift, disparity shimmer, and temporal judder may become part of the experience.

See the [Troubleshooting Guide](docs/TROUBLESHOOTING.md) for detailed quality expectations and optimization tips.
