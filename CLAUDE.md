# Depth Surge 3D - Claude Development Notes

## Project Overview
A comprehensive 2D to 3D VR video converter using Depth Anything V2 for depth estimation and advanced FFmpeg processing for high-quality stereo pair generation.

## Key Features Implemented
- **Command-line Interface**: Full-featured CLI with time range selection, resolution control, and VR format options
- **Web UI**: Modern dark-themed Flask interface with real-time progress tracking and live frame previews
- **60fps Frame Interpolation**: Motion-compensated interpolation using FFmpeg's minterpolate
- **Intelligent Upscaling**: Lanczos algorithm for quality enhancement to 1080p/4K
- **Audio Preservation**: Time-synchronized audio extraction and combination
- **Symmetric Stereo Generation**: Proper half-disparity shifts for both left and right eyes
- **Multiple VR Formats**: Side-by-side and over-under variants (LR/RL)

## Technical Architecture
- **Backend**: Flask + SocketIO for real-time communication
- **AI Model**: Depth Anything V2 (ViT-Large) for depth estimation
- **Video Processing**: FFmpeg for frame extraction, enhancement, and final video assembly
- **Progress Tracking**: WebSocket-based real-time updates with sub-progress bars
- **GPU Acceleration**: CUDA support with automatic fallback to CPU

## Development Commands
```bash
# Setup and testing
./setup.sh                    # Initial setup with dependencies
./test.sh                     # Verify installation and system info
./run_ui.sh                   # Launch web interface (auto-opens browser)

# Command-line usage
./start.sh 1:18 1:33          # Quick processing with time range
python stereo_projector.py --help  # Full CLI options

# Development workflow
git status                    # Check current state
git add -A && git commit -m "message"  # Commit changes
```

## Architecture Decisions
1. **Default to Source FPS**: Avoid unnecessary interpolation that may degrade quality
2. **Symmetric Stereo Pairs**: Both eyes receive proper disparity shifts (not just one)
3. **Dark Theme UI**: Desktop-optimized interface for professional use
4. **Real-time Preview**: Optional live frame previews with performance toggle
5. **Timestamped Output**: Organized output directories with video name and timestamp

## Performance Characteristics
- **GPU Processing**: ~2-4 seconds per output frame on RTX 4070 Ti SUPER
- **Frame Enhancement**: +30% time for interpolation, +20% for upscaling
- **Memory Usage**: Processes frames sequentially to manage VRAM
- **Preview Updates**: Every 5th frame for smooth real-time feedback

## Output Structure
```
output/
├── video_name_20250615_143022/
│   ├── frames/              # Enhanced extracted frames
│   ├── depth_maps/          # AI-generated depth maps
│   ├── left_frames/         # Left stereo images
│   ├── right_frames/        # Right stereo images
│   ├── vr_frames/          # Combined VR frames
│   └── video_name_3D_side_by_side_0118-0133.mp4
```

## VR Compatibility
- **Oculus/Meta Headsets**: Side-by-side format
- **HTC Vive**: Over-under format
- **Cardboard VR**: Both formats supported
- **3D Video Players**: Standard VR format compliance

## Known Limitations
- **Processing Time**: Can be significant for long/high-resolution videos
- **Depth Quality**: Works best with scenes containing varied depth information
- **Stereo Effect**: Approximate stereo generation, not true stereo capture
- **Memory Requirements**: GPU processing requires adequate VRAM

## Future Enhancements
- **Batch Processing**: Multiple video queue support
- **Advanced Stereo Parameters**: Fine-tuning for different VR headsets
- **Preview Optimization**: Faster preview generation
- **Export Formats**: Additional VR-compatible output formats

## Dependencies
- **Core**: Python 3.8+, PyTorch 2.0+, OpenCV 4.8+
- **Web UI**: Flask 3.0+, SocketIO 5.3+, Bootstrap 5.3
- **Video**: FFmpeg with full codec support
- **AI Model**: Depth Anything V2 (pre-downloaded in /models)

## Browser Compatibility
- **Chrome/Edge**: Full feature support
- **Firefox**: Full feature support
- **Safari**: Basic support (some WebSocket limitations)

## System Requirements
- **Minimum**: 8GB RAM, modern CPU, 4GB storage
- **Recommended**: 16GB RAM, CUDA GPU, 10GB storage
- **Optimal**: 32GB RAM, RTX 4070+ GPU, SSD storage

## Troubleshooting
- **"uv.lock parse error"**: Script automatically falls back to virtual environment
- **"xFormers not available"**: Warning is normal, doesn't affect functionality
- **Slow processing**: Verify GPU acceleration or use `--device cpu`
- **Out of memory**: Reduce resolution with `--resolution 720p`

## Code Quality
- **Error Handling**: Comprehensive try-catch with graceful fallbacks
- **Progress Tracking**: Detailed progress callbacks with stage information
- **Memory Management**: Efficient frame processing and cleanup
- **Cross-platform**: Windows, macOS, and Linux compatibility

## Git Workflow
The project uses conventional commits with detailed descriptions and co-authorship attribution to Claude Code.