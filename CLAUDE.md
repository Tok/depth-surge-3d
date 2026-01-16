# Depth Surge 3D - Claude Development Notes

## Project Overview
A comprehensive 2D to 3D VR video converter using Video-Depth-Anything V2 or Depth-Anything V3 for depth estimation and advanced FFmpeg processing for high-quality stereo pair generation.

**Current Version**: 0.8.0 (depth-anything-v3 branch)

## Key Features Implemented
- **Command-line Interface**: Full-featured CLI with time range selection, resolution control, and VR format options
- **Web UI**: Modern dark-themed Flask interface with real-time progress tracking and live frame previews
- **Dual Depth Model Support**: Choose between Video-Depth-Anything V2 (temporal consistency) or Depth-Anything V3 (better VRAM efficiency)
- **Resume Capability**: Intelligent step-level resume that skips completed processing stages
- **Audio Preservation**: Immediate lossless FLAC extraction with time-synchronized audio in final output
- **Symmetric Stereo Generation**: Proper half-disparity shifts for both left and right eyes
- **Multiple VR Formats**: Side-by-side and over-under variants with auto-detection
- **Generation-Specific Storage**: Self-contained timestamped output directories with original video
- **Temporal Consistency**: Video-Depth-Anything V2 with 32-frame sliding windows for smooth depth transitions

## Technical Architecture
- **Backend**: Flask + SocketIO (threading async_mode) for real-time communication
- **AI Models**:
  - Video-Depth-Anything V2 (Large/Base/Small) for temporal-consistent depth estimation
  - Depth-Anything V3 (Giant/Large/Base/Small) for improved VRAM efficiency
- **Video Processing**: FFmpeg for frame extraction, enhancement, and final video assembly
- **Progress Tracking**: 7-step weighted progress system with WebSocket-based real-time updates
- **GPU Acceleration**: CUDA support with automatic fallback to CPU
- **Storage Architecture**: Generation-specific directories eliminating redundant uploads

## Depth Model Comparison

### Video-Depth-Anything V2 (Default)
- **Best for**: Temporal consistency across video frames
- **Processing**: 32-frame sliding windows with overlap for smooth transitions
- **VRAM**: Higher memory usage, may struggle on 6GB GPUs with high-resolution videos
- **Temporal Artifacts**: Minimal, excellent for smooth video output
- **Use with**: `--depth-model-version v2` (default)

### Depth-Anything V3 (New)
- **Best for**: Limited VRAM (6GB GPUs like RTX 3060), faster processing
- **Processing**: Frame-by-frame independent processing
- **VRAM**: Significantly lower memory usage, better for high-resolution videos
- **Temporal Artifacts**: May have slight frame-to-frame variations
- **Models**: small, base, large (default), large-metric, giant
- **Use with**: `--depth-model-version v3 --model large`

### When to Use Each Model

**Use V2 if:**
- You have ≥12GB VRAM
- You need maximum temporal consistency
- Processing speed is not critical

**Use V3 if:**
- You have limited VRAM (6-8GB)
- You need faster processing
- You can accept minor frame-to-frame variations
- You're processing high-resolution (4K+) videos

## Development Commands
```bash
# Setup and testing
./setup.sh                    # Initial setup with dependencies
./test.sh                     # Verify installation and system info
./run_ui.sh                   # Launch web interface (auto-opens browser)

# Command-line usage (V2 - default)
./start.sh 1:00 2:00          # Quick processing with time range
python depth_surge_3d.py --help  # Full CLI options

# Command-line usage (V3 - better VRAM efficiency)
python depth_surge_3d.py input.mp4 --depth-model-version v3
python depth_surge_3d.py input.mp4 --depth-model-version v3 --model base  # Use smaller model
python depth_surge_3d.py input.mp4 --depth-model-version v3 --metric      # Use metric depth

# Development workflow
git status                    # Check current state
git add -A && git commit -m "message"  # Commit changes
```

## Architecture Decisions
1. **Modular Design**: Separated concerns with dedicated modules for depth estimation, image processing, and video assembly
2. **Symmetric Stereo Pairs**: Both eyes receive proper disparity shifts (not just one)
3. **Dark Theme UI**: Desktop-optimized interface for professional use
4. **Progress Tracking**: Unified interface supporting both CLI (tqdm) and WebSocket (Flask) progress reporting
5. **Generation-Specific Storage**: Each processing session in self-contained timestamped directory with original video
6. **Immediate Audio Extraction**: Audio extracted to lossless FLAC on upload for reuse in final video
7. **Step-Level Resume**: Comprehensive resume functionality that skips completed steps (7-step pipeline)
8. **Temporal Consistency**: Video-Depth-Anything processes in 32-frame chunks with overlap for smooth depth transitions

## Performance Characteristics
- **GPU Processing**: ~2-4 seconds per output frame on RTX 4070 Ti SUPER
- **Frame Enhancement**: +30% time for interpolation, +20% for upscaling
- **Memory Usage**: Processes frames sequentially to manage VRAM
- **Preview Updates**: Every 5th frame for smooth real-time feedback

## Output Structure
```
output/
└── timestamp_videoname_timestamp/
    ├── original_video.mp4      # Source video (uploaded or copied)
    ├── original_audio.flac     # Pre-extracted lossless audio
    ├── frames/                 # Enhanced extracted frames
    ├── depth_maps/            # AI-generated depth maps (if --keep-intermediates)
    ├── left_frames/           # Left stereo images (if --keep-intermediates)
    ├── right_frames/          # Right stereo images (if --keep-intermediates)
    ├── vr_frames/             # Combined VR frames
    ├── settings.json          # Processing parameters and metadata
    └── videoname_3D_side_by_side_0118-0133.mp4  # Final output
```

**Key Changes from Previous Architecture:**
- **No uploads/ directory**: Videos saved directly to generation-specific output directory
- **Self-contained sessions**: All files for one conversion in one timestamped directory
- **Pre-extracted audio**: Lossless FLAC extracted immediately, reused in final video
- **Resume-friendly**: Original video always available in output directory for resume functionality

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
- **Core**: Python 3.8+, PyTorch 2.0+, OpenCV 4.8+, CUDA 13.0+ (required for GPU)
- **Web UI**: Flask 3.0+, SocketIO 5.3+, Bootstrap 5.3
- **Video**: FFmpeg with full codec support
- **AI Models**:
  - Video-Depth-Anything V2 (pre-downloaded in /models, default)
  - Depth-Anything V3 (auto-downloaded from HuggingFace on first use)
  - xformers (required for DA3, optional for DA2)

## Browser Compatibility
- **Chrome/Edge**: Full feature support
- **Firefox**: Full feature support
- **Safari**: Basic support (some WebSocket limitations)

## System Requirements
- **Minimum**: 8GB RAM, modern CPU, 4GB storage
- **Recommended**: 16GB RAM, CUDA 13.0+ GPU, 10GB storage
- **Optimal**: 32GB RAM, CUDA 13.0+ with RTX 4070+ GPU, SSD storage

## Troubleshooting
- **"uv.lock parse error"**: Script automatically falls back to virtual environment
- **"xFormers not available"**: Warning is normal, doesn't affect functionality
- **Slow processing**: Verify GPU acceleration or use `--device cpu`
- **Out of memory**: Reduce resolution with `--resolution 720p`

## Code Quality Standards

### Refactoring Rules (Strict Requirements)
1. **Complexity Limit**: All functions MUST have McCabe complexity ≤10
   - Use helper methods to break down complex logic
   - Extract nested loops and conditionals into separate functions
   - Aim for single responsibility per function

2. **Type Hints**: Complete type annotations required on all functions
   - Parameters must have type hints
   - Return values must have type hints
   - Use typing module for complex types (List, Dict, Optional, Tuple, etc.)

3. **Error Handling**: Comprehensive try-catch blocks with graceful fallbacks
   - Catch specific exceptions where possible
   - Log errors with context
   - Provide user-friendly error messages

4. **Documentation**: Clear docstrings with parameter descriptions
   - Use triple-quoted strings for all functions
   - Document parameters, return values, and exceptions
   - Include usage examples for complex functions

5. **Code Style**: Black formatting, flake8 linting (MUST pass before committing)
   - Run `black src/ tests/` before committing (REQUIRED)
   - Run `flake8 src/ tests/` and fix all violations
   - Line length: 88 characters (Black default)
   - All code must be formatted with Black - CI will reject unformatted code

6. **Testing Requirements**: Comprehensive unit tests required for all new code
   - Write unit tests for all new functions and classes
   - Maintain minimum 80% code coverage
   - All tests must pass before committing: `pytest tests/ -v`
   - Use mocking for external dependencies (models, APIs, file I/O)
   - Test edge cases and error conditions

### Testing Workflow

Before committing any code changes:
```bash
# 1. Format code with Black (REQUIRED)
.venv/bin/black src/ tests/

# 2. Run linting
.venv/bin/flake8 src/ tests/

# 3. Run all tests
.venv/bin/pytest tests/ -v

# 4. Commit only if all checks pass
git add -A
git commit -m "your message"
```

### CI/CD Pipeline

All pushes and pull requests automatically trigger GitHub Actions:
- **Multi-environment testing**: Ubuntu + macOS × Python 3.8-3.11 (8 configurations)
- **Code quality checks**: Black formatting, flake8 linting, mypy type checking
- **Security scanning**: Dependency vulnerability checks with safety
- **Test execution**: 80+ unit tests + integration tests
- **Coverage reporting**: Uploaded to Codecov

**Pull requests will be blocked if:**
- Code is not formatted with Black
- Tests fail on any platform/Python version
- Code quality checks fail
- Coverage drops below threshold

### Code Quality Metrics
- **Error Handling**: Comprehensive try-catch with graceful fallbacks
- **Progress Tracking**: Detailed progress callbacks with stage information
- **Memory Management**: Efficient frame processing with chunked depth map generation
- **Cross-platform**: Windows, macOS, and Linux compatibility

## Processing Pipeline (7 Steps)

Each step has resume capability that checks for existing intermediate files:

1. **Frame Extraction**: FFmpeg extracts frames, skipped if frames/ exists
2. **Depth Map Generation**: Video-Depth-Anything creates depth maps in 32-frame chunks, skipped if depth_maps/ exists (when --keep-intermediates)
3. **Disparity Conversion**: Converts depth to horizontal pixel shifts
4. **Stereo Pair Creation**: Generates left/right eye images, skipped if stereo pairs exist (when --keep-intermediates)
5. **Fisheye Distortion**: Optional lens simulation, skipped if distorted frames exist
6. **VR Frame Assembly**: Combines stereo pairs into side-by-side or over-under, skipped if vr_frames/ exists
7. **Audio Integration**: Creates final video with pre-extracted FLAC audio

## Documentation Structure

- **README.md** (concise): Quick start, key features, links to detailed docs
- **docs/INSTALLATION.md**: Detailed setup, model management, verification
- **docs/USAGE.md**: Command-line and web UI usage examples
- **docs/PARAMETERS.md**: All CLI options, stereo tuning, artifact management
- **docs/TROUBLESHOOTING.md**: Common issues, performance tips, limitations
- **docs/ARCHITECTURE.md**: Technical pipeline, processing details, code structure

## Git Workflow
The project uses conventional commits with detailed descriptions.