# Depth Surge 3D - Development Guide

## Project Overview
2D to 3D VR video converter using AI depth estimation (Depth-Anything V3 or Video-Depth-Anything V2) with FFmpeg processing for stereo pair generation.

**Current Version**: 0.8.0

## Quick Reference

### Tech Stack
- **Backend**: Flask + SocketIO (threading) for real-time progress
- **AI Models**: Depth-Anything V3 (default, ~50% less VRAM) or Video-Depth-Anything V2 (temporal consistency)
- **Video**: FFmpeg with CUDA acceleration (NVENC encoding, hardware decoding)
- **UI**: Dark-themed Bootstrap 5 with live previews

### Development Commands
```bash
./setup.sh          # Initial setup
./run_ui.sh         # Launch web UI (http://localhost:5000)
./test.sh           # Verify installation

# CLI examples
python depth_surge_3d.py input.mp4
python depth_surge_3d.py input.mp4 -s 1:00 -e 2:00 --resolution 1080p
```

## Code Quality Standards (REQUIRED)

**Before committing, ALWAYS run:**
```bash
black .                    # Format code (required, no exceptions)
flake8 src/ tests/        # Lint code (must pass)
pytest tests/unit -v      # Run unit tests
```

### Formatting Rules
1. **Black**: All Python code MUST be formatted with black (line length: 100)
   - Run `black .` before every commit
   - CI will fail if code is not black-formatted

2. **Flake8**: Code MUST pass flake8 linting
   - Max line length: 127 characters
   - Max complexity: 10
   - Ignore: E203, W503 (black compatibility)

3. **Type Hints**: Add type annotations to all new functions
   - Use `from typing import` for complex types
   - Example: `def process(data: Dict[str, Any]) -> Optional[bool]:`

4. **Docstrings**: Required for all public functions
   - Use triple quotes
   - Document parameters, return values, exceptions
   - Include usage examples for complex functions

### Git Workflow
```bash
git status
git add -A
git commit -m "type: brief description

Detailed explanation if needed.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

## Key Features

### Depth Models
- **V3** (default): Lower VRAM, faster, frame-by-frame processing
- **V2**: Better temporal consistency, 32-frame sliding windows, higher VRAM

Choose V3 for: Limited VRAM (6-8GB), 4K videos, faster processing
Choose V2 for: Maximum temporal smoothness, ≥12GB VRAM

### Processing Pipeline (7 Steps)
1. Frame Extraction → 2. Depth Maps → 3. Load Frames → 4. Stereo Pairs → 5. Fisheye Distortion → 6. VR Assembly → 7. Audio Integration

Each step has resume capability (skips if intermediate files exist).

### Hardware Acceleration
- **CUDA frame decoding**: `-hwaccel cuda` for faster frame extraction
- **NVENC encoding**: H.265 hardware encoding (~10x faster, requires NVIDIA GPU)
- **Auto-fallback**: Gracefully falls back to software encoding if NVENC unavailable

### UI Settings (Web Interface)
- Depth Model: V3 (default) vs V2
- Depth Resolution: Auto (matches source) or manual (518px-4K)
- Video Encoder: Auto (tries NVENC) / NVENC / libx264 / libx265
- Model Size: Small (4GB) / Base (8GB) / Large (12GB)
- VR Format: Side-by-side / Over-under
- Time Range: Start/end timestamps

## Architecture

### Output Structure
```
output/
└── timestamp_videoname/
    ├── original_video.mp4
    ├── original_audio.flac
    ├── frames/              # Extracted frames
    ├── vr_frames/          # Final VR frames
    ├── settings.json       # Processing metadata
    └── videoname_3D_side_by_side.mp4
```

### Depth Resolution Logic
**IMPORTANT**: Depth maps should NEVER exceed source frame resolution.
- 4K video → max 2160px depth
- 1080p video → max 1080px depth
- 720p video → max 720px depth
- Auto mode: matches actual frame dimensions

### Memory Management
- DA3: Processes in small chunks (4-24 frames) based on resolution
- V2: 32-frame sliding windows with 10-frame overlap
- Chunk sizes auto-adjust based on VRAM availability

## Common Issues

### FFmpeg Errors
- **"Unknown decoder hevc_nvenc"**: Encoder used as decoder (check `-c:v` placement)
- **NVENC not available**: Auto-falls back to libx264, no action needed
- **Frame extraction fails**: Check CUDA drivers, falls back to CPU decode

### VRAM Issues
- Use smaller model: Base instead of Large
- Lower depth resolution: 720p instead of 1080p
- Use V3 instead of V2: ~50% less VRAM

### Quality Issues
- Low depth quality: Check depth resolution setting (should match source)
- Temporal jitter: Use V2 for better temporal consistency
- Artifacts: Increase depth resolution if VRAM allows

## Testing

### Unit Tests
```bash
pytest tests/unit -v --cov=src/depth_surge_3d
```

### Integration Tests
```bash
pytest tests/integration -v -m integration
```

### CI/CD
- Single Ubuntu VM with Python 3.11
- Runs: black check, flake8, mypy (continue-on-error), unit + integration tests
- Coverage uploaded to Codecov

## Performance

### Typical Processing Times (RTX 4070 Ti SUPER)
- **V3**: ~2-3 seconds/frame
- **V2**: ~3-4 seconds/frame
- **1-minute 1080p clip @ 30fps**: ~2-3 hours with V3

### Optimization Tips
1. Use V3 for better speed
2. Enable NVENC if available
3. Lower depth resolution for speed over quality
4. Process shorter time ranges for testing

## Documentation Structure

- **README.md**: Quick start, features overview
- **docs/INSTALLATION.md**: Detailed setup
- **docs/USAGE.md**: CLI and web UI examples
- **docs/PARAMETERS.md**: All options explained
- **docs/TROUBLESHOOTING.md**: Common issues
- **docs/ARCHITECTURE.md**: Technical deep dive

## Important Notes

1. **Version Bump**: Update version in CLAUDE.md and package files when releasing
2. **Breaking Changes**: Document in CHANGELOG.md and migration guide
3. **Dependencies**: Lock versions in requirements.txt, test before merging
4. **VRAM Limits**: Test on 8GB GPU before recommending settings
5. **Cross-platform**: Test on Windows, macOS, Linux if changing system calls

## Recent Changes (v0.8.0)

- Added Depth-Anything V3 support (default)
- CUDA hardware acceleration (NVENC + hardware decoding)
- Configurable depth map resolution
- Video encoder selection (Auto/NVENC/Software)
- Favicon with distorted grid design
- Suppressed gsplat warnings
- Fixed depth resolution capping at source frame size
- Fixed FFmpeg encoder bug (hevc_nvenc as decoder)
- Fixed time range selection with CUDA
- Comprehensive unit tests and CI/CD

---

**For detailed historical information, see git history or archived CLAUDE.md versions.**
