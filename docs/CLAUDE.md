# Depth Surge 3D - Development Guide

## Project Overview
2D to 3D VR video converter using AI depth estimation (Depth-Anything V3 or Video-Depth-Anything V2) with FFmpeg processing for stereo pair generation.

**Current Version**: 0.8.1

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

**See [CODING_GUIDE.md](CODING_GUIDE.md) for comprehensive coding standards including:**
- Functional programming principles
- Type hints, documentation, and complexity limits
- Testing requirements and best practices
- Refactoring guidelines

**Quick Pre-Commit Checklist:**
```bash
black src/ tests/              # Format code (required, no exceptions)
flake8 src/ tests/             # Lint code (must pass)
pytest tests/unit -v           # Run unit tests
```

**Current Coverage: 23% → Target: 70%+**

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
- **CODING_GUIDE.md**: Coding standards and refactoring guide
- **CLAUDE.md**: Development guide (this file)
- **docs/INSTALLATION.md**: Detailed setup
- **docs/USAGE.md**: CLI and web UI examples
- **docs/PARAMETERS.md**: All options explained
- **docs/TROUBLESHOOTING.md**: Common issues
- **docs/ARCHITECTURE.md**: Technical deep dive
- **docs/CONTRIBUTING.md**: Contribution workflow and CI setup

## Important Notes

1. **Version Bump**: Update version in CLAUDE.md and package files when releasing
2. **Breaking Changes**: Document in CHANGELOG.md and migration guide
3. **Dependencies**: Lock versions in requirements.txt, test before merging
4. **VRAM Limits**: Test on 8GB GPU before recommending settings
5. **Cross-platform**: Test on Windows, macOS, Linux if changing system calls

## Recent Changes (v0.8.1)

### Quality & Stability Improvements
- **Modernized type hints**: Migrated to PEP 585 built-in generics (`dict`, `list`, `tuple`) and PEP 604 union syntax (`X | None`)
- **Fixed UI stuck at 100%**: Added socketio.sleep() to ensure completion messages are sent before thread termination
- **Fixed video file locks**: Added try/finally blocks to ensure cv2.VideoCapture.release() is always called
- **Suppressed DA3 library warnings**: Fully suppressed gsplat dependency warnings using stdout/stderr redirection
- **Reduced code complexity**: Refactored load_model() method to pass flake8 C901 complexity checks

### Testing & Coverage
- **Coverage: 76% → 89%** (exceeded 85% target by 4 percentage points!)
- Added comprehensive video_processor tests (25 tests, 51% coverage on main orchestrator)
- Added io_operations edge case tests (100% coverage)
- All 541 tests passing with 89% overall coverage

### Code Quality
- All type hints now use modern Python 3.10+ syntax
- Fixed invalid type defaults in public APIs
- Flake8 and black compliant across entire codebase

## Previous Release (v0.8.0)

- Added Depth-Anything V3 support (default)
- CUDA hardware acceleration (NVENC + hardware decoding)
- Configurable depth map resolution
- Video encoder selection (Auto/NVENC/Software)
- Favicon with distorted grid design
- Fixed depth resolution capping at source frame size
- Fixed FFmpeg encoder bug (hevc_nvenc as decoder)
- Fixed time range selection with CUDA
- Comprehensive unit tests and CI/CD

---

**For detailed historical information, see git history or archived CLAUDE.md versions.**
