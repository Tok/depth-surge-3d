# Depth Surge 3D - Development Guide

**Version**: 0.9.0
**Purpose**: 2D to 3D VR video converter using AI depth estimation (Depth-Anything V3/V2)

---

## Quick Start

```bash
./setup.sh          # Initial setup
./run_ui.sh         # Launch web UI (http://localhost:5000)
./test.sh           # Verify installation

# CLI usage
python depth_surge_3d.py input.mp4
python depth_surge_3d.py input.mp4 --vr-resolution 16x9-1080p -s 1:00 -e 2:00
```

---

## Code Quality (CRITICAL - ALWAYS DO THIS BEFORE COMMIT)

**MANDATORY pre-commit quality gate:**

```bash
# Run ALL checks at once (RECOMMENDED - mirrors CI exactly)
./scripts/pre-commit-checks.sh          # Linux/macOS
# OR
.\scripts\pre-commit-checks.ps1         # Windows

# This runs:
# 1. Black formatting check (--check mode)
# 2. Flake8 linting (must have 0 errors)
# 3. Unit tests with coverage (must be ≥ 85%)
```

**Alternative: Run checks individually**

```bash
# 1. Format code (required, no exceptions)
black src/ tests/ app.py

# 2. Lint code (must pass with 0 errors)
flake8 src/ tests/ app.py --count --show-source --statistics

# 3. Run unit tests (MUST pass, no exceptions)
./scripts/run-unit-tests.sh              # Linux/macOS (includes coverage)
# OR
.\scripts\run-unit-tests.ps1             # Windows (includes coverage)
```

**DO NOT commit if pre-commit checks fail. No exceptions.**

**Complete coding standards**: See [CODING_GUIDE.md](CODING_GUIDE.md)
- Functional programming patterns
- Type hints (modern Python 3.10+ syntax: `dict`, `list`, `X | None`)
- Max complexity: 10 (enforced by flake8)
- Pure functions, immutability, composition
- Current coverage: 92% (minimum: 85% enforced, goal: 90%+)

**Git Commit Format:**
```
type: brief description

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

---

## Tech Stack

- **Backend**: Flask + SocketIO (threading)
- **AI Models**: Depth-Anything V3 (default, lower VRAM) / Video-Depth-Anything V2 (temporal consistency)
- **Video**: FFmpeg with CUDA acceleration (NVENC encoding, hardware decoding)
- **UI**: Bootstrap 5 dark theme with real-time progress

---

## Architecture Essentials

### Processing Pipeline (7 Steps)
1. Frame Extraction → 2. Depth Maps → 3. Load Frames → 4. Stereo Pairs → 5. Fisheye Distortion → 6. VR Assembly → 7. Audio Integration

Each step has resume capability (skips if intermediate files exist).

### Output Structure
```
output/videoname_timestamp/
├── 00_original_frames/
├── 02_depth_maps/
├── 04_left_frames/ + 04_right_frames/
├── 99_vr_frames/
├── settings.json
└── videoname_3D_side_by_side.mp4
```

### Depth Models
- **V3** (default): Lower VRAM (~50%), faster, frame-by-frame
- **V2**: Better temporal consistency, 32-frame windows, higher VRAM

### Memory Management
- Smart VRAM-based chunk sizing (auto-adjusts based on available GPU memory)
- V3: 4-24 frames/chunk based on resolution
- V2: 32-frame sliding windows with 10-frame overlap

---

## Important Rules

1. **Depth resolution**: NEVER exceed source frame resolution
2. **Type hints**: Use modern syntax (`dict` not `Dict`, `X | None` not `Optional[X]`)
3. **Complexity**: All functions ≤ 10 McCabe complexity (enforced)
4. **Black + Flake8**: ALWAYS run before commit (no exceptions)
5. **Testing**: Write unit tests for all pure functions in `utils/`

---

## Common Commands

```bash
# Pre-commit quality gate (RECOMMENDED - run before every commit)
./scripts/pre-commit-checks.sh            # Linux/macOS (all checks at once)
.\scripts\pre-commit-checks.ps1           # Windows (all checks at once)

# Development
black src/ tests/ app.py                   # Format code
flake8 src/ tests/ app.py                  # Lint code
./scripts/run-unit-tests.sh               # Unit tests with coverage (Linux/macOS)
.\scripts\run-unit-tests.ps1              # Unit tests with coverage (Windows)
pytest tests/integration -v -m integration # Integration tests (requires GPU)

# CI checks (what runs in GitHub Actions)
black --check src/ tests/ app.py           # Check formatting
flake8 src/ tests/ app.py --count --show-source --statistics
pytest tests/unit -v --cov=src/depth_surge_3d --cov-report=xml --cov-fail-under=85

# Debugging
radon cc src/depth_surge_3d/ -a -nc        # Find complex functions (>10)
vulture src/depth_surge_3d/                # Find dead code
mypy src/depth_surge_3d/ --ignore-missing-imports
```

---

## Documentation Structure

- **README.md**: Quick start, features overview
- **CODING_GUIDE.md**: Coding standards and refactoring guide (comprehensive)
- **CLAUDE.md**: Development guide (this file - keep concise!)
- **docs/INSTALLATION.md**: Detailed setup instructions
- **docs/USAGE.md**: CLI and web UI examples
- **docs/PARAMETERS.md**: All options explained
- **docs/TROUBLESHOOTING.md**: Common issues and solutions
- **docs/ARCHITECTURE.md**: Technical deep dive
- **docs/CONTRIBUTING.md**: Contribution workflow

---

## Experimental Branches

### ⚠️ `experimental/optical-flow-parked` - DO NOT MERGE
Complete optical flow implementation (2000+ lines, 35 passing tests) that is **intentionally parked** due to fundamental theoretical limitations. V2 already provides temporal consistency. Kept for reference only.

---

## Performance Benchmarks (RTX 4070 Ti SUPER)

- **V3**: ~2-3 seconds/frame
- **V2**: ~3-4 seconds/frame
- **1-minute 1080p @ 30fps**: ~2-3 hours with V3

---

## CI/CD

- **Platform**: Ubuntu (GitHub Actions)
- **Python**: 3.11
- **Jobs**: Code Quality (black, flake8), Tests (unit + integration), Coverage (Codecov)
- **Required**: All flake8 checks must pass (0 errors)

---

## Keeping This File Compact

**IMPORTANT**: Keep CLAUDE.md concise (100-150 lines target).

- Remove historical information (use git history instead)
- Focus on essential commands and current architecture
- Point to comprehensive docs (CODING_GUIDE.md, etc.) instead of duplicating
- Update when architecture/workflow changes
- Delete outdated sections immediately
- Keep code quality reminders prominent
