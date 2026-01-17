# Coding Standards - Depth Surge 3D

**Scope:** These standards apply to all new code and refactoring work. They prioritize:
- **Functional programming patterns** - Pure functions, immutability, composition
- **Code quality** - Type hints, documentation, complexity limits
- **Testability** - Small, testable units with comprehensive coverage

**Target Coverage: 70%+** (currently 23%)

---

## Functional Programming Principles (Python-Adapted)

### 1. Prefer Expressions Over Statements
- Use ternary operators, logical operators, and comprehensions instead of verbose if/else blocks
- Use list/dict/set comprehensions over imperative loops where readable

**Bad:**
```python
if depth_model == "v3":
    chunk_size = 24
else:
    chunk_size = 32
```

**Good:**
```python
chunk_size = 24 if depth_model == "v3" else 32
```

### 2. Small Pure Functions
- **Maximum 20 lines per function**
- **Single responsibility** - one function does one thing well
- **Side-effect free** where possible - same input always gives same output
- Return values, don't modify arguments

**Bad:**
```python
def process_frames(frames):
    # 50 lines of mixed logic
    frames['modified'] = True  # Side effect!
    return frames
```

**Good:**
```python
def calculate_depth_scale(width: int, height: int) -> float:
    """Pure calculation - no side effects."""
    megapixels = (width * height) / 1_000_000
    return 1.0 if megapixels < 2.0 else 0.5

def create_modified_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Returns new list, doesn't modify input."""
    return [frame.copy() for frame in frames]
```

### 3. Static Context for Pure Functions
- Group related pure functions in utility modules
- Use module-level functions, not classes, for stateless utilities
- Organize in: `src/depth_surge_3d/utils/` package
  - `frame_utils.py` - Frame manipulation helpers
  - `depth_utils.py` - Depth map processing
  - `video_utils.py` - Video metadata and validation
  - `transform_utils.py` - Geometric transformations

### 4. Avoid Void Functions
- Functions should return values
- Separate side effects (I/O, logging, state changes) from pure logic
- Use verbs for side-effect functions: `save_depth_map()`, `log_progress()`
- Use nouns for pure functions: `calculate_stereo_offset()`, `get_frame_dimensions()`

### 5. Use Functional List Operations
- Prefer `map()`, `filter()`, comprehensions over imperative loops
- Use `itertools` and `functools` for advanced functional patterns

**Bad:**
```python
scaled_frames = []
for frame in frames:
    if frame.shape[0] > 1080:
        scaled_frames.append(cv2.resize(frame, (1920, 1080)))
```

**Good:**
```python
scaled_frames = [
    cv2.resize(frame, (1920, 1080))
    for frame in frames
    if frame.shape[0] > 1080
]
```

### 6. Minimize and Group State
- Keep mutable state isolated and well-contained
- Use dataclasses for immutable data structures
- Make state explicit in function signatures

### 7. No Magic Numbers
- Extract all magic numbers to named constants
- Use configuration objects or dataclasses for grouped constants
- Put constants in `src/depth_surge_3d/core/constants.py`

**Bad:**
```python
if width > 1920:
    depth_resolution = 1080
```

**Good:**
```python
# In constants.py
MAX_HD_WIDTH = 1920
HD_DEPTH_RESOLUTION = 1080

# In code
if width > MAX_HD_WIDTH:
    depth_resolution = HD_DEPTH_RESOLUTION
```

### 8. Immutable by Default
- Use tuples over lists when data won't change
- Use `dataclass(frozen=True)` for immutable data structures
- Return new objects instead of modifying existing ones

**Example:**
```python
from dataclasses import dataclass, replace

@dataclass(frozen=True)
class DepthSettings:
    resolution: int
    model: str
    batch_size: int

def update_resolution(settings: DepthSettings, new_res: int) -> DepthSettings:
    return replace(settings, resolution=new_res)
```

### 9. Function Composition
- Design small functions that can be composed together
- Use `functools.partial` and pipelines
- Think in pipelines: frames → depth_maps → stereo_pairs → vr_frames

---

## Python-Specific Standards (STRICT REQUIREMENTS)

### 1. Complexity Limit (ENFORCED)
**All functions MUST have McCabe complexity ≤ 10**

- Use helper methods to break down complex logic
- Extract nested loops and conditionals into separate functions
- Aim for single responsibility per function

**Check complexity:**
```bash
radon cc src/depth_surge_3d/ -a -nc  # Show only complex functions
```

**Bad (complexity 11):**
```python
def determine_chunk_size(width, height, model):
    if model == "v3":
        if width > 3840:
            return 4
        elif width > 2560:
            return 6
        elif width > 1920:
            return 12
        else:
            return 24
    else:
        if width > 3840:
            return 2
        elif width > 2560:
            return 4
        # ... more nesting
```

**Good (complexity 5):**
```python
def _get_v3_chunk_size(width: int) -> int:
    """V3 model chunk sizes by resolution."""
    if width > 3840: return 4
    if width > 2560: return 6
    if width > 1920: return 12
    return 24

def _get_v2_chunk_size(width: int) -> int:
    """V2 model chunk sizes by resolution."""
    if width > 3840: return 2
    if width > 2560: return 4
    return 8

def determine_chunk_size(width: int, height: int, model: str) -> int:
    return _get_v3_chunk_size(width) if model == "v3" else _get_v2_chunk_size(width)
```

### 2. Type Hints (REQUIRED)
**Complete type annotations required on ALL functions**

```python
from typing import Optional
import numpy as np

def process_depth_map(
    depth_map: np.ndarray,
    target_size: tuple[int, int],
    normalize: bool = True
) -> np.ndarray:
    """Resize and normalize depth map.

    Args:
        depth_map: Input depth map (H, W)
        target_size: Target (width, height) in pixels
        normalize: Whether to normalize to 0-1 range

    Returns:
        Processed depth map of shape (target_h, target_w)
    """
    ...
```

**Use modern type hints (Python 3.9+):**
```python
# Prefer these over typing module
def get_frames(count: int) -> list[np.ndarray]:  # Not List[np.ndarray]
    return [np.zeros((1080, 1920, 3)) for _ in range(count)]

def get_audio_path(video_path: str) -> str | None:  # Not Optional[str]
    audio_file = video_path.replace('.mp4', '.flac')
    return audio_file if Path(audio_file).exists() else None
```

### 3. Error Handling (REQUIRED)
**Comprehensive try-catch blocks with graceful fallbacks**

```python
def load_depth_model(model_path: str) -> torch.nn.Module | None:
    """Load depth estimation model with error handling.

    Returns None on failure rather than crashing.
    """
    try:
        model = torch.load(model_path)
        return model
    except FileNotFoundError:
        logger.error(f"Model not found: {model_path}")
        return None
    except torch.cuda.OutOfMemoryError:
        logger.error(f"Insufficient VRAM to load model: {model_path}")
        return None
    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}")
        return None
```

**Rules:**
- Catch specific exceptions where possible
- Log errors with context
- Provide user-friendly error messages
- Never use bare `except:` - use `except Exception:` at minimum

### 4. Documentation (PRAGMATIC APPROACH)

**Three levels of documentation:**

**Level 1: Simple/Internal Functions** - Type hints + one-liner
```python
def calculate_stereo_offset(depth: float, convergence: float) -> float:
    """Compute pixel offset for stereo pair from normalized depth."""
    return depth * convergence * 100
```

**Level 2: Public API / Complex Logic** - Full docstring
```python
def create_stereo_pair(
    frame: np.ndarray,
    depth_map: np.ndarray,
    convergence: float,
    separation: float
) -> tuple[np.ndarray, np.ndarray]:
    """Generate left and right eye views from frame and depth.

    Args:
        frame: Source RGB frame (H, W, 3)
        depth_map: Normalized depth map (H, W), 0=near, 1=far
        convergence: Convergence distance (0.0-1.0)
        separation: Eye separation multiplier (typically 1.0)

    Returns:
        Tuple of (left_frame, right_frame) as numpy arrays

    Raises:
        ValueError: If frame and depth_map shapes don't match
    """
    ...
```

**Level 3: API Endpoints** - Full documentation with examples
```python
@app.route('/api/process', methods=['POST'])
def process_video_api():
    """Process 2D video to 3D VR format via REST API.

    Request JSON:
        {
            "video_path": "/path/to/video.mp4",
            "depth_model": "v3" | "v2",
            "vr_format": "side_by_side" | "over_under",
            "resolution": "1080p" | "2k" | "4k"
        }

    Response JSON:
        {
            "status": "success" | "error",
            "output_path": "/path/to/output.mp4",
            "processing_time": 1234.5,
            "frame_count": 1800
        }

    Status Codes:
        200: Success
        400: Invalid parameters
        500: Processing error

    Example:
        >>> import requests
        >>> response = requests.post(
        ...     'http://localhost:5000/api/process',
        ...     json={"video_path": "input.mp4", "depth_model": "v3"}
        ... )
        >>> response.json()
        {"status": "success", "output_path": "output/result.mp4"}
    """
    ...
```

**When to skip docstrings:**
- Function name + type hints make it obvious: `def lerp(a: float, b: float, t: float) -> float`
- Getters with clear names: `def get_frame_count(video_path: str) -> int`
- Simple wrappers: `def is_nvenc_available() -> bool`

### 5. Code Style (ENFORCED)

**Black formatting, flake8 linting (must pass)**

```bash
# Format code
black src/ tests/ --line-length 100

# Lint code
flake8 src/ tests/ --max-line-length 127 --max-complexity 10

# Type checking
mypy src/depth_surge_3d/ --ignore-missing-imports
```

**Configuration (already in place):**
- Black: line length 100
- Flake8: max line length 127, max complexity 10, ignore E203, W503
- MyPy: continue-on-error (warnings only)

---

## Code Quality Standards

### 1. No Duplicate Code (DRY Principle)
- Extract common functionality into shared utilities
- If you copy-paste, you must refactor into a shared function
- Maximum 3 lines of similar code before extraction required

**Bad:**
```python
# In file A
if video_path.endswith('.mp4') or video_path.endswith('.avi') or video_path.endswith('.mov'):
    process(video_path)

# In file B
if video_path.endswith('.mp4') or video_path.endswith('.avi') or video_path.endswith('.mov'):
    validate(video_path)
```

**Good:**
```python
# In utils/video_utils.py
SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv'}

def is_supported_video(video_path: str) -> bool:
    """Check if video format is supported."""
    return Path(video_path).suffix.lower() in SUPPORTED_VIDEO_FORMATS

# In files A and B
if is_supported_video(video_path):
    process(video_path)
```

### 2. No Dead Code
- Remove unused functions, variables, and imports
- Run `vulture` to find dead code: `vulture src/depth_surge_3d/`
- Comment out during development, delete before commit

### 3. Pure Functions First
- Separate data transformation from side effects
- Pure functions go in `utils/` modules
- Side effects go in `processing/` and `models/` modules

**Example:**
```python
# utils/depth_utils.py (pure)
def normalize_depth_map(depth: np.ndarray) -> np.ndarray:
    """Normalize depth to 0-1 range (pure function)."""
    return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

# processing/video_processor.py (orchestration with side effects)
def save_depth_maps(depths: list[np.ndarray], output_dir: Path) -> None:
    """Save depth maps to disk (side effect function)."""
    for i, depth in enumerate(depths):
        cv2.imwrite(str(output_dir / f"depth_{i:06d}.png"), depth)
```

### 4. Immutable Data Patterns
```python
# Bad - mutates input
def add_timestamp(metadata: dict) -> dict:
    metadata['timestamp'] = time.time()
    return metadata

# Good - returns new dict
def add_timestamp(metadata: dict) -> dict:
    return {**metadata, 'timestamp': time.time()}

# Best - use dataclasses
from dataclasses import dataclass, replace
from datetime import datetime

@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    timestamp: datetime | None = None

def add_timestamp(metadata: VideoMetadata) -> VideoMetadata:
    return replace(metadata, timestamp=datetime.now())
```

### 5. Explicit Dependencies
Function parameters should clearly show what data is needed:

```python
# Bad - unclear what's needed
def process(config):
    width = config['video']['resolution']['width']
    height = config['video']['resolution']['height']
    # ...

# Good - explicit parameters
def process(width: int, height: int, fps: float, depth_model: str):
    # ...
```

### 6. Descriptive Naming
- **Functions:** `calculate_aspect_ratio()`, `get_next_frame()`, `save_depth_map()`
- **Variables:** `frame_count`, `is_cuda_available`, `depth_resolution`
- **Constants:** `MAX_RESOLUTION`, `DEFAULT_FPS`, `NVENC_PRESET`
- **No abbreviations** unless widely known (fps, gpu, vram, cuda are OK)

### 7. Low Cyclomatic Complexity
**Maximum complexity: 10** (enforced by radon and flake8)

Use guard clauses to reduce nesting:

```python
# Bad - high complexity (11)
def get_encoder(use_nvenc, has_gpu, quality):
    if use_nvenc:
        if has_gpu:
            if quality == 'high':
                return 'hevc_nvenc'
            else:
                return 'h264_nvenc'
        else:
            return 'libx264'
    else:
        if quality == 'high':
            return 'libx265'
        else:
            return 'libx264'

# Good - guard clauses (complexity 4)
def get_encoder(use_nvenc: bool, has_gpu: bool, quality: str) -> str:
    """Select video encoder based on hardware and quality settings."""
    if not use_nvenc:
        return 'libx265' if quality == 'high' else 'libx264'
    if not has_gpu:
        return 'libx264'
    return 'hevc_nvenc' if quality == 'high' else 'h264_nvenc'
```

### 8. Single Responsibility
Each function should do one thing well:

```python
# Bad - multiple responsibilities
def load_and_process_and_save_frames(video_path):
    frames = load_frames(video_path)
    processed = apply_depth_estimation(frames)
    save_frames(processed)

# Good - separate responsibilities
def load_frames(video_path: str) -> list[np.ndarray]: ...
def estimate_depth(frames: list[np.ndarray]) -> list[np.ndarray]: ...
def save_frames(frames: list[np.ndarray], output_dir: Path) -> None: ...

# Compose in orchestrator
def run_depth_pipeline(video_path: str, output_dir: Path) -> None:
    frames = load_frames(video_path)
    depth_maps = estimate_depth(frames)
    save_frames(depth_maps, output_dir)
```

### 9. Early Returns (Guard Clauses)
```python
# Bad - nested
def get_depth_resolution(settings):
    if settings:
        if 'depth_resolution' in settings:
            if settings['depth_resolution']:
                return settings['depth_resolution']
            else:
                return 1080
        else:
            return 1080
    else:
        return 1080

# Good - guard clauses
def get_depth_resolution(settings: dict | None) -> int:
    if not settings:
        return 1080
    if 'depth_resolution' not in settings:
        return 1080
    return settings['depth_resolution'] or 1080
```

---

## Testing Standards

### Test Organization
```
tests/
├── unit/                    # Fast, isolated tests (no GPU)
│   ├── test_depth_utils.py
│   ├── test_frame_utils.py
│   └── test_video_utils.py
└── integration/             # Full-stack tests (requires GPU)
    ├── test_depth_pipeline.py
    └── test_video_processing.py
```

### Running Tests
```bash
# Unit tests only (fast, CI-friendly)
pytest tests/unit/ -v

# Unit tests with coverage
pytest tests/unit/ --cov=src/depth_surge_3d --cov-report=html --cov-report=term

# Integration tests (requires GPU)
pytest tests/integration/ -v -m integration

# Require minimum coverage threshold (70%)
pytest tests/unit/ --cov=src/depth_surge_3d --cov-fail-under=70
```

### Testing Best Practices
- **Write unit tests for all pure functions** in `utils/` modules
- **Integration tests for full pipelines** (frame extraction → depth → stereo → encoding)
- **Mock external dependencies** in unit tests (FFmpeg, CUDA, filesystem, network)
- **Target 70%+ coverage** for utility modules
- **Use `@pytest.mark.integration`** for tests requiring GPU
- **Separate unit/integration** to keep CI fast

**Example Unit Test:**
```python
import numpy as np
from depth_surge_3d.utils.depth_utils import normalize_depth_map

def test_normalize_depth_map_range():
    """Test depth normalization to 0-1 range."""
    depth = np.array([[10.0, 20.0], [30.0, 40.0]])
    normalized = normalize_depth_map(depth)

    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert normalized.shape == depth.shape

def test_normalize_depth_map_constant():
    """Test depth normalization with constant values."""
    depth = np.full((100, 100), 5.0)
    normalized = normalize_depth_map(depth)

    # Should handle zero range gracefully
    assert not np.isnan(normalized).any()
    assert not np.isinf(normalized).any()
```

**Example Integration Test:**
```python
import pytest
from pathlib import Path

@pytest.mark.integration
def test_full_depth_estimation_pipeline(sample_video_path):
    """Test complete depth estimation workflow."""
    from depth_surge_3d.models.video_depth_estimator_da3 import VideoDepthEstimatorDA3

    estimator = VideoDepthEstimatorDA3(model_size="small")
    estimator.load_model()

    # Process test video clip
    frames = load_test_frames(sample_video_path, max_frames=10)
    depth_maps = estimator.estimate_depth_batch(frames)

    assert len(depth_maps) == len(frames)
    assert depth_maps[0].shape == frames[0].shape[:2]
    assert depth_maps[0].dtype == np.float32
```

### Coverage Goals
- **Overall:** 70%+ (currently 23%)
- **Utils modules:** 80%+ (pure functions are easy to test)
- **Processing modules:** 60%+ (orchestration, harder to test)
- **Models:** 50%+ (heavy dependencies, integration-focused)

---

## Refactoring Priorities (Execution Order)

When refactoring a module, follow this order:

1. **Add type hints** to all functions (enables better tooling)
2. **Add minimal docstrings** (only for public APIs or non-obvious logic)
3. **Eliminate code duplication** through utility extraction
4. **Extract magic numbers** to named constants
5. **Separate pure calculations** from side effects
6. **Convert imperative loops** to functional operations (where readable)
7. **Group related pure functions** into utility modules
8. **Break down complex functions** (complexity > 10)
9. **Minimize mutable state** and make it explicit
10. **Add error handling** and input validation
11. **Remove dead code** and unused imports
12. **Format with Black** and **lint with flake8**
13. **Write unit tests** for extracted pure functions

---

## Tools and Commands

### Code Quality Checks
```bash
# Format code (required before commit)
black src/ tests/ --line-length 100

# Check formatting (CI)
black src/ tests/ --check --line-length 100

# Lint
flake8 src/ tests/ --max-line-length 127 --max-complexity 10

# Find dead code
vulture src/depth_surge_3d/

# Check complexity
radon cc src/depth_surge_3d/ -a -nc  # Show only complex functions
radon cc src/depth_surge_3d/ -a      # Show all

# Type checking
mypy src/depth_surge_3d/ --ignore-missing-imports
```

### Testing
```bash
# Unit tests only (fast)
pytest tests/unit/ -v

# With coverage report
pytest tests/unit/ -v --cov=src/depth_surge_3d --cov-report=html --cov-report=term

# Integration tests (requires GPU)
pytest tests/integration/ -v -m integration

# All tests
pytest tests/ -v

# Enforce coverage threshold
pytest tests/unit/ --cov=src/depth_surge_3d --cov-fail-under=70
```

---

## Example Refactoring

### Before: Imperative, Complex, No Types
```python
def process_video(video_path, settings):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[0] > 1920:
            frame = cv2.resize(frame, (1920, 1080))
        frames.append(frame)
    cap.release()

    depths = []
    for frame in frames:
        depth = estimate_depth(frame)
        if depth is not None:
            depths.append(depth)

    return frames, depths
```

### After: Functional, Typed, Tested
```python
from pathlib import Path
import cv2
import numpy as np

# Constants
MAX_FRAME_HEIGHT = 1920
MAX_FRAME_WIDTH = 1080

# Pure functions
def should_resize_frame(frame: np.ndarray) -> bool:
    """Check if frame exceeds maximum dimensions."""
    return frame.shape[0] > MAX_FRAME_HEIGHT

def resize_frame(frame: np.ndarray) -> np.ndarray:
    """Resize frame to standard HD resolution."""
    return cv2.resize(frame, (MAX_FRAME_WIDTH, MAX_FRAME_HEIGHT))

def process_frame(frame: np.ndarray) -> np.ndarray:
    """Resize frame if needed (pure function)."""
    return resize_frame(frame) if should_resize_frame(frame) else frame

# Side-effect functions
def extract_frames(video_path: str) -> list[np.ndarray]:
    """Extract all frames from video file.

    Args:
        video_path: Path to input video

    Returns:
        List of BGR frames as numpy arrays

    Raises:
        FileNotFoundError: If video file doesn't exist
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    frames = []
    cap = cv2.VideoCapture(video_path)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed = process_frame(frame)
            frames.append(processed)
    finally:
        cap.release()

    return frames

def estimate_depth_batch(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Estimate depth for frame batch, filtering None results."""
    return [
        depth
        for frame in frames
        if (depth := estimate_depth(frame)) is not None
    ]

# Orchestration
def process_video(
    video_path: str,
    settings: dict
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Process video: extract frames and estimate depth.

    Complexity: 3 (was 8)
    """
    frames = extract_frames(video_path)
    depths = estimate_depth_batch(frames)
    return frames, depths
```

**With tests:**
```python
# tests/unit/test_frame_processing.py
import numpy as np
from depth_surge_3d.utils.frame_utils import should_resize_frame, resize_frame

def test_should_resize_frame_large():
    frame = np.zeros((2160, 3840, 3))  # 4K
    assert should_resize_frame(frame) is True

def test_should_resize_frame_normal():
    frame = np.zeros((1080, 1920, 3))  # HD
    assert should_resize_frame(frame) is False

def test_resize_frame_dimensions():
    frame = np.zeros((2160, 3840, 3))
    resized = resize_frame(frame)
    assert resized.shape == (MAX_FRAME_HEIGHT, MAX_FRAME_WIDTH, 3)
```

---

## Anti-Patterns to Avoid

### ❌ Don't Do This:
```python
# Mixing side effects with logic
def calculate_and_save_depth(frame):
    depth = estimate_depth(frame)
    cv2.imwrite('depth.png', depth)  # Side effect!
    return depth

# Modifying inputs
def normalize_frames(frames: list):
    for i in range(len(frames)):
        frames[i] = frames[i] / 255.0  # Mutates input!
    return frames

# Unclear function purpose
def do_stuff(x, y, z):  # What does this do?
    ...

# Deep nesting
def process(frame):
    if frame is not None:
        if frame.shape[0] > 0:
            if frame.shape[1] > 0:
                if frame.dtype == np.uint8:
                    return frame
```

### ✅ Do This Instead:
```python
# Separate calculation from I/O
def calculate_depth(frame: np.ndarray) -> np.ndarray:
    """Pure depth estimation."""
    return estimate_depth(frame)

def save_depth_map(depth: np.ndarray, path: Path) -> None:
    """I/O side effect."""
    cv2.imwrite(str(path), depth)

# Return new list
def normalize_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Returns normalized copies, doesn't modify input."""
    return [frame / 255.0 for frame in frames]

# Clear function names
def calculate_aspect_ratio(width: int, height: int) -> float:
    """Computes width/height ratio."""
    return width / height

# Guard clauses
def process(frame: np.ndarray | None) -> np.ndarray | None:
    if frame is None:
        return None
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        return None
    if frame.dtype != np.uint8:
        return None
    return frame
```

---

## Commit Message Format

```
refactor: <module_name> - apply functional patterns

Changes:
- Extract pure functions: calculate_X(), transform_Y()
- Add type hints to all functions
- Reduce complexity from 15 to 8 (calculate_stereo_offset)
- Move constants to constants.py
- Add unit tests for pure functions

Complexity: 15 → 8
Coverage: 23% → 45% (+22%)
```

---

## Review Checklist

Before committing refactored code, verify:

- [ ] All functions have type hints
- [ ] Public functions have docstrings
- [ ] No function has complexity > 10 (run `radon cc`)
- [ ] No magic numbers (all extracted to constants)
- [ ] No code duplication
- [ ] Black formatting applied (`black src/ tests/`)
- [ ] Flake8 passes with no errors
- [ ] Unit tests written for pure functions
- [ ] Coverage improved (check `pytest --cov`)
- [ ] No dead code or unused imports
