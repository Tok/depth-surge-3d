# Contributing to Depth Surge 3D

Thank you for your interest in contributing to Depth Surge 3D! This guide will help you set up your development environment and understand our workflow.

## Development Setup

### Initial Setup

```bash
git clone https://github.com/Tok/depth-surge-3d.git
cd depth-surge-3d
./setup.sh
```

### Code Quality Requirements

**Before every commit, you MUST run:**

```bash
black src/ tests/              # Format code (required, no exceptions)
flake8 src/ tests/             # Lint code (must pass)
pytest tests/unit -v           # Run unit tests (must pass)
```

All pull requests must pass CI checks which include:
- Black formatting (line length: 100)
- Flake8 linting (max complexity: 10, max line length: 127)
- MyPy type checking (continue-on-error)
- Unit and integration tests
- Code coverage reporting

### Running Tests

```bash
# Unit tests with coverage
pytest tests/unit -v --cov=src/depth_surge_3d --cov-report=term

# Integration tests
pytest tests/integration -v -m integration

# All tests
pytest tests/ -v
```

## CI/CD Setup

### Codecov Integration

The project uses Codecov for code coverage reporting. To enable codecov badges and reports:

1. **Get Codecov Token**:
   - Go to [codecov.io](https://codecov.io/) and sign in with GitHub
   - Add your forked repository
   - Copy the repository upload token

2. **Add Token to GitHub Secrets**:
   - Go to your GitHub repository settings
   - Navigate to **Settings → Secrets and variables → Actions**
   - Click **New repository secret**
   - Name: `CODECOV_TOKEN`
   - Value: Paste your Codecov token
   - Click **Add secret**

3. **Verify Setup**:
   - Push a commit to trigger CI
   - Check the "Upload coverage to Codecov" step in GitHub Actions
   - Verify the badge in README.md shows coverage percentage

### Workflow Triggers

CI runs automatically on:
- Pushes to `main`, `dev`, or `depth-anything-v3` branches
- Pull requests targeting `main` or `dev`

## Coding Standards

### Type Hints

All functions must have complete type annotations:

```python
from typing import Optional, List, Dict, Tuple

def process_frames(
    frames: List[np.ndarray],
    settings: Dict[str, Any],
    verbose: bool = False,
) -> Tuple[List[np.ndarray], Dict[str, float]]:
    """Process video frames with depth estimation."""
    pass
```

### Complexity Limit

All functions must have McCabe complexity ≤ 10:
- Break down complex functions into smaller helpers
- Extract nested loops and conditionals
- Prefer composition over deeply nested logic

### Error Handling

Use specific exceptions with context:

```python
try:
    result = risky_operation()
except FileNotFoundError as e:
    logger.error(f"Configuration file not found: {e}")
    raise ConfigurationError(f"Missing config: {e}") from e
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

### Documentation

All functions require docstrings:

```python
def estimate_depth(
    frames: np.ndarray,
    model_name: str = "large",
) -> np.ndarray:
    """
    Estimate depth maps from video frames.

    Args:
        frames: Input frames (shape: [N, H, W, 3], BGR format)
        model_name: Model size (small, base, large)

    Returns:
        Depth maps (shape: [N, H, W], normalized 0-1 range)

    Raises:
        RuntimeError: If model is not loaded
        ValueError: If frame format is invalid
    """
    pass
```

## Git Workflow

### Branch Strategy

- `main`: Stable releases only
- `dev`: Active development branch
- Feature branches: `feature/description`
- Bug fixes: `bugfix/description`

### Commit Messages

Use conventional commits format:

```
type(scope): brief description

Detailed explanation if needed.

Co-Authored-By: Your Name <email@example.com>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no logic changes)
- `refactor`: Code restructuring
- `perf`: Performance improvements
- `test`: Test additions/changes
- `chore`: Build process, dependencies

**Examples:**
```
feat(depth): add Depth Anything V3 support

- Implemented DA3 wrapper with improved memory efficiency
- Added configurable depth resolution UI setting
- Updated documentation with DA3 usage examples

fix(encoder): correct FFmpeg NVENC usage

Previously used hevc_nvenc as decoder, causing errors.
Now properly detects NVENC availability and uses as encoder.

Fixes #42
```

### Pull Request Process

1. Create feature branch from `dev`
2. Make changes following code quality standards
3. Run all tests and formatters locally
4. Push and create PR to `dev` (not `main`)
5. Ensure CI passes (all checks green)
6. Request review from maintainers
7. Address review feedback
8. Merge when approved

## Testing Guidelines

### Unit Tests

Test individual functions in isolation:

```python
def test_normalize_depth_maps():
    """Test depth map normalization to 0-1 range."""
    depths = np.array([[10, 20], [30, 40]])
    normalized = normalize_depths(depths)

    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert normalized.shape == depths.shape
```

### Integration Tests

Test component interactions:

```python
@pytest.mark.integration
def test_depth_estimation_pipeline():
    """Test complete depth estimation workflow."""
    frames = load_test_frames()
    estimator = VideoDepthEstimatorDA3()
    estimator.load_model()

    depth_maps = estimator.estimate_depth_batch(frames)

    assert depth_maps.shape[0] == frames.shape[0]
    assert depth_maps.dtype == np.float32
```

### Test Organization

```
tests/
├── unit/                  # Fast, isolated tests
│   ├── test_core.py
│   ├── test_models.py
│   └── test_utils.py
└── integration/          # Slower, multi-component tests
    ├── test_pipeline.py
    └── test_video_processing.py
```

## Performance Considerations

- Process frames in batches to manage memory
- Use `torch.no_grad()` for inference
- Clear GPU cache after heavy operations
- Profile before optimizing (use `cProfile` or `line_profiler`)

## Questions or Issues?

- **Bug reports**: Open an issue with minimal reproduction steps
- **Feature requests**: Open an issue with use case description
- **Questions**: Check docs/ first, then open a discussion

Thank you for contributing to Depth Surge 3D!
