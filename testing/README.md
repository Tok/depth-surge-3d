# Testing and Verification Tools

This directory contains testing and verification tools for the Stereo Projector.

## Test Generation Scripts

### `create_simple_test.py`
Creates simple geometric test patterns with known depth structure for verifying stereo disparity calculations.
- Generates shapes at different depths (background, rectangles, circles)
- Outputs PPM format compatible with FFmpeg
- Used to create predictable test cases

### `create_test_video.py`
Creates video test files with motion and known depth patterns.
- Generates moving objects at different depths
- Useful for testing motion handling and temporal consistency

## Verification Scripts

### `verify_disparity.py`
Manual disparity verification using OpenCV stereo matching.
- Analyzes specific regions in stereo pairs
- Compares expected vs actual disparity values
- Requires OpenCV for stereo correspondence

### `simple_disparity_check.py`
PIL-based disparity verification without OpenCV dependencies.
- Simple edge detection and shift measurement
- Works in environments without OpenCV
- Good for basic verification

### `analyze_stereo.py`
Comprehensive stereo pair analysis tool.
- Generates disparity maps and visualizations
- Creates summary images showing analysis results
- Validates depth-to-disparity formula correctness

## Mathematical Analysis

### `fix_stereo.py`
Mathematical analysis of stereo disparity formula issues.
- Documents the original formula problems
- Shows why 100-1000px disparities were unrealistic
- Demonstrates the corrected linear mapping approach

## Usage Examples

```bash
# Generate test pattern
python3 testing/create_simple_test.py

# Process test with stereo projector
python3 stereo_projector.py testing/simple_test.mp4 --vr-resolution square-3k

# Verify results
python3 testing/simple_disparity_check.py output --frame 1

# Full analysis (requires OpenCV)
python3 testing/analyze_stereo.py output/6_left_frames_final/frame_000001.png output/6_right_frames_final/frame_000001.png
```

## Requirements

- **Basic tools**: Python 3.8+, PIL/Pillow, NumPy
- **Advanced tools**: OpenCV (for verify_disparity.py and analyze_stereo.py)
- **FFmpeg**: For video test generation and processing

## Test Validation Results

The testing framework confirmed:
- ✅ Fisheye distortion working correctly for VR projection
- ✅ Stereo disparity linear mapping producing realistic 10-96px range
- ✅ Correct depth ordering (close objects = larger disparity)
- ✅ Square VR format optimization functional