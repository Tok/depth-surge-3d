# Technical Architecture

## Overview

Depth Surge 3D uses a multi-stage processing pipeline that combines AI-powered depth estimation with stereoscopic rendering techniques to convert monocular 2D video into 3D VR format.

## How It Works

The conversion process combines computer vision with stereoscopic rendering:

1. **AI Depth Analysis**: Video-Depth-Anything processes video frames with temporal consistency to create smooth depth maps across time, estimating the 3D structure of the scene without requiring specialized camera equipment.

2. **Stereo Pair Generation**: Using the predicted depth information, the system generates separate left and right eye images by shifting pixels based on their distance from the viewer - closer objects appear more separated, distant objects less so.

3. **VR Optimization**: The stereo pairs are processed through configurable fisheye distortion and projection models to match different VR headset optics.

4. **Format Adaptation**: Final output can be rendered in standard VR formats (side-by-side or over-under) with resolution options from preview quality to 8K.

The result is 3D video that maintains the original content's motion and timing while adding depth perception suitable for VR viewing.

## Processing Pipeline

The conversion follows a 7-step pipeline with resume capability:

### Step 1: Video Preprocessing
- **Frame Extraction**: FFmpeg extracts frames from the source video
- **Time Range Handling**: Processes only the specified time range if `-s`/`-e` flags are used
- **Optional Enhancement**: Resolution upscaling and frame interpolation
- **Audio Extraction**: Immediately extracts audio to lossless FLAC format for reuse

**Resume**: Skips if frames already exist in the output directory

### Step 2: AI Depth Analysis
- **Model**: Video-Depth-Anything neural network with temporal consistency
- **Chunked Processing**: Processes frames in 32-frame sliding windows with overlap
- **Output**: Grayscale depth maps showing estimated distance of every pixel
- **Memory Management**: Automatic batching to prevent CUDA out-of-memory errors

**Resume**: Skips if depth maps already exist (when `--keep-intermediates` is enabled)

### Step 3: Disparity Conversion
- **Depth to Disparity**: Converts depth values to horizontal pixel shifts
- **Stereo Parameters**: Uses baseline and focal length to calculate shift amounts
- **Normalization**: Ensures disparity values are appropriate for the target resolution

### Step 4: Stereo Image Generation
- **Symmetric Generation**: Creates both left and right eye images with proper disparity
- **Occlusion-aware warping**: Pixels are splatted to the eye's view with a z-buffer (nearer
  wins), so foreground edges occlude correctly instead of smearing over the background
- **Hole Filling**: Revealed (disoccluded) areas are an explicit mask; they're filled by mirroring
  the background-side neighbour's texture, deterministic and therefore stable over time
- **Quality Modes**: Fast (mirror fill) or Advanced/High (mirror fill + softened seams)

**Resume**: Skips if stereo pairs already exist (when `--keep-intermediates` is enabled)

### Step 5: VR Lens Simulation (Optional)
- **Fisheye Distortion**: Applies lens distortion based on projection model
- **Projection Models**: Stereographic, equidistant, equisolid, orthographic
- **Field of View**: Configurable FOV from 75° to 180°
- **Headset Compatibility**: Matches VR headset optical characteristics

**Resume**: Skips if distorted frames already exist (when distortion is enabled)

### Step 6: Format Assembly
- **VR Format Creation**: Combines stereo pairs into side-by-side or over-under format
- **Resolution Scaling**: Scales to target VR resolution with proper aspect ratio
- **Center Cropping**: Optional cropping to remove edge artifacts

**Resume**: Skips if VR frames already exist

### Step 7: Audio Integration
- **Audio Synchronization**: Combines VR frames with pre-extracted audio
- **Time Alignment**: Maintains perfect lip-sync with original video
- **Codec Selection**: H.264 video with AAC audio for wide compatibility
- **Quality Settings**: High-bitrate encoding for VR-quality output

## Video Enhancement Features

### Frame Interpolation
- **Algorithm**: FFmpeg's minterpolate with motion compensation
- **Target FPS**: Configurable (default 60fps for smooth VR)
- **Motion Vectors**: Estimates motion between frames for smooth interpolation
- **Trade-off**: Adds processing time but significantly improves VR comfort

### Upscaling
- **Algorithm**: Lanczos resampling (high-quality)
- **Minimum Resolution**: Enforces minimum output quality (default 1080p)
- **Aspect Ratio**: Preserves original aspect ratio
- **Smart Upscaling**: Only upscales if source is below target resolution

### Audio Preservation
- **Immediate Extraction**: Audio extracted to FLAC on upload/initial processing
- **Lossless Format**: Preserves full audio quality for final video
- **Time Range Sync**: Automatically syncs audio with processed time range
- **Reuse**: Pre-extracted audio reused in resume scenarios

## Video-Depth-Anything Integration

### Model Architecture
- **Base**: Depth Anything V2 architecture
- **Temporal Consistency**: Specialized for smooth depth across video frames
- **Window Size**: 32-frame chunks with overlap
- **Available Sizes**:
  - Small (24.8M params) - Fast, lower quality
  - Base (97.5M params) - Balanced
  - Large (335.3M params) - Best quality (default)

### Depth Estimation Process
1. **Frame Batching**: Groups frames into 32-frame windows
2. **Overlap Processing**: Windows overlap to ensure temporal smoothness
3. **Feature Extraction**: Vision Transformer (ViT) extracts image features
4. **Depth Prediction**: Predicts relative depth for each pixel
5. **Temporal Alignment**: Ensures consistent depth values across frames

### Memory Optimization
- **Chunked Processing**: Processes depth maps in small batches (default: 5 frames)
- **CUDA Management**: Explicit GPU memory cleanup between batches
- **Adaptive Batching**: Automatically reduces batch size if OOM errors occur

## Storage Architecture

### Generation-Specific Directories
Each processing session creates a self-contained timestamped directory:

```
output/
└── timestamp_videoname_timestamp/
    ├── original_video.mp4      # Source video
    ├── original_audio.flac     # Pre-extracted audio
    ├── frames/                 # Extracted frames
    ├── depth_maps/            # AI depth maps (optional)
    ├── left_frames/           # Left eye images (optional)
    ├── right_frames/          # Right eye images (optional)
    ├── vr_frames/             # Final VR frames
    ├── settings.json          # Processing parameters
    └── output.mp4             # Final 3D video
```

**Benefits**:
- **Self-contained**: All files for one conversion in one directory
- **Resume-friendly**: Original video always available for resume
- **No duplication**: Eliminates redundant uploads/ directory
- **Easy cleanup**: Delete entire directory when done

### Settings File
Each processing session saves complete metadata:
- Processing parameters (all CLI flags)
- Video properties (resolution, fps, duration)
- Timestamps (created, completed, duration)
- Status tracking (in_progress, completed, failed)
- Output information (filename, format, resolution)

## Performance Characteristics

### Processing Speed
- **GPU (RTX 4070+ class)**: ~2-4 seconds per output frame
- **GPU (Mid-range)**: ~5-10 seconds per output frame
- **CPU**: ~30-60 seconds per output frame
- **Overall**: ~10x faster with GPU acceleration

### Memory Usage
- **GPU VRAM**: ~2-4GB for depth estimation
- **System RAM**: ~2-8GB depending on resolution
- **Disk Space**: ~10-50GB for intermediate files (high-res, long clips)

### Chunked Processing
- **Default batch size**: 5 depth maps at a time
- **Automatic adjustment**: Reduces batch size on OOM errors
- **Memory cleanup**: Explicit CUDA cache clearing between batches
- **Progress tracking**: Per-frame progress updates

## Code Structure

### Main Components

**`depth_surge_3d.py`**: Command-line interface and argument parsing

**`app.py`**: Flask web server with SocketIO for real-time updates

**`src/depth_surge_3d/processing/video_processor.py`**: Core processing pipeline
- VideoProcessor class with 7-step processing
- Resume capability with step-level skipping
- Progress tracking and callbacks

**`src/depth_surge_3d/models/video_depth_estimator.py`**: Depth estimation wrapper
- VideoDepthAnything integration
- Model loading and management
- Chunked depth map generation

**`src/depth_surge_3d/processing/image_processor.py`**: Image manipulation
- Stereo pair generation
- Fisheye distortion
- Hole filling algorithms

**`src/depth_surge_3d/utils/`**: Utility modules
- Resolution calculation and validation
- VR format assembly
- Settings management

### Refactoring Standards

The codebase follows strict quality standards:
- **Complexity**: All functions maintained at ≤10 McCabe complexity
- **Type Hints**: Complete type annotations on all functions
- **Error Handling**: Comprehensive try-catch with graceful fallbacks
- **Documentation**: Clear docstrings with parameter descriptions
- **Code Style**: Black formatting, flake8 linting

## Browser Compatibility

### Web UI
- **Chrome/Edge**: Full feature support
- **Firefox**: Full feature support
- **Safari**: Basic support (some WebSocket limitations)

### Real-time Features
- **SocketIO**: WebSocket-based progress updates
- **Threading**: async_mode='threading' for concurrent processing
- **Progress Tracking**: Sub-progress bars for detailed feedback
- **Frame Previews**: Live depth map and stereo frame previews
