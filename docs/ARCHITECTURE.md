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

The conversion has nine steps. Steps 2-8 work through the video in batches (120 frames by
default) and save their progress after every batch, so an interrupted run continues where it
stopped.

- **Batched mode** (default) runs steps 2-8 for one batch, then for the next: finished video
  segments appear early and an interruption loses at most one batch.
- **Sequential mode** finishes the depth maps for the whole video first.

### Step 1: Frame Extraction
- FFmpeg extracts the frames of the selected time range (hardware decoding if available)
- The web UI extracts the audio to lossless FLAC at upload

### Step 2: Depth Maps
- **Models**: Video-Depth-Anything (temporally consistent) or Depth Anything V3 (per frame)
- **Streaming**: frames are estimated in chunks. Chunks of the temporal model overlap; the
  overlapping frames are aligned in scale/shift and crossfaded. Per-frame output is aligned to a
  running reference
- **Normalization**: a running range that widens at once and narrows slowly, so depth maps can
  be written before the whole video has been seen
- **Output**: 16-bit PNG depth maps, written as soon as a chunk is final
- **Memory**: a chunk that no longer fits the GPU is retried at lower resolution (temporal
  model) or in halves (per-frame model)

### Steps 3-5 and 7: Stereo Pairs, Fisheye, Crop, VR Assembly (fused)
These run in one worker process per frame, passing the images in memory:
- **Stereo pairs**: depth is converted to disparity; pixels are splatted to each eye's view with
  a z-buffer (nearer wins), and revealed areas are filled by mirroring the background-side
  neighbour's texture
- **Fisheye distortion** (optional): stereographic, equidistant, equisolid or orthographic
  projection into a square view per eye. *Frame fit* sets how much of a wide frame the view
  shows: filled (sides cut off) to whole frame (black above and below)
- **Crop** to the per-eye view, then **VR assembly** side by side or over-under

The separate step processors do the same work with a PNG between every step. They are used to
redo single steps after a settings change (when intermediates were kept) and around upscaling.

### Step 6: AI Upscaling (optional)
- Real-ESRGAN on the cropped eyes, between the fused pass and VR assembly

### Step 8: Video Encoding
- Each batch's VR frames are encoded into a video segment (NVENC H.265 at constant quality, or
  x264/x265) at the source frame rate
- With frame interpolation the video is encoded in one pass after all batches instead

### Step 9: Final Video
- The segments are joined without re-encoding and the matching part of the audio is added

### Resume
`pipeline_state.json` records, per step, how many frames are finished and a fingerprint of the
settings that determine its output. A run in an existing directory continues from there; changed
settings restart only the steps they affect. The depth stream keeps the data needed to continue
its alignment and normalization exactly. A resumed run produces the same video as an
uninterrupted one.

### Intermediate files
Off by default: stereo, distorted and cropped images are never written, and extracted frames,
depth maps and VR frames are deleted batch by batch once the steps that read them are done.
With `--keep-intermediates` every step's images stay in the batch directory.

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
Each conversion has its own directory. Only directories the run writes to are created:

```
output/
└── videoname_timestamp/
    ├── original_video.mp4           # Source video (web UI uploads)
    ├── original_audio.flac          # Pre-extracted audio (web UI)
    ├── 00_original_frames/          # Extracted frames
    ├── 02_depth_maps/               # 16-bit depth maps
    ├── 04_left_frames/, 04_right_frames/        # with kept intermediates
    ├── 05_left_distorted/, 05_right_distorted/  # with kept intermediates and fisheye
    ├── 06_left_cropped/, 06_right_cropped/      # with kept intermediates or upscaling
    ├── 07_left_upscaled/, 07_right_upscaled/    # with upscaling
    ├── 98_video_segments/           # one encoded segment per batch
    ├── 99_vr_frames/                # assembled VR frames
    ├── pipeline_state.json          # resume state
    ├── depth_stream_state.npz       # depth alignment state for resuming
    ├── videoname_<timestamp>-settings.json
    └── videoname_3D_<format>_<resolution>.mp4   # Final 3D video
```

Without kept intermediates the numbered directories are emptied batch by batch and removed after
a successful run, leaving the video and its settings file.

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
- **Disk Space**: about 2 MB per extracted 1080p frame; about 18 MB per frame with kept intermediates (1080p source, 1536 px eyes). An estimate is shown before a job starts

### Chunked Processing
- **Depth chunks**: sized from free RAM (temporal model) or VRAM (per-frame model)
- **Automatic adjustment**: a chunk that runs out of GPU memory is retried smaller
- **Memory cleanup**: CUDA cache cleared between chunks; GPU memory stays flat across batches
- **Progress tracking**: per step, per batch and overall, in the web UI and the console

## Code Structure

### Main Components

**`depth_surge_3d.py`**: Command-line interface and argument parsing

**`app.py`**: Flask web server with SocketIO for real-time updates

**`src/depth_surge_3d/processing/video_processor.py`**: Core processing pipeline
- VideoProcessor class driving the 9-step pipeline
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
