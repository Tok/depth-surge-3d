# Parameters Reference

## Command Line Options

### Basic Options

- **`input_video`**: Path to input video file
- **`-o, --output`**: Output directory (default: ./output)
- **`-m, --model`**: Path to Video-Depth-Anything model file (auto-downloads if missing)
- **`-f, --format`**: VR format - 'side_by_side' or 'over_under' (default: side_by_side)
- **`--device`**: Processing device - 'cpu', 'cuda', or 'auto' (default: auto)

### Time Range Options

- **`-s, --start`**: Start time in mm:ss or hh:mm:ss format (e.g., 01:30 or 00:01:30)
- **`-e, --end`**: End time in mm:ss or hh:mm:ss format (e.g., 03:45 or 00:03:45)

### Resolution Options

- **`--vr-resolution`**: Per-eye resolution: a preset, `custom:WIDTHxHEIGHT`, or `auto` (default). Auto picks a square preset when fisheye is enabled (VR180 eyes are square) and a preset matching the source aspect for flat 3D (`--no-distortion`). If the content's aspect differs from the chosen resolution, it's fitted with black bars, never stretched.
  - **Square**: square-480, square-720, square-1k, square-2k, square-3k, square-4k, square-5k
  - **16:9**: 16x9-480p, 16x9-720p, 16x9-1080p, 16x9-1440p, 16x9-4k, 16x9-5k, 16x9-8k
  - **Legacy Wide**: wide-2k, wide-4k, ultrawide
  - **Cinema**: cinema-2k, cinema-4k (ultra-wide, recommend over-under)
  - **Custom**: custom:WIDTHxHEIGHT (e.g., custom:1920x1080)

- **`--resolution`**: Minimum resolution - '720p', '1080p', '4k', or 'original' (default: 1080p)
- **`--fps`**: Target framerate for output video (default: 60)

### Stereo Parameters

- **`-b, --baseline`**: Stereo baseline distance (default: 0.065 - average human IPD)
  - Range: 0.02m (subtle) to 0.15m (very strong)
  - Larger baseline = stronger 3D effect

- **`-fl, --focal-length`**: Virtual focal length (default: 1000)
  - Range: 500px (wide-angle) to 2000px (telephoto)
  - Higher focal length = objects appear closer

### Fisheye/Distortion Options

- **`--fisheye-projection`**: Projection model (default: stereographic)
  - Options: stereographic, equidistant, equisolid, orthographic

- **`--fisheye-fov`**: Field of view in degrees (default: 105, range: 75-180)

- **`--crop-factor`**: Center crop factor (default: 1.0 - no crop)
  - Range: 0.5 to 1.0
  - Lower values = more aggressive cropping

### Quality Options

- **`--hole-fill-quality`**: How revealed background next to near objects is filled - 'fast', 'advanced' or 'high' (default: fast)
  - **fast**: ~2-4 seconds per frame, basic gap filling
  - **advanced**: ~8-15 seconds per frame, sophisticated depth-guided filling

### Output Options

- **`--no-audio`**: Do not preserve audio from original video (audio preserved by default)
- **`--no-intermediates`**: Don't keep intermediate files: disk use stays at about one batch of frames plus the video (files of finished frames are deleted as the job goes)
- **`--resume DIRECTORY`**: Continue an interrupted job where it stopped (original settings and model)
- **`--assemble-partial DIRECTORY`**: Create a video from the VR frames an unfinished job has completed so far
- **`--reuse DIRECTORY`**: Run in an existing output directory with the settings of this command line; frames, depth maps and later steps are reused as far as the settings allow
- **`--video-encoder`**: `auto` (NVENC H.265 if available, else H.264), `nvenc`, `libx264`, `libx265`

### Quality Options

- **`--depth-resolution`**: Depth model input resolution: `auto` (matches the source), `518`, `720`, `1080`, `1440`, `2160`
- **`--super-sample`**: Resolution the stereo views are rendered at: `auto`, `none`, `2x`, `1080p`, `4k`

### Processing Mode

- **`--mode`**: `batched` (default) runs steps 2-7 batch by batch, so finished VR frames appear early; `sequential` finishes each step for the whole video first
- **`--batch-size`**: Frames per batch, 10-2000 (default: 100). Progress is saved after every batch in both modes

## Stereo Parameters Deep Dive

### Baseline and Focal Length Relationship

The **baseline** (distance between virtual cameras) and **focal length** work together to control 3D depth strength:

**Tuning Strategy**:
```bash
# Subtle 3D for comfortable viewing
--baseline 0.04 --focal-length 800

# Strong 3D for dramatic effect
--baseline 0.10 --focal-length 1200

# Balanced (default)
--baseline 0.065 --focal-length 1000
```

### Managing Artifacts and "Invented Pixels"

**Common Artifacts**:
- **Stretching/warping**: Objects appear distorted at depth boundaries
- **Floating pixels**: Disconnected visual elements
- **Edge artifacts**: Jagged or broken object boundaries
- **"Invented pixels"**: AI fills gaps with estimated content that may not match reality

**Artifact Reduction Strategies**:

1. **Reduce 3D Strength** (most effective):
   ```bash
   # Conservative settings for clean results
   --baseline 0.035 --focal-length 700
   ```

2. **Adjust Hole Filling**:
   - `--hole-fill-quality fast`: Fills revealed areas by mirroring the neighbouring background
     (deterministic, so it doesn't flicker between frames)
   - `--hole-fill-quality advanced` (or `high`): Same, plus softened seams around the filled areas

3. **Crop More Aggressively**:
   ```bash
   # Remove problematic edges
   --crop-factor 0.8
   ```

**Trade-offs to Understand**:
- **Strong 3D vs. Clean Image**: More dramatic depth = more artifacts
- **Hole Filling Quality vs. Performance**: Advanced only adds seam softening and costs little extra
- **Edge Preservation vs. 3D Effect**: Keeping original content vs. creating convincing stereo

**When to Reduce 3D Strength**:
- Content with many depth discontinuities (trees, hair, complex objects)
- Fast camera movement or quick subject motion
- Scenes with reflective surfaces or transparent objects
- When artifacts are more distracting than the 3D effect is beneficial

**Remember**: Subtle, clean 3D is often more enjoyable than aggressive 3D with artifacts. Start conservative and increase strength only if the content handles it well.

## VR Format Selection

### Format Recommendations

- **Ultra-wide content (>2.2:1)**: Cinema formats with **over-under** layout recommended
- **Wide content (>1.6:1)**: Wide formats, consider **over-under** for better preservation
- **Standard content**: Square formats work best with **side-by-side**

### Resolution Options by Aspect Ratio

**Square formats** (1:1 aspect ratio - optimized for VR headsets):
- square-480 → 480×480 per eye (quick testing)
- square-720 → 720×720 per eye (preview)
- square-1k → 1024×1024 per eye (good quality)
- square-2k → 2048×2048 per eye (high quality)
- square-3k → 3072×3072 per eye (very high quality)
- square-4k → 4096×4096 per eye (ultra quality)
- square-5k → 5120×5120 per eye (maximum quality)

**16:9 Standard Formats**:
- 16x9-480p → 854×480 per eye (quick testing)
- 16x9-720p → 1280×720 per eye (HD)
- 16x9-1080p → 1920×1080 per eye (Full HD)
- 16x9-1440p → 2560×1440 per eye (QHD)
- 16x9-4k → 3840×2160 per eye (Ultra HD)
- 16x9-5k → 5120×2880 per eye (5K)
- 16x9-8k → 7680×4320 per eye (8K)

**Custom Resolutions**:
```bash
# Command Line
python depth_surge_3d.py --vr-resolution custom:1920x1080 input.mp4
python depth_surge_3d.py --vr-resolution custom:2560x1600 input.mp4  # Custom aspect ratio
```

### Auto-Detection

The system automatically detects your content's aspect ratio and recommends the best format and resolution combination based on:
- Source video aspect ratio
- Content type (wide, ultra-wide, standard)
- VR format selected (side-by-side or over-under)

## Performance vs. Quality Trade-offs

**Hole Filling Quality Impact**:
- **Fast**: mirrors background texture into revealed areas (tens of ms per frame)
- **Advanced / high**: additionally softens the seams of filled areas (slightly slower)

**Optimization Recommendations**:
```bash
# Fast processing for testing
--hole-fill-quality fast --vr-resolution 16x9-720p

# Balanced quality/speed
--hole-fill-quality fast --vr-resolution 16x9-1080p

# Maximum quality (slow)
--hole-fill-quality advanced --vr-resolution 16x9-4k
```
