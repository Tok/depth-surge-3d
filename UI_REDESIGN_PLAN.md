# UI Redesign Plan - Processing Steps Visualization

## Objective
Reorganize the web UI settings panel to match the actual processing pipeline, making it intuitive to understand what gets cropped/super sampled and when.

## Current Problems
1. Settings are grouped by feature type, not processing order
2. No visual connection to the 7-step processing pipeline
3. Hard to understand when resolution changes happen
4. Cropping and distortion feel disconnected from the pipeline
5. "Processing Mode" selector still present (should be removed - no longer applicable)

## Processing Pipeline (from video_processor.py)

```
Step 1: Extract Frames        → directories/01_extracted
Step 2: Generate Depth Maps    → directories/02_depth_maps
Step 3: Load into Memory       → (internal processing)
Step 4: Generate Stereo Pairs  → directories/04_stereo_pairs (left/right)
Step 5: Apply Distortion       → directories/05_distorted
Step 6: Final Processing       → directories/06_cropped, 07_final
Step 7: Create VR Video        → output.mp4
```

## Proposed UI Structure

### Section 1: Input & Frame Extraction (Step 1)
**Icon:** 📹 Extract
**What happens:** Video is decoded, frames extracted, optionally upscaled

```
├─ Time Range Selection
│  ├─ Start Time
│  ├─ End Time
│  └─ Estimated frames display
├─ Source Processing
│  ├─ Target Resolution (720p/1080p/4K/original)
│  └─ Super Sampling (auto/none/force)
└─ Frame Rate
   └─ Target FPS (original/30/60/120)
```

**Visual indicator:** "Frame resolution after extraction: 1920×1080 @ 60fps"

### Section 2: Depth Analysis (Step 2)
**Icon:** 🧠 Depth
**What happens:** AI model analyzes frames to create depth maps

```
├─ Model Configuration
│  ├─ Model Size (vits/vitb/vitl)
│  ├─ Model Type (metric/relative)
│  └─ Processing Device (auto/cuda/cpu)
└─ Info Display
   └─ "Video-Depth-Anything processes in 32-frame chunks with temporal consistency"
```

**Visual indicator:** "Depth maps: [frame_count] × model_size"

### Section 3: Stereo Pair Generation (Step 4)
**Icon:** 👁️ Stereo
**What happens:** Creates separate left/right eye views using depth information

```
├─ Camera Parameters
│  ├─ Stereo Baseline (0.02-0.15m, default: 0.065)
│  ├─ Focal Length (500-2000px, default: 1000)
│  └─ 3D Strength Presets (Subtle/Balanced/Strong)
└─ Artifact Management Tips
   └─ "Reduce baseline to 0.035-0.05 if you see warping or artifacts"
```

**Visual indicator:** Preview showing [L][R] with disparity visualization

### Section 4: VR Lens Simulation (Step 5)
**Icon:** 🔮 Distortion
**What happens:** Applies fisheye projection for VR headset optics

```
├─ Fisheye Settings
│  ├─ ☑ Apply Fisheye Distortion
│  ├─ Projection Type (equidistant/stereographic/equisolid/orthographic)
│  └─ Field of View (75-180°, default: 180)
└─ Preview
   └─ Visual showing rectangular → fisheye transformation
```

**Visual indicator:** "FOV: 180° → hemispherical projection"

### Section 5: Final Processing (Step 6)
**Icon:** ✂️ Finalize
**What happens:** Crops, fills holes, prepares final frames

```
├─ Cropping
│  ├─ Crop Factor (0.5-1.0, default: 1.0)
│  └─ Fisheye Crop Factor (0.5-1.0, default: 0.7)
├─ Quality
│  └─ Hole Fill Quality (fast/advanced)
└─ Info
   └─ "Cropping removes edge artifacts from stereo generation"
```

**Visual indicator:** "Final frame size: 1920×1080 after 0.7× crop"

### Section 6: VR Format & Output (Step 7)
**Icon:** 📦 Output
**What happens:** Combines stereo pairs into final VR video format

```
├─ VR Format
│  ├─ Format Type (side_by_side/over_under + LR/RL variants)
│  └─ VR Resolution per Eye (auto/square/16:9/cinema/custom)
├─ Custom Resolution (if selected)
│  ├─ Width per eye
│  └─ Height per eye
├─ Output Options
│  ├─ ☑ Preserve Audio
│  └─ ☑ Keep Intermediate Files
└─ Experimental
   └─ ☐ Frame Interpolation (doubles FPS, may cause artifacts)
```

**Visual indicator:** "Final output: 3840×1080 side-by-side (1920×1080 per eye)"

## Visual Design Improvements

### Step Numbers
Each section gets a clear step number badge:
```html
<div class="pipeline-step">
  <span class="step-number">1</span>
  <h6>Input & Frame Extraction</h6>
</div>
```

### Pipeline Flow Visualization
Add a small visual pipeline at the top showing:
```
[Video] → [Frames] → [Depth] → [Stereo] → [Distort] → [Crop] → [VR Output]
   ↓         ↓          ↓          ↓           ↓          ↓          ↓
 [1080p]  [60fps]   [32-frame]  [L][R]    [fisheye]  [0.7×]   [SBS 1920×1080]
```

### Collapsible Sections
Each step can be collapsed to save space:
- Expand current step being processed
- Collapse completed steps
- Keep next step visible

### Live Updates During Processing
As processing happens, highlight the current step:
- **Step 1:** ✅ Complete (120 frames extracted)
- **Step 2:** 🔄 Processing (45/120 depth maps)
- **Step 3:** ⏸️ Waiting
- etc.

## Directory Output Organization
Match UI sections to output directories:

```
output/video_name_timestamp/
├── 01_extracted/         ← Step 1: Frame Extraction
├── 02_depth_maps/        ← Step 2: Depth Analysis
├── 03_frames_loaded/     ← Step 3: Loading (optional debug)
├── 04_stereo_pairs/      ← Step 4: Stereo Generation
│   ├── left/
│   └── right/
├── 05_distorted/         ← Step 5: VR Lens Simulation
│   ├── left/
│   └── right/
├── 06_cropped/           ← Step 6: Cropping
│   ├── left/
│   └── right/
├── 07_final/             ← Step 6: Final Processing
│   ├── left/
│   └── right/
└── video_name_final.mp4  ← Step 7: VR Video Output
```

## Changes to Remove

### Remove "Processing Mode" Selector
The serial/batch mode distinction no longer exists. Remove:
- Line 853-859 in index.html (Processing Mode selector)
- All related JavaScript handling
- All UI state management for processing mode

### Update Default Values
- Fisheye FOV: 105° → 180°
- Fisheye Crop Factor: 1.0 → 0.7

## Implementation Priority

### Phase 1: Critical Updates (do now)
1. Remove "Processing Mode" selector completely
2. Update default values (FOV, crop factor)
3. Reorganize settings into 6 step-based sections
4. Add step numbers and pipeline visualization

### Phase 2: Enhanced Visualization (later)
1. Add collapsible sections with expand/collapse
2. Live processing step highlighting
3. Resolution/size indicators for each step
4. Visual preview of transformations

### Phase 3: Polish (optional)
1. Interactive pipeline diagram
2. Hover tooltips explaining each transformation
3. Settings presets (Beginner/Advanced/Expert)
4. Recommended settings based on source video

## CSS Styling for Steps

```css
.pipeline-step {
    background: var(--bs-card-bg);
    border-left: 3px solid var(--accent-lime);
    padding: 15px;
    margin: 10px 0;
    position: relative;
}

.pipeline-step.active {
    border-left-color: var(--accent-green);
    box-shadow: 0 0 10px rgba(0, 255, 65, 0.2);
}

.pipeline-step.completed {
    border-left-color: #2a2a2a;
    opacity: 0.7;
}

.step-number {
    display: inline-block;
    width: 28px;
    height: 28px;
    line-height: 28px;
    text-align: center;
    background: var(--accent-lime);
    color: #000;
    font-weight: bold;
    border-radius: 50%;
    margin-right: 10px;
}

.step-indicator {
    font-size: 0.85rem;
    color: var(--accent-lime);
    margin-top: 5px;
    font-family: 'Courier New', monospace;
}
```

## Benefits of This Redesign

1. **Intuitive Understanding:** Users immediately see the processing order
2. **Better Mental Model:** Settings grouped by when they're applied in the pipeline
3. **Debugging Aid:** Output directories match UI sections
4. **Educational:** Shows how 2D→3D conversion actually works
5. **Progressive Disclosure:** Advanced users can expand sections, beginners see simplified view
6. **Live Feedback:** During processing, users see exactly what step is happening
7. **Artifact Prevention:** Grouping stereo + distortion + crop helps users understand artifact sources

## Example User Flow

1. User uploads video → sees "1920×1080 @ 30fps" detected
2. Step 1 shows: "Will extract as 1920×1080 @ 60fps (FPS doubled)"
3. Step 3 shows: "Stereo baseline 0.065m will create ~65px disparity"
4. Step 4 shows: "180° FOV fisheye → hemispherical projection"
5. Step 5 shows: "0.7× crop → final frames 1344×756"
6. Step 6 shows: "SBS output will be 2688×756 total"

User now understands exactly what will happen at each step!
