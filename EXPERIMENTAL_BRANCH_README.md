# Experimental Branch: Optical Flow Motion Compensation

## ⚠️ THIS BRANCH IS PARKED - DO NOT MERGE

This branch contains a fully-implemented optical flow motion compensation feature that is **intentionally not merged** into main/dev.

## What This Branch Contains

A complete implementation of optical flow-based depth map warping:
- UniMatch + RAFT optical flow estimation
- Motion compensator with scene cut detection and occlusion handling
- Comprehensive logging and metrics
- Intermediate output (flow visualizations, depth comparisons)
- Full UI integration
- 35 unit tests (all passing)

**Total work**: ~2000 lines of code across 15+ files

## Why It's Parked (Not Merged)

After implementation and analysis, we determined this approach has fundamental issues:

### 1. **Wrong Order of Operations**
- Depth inconsistencies happen **during** depth estimation
- Optical flow tries to fix them **after-the-fact** by warping
- It's addressing symptoms, not root causes

### 2. **V2 Already Solves This Problem**
- Video-Depth-Anything V2 has built-in temporal consistency (32-frame windows)
- Adding optical flow on top is redundant and may degrade quality
- We're "fixing" a problem that V2 is designed to handle natively

### 3. **Theoretical Limitations**
- Optical flow tracks 2D pixel motion (brightness constancy assumption)
- Depth changes are 3D phenomena (camera motion, object rotation, perspective changes)
- Warping frame N's depth to N+1 based on 2D motion ignores 3D geometry
- **You can't reliably reconstruct 3D from 2D motion alone**

### 4. **Error Propagation**
- If frame N has incorrect depth, warping it to frame N+1 spreads that error
- Doesn't fix the underlying estimation problem
- May accumulate errors over long sequences

### 5. **Performance Cost vs Benefit**
- **Cost**: +10-20% processing time, +2-4GB VRAM
- **Benefit**: Marginal improvement (5-15% variance reduction)
- **Better alternative**: Just use V2 instead of V3

### 6. **Limited Applicability**
Might provide minimal benefit only in very specific cases:
- V3 (frame-by-frame) with static scenes and camera jitter
- Small, slow motions where 2D flow approximates 3D motion
- **But in these cases, using V2 is still the better solution**

## What Would Actually Work

If we wanted to properly address temporal consistency:

1. **Use V2** - It's literally designed for this
2. **Optical flow as model input** - Feed flow to depth estimator during inference
3. **Depth-guided flow** - Estimate flow in 3D scene space, not 2D image space
4. **Test-time optimization** - Use flow as consistency loss during inference

These are fundamentally different architectures requiring model-level changes.

## Technical Implementation Quality

The code itself is well-implemented:
- ✅ Clean architecture with factory pattern
- ✅ Proper error handling and fallbacks
- ✅ Comprehensive test coverage (35 tests, 100% passing)
- ✅ Good logging and observability
- ✅ Proper memory management
- ✅ Full UI integration

**The implementation is solid - the concept itself is the issue.**

## Files Modified in This Branch

### New Files
- `src/depth_surge_3d/models/optical_flow_estimator.py` (373 lines)
- `src/depth_surge_3d/models/motion_compensator.py` (286 lines)
- `tests/unit/test_optical_flow_estimator.py` (240 lines)
- `tests/unit/test_motion_compensator.py` (286 lines)
- `OPTICAL_FLOW_IMPROVEMENTS.md`
- `EXPERIMENTAL_BRANCH_README.md` (this file)

### Modified Files
- `src/depth_surge_3d/core/constants.py` - Added optical flow settings
- `src/depth_surge_3d/processing/video_processor.py` - Added Step 2.5
- `depth_surge_3d.py` - Added CLI arguments
- `templates/index.html` - Added UI controls (Step 2.5)

## How to Use This Branch (If You Must)

```bash
# Switch to experimental branch
git checkout experimental/optical-flow-parked

# Install dependencies (if needed)
pip install torchvision>=0.15.0

# Run with optical flow
python depth_surge_3d.py video.mp4 \
  --enable-optical-flow \
  --optical-flow-blend 0.5 \
  --keep-intermediates

# Check output
ls output/video_name_timestamp/optical_flow/
```

See `OPTICAL_FLOW_IMPROVEMENTS.md` for detailed usage and metrics interpretation.

## Lessons Learned

1. **Theoretical validation first** - Should have more thoroughly analyzed whether post-hoc depth warping makes sense before implementing
2. **Prototype before full implementation** - Could have tested on a few frames first to validate benefit
3. **Question the premise** - "Does optical flow motion compensation make sense?" was the right question to ask earlier
4. **Sometimes the answer is "use existing features"** - V2's temporal consistency is the right solution

## Future Directions

If we want to improve temporal consistency beyond V2:
- Investigate V2 hyperparameters (window size, overlap)
- Look into depth-specific temporal filtering (not optical flow)
- Consider ensemble methods (V2 + V3 weighted average)
- Research proper video depth architectures (not post-processing hacks)

## Status

- **Branch**: `experimental/optical-flow-parked`
- **Created**: 2026-01-17
- **Status**: Complete implementation, intentionally not merged
- **Tests**: 35/35 passing
- **Decision**: Park and document, do not merge

---

This branch serves as a reference for:
1. What we tried
2. Why it doesn't work
3. What we learned
4. What would work better

It's kept for historical/educational purposes, not for production use.
