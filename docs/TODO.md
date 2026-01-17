# Depth Surge 3D - Future Improvements

## High Priority

### 🎥 Video Model Setup
- [x] Integrate Video-Depth-Anything model ✓ (2025-10-12)
  - Repository: https://github.com/DepthAnything/Video-Depth-Anything
  - Created VideoDepthEstimator class for temporal consistency
  - Processes frames in 32-frame sliding windows with 10-frame overlap
- [x] Remove Depth Anything V2 completely ✓ (2025-10-12)
  - Simplified to single video-only architecture
  - Removed serial/batch mode distinction
  - Deleted ~850 lines of redundant code
  - Cleaner mental model: one model, one processor, one way
- [ ] Download and configure video model weights
  - Need to download: video_depth_anything_vitl.pth (or smaller variants)
  - Run: wget https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth
  - Place in: models/Video-Depth-Anything-Large/
- [ ] Update main CLI to work with new architecture
  - Remove --processing-mode flag (no longer needed)
  - Update imports to use VideoProcessor
  - Test end-to-end with downloaded model

## Medium Priority

### 🎯 Motion & Temporal Consistency
- ~~[PARKED]~~ Optical Flow Integration
  - **Status**: Fully implemented, tested, and intentionally NOT merged
  - **Branch**: `experimental/optical-flow-parked` (do not merge)
  - **Why parked**: Fundamental theoretical limitations (see EXPERIMENTAL_BRANCH_README.md on branch)
    - Post-hoc depth warping doesn't address root causes
    - V2 already has temporal consistency (32-frame windows)
    - 2D optical flow can't properly handle 3D depth changes
    - Error propagation, not error correction
  - **Lessons learned**: Use V2 for temporal consistency, not post-processing hacks
  - **Implementation quality**: Complete with 35 passing tests, full UI, metrics, logging
  - **Conclusion**: The right answer is "use V2", not "add optical flow"
- [ ] V2 Temporal Window Tuning (better alternative to optical flow)
  - Investigate optimal window sizes for different content types
  - Test overlap percentages for best temporal consistency
  - Benchmark memory/quality tradeoffs

### ⚡ Performance Enhancements
- [x] GPU memory optimization ✓ (2026-01-17)
  - Created vram_manager.py for VRAM detection and monitoring
  - Implemented smart batch sizing based on available memory
  - Chunk sizes auto-adjust: 4-24 frames based on VRAM and resolution
- [x] Parallel frame processing ✓ (2026-01-17)
  - Stereo pair generation parallelized with multiprocessing.Pool
  - Uses cpu_count() - 2 workers for optimal performance
- [x] Cache depth maps across runs ✓ (2026-01-17)
  - Implemented depth_cache.py with BLAKE2b content hashing
  - Caches in ~/.cache/depth-surge-3d/depth_cache
  - Skips re-computing if video and depth settings unchanged

### 🎨 UI/UX Improvements
- [ ] Real-time preview while processing
  - Show current frame being processed
  - Display depth map visualization
- [ ] Progress time estimates
  - Better ETA calculation based on actual frame processing time
- [ ] Drag-and-drop video upload
- [ ] Queue multiple videos for batch processing
- [ ] Resume interrupted processing automatically

### 🎛️ Advanced Settings
- [ ] Fine-tune stereo parameters per VR headset
  - Presets for Quest 2/3, Vive, PSVR2, etc.
- [ ] Custom depth model selection
  - Support multiple depth estimation models
  - Allow model switching without reinstall
- [x] Advanced hole-filling algorithms ✓ (2026-01-17)
  - Implemented adaptive multi-pass inpainting (fast/advanced/high)
  - Automatic radius calculation based on hole size
  - Edge-preserving bilateral filtering
- [ ] ML-based hole filling (future enhancement)

## Low Priority

### 📦 Export Formats
- [ ] Additional VR-compatible formats
  - Oculus-specific formats
  - SteamVR-optimized output
- [ ] 360° video support
  - Equirectangular projection
  - Cubemap support

### 🧪 Experimental Features
- [ ] Dynamic depth adjustment
  - Auto-tune baseline/focal based on scene content
- [ ] Scene detection
  - Different 3D strength for different scene types
  - Cut detection to reset temporal models
- [ ] AI upscaling integration
  - Real-ESRGAN or similar for enhanced quality

### 📊 Analysis & Debugging
- [ ] Depth map quality metrics
  - Temporal consistency scoring
  - Edge coherence analysis
- [ ] Frame comparison tools
  - Side-by-side before/after viewer
  - Depth map visualization tools

## Documentation
- [ ] Update CLAUDE.md with simplified architecture
  - Document Video-Depth-Anything as sole depth model
  - Remove serial/batch mode confusion
  - Update model loading information
  - Clarify memory requirements (loads all frames)
- [ ] Create VR headset compatibility matrix
- [ ] Document performance benchmarks by GPU
  - Video-Depth-Anything performance metrics
  - Memory usage for different video lengths

## Technical Debt
- [ ] Clean up commented debug code
- [ ] Standardize error handling patterns
- [x] Add type hints throughout ✓ (2026-01-17)
  - Modern Python 3.10+ syntax (dict, list, X | None)
  - Full type coverage across all modules
- [x] Comprehensive test suite ✓ (2026-01-17)
  - 600 unit tests, 89.45% coverage (target: 90%, gap: 14 lines)
  - 100% coverage: vram_manager, depth_cache, batch_analysis, image_processing
  - 97% coverage: progress tracking
  - Integration tests for full pipeline
- [ ] Performance regression tests

---

## Completed ✓

### 2025-10-13 Session - Fisheye Fixes & Console Polish
- [x] Fixed fisheye distortion to properly fill rectangular VR frames (180° FOV)
  - Updated defaults: FOV 105° → 180°, fisheye_crop_factor 1.0 → 0.7
  - Fixed coordinate calculation to use clipping instead of circular masking
  - Changed border mode to BORDER_REFLECT_101 to eliminate black borders
  - Result: Clean rectangular frames [__][__] instead of circular crops [([])][([])]
- [x] Added intermediate frame saving for 06 (cropped) and 07 (final) directories
  - Fixed empty directories when keep_intermediates=true
  - All processing steps now properly saved for debugging
- [x] Implemented console color system matching UI theme
  - Created console.py utility with ANSI colors
  - Lime green (#39ff14) step completion arrows "  -> "
  - Yellow warnings, red errors matching design system
  - Removed all emojis from console output (✓, ✗, ⚠️)
- [x] Added localStorage versioning (v2) for web UI settings
  - Automatic reset of outdated cached settings
  - Prevents old defaults from overriding new improvements
- [x] Updated HTML template defaults to match Python constants

### 2025-10-12 Session - Video-Only Architecture
- [x] Integrated Video-Depth-Anything model for temporal consistency
- [x] Created VideoDepthEstimator class with temporal processing support
- [x] Completely refactored BatchProcessor into VideoProcessor
- [x] **Removed Depth Anything V2 completely** (~850 lines deleted)
  - Deleted depth_estimator.py (V2 wrapper)
  - Deleted serial mode processing
  - Removed batch/serial mode distinction
  - Removed depth_anything_v2_repo/
  - Simplified constants.py (single model config)
- [x] Updated constants.py with simplified video-only configurations
- [x] Fixed .gitignore to properly handle model code vs model weights
- [x] Simplified progress tracking (removed serial mode support)

### Previous Sessions
- [x] Fix Depth Anything V2 import issue
- [x] Test full pipeline with sample video
- [x] Update documentation to match current state
- [x] UI redesign - "Terminal Precision" aesthetic
- [x] Fix 'frames' KeyError with --no-intermediates
- [x] Workflow-guided button shininess
- [x] Dense grid background animation
