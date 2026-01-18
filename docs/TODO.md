# Depth Surge 3D - Future Improvements

## High Priority

### 🎥 Dual Model Architecture
- [x] Video-Depth-Anything V2 integration ✓ (2025-10-12)
  - Repository: https://github.com/DepthAnything/Video-Depth-Anything
  - Temporal consistency via 32-frame sliding windows with 10-frame overlap
  - Auto-download support for model weights
- [x] Depth Anything V3 integration ✓ (2026-01-17)
  - Frame-by-frame processing with ~50% less VRAM
  - Hugging Face model loading (small/base/large/giant variants)
  - Faster processing, no temporal window overhead
- [x] CLI dual model support ✓ (2026-01-17)
  - `--depth-model-version {v2,v3}` flag
  - V2 for temporal consistency, V3 for lower VRAM
  - Auto-download for both model types

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
- [x] V2 Temporal Window Tuning ✓ (2026-01-17)
  - Experimental UI sliders for window size (16-64) and overlap (4-20)
  - Configurable via settings chain (UI → StereoProjector → VideoDepthEstimator)
  - Warning about quality impact when deviating from trained default (32 frames)
  - Allows memory/quality tradeoff experimentation

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
- [x] Real-time preview while processing ✓ (2026-01-18)
  - Shows current depth maps, stereo pairs, and VR frames via websocket
  - Configurable update frequency (1-5 seconds, default: 2s)
  - Toggle on/off in UI settings
  - Bandwidth-optimized: ~25-100 KB/sec, negligible performance impact

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
- [x] AI upscaling integration ✓ (2026-01-18)
  - **Implemented: Real-ESRGAN**
    - Standalone RRDB network (vendored from ai-forever/Real-ESRGAN)
    - No external wrapper dependencies (torch incompatibility issues)
    - Models: x2, x4, x4-conservative (auto-download from GitHub releases)
    - Optional Step 6 in pipeline (after cropping, before VR assembly)
    - Dynamic UI styling: blue when enabled, gray when disabled
    - Progress tracking: per-frame updates for slow operations
    - VRAM overhead: ~2-4GB depending on model
  - **Future enhancements:**
    - BasicVSR++ for video-specific temporal consistency
    - SwinIR as deterministic alternative
    - Batch processing for faster throughput

### 📊 Analysis & Debugging
- [ ] Depth map quality metrics
  - Temporal consistency scoring
  - Edge coherence analysis
- [ ] Frame comparison tools
  - Side-by-side before/after viewer
  - Depth map visualization tools

## Documentation
- [ ] Create VR headset compatibility matrix
- [ ] Document performance benchmarks by GPU
  - V2 vs V3 performance comparison
  - Memory usage for different video lengths and resolutions
  - Frame processing speed by model variant

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
