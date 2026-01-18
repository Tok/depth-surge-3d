# Depth Surge 3D - Future Improvements

**Active development roadmap for v0.10.0+**

Completed tasks are archived in [docs/archive/completed-tasks.md](archive/completed-tasks.md)

---

## Future Enhancements (v0.10.0+)

The following items are planned for future releases but not critical for current functionality:

### 🎯 VR Headset Presets
- [ ] Fine-tune stereo parameters per VR headset
  - Presets for Quest 2/3, Vive, PSVR2, etc.
  - Documentation complete (see VR_HEADSET_COMPATIBILITY.md)
  - Implementation: Create preset system in StereoProjector with auto-detection
  - Benefit: One-click optimal settings per device

### 🔄 Depth Model Enhancements
- [ ] Depth-Anything V4 integration (when released)
  - Monitor for next generation improvements
  - Maintain backward compatibility with V2/V3
  - Note: MiDaS and ZoeDepth are legacy models and won't be added

### 📦 Advanced Export Formats
- [ ] Additional VR-compatible formats
  - Oculus-specific formats with metadata
  - SteamVR-optimized output profiles
  - Native Quest APK packaging
- [ ] 360° video support
  - Equirectangular projection for 360° depth estimation
  - Cubemap support for better quality
  - Benefit: Immersive 360° VR experiences

### 🧪 ML Enhancements
- [ ] ML-based hole filling
  - LaMa inpainting for large mask regions
  - Better quality than traditional CV methods
  - Trade-off: ~200MB model, slower processing
  - Note: Current CV methods already excellent for our use case
- [ ] Advanced AI upscaling options
  - BasicVSR++ for video-specific temporal consistency
  - SwinIR as deterministic alternative
  - Batch processing for faster throughput

### 📊 Analysis Tools
- [ ] Depth map quality metrics
  - Temporal consistency scoring across frames
  - Edge coherence analysis
  - Automated quality reports
- [ ] Frame comparison tools
  - Side-by-side before/after viewer
  - Depth map visualization with color maps
  - Interactive quality assessment UI

### ⚡ Performance
- [ ] Performance regression tests
  - Automated benchmarking suite
  - GPU-specific performance tracking
  - CI integration for detecting slowdowns
- [ ] tqdm dashboard integration
  - Replace custom progress tracking with tqdm.auto
  - Rich console dashboards with multiple progress bars
  - Better ETA estimates and throughput metrics
  - Benefit: Industry-standard progress visualization

---

## Notes

- Items are prioritized based on user feedback and demand
- Breaking changes require major version bump
- All enhancements must maintain ≥85% test coverage
- See [CODING_GUIDE.md](CODING_GUIDE.md) for contribution guidelines
