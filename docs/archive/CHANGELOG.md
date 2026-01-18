# Changelog

All notable changes to Depth Surge 3D will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-01-18

### Added

#### Dual Depth Model Architecture
- **Depth Anything V3 integration** (default model)
  - Frame-by-frame processing with ~50% less VRAM usage compared to V2
  - Hugging Face model loading supporting small/base/large/giant variants
  - Faster processing without temporal window overhead
  - Auto-download support for model weights
- **Video-Depth-Anything V2 support**
  - Temporal consistency via 32-frame sliding windows with 10-frame overlap
  - Superior motion coherence for video content
  - Configurable window size (16-64 frames) and overlap (4-20 frames)
  - Experimental temporal window tuning UI sliders
- **CLI dual model selection**
  - `--depth-model-version {v2,v3}` flag for model selection
  - V2 recommended for temporal consistency, V3 for lower VRAM requirements
  - Inline model documentation links in UI

#### Real-Time Preview System
- **Live preview during processing** via WebSocket
  - Shows current depth maps, stereo pairs, and VR frames in real-time
  - Configurable update frequency (1-5 seconds, default: 2s)
  - Toggle on/off in UI settings
  - Bandwidth-optimized: ~25-100 KB/sec with negligible performance impact
  - Dynamic labels that reflect current processing stage
  - Upscale preview support showing enhanced frames

#### AI Upscaling
- **Real-ESRGAN integration** for enhanced output quality
  - Standalone RRDB network (vendored from ai-forever/Real-ESRGAN)
  - Models: x2, x4, x4-conservative with auto-download from GitHub releases
  - Positioned as Step 6.5 in pipeline (after VR cropping, before assembly)
  - Dynamic UI styling: blue when enabled, gray when disabled
  - Per-frame progress tracking for long operations
  - VRAM overhead: ~2-4GB depending on model variant
  - Auto-install in run_ui.sh script

#### Performance Optimizations
- **GPU memory optimization** via smart VRAM management
  - Created vram_manager.py for VRAM detection and monitoring
  - Implemented smart batch sizing based on available memory
  - Chunk sizes auto-adjust: 4-24 frames based on VRAM and resolution
- **Parallel frame processing**
  - Stereo pair generation parallelized with multiprocessing.Pool
  - Uses cpu_count() - 2 workers for optimal performance
- **Depth map caching system**
  - Global depth map cache implemented with depth_cache.py
  - BLAKE2b content hashing for cache key generation
  - Caches in ~/.cache/depth-surge-3d/depth_cache
  - Skips re-computing if video and depth settings unchanged
  - Significant speedup for repeated processing with same settings
- **Auto-resume detection** for interrupted processing
  - Intelligent step-level resume capability
  - Automatically detects and skips completed steps

#### Advanced Image Processing
- **Advanced hole-filling algorithms**
  - Adaptive multi-pass inpainting (fast/advanced/high quality modes)
  - Automatic radius calculation based on hole size
  - Edge-preserving bilateral filtering
  - Residual inpainting for second pass enhancement
  - Significant quality improvement for stereo pair generation

#### UI/UX Enhancements
- **Drag-and-drop video upload** to web UI
  - Modern file upload interface
  - Visual feedback during drag operations
- **Progress ETA time estimates** for better visibility
  - Real-time estimation of remaining processing time
  - Per-step time tracking
- **VR headset presets** for popular devices
  - Presets for Quest 2/3, Vive, PSVR2, and more
  - Optimized settings per device
- **Improved progress bar readability**
  - Black bold text on green background for better contrast
- **Windows launcher scripts**
  - Simplified launch process on Windows
  - Reduced complexity for Windows users

#### Developer Tools
- **Unit test runner scripts**
  - `scripts/run-unit-tests.sh` for Linux/macOS
  - `scripts/run-unit-tests.ps1` for Windows
  - Automatic virtual environment activation
  - Support for --coverage flag with pytest-cov
  - Support for --verbose flag for detailed output
  - Proper exit code propagation for CI integration
- **Comprehensive test coverage**
  - 600 unit tests (585 passing)
  - 89.45% code coverage (target: 90%)
  - 100% coverage for: vram_manager, depth_cache, batch_analysis, image_processing
  - 97% coverage for progress tracking
  - Edge case tests for multiple modules
  - Residual inpainting test coverage
  - Multiprocessing-compatible test suite

#### Documentation & Code Quality
- **Updated CLAUDE.md developer guide**
  - Condensed to be concise and focused
  - Added mandatory pre-commit unit test requirement
  - Referenced new test runner scripts
  - Clarified that all pre-commit checks MUST pass
- **TODO.md updates**
  - Documented parked optical flow experimental branch
  - Cleaned up completed work
  - Added AI upscaling research findings
  - Removed YAGNI features
- **Complexity reduction**
  - All functions meet C901 complexity requirements (≤10)
  - Refactored complex functions for maintainability

### Changed

- **Default depth model** changed from V2 to V3 for better performance
- **Processing pipeline restructured** with clean 8-step architecture
  - Step 1: Frame Extraction
  - Step 2: Depth Map Generation
  - Step 3: Load Frames
  - Step 4: Stereo Pair Creation
  - Step 5: Fisheye Distortion (optional)
  - Step 6: VR Cropping
  - Step 6.5: AI Upscaling (optional) - moved after cropping for efficiency
  - Step 7: VR Assembly
  - Step 8: Audio Integration
- **UI layout reorganization**
  - Live preview panel positioned above progress section for better visibility
  - Output directory section moved with preview
  - Progress bars at bottom of interface
  - Removed redundant model documentation text
- **Progress tracking** enhanced with preview frame updates and ETA estimates
- **Version numbering** updated to 0.9.0 across all modules
- **Cropped frames** now always saved when upscaling is enabled
- **Settings persistence** improved for UI state management

### Fixed

#### UI Issues
- **UI freeze after processing completion** (multiple fixes)
  - Eliminated modal event listeners causing race conditions
  - Reset state immediately before showing completion modal
  - Proper cleanup of preview images and labels
  - Enable process button when completing to prevent freeze
  - Properly reset UI state after completion popup closes
- **Preview label initialization**
  - Labels now properly update based on processing stage
  - Dynamic stage-based labels: "Depth Map", "Stereo Pair (Left)", "Upscaled (Left)", "VR Output"

#### Performance & Stability
- **FFmpeg extraction optimization**
  - Prevent NumPy recompilation during FFmpeg operations
  - Improved extraction performance and stability
- **Multiprocessing compatibility**
  - Fixed stereo pair test to work with parallel processing
  - Proper worker pool management

#### Code Quality
- **Black formatting compliance**
  - All code formatted to Black standards
  - Triggered CI to recheck formatting
  - Resolved all flake8 errors
  - Removed unused imports and f-string issues
- **Complexity reduction**
  - All functions now meet C901 complexity requirements (≤10)
  - Improved code maintainability

#### Progress Tracking & UI Accuracy
- **8-step progress bar architecture**
  - Updated from 7 to 8 steps to match current pipeline
  - Added separate "AI Upscaling" step (Step 6)
  - Fixed progress weights distribution (now sums to exactly 100%)
  - Progress bar now shows correct step name during ESRGAN processing
- **Adaptive ETA calculation**
  - Measures actual time-per-frame for current step (especially ESRGAN)
  - Estimates remaining time using measured step rate, not overall average
  - Fixes wildly inaccurate ETAs during slow upscaling steps (2-3x slower)
  - Blends step-based and fallback estimates for robustness
- **Upscale preview functionality**
  - Preview frames now appear regardless of keep_intermediates setting
  - Creates temporary directories for preview when intermediates disabled
  - Fixed preview not showing during AI upscaling operations

#### File Management & Paths
- **Original filename preservation**
  - Videos no longer renamed to "original_video.ext" on upload
  - Output files maintain connection to source video name
  - Added find_source_video() helper to locate videos in directories
  - Intelligently excludes processed outputs (files with _3D_ marker)
- **Absolute path resolution**
  - OUTPUT_FOLDER now uses absolute paths to prevent resolution issues
  - Upload endpoint returns absolute paths via .resolve()
  - Process endpoint resolves incoming paths for consistency
  - Fixes "output dir not found" errors caused by relative path mismatches

#### Modal & UI State Management
- **Bootstrap modal backdrop cleanup**
  - Fixed modal backdrop remaining after modal close (blocking interaction)
  - Added explicit cleanup on 'hidden.bs.modal' event
  - Removes all .modal-backdrop elements and resets body state
  - Fixes entire page being stuck/unclickable after processing completion
- **Process button state management**
  - Properly disables after completion (not just hidden)
  - Re-enables only when new video is uploaded
  - Prevents clicking when no video loaded

#### Visual Design
- **Background grid gravity well effect**
  - Grid now converges towards header like general relativity spacetime diagram
  - Transform origin set above viewport (-100vh) for convergence effect
  - Increased perspective (60deg rotation) for dramatic warping
  - Added gradient mask to fade grid near convergence point
  - Visualizes depth/3D theme matching the app's purpose

#### Setup & Installation
- **Setup script improvements** (PR #10 by @danrossi)
  - Fixed Windows setup.ps1 variable name bug ($modelFile vs $model_path)
  - Moved GPU torch install to pip-only path (UV handles it automatically)
  - Added UV command line documentation with WSL support
  - WSL users: export UV_LINK_MODE=copy for proper linking

#### Code Quality & CI
- **Black formatting consistency**
  - Pinned Black version to 25.1.0 in CI workflow
  - Ensures consistent formatting checks across environments
  - Resolves formatting errors that don't reproduce locally
- **Filename sanitization**
  - Upload handler now sanitizes filenames to prevent path issues
  - Removes invalid characters: <>:"/\|?*
  - Frontend validates uploadedOutputDir before processing

#### Documentation & Links
- **Depth Anything V3 repository link** corrected in documentation
- **Settings persistence** for AI upscaling and other UI state
- **Cropped frames pipeline** fixed to always save frames for upscaling

#### Code Architecture & Modularization
- **VideoProcessor refactored into specialized modules**
  - Reduced from 2002 LOC monolith to 7 focused modules (~500 LOC each)
  - Created `depth_processor.py` for depth map generation with caching
  - Created `stereo_generator.py` for stereo pair creation
  - Created `distortion_processor.py` for fisheye distortion and cropping
  - Created `frame_upscaler.py` for AI upscaling orchestration
  - Created `vr_assembler.py` for VR frame assembly
  - Created `video_encoder.py` for video encoding with FFmpeg
  - Created `pipeline_orchestrator.py` for high-level pipeline coordination
  - Improves maintainability, testability, and separation of concerns
- **Module structure reorganization**
  - Moved inference modules to `inference/depth/` and `inference/upscaling/`
  - Separated processing into `frames/`, `video/`, and `orchestration/`
  - Organized utils into `domain/`, `imaging/`, and `system/` categories
  - All imports updated to reflect new structure
- **Real-ESRGAN model checksum corrections**
  - Fixed SHA-256 checksums for x2plus and x4-conservative models
  - Models now download and verify correctly without false corruption errors
- **Console output restoration**
  - Restored colored arrow (→) step messages lost during refactor
  - Lime green arrows for step completion, blue arrows for file paths
  - Compact format: no blank lines between left/right frame pairs
  - Clean separation between pipeline steps
- **Integration test compatibility**
  - Fixed import paths in CI tests after modularization
  - All 4 integration tests now pass in CI pipeline
- **Audio pre-extraction debugging**
  - Added console output showing extraction and lookup paths
  - Added `-y` flag to ffmpeg for file overwrites
  - Improved error messages for troubleshooting
- **UI styling consistency**
  - Drag-and-drop zone recolored to lime theme (#39ff14)
  - Matches Terminal Precision theme throughout interface

#### Security & Code Quality
- **Codex security review findings addressed**
  - Fixed all security vulnerabilities identified in review
  - Improved input validation and sanitization
  - Enhanced error handling and logging
- **Test coverage improvements**
  - Increased from 78% to 91% (723 unit tests + 4 integration tests)
  - Minimum coverage enforced at 85%, goal documented at 90%
  - Created comprehensive tests for all new processor modules
  - Removed redundant test files, suppressed multiprocessing warnings
- **Automated import path checking**
  - Created test_imports.py to detect orphaned import paths
  - Prevents import errors after refactoring
  - Scans all Python files for old import patterns

### Performance

- **V3 processing speed**: ~2-3 seconds/frame (RTX 4070 Ti SUPER)
- **V2 processing speed**: ~3-4 seconds/frame (RTX 4070 Ti SUPER)
- **1-minute 1080p @ 30fps**: ~2-3 hours with V3 on modern GPU
- **Memory usage**: 50% reduction with V3 compared to V2
- **Cache effectiveness**: Near-instant depth map retrieval for cached content

### Technical Debt

- **Type hints**: Modern Python 3.10+ syntax throughout (`dict` not `Dict`, `X | None` not `Optional[X]`)
- **Code complexity**: All functions ≤ 10 McCabe complexity (enforced by flake8)
- **Functional programming**: Emphasis on pure functions, immutability, and composition

## [0.8.1] - 2025-XX-XX

### Initial Release
- Basic 2D to 3D VR conversion using Video-Depth-Anything V2
- Web UI with Flask and SocketIO
- CUDA hardware acceleration
- Multiple VR formats (side-by-side, over-under)
- Resume capability
- Audio preservation

---

## Release Planning

### v0.9.1 (Upcoming)
- Performance regression tests
- VR headset-specific presets (Quest 2/3, Vive, PSVR2)
- Additional export format optimizations

### Future Releases
- VR headset-specific presets (Quest 2/3, Vive, PSVR2)
- Custom depth model selection
- ML-based hole filling
- Additional VR-compatible export formats
- 360° video support
- Depth map quality metrics
- Frame comparison tools

---

[0.9.0]: https://github.com/Tok/depth-surge-3d/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/Tok/depth-surge-3d/releases/tag/v0.8.1
