# Codex Review Instructions for v0.9.0

**Date**: 2026-01-18
**Version**: 0.9.0 (pre-release)
**Previous Review**: v0.8.1 (2026-01-17)

---

## Overview

This is a pre-release review for v0.9.0, which adds significant features including:
- Real-time preview system via WebSocket
- AI upscaling with Real-ESRGAN
- Improved progress tracking with adaptive ETA
- UI/UX enhancements and bug fixes
- Enhanced file management and path handling

---

## What's New Since Last Review

### Major Features (Since 2026-01-17)

1. **Real-Time Preview System**
   - WebSocket-based live preview showing depth maps, stereo pairs, and VR frames
   - Configurable update frequency (1-5s, default 2s)
   - Base64-encoded image transmission (~50-200KB per frame)
   - Toggle control in UI settings
   - **Location**: `app.py:ProgressCallback.send_preview_frame()`, `templates/index.html:handleFramePreview()`

2. **AI Upscaling Integration**
   - Real-ESRGAN with standalone RRDB network (vendored from ai-forever/Real-ESRGAN)
   - Models: x2, x4, x4-conservative with auto-download
   - Positioned as optional Step 6 in pipeline
   - Per-frame progress tracking
   - **Location**: `src/depth_surge_3d/models/upscaler.py`

3. **8-Step Pipeline Architecture**
   - Restructured from 7 to 8 steps with dedicated upscaling step
   - Progress weights: [2%, 35%, 20%, 8%, 2%, 18%, 8%, 7%] = 100%
   - **Location**: `app.py:ProgressCallback.steps`, `src/depth_surge_3d/core/constants.py:PROGRESS_STEP_WEIGHTS`

4. **Adaptive ETA Calculation**
   - Per-step time-per-frame measurement
   - Addresses ESRGAN being 2-3x slower than other steps
   - Blends step-based and fallback estimates
   - **Location**: `app.py:ProgressCallback._calculate_eta()`

5. **File Management Improvements**
   - Preserves original filenames (no more "original_video.ext")
   - Absolute path resolution for OUTPUT_FOLDER
   - Intelligent source video detection
   - **Location**: `app.py:find_source_video()`, upload/process endpoints

### Bug Fixes (2026-01-18 Session)

1. **UI Freeze After Completion**
   - Bootstrap modal backdrop not cleaning up properly
   - **Location**: `templates/index.html:handleProcessingComplete/Error()`

2. **Progress Bar Accuracy**
   - 8-step architecture matching actual pipeline
   - Correct step names during processing
   - **Location**: `app.py`, `constants.py`

3. **Upscale Preview Not Appearing**
   - Preview works regardless of keep_intermediates setting
   - **Location**: `video_processor.py:_upscale_frame_pair()`

4. **Path Resolution Issues**
   - Absolute paths prevent "output dir not found" errors
   - **Location**: `app.py:upload_video()`, `start_processing()`

5. **Setup Script Fixes** (PR #10 by @danrossi)
   - Windows script variable bug fix
   - UV installation documentation
   - GPU torch installation logic
   - **Location**: `setup.sh`, `setup.ps1`, `README.md`

---

## Review Focus Areas

### 1. Code Quality & Standards ⭐ CRITICAL
- **Black formatting**: All Python files formatted (version 25.1.0 pinned in CI)
- **Flake8 compliance**: 0 errors, all functions ≤10 complexity
- **Type hints**: Modern Python 3.10+ syntax (`dict`, `X | None`)
- **Test coverage**: 89.45% (target: 90%, gap: 14 lines)
- **Tests**: 600 unit tests, all passing

**Validate**:
```bash
black --check src/ tests/
flake8 src/ tests/ --count --show-source --statistics
pytest tests/unit -v --cov
```

### 2. Architecture & Design Patterns

**Key Architectural Changes**:
- 8-step pipeline with clean separation of concerns
- WebSocket integration for real-time updates
- Standalone RRDB network (no external wrappers)
- Path handling refactor for robustness

**Review**:
- Is the separation of concerns appropriate?
- Are dependencies manageable? (Real-ESRGAN vendored to avoid conflicts)
- Is error handling consistent throughout?
- Are there any obvious performance bottlenecks?

### 3. Security & Input Validation

**Areas to Check**:
- `app.py:upload_video()` - filename sanitization (line 806)
- `app.py:find_source_video()` - path traversal prevention
- WebSocket preview - base64 encoding, size limits
- Path resolution - absolute paths prevent traversal

**Questions**:
- Is filename sanitization sufficient?
- Are there any injection risks in paths or filenames?
- Is WebSocket preview data properly validated?

### 4. Performance & Resource Management

**New Resource-Intensive Features**:
- Real-time preview (base64 encoding + WebSocket transmission)
- Real-ESRGAN upscaling (~2-4GB VRAM overhead)
- Adaptive ETA calculation (minimal overhead)

**Validate**:
- Preview throttling (2s default) adequate?
- VRAM management for upscaling appropriate?
- No memory leaks in preview/modal cleanup?

### 5. User Experience

**UI/UX Improvements**:
- Real-time preview visibility
- Accurate progress tracking with ETA
- Background grid gravity well effect
- Modal backdrop cleanup
- Original filename preservation

**Test**:
- Does the preview provide value without cluttering UI?
- Are progress updates and ETAs accurate enough?
- Is the gravity well effect visually appealing and not distracting?
- Does the UI properly reset after completion?

### 6. Documentation Quality

**Updated Documentation**:
- `CHANGELOG.md` - Comprehensive v0.9.0 changes (227 lines)
- `README.md` - UV command line examples, WSL notes
- `docs/CLAUDE.md` - Development guide
- `docs/TODO.md` - Completed items marked
- Code comments in key areas

**Review**:
- Is CHANGELOG.md clear and complete?
- Are setup instructions accurate for all platforms?
- Is inline code documentation sufficient?

---

## Specific Code Review Targets

### High Priority

1. **`app.py:ProgressCallback.send_preview_frame()` (lines 290-340)**
   - Base64 encoding security
   - Resource cleanup
   - Error handling

2. **`src/depth_surge_3d/models/upscaler.py` (entire file)**
   - RRDB network vendored correctly
   - Model auto-download security
   - VRAM management

3. **`app.py:find_source_video()` (lines 205-223)**
   - Path traversal prevention
   - Pattern matching (_3D_ exclusion)

4. **`templates/index.html:handleProcessingComplete()` (lines 2373-2466)**
   - Modal cleanup logic
   - State management
   - Memory leak prevention

5. **`video_processor.py:_upscale_frame_pair()` (lines 1565-1614)**
   - Preview creation logic
   - Directory handling when keep_intermediates=False
   - Error handling

### Medium Priority

6. **`app.py:_calculate_eta()` (lines 343-419)**
   - ETA algorithm correctness
   - Edge cases (division by zero, negative times)

7. **`.github/workflows/ci.yml`**
   - Black version pinning (line 79)
   - Test coverage requirements

8. **Setup scripts** (`setup.sh`, `setup.ps1`)
   - Cross-platform compatibility
   - Error handling in model downloads

---

## Known Issues / Trade-offs

### Acceptable Trade-offs
1. **Preview bandwidth**: ~25-100KB/sec sustained (acceptable for local/cloud)
2. **Upscaling VRAM**: 2-4GB overhead (documented, user-configurable)
3. **Real-ESRGAN vendored**: Avoids torch compatibility issues (intentional)

### Items Deferred to v0.9.1
1. Performance regression tests
2. VR headset-specific presets
3. Additional export format optimizations

---

## Testing Checklist

**Manual Testing Required**:
- [ ] Upload video with special characters in filename
- [ ] Process with preview enabled/disabled
- [ ] Process with upscaling enabled/disabled
- [ ] Verify UI cleanup after success modal
- [ ] Verify UI cleanup after error modal
- [ ] Check ETA accuracy during ESRGAN step
- [ ] Verify original filename in output
- [ ] Test on Windows (if possible)
- [ ] Test with UV package manager (WSL)

**Automated Testing**:
- [x] All 600 unit tests passing
- [x] Black formatting check (version 25.1.0)
- [x] Flake8 linting (0 errors)
- [ ] CI passing on GitHub Actions

---

## Questions for Codex

1. **Architecture**: Is the 8-step pipeline organization clear and maintainable?
2. **Security**: Any concerns with filename sanitization or path handling?
3. **Performance**: Are preview throttling and VRAM estimates reasonable?
4. **Code Quality**: Any patterns that violate best practices?
5. **Documentation**: Is CHANGELOG.md complete and clear?
6. **Dependencies**: Vendoring Real-ESRGAN - acceptable or concerning?
7. **Error Handling**: Consistent patterns throughout new features?
8. **User Experience**: Preview + ETA improvements valuable or cluttering?

---

## Success Criteria for v0.9.0

- [ ] All tests passing (600 unit tests, 89.45% coverage)
- [ ] CI pipeline green (Black, flake8, tests)
- [ ] CHANGELOG.md comprehensive and accurate
- [ ] No critical security vulnerabilities
- [ ] Performance acceptable for target use cases
- [ ] UI/UX improvements validated
- [ ] Documentation complete and accurate
- [ ] Cross-platform compatibility verified

---

## Post-Review Actions

Based on Codex feedback:
1. Address any critical issues immediately
2. Document non-critical issues for v0.9.1
3. Update CHANGELOG if needed
4. Create GitHub release draft
5. Tag v0.9.0 when approved

---

## Contact

**Repository**: https://github.com/Tok/depth-surge-3d
**Branch**: `dev`
**Commits since v0.8.1**: 68
**Lines Changed**: +5,245 / -2,135

**Key Contributors**:
- Zirteq (primary development)
- Claude Sonnet 4.5 (co-authored)
- @danrossi (PR #10: setup scripts)
