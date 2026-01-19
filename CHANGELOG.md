# Changelog

All notable changes to Depth Surge 3D will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.2] - 2026-01-19

### Fixed
- **Critical bug fix**: Added null checks for `progress_tracker` in depth_processor.py
  - Fixes crash when using CLI without web interface (#14)
  - Error: `'NoneType' object has no attribute 'update_progress'`
  - Added guards at lines 111 and 399 in `src/depth_surge_3d/processing/frames/depth_processor.py`
  - All 770 tests passing

### Added
- **Python version pinning**: Added `.python-version` file pinned to Python 3.12
  - Improves development environment consistency
  - Recommended by contributor for uv project stability
- **Windows test script**: Added `test.ps1` for PowerShell users
  - Matches functionality of `test.sh` for cross-platform consistency
  - Verifies Python dependencies, CUDA, model files, and FFmpeg

### Changed
- **UI reorganization**: Improved logical grouping of settings in web UI
  - **Step 7 (VR Assembly)**: Now contains VR Format, Headset Preset, and VR Resolution
  - **Step 8 (Video Encoding & Output)**: Focused on encoding, audio, and file management
  - Clearer separation: assembly settings vs. output encoding settings
- **Dependency configuration**: Merged PR #13 from @danrossi
  - Added `depth-anything-3` package with git source configuration to `pyproject.toml`
  - Enables proper installation via uv package manager
  - Resolves installation issues for Python 3.9-3.12 users (#11)
- **Script colors**: Updated all user-facing scripts to use exact CSS colors
  - Lime green: `#39ff14` (RGB 57, 255, 20) - matches `--accent-lime`
  - Cyan: `#00d9ff` (RGB 0, 217, 255) - matches info/cyan
  - Consistent branding across CLI scripts and web UI

### Documentation
- Reorganized project structure
  - Moved `TODO.md` to `docs/` directory for better organization
  - Archived `codex-review.md` to `docs/archive/`
  - Updated contributor documentation links
- Rewrote CONTRIBUTING.md with separate sections for human vs AI contributors
  - Human contributors: Relaxed requirements, focus on ideas over perfection
  - AI contributors: Strict requirements, points to CLAUDE.md
  - Acknowledges AI may refactor human contributions later
- Updated example_settings.json to v0.9.2 with current settings

### Contributors
- Special thanks to @danrossi for identifying and helping resolve installation issues

---

## Previous Releases

For changelog entries prior to v0.9.2, see [docs/archive/CHANGELOG.md](docs/archive/CHANGELOG.md)

[Unreleased]: https://github.com/Tok/depth-surge-3d/compare/v0.9.2...HEAD
[0.9.2]: https://github.com/Tok/depth-surge-3d/compare/v0.9.1...v0.9.2
