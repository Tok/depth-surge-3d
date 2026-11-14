# Changelog

All notable changes to Depth Surge 3D will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Batched processing mode** (default; `--mode batched`, `--batch-size`, "Processing Mode" in the
  web UI): steps 2-7 run for one batch of frames (default 120), then for the next, so finished VR
  frames appear early. `--mode sequential` keeps the old order (each step for the whole video).
- **Real resume**: progress is saved after every batch (`pipeline_state.json`). A resumed job
  continues after the last finished batch and produces the same output as an uninterrupted run.
  Re-running a directory with changed settings only repeats the steps those settings affect.
- **Partial videos**: `--assemble-partial DIR` / "Assemble Partial Video" encodes the VR frames an
  unfinished job has completed, with the matching part of the audio.
- **CLI progress bars**: tqdm bars per step plus overall progress and ETA, in the web UI's colours.
  In batched mode a batch counter is shown and the inner steps count frames of the current batch.
  The web UI shows the same batch counter with the looped steps nested below it.
- **Job queue** in the web UI: a job started while another is running is queued. The new Jobs
  list shows the running job, the queue and unfinished jobs, each with Resume / Assemble / Open.
- **Disk space estimate** before starting (web UI and CLI), with a warning if it exceeds the
  free space. Sizes per pixel were measured on a 1080p camera video; other content differs.
- Stopping a job in the web UI offers to create a video from the frames finished so far.
- CLI: `--depth-resolution`, `--super-sample`, `--video-encoder`, and `--reuse DIR` to run in an
  existing output directory with new settings, reusing its frames and depth maps.
- Frame extraction and video encoding report progress (both steps used to jump from 0 to 100%).
- Web UI: selected time range shows its duration and frame count; a reloaded page re-attaches to
  the running job; output paths are copied by clicking them.

### Changed
- Depth maps are written to `02_depth_maps/` as 16-bit PNGs as soon as each chunk is final
  (previously 8-bit, and only after the whole video). Memory use no longer grows with video
  length: frames and depth maps are read from disk per batch instead of being held in RAM.
- Depth is normalized with a running range (widens at once, narrows slowly) instead of one range
  computed after the whole video, which is what makes writing depth maps early possible.
- **Fused frame pipeline**: stereo generation, distortion, cropping and VR assembly run in one
  worker per frame with the images passed in memory, instead of writing and re-reading a PNG
  between every step. Output is identical. Intermediate images are written only if they are kept.
  The separate steps (now multi-threaded) are still used to redo single steps after a settings
  change, and around AI upscaling.
- **Segmented encoding**: the video is encoded in one segment per batch and the segments are
  joined (without re-encoding) at the end. Partial videos are joined from the finished segments.
- **Fisheye frame fit** (`--fisheye-fit`, "Frame Fit" slider, default 0.5): the square fisheye
  view used to be filled by the frame's height, cutting about 44% of a 16:9 video's width. The
  view can now be zoomed out: 0 fills the view (old behaviour), 1 shows the whole frame with
  black above and below. The default shows 78% of a 16:9 frame's width. The web UI says what the
  setting means for the uploaded video and only warns at the extremes; a square target for a wide
  video without fisheye gets a notice about black bars.
- **Intermediate files are no longer kept by default** (`--keep-intermediates` / the checkbox
  turns them on; `--no-intermediates` is still accepted). Off now means bounded disk use: stereo/distorted/cropped images are
  never written, and original frames, depth maps and VR frames are deleted batch by batch once
  the steps that read them are done (VR frames after their segment is encoded). An interrupted
  job can still be resumed; changing settings on such a job starts over.
- **Nothing is written to `~/.cache` any more.** Real-ESRGAN weights are stored in
  `models/Real-ESRGAN/`. The global depth map cache (`--cache-info`, `--cache-clear`) is removed:
  depth maps live in the batch directory and are reused there (`--reuse`, or starting a job again
  in the same directory). `~/.cache/depth-surge-3d` from earlier versions can be deleted.

### Fixed
- **NVENC quality**: hardware-encoded videos used the encoder's default bitrate (about 2 Mbit/s,
  whatever the resolution). NVENC now encodes at constant quality (CQ 20).
- **Output frame rate**: videos were encoded at 30 fps (or the target FPS) regardless of the
  source frame rate, changing playback speed for e.g. 24 fps sources. Frames are now read at the
  source rate; a target FPS is produced on output. Frame interpolation (experimental) is applied.
- **Audio with a time range**: audio started at the beginning of the source instead of at the
  selected start time.
- FFmpeg could pick the source's video stream instead of the VR frames (explicit stream mapping).
- **Stop button**: stopping was reported as "Video processing failed"; it now pauses the job.
- **Web UI progress**: updates were throttled to one per 10 seconds, and progress stopped after a
  socket reconnect.
- `--resume` crashed on saved settings it didn't accept and loaded the default model instead of
  the job's model; `--upscale-model` was ignored by the CLI.
- Web UI: uploading a video resets the time range to the full video; headset presets no longer
  raise a script error; the depth model is unloaded after a job (VRAM was kept until the next job).

## [0.10.0] - 2026-11-06

### Fixed
- **Stereo generation crash** (#16, #17): `ValueError: operands could not be broadcast together`.
  Super sampling upscaled frames for depth estimation but not for stereo generation, so every
  depth map was 2x the size of its frame. Frames now go to the depth model at source resolution,
  and super sampling happens in the stereo step (frame and depth upscaled together, aspect
  preserved, disparity scaled to match). Depth maps of any size, e.g. from the cache, are resized
  to their frame.
- **Stereo generation hang** (#17): forked workers deadlocked on locks inherited from the
  parent's torch/CUDA/OpenCV threads. Workers now start with `spawn`, are capped at 16, run
  OpenCV single-threaded, and a crashed pool falls back to in-process generation instead of
  hanging forever.
- **Flat or garbled 3D output** (#16):
  - Disparity was computed as `1 / normalized_depth`, so the 0-valued pixel in every frame was
    shifted by thousands of pixels. Disparity is now linear in inverse depth and bounded by
    `baseline * focal_length / 1 m` (65 px with defaults); the farthest point has zero parallax.
  - Depth Anything V3 and metric models output depth (far = large) while the relative
    Video-Depth-Anything model outputs inverse depth (near = large). Both are now converted to
    normalized inverse depth.
  - The global depth cache saved 16-bit PNGs but read them back as 8-bit, so every cache hit
    returned depth ~0. Old cache entries are invalidated (cache format version 2).
- **`--vr-resolution custom:WIDTHxHEIGHT` rejected by the CLI** (#17); invalid values now get a
  clear error message.
- CLI crash `'NoneType' object has no attribute 'update_progress'` in the distortion, upscaling
  and assembly steps (same class as #14), via a no-op progress tracker for UI-less runs.
- Output filenames for custom resolutions no longer contain `:` (invalid on Windows).
- Video-Depth-Anything crashed with `No operator found for memory_efficient_attention_forward`
  when the installed xformers didn't match the installed PyTorch build. Without xformers, its naive
  attention needed ~20 GB at 720p. Missing or mismatched xformers kernels are now replaced with
  PyTorch's `scaled_dot_product_attention`.
- **Depth flicker / temporal consistency**: Video-Depth-Anything's temporal consistency was undone
  by the pipeline. Videos were cut into independent ~32-frame chunks, each with its own
  depth scale, and every frame was then normalized to 0-1 separately. On a 10 s test clip the
  largest frame-to-frame depth jump drops from 0.0148 to 0.0021 (7x) and the 99th percentile from
  0.0079 to 0.0020 (4x), with peak RAM unchanged.
  - Temporally consistent models (VDA) now get long, RAM-sized chunks; consecutive chunks overlap
    by `temporal_window_overlap` frames (the web UI slider now has an effect), are scale/shift
    aligned on the overlap and crossfaded.
  - Per-frame models (DA3) align every frame to a running reference: about 23% less frame-to-frame
    flicker on the same clip.
  - One robust normalization range (0.5th-99.5th percentile) for the whole video replaces
    per-frame min-max. Old depth cache entries are invalidated (cache format version 3).
  - Removed the VDA wrapper's own 32-frame chunking, which also skipped alignment;
    Video-Depth-Anything handles long inputs itself with constant VRAM.
- **Web UI jobs failing on longer videos**: a Flask `teardown_appcontext` hook, which runs after
  *every* request, called `pkill -f "ffmpeg.*depth-surge"`. Any browser request during a job
  killed its frame extraction, audio extraction or encoding ("received signal 15", "Video
  processing failed"). Short clips usually escaped. Process cleanup now only runs on shutdown and
  only targets ffmpeg processes started by the server.
- **Web UI not resetting between jobs**: a new job started with the previous job's final progress
  (e.g. 59.7% "Applying distortion"), the previous job's output folder stayed on screen, and the
  progress bar kept its finished/failed styling. Progress state is reset server- and client-side
  when a job starts. The old output card hides on a new upload. The broken
  `resetProgressDisplay()` (stale element IDs) was rewritten.
- `/status` returned HTTP 500 for as long as a job was running (thread object not JSON
  serializable).
- Removed the unused `uploads/` folder (uploads are stored in their batch folder). Stitching's audio
  detection now looks in the batch folder instead of `uploads/`.
- Metric Video-Depth-Anything downloads returned 404: the weights moved to the
  `depth-anything/Metric-Video-Depth-Anything-*` repos. Automatic downloads also used the
  relative-model URL for metric files, and now download atomically (no half-written models).
- Web UI resume ignored the batch's saved settings and used stale hardcoded values
  (fisheye FOV 105, crop factor 1.0); it now continues with the saved settings.
- `setup.ps1` cloned Video-Depth-Anything to the wrong folder (`video_depth_anything_repo`
  instead of `vendor/Video-Depth-Anything`), so the V2 backend couldn't load on Windows.
- A unit test re-downloaded the 1.5 GB VDA-Large model into `models/` on every test run. Unit
  tests now block real downloads, and the suite runs about twice as fast.
- **Stretched output**: with fisheye enabled (the default), `auto` resolution picked a 16:9 per-eye
  size for landscape video, and the square fisheye crop was stretched to it (~1.8x too wide).
  Auto now picks a square per-eye preset for fisheye output. Whenever content and per-eye aspect
  differ (an explicit 16:9 eye, portrait video), the image is fitted with black bars instead
  of being stretched.
- **`--no-intermediates` / "Keep intermediates" off** broke the pipeline: stereo pairs and
  distorted frames weren't written, so later steps found nothing. AI-upscaled frames weren't
  written either, so assembly silently used the non-upscaled frames. Pipeline frames are now
  always written, and the intermediate folders are removed after a successful run.
- Video-Depth-Anything's out-of-memory retry at lower resolution always failed as well: it ran
  inside the `except` block, where the traceback kept the failed attempt's GPU memory alive.
- `run_ui.sh` stopped processes by broad name patterns (`pkill -f "depth_surge_3d\.py"`,
  `"python.*app\.py"`, `"ffmpeg.*depth-surge"`). Stopping the web UI also killed CLI
  conversions in other terminals, other web UI instances, and any process whose command line
  matched. It now stops only the server it started (and that server's children). Also removed
  its unpinned `pip install flask flask-socketio`.
- Stopping the web UI printed a crash dump ("*** Aborted ... SIGTERM received ... stack trace"):
  `pycolmap` (a DA3 dependency) installs glog's crash handler, which intercepts SIGTERM.
  `run_ui.sh` now stops the server with SIGINT (graceful), and duplicate signals are ignored.
- Web UI progress went backwards when cropping started (the crop step reported a step name the
  progress tracker didn't know) and jumped over disabled steps (e.g. 18% for upscaling when it's
  off). Disabled steps now carry no weight.
- Web UI output files were named `..._auto.mp4` instead of after the resolution actually used.

### Added
- **Python 3.13 support** (#11): open3d 0.20 ships 3.13 wheels, and uv overrides
  Depth-Anything-3's `numpy<2` pin.
- Web UI "Force 1080p" / "Force 4K" super sampling options now do what they say.
- `scripts/download_models.sh` / `.ps1` (backed by `scripts/download_models.py`) download every
  supported model: Video-Depth-Anything Small/Base/Large in relative and metric versions, and
  all Depth Anything V3 variants, plus a `recommended` set for the default settings. The script
  reads model locations from the application's constants and shows install status and licences.
  The old script downloaded the Depth-Anything-V2 *image* models, which the project doesn't use.

### Changed
- **Web UI**:
  - Super sampling is now visible with the new modes (Auto, the default, matching the backend;
    None, 2x, at least 1080p, at least 4K). It was hidden and fixed at "none".
  - Hole filling options describe the new mirror fill.
  - The temporal "window size" slider was removed (it had no effect; VDA's window is fixed). The
    remaining overlap slider now controls chunk alignment.
  - The orthographic projection now uses the backend's `orthogonal` name.
  - The fisheye crop help text matches the fitted projection.
  - Saved browser settings are reset once (settings version 3) so the new defaults apply.
- **Occlusion-aware stereo view synthesis**: Eye views were made by backward warping with the
  disparity at the *target* position. That smeared and duplicated foreground edges over the
  background (ghosted edge lines, chopped corners), so there were never real holes. Views are now
  forward-warped with a z-buffer (nearer pixel wins). Revealed background is an explicit hole mask,
  filled by mirroring the background-side neighbour's texture. That's deterministic, so it doesn't
  flicker. Bands of 1 px cracks and "flying pixels" from soft depth edges are merged into a single
  hole first.
  - The old hole filler inpainted every pure-black pixel, including genuinely dark parts of the
    footage. It's no longer used for stereo.
  - `--hole-fill-quality high` previously skipped hole filling entirely; `advanced`/`high` now
    mean mirror fill plus softened seams.
- **Fisheye edge streaks removed**: The fisheye mapping sampled up to 2x beyond the source's
  top/bottom edge and clamped those samples into stretched streaks. The projection is now fitted
  so the circle kept by the fisheye crop contains only real image content. Anything outside the
  source (the corners of the square crop) is black, the standard circular-fisheye look. Content
  appears about 18% smaller at the default crop factor. Fisheye maps are computed once per video
  instead of twice per frame.
- Supported Python versions: 3.10-3.13 (3.9 is end-of-life; Python 3.14+ only warns).
- Dependencies upgraded, and `uv.lock` is now committed (it was gitignored, so every `uv sync`
  re-resolved from scratch; the local copy was also missing Depth-Anything-3): PyTorch 2.15
  (CUDA 13.2, needs NVIDIA driver 580+), numpy 2.x, OpenCV 5, open3d 0.20, Flask-SocketIO 5.7.
- Added `addict`: Depth-Anything-3 imports it but doesn't declare it (it used to come with
  open3d < 0.20).
- Removed unused dependencies: transformers, eventlet, six, decord, matplotlib, imageio,
  imageio-ffmpeg.
- Dev dependencies moved from the deprecated `tool.uv.dev-dependencies` to `[dependency-groups]`.
- CI now installs from `uv.lock`, tests Python 3.10-3.13, and uses current action versions.
- `settings.json` records the actual package version instead of a hardcoded "0.9.0".

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

[Unreleased]: https://github.com/Tok/depth-surge-3d/compare/v0.10.0...HEAD
[0.10.0]: https://github.com/Tok/depth-surge-3d/compare/v0.9.2...v0.10.0
[0.9.2]: https://github.com/Tok/depth-surge-3d/compare/v0.9.1...v0.9.2
