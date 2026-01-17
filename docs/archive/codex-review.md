# Codex Review Findings

Reviewed: 2026-01-16

## Scope
- Core processing pipeline (`VideoProcessor`, `StereoProjector`)
- Video I/O and FFmpeg usage
- Utility helpers related to video/audio handling

## Findings

### High: FFmpeg frame extraction uses an invalid CUDA flag token
**Location:** `src/depth_surge_3d/core/stereo_projector.py`

**What:** `extract_frames` builds the FFmpeg command with `"-hwaccel cuda"` as a
single argument. FFmpeg expects `-hwaccel` and `cuda` as separate arguments. In
practice, this results in `Unrecognized option 'hwaccel cuda'`, so frame
extraction fails before any processing begins.

**Why it matters:** CLI and test flows that rely on `StereoProjector` will fail
on first use, even on CUDA-capable systems.

**Suggested fix:** Split the flag into separate list items, and consider a CPU
fallback if CUDA is unavailable.

---

### High: NVENC-only output without fallback or proper option ordering
**Location:** `src/depth_surge_3d/core/stereo_projector.py`

**What:** `create_output_video` always uses `hevc_nvenc` and places `-c` before
the input image sequence. That ordering applies the codec option to the next
*input*, which can cause FFmpeg to error out. On machines without NVENC,
encoding fails outright because there is no software fallback.

**Why it matters:** CLI processing fails on CPUs or GPUs without NVENC support,
and can fail even on NVENC-capable machines due to option ordering.

**Suggested fix:** Move codec options after the input sequence and add a
software fallback (e.g. `libx264`) when NVENC is not available.

---

### Medium: CUDA-only frame extraction in the temporal pipeline
**Location:** `src/depth_surge_3d/processing/video_processor.py`

**What:** `_extract_frames` always injects `-hwaccel cuda` for FFmpeg. If CUDA
or NVDEC is not available, extraction returns an error and the pipeline halts
without retrying on CPU.

**Why it matters:** Web UI processing fails on non-NVIDIA machines or when
FFmpeg lacks CUDA support, even though CPU extraction would work.

**Suggested fix:** Detect CUDA availability and fall back to CPU decode when
`ffmpeg` fails with CUDA flags.

---

### Low: `get_video_properties` can return empty data on zero FPS
**Location:** `src/depth_surge_3d/utils/file_operations.py`

**What:** `duration` is computed as `frame_count / fps` without guarding for
`fps == 0`. Some videos return `0` FPS via OpenCV; this triggers a `ZeroDivisionError`,
and the function returns an empty dict.

**Why it matters:** Downstream code treats an empty dict as “invalid video” and
aborts processing, even when the file is readable by FFmpeg.

**Suggested fix:** Guard against zero FPS and set duration to `0` or compute
duration via FFprobe as a fallback.

