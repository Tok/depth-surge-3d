"""
Rough disk space estimate for a conversion.

The pipeline works on PNG frames, which need far more space than the videos involved.
The sizes per pixel were measured on a 1080p camera video converted to 1536 px square
eyes; other content can differ (noisy footage compresses worse, flat or animated footage
better). The estimate is meant to tell "needs 200 GB" from "needs 20 GB", not to be exact.
"""

from __future__ import annotations

from typing import Any

from ..imaging.image_processing import calculate_super_sample_size

# Measured compressed size per pixel, by kind of image
SOURCE_FRAME_BYTES_PER_PIXEL = 1.1  # extracted frames (camera noise at native resolution)
DEPTH_BYTES_PER_PIXEL = 1.0  # 16-bit grayscale depth maps
STEREO_BYTES_PER_PIXEL = 0.65  # left/right views (smoother: rendered above source resolution)
DISTORTED_BYTES_PER_PIXEL = 0.35  # fisheye views (black outside the image circle)
EYE_BYTES_PER_PIXEL = 0.65  # cropped / upscaled eyes and assembled VR frames
VIDEO_BYTES_PER_PIXEL = 0.01  # encoded video (hardware encoding is well below this)
_UPSCALE_FACTORS = {"x2": 2, "x4": 4, "x4-conservative": 4}


def estimate_disk_usage(
    num_frames: int, source_width: int, source_height: int, settings: dict[str, Any]
) -> dict[str, Any]:
    """
    PURE: Estimate the peak disk space a conversion needs.

    With intermediates kept, every step's frames stay on disk. Without them the extracted
    frames are the largest part (deleted as the job goes), plus about one batch of
    everything else and the video.

    Args:
        num_frames: Frames to convert (the selected time range)
        source_width: Source video width
        source_height: Source video height
        settings: Processing settings; needs per_eye_width/per_eye_height and
            vr_output_width/vr_output_height

    Returns:
        Dict with ``total`` (peak bytes), ``final_video`` (bytes of the result) and
        ``breakdown`` (bytes per kind of file, at the peak)
    """
    keep = bool(settings.get("keep_intermediates", True))
    batch = max(1, int(settings.get("batch_size") or 100))
    # Frames on disk at once: all of them, or roughly two batches (one being read, one written)
    live = num_frames if keep else min(num_frames, 2 * batch)

    source_px = source_width * source_height
    eye_w = int(settings.get("per_eye_width") or source_width)
    eye_h = int(settings.get("per_eye_height") or source_height)
    eye_px = eye_w * eye_h
    vr_px = int(settings.get("vr_output_width") or eye_w * 2) * int(
        settings.get("vr_output_height") or eye_h
    )
    stereo_w, stereo_h = calculate_super_sample_size(
        source_width, source_height, eye_w, eye_h, str(settings.get("super_sample", "auto"))
    )
    stereo_px = stereo_w * stereo_h
    upscale = _UPSCALE_FACTORS.get(str(settings.get("upscale_model", "none")), 0)

    breakdown = {
        # all frames are extracted before anything else runs, so they peak in full
        "original_frames": num_frames * source_px * SOURCE_FRAME_BYTES_PER_PIXEL,
        "depth_maps": live * source_px * DEPTH_BYTES_PER_PIXEL,
        "vr_frames": live * vr_px * EYE_BYTES_PER_PIXEL,
        # segments while converting plus the final file
        "video": 2 * num_frames * vr_px * VIDEO_BYTES_PER_PIXEL,
    }
    if keep:
        breakdown["stereo_frames"] = 2 * num_frames * stereo_px * STEREO_BYTES_PER_PIXEL
        if settings.get("apply_distortion", True):
            breakdown["distorted_frames"] = 2 * num_frames * stereo_px * DISTORTED_BYTES_PER_PIXEL
        breakdown["cropped_frames"] = 2 * num_frames * eye_px * EYE_BYTES_PER_PIXEL
    if upscale:
        # cropped frames feed the upscaler, so they exist even without intermediates
        breakdown.setdefault("cropped_frames", 2 * live * eye_px * EYE_BYTES_PER_PIXEL)
        breakdown["upscaled_frames"] = 2 * live * eye_px * upscale**2 * EYE_BYTES_PER_PIXEL

    sizes = {name: int(size) for name, size in breakdown.items()}
    return {
        "total": sum(sizes.values()),
        "final_video": sizes["video"] // 2,
        "breakdown": sizes,
    }
