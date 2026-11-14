"""
Image processing utilities for depth estimation and stereo projection.

This module contains pure functions for image manipulation, depth processing,
and geometric transformations without side effects.
"""

from __future__ import annotations

import functools
import math

import cv2
import numpy as np
from typing import Any

from ...core.constants import MIN_DEPTH_VALUE, MAX_DEPTH_VALUE, STEREO_NEAR_PLANE_M

# Holes closer than this (px) are merged into one disocclusion region before filling
HOLE_MERGE_WIDTH = 5


def resize_image(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    interpolation: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    """
    Resize image to target dimensions.

    Args:
        image: Input image array
        target_width: Target width
        target_height: Target height
        interpolation: OpenCV interpolation method

    Returns:
        Resized image array
    """
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def fit_to_canvas(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    PURE: Scale an image into a target size without distorting it.

    If the aspect ratios match (within a pixel), this is a plain resize. Otherwise the image is
    scaled to fit and centred on a black canvas (letterbox/pillarbox) instead of being stretched.

    Args:
        image: Input image (H, W) or (H, W, C)
        target_width: Output width
        target_height: Output height

    Returns:
        Image of exactly (target_height, target_width)
    """
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    if abs(new_w - target_width) <= 1 and abs(new_h - target_height) <= 1:
        return resize_image(image, target_width, target_height)

    scaled = resize_image(image, new_w, new_h)
    canvas = np.zeros((target_height, target_width) + image.shape[2:], dtype=image.dtype)
    y0 = (target_height - new_h) // 2
    x0 = (target_width - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = scaled
    return canvas


def normalize_depth_map(depth_map: np.ndarray) -> np.ndarray:
    """
    Normalize depth map to 0-1 range.

    Args:
        depth_map: Raw depth map array

    Returns:
        Normalized depth map array
    """
    if depth_map.max() == depth_map.min():
        return np.zeros_like(depth_map)

    normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    return np.clip(normalized, MIN_DEPTH_VALUE, MAX_DEPTH_VALUE)


def depth_to_inverse_depth(depth: np.ndarray, low_percentile: float = 1.0) -> np.ndarray:
    """
    Convert a depth map (larger = farther) to inverse depth (larger = nearer).

    The pipeline works on inverse depth, which is what relative depth models such as
    Video-Depth-Anything output natively. Models that output depth (Depth Anything V3,
    metric models) must be converted first. Depths are clamped at a low percentile of the
    valid values so a few near-zero pixels can't blow up the range.

    Args:
        depth: Depth map (any positive scale); non-positive values are treated as invalid
        low_percentile: Percentile of valid depths used as the minimum depth

    Returns:
        Inverse depth map (float32, >= 0); invalid pixels become 0 (infinitely far)
    """
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.zeros_like(depth)

    min_depth = max(float(np.percentile(depth[valid], low_percentile)), 1e-6)
    inverse = np.zeros_like(depth)
    inverse[valid] = 1.0 / np.maximum(depth[valid], min_depth)
    return inverse


def depth_to_disparity(depth_map: np.ndarray, baseline: float, focal_length: float) -> np.ndarray:
    """
    Convert a normalized inverse-depth map to a pixel disparity map.

    Depth maps throughout the pipeline are normalized inverse depth ("nearness"):
    1.0 is the nearest point in the frame and 0.0 the farthest. The nearest point is
    placed at STEREO_NEAR_PLANE_M meters and the farthest at infinity (zero disparity),
    so disparity = baseline * focal_length * nearness / STEREO_NEAR_PLANE_M.

    Args:
        depth_map: Normalized inverse depth map (0-1 range, 1 = near)
        baseline: Stereo baseline in meters
        focal_length: Camera focal length in pixels

    Returns:
        Disparity map array in pixels (float32, >= 0)
    """
    nearness = np.clip(np.nan_to_num(depth_map, nan=0.0), MIN_DEPTH_VALUE, MAX_DEPTH_VALUE)
    max_disparity = (baseline * focal_length) / STEREO_NEAR_PLANE_M
    return (nearness * max_disparity).astype(np.float32)


def stereo_depth_stats(
    depth_map: np.ndarray, baseline: float, focal_length: float, max_samples: int = 100_000
) -> dict[str, float]:
    """
    PURE: How much stereo effect a normalized depth map produces.

    Args:
        depth_map: Normalized inverse depth map (0-1 range, 1 = near)
        baseline: Stereo baseline in meters
        focal_length: Camera focal length in pixels
        max_samples: Maximum number of pixels looked at

    Returns:
        Dict with ``depth_spread`` (share of the 0-1 depth range the frame uses, between
        its 2nd and 98th percentile) and ``shift_px`` (difference in disparity between
        those near and far parts, in pixels; 0 means no 3D effect)
    """
    values = np.asarray(depth_map, dtype=np.float32).reshape(-1)
    values = values[:: max(1, values.size // max_samples)]
    values = np.clip(np.nan_to_num(values, nan=0.0), MIN_DEPTH_VALUE, MAX_DEPTH_VALUE)
    if values.size == 0:
        return {"depth_spread": 0.0, "shift_px": 0.0}
    far, near = np.percentile(values, [2, 98])
    spread = float(near - far)
    return {
        "depth_spread": round(spread, 4),
        "shift_px": round(spread * baseline * focal_length / STEREO_NEAR_PLANE_M, 1),
    }


def match_depth_to_frame(depth_map: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize a depth map to the given frame size if it doesn't already match.

    Args:
        depth_map: Depth map array (H, W)
        width: Target width in pixels
        height: Target height in pixels

    Returns:
        Depth map of shape (height, width)
    """
    if depth_map.shape[:2] == (height, width):
        return depth_map
    return cv2.resize(depth_map.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)


def calculate_super_sample_size(
    frame_width: int,
    frame_height: int,
    per_eye_width: int,
    per_eye_height: int,
    super_sample: str = "auto",
) -> tuple[int, int]:
    """
    Determine the working resolution for stereo warping.

    Warping at a higher resolution than the source reduces resampling artifacts when
    the per-eye output is larger than the source. The source aspect ratio is preserved.

    Modes:
        none:  work at source resolution
        auto:  upscale so the frame covers the per-eye output, at most 2x the source
        2x:    upscale by 2x
        1080p: upscale so the shorter side is at least 1080 px
        4k:    upscale so the shorter side is at least 2160 px
        any other value is treated like "auto"; the frame is never downscaled

    Args:
        frame_width: Source frame width
        frame_height: Source frame height
        per_eye_width: Target per-eye width
        per_eye_height: Target per-eye height
        super_sample: Super sampling mode

    Returns:
        Tuple of (width, height) for stereo warping
    """
    if super_sample == "none" or frame_width <= 0 or frame_height <= 0:
        return frame_width, frame_height

    short_side_targets = {"1080p": 1080, "4k": 2160}
    if super_sample == "2x":
        scale = 2.0
    elif super_sample in short_side_targets:
        scale = max(short_side_targets[super_sample] / min(frame_width, frame_height), 1.0)
    else:
        scale = max(per_eye_width / frame_width, per_eye_height / frame_height)
        scale = min(max(scale, 1.0), 2.0)

    return int(round(frame_width * scale)), int(round(frame_height * scale))


def create_shifted_image(
    image: np.ndarray, disparity_map: np.ndarray, direction: str = "left"
) -> np.ndarray:
    """
    Create shifted image for stereo pair using disparity map.

    Args:
        image: Source image array
        disparity_map: Disparity map array
        direction: "left" or "right" for shift direction

    Returns:
        Shifted image array
    """
    height, width = image.shape[:2]
    shift_multiplier = -0.5 if direction == "left" else 0.5

    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    # Calculate shifted coordinates
    x_shifted = x_coords + (disparity_map * shift_multiplier)

    # Clip coordinates to valid range
    x_shifted = np.clip(x_shifted, 0, width - 1)

    # Create output image
    if len(image.shape) == 3:
        shifted_image = np.zeros_like(image)
        for channel in range(image.shape[2]):
            shifted_image[:, :, channel] = cv2.remap(
                image[:, :, channel],
                x_shifted.astype(np.float32),
                y_coords.astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
    else:
        shifted_image = cv2.remap(
            image,
            x_shifted.astype(np.float32),
            y_coords.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    return shifted_image


def forward_warp_disparity(disparity: np.ndarray, shift_scale: float) -> np.ndarray:
    """
    PURE: Splat a disparity map into the target view with a z-buffer.

    Every source pixel moves horizontally by ``shift_scale * disparity`` and is written to
    the two nearest target pixels (closes 1 px cracks on stretched surfaces). Where several
    pixels land on the same target, the nearest one (largest disparity) wins, which gives
    correct occlusion edges.

    Args:
        disparity: Source-view disparity in pixels (H, W)
        shift_scale: Fraction of the disparity to shift by (+0.5 = left eye, -0.5 = right eye)

    Returns:
        Target-view disparity (H, W, float32); -inf where nothing landed (disocclusions)
    """
    height, width = disparity.shape[:2]
    disp = np.asarray(disparity, dtype=np.float32)
    target_x = np.arange(width, dtype=np.float32)[None, :] + shift_scale * disp
    x0 = np.floor(target_x).astype(np.int64)
    row_offsets = (np.arange(height, dtype=np.int64) * width)[:, None]

    zbuffer = np.full(height * width, -np.inf, dtype=np.float32)
    for xi in (x0, x0 + 1):
        inside = (xi >= 0) & (xi < width)
        np.maximum.at(zbuffer, (row_offsets + xi)[inside], disp[inside])
    return zbuffer.reshape(height, width)


def stereo_source_coordinates(
    disparity: np.ndarray, direction: str = "left"
) -> tuple[np.ndarray, np.ndarray]:
    """
    PURE: Per target pixel, the source x coordinate to sample for one eye.

    Visible pixels sample at the position given by the forward-warped disparity.
    Disoccluded pixels (background revealed next to a foreground edge) mirror the
    texture of their background-side neighbour, i.e. the neighbour with the smaller
    disparity, so foreground is never duplicated into the hole and the result is
    deterministic (temporally stable).

    Args:
        disparity: Source-view disparity in pixels (H, W)
        direction: "left" or "right" eye

    Returns:
        Tuple of (map_x float32 (H, W), hole_mask bool (H, W))
    """
    shift = 0.5 if direction == "left" else -0.5
    target_disp = forward_warp_disparity(disparity, shift)
    height, width = target_disp.shape
    found = np.isfinite(target_disp)
    # Soft depth edges stretch into bands of 1 px cracks with "flying" in-between pixels;
    # merge them into one hole so it gets filled as a unit from the true background,
    # and grow it slightly so flying pixels at its border are replaced as well
    kernel = np.ones((1, HOLE_MERGE_WIDTH), np.uint8)
    holes_grown = cv2.dilate(
        cv2.morphologyEx((~found).astype(np.uint8), cv2.MORPH_CLOSE, kernel), kernel
    )
    valid = found & (holes_grown == 0)
    cols = np.arange(width)
    xs = np.broadcast_to(cols.astype(np.float32), (height, width))
    map_x = xs - shift * np.where(valid, target_disp, 0.0)
    holes = ~valid
    if not holes.any():
        return map_x.astype(np.float32), holes

    # Nearest visible pixel to the left and to the right of every pixel in its row
    left_nb = np.maximum.accumulate(np.where(valid, cols, -1), axis=1)
    right_nb = np.minimum.accumulate(np.where(valid, cols, width)[:, ::-1], axis=1)[:, ::-1]
    has_left, has_right = left_nb >= 0, right_nb < width
    left_c, right_c = np.clip(left_nb, 0, width - 1), np.clip(right_nb, 0, width - 1)
    disp_left = np.where(has_left, np.take_along_axis(target_disp, left_c, axis=1), np.inf)
    disp_right = np.where(has_right, np.take_along_axis(target_disp, right_c, axis=1), np.inf)

    # Background side = smaller disparity
    use_left = has_left & (~has_right | (disp_left <= disp_right))
    neighbour = np.where(use_left, left_c, right_c)
    neighbour_src = np.take_along_axis(map_x, neighbour, axis=1)
    mirrored = neighbour_src + (neighbour - cols[None, :])  # reflect away from the hole

    fillable = holes & (has_left | has_right)
    map_x = np.where(fillable, mirrored, map_x)
    return map_x.astype(np.float32), holes


def create_stereo_view(
    image: np.ndarray, disparity: np.ndarray, direction: str = "left", hole_fill: str = "fast"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesize one eye's view with occlusion-aware warping.

    Args:
        image: Source image (H, W) or (H, W, C)
        disparity: Disparity in pixels (H, W), same size as ``image``
        direction: "left" or "right" eye
        hole_fill: "fast" (mirror background into disocclusions), "advanced"/"high"
            (additionally soften the seams around filled regions); anything else leaves
            holes black

    Returns:
        Tuple of (view, hole_mask)
    """
    height, width = image.shape[:2]
    map_x, holes = stereo_source_coordinates(disparity, direction)
    map_y = np.broadcast_to(np.arange(height, dtype=np.float32)[:, None], (height, width)).astype(
        np.float32
    )
    view = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    if not holes.any():
        return view, holes

    if hole_fill in ("advanced", "high"):
        # Soften only the seams at the hole border; blurring the whole fill would smear
        # textured backgrounds
        seams = cv2.morphologyEx(
            holes.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
        )
        alpha = np.clip(cv2.GaussianBlur(seams.astype(np.float32), (0, 0), 1.0) * 1.5, 0, 1)
        blurred = cv2.GaussianBlur(view, (0, 0), 1.5)
        if view.ndim == 3:
            alpha = alpha[:, :, None]
        view = (view * (1.0 - alpha) + blurred * alpha).astype(image.dtype)
    elif hole_fill != "fast":
        view = view.copy()
        view[holes] = 0

    return view, holes


def render_stereo_pair(
    frame: np.ndarray, depth_map: np.ndarray, settings: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Render a left/right stereo pair from a frame and its normalized inverse-depth map.

    The depth map may come at any resolution (e.g. from the depth cache); it is resized
    to the warp resolution. When super sampling is enabled the frame is upscaled first
    and the disparity is scaled by the same factor, so the 3D strength is independent of
    the super sampling mode.

    Args:
        frame: Source frame (H, W, C) or (H, W)
        depth_map: Normalized inverse depth map (any resolution, 1 = near)
        settings: Needs baseline, focal_length, hole_fill_quality; optionally
            super_sample, per_eye_width, per_eye_height

    Returns:
        Tuple of (left_image, right_image) at the warp resolution
    """
    src_h, src_w = frame.shape[:2]
    work_w, work_h = calculate_super_sample_size(
        src_w,
        src_h,
        int(settings.get("per_eye_width") or src_w),
        int(settings.get("per_eye_height") or src_h),
        str(settings.get("super_sample", "none")),
    )
    if (work_w, work_h) != (src_w, src_h):
        frame = cv2.resize(frame, (work_w, work_h), interpolation=cv2.INTER_CUBIC)

    depth_map = match_depth_to_frame(depth_map, work_w, work_h)
    disparity_map = depth_to_disparity(depth_map, settings["baseline"], settings["focal_length"])
    disparity_map *= work_w / src_w

    hole_fill = str(settings.get("hole_fill_quality", "fast"))
    left_img, _ = create_stereo_view(frame, disparity_map, "left", hole_fill)
    right_img, _ = create_stereo_view(frame, disparity_map, "right", hole_fill)
    return left_img, right_img


def apply_center_crop(image: np.ndarray, crop_factor: float) -> np.ndarray:
    """
    Apply center crop to image.

    Args:
        image: Input image array
        crop_factor: Crop factor (1.0 = no crop, 0.5 = crop to half size)

    Returns:
        Cropped image array
    """
    if crop_factor >= 1.0:
        return image

    height, width = image.shape[:2]

    # Calculate crop dimensions
    crop_width = int(width * crop_factor)
    crop_height = int(height * crop_factor)

    # Calculate crop coordinates
    x_start = (width - crop_width) // 2
    y_start = (height - crop_height) // 2
    x_end = x_start + crop_width
    y_end = y_start + crop_height

    return image[y_start:y_end, x_start:x_end]


def _fisheye_source_radius(
    rho: np.ndarray | float, max_radius: float, fov_radians: float, projection_type: str
) -> np.ndarray:
    """PURE: Source radius sampled for a normalized output radius (1.0 = inscribed circle)."""
    theta = np.asarray(rho, dtype=np.float64) * (fov_radians / 2)
    if projection_type == "stereographic":
        return 2 * max_radius * np.tan(theta / 2) / np.tan(fov_radians / 4)
    if projection_type == "equidistant":
        return max_radius * theta / (fov_radians / 2)
    if projection_type == "equisolid":
        return 2 * max_radius * np.sin(theta / 2) / np.sin(fov_radians / 4)
    return max_radius * np.sin(theta) / np.sin(fov_radians / 2)  # orthogonal


def fisheye_zoom_out(width: int, height: int, frame_fit: float) -> float:
    """
    PURE: Factor by which the fisheye view is zoomed out to show more of the frame.

    Args:
        width: Frame width
        height: Frame height
        frame_fit: 0.0 (fill the view) to 1.0 (show the whole frame)

    Returns:
        1.0 for no zoom, up to the frame's long/short side ratio for the whole frame
    """
    short_side, long_side = min(width, height), max(width, height)
    if short_side <= 0:
        return 1.0
    return 1.0 + min(max(float(frame_fit), 0.0), 1.0) * (long_side / short_side - 1.0)


def calculate_fisheye_coordinates(
    width: int,
    height: int,
    fov_degrees: float,
    projection_type: str = "stereographic",
    fit_radius: float | None = None,
    frame_fit: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate coordinate mappings for fisheye projection.

    Args:
        width: Output width
        height: Output height
        fov_degrees: Field of view in degrees
        projection_type: Type of fisheye projection
        fit_radius: Normalized output radius (1.0 = inscribed circle) that should land
            exactly on the source's top/bottom edge, typically the fisheye crop factor.
            The projections sample up to 2x beyond the source there, which used to be
            clamped into edge streaks. With fit_radius, everything inside that circle comes
            from real image content and coordinates outside the source are returned
            unclamped (render them as black). None keeps the legacy clamped mapping.
        frame_fit: Zooms the view out for frames that aren't square (needs fit_radius).
            0.0: the circle at fit_radius touches the frame's long edges (top/bottom of a
            wide video), so the view is filled and the short edges are cut off.
            1.0: it touches the short edges (left/right of a wide video), so the whole
            frame is visible, with black above and below. Values between interpolate.

    Returns:
        Tuple of (x_map, y_map) coordinate arrays
    """
    y, x = np.mgrid[0:height, 0:width]
    center_x, center_y = width / 2, height / 2
    x_centered = x - center_x
    y_centered = y - center_y

    radius = np.sqrt(x_centered**2 + y_centered**2)
    angle = np.arctan2(y_centered, x_centered)

    fov_radians = math.radians(fov_degrees)
    max_radius = min(width, height) / 2

    r_fisheye = _fisheye_source_radius(
        radius / max_radius, max_radius, fov_radians, projection_type
    )

    if fit_radius is not None:
        rho_fit = min(max(float(fit_radius), 0.05), 1.0)  # crops never extend past the short side
        r_fit = float(_fisheye_source_radius(rho_fit, max_radius, fov_radians, projection_type))
        if np.isfinite(r_fit) and r_fit > 0:
            # land on the outermost pixel centre, not half a pixel beyond it
            zoom_out = fisheye_zoom_out(width, height, frame_fit)
            r_fisheye = r_fisheye * (zoom_out * (max_radius - 0.5) / r_fit)

    x_fisheye = r_fisheye * np.cos(angle) + center_x
    y_fisheye = r_fisheye * np.sin(angle) + center_y

    if fit_radius is None:
        # Legacy behaviour: clamp to image bounds (stretches edge pixels into streaks)
        x_fisheye = np.clip(x_fisheye, 0, width - 1)
        y_fisheye = np.clip(y_fisheye, 0, height - 1)

    return x_fisheye.astype(np.float32), y_fisheye.astype(np.float32)


@functools.lru_cache(maxsize=8)
def _cached_fisheye_maps(
    width: int,
    height: int,
    fov_degrees: float,
    projection_type: str,
    fit_radius: float | None,
    frame_fit: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fisheye maps depend only on size and settings: compute once per video, not per frame."""
    x_map, y_map = calculate_fisheye_coordinates(
        width, height, fov_degrees, projection_type, fit_radius, frame_fit
    )
    x_map.flags.writeable = False
    y_map.flags.writeable = False
    return x_map, y_map


def apply_fisheye_distortion(
    image: np.ndarray,
    fov_degrees: float,
    projection_type: str = "stereographic",
    fit_radius: float | None = None,
    frame_fit: float = 0.0,
) -> np.ndarray:
    """
    Apply fisheye distortion to image.

    Args:
        image: Input image array
        fov_degrees: Field of view in degrees
        projection_type: Type of fisheye projection
        fit_radius: See calculate_fisheye_coordinates. When set, areas outside the source
            are black (standard circular fisheye) instead of stretched edge pixels.
        frame_fit: See calculate_fisheye_coordinates (0.0 fills the view, 1.0 shows the
            whole frame).

    Returns:
        Fisheye distorted image array
    """
    height, width = image.shape[:2]
    x_map, y_map = _cached_fisheye_maps(
        width, height, float(fov_degrees), projection_type, fit_radius, float(frame_fit)
    )

    if fit_radius is None:
        return cv2.remap(image, x_map, y_map, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return cv2.remap(
        image, x_map, y_map, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


def apply_fisheye_square_crop(
    image: np.ndarray, target_width: int, target_height: int, crop_factor: float = 1.0
) -> np.ndarray:
    """
    Apply fisheye-aware square cropping and scaling.

    Args:
        image: Fisheye distorted image
        target_width: Target output width
        target_height: Target output height
        crop_factor: Crop factor for the fisheye circle

    Returns:
        Cropped and scaled image array
    """
    height, width = image.shape[:2]

    # Find the fisheye circle (largest inscribed circle)
    circle_radius = min(width, height) // 2
    center_x, center_y = width // 2, height // 2

    # Apply crop factor to radius
    effective_radius = int(circle_radius * crop_factor)

    # Calculate crop bounds
    x_start = center_x - effective_radius
    y_start = center_y - effective_radius
    x_end = center_x + effective_radius
    y_end = center_y + effective_radius

    # Ensure bounds are within image
    x_start = max(0, x_start)
    y_start = max(0, y_start)
    x_end = min(width, x_end)
    y_end = min(height, y_end)

    # Crop to square
    cropped = image[y_start:y_end, x_start:x_end]

    # Scale to the target size; a non-square target gets black bars instead of a stretched circle
    if cropped.size > 0:
        scaled = fit_to_canvas(cropped, target_width, target_height)
    else:
        # Fallback if crop failed
        scaled = fit_to_canvas(image, target_width, target_height)

    return scaled


def create_vr_frame(left_image: np.ndarray, right_image: np.ndarray, vr_format: str) -> np.ndarray:
    """
    Combine left and right images into VR format.

    Args:
        left_image: Left eye image
        right_image: Right eye image
        vr_format: "side_by_side" or "over_under"

    Returns:
        Combined VR frame
    """
    if vr_format == "side_by_side":
        return np.hstack([left_image, right_image])
    elif vr_format == "over_under":
        return np.vstack([left_image, right_image])
    else:
        # Default to side_by_side
        return np.hstack([left_image, right_image])


def _create_hole_mask(image: np.ndarray) -> np.ndarray:
    """Create binary mask for holes (black pixels) in image."""
    if len(image.shape) == 3:
        mask = np.all(image == 0, axis=2).astype(np.uint8)
    else:
        mask = (image == 0).astype(np.uint8)

    # Dilate mask slightly to catch edge artifacts
    kernel = np.ones((3, 3), np.uint8)
    return np.asarray(cv2.dilate(mask, kernel, iterations=1))


def _calculate_adaptive_radius(mask: np.ndarray) -> int:
    """Calculate adaptive inpaint radius based on largest hole size."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:  # Only background
        return 3

    # Find largest hole (excluding background at index 0)
    max_area = max(stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels))

    # Adaptive radius: larger for bigger holes
    # sqrt gives good scaling: 100px hole -> ~10px radius
    return max(3, min(int(np.sqrt(max_area) * 0.5), 15))


def _apply_high_quality_inpaint(image: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """Apply high quality multi-pass inpainting."""
    # First pass: Navier-Stokes for structure
    filled = cv2.inpaint(image, mask, radius, cv2.INPAINT_NS)

    # Second pass: TELEA on remaining artifacts
    residual_mask = _create_hole_mask(filled)
    if np.any(residual_mask):
        filled = cv2.inpaint(filled, residual_mask, radius // 2, cv2.INPAINT_TELEA)

    # Third pass: Bilateral filter to smooth inpainted regions while preserving edges
    return cv2.bilateralFilter(filled, 5, 50, 50)


def hole_fill_image(
    image: np.ndarray, mask: np.ndarray | None = None, method: str = "fast"
) -> np.ndarray:
    """
    Fill holes in image using advanced inpainting with adaptive parameters.

    Args:
        image: Input image with holes
        mask: Binary mask of holes (None for auto-detection)
        method: "fast", "advanced", or "high" hole filling

    Returns:
        Image with filled holes
    """
    if mask is None:
        mask = _create_hole_mask(image)

    if not np.any(mask):
        return image  # No holes to fill

    adaptive_radius = _calculate_adaptive_radius(mask)

    if method == "high":
        return _apply_high_quality_inpaint(image, mask, adaptive_radius)
    elif method == "advanced":
        # Advanced: Navier-Stokes with adaptive radius
        filled = cv2.inpaint(image, mask, adaptive_radius, cv2.INPAINT_NS)
        return cv2.bilateralFilter(filled, 3, 30, 30)
    else:
        # Fast: TELEA with smaller radius
        return cv2.inpaint(image, mask, max(adaptive_radius // 2, 3), cv2.INPAINT_TELEA)


def validate_image_array(image: np.ndarray) -> bool:
    """
    Validate that array is a proper image.

    Args:
        image: Image array to validate

    Returns:
        True if valid image array
    """
    if not isinstance(image, np.ndarray):
        return False

    if len(image.shape) not in [2, 3]:
        return False

    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        return False

    if image.size == 0:
        return False

    return True


def calculate_image_statistics(image: np.ndarray) -> dict:
    """
    Calculate basic statistics for an image.

    Args:
        image: Input image array

    Returns:
        Dictionary with image statistics
    """
    if not validate_image_array(image):
        return {}

    stats = {
        "shape": image.shape,
        "dtype": str(image.dtype),
        "min": float(image.min()),
        "max": float(image.max()),
        "mean": float(image.mean()),
        "std": float(image.std()),
    }

    if len(image.shape) == 3:
        stats["channels"] = image.shape[2]
        # Per-channel statistics
        for i in range(image.shape[2]):
            channel_name = (
                ["blue", "green", "red", "alpha"][i] if image.shape[2] <= 4 else f"channel_{i}"
            )
            stats[f"{channel_name}_mean"] = float(image[:, :, i].mean())

    return stats
