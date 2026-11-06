"""
Image processing utilities for depth estimation and stereo projection.

This module contains pure functions for image manipulation, depth processing,
and geometric transformations without side effects.
"""

from __future__ import annotations

import cv2
import numpy as np
import math
from typing import Any

from ...core.constants import MIN_DEPTH_VALUE, MAX_DEPTH_VALUE, STEREO_NEAR_PLANE_M


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

    left_img = create_shifted_image(frame, disparity_map, "left")
    right_img = create_shifted_image(frame, disparity_map, "right")

    hole_fill_quality = settings.get("hole_fill_quality", "none")
    if hole_fill_quality in ["fast", "advanced"]:
        left_img = hole_fill_image(left_img, method=hole_fill_quality)
        right_img = hole_fill_image(right_img, method=hole_fill_quality)

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


def calculate_fisheye_coordinates(
    width: int, height: int, fov_degrees: float, projection_type: str = "stereographic"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate coordinate mappings for fisheye projection.

    Args:
        width: Output width
        height: Output height
        fov_degrees: Field of view in degrees
        projection_type: Type of fisheye projection

    Returns:
        Tuple of (x_map, y_map) coordinate arrays
    """
    # Create coordinate grids
    y, x = np.mgrid[0:height, 0:width]

    # Center coordinates
    center_x, center_y = width / 2, height / 2

    # Convert to centered coordinates
    x_centered = x - center_x
    y_centered = y - center_y

    # Calculate radius and angle
    radius = np.sqrt(x_centered**2 + y_centered**2)
    angle = np.arctan2(y_centered, x_centered)

    # Calculate maximum radius for given FOV
    fov_radians = math.radians(fov_degrees)
    max_radius = min(width, height) / 2

    # Apply projection based on type
    if projection_type == "stereographic":
        # Stereographic projection
        theta = (radius / max_radius) * (fov_radians / 2)
        r_fisheye = 2 * max_radius * np.tan(theta / 2) / np.tan(fov_radians / 4)
    elif projection_type == "equidistant":
        # Equidistant projection
        theta = (radius / max_radius) * (fov_radians / 2)
        r_fisheye = max_radius * theta / (fov_radians / 2)
    elif projection_type == "equisolid":
        # Equisolid projection
        theta = (radius / max_radius) * (fov_radians / 2)
        r_fisheye = 2 * max_radius * np.sin(theta / 2) / np.sin(fov_radians / 4)
    else:  # orthogonal
        # Orthogonal projection
        theta = (radius / max_radius) * (fov_radians / 2)
        r_fisheye = max_radius * np.sin(theta) / np.sin(fov_radians / 2)

    # Convert back to cartesian coordinates
    x_fisheye = r_fisheye * np.cos(angle) + center_x
    y_fisheye = r_fisheye * np.sin(angle) + center_y

    # Clamp coordinates to image bounds instead of masking to circle
    # This allows the fisheye to fill the entire rectangular frame
    x_fisheye = np.clip(x_fisheye, 0, width - 1)
    y_fisheye = np.clip(y_fisheye, 0, height - 1)

    return x_fisheye.astype(np.float32), y_fisheye.astype(np.float32)


def apply_fisheye_distortion(
    image: np.ndarray, fov_degrees: float, projection_type: str = "stereographic"
) -> np.ndarray:
    """
    Apply fisheye distortion to image.

    Args:
        image: Input image array
        fov_degrees: Field of view in degrees
        projection_type: Type of fisheye projection

    Returns:
        Fisheye distorted image array
    """
    height, width = image.shape[:2]

    # Get coordinate mappings
    x_map, y_map = calculate_fisheye_coordinates(width, height, fov_degrees, projection_type)

    # Apply remapping with reflection to avoid black borders
    distorted = cv2.remap(image, x_map, y_map, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return distorted


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

    # Resize to target dimensions
    if cropped.size > 0:
        scaled = resize_image(cropped, target_width, target_height)
    else:
        # Fallback if crop failed
        scaled = resize_image(image, target_width, target_height)

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
