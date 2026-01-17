"""
Image processing utilities for depth estimation and stereo projection.

This module contains pure functions for image manipulation, depth processing,
and geometric transformations without side effects.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import math

from ..core.constants import MIN_DEPTH_VALUE, MAX_DEPTH_VALUE


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


def depth_to_disparity(depth_map: np.ndarray, baseline: float, focal_length: float) -> np.ndarray:
    """
    Convert depth map to disparity map for stereo generation.

    Args:
        depth_map: Normalized depth map (0-1 range)
        baseline: Stereo baseline in meters
        focal_length: Camera focal length in pixels

    Returns:
        Disparity map array
    """
    # Avoid division by zero
    safe_depth = np.where(depth_map > 0.001, depth_map, 0.001)

    # Convert normalized depth to actual depth (assuming 1 unit = 10 meters)
    actual_depth = safe_depth * 10.0

    # Calculate disparity: d = (baseline * focal_length) / depth
    disparity = (baseline * focal_length) / actual_depth

    return disparity


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
) -> Tuple[np.ndarray, np.ndarray]:
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


def hole_fill_image(
    image: np.ndarray, mask: Optional[np.ndarray] = None, method: str = "fast"
) -> np.ndarray:
    """
    Fill holes in image using inpainting.

    Args:
        image: Input image with holes
        mask: Binary mask of holes (None for auto-detection)
        method: "fast" or "advanced" hole filling

    Returns:
        Image with filled holes
    """
    if mask is None:
        # Auto-detect holes (black pixels)
        if len(image.shape) == 3:
            mask = np.all(image == 0, axis=2).astype(np.uint8)
        else:
            mask = (image == 0).astype(np.uint8)

    if not np.any(mask):
        return image  # No holes to fill

    if method == "advanced":
        # Use Navier-Stokes based inpainting (slower but better quality)
        filled = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
    else:
        # Use fast marching method (faster)
        filled = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    return filled


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
