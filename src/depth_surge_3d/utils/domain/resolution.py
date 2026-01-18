"""
Resolution utilities for VR output processing.

This module contains pure functions for resolution parsing, validation,
and auto-detection based on source content.
"""

from __future__ import annotations

from typing import Any

from ...core.constants import VR_RESOLUTIONS


def parse_custom_resolution(resolution_string: str) -> tuple[int, int | None]:
    """
    Parse custom resolution string into width and height.

    Args:
        resolution_string: Format "custom:WIDTHxHEIGHT" (e.g., "custom:1920x1080")

    Returns:
        Tuple of (width, height) or None if invalid
    """
    if not resolution_string.startswith("custom:"):
        return None

    try:
        custom_res = resolution_string.replace("custom:", "")
        width_str, height_str = custom_res.split("x")
        width, height = int(width_str), int(height_str)

        # Basic validation
        if width <= 0 or height <= 0:
            return None
        if width > 10000 or height > 10000:  # Reasonable upper limit
            return None

        return (width, height)
    except (ValueError, IndexError):
        return None


def get_resolution_dimensions(vr_resolution: str) -> tuple[int, int]:
    """
    Get resolution dimensions for a given VR resolution setting.

    Args:
        vr_resolution: Resolution setting (preset or custom format)

    Returns:
        Tuple of (per_eye_width, per_eye_height)

    Raises:
        ValueError: If resolution setting is invalid
    """
    if vr_resolution in VR_RESOLUTIONS:
        return VR_RESOLUTIONS[vr_resolution]

    custom_resolution = parse_custom_resolution(vr_resolution)
    if custom_resolution:
        return custom_resolution

    raise ValueError(f"Invalid resolution setting: {vr_resolution}")


def calculate_vr_output_dimensions(
    per_eye_width: int, per_eye_height: int, vr_format: str
) -> tuple[int, int]:
    """
    Calculate final VR output dimensions based on per-eye dimensions and format.

    Args:
        per_eye_width: Width of each eye
        per_eye_height: Height of each eye
        vr_format: "side_by_side" or "over_under"

    Returns:
        Tuple of (total_width, total_height)
    """
    if vr_format == "side_by_side":
        return (per_eye_width * 2, per_eye_height)
    elif vr_format == "over_under":
        return (per_eye_width, per_eye_height * 2)
    else:
        # Default to side_by_side
        return (per_eye_width * 2, per_eye_height)


def calculate_aspect_ratio(width: int, height: int) -> float:
    """
    Calculate aspect ratio from width and height.

    Args:
        width: Image width
        height: Image height

    Returns:
        Aspect ratio as width/height
    """
    if height == 0:
        return 1.0
    return width / height


def classify_aspect_ratio(aspect_ratio: float) -> str:
    """
    Classify aspect ratio into categories.

    Args:
        aspect_ratio: Aspect ratio value

    Returns:
        Category: "ultra_wide", "wide", or "standard"
    """
    if aspect_ratio >= 2.3:  # Cinema and ultra-wide
        return "ultra_wide"
    elif aspect_ratio >= 1.6:  # 16:9 and similar wide formats
        return "wide"
    else:  # Square and portrait
        return "standard"


def auto_detect_resolution(source_width: int, source_height: int, vr_format: str) -> str:
    """
    Auto-detect optimal VR resolution based on source dimensions.

    Args:
        source_width: Source video width
        source_height: Source video height
        vr_format: Target VR format

    Returns:
        Recommended resolution setting
    """
    aspect_ratio = calculate_aspect_ratio(source_width, source_height)
    category = classify_aspect_ratio(aspect_ratio)

    # Get source pixel count for quality matching
    source_pixels = source_width * source_height

    # Select resolution based on category and source quality
    if category == "ultra_wide":
        if source_pixels >= 8000000:  # 4K+ sources
            return "cinema-4k"
        else:
            return "cinema-2k"
    elif category == "wide":
        if source_pixels >= 8000000:  # 4K+ sources
            return "16x9-4k"
        elif source_pixels >= 2000000:  # 1080p+ sources
            return "16x9-1080p"
        else:
            return "16x9-720p"
    else:  # standard/square
        if source_pixels >= 8000000:  # 4K+ sources
            return "square-4k"
        elif source_pixels >= 2000000:  # 1080p+ sources
            return "square-2k"
        else:
            return "square-1k"


def get_format_recommendation(aspect_ratio: float) -> str:
    """
    Recommend VR format based on aspect ratio.

    Args:
        aspect_ratio: Source aspect ratio

    Returns:
        Recommended format: "side_by_side" or "over_under"
    """
    if aspect_ratio >= 2.0:  # Ultra-wide content
        return "over_under"
    else:
        return "side_by_side"


def validate_resolution_settings(
    vr_resolution: str, vr_format: str, source_width: int, source_height: int
) -> dict[str, Any]:
    """
    Validate resolution settings and provide recommendations.

    Args:
        vr_resolution: Target VR resolution
        vr_format: Target VR format
        source_width: Source video width
        source_height: Source video height

    Returns:
        Dictionary with validation results and recommendations
    """
    result = {
        "valid": True,
        "warnings": [],
        "recommendations": [],
        "final_resolution": None,
        "final_format": vr_format,
    }

    try:
        if vr_resolution == "auto":
            auto_res = auto_detect_resolution(source_width, source_height, vr_format)
            result["final_resolution"] = auto_res
            result["recommendations"].append(f"Auto-detected resolution: {auto_res}")
        else:
            per_eye_width, per_eye_height = get_resolution_dimensions(vr_resolution)
            result["final_resolution"] = vr_resolution

        # Check format recommendation
        source_aspect = calculate_aspect_ratio(source_width, source_height)
        recommended_format = get_format_recommendation(source_aspect)

        if vr_format != recommended_format and source_aspect >= 2.0:
            result["warnings"].append(
                f"Wide content (aspect ratio {source_aspect:.2f}) works better with {recommended_format} format"
            )
            result["recommendations"].append(f"Consider using --format {recommended_format}")

    except ValueError as e:
        result["valid"] = False
        result["warnings"].append(str(e))

    return result


def get_available_resolutions() -> dict[str, list[dict[str, Any]]]:
    """
    Get categorized list of available resolutions.

    Returns:
        Dictionary with resolution categories and their options
    """
    categorized: dict[str, list[dict[str, Any]]] = {
        "square": [],
        "16x9": [],
        "wide": [],
        "cinema": [],
        "legacy": [],
    }

    for res_name, (width, height) in VR_RESOLUTIONS.items():
        if res_name.startswith("square-"):
            categorized["square"].append(
                {
                    "name": res_name,
                    "description": f"{width}×{height} per eye",
                    "dimensions": (width, height),
                }
            )
        elif res_name.startswith("16x9-"):
            categorized["16x9"].append(
                {
                    "name": res_name,
                    "description": f"{width}×{height} per eye",
                    "dimensions": (width, height),
                }
            )
        elif res_name.startswith("cinema-"):
            categorized["cinema"].append(
                {
                    "name": res_name,
                    "description": f"{width}×{height} per eye",
                    "dimensions": (width, height),
                }
            )
        else:
            categorized["legacy"].append(
                {
                    "name": res_name,
                    "description": f"{width}×{height} per eye",
                    "dimensions": (width, height),
                }
            )

    return categorized
