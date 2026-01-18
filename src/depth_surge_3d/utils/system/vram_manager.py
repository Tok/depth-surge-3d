"""
VRAM management utilities for adaptive batch sizing.

Dynamically calculates optimal batch sizes based on available GPU memory,
model requirements, and frame resolution.
"""

from __future__ import annotations

import torch


def get_available_vram() -> float:
    """
    Get available VRAM in gigabytes.

    Returns:
        Available VRAM in GB, or 0.0 if CUDA is not available
    """
    if not torch.cuda.is_available():
        return 0.0

    try:
        mem_free, mem_total = torch.cuda.mem_get_info()
        return mem_free / (1024**3)  # Convert bytes to GB
    except Exception:
        return 0.0


def get_total_vram() -> float:
    """
    Get total VRAM in gigabytes.

    Returns:
        Total VRAM in GB, or 0.0 if CUDA is not available
    """
    if not torch.cuda.is_available():
        return 0.0

    try:
        mem_free, mem_total = torch.cuda.mem_get_info()
        return mem_total / (1024**3)  # Convert bytes to GB
    except Exception:
        return 0.0


def estimate_frame_vram_usage(
    frame_width: int, frame_height: int, depth_resolution: int, model_version: str = "v3"
) -> float:
    """
    Estimate VRAM usage per frame in GB.

    Args:
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        depth_resolution: Depth processing resolution
        model_version: "v2" or "v3"

    Returns:
        Estimated VRAM usage per frame in GB
    """
    # Calculate memory for frame storage (RGB, float32)
    frame_pixels = frame_width * frame_height
    frame_memory = frame_pixels * 3 * 4 / (1024**3)  # 3 channels, 4 bytes per float

    # Calculate memory for depth map
    depth_pixels = depth_resolution * depth_resolution
    depth_memory = depth_pixels * 4 / (1024**3)  # float32

    # Model-specific overhead
    if model_version == "v2":
        # V2 uses temporal windows and has higher memory overhead
        model_overhead = 0.15  # ~150MB per frame for temporal processing
    else:
        # V3 is more memory efficient
        model_overhead = 0.08  # ~80MB per frame

    return frame_memory + depth_memory + model_overhead


def calculate_optimal_chunk_size(
    frame_width: int,
    frame_height: int,
    depth_resolution: int,
    model_version: str = "v3",
    model_size: str = "base",
    safety_margin: float = 0.3,
) -> int:
    """
    Calculate optimal chunk size based on available VRAM.

    Args:
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        depth_resolution: Depth processing resolution
        model_version: "v2" or "v3"
        model_size: "small", "base", or "large"
        safety_margin: Reserve this fraction of VRAM (0.0-1.0)

    Returns:
        Optimal chunk size (number of frames)
    """
    available_vram = get_available_vram()

    # If no CUDA, return conservative CPU batch size
    if available_vram == 0:
        return 4  # Small batch for CPU processing

    # Estimate base model memory requirements (loaded once)
    model_memory = {
        "small": 1.5,  # ~1.5GB
        "base": 2.5,  # ~2.5GB
        "large": 4.0,  # ~4GB
    }.get(model_size, 2.5)

    # V2 has higher base memory due to temporal model
    if model_version == "v2":
        model_memory *= 1.3

    # Calculate usable VRAM after safety margin and model
    usable_vram = available_vram * (1.0 - safety_margin) - model_memory

    if usable_vram <= 0:
        # Not enough VRAM, return minimum chunk size
        return 2 if model_version == "v3" else 4  # V3 can handle smaller batches

    # Estimate per-frame VRAM usage
    per_frame_vram = estimate_frame_vram_usage(
        frame_width, frame_height, depth_resolution, model_version
    )

    # Calculate how many frames fit in usable VRAM
    optimal_chunks = int(usable_vram / per_frame_vram)

    # Apply constraints
    min_chunk = 2 if model_version == "v3" else 4  # V3 more flexible
    max_chunk = 32 if model_version == "v2" else 24  # V2 prefers 32-frame windows

    # Clamp to reasonable range
    optimal_chunks = max(min_chunk, min(optimal_chunks, max_chunk))

    return optimal_chunks


def get_vram_info() -> dict[str, float]:
    """
    Get comprehensive VRAM information.

    Returns:
        Dictionary with 'total', 'available', 'used', and 'usage_percent'
    """
    if not torch.cuda.is_available():
        return {"total": 0.0, "available": 0.0, "used": 0.0, "usage_percent": 0.0}

    try:
        mem_free, mem_total = torch.cuda.mem_get_info()
        mem_used = mem_total - mem_free

        return {
            "total": mem_total / (1024**3),
            "available": mem_free / (1024**3),
            "used": mem_used / (1024**3),
            "usage_percent": (mem_used / mem_total) * 100 if mem_total > 0 else 0.0,
        }
    except Exception:
        return {"total": 0.0, "available": 0.0, "used": 0.0, "usage_percent": 0.0}
