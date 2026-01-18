"""System and hardware utilities.

Modules for CUDA detection, VRAM management, and console output.
"""

from .vram_manager import (
    get_available_vram,
    get_total_vram,
    estimate_frame_vram_usage,
    calculate_optimal_chunk_size,
    get_vram_info,
)
from .console import (
    success,
    error,
    warning,
    info,
    dim,
    title_bar,
    step_complete,
    saved_to,
)

__all__ = [
    # VRAM
    "get_available_vram",
    "get_total_vram",
    "estimate_frame_vram_usage",
    "calculate_optimal_chunk_size",
    "get_vram_info",
    # Console
    "success",
    "error",
    "warning",
    "info",
    "dim",
    "title_bar",
    "step_complete",
    "saved_to",
]
