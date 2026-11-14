"""Domain-specific utilities.

Depth alignment, resolution management, and progress tracking.
"""

from .resolution import (
    parse_custom_resolution,
    get_resolution_dimensions,
    calculate_vr_output_dimensions,
    calculate_aspect_ratio,
    classify_aspect_ratio,
    auto_detect_resolution,
    get_format_recommendation,
    validate_resolution_settings,
    get_available_resolutions,
)
from .progress import (
    ProgressReporter,
    ConsoleProgressReporter,
    ProgressTracker,
    ProgressCallback,
    create_progress_tracker,
    calculate_eta,
    format_time_duration,
)

__all__ = [
    # Resolution
    "parse_custom_resolution",
    "get_resolution_dimensions",
    "calculate_vr_output_dimensions",
    "calculate_aspect_ratio",
    "classify_aspect_ratio",
    "auto_detect_resolution",
    "get_format_recommendation",
    "validate_resolution_settings",
    "get_available_resolutions",
    # Progress
    "ProgressReporter",
    "ConsoleProgressReporter",
    "ProgressTracker",
    "ProgressCallback",
    "create_progress_tracker",
    "calculate_eta",
    "format_time_duration",
]
