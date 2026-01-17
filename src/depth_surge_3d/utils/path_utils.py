"""
Pure utility functions for path and string manipulation.

This module contains ONLY pure functions with no side effects:
- No filesystem I/O
- No subprocess calls
- No external state mutation
- Deterministic output for given inputs

All functions are safe to use in functional contexts and easy to test.
"""

from __future__ import annotations

import os
from pathlib import Path


def parse_time_string(time_str: str) -> float | None:
    """
    Parse time string in mm:ss or hh:mm:ss format to seconds.

    Args:
        time_str: Time string in "mm:ss" or "hh:mm:ss" format

    Returns:
        Time in seconds, or None if invalid format

    Examples:
        >>> parse_time_string("1:30")
        90.0
        >>> parse_time_string("01:05:30")
        3930.0
    """
    try:
        parts = time_str.strip().split(":")
        if len(parts) == 2:  # mm:ss
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:  # hh:mm:ss
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        return None
    except (ValueError, AttributeError):
        return None


def calculate_frame_range(
    total_frames: int,
    fps: float,
    start_time: str | None = None,
    end_time: str | None = None,
) -> tuple[int, int]:
    """
    Calculate frame range from time specifications.

    Pure function that computes frame indices based on time ranges.

    Args:
        total_frames: Total number of frames in video
        fps: Frames per second
        start_time: Start time in "mm:ss" or "hh:mm:ss" format
        end_time: End time in "mm:ss" or "hh:mm:ss" format

    Returns:
        Tuple of (start_frame, end_frame) indices

    Examples:
        >>> calculate_frame_range(1000, 30.0, "0:10", "0:20")
        (300, 600)
    """
    start_frame = 0
    end_frame = total_frames

    if start_time:
        start_seconds = parse_time_string(start_time)
        if start_seconds is not None:
            start_frame = max(0, int(start_seconds * fps))

    if end_time:
        end_seconds = parse_time_string(end_time)
        if end_seconds is not None:
            end_frame = min(total_frames, int(end_seconds * fps))

    # Ensure valid range
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_frames))

    return start_frame, end_frame


def generate_frame_filename(index: int, prefix: str = "frame") -> str:
    """
    Generate standardized frame filename.

    Pure function for consistent frame naming.

    Args:
        index: Frame index number
        prefix: Filename prefix

    Returns:
        Frame filename string

    Examples:
        >>> generate_frame_filename(42)
        'frame_000042.png'
        >>> generate_frame_filename(5, "depth")
        'depth_000005.png'
    """
    return f"{prefix}_{index:06d}.png"


def generate_output_filename(
    base_name: str,
    vr_format: str = "side_by_side",
    resolution: str | None = None,
    processing_mode: str | None = None,  # Deprecated, kept for backwards compatibility
) -> str:
    """
    Generate output video filename from components.

    Pure function that constructs standardized output filenames.

    Args:
        base_name: Base name for the output file
        vr_format: VR format ('side_by_side', 'over_under')
        resolution: Resolution string (e.g., '1080p', '4k')
        processing_mode: Deprecated parameter, ignored

    Returns:
        Output filename string

    Examples:
        >>> generate_output_filename("myvideo", "side_by_side", "1080p")
        'myvideo_3D_side-by-side_1080p.mp4'
    """
    # Extract stem from path and sanitize
    if base_name:
        clean_base = Path(base_name).stem
        safe_base = sanitize_filename(clean_base)
    else:
        safe_base = "output"

    # Convert underscores to hyphens in format for readability
    safe_format = vr_format.replace("_", "-")

    # Build filename parts
    parts = [safe_base, "3D", safe_format]

    if resolution:
        parts.append(resolution)

    filename = "_".join(parts) + ".mp4"
    return filename


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for cross-platform compatibility.

    Pure function that removes invalid characters and normalizes filenames.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename

    Examples:
        >>> sanitize_filename('my<video>.mp4')
        'my_video_.mp4'
        >>> sanitize_filename('___test___.txt')
        'test.txt'
    """
    # Replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Remove multiple underscores
    while "__" in filename:
        filename = filename.replace("__", "_")

    # Trim and limit length
    filename = filename.strip("_")
    if len(filename) > 200:  # Reasonable filename length limit
        name, ext = os.path.splitext(filename)
        filename = name[: 200 - len(ext)] + ext

    return filename


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Pure function for consistent size formatting.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string

    Examples:
        >>> format_file_size(0)
        '0 B'
        >>> format_file_size(1536)
        '1.5 KB'
        >>> format_file_size(1073741824)
        '1.0 GB'
    """
    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    size = float(size_bytes)

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"


def estimate_output_size(
    frame_count: int,
    width: int,
    height: int,
    vr_format: str = "side_by_side",
    target_fps: int = 60,
) -> dict[str, int]:
    """
    Estimate output file sizes.

    Pure function for size estimation based on video parameters.

    Args:
        frame_count: Number of frames
        width: Frame width
        height: Frame height
        vr_format: VR format
        target_fps: Target frames per second

    Returns:
        Dictionary with size estimates in bytes
    """
    # Calculate output dimensions
    if vr_format == "side_by_side":
        output_width = width * 2
        output_height = height
    else:  # over_under
        output_width = width
        output_height = height * 2

    # Rough estimation based on typical compression ratios
    pixels_per_frame = output_width * output_height
    bytes_per_pixel = 0.1  # H.265 typical compression

    video_size = int(frame_count * pixels_per_frame * bytes_per_pixel)
    audio_size = int(frame_count * 4000 / target_fps)  # ~4KB per second of audio

    return {
        "video": video_size,
        "audio": audio_size,
        "total": video_size + audio_size,
        "frames": frame_count,
    }


def format_time_duration(seconds: float) -> str:
    """
    Format time duration in human-readable format.

    Pure function for consistent time formatting.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string (HH:MM:SS)

    Examples:
        >>> format_time_duration(90.5)
        '00:01:30'
        >>> format_time_duration(3661)
        '01:01:01'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
