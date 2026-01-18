"""I/O operations for Depth Surge 3D.

This module handles all file system operations including:
- Video validation and properties
- Directory management
- Frame I/O (loading/saving)
- Processing state persistence
"""

from .operations import (
    # Video operations
    validate_video_file,
    validate_image_file,
    get_video_properties,
    get_video_info_ffprobe,
    # Directory operations
    create_output_directories,
    get_frame_files,
    calculate_directory_size,
    cleanup_intermediate_files,
    get_available_space,
    # Processing state
    save_processing_settings,
    load_processing_settings,
    update_processing_status,
    find_settings_file,
    can_resume_processing,
    analyze_processing_progress,
    # FFmpeg
    verify_ffmpeg_installation,
)

__all__ = [
    # Video operations
    "validate_video_file",
    "validate_image_file",
    "get_video_properties",
    "get_video_info_ffprobe",
    # Directory operations
    "create_output_directories",
    "get_frame_files",
    "calculate_directory_size",
    "cleanup_intermediate_files",
    "get_available_space",
    # Processing state
    "save_processing_settings",
    "load_processing_settings",
    "update_processing_status",
    "find_settings_file",
    "can_resume_processing",
    "analyze_processing_progress",
    # FFmpeg
    "verify_ffmpeg_installation",
]
