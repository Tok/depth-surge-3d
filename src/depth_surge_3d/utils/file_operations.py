"""
File operations utilities for video and image processing.

This module contains pure functions for file handling, validation,
and path operations without side effects.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import time
import cv2

from ..core.constants import (
    SUPPORTED_VIDEO_FORMATS,
    SUPPORTED_IMAGE_FORMATS,
    OUTPUT_IMAGE_FORMAT,
    OUTPUT_VIDEO_FORMAT,
    INTERMEDIATE_DIRS,
)


def validate_video_file(video_path: str) -> bool:
    """
    Validate if file is a supported video format.

    Args:
        video_path: Path to video file

    Returns:
        True if valid video file
    """
    if not os.path.exists(video_path):
        return False

    file_ext = Path(video_path).suffix.lower()
    return file_ext in SUPPORTED_VIDEO_FORMATS


def validate_image_file(image_path: str) -> bool:
    """
    Validate if file is a supported image format.

    Args:
        image_path: Path to image file

    Returns:
        True if valid image file
    """
    if not os.path.exists(image_path):
        return False

    file_ext = Path(image_path).suffix.lower()
    return file_ext in SUPPORTED_IMAGE_FORMATS


def get_video_properties(video_path: str) -> Dict[str, Any]:
    """
    Get video properties using OpenCV.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video properties
    """
    properties = {}

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return properties

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Guard against zero FPS to prevent division by zero
        if fps > 0:
            duration = frame_count / fps
        else:
            duration = 0.0

        properties.update(
            {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
            }
        )

        cap.release()

    except Exception:
        # Return empty dict if unable to read properties
        pass

    return properties


def get_video_info_ffprobe(video_path: str) -> Dict[str, Any]:
    """
    Get detailed video information using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video information
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            import json

            return json.loads(result.stdout)

    except Exception:
        pass

    return {}


def calculate_frame_range(
    total_frames: int,
    fps: float,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Calculate frame range from time specifications.

    Args:
        total_frames: Total frames in video
        fps: Frames per second
        start_time: Start time in "mm:ss" or "hh:mm:ss" format
        end_time: End time in "mm:ss" or "hh:mm:ss" format

    Returns:
        Tuple of (start_frame, end_frame)
    """
    start_frame = 0
    end_frame = total_frames

    if start_time:
        start_seconds = parse_time_string(start_time)
        if start_seconds is not None:
            start_frame = int(start_seconds * fps)

    if end_time:
        end_seconds = parse_time_string(end_time)
        if end_seconds is not None:
            end_frame = int(end_seconds * fps)

    # Ensure valid range
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_frames))

    return start_frame, end_frame


def parse_time_string(time_str: str) -> Optional[float]:
    """
    Parse time string into seconds.

    Args:
        time_str: Time in "mm:ss" or "hh:mm:ss" format

    Returns:
        Time in seconds or None if invalid
    """
    try:
        parts = time_str.split(":")

        if len(parts) == 2:  # mm:ss
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:  # hh:mm:ss
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds

    except (ValueError, TypeError):
        pass

    return None


def create_output_directories(
    base_path: Path, keep_intermediates: bool = True
) -> Dict[str, Path]:
    """
    Create output directory structure.

    Args:
        base_path: Base output directory path
        keep_intermediates: Whether to create intermediate directories

    Returns:
        Dictionary mapping directory names to paths
    """
    directories = {"base": base_path}

    # Always create base directory
    base_path.mkdir(parents=True, exist_ok=True)

    if keep_intermediates:
        for dir_name, dir_path in INTERMEDIATE_DIRS.items():
            full_path = base_path / dir_path
            full_path.mkdir(exist_ok=True)
            directories[dir_name] = full_path

    return directories


def get_frame_files(frames_dir: Path) -> List[Path]:
    """
    Get sorted list of frame files from directory.

    Args:
        frames_dir: Directory containing frame files

    Returns:
        Sorted list of frame file paths
    """
    if not frames_dir.exists():
        return []

    frame_files = []
    for ext in SUPPORTED_IMAGE_FORMATS:
        frame_files.extend(frames_dir.glob(f"*{ext}"))

    # Sort numerically by filename
    def sort_key(path):
        try:
            # Extract number from filename
            stem = path.stem
            if stem.startswith("frame_"):
                return int(stem.split("_")[1])
            else:
                return int("".join(filter(str.isdigit, stem)))
        except (ValueError, IndexError):
            return 0

    return sorted(frame_files, key=sort_key)


def generate_frame_filename(index: int, prefix: str = "frame") -> str:
    """
    Generate standardized frame filename.

    Args:
        index: Frame index
        prefix: Filename prefix

    Returns:
        Formatted filename
    """
    return f"{prefix}_{index:06d}{OUTPUT_IMAGE_FORMAT}"


def generate_output_filename(
    base_name: str,
    vr_format: str,
    vr_resolution: str,
    processing_mode: Optional[
        str
    ] = None,  # Deprecated, kept for backwards compatibility
) -> str:
    """
    Generate output filename with metadata.

    Args:
        base_name: Base name from input file
        vr_format: VR format used
        vr_resolution: VR resolution used
        processing_mode: (Deprecated) Processing mode - no longer used

    Returns:
        Generated filename
    """
    # Clean base name
    clean_base = Path(base_name).stem

    # Add metadata (no longer include processing_mode)
    metadata_parts = [clean_base, vr_format.replace("_", "-"), vr_resolution]

    filename = "_".join(metadata_parts) + OUTPUT_VIDEO_FORMAT

    # Sanitize filename
    filename = sanitize_filename(filename)

    return filename


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for cross-platform compatibility.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
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


def calculate_directory_size(directory: Path) -> int:
    """
    Calculate total size of directory in bytes.

    Args:
        directory: Directory path

    Returns:
        Total size in bytes
    """
    total_size = 0

    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except (OSError, PermissionError):
        pass

    return total_size


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
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


def _should_keep_file(file_path: Path, keep_patterns: List[str]) -> bool:
    """
    Check if file should be kept based on pattern matching.

    Args:
        file_path: Path to file to check
        keep_patterns: List of patterns to keep

    Returns:
        True if file should be kept
    """
    for pattern in keep_patterns:
        if file_path.match(pattern):
            return True
    return False


def _remove_file_safe(file_path: Path) -> bool:
    """
    Safely remove a file, handling errors.

    Args:
        file_path: Path to file to remove

    Returns:
        True if file was removed successfully
    """
    try:
        file_path.unlink()
        return True
    except OSError:
        return False


def _cleanup_directory(directory: Path, keep_patterns: List[str]) -> int:
    """
    Clean up files in a single directory.

    Args:
        directory: Directory to clean
        keep_patterns: Patterns to keep

    Returns:
        Number of files removed
    """
    removed_count = 0

    for file_path in directory.rglob("*"):
        if file_path.is_file():
            if not _should_keep_file(file_path, keep_patterns):
                if _remove_file_safe(file_path):
                    removed_count += 1

    return removed_count


def cleanup_intermediate_files(
    base_path: Path, keep_patterns: Optional[List[str]] = None
) -> int:
    """
    Clean up intermediate files, optionally keeping certain patterns.

    Args:
        base_path: Base directory containing intermediate files
        keep_patterns: List of patterns to keep (glob patterns)

    Returns:
        Number of files removed
    """
    removed_count = 0
    keep_patterns = keep_patterns or []

    try:
        for intermediate_dir in INTERMEDIATE_DIRS.values():
            full_dir = base_path / intermediate_dir
            if full_dir.exists():
                removed_count += _cleanup_directory(full_dir, keep_patterns)

    except (OSError, PermissionError):
        pass

    return removed_count


def verify_ffmpeg_installation() -> bool:
    """
    Verify that FFmpeg is installed and accessible.

    Returns:
        True if FFmpeg is available
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False


def get_available_space(directory: Path) -> int:
    """
    Get available disk space for directory in bytes.

    Args:
        directory: Directory path to check

    Returns:
        Available space in bytes
    """
    try:
        stat = os.statvfs(directory)
        return stat.f_bavail * stat.f_frsize
    except (OSError, AttributeError):
        # Fallback for systems without statvfs
        try:
            import shutil

            _, _, free = shutil.disk_usage(directory)
            return free
        except (OSError, ImportError):
            return 0


def estimate_output_size(
    frame_count: int, vr_width: int, vr_height: int, keep_intermediates: bool = True
) -> Dict[str, int]:
    """
    Estimate storage requirements for processing.

    Args:
        frame_count: Number of frames to process
        vr_width: VR output width
        vr_height: VR output height
        keep_intermediates: Whether intermediate files will be kept

    Returns:
        Dictionary with size estimates in bytes
    """
    # Rough estimates based on typical compression ratios
    estimates = {}

    # VR frame size (PNG format, roughly 3 bytes per pixel with compression)
    vr_frame_size = vr_width * vr_height * 3
    estimates["vr_frames"] = vr_frame_size * frame_count

    if keep_intermediates:
        # Original frames (smaller than VR frames)
        estimates["original_frames"] = vr_frame_size * 0.5 * frame_count

        # Depth maps (single channel)
        estimates["depth_maps"] = vr_width * vr_height * frame_count

        # Stereo pairs (2x VR frame size)
        estimates["stereo_pairs"] = vr_frame_size * 2 * frame_count

        # Other intermediate files
        estimates["other_intermediates"] = vr_frame_size * 0.5 * frame_count

    # Final video (much smaller due to compression)
    estimates["final_video"] = vr_frame_size * 0.1 * frame_count

    # Total estimate
    estimates["total"] = sum(estimates.values())

    return estimates


def save_processing_settings(
    output_dir: Path,
    batch_name: str,
    settings: Dict[str, Any],
    video_properties: Dict[str, Any],
    source_video_path: str,
) -> Path:
    """
    Save processing settings to a JSON file in the output directory.

    Args:
        output_dir: Output directory path
        batch_name: Name of the processing batch/job
        settings: Complete processing settings dictionary
        video_properties: Video metadata
        source_video_path: Path to source video file

    Returns:
        Path to the saved settings file
    """
    settings_data = {
        "metadata": {
            "batch_name": batch_name,
            "source_video": str(source_video_path),
            "source_video_name": Path(source_video_path).name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "created_timestamp": time.time(),
            "project_version": "1.0.0",
            "processing_status": "in_progress",
        },
        "video_properties": video_properties,
        "processing_settings": settings,
        "output_info": {
            "output_directory": str(output_dir),
            "expected_output_filename": generate_output_filename(
                Path(source_video_path).name,
                settings["vr_format"],
                settings["vr_resolution"],
            ),
        },
    }

    # Save settings file
    settings_file = output_dir / f"{batch_name}-settings.json"

    try:
        with open(settings_file, "w", encoding="utf-8") as f:
            json.dump(settings_data, f, indent=2, ensure_ascii=False)

        print(f"Settings saved to: {settings_file}")
        return settings_file

    except Exception as e:
        print(f"Warning: Could not save settings file: {e}")
        # Create a minimal fallback file
        fallback_data = {
            "metadata": {
                "batch_name": batch_name,
                "error": f"Failed to save full settings: {e}",
            },
            "processing_settings": settings,
        }
        try:
            with open(settings_file, "w", encoding="utf-8") as f:
                json.dump(fallback_data, f, indent=2)
        except Exception:
            pass  # If we can't even save the fallback, continue without it

        return settings_file


def load_processing_settings(settings_file: Path) -> Optional[Dict[str, Any]]:
    """
    Load processing settings from a JSON file.

    Args:
        settings_file: Path to settings file

    Returns:
        Dictionary with loaded settings data or None if failed
    """
    try:
        if not settings_file.exists():
            print(f"Settings file not found: {settings_file}")
            return None

        with open(settings_file, "r", encoding="utf-8") as f:
            settings_data = json.load(f)

        print(f"Settings loaded from: {settings_file}")
        return settings_data

    except Exception as e:
        print(f"Error loading settings file: {e}")
        return None


def update_processing_status(
    settings_file: Path, status: str, additional_info: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Update the processing status in the settings file.

    Args:
        settings_file: Path to settings file
        status: New status ('in_progress', 'completed', 'failed', 'paused')
        additional_info: Additional metadata to add

    Returns:
        True if update was successful
    """
    try:
        settings_data = load_processing_settings(settings_file)
        if not settings_data:
            return False

        # Update status and timestamp
        settings_data["metadata"]["processing_status"] = status
        settings_data["metadata"]["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        settings_data["metadata"]["last_updated_timestamp"] = time.time()

        if status == "completed":
            settings_data["metadata"]["completed_at"] = time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            settings_data["metadata"]["completed_timestamp"] = time.time()

            # Calculate processing duration
            if "created_timestamp" in settings_data["metadata"]:
                duration = time.time() - settings_data["metadata"]["created_timestamp"]
                settings_data["metadata"]["processing_duration_seconds"] = duration
                settings_data["metadata"]["processing_duration_formatted"] = (
                    format_time_duration(duration)
                )

        # Add any additional info
        if additional_info:
            if "runtime_info" not in settings_data:
                settings_data["runtime_info"] = {}
            settings_data["runtime_info"].update(additional_info)

        # Save updated settings
        with open(settings_file, "w", encoding="utf-8") as f:
            json.dump(settings_data, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"Warning: Could not update settings file status: {e}")
        return False


def find_settings_file(
    output_dir: Path, batch_name: Optional[str] = None
) -> Optional[Path]:
    """
    Find a settings file in the output directory.

    Args:
        output_dir: Output directory to search
        batch_name: Specific batch name to look for (if None, finds any)

    Returns:
        Path to settings file or None if not found
    """
    try:
        if batch_name:
            # Look for specific batch
            settings_file = output_dir / f"{batch_name}-settings.json"
            if settings_file.exists():
                return settings_file
        else:
            # Find any settings file
            for file_path in output_dir.glob("*-settings.json"):
                return file_path

        return None

    except Exception:
        return None


def can_resume_processing(output_dir: Path) -> Dict[str, Any]:
    """
    Check if processing can be resumed from the given directory.

    Args:
        output_dir: Output directory to check

    Returns:
        Dictionary with resume information
    """
    result = {
        "can_resume": False,
        "settings_file": None,
        "batch_name": None,
        "status": None,
        "progress_info": None,
        "recommendations": [],
    }

    try:
        # Find settings file
        settings_file = find_settings_file(output_dir)
        if not settings_file:
            result["recommendations"].append("No settings file found - cannot resume")
            return result

        # Load settings
        settings_data = load_processing_settings(settings_file)
        if not settings_data:
            result["recommendations"].append("Could not load settings file")
            return result

        result["settings_file"] = settings_file
        result["batch_name"] = settings_data.get("metadata", {}).get("batch_name")
        result["status"] = settings_data.get("metadata", {}).get("processing_status")

        # Check if can resume based on status
        if result["status"] in ["completed"]:
            result["recommendations"].append("Processing already completed")
            return result
        elif result["status"] in ["in_progress", "paused", "failed"]:
            result["can_resume"] = True

            # Analyze progress
            progress_info = analyze_processing_progress(output_dir, settings_data)
            result["progress_info"] = progress_info

            if progress_info["frames_processed"] > 0:
                result["recommendations"].append(
                    f"Found {progress_info['frames_processed']} processed frames - can resume from where left off"
                )
            else:
                result["recommendations"].append(
                    "No processed frames found - will restart from beginning"
                )

        return result

    except Exception as e:
        result["recommendations"].append(f"Error checking resume capability: {e}")
        return result


def analyze_processing_progress(
    output_dir: Path, settings_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze how much processing has been completed.

    Args:
        output_dir: Output directory path
        settings_data: Loaded settings data

    Returns:
        Dictionary with progress analysis
    """
    progress = {
        "frames_processed": 0,
        "vr_frames_created": 0,
        "intermediate_stages": {},
        "can_resume_from_intermediates": False,
    }

    try:
        # Check each intermediate directory
        for stage_name, dir_name in INTERMEDIATE_DIRS.items():
            stage_dir = output_dir / dir_name
            if stage_dir.exists():
                frame_count = len(list(stage_dir.glob("*.png")))
                progress["intermediate_stages"][stage_name] = {
                    "directory": str(stage_dir),
                    "frames_found": frame_count,
                }

                if stage_name == "vr_frames":
                    progress["vr_frames_created"] = frame_count

        # Determine overall progress
        if progress["vr_frames_created"] > 0:
            progress["frames_processed"] = progress["vr_frames_created"]
            progress["can_resume_from_intermediates"] = True
        elif "depth_maps" in progress["intermediate_stages"]:
            depth_frames = progress["intermediate_stages"]["depth_maps"]["frames_found"]
            if depth_frames > 0:
                progress["frames_processed"] = depth_frames
                progress["can_resume_from_intermediates"] = True

        return progress

    except Exception as e:
        print(f"Warning: Could not analyze processing progress: {e}")
        return progress


def format_time_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"
