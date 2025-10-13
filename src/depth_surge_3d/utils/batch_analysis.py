#!/usr/bin/env python3
"""
Batch Analysis Utilities
Extracted from app.py for batch directory analysis and video creation
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..core.constants import INTERMEDIATE_DIRS


# Lazy import cv2 to avoid blocking module loading when cv2 is not available
def _get_cv2():
    """Lazy import cv2 only when needed."""
    try:
        import cv2

        return cv2
    except ImportError:
        raise ImportError("opencv-python is required for image processing. Install with: pip install opencv-python")


def analyze_batch_directory(batch_path: Path) -> Dict[str, Any]:
    """
    Analyze batch directory to determine available processing stages and settings.

    Args:
        batch_path: Path to the batch directory

    Returns:
        Dictionary with analysis results including stages, frame count, etc.
    """
    batch_path = Path(batch_path)

    analysis = {
        "frame_count": 0,
        "vr_format": "unknown",
        "resolution": "unknown",
        "highest_stage": "none",
        "has_audio": False,
        "settings_summary": "unknown",
    }

    # Check for different processing stages using constants
    stages = {
        INTERMEDIATE_DIRS["vr_frames"]: "Final VR frames",
        INTERMEDIATE_DIRS["left_final"]: "Final left frames",
        INTERMEDIATE_DIRS["right_final"]: "Final right frames",
        INTERMEDIATE_DIRS["left_distorted"]: "Distorted left frames",
        INTERMEDIATE_DIRS["right_distorted"]: "Distorted right frames",
        INTERMEDIATE_DIRS["left_cropped"]: "Cropped left frames",
        INTERMEDIATE_DIRS["right_cropped"]: "Cropped right frames",
        INTERMEDIATE_DIRS["left_frames"]: "Basic left frames",
        INTERMEDIATE_DIRS["right_frames"]: "Basic right frames",
        INTERMEDIATE_DIRS["depth_maps"]: "Depth maps",
        INTERMEDIATE_DIRS["supersampled"]: "Super sampled frames",
        INTERMEDIATE_DIRS["frames"]: "Original frames",
    }

    highest_stage_num = 0
    for stage_dir, stage_name in stages.items():
        stage_path = batch_path / stage_dir
        if stage_path.exists():
            png_files = list(stage_path.glob("*.png"))
            if png_files:
                analysis["frame_count"] = max(analysis["frame_count"], len(png_files))
                current_stage_num = _get_stage_number(stage_dir)
                if current_stage_num > highest_stage_num:
                    highest_stage_num = current_stage_num
                    analysis["highest_stage"] = stage_name

    # Detect VR format and resolution from highest stage
    if highest_stage_num >= 50:  # Final frames available
        sample_frame_dirs = [
            d
            for d in [INTERMEDIATE_DIRS["left_final"], INTERMEDIATE_DIRS["right_final"], INTERMEDIATE_DIRS["vr_frames"]]
            if (batch_path / d).exists()
        ]
        for frame_dir in sample_frame_dirs:
            frame_path = batch_path / frame_dir
            sample_frames = list(frame_path.glob("*.png"))
            if sample_frames:
                try:
                    cv2 = _get_cv2()
                    sample_img = cv2.imread(str(sample_frames[0]))
                    if sample_img is not None:
                        h, w = sample_img.shape[:2]
                        analysis["resolution"] = f"{w}x{h}"

                        # Detect format based on aspect ratio
                        if frame_dir == INTERMEDIATE_DIRS["vr_frames"]:
                            if w > h * 1.5:
                                analysis["vr_format"] = "side_by_side"
                            else:
                                analysis["vr_format"] = "over_under"
                        break
                except Exception:
                    continue

    # Try to load settings
    settings_files = list(batch_path.glob("*-settings.json"))
    if settings_files:
        try:
            with open(settings_files[0], "r") as f:
                settings = json.load(f)
                analysis["settings_summary"] = _summarize_settings(settings)
        except Exception:
            pass

    # Check for audio
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    for ext in video_extensions:
        if list(batch_path.parent.parent.glob(f"uploads/{ext}")):  # Check uploads dir
            analysis["has_audio"] = True
            break

    return analysis


def create_video_from_batch(batch_path: Path, settings: Dict[str, Any]) -> Optional[Path]:
    """
    Create video from batch frames using FFmpeg.

    Args:
        batch_path: Path to batch directory containing frames
        settings: Settings dictionary with frame_source, quality, etc.

    Returns:
        Path to created video file or None if failed
    """
    batch_path = Path(batch_path)
    frame_source = settings.get("frame_source", "auto")
    quality = settings.get("quality", "medium")
    fps = settings.get("fps", "original")
    include_audio = settings.get("include_audio", False)
    output_filename = settings.get("output_filename")

    # Determine frame directory to use
    if frame_source == "auto":
        # Auto-detect highest available stage
        stages = [INTERMEDIATE_DIRS["vr_frames"], INTERMEDIATE_DIRS["left_final"], INTERMEDIATE_DIRS["right_final"]]
        frame_dir = None
        for stage in stages:
            stage_path = batch_path / stage
            if stage_path.exists() and list(stage_path.glob("*.png")):
                frame_dir = stage_path
                break
    else:
        # Use specified stage
        stage_mapping = {
            "vr_frames": INTERMEDIATE_DIRS["vr_frames"],
            "left_right_final": INTERMEDIATE_DIRS["left_final"],
            "left_right_fisheye": INTERMEDIATE_DIRS["left_distorted"],
            "left_right_basic": INTERMEDIATE_DIRS["left_frames"],
        }
        stage_name = stage_mapping.get(frame_source, INTERMEDIATE_DIRS["vr_frames"])
        frame_dir = batch_path / stage_name

    if not frame_dir or not frame_dir.exists():
        raise ValueError(f"No frames found in selected stage: {frame_source}")

    # Generate output filename
    if not output_filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_name = batch_path.name
        quality_suffix = f"_{quality}" if quality != "medium" else ""
        output_filename = f"{batch_name}_stitched_{timestamp}{quality_suffix}.mp4"

    output_path = batch_path / output_filename

    # Build FFmpeg command
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(frame_dir / "frame_%06d.png")]

    # Quality settings
    quality_settings = {
        "low": ["-crf", "28", "-preset", "fast"],
        "medium": ["-crf", "23", "-preset", "medium"],
        "high": ["-crf", "18", "-preset", "slow"],
        "lossless": ["-crf", "0", "-preset", "medium"],
    }

    cmd.extend(quality_settings.get(quality, quality_settings["medium"]))
    cmd.extend(["-pix_fmt", "yuv420p", str(output_path)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)

        return output_path
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")
        return None


def _get_stage_number(stage_dir: str) -> int:
    """Extract numeric stage number from directory name."""
    try:
        # Extract number from directory name (e.g., "99_vr_frames" -> 99)
        return int(stage_dir.split("_")[0])
    except (ValueError, IndexError):
        return 0


def _summarize_settings(settings: Dict[str, Any]) -> str:
    """Create a human-readable summary of settings."""
    summary_parts = []

    if "vr_format" in settings:
        summary_parts.append(f"Format: {settings['vr_format']}")

    if "vr_resolution" in settings:
        summary_parts.append(f"Resolution: {settings['vr_resolution']}")

    if "processing_mode" in settings:
        summary_parts.append(f"Mode: {settings['processing_mode']}")

    if "super_sample" in settings and settings["super_sample"] != "none":
        summary_parts.append(f"Super-sample: {settings['super_sample']}")

    if "fisheye_enabled" in settings and settings["fisheye_enabled"]:
        summary_parts.append("Fisheye: enabled")

    return ", ".join(summary_parts) if summary_parts else "Standard processing"
