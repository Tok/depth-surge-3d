#!/usr/bin/env python3
"""
Depth Surge 3D Web UI
Flask application for converting 2D videos to immersive 3D VR format
"""

from __future__ import annotations

import os
import time
import uuid
import subprocess
import platform
import argparse
import sys
import signal
import base64
from datetime import datetime
from pathlib import Path
from typing import Any
import cv2
import numpy as np

# Set PyTorch memory allocator config BEFORE importing torch
# This helps prevent memory fragmentation on GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Suppress warnings from dependencies
import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=SyntaxWarning)  # moviepy old regex patterns

# Import our constants and utilities
from src.depth_surge_3d.core.constants import (  # noqa: E402
    INTERMEDIATE_DIRS,
    MODEL_PATHS,
    MODEL_PATHS_METRIC,
    SOCKETIO_PING_TIMEOUT,
    SOCKETIO_PING_INTERVAL,
    SOCKETIO_SLEEP_YIELD,
    INITIAL_PROCESSING_DELAY,
    BYTES_TO_GB_DIVISOR,
    FPS_ROUND_DIGITS,
    DURATION_ROUND_DIGITS,
    ASPECT_RATIO_ROUND_DIGITS,
    ASPECT_RATIO_SBS_THRESHOLD,
    ASPECT_RATIO_OU_THRESHOLD,
    SESSION_ID_DISPLAY_LENGTH,
    PROGRESS_UPDATE_INTERVAL,
    PROGRESS_DECIMAL_PLACES,
    PROGRESS_STEP_WEIGHTS,
    PREVIEW_UPDATE_INTERVAL,
    PREVIEW_DOWNSCALE_WIDTH,
    FFMPEG_OVERWRITE_FLAG,
    FFMPEG_CRF_HIGH_QUALITY,
    FFMPEG_CRF_MEDIUM_QUALITY,
    FFMPEG_CRF_FAST_QUALITY,
    FFMPEG_DEFAULT_PRESET,
    FFMPEG_PIX_FORMAT,
    VIDEO_CREATION_TIMEOUT,
    DEFAULT_FALLBACK_FPS,
    DEFAULT_SERVER_PORT,
    DEFAULT_SERVER_HOST,
    SIGNAL_SHUTDOWN_TIMEOUT,
)
from src.depth_surge_3d.utils.system.console import warning as console_warning  # noqa: E402

from flask import Flask, render_template, request, jsonify  # noqa: E402
from flask_socketio import SocketIO  # noqa: E402

# NOTE: torch is imported later (line ~960) to avoid CUDA initialization issues

# Add src to path for package imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from depth_surge_3d.rendering import create_stereo_projector  # noqa: E402
from depth_surge_3d.processing import VideoProcessor  # noqa: E402

# Global flags and state
VERBOSE = False
SHUTDOWN_FLAG = False
ACTIVE_PROCESSES = set()


def vprint(*args: Any, **kwargs: Any) -> None:
    """Print only if verbose mode is enabled"""
    if VERBOSE:
        print(*args, **kwargs)


def _get_version_info() -> tuple[str, str]:
    """Get version and git commit ID"""
    try:
        from depth_surge_3d import __version__

        version = __version__
    except ImportError:
        version = "dev"

    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=1,
        ).stdout.strip()
        if not git_commit:
            git_commit = "unknown"
    except Exception:
        git_commit = "unknown"

    return version, git_commit


def _rgb_to_ansi256(r: int, g: int, b: int) -> int:
    """Convert RGB (0-5 range) to ANSI 256 color code"""
    return 16 + 36 * r + 6 * g + b


def _get_lime_color(position: float) -> str:
    """Lerp between dark green and brand lime (#39ff14) for smooth gradient"""
    # Brand lime color from CSS: #39ff14 = RGB(57, 255, 20)
    # Converted to ANSI 0-5 scale: (1, 5, 0)
    # Create gradient from dark green to brand lime
    start_rgb = (0, 3, 0)  # Dark green
    end_rgb = (1, 5, 0)  # Brand lime (#39ff14)

    # Lerp each RGB component with rounding for smoothness
    r = round(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * position)
    g = round(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * position)
    b = round(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * position)

    # Clamp to valid range (0-5)
    r = max(0, min(5, r))
    g = max(0, min(5, g))
    b = max(0, min(5, b))

    # Convert to ANSI 256 color code
    color_code = _rgb_to_ansi256(r, g, b)
    return f"\033[38;5;{color_code}m"


def _print_banner_border(border_width: int, char: str, row_offset: int, total_rows: int) -> None:
    """Print border with diagonal gradient matching text"""
    margin = 0.0  # No margin - let gradient span full banner
    max_diagonal = total_rows + border_width

    border = " " + _get_lime_color(0.0 - margin) + "█"  # Leading space
    for i in range(border_width):
        # Calculate diagonal position with stretched gradient
        raw_pos = (row_offset + i) / max_diagonal
        stretched_pos = (raw_pos * (1 + 2 * margin)) - margin
        border += f"{_get_lime_color(stretched_pos)}{char}"
    border += _get_lime_color(1.0 + margin) + "█\033[0m"
    print(border)


def _print_banner_line(line: str, row_idx: int, num_rows: int, max_diagonal: int) -> None:
    """Print single banner line with diagonal gradient"""
    reset = "\033[0m"
    margin = 0.0  # No margin - let gradient span full banner

    # Left border with leading space
    left_pos = row_idx / (num_rows + 1)
    colored_line = f" {_get_lime_color(left_pos)}█{reset} "  # Leading space

    # Apply diagonal gradient to entire text
    for col_idx, char in enumerate(line):
        raw_pos = (row_idx + col_idx) / max_diagonal
        stretched_pos = (raw_pos * (1 + 2 * margin)) - margin
        colored_line += f"{_get_lime_color(stretched_pos)}{char}"

    # Right border
    right_pos = (row_idx + 1) / (num_rows + 1)
    colored_line += f"{reset} {_get_lime_color(right_pos)}█{reset}"
    print(colored_line)


def print_banner() -> None:
    """Print Depth Surge 3D banner with diagonal lime gradient"""
    version, git_commit = _get_version_info()

    blue_accent = "\033[38;5;39m"
    bright_blue = "\033[38;5;51m"
    reset = "\033[0m"

    banner_lines = [
        "░█▀▄░█▀▀░█▀█░▀█▀░█░█░░░█▀▀░█░█░█▀▄░█▀▀░█▀▀░░░▀▀█░█▀▄░",
        "░█░█░█▀▀░█▀▀░░█░░█▀█░░░▀▀█░█░█░█▀▄░█░█░█▀▀░░░░▀▄░█░█░",
        "░▀▀░░▀▀▀░▀░░░░▀░░▀░▀░░░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░░░▀▀░░▀▀░░",
    ]

    print()

    # Calculate dimensions
    border_width = len(banner_lines[0]) + 2
    num_rows = len(banner_lines)
    line_length = len(banner_lines[0])
    max_diagonal = num_rows + line_length - 2

    # Print banner with borders (diagonal gradient on borders too)
    total_rows = num_rows + 2  # +2 for top and bottom borders
    _print_banner_border(border_width, "▀", row_offset=0, total_rows=total_rows)
    for row_idx, line in enumerate(banner_lines):
        _print_banner_line(
            line, row_idx + 1, num_rows, max_diagonal
        )  # +1 to account for top border
    _print_banner_border(border_width, "▄", row_offset=num_rows + 1, total_rows=total_rows)

    # GitHub repo link (with leading space for alignment)
    repo_link = "https://github.com/Tok/depth-surge-3d"
    padding = (border_width - len(repo_link)) // 2 + 1  # +1 for leading space
    print(f"{' ' * padding}{blue_accent}{repo_link}{reset}")

    # Version and commit info (with leading space for alignment)
    version_info = f"v{version} [{git_commit}]"
    version_padding = (border_width - len(version_info)) // 2 + 1  # +1 for leading space
    print(
        f"{' ' * version_padding}v{bright_blue}{version}{reset} [{bright_blue}{git_commit}{reset}]"
    )

    print()


def cleanup_processes() -> None:
    """Clean up any active processing threads or subprocesses"""
    global SHUTDOWN_FLAG

    SHUTDOWN_FLAG = True
    vprint("Cleaning up active processes...")

    # Kill any ffmpeg processes related to this app
    try:
        subprocess.run(["pkill", "-f", "ffmpeg.*depth-surge"], check=False, capture_output=True)
    except Exception:
        pass

    # Clean up any tracked processes
    for proc in list(ACTIVE_PROCESSES):
        try:
            if hasattr(proc, "terminate"):
                proc.terminate()
            elif hasattr(proc, "kill"):
                proc.kill()
        except Exception:
            pass

    ACTIVE_PROCESSES.clear()
    vprint("Process cleanup completed")


def signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals"""
    print(f"\nReceived signal {signum}, shutting down gracefully...")

    # Stop any active processing
    if current_processing["active"]:
        print("   Stopping active video processing...")
        current_processing["stop_requested"] = True
        # Wait a moment for cleanup
        if current_processing["thread"] and current_processing["thread"].is_alive():
            current_processing["thread"].join(timeout=SIGNAL_SHUTDOWN_TIMEOUT)

    # Clean up all processes
    cleanup_processes()
    sys.exit(0)


app = Flask(__name__)
app.config["SECRET_KEY"] = "depth-surge-3d-secret"
app.config["OUTPUT_FOLDER"] = str(Path("output").resolve())
# Use threading async_mode and disable ping timeout for long-running tasks
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
    async_mode="threading",
    ping_timeout=SOCKETIO_PING_TIMEOUT,
    ping_interval=SOCKETIO_PING_INTERVAL,
)


@app.teardown_appcontext
def cleanup_on_teardown(error: Exception | None) -> None:
    """Clean up processes when Flask shuts down"""
    if error:
        vprint(f"App teardown due to error: {error}")
    cleanup_processes()


# Global variables for processing state
current_processing = {
    "active": False,
    "progress": 0,
    "stage": "",
    "total_frames": 0,
    "current_frame": 0,
    "session_id": None,
    "stop_requested": False,
    "thread": None,
}


def ensure_directories() -> None:
    """Ensure output directory exists"""
    Path(app.config["OUTPUT_FOLDER"]).mkdir(exist_ok=True)


def get_video_info(video_path: str | Path) -> dict[str, Any] | None:
    """Extract video information using OpenCV"""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        # Calculate aspect ratio
        aspect_ratio = width / height if height > 0 else 1.0

        return {
            "width": width,
            "height": height,
            "fps": round(fps, FPS_ROUND_DIGITS),
            "duration": round(duration, DURATION_ROUND_DIGITS),
            "frame_count": frame_count,
            "aspect_ratio": round(aspect_ratio, ASPECT_RATIO_ROUND_DIGITS),
            "aspect_ratio_text": f"{width}:{height}",
        }
    finally:
        cap.release()


def find_source_video(directory: Path) -> Path | None:
    """
    Find the source video file in an output directory.

    Looks for video files excluding processed outputs (those with _3D_ in name).

    Args:
        directory: Directory to search

    Returns:
        Path to source video, or None if not found
    """
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"]

    for ext in video_extensions:
        for video_file in directory.glob(f"*{ext}"):
            # Skip processed output files (they contain _3D_)
            if "_3D_" not in video_file.name:
                return video_file

    return None


def get_system_info() -> dict[str, Any]:
    """Get system information including GPU details"""
    import torch  # Import here to avoid early CUDA initialization

    info = {
        "gpu_device": "CPU",
        "vram_usage": "N/A",
        "device_mode": "CPU",
        "cuda_available": False,
    }

    try:
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["gpu_device"] = torch.cuda.get_device_name(0)
            info["device_mode"] = "GPU"

            # Get VRAM usage
            try:
                allocated = torch.cuda.memory_allocated() / BYTES_TO_GB_DIVISOR
                total_memory = (
                    torch.cuda.get_device_properties(0).total_memory / BYTES_TO_GB_DIVISOR
                )
                info["vram_usage"] = f"{allocated:.1f}GB / {total_memory:.1f}GB"
            except Exception as vram_error:
                vprint(f"Error getting VRAM details: {vram_error}")
                info["vram_usage"] = "N/A"
    except Exception as e:
        vprint(f"Error getting GPU info: {e}")

    return info


class ProgressCallback:
    """Enhanced callback class to track processing progress for both serial and batch modes"""

    def __init__(
        self,
        session_id: str,
        total_frames: int,
        processing_mode: str = "serial",
        enable_live_preview: bool = True,
        preview_update_interval: float = PREVIEW_UPDATE_INTERVAL,
    ) -> None:
        self.session_id = session_id
        self.total_frames = total_frames
        self.processing_mode = processing_mode
        self.current_frame = 0
        self.last_update_time = 0
        self.current_phase = "extraction"  # extraction, processing, video
        self.step_start_times = {}  # Track start time for each step
        self.current_step_name = None
        self.start_time = time.time()  # For ETA calculation
        self.step_frame_times = []  # Track recent frame times for current step
        self.step_start_progress = 0  # Track progress when step started

        # Preview tracking
        self.enable_live_preview = enable_live_preview
        self.last_preview_time = 0
        self.preview_interval = preview_update_interval
        self.preview_downscale_width = PREVIEW_DOWNSCALE_WIDTH

        # Step tracking (used for all modes)
        # Note: These must match the step names sent from video_processor.py
        self.steps = [
            "Frame Extraction",  # Step 1: FFmpeg extracts frames
            "Depth Map Generation",  # Step 2: AI generates depth maps
            "Stereo Pair Creation",  # Step 3: Create L/R stereo pairs (includes frame loading)
            "Fisheye Distortion",  # Step 4: Apply distortion (optional)
            "Crop Frames",  # Step 5: Crop frames for VR
            "AI Upscaling",  # Step 6: AI upscaling (optional)
            "VR Assembly",  # Step 7: Assemble final VR frames
            "Video Creation",  # Step 8: FFmpeg creates video
        ]
        # Weighted progress based on actual timing measurements
        # [2%, 35%, 20%, 8%, 2%, 18%, 8%, 7%] = 100%
        self.step_weights = PROGRESS_STEP_WEIGHTS
        self.current_step_index = 0
        self.step_progress = 0
        self.step_total = 0

    def send_preview_frame(
        self,
        frame_path: Path,
        frame_type: str,
        frame_number: int,
    ) -> None:
        """
        Send preview frame via websocket.

        Args:
            frame_path: Path to the frame image file
            frame_type: Type of frame ("depth_map", "stereo_left", "stereo_right", "vr_frame")
            frame_number: Frame number being processed
        """
        # Check if preview is enabled
        if not self.enable_live_preview:
            return

        current_time = time.time()

        # Throttle preview updates
        if current_time - self.last_preview_time < self.preview_interval:
            return

        try:
            # Read frame
            frame = cv2.imread(str(frame_path))
            if frame is None:
                vprint(f"Preview: Failed to read frame {frame_path}")
                return

            # Validate dimensions
            height, width = frame.shape[:2]
            if width <= 0 or height <= 0:
                vprint(f"Preview: Invalid dimensions {width}x{height} for {frame_path}")
                return

            # Cap input size to prevent excessive processing
            MAX_INPUT_DIM = 8192  # Reject frames larger than 8K
            if width > MAX_INPUT_DIM or height > MAX_INPUT_DIM:
                vprint(
                    f"Preview: Frame too large ({width}x{height}), "
                    f"exceeds max {MAX_INPUT_DIM}px"
                )
                return

            # Downscale for transmission
            scale = self.preview_downscale_width / width
            new_width = self.preview_downscale_width
            new_height = int(height * scale)
            frame_small = cv2.resize(frame, (new_width, new_height))

            # Encode to base64
            _, buffer = cv2.imencode(".png", frame_small)
            img_base64 = base64.b64encode(buffer).decode("utf-8")

            # Cap base64 payload size
            MAX_PAYLOAD_KB = 500
            payload_size_kb = len(img_base64) / 1024
            if payload_size_kb > MAX_PAYLOAD_KB:
                vprint(
                    f"Preview: Payload too large ({payload_size_kb:.1f}KB), "
                    f"skipping {frame_type} frame {frame_number}"
                )
                return

            # Send via socketio
            preview_data = {
                "frame_type": frame_type,
                "frame_number": frame_number,
                "image_data": f"data:image/png;base64,{img_base64}",
                "dimensions": {"width": new_width, "height": new_height},
            }

            socketio.emit("frame_preview", preview_data, room=self.session_id)

            self.last_preview_time = current_time

        except Exception as e:
            # Log error but don't interrupt processing
            vprint(
                f"Preview: Error sending {frame_type} frame {frame_number}: "
                f"{type(e).__name__}: {e}"
            )

    def send_preview_frame_from_array(
        self,
        frame_array: np.ndarray,
        frame_type: str,
        frame_number: int,
    ) -> None:
        """
        Send preview frame directly from numpy array (no disk I/O).

        Args:
            frame_array: Frame as numpy array (BGR format)
            frame_type: Type of frame ("depth_map", "stereo_left", "upscaled_left", etc)
            frame_number: Frame number being processed
        """
        # Check if preview is enabled
        if not self.enable_live_preview:
            return

        current_time = time.time()

        # Throttle preview updates
        if current_time - self.last_preview_time < self.preview_interval:
            return

        try:
            # Validate dimensions
            height, width = frame_array.shape[:2]
            if width <= 0 or height <= 0:
                vprint(f"Preview: Invalid dimensions {width}x{height}")
                return

            # Cap input size
            MAX_INPUT_DIM = 8192
            if width > MAX_INPUT_DIM or height > MAX_INPUT_DIM:
                vprint(
                    f"Preview: Frame too large ({width}x{height}), "
                    f"exceeds max {MAX_INPUT_DIM}px"
                )
                return

            # Downscale for transmission
            scale = self.preview_downscale_width / width
            new_width = self.preview_downscale_width
            new_height = int(height * scale)
            frame_small = cv2.resize(frame_array, (new_width, new_height))

            # Encode to base64
            _, buffer = cv2.imencode(".png", frame_small)
            img_base64 = base64.b64encode(buffer).decode("utf-8")

            # Cap base64 payload size
            MAX_PAYLOAD_KB = 500
            payload_size_kb = len(img_base64) / 1024
            if payload_size_kb > MAX_PAYLOAD_KB:
                vprint(
                    f"Preview: Payload too large ({payload_size_kb:.1f}KB), "
                    f"skipping {frame_type} frame {frame_number}"
                )
                return

            # Send via socketio
            preview_data = {
                "frame_type": frame_type,
                "frame_number": frame_number,
                "image_data": f"data:image/png;base64,{img_base64}",
                "dimensions": {"width": new_width, "height": new_height},
            }

            socketio.emit("frame_preview", preview_data, room=self.session_id)

            self.last_preview_time = current_time

        except Exception as e:
            # Log error but don't interrupt processing
            vprint(
                f"Preview: Error sending {frame_type} frame {frame_number}: "
                f"{type(e).__name__}: {e}"
            )

    def _calculate_eta(self, current_progress: float) -> str | None:
        """
        Calculate estimated time remaining using adaptive per-step timing.

        This uses actual measured time-per-frame for the current step and
        weights for remaining steps, providing much more accurate ETAs
        especially for slow steps like ESRGAN upscaling.

        Args:
            current_progress: Current progress percentage (0-100)

        Returns:
            Formatted ETA string (e.g., "5m 23s") or None if not enough data
        """
        if current_progress <= 0:
            return None

        current_time = time.time()
        elapsed = current_time - self.start_time

        # Need at least 3 seconds of data for reasonable estimate
        if elapsed < 3:
            return None

        # Strategy: Estimate remaining time for current step + remaining steps
        remaining_time = 0

        # 1. Estimate remaining time for current step
        if self.step_total > 0 and self.step_progress > 0:
            # Use actual measured rate for current step
            step_start_time = self.step_start_times.get(self.current_step_name, self.start_time)
            step_elapsed = current_time - step_start_time

            # Need at least 2 seconds into current step for accurate measurement
            if step_elapsed >= 2:
                frames_completed = self.step_progress
                frames_remaining = self.step_total - self.step_progress

                # Calculate time per frame for this step
                time_per_frame = step_elapsed / max(frames_completed, 1)

                # Estimate remaining time for this step
                step_remaining_time = time_per_frame * frames_remaining
                remaining_time += step_remaining_time

        # 2. Estimate time for remaining steps using weight-based projection
        # Use the overall rate as a fallback for future steps
        if self.current_step_index < len(self.steps) - 1:
            # Estimate time based on overall average rate (fallback for steps we haven't measured)
            if current_progress > 5:  # Need some progress to estimate
                avg_time_per_percent = elapsed / current_progress
                remaining_percent = 100 - current_progress
                fallback_estimate = avg_time_per_percent * remaining_percent

                # Weight between step-based estimate and fallback
                # Use step-based more heavily when we have good data
                if remaining_time > 0:
                    # We have a step-based estimate, use it primarily
                    remaining_time = remaining_time * 0.8 + fallback_estimate * 0.2
                else:
                    # Fall back to overall rate
                    remaining_time = fallback_estimate

        # Fallback: use simple overall progress rate if we don't have step data
        if remaining_time <= 0 and current_progress > 0:
            progress_ratio = current_progress / 100.0
            estimated_total_time = elapsed / progress_ratio
            remaining_time = estimated_total_time - elapsed

        if remaining_time < 0:
            return None

        return self._format_time(remaining_time)

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time (e.g., '5m 23s' or '2h 15m')."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:  # Less than 1 hour
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:  # 1 hour or more
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def update_progress(  # noqa: C901
        self,
        stage: str,
        frame_num: int | None = None,
        phase: str | None = None,
        step_name: str | None = None,
        step_progress: int | None = None,
        step_total: int | None = None,
    ) -> None:
        import time

        # Check if stop has been requested
        if current_processing.get("stop_requested", False):
            raise InterruptedError("Processing stopped by user request")

        # Throttle updates to avoid threading issues
        current_time = time.time()
        if current_time - self.last_update_time < PROGRESS_UPDATE_INTERVAL:
            return
        self.last_update_time = current_time

        # Update phase if provided
        if phase:
            self.current_phase = phase

        # Track step changes and timing
        if step_name and step_name != self.current_step_name:
            # New step started
            self.current_step_name = step_name
            self.step_start_times[step_name] = current_time
            self.step_frame_times = []  # Reset frame times for new step
            self.step_start_progress = 0  # Reset progress for new step

            # Update step index
            if step_name in self.steps:
                self.current_step_index = self.steps.index(step_name)

        # Update step progress
        if step_progress is not None:
            self.step_progress = step_progress
        if step_total is not None:
            self.step_total = step_total

        if frame_num is not None:
            self.current_frame = frame_num
            current_processing["current_frame"] = frame_num

        current_processing["stage"] = stage

        # Calculate overall progress using weighted steps
        step_progress_ratio = (
            (self.step_progress / max(self.step_total, 1)) if self.step_total > 0 else 0
        )

        # Sum weights of all completed steps
        cumulative_weight = sum(self.step_weights[: self.current_step_index])

        # Add weighted progress of current step
        if self.current_step_index < len(self.step_weights):
            cumulative_weight += step_progress_ratio * self.step_weights[self.current_step_index]

        progress = round(cumulative_weight * 100, PROGRESS_DECIMAL_PLACES)

        current_processing["progress"] = round(progress, PROGRESS_DECIMAL_PLACES)
        current_processing["phase"] = self.current_phase
        current_processing["processing_mode"] = self.processing_mode
        current_processing["step_name"] = step_name or self.current_step_name
        current_processing["step_progress"] = self.step_progress
        current_processing["step_total"] = self.step_total
        current_processing["step_index"] = self.current_step_index

        # Calculate ETA
        eta_str = self._calculate_eta(progress)

        # Emit progress update (always include step data for UI)
        progress_data = {
            "progress": current_processing["progress"],
            "stage": stage,
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "phase": self.current_phase,
            "processing_mode": self.processing_mode,
            "step_name": self.current_step_name or "",
            "step_progress": self.step_progress,
            "step_total": self.step_total,
            "step_index": self.current_step_index,
            "total_steps": len(self.steps),
            "eta": eta_str,  # Add ETA
        }

        # Console output - show both overall and step progress
        step_percent = (
            (self.step_progress / max(self.step_total, 1)) * 100 if self.step_total > 0 else 0
        )
        eta_suffix = f" | ETA: {eta_str}" if eta_str else ""
        progress_msg = (
            f"Overall: {progress:05.1f}% | "
            f"Step: {step_percent:03.0f}% ({self.step_progress:04d}/{self.step_total:04d}) | "
            f"{stage}{eta_suffix}"
        )
        print(progress_msg)

        try:
            # Emit progress (socketio.start_background_task handles context automatically)
            socketio.emit("progress_update", progress_data, room=self.session_id)
            # Yield control to allow message to be sent immediately (fixes buffering issue)
            socketio.sleep(SOCKETIO_SLEEP_YIELD)
        except Exception as e:
            print(console_warning(f"Error emitting progress: {e}"))
            import traceback

            traceback.print_exc()

    def get_step_duration(self):
        """Get duration of current step in seconds."""
        import time

        if self.current_step_name and self.current_step_name in self.step_start_times:
            return time.time() - self.step_start_times[self.current_step_name]
        return 0

    def finish(self, message: str = "Processing complete"):
        """Finish progress tracking (compatibility with ProgressTracker interface)."""
        print(f"{message}")
        try:
            socketio.emit(
                "processing_complete",
                {"success": True, "message": message},
                room=self.session_id,
            )
            # Allow time for message to be sent
            socketio.sleep(SOCKETIO_SLEEP_YIELD)
        except Exception as e:
            print(console_warning(f"Error emitting completion: {e}"))


def process_video_async(  # noqa: C901
    session_id: str, video_path: str | Path, settings: dict[str, Any], output_dir: str | Path
) -> None:
    """Process video in background thread"""
    import torch  # Import here to avoid CUDA initialization issues in main thread

    try:
        current_processing["active"] = True
        current_processing["session_id"] = session_id
        current_processing["stop_requested"] = False

        # Check CUDA availability in this thread
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, using CPU")

        # Initialize projector with depth model
        # Get depth model version (v2 or v3)
        depth_model_version = settings.get("depth_model_version", "v3")  # Default to V3
        model_size = settings.get("model_size", "vitb")  # Default to Base for 16GB GPUs
        use_metric = settings.get("use_metric_depth", True)  # Default to metric depth

        device = settings.get("device", "auto")
        if device == "auto":
            device = "cuda" if cuda_available else "cpu"

        # Fail fast if GPU requested but not available
        if device == "cuda" and not cuda_available:
            error_msg = (
                "GPU (CUDA) requested but not available. "
                "Please select 'Auto' or 'Force CPU' in Processing Device settings."
            )
            raise Exception(error_msg)

        # For V2, select the appropriate model path based on size and metric/relative
        # For V3, model_path is just the model name (e.g., "large", "base", "small")
        if depth_model_version == "v2":
            if use_metric:
                model_paths_dict = MODEL_PATHS_METRIC
                depth_type = "Metric"
            else:
                model_paths_dict = MODEL_PATHS
                depth_type = "Relative"

            model_path = settings.get(
                "model_path",
                model_paths_dict.get(model_size, MODEL_PATHS_METRIC["vitb"]),
            )
            print(
                f"Loading Video-Depth-Anything V2: {model_size.upper()} {depth_type} from: {model_path}"
            )
        else:
            # For V3, map model_size to DA3 model names
            da3_model_map = {"vits": "small", "vitb": "base", "vitl": "large"}
            model_path = da3_model_map.get(model_size, "large")
            print(f"Loading Depth-Anything V3: {model_path.upper()} model (metric: {use_metric})")

        print(f"Using device: {device.upper()}")

        projector = create_stereo_projector(
            model_path,
            device,
            metric=use_metric,
            depth_model_version=depth_model_version,
        )

        # Ensure the model is loaded before processing
        if not projector.depth_estimator.load_model():
            raise Exception("Failed to load depth estimation model")

        # Get video info for progress tracking
        video_info = get_video_info(video_path)
        if not video_info:
            raise Exception("Could not read video file")

        # Calculate expected frame count based on time range and ORIGINAL fps (since we extract at original fps)
        start_time = settings.get("start_time")
        end_time = settings.get("end_time")

        # Always use original FPS for frame count calculation (interpolation happens at the end)
        original_fps = video_info["fps"]

        # Calculate actual frame range that will be extracted (matching VideoProcessor logic)
        from depth_surge_3d.utils.path_utils import calculate_frame_range

        start_frame, end_frame = calculate_frame_range(
            video_info["frame_count"], original_fps, start_time, end_time
        )
        expected_frames = end_frame - start_frame

        processing_mode = settings.get("processing_mode", "serial")
        enable_live_preview = settings.get("enable_live_preview", True)
        preview_update_interval = settings.get("preview_update_interval", PREVIEW_UPDATE_INTERVAL)
        callback = ProgressCallback(
            session_id,
            expected_frames,
            processing_mode,
            enable_live_preview,
            preview_update_interval,
        )

        # Give client time to join the session room before starting processing
        socketio.sleep(INITIAL_PROCESSING_DELAY)

        # Use the appropriate processor based on processing mode
        if processing_mode == "batch":
            from depth_surge_3d.processing.batch_processor import BatchProcessor

            processor = BatchProcessor(projector.depth_estimator)
        else:
            processor = VideoProcessor(projector.depth_estimator)

        # Calculate resolution settings that VideoProcessor expects
        from depth_surge_3d.utils.domain.resolution import (
            get_resolution_dimensions,
            calculate_vr_output_dimensions,
            auto_detect_resolution,
        )

        # Resolve VR resolution if auto
        vr_resolution = settings.get("vr_resolution", "auto")
        if vr_resolution == "auto":
            vr_resolution = auto_detect_resolution(
                video_info["width"],
                video_info["height"],
                settings.get("vr_format", "side_by_side"),
            )

        # Get resolution dimensions
        per_eye_width, per_eye_height = get_resolution_dimensions(vr_resolution)
        vr_output_width, vr_output_height = calculate_vr_output_dimensions(
            per_eye_width, per_eye_height, settings.get("vr_format", "side_by_side")
        )

        # Add calculated dimensions to settings
        settings.update(
            {
                "per_eye_width": per_eye_width,
                "per_eye_height": per_eye_height,
                "vr_output_width": vr_output_width,
                "vr_output_height": vr_output_height,
                "source_width": video_info["width"],
                "source_height": video_info["height"],
                "source_fps": video_info["fps"],
            }
        )

        success = processor.process(
            video_path=video_path,
            output_dir=output_dir,
            video_properties=video_info,
            settings=settings,
            progress_callback=callback,
        )

        if not success:
            raise Exception("Video processing failed")

        # Processing complete
        try:
            socketio.emit(
                "processing_complete",
                {
                    "success": True,
                    "output_dir": str(output_dir),
                    "message": "Video processing completed successfully!",
                },
                room=session_id,
            )
            # Allow time for message to be sent before thread terminates
            socketio.sleep(SOCKETIO_SLEEP_YIELD)
        except Exception as e:
            print(console_warning(f"Error emitting completion: {e}"))

    except InterruptedError as e:
        # Handle user-requested stop
        try:
            socketio.emit(
                "processing_stopped",
                {"success": True, "message": str(e)},
                room=session_id,
            )
            # Allow time for message to be sent before thread terminates
            socketio.sleep(SOCKETIO_SLEEP_YIELD)
        except Exception as emit_error:
            print(console_warning(f"Error emitting stop: {emit_error}"))

    except Exception as e:
        try:
            socketio.emit("processing_error", {"success": False, "error": str(e)}, room=session_id)
            # Allow time for message to be sent before thread terminates
            socketio.sleep(SOCKETIO_SLEEP_YIELD)
        except Exception as emit_error:
            print(console_warning(f"Error emitting error: {emit_error}"))
        print(f"Processing error: {e}")  # Always print errors

    finally:
        current_processing["active"] = False
        current_processing["session_id"] = None
        current_processing["stop_requested"] = False
        current_processing["thread"] = None


@app.route("/")
def index():
    """Main page"""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video() -> tuple[dict[str, Any], int] | tuple[Any, int]:
    """Handle video upload - saves directly to output directory with audio extraction"""
    from src.depth_surge_3d.utils.path_utils import sanitize_filename

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Create timestamped output directory with sanitized filename
    original_filename = sanitize_filename(file.filename)
    video_name = Path(original_filename).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(app.config["OUTPUT_FOLDER"]) / f"{int(time.time())}_{video_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    vprint(f"Created output directory: {output_dir}")
    vprint(f"  Directory exists: {output_dir.exists()}")

    # Save video to output directory preserving original filename
    video_path = output_dir / original_filename
    vprint(f"Saving video as: {video_path.name}")
    file.save(video_path)
    vprint(f"  Video saved: {video_path.exists()}")

    # Get video information
    video_info = get_video_info(video_path)
    if not video_info:
        return jsonify({"error": "Invalid video file"}), 400

    # Extract high-quality audio to FLAC immediately (if video has audio)
    audio_path = output_dir / "original_audio.flac"
    try:
        # First check if video has an audio stream
        probe_result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if probe_result.stdout.strip() == "audio":
            # Video has audio, extract it
            print(f"Extracting audio to: {audio_path}")
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",  # Overwrite existing files
                    "-i",
                    str(video_path),
                    "-vn",  # No video
                    "-acodec",
                    "flac",  # FLAC codec for lossless audio
                    "-compression_level",
                    "8",  # Maximum compression (still lossless)
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                print(f"Warning: Audio extraction failed: {result.stderr}")
                audio_path = None
            elif audio_path.exists():
                print(f"Audio successfully extracted: {audio_path}")
            else:
                print(
                    f"Warning: Audio extraction reported success but file not found at: {audio_path}"
                )
                audio_path = None
        else:
            # No audio stream in video
            audio_path = None
    except Exception as e:
        print(f"Warning: Audio extraction error: {e}")
        audio_path = None

    return jsonify(
        {
            "success": True,
            "filename": video_path.name,
            "output_dir": str(output_dir.resolve()),  # Ensure absolute path
            "video_info": video_info,
            "has_audio": audio_path is not None and audio_path.exists(),
        }
    )


@app.route("/process", methods=["POST"])
def start_processing() -> tuple[dict[str, Any], int] | tuple[Any, int]:
    """Start video processing"""
    if current_processing["active"]:
        return jsonify({"error": "Processing already in progress"}), 400

    data = request.json
    output_dir_str = data.get("output_dir")
    settings = data.get("settings", {})

    if not output_dir_str:
        return jsonify({"error": "No output directory provided"}), 400

    output_dir = Path(output_dir_str).resolve()  # Resolve to absolute path
    vprint(f"Processing request for output directory: {output_dir}")

    # Security: Ensure output_dir is within OUTPUT_FOLDER (prevent path traversal)
    allowed_base = Path(app.config["OUTPUT_FOLDER"]).resolve()
    try:
        output_dir.relative_to(allowed_base)
    except ValueError:
        vprint("ERROR: Path traversal attempt detected!")
        vprint(f"  Requested path: {output_dir}")
        vprint(f"  Allowed base: {allowed_base}")
        return (
            jsonify({"error": "Invalid output directory: must be within output folder"}),
            403,
        )

    if not output_dir.exists():
        vprint("ERROR: Output directory not found!")
        vprint(f"  Requested path: {output_dir_str}")
        vprint(f"  Resolved path: {output_dir}")
        vprint(f"  Exists: {output_dir.exists()}")
        vprint(f"  Current working directory: {Path.cwd()}")
        return jsonify({"error": f"Output directory not found: {output_dir}"}), 404

    # Find the source video file in output directory
    video_path = find_source_video(output_dir)

    if not video_path:
        return (
            jsonify({"error": "Source video file not found in output directory"}),
            404,
        )

    # Generate session ID
    session_id = str(uuid.uuid4())

    # Start processing in background using socketio's method for proper context handling
    thread = socketio.start_background_task(
        process_video_async, session_id, video_path, settings, output_dir
    )
    current_processing["thread"] = thread

    return jsonify({"success": True, "session_id": session_id, "output_dir": str(output_dir)})


@app.route("/stop", methods=["POST"])
def stop_processing() -> dict[str, Any]:
    """Stop current processing"""
    data = request.json
    session_id = data.get("session_id")

    if not current_processing["active"]:
        return jsonify({"success": False, "error": "No processing currently active"})

    if session_id != current_processing["session_id"]:
        return jsonify({"success": False, "error": "Invalid session ID"})

    # Request stop
    current_processing["stop_requested"] = True

    return jsonify({"success": True, "message": "Stop request sent"})


@app.route("/resume", methods=["POST"])
def resume_processing():
    """Resume processing from a previous interrupted batch"""
    if current_processing["active"]:
        return jsonify({"error": "Processing already in progress"}), 400

    data = request.json
    output_dir = data.get("output_dir")

    if not output_dir:
        return jsonify({"error": "No output directory provided"}), 400

    output_path = Path(output_dir)
    if not output_path.exists():
        return jsonify({"error": "Output directory does not exist"}), 404

    # Look for source video in the output directory
    source_video = find_source_video(output_path)

    if not source_video:
        return (
            jsonify({"error": "Could not find source video file in output directory for resuming"}),
            404,
        )

    # Try to detect settings from existing files/directories
    settings = detect_resume_settings(output_path)

    # Generate session ID
    session_id = str(uuid.uuid4())

    # Start processing in background using socketio's method for proper context handling
    thread = socketio.start_background_task(
        process_video_async, session_id, source_video, settings, output_path
    )
    current_processing["thread"] = thread

    return jsonify({"success": True, "session_id": session_id, "output_dir": str(output_path)})


def detect_resume_settings(output_path):
    """Detect processing settings from existing output directory"""
    settings = {
        "vr_format": "side_by_side",
        "baseline": 0.065,
        "focal_length": 1000,
        "preserve_audio": True,
        "keep_intermediates": True,
        "device": "auto",
        "super_sample": "auto",
        "apply_distortion": True,
        "fisheye_projection": "stereographic",
        "fisheye_fov": 105,
        "crop_factor": 1.0,
        "vr_resolution": "auto",
        "fisheye_crop_factor": 1.0,
        "hole_fill_quality": "fast",
    }

    # Try to detect VR format from directory structure
    if (output_path / INTERMEDIATE_DIRS["vr_frames"]).exists():
        # Check if there are side-by-side or over-under frames
        vr_files = list((output_path / INTERMEDIATE_DIRS["vr_frames"]).glob("*.png"))
        if vr_files:
            sample_frame = cv2.imread(str(vr_files[0]))
            if sample_frame is not None:
                height, width = sample_frame.shape[:2]
                if width > height * ASPECT_RATIO_SBS_THRESHOLD:  # Likely side-by-side
                    settings["vr_format"] = "side_by_side"
                elif height > width * ASPECT_RATIO_OU_THRESHOLD:  # Likely over-under
                    settings["vr_format"] = "over_under"

    return settings


@app.route("/status")
def get_status():
    """Get current processing status"""
    return jsonify(current_processing)


@app.route("/system_info")
def get_system_info_endpoint():
    """Get system information"""
    return jsonify(get_system_info())


@app.route("/detect_resumable")
def detect_resumable_jobs():
    """Detect incomplete processing jobs that can be resumed"""
    output_folder = Path(app.config["OUTPUT_FOLDER"])
    if not output_folder.exists():
        return jsonify({"success": True, "jobs": []})

    resumable_jobs = []

    try:
        # Scan output directories for incomplete jobs
        for batch_dir in output_folder.iterdir():
            if not batch_dir.is_dir():
                continue

            # Check for source video file
            source_video = find_source_video(batch_dir)

            if not source_video:
                continue

            # Check if processing is incomplete (has frames but no final video)
            has_frames = (batch_dir / INTERMEDIATE_DIRS["frames"]).exists()
            has_final_video = any(batch_dir.glob("*_3D_*.mp4"))

            if has_frames and not has_final_video:
                # This is a resumable job
                analysis = analyze_batch_directory(batch_dir)
                resumable_jobs.append(
                    {
                        "path": str(batch_dir),
                        "name": batch_dir.name,
                        "highest_stage": analysis.get("highest_stage", "unknown"),
                        "frame_count": analysis.get("frame_count", 0),
                        "vr_format": analysis.get("vr_format", "unknown"),
                        "resolution": analysis.get("resolution", "unknown"),
                    }
                )

        # Sort by modification time (most recent first)
        resumable_jobs.sort(key=lambda x: Path(x["path"]).stat().st_mtime, reverse=True)

        return jsonify({"success": True, "jobs": resumable_jobs})

    except Exception as e:
        vprint(f"Error detecting resumable jobs: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/open_directory", methods=["POST"])
def open_directory():
    """Open directory in file explorer"""
    data = request.json
    directory_path = data.get("path")

    if not directory_path or not os.path.exists(directory_path):
        return jsonify({"success": False, "error": "Invalid directory path"})

    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(directory_path)
        elif system == "Darwin":  # macOS
            subprocess.run(["open", directory_path])
        else:  # Linux and others
            subprocess.run(["xdg-open", directory_path])

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


def analyze_batch_directory(batch_path):  # noqa: C901
    """Analyze batch directory to determine available processing stages and settings"""
    analysis = {
        "frame_count": 0,
        "vr_format": "unknown",
        "resolution": "unknown",
        "highest_stage": "none",
        "has_audio": False,
        "settings_summary": "unknown",
    }

    # Check for different processing stages
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
            frame_count = len(list(stage_path.glob("*.png"))) + len(list(stage_path.glob("*.jpg")))
            if frame_count > 0:
                stage_num = int(stage_dir.split("_")[0])
                if stage_num > highest_stage_num:
                    highest_stage_num = stage_num
                    analysis["highest_stage"] = stage_name
                    analysis["frame_count"] = frame_count

    # Detect VR format and resolution from highest stage
    if highest_stage_num >= 6:  # Final frames available
        sample_frame_dirs = [
            d
            for d in [
                INTERMEDIATE_DIRS["left_final"],
                INTERMEDIATE_DIRS["right_final"],
                INTERMEDIATE_DIRS["vr_frames"],
            ]
            if (batch_path / d).exists()
        ]
        for frame_dir in sample_frame_dirs:
            frame_path = batch_path / frame_dir
            sample_frames = list(frame_path.glob("*.png"))
            if sample_frames:
                try:
                    sample_img = cv2.imread(str(sample_frames[0]))
                    if sample_img is not None:
                        h, w = sample_img.shape[:2]
                        analysis["resolution"] = f"{w}x{h}"

                        # Detect format based on aspect ratio
                        if frame_dir == INTERMEDIATE_DIRS["vr_frames"]:
                            if w > h * ASPECT_RATIO_SBS_THRESHOLD:
                                analysis["vr_format"] = "side_by_side"
                            elif h > w * ASPECT_RATIO_OU_THRESHOLD:
                                analysis["vr_format"] = "over_under"
                            else:
                                analysis["vr_format"] = "square"
                        break
                except Exception:
                    pass

    # Check for pre-extracted audio file or source video in batch directory
    audio_file = batch_path / "original_audio.flac"
    if audio_file.exists():
        analysis["has_audio"] = True
    else:
        # Check for source video file
        source_video = find_source_video(batch_path)
        if source_video:
            analysis["has_audio"] = True

    # Generate settings summary
    if analysis["vr_format"] != "unknown" and analysis["resolution"] != "unknown":
        analysis["settings_summary"] = f"{analysis['vr_format']}, {analysis['resolution']}"

    return analysis


def create_video_from_batch(batch_path, settings):  # noqa: C901
    """Create video from batch frames using FFmpeg"""
    frame_source = settings.get("frame_source", "auto")
    quality = settings.get("quality", "medium")
    fps = settings.get("fps", "original")
    include_audio = settings.get("include_audio", False)
    output_filename = settings.get("output_filename")

    # Determine frame directory to use
    if frame_source == "auto":
        # Auto-detect highest available stage
        stages = [
            INTERMEDIATE_DIRS["vr_frames"],
            INTERMEDIATE_DIRS["left_final"],
            INTERMEDIATE_DIRS["right_final"],
        ]
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
        raise Exception(f"Frame directory not found: {frame_dir}")

    frame_files = sorted(frame_dir.glob("*.png"))
    if not frame_files:
        raise Exception(f"No frames found in {frame_dir}")

    # Determine output filename
    if not output_filename:
        batch_name = batch_path.name
        timestamp = datetime.now().strftime("%H%M%S")
        output_filename = f"{batch_name}_stitched_{timestamp}.mp4"

    output_path = batch_path / output_filename

    # Build FFmpeg command
    cmd = ["ffmpeg", FFMPEG_OVERWRITE_FLAG]

    # Input frames
    cmd.extend(["-framerate", str(fps) if fps != "original" else str(DEFAULT_FALLBACK_FPS)])
    cmd.extend(["-i", str(frame_dir / "frame_%06d.png")])

    # Quality settings
    quality_map = {
        "high": FFMPEG_CRF_HIGH_QUALITY,
        "medium": FFMPEG_CRF_MEDIUM_QUALITY,
        "fast": FFMPEG_CRF_FAST_QUALITY,
    }
    crf = quality_map.get(quality, FFMPEG_CRF_MEDIUM_QUALITY)

    cmd.extend(["-c:v", "libx264", "-crf", crf, "-preset", FFMPEG_DEFAULT_PRESET])
    cmd.extend(["-pix_fmt", FFMPEG_PIX_FORMAT])  # For compatibility

    # Add audio if requested and available - use pre-extracted audio file
    if include_audio:
        # Look for pre-extracted audio file in batch directory
        audio_file = batch_path / "original_audio.flac"
        if audio_file.exists():
            cmd.extend(["-i", str(audio_file)])
            cmd.extend(["-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0"])
        else:
            # Fallback: look for source video in batch directory
            source_video = find_source_video(batch_path)
            if source_video:
                cmd.extend(["-i", str(source_video)])
                cmd.extend(["-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0"])

    cmd.append(str(output_path))

    # Execute FFmpeg
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=VIDEO_CREATION_TIMEOUT)
        if result.returncode != 0:
            raise Exception(f"FFmpeg error: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise Exception("Video creation timed out")

    return output_path


@app.route("/analyze_batch", methods=["POST"])
def analyze_batch():
    """Analyze a batch directory for video stitching"""
    data = request.json
    batch_dir = data.get("batch_dir")

    if not batch_dir:
        return jsonify({"success": False, "error": "No batch directory provided"})

    batch_path = Path(batch_dir)
    if not batch_path.exists():
        return jsonify({"success": False, "error": "Batch directory does not exist"})

    try:
        analysis = analyze_batch_directory(batch_path)
        return jsonify({"success": True, "analysis": analysis})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/stitch_video", methods=["POST"])
def stitch_video():
    """Create video from batch frames"""
    data = request.json
    batch_dir = data.get("batch_dir")

    if not batch_dir:
        return jsonify({"success": False, "error": "No batch directory provided"})

    batch_path = Path(batch_dir)
    if not batch_path.exists():
        return jsonify({"success": False, "error": "Batch directory does not exist"})

    try:
        output_path = create_video_from_batch(batch_path, data)
        return jsonify({"success": True, "output_path": str(output_path)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@socketio.on("connect")
def handle_connect():
    vprint(f"Client connected: {request.sid}")


@socketio.on("disconnect")
def handle_disconnect():
    vprint(f"Client disconnected: {request.sid}")

    # Only stop processing if there are no other connected clients
    # Don't immediately stop on disconnect since users might refresh the page
    # The processing will continue and can be rejoined


@socketio.on("join_session")
def handle_join_session(data):
    session_id = data.get("session_id")
    if session_id:
        # Join the session room for progress updates
        from flask_socketio import join_room

        join_room(session_id)
        vprint(f"Client {request.sid} joined session {session_id[:SESSION_ID_DISPLAY_LENGTH]}...")

        # Send initial status to joined client
        try:
            initial_data = {
                "progress": current_processing.get("progress", 0),
                "stage": current_processing.get("stage", "Initializing..."),
                "current_frame": current_processing.get("current_frame", 0),
                "total_frames": current_processing.get("total_frames", 0),
                "phase": current_processing.get("phase", "extraction"),
                "step_index": current_processing.get("step_index", 0),
                "step_progress": current_processing.get("step_progress", 0),
                "step_total": current_processing.get("step_total", 0),
            }
            socketio.emit("progress_update", initial_data, room=session_id)
        except Exception as e:
            print(console_warning(f"Error emitting initial progress: {e}"))


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Depth Surge 3D Web UI")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (shows GET/SET requests and client details)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help=f"Port to run the server on (default: {DEFAULT_SERVER_PORT})",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_SERVER_HOST,
        help=f"Host to bind to (default: {DEFAULT_SERVER_HOST})",
    )
    args = parser.parse_args()

    # Set global verbose flag
    VERBOSE = args.verbose

    ensure_directories()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Always print banner on startup
    print_banner()

    # Only print additional startup message if not already printed by run_ui.sh
    if not os.environ.get("DEPTH_SURGE_UI_SCRIPT"):
        print("Starting Depth Surge 3D Web UI...")
        print(f"Navigate to http://localhost:{args.port}")

    # Suppress Flask/Werkzeug production warnings for desktop application
    if not args.verbose:
        import logging

        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        # Suppress the specific production deployment warning
        import warnings

        warnings.filterwarnings("ignore", message=".*Werkzeug.*production.*")

    socketio.run(
        app,
        host=args.host,
        port=args.port,
        debug=args.verbose,
        allow_unsafe_werkzeug=True,
    )
