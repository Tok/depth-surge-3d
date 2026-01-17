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
from datetime import datetime
from pathlib import Path
from typing import Any
import cv2

# Set PyTorch memory allocator config BEFORE importing torch
# This helps prevent memory fragmentation on GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Suppress warnings from dependencies
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)  # moviepy old regex patterns

# Import our constants and utilities
from src.depth_surge_3d.core.constants import (
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
from src.depth_surge_3d.utils.console import warning as console_warning

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

# NOTE: torch is imported later (line ~960) to avoid CUDA initialization issues

# Add src to path for package imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from depth_surge_3d.core.stereo_projector import create_stereo_projector
from depth_surge_3d.processing.video_processor import VideoProcessor

# Global flags and state
VERBOSE = False
SHUTDOWN_FLAG = False
ACTIVE_PROCESSES = set()


def vprint(*args: Any, **kwargs: Any) -> None:
    """Print only if verbose mode is enabled"""
    if VERBOSE:
        print(*args, **kwargs)


def cleanup_processes() -> None:
    """Clean up any active processing threads or subprocesses"""
    global ACTIVE_PROCESSES, SHUTDOWN_FLAG

    SHUTDOWN_FLAG = True
    vprint("Cleaning up active processes...")

    # Kill any ffmpeg processes related to this app
    try:
        subprocess.run(["pkill", "-f", "ffmpeg.*depth-surge"], check=False, capture_output=True)
    except:
        pass

    # Clean up any tracked processes
    for proc in list(ACTIVE_PROCESSES):
        try:
            if hasattr(proc, "terminate"):
                proc.terminate()
            elif hasattr(proc, "kill"):
                proc.kill()
        except:
            pass

    ACTIVE_PROCESSES.clear()
    vprint("Process cleanup completed")


def signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals"""
    global current_processing
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
app.config["OUTPUT_FOLDER"] = "output"
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

    def __init__(self, session_id: str, total_frames: int, processing_mode: str = "serial") -> None:
        self.session_id = session_id
        self.total_frames = total_frames
        self.processing_mode = processing_mode
        self.current_frame = 0
        self.last_update_time = 0
        self.current_phase = "extraction"  # extraction, processing, video
        self.step_start_times = {}  # Track start time for each step
        self.current_step_name = None

        # Step tracking (used for all modes)
        # Note: These must match the step names sent from video_processor.py
        self.steps = [
            "Frame Extraction",  # Step 1: FFmpeg extracts frames
            "Depth Map Generation",  # Step 2: AI generates depth maps
            "Frame Loading",  # Step 3: Load frames for stereo
            "Stereo Pair Creation",  # Step 4: Create L/R stereo pairs
            "Fisheye Distortion",  # Step 5: Apply distortion
            "Final Processing",  # Step 6: Create VR frames
            "Video Creation",  # Step 7: FFmpeg creates video
        ]
        # Weighted progress based on actual timing measurements
        # [1%, 17%, 1%, 31%, 38%, 6%, 7%] = 100%
        self.step_weights = PROGRESS_STEP_WEIGHTS
        self.current_step_index = 0
        self.step_progress = 0
        self.step_total = 0

    def update_progress(
        self,
        stage: str,
        frame_num: int | None = None,
        phase: str | None = None,
        step_name: str | None = None,
        step_progress: int | None = None,
        step_total: int | None = None,
    ) -> None:
        global current_processing
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
        }

        # Console output - show both overall and step progress
        step_percent = (
            (self.step_progress / max(self.step_total, 1)) * 100 if self.step_total > 0 else 0
        )
        progress_msg = (
            f"Overall: {progress:05.1f}% | "
            f"Step: {step_percent:03.0f}% ({self.step_progress:04d}/{self.step_total:04d}) | "
            f"{stage}"
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


def process_video_async(
    session_id: str, video_path: str | Path, settings: dict[str, Any], output_dir: str | Path
) -> None:
    """Process video in background thread"""
    global current_processing
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
        callback = ProgressCallback(session_id, expected_frames, processing_mode)

        # Give client time to join the session room before starting processing
        socketio.sleep(INITIAL_PROCESSING_DELAY)

        # Use the appropriate processor based on processing mode
        if processing_mode == "batch":
            from depth_surge_3d.processing.batch_processor import BatchProcessor

            processor = BatchProcessor(projector.depth_estimator)
        else:
            processor = VideoProcessor(projector.depth_estimator)

        # Calculate resolution settings that VideoProcessor expects
        from depth_surge_3d.utils.resolution import (
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
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Create timestamped output directory
    original_filename = file.filename
    video_name = Path(original_filename).stem
    file_ext = Path(original_filename).suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(app.config["OUTPUT_FOLDER"]) / f"{int(time.time())}_{video_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save video to output directory as "original_video.ext"
    video_path = output_dir / f"original_video{file_ext}"
    file.save(video_path)

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
            result = subprocess.run(
                [
                    "ffmpeg",
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
            "output_dir": str(output_dir),
            "video_info": video_info,
            "has_audio": audio_path is not None and audio_path.exists(),
        }
    )


@app.route("/process", methods=["POST"])
def start_processing() -> tuple[dict[str, Any], int] | tuple[Any, int]:
    """Start video processing"""
    global current_processing

    if current_processing["active"]:
        return jsonify({"error": "Processing already in progress"}), 400

    data = request.json
    output_dir_str = data.get("output_dir")
    settings = data.get("settings", {})

    if not output_dir_str:
        return jsonify({"error": "No output directory provided"}), 400

    output_dir = Path(output_dir_str)
    if not output_dir.exists():
        return jsonify({"error": "Output directory not found"}), 404

    # Find the original video file in output directory
    video_path = None
    for ext in [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"]:
        candidate = output_dir / f"original_video{ext}"
        if candidate.exists():
            video_path = candidate
            break

    if not video_path:
        return (
            jsonify({"error": "Original video file not found in output directory"}),
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
    global current_processing

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
    global current_processing

    if current_processing["active"]:
        return jsonify({"error": "Processing already in progress"}), 400

    data = request.json
    output_dir = data.get("output_dir")

    if not output_dir:
        return jsonify({"error": "No output directory provided"}), 400

    output_path = Path(output_dir)
    if not output_path.exists():
        return jsonify({"error": "Output directory does not exist"}), 404

    # Look for original video in the output directory itself
    original_video = None
    for ext in [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"]:
        candidate = output_path / f"original_video{ext}"
        if candidate.exists():
            original_video = candidate
            break

    if not original_video:
        return (
            jsonify(
                {"error": "Could not find original video file in output directory for resuming"}
            ),
            404,
        )

    # Try to detect settings from existing files/directories
    settings = detect_resume_settings(output_path)

    # Generate session ID
    session_id = str(uuid.uuid4())

    # Start processing in background using socketio's method for proper context handling
    thread = socketio.start_background_task(
        process_video_async, session_id, original_video, settings, output_path
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


def analyze_batch_directory(batch_path):
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

    # Check for pre-extracted audio file or original video in batch directory
    audio_file = batch_path / "original_audio.flac"
    if audio_file.exists():
        analysis["has_audio"] = True
    else:
        # Check for original video file
        for video_ext in [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"]:
            if (batch_path / f"original_video{video_ext}").exists():
                analysis["has_audio"] = True
                break

    # Generate settings summary
    if analysis["vr_format"] != "unknown" and analysis["resolution"] != "unknown":
        analysis["settings_summary"] = f"{analysis['vr_format']}, {analysis['resolution']}"

    return analysis


def create_video_from_batch(batch_path, settings):
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
            # Fallback: look for original video in batch directory
            video_file = None
            for ext in [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"]:
                candidate = batch_path / f"original_video{ext}"
                if candidate.exists():
                    video_file = candidate
                    break
            if video_file:
                cmd.extend(["-i", str(video_file)])
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
    global current_processing
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

    import torch  # Import here to avoid issues during startup

    # Only print startup message if not already printed by run_ui.sh
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
