"""
Constants and default configuration for Depth Surge 3D.

This module contains all default settings, magic numbers, and configuration
values used throughout the application.
"""

from __future__ import annotations

# Version and project info
PROJECT_NAME = "Depth Surge 3D"
DEFAULT_OUTPUT_DIR = "./output"

# Model configuration (Video-Depth-Anything only)
DEFAULT_MODEL_PATH = "models/Video-Depth-Anything-Large/video_depth_anything_vitl.pth"
VIDEO_DEPTH_ANYTHING_REPO_DIR = "vendor/Video-Depth-Anything"

# Depth estimation model parameters
DEPTH_MODEL_INPUT_SIZE = 518  # Input resolution for depth estimation model
DEPTH_MODEL_CHUNK_SIZE = 32  # Number of frames to process in one batch (temporal window)
DEPTH_MODEL_CHUNK_OVERLAP = 10  # Number of frames to overlap between chunks
DEPTH_MODEL_DEFAULT_FPS = 30  # Default FPS for depth estimation when original cannot be determined

# Video model configurations
MODEL_CONFIGS = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "num_frames": DEPTH_MODEL_CHUNK_SIZE,
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
        "num_frames": DEPTH_MODEL_CHUNK_SIZE,
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "num_frames": DEPTH_MODEL_CHUNK_SIZE,
    },
}

# Default processing settings
DEFAULT_SETTINGS = {
    "baseline": 0.065,  # meters - average human IPD
    "focal_length": 1000,
    "vr_format": "side_by_side",
    "vr_resolution": "auto",
    "fisheye_projection": "stereographic",
    "fisheye_fov": 180,  # degrees - full 180° dome view
    "crop_factor": 1.0,  # default: 1.0 (no crop)
    "fisheye_crop_factor": 0.7,  # default: 0.7 (zoom into center ~70%, crops ~20% each edge to hide distortion)
    "hole_fill_quality": "fast",
    "super_sample": "auto",
    "target_fps": 60,
    "min_resolution": "1080p",
    "preserve_audio": True,
    "keep_intermediates": True,
    "apply_distortion": True,
    "output_dir": "./output",
    "experimental_frame_interpolation": False,  # Experimental feature with quality warnings
    "temporal_window_size": DEPTH_MODEL_CHUNK_SIZE,  # V2 only: frames per temporal window
    "temporal_window_overlap": DEPTH_MODEL_CHUNK_OVERLAP,  # V2 only: frame overlap between windows
    "upscale_model": "none",  # AI upscaling: none, x2, x4, x4-conservative
}

# VR resolution configurations (per eye)
VR_RESOLUTIONS: dict[str, tuple[int, int]] = {
    # Square formats (optimized for VR headsets)
    "square-480": (480, 480),
    "square-720": (720, 720),
    "square-1k": (1080, 1080),
    "square-2k": (1536, 1536),
    "square-3k": (1920, 1920),
    "square-4k": (2048, 2048),
    "square-5k": (2560, 2560),
    # 16:9 formats (standard aspect ratio)
    "16x9-480p": (854, 480),
    "16x9-720p": (1280, 720),
    "16x9-1080p": (1920, 1080),
    "16x9-1440p": (2560, 1440),
    "16x9-4k": (3840, 2160),
    "16x9-5k": (5120, 2880),
    "16x9-8k": (7680, 4320),
    # Legacy wide formats
    "ultrawide": (3840, 2160),
    "wide-2k": (2560, 1440),
    "wide-4k": (3840, 2160),
    # Cinema formats (ultra-wide aspect ratios)
    "cinema-2k": (2048, 858),
    "cinema-4k": (4096, 1716),
}

# Resolution categories for auto-detection
RESOLUTION_CATEGORIES = {
    "ultra_wide": ["cinema-2k", "cinema-4k"],
    "wide": [
        "16x9-720p",
        "16x9-1080p",
        "16x9-1440p",
        "16x9-4k",
        "16x9-5k",
        "16x9-8k",
        "wide-2k",
        "wide-4k",
        "ultrawide",
    ],
    "standard": [
        "square-480",
        "square-720",
        "square-1k",
        "square-2k",
        "square-3k",
        "square-4k",
        "square-5k",
    ],
}

# VR Headset Presets (optimized settings for popular headsets)
VR_HEADSET_PRESETS: dict[str, dict[str, int | float | str]] = {
    "quest-3": {
        "name": "Meta Quest 3",
        "per_eye_width": 2064,
        "per_eye_height": 2208,
        "fisheye_fov": 110,  # degrees
        "description": "Optimized for Meta Quest 3 (2064x2208 per eye, 110° FOV)",
    },
    "quest-2": {
        "name": "Meta Quest 2",
        "per_eye_width": 1832,
        "per_eye_height": 1920,
        "fisheye_fov": 100,  # degrees
        "description": "Optimized for Meta Quest 2 (1832x1920 per eye, 100° FOV)",
    },
    "psvr2": {
        "name": "PlayStation VR2",
        "per_eye_width": 2000,
        "per_eye_height": 2040,
        "fisheye_fov": 110,  # degrees
        "description": "Optimized for PSVR2 (2000x2040 per eye, 110° FOV)",
    },
    "valve-index": {
        "name": "Valve Index",
        "per_eye_width": 1440,
        "per_eye_height": 1600,
        "fisheye_fov": 130,  # degrees
        "description": "Optimized for Valve Index (1440x1600 per eye, 130° FOV)",
    },
    "vive-pro-2": {
        "name": "HTC Vive Pro 2",
        "per_eye_width": 2448,
        "per_eye_height": 2448,
        "fisheye_fov": 120,  # degrees
        "description": "Optimized for HTC Vive Pro 2 (2448x2448 per eye, 120° FOV)",
    },
    "pico-4": {
        "name": "Pico 4",
        "per_eye_width": 2160,
        "per_eye_height": 2160,
        "fisheye_fov": 105,  # degrees
        "description": "Optimized for Pico 4 (2160x2160 per eye, 105° FOV)",
    },
}

# Progress tracking configuration
PROGRESS_UPDATE_INTERVAL = 0.1  # seconds - throttle for progress updates (10 updates/sec)
PROGRESS_DECIMAL_PLACES = 1  # decimal places for progress percentage
PROGRESS_STEP_WEIGHTS = [
    0.02,  # Step 1: Frame Extraction (fast)
    0.35,  # Step 2: Depth Map Generation (slow - AI depth estimation)
    0.20,  # Step 3: Stereo Pair Creation (moderate - includes warping, hole-filling)
    0.08,  # Step 4: Fisheye Distortion (fast, optional)
    0.02,  # Step 5: Crop Frames (fast)
    0.18,  # Step 6: AI Upscaling (slow, optional)
    0.08,  # Step 7: VR Assembly (moderate - side-by-side combining)
    0.07,  # Step 8: Video Creation (moderate - FFmpeg encoding)
]  # Weighted progress distribution (sums to 1.00)
PROCESSING_STEPS = [
    "Frame Extraction",
    "Super Sampling",
    "Depth Map Generation",
    "Stereo Pair Creation",
    "Fisheye Distortion",
    "Final Processing",
    "Video Creation",
]

# Threading configuration
MAX_WORKERS_DEFAULT = 4
MAX_WORKERS_GPU = 2  # Limited for GPU memory
MAX_WORKERS_IO = 8  # For file operations

# SocketIO/Network configuration
SOCKETIO_PING_TIMEOUT = 120  # seconds - timeout for long-running tasks
SOCKETIO_PING_INTERVAL = 25  # seconds - ping interval
SOCKETIO_SLEEP_YIELD = 0  # seconds - immediate yield for message sending
INITIAL_PROCESSING_DELAY = 0.5  # seconds - delay before starting processing

# Memory/Performance constants
BYTES_TO_GB_DIVISOR = 1024**3  # Conversion factor for bytes to gigabytes
FPS_ROUND_DIGITS = 2  # decimal places for FPS values
DURATION_ROUND_DIGITS = 2  # decimal places for duration values
ASPECT_RATIO_ROUND_DIGITS = 3  # decimal places for aspect ratio values

# Aspect ratio detection thresholds
ASPECT_RATIO_SBS_THRESHOLD = 1.5  # width > height * threshold = side-by-side
ASPECT_RATIO_OU_THRESHOLD = 1.5  # height > width * threshold = over-under

# Session/Display formatting
SESSION_ID_DISPLAY_LENGTH = 8  # characters to show from session ID in logs

# File format settings
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"]
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
OUTPUT_IMAGE_FORMAT = ".png"
OUTPUT_VIDEO_FORMAT = ".mp4"

# FFmpeg configuration
FFMPEG_OVERWRITE_FLAG = "-y"  # Overwrite output files without asking
FFMPEG_CRF_HIGH_QUALITY = "18"  # CRF value for high quality encoding
FFMPEG_CRF_MEDIUM_QUALITY = "23"  # CRF value for medium quality encoding
FFMPEG_CRF_FAST_QUALITY = "28"  # CRF value for fast/lower quality encoding
FFMPEG_DEFAULT_PRESET = "medium"  # Default encoding preset
FFMPEG_PIX_FORMAT = "yuv420p"  # Pixel format for compatibility
VIDEO_CREATION_TIMEOUT = 300  # seconds - 5 minute timeout for video creation
DEFAULT_FALLBACK_FPS = 30  # FPS to use when original FPS cannot be determined

# Server configuration (for Flask web UI)
DEFAULT_SERVER_PORT = 5000  # Default port for web interface
DEFAULT_SERVER_HOST = "0.0.0.0"  # Default host binding (all interfaces)
SIGNAL_SHUTDOWN_TIMEOUT = 5  # seconds - timeout for graceful shutdown

# Image processing constants
DEFAULT_INTERPOLATION = "cv2.INTER_CUBIC"
DEPTH_MAP_SCALE = 255  # Scale factor for converting float depth [0-1] to uint8 [0-255]
DEPTH_MAP_SCALE_FLOAT = 255.0  # Float version for division operations
MIN_DEPTH_VALUE = 0.0
MAX_DEPTH_VALUE = 1.0

# Progress reporting
PROGRESS_UPDATE_INTERVAL = 10  # Update progress every N frames processed

# Fisheye projection constants
FISHEYE_PROJECTIONS = ["equidistant", "stereographic", "equisolid", "orthogonal"]
MIN_FOV = 75  # degrees
MAX_FOV = 180  # degrees

# Hole filling methods
HOLE_FILL_METHODS = ["fast", "advanced", "high"]

# AI upscaling models
UPSCALE_MODELS = ["none", "x2", "x4", "x4-conservative"]

# Directory names for intermediate files
# Numbered to match processing steps (6-8 steps total based on optional features):
# Step 1: Extract Frames (00_original_frames)
# Step 2: Generate Depth Maps (02_depth_maps)
# Step 3: Create Stereo Pairs (04_left_frames, 04_right_frames)
# Step 4: Apply Distortion - OPTIONAL (05_left_distorted, 05_right_distorted)
# Step 5: Crop Frames (06_left_cropped, 06_right_cropped)
# Step 6: Upscale Frames - OPTIONAL (07_left_upscaled, 07_right_upscaled)
# Step 7: Assemble VR Frames (99_vr_frames)
# Step 8: Create Final Video
INTERMEDIATE_DIRS = {
    "frames": "00_original_frames",  # Step 1: Extracted input frames
    "supersampled": "01_supersampled_frames",  # Legacy: Super sampling (deprecated)
    "depth_maps": "02_depth_maps",  # Step 2: AI-generated depth maps
    "left_frames": "04_left_frames",  # Step 3: Stereo pair - left eye
    "right_frames": "04_right_frames",  # Step 3: Stereo pair - right eye
    "left_distorted": "05_left_distorted",  # Step 4: Fisheye distortion - left (optional)
    "right_distorted": "05_right_distorted",  # Step 4: Fisheye distortion - right (optional)
    "left_cropped": "06_left_cropped",  # Step 5: Center cropped - left
    "right_cropped": "06_right_cropped",  # Step 5: Center cropped - right
    "left_upscaled": "07_left_upscaled",  # Step 6: AI upscaled - left (optional)
    "right_upscaled": "07_right_upscaled",  # Step 6: AI upscaled - right (optional)
    "left_final": "08_left_final",  # Legacy: kept for compatibility
    "right_final": "08_right_final",  # Legacy: kept for compatibility
    "vr_frames": "99_vr_frames",  # Step 7: Final VR assembled frames
}

# Depth Anything V3 model names (Hugging Face IDs)
DA3_MODEL_NAMES = {
    "small": "depth-anything/DA3-SMALL",
    "base": "depth-anything/DA3-BASE",
    "large": "depth-anything/DA3MONO-LARGE",
    "large-metric": "depth-anything/DA3METRIC-LARGE",
    "giant": "depth-anything/DA3-GIANT-1.1",
    "giant-large": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
}

# Default DA3 model
DEFAULT_DA3_MODEL = "large"  # DA3MONO-LARGE for monocular depth

# Model download URLs (Video-Depth-Anything V2)
# Keys match encoder types in MODEL_CONFIGS (vits, vitb, vitl)
# Relative depth models (trained on general scenes)
MODEL_DOWNLOAD_URLS = {
    "vits": (
        "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/"
        "resolve/main/video_depth_anything_vits.pth"
    ),
    "vitb": (
        "https://huggingface.co/depth-anything/Video-Depth-Anything-Base/"
        "resolve/main/video_depth_anything_vitb.pth"
    ),
    "vitl": (
        "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/"
        "resolve/main/video_depth_anything_vitl.pth"
    ),
}

# Metric depth models (trained on Virtual KITTI and IRS for accurate depth values)
MODEL_DOWNLOAD_URLS_METRIC = {
    "vits": (
        "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/"
        "resolve/main/video_depth_anything_vits_metric.pth"
    ),
    "vitb": (
        "https://huggingface.co/depth-anything/Video-Depth-Anything-Base/"
        "resolve/main/video_depth_anything_vitb_metric.pth"
    ),
    "vitl": (
        "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/"
        "resolve/main/video_depth_anything_vitl_metric.pth"
    ),
}

# Model paths (relative depth)
MODEL_PATHS = {
    "vits": "models/Video-Depth-Anything-Small/video_depth_anything_vits.pth",
    "vitb": "models/Video-Depth-Anything-Base/video_depth_anything_vitb.pth",
    "vitl": "models/Video-Depth-Anything-Large/video_depth_anything_vitl.pth",
}

# Model paths (metric depth)
MODEL_PATHS_METRIC = {
    "vits": "models/Video-Depth-Anything-Small/video_depth_anything_vits_metric.pth",
    "vitb": "models/Video-Depth-Anything-Base/video_depth_anything_vitb_metric.pth",
    "vitl": "models/Video-Depth-Anything-Large/video_depth_anything_vitl_metric.pth",
}

# Error messages
ERROR_MESSAGES = {
    "model_not_found": "Model file not found. Please download Video-Depth-Anything model.",
    "repo_not_found": (
        "Video-Depth-Anything repository not found. "
        "Please clone from https://github.com/DepthAnything/Video-Depth-Anything"
    ),
    "invalid_video": "Invalid video file or format not supported.",
    "insufficient_memory": "Insufficient GPU memory. Try using a smaller model or CPU processing.",
    "ffmpeg_not_found": "FFmpeg not found. Please install FFmpeg for video processing.",
}

# Validation ranges
VALIDATION_RANGES = {
    "baseline": (0.01, 0.5),  # meters
    "focal_length": (100, 5000),  # pixels
    "fisheye_fov": (MIN_FOV, MAX_FOV),  # degrees (75-180)
    "crop_factor": (0.5, 1.0),  # ratio for non-fisheye crop
    "fisheye_crop_factor": (
        0.5,
        2.0,
    ),  # ratio for fisheye crop (1.0=inscribed circle, <1.0=zoom in, >1.0=show curved edges)
    "target_fps": (1, 120),  # fps
}

# Depth processing chunk sizes and resolution thresholds
# Resolution thresholds in pixels (for depth map processing)
RESOLUTION_4K = 2160
RESOLUTION_1440P = 1440
RESOLUTION_1080P = 1080
RESOLUTION_720P = 720
RESOLUTION_SD = 640

# Megapixel thresholds for auto-resolution detection
MEGAPIXELS_4K = 8.0  # 4K is ~8.3MP
MEGAPIXELS_1080P = 2.0  # 1080p is ~2.1MP
MEGAPIXELS_720P = 1.0  # 720p is ~0.9MP

# Chunk sizes for different resolutions (frames per batch for DA3)
CHUNK_SIZE_4K = 4  # 2160p and above
CHUNK_SIZE_1440P = 6  # 1440p
CHUNK_SIZE_1080P = 8  # Auto mode for 1080p source
CHUNK_SIZE_1080P_MANUAL = 12  # Manual 1080p depth setting
CHUNK_SIZE_720P = 16  # 720p and manual settings
CHUNK_SIZE_SD = 24  # Auto mode for SD sources
CHUNK_SIZE_SMALL = 32  # Very small resolutions

# Preview configuration (real-time frame preview via websocket)
PREVIEW_UPDATE_INTERVAL = 2.0  # seconds between preview updates
PREVIEW_FRAME_SAMPLE_RATE = 30  # Send every Nth frame (if time interval allows)
PREVIEW_DOWNSCALE_WIDTH = 480  # Downscale width for transmission (pixels)
PREVIEW_JPEG_QUALITY = 85  # JPEG quality (if using JPEG instead of PNG)
