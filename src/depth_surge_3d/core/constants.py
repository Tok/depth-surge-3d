"""
Constants and default configuration for Depth Surge 3D.

This module contains all default settings, magic numbers, and configuration
values used throughout the application.
"""

from typing import Dict, Tuple, Any
from pathlib import Path

# Version and project info
PROJECT_NAME = "Depth Surge 3D"
DEFAULT_OUTPUT_DIR = "./output"

# Model configuration
DEFAULT_MODEL_PATH = "models/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth"
DEPTH_ANYTHING_REPO_DIR = "depth_anything_v2_repo"

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# Default processing settings
DEFAULT_SETTINGS = {
    "baseline": 0.065,  # meters - average human IPD
    "focal_length": 1000,
    "vr_format": "side_by_side",
    "vr_resolution": "auto",
    "processing_mode": "serial",
    "fisheye_projection": "stereographic",
    "fisheye_fov": 105,  # degrees
    "crop_factor": 1.0,  # no crop by default
    "fisheye_crop_factor": 1.0,  # no crop by default
    "hole_fill_quality": "fast",
    "super_sample": "auto",
    "target_fps": 60,
    "min_resolution": "1080p",
    "preserve_audio": True,
    "keep_intermediates": True,
    "apply_distortion": True,
}

# VR resolution configurations (per eye)
VR_RESOLUTIONS: Dict[str, Tuple[int, int]] = {
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
    "wide": ["16x9-720p", "16x9-1080p", "16x9-1440p", "16x9-4k", "16x9-5k", "16x9-8k", "wide-2k", "wide-4k", "ultrawide"],
    "standard": ["square-480", "square-720", "square-1k", "square-2k", "square-3k", "square-4k", "square-5k"],
}

# Progress tracking configuration
PROGRESS_UPDATE_INTERVAL = 0.1  # seconds
BATCH_PROCESSING_STEPS = [
    "Frame Extraction",
    "Super Sampling", 
    "Depth Map Generation",
    "Stereo Pair Creation",
    "Fisheye Distortion",
    "Final Processing",
    "Video Creation"
]

# Threading configuration
MAX_WORKERS_DEFAULT = 4
MAX_WORKERS_GPU = 2  # Limited for GPU memory
MAX_WORKERS_IO = 8   # For file operations

# File format settings
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"]
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
OUTPUT_IMAGE_FORMAT = ".png"
OUTPUT_VIDEO_FORMAT = ".mp4"

# Image processing constants
DEFAULT_INTERPOLATION = "cv2.INTER_CUBIC"
DEPTH_MAP_SCALE = 255
MIN_DEPTH_VALUE = 0.0
MAX_DEPTH_VALUE = 1.0

# Fisheye projection constants
FISHEYE_PROJECTIONS = ["equidistant", "stereographic", "equisolid", "orthogonal"]
MIN_FOV = 75  # degrees
MAX_FOV = 180  # degrees

# Hole filling methods
HOLE_FILL_METHODS = ["fast", "advanced"]

# Directory names for intermediate files
INTERMEDIATE_DIRS = {
    "frames": "1_frames",
    "supersampled": "2_supersampled_frames",
    "depth_maps": "3_depth_maps",
    "left_frames": "4_left_frames",
    "right_frames": "5_right_frames",
    "left_distorted": "6_left_distorted",
    "right_distorted": "7_right_distorted",
    "left_final": "8_left_final",
    "right_final": "9_right_final",
    "vr_frames": "10_vr_frames",
}

# Model download URLs
MODEL_DOWNLOAD_URLS = {
    "small": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
    "base": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
    "large": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth",
}

MODEL_PATHS = {
    "small": "models/Depth-Anything-V2-Small/depth_anything_v2_vits.pth",
    "base": "models/Depth-Anything-V2-Base/depth_anything_v2_vitb.pth",
    "large": "models/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth",
}

# Error messages
ERROR_MESSAGES = {
    "model_not_found": "Model file not found. Please run ./download_models.sh to download required models.",
    "repo_not_found": "Depth-Anything-V2 repository not found. Please run ./setup.sh to download dependencies.",
    "invalid_video": "Invalid video file or format not supported.",
    "insufficient_memory": "Insufficient GPU memory. Try using a smaller model or CPU processing.",
    "ffmpeg_not_found": "FFmpeg not found. Please install FFmpeg for video processing.",
}

# Validation ranges
VALIDATION_RANGES = {
    "baseline": (0.01, 0.5),  # meters
    "focal_length": (100, 5000),  # pixels
    "fisheye_fov": (MIN_FOV, MAX_FOV),  # degrees
    "crop_factor": (0.1, 2.0),  # ratio
    "target_fps": (1, 120),  # fps
} 