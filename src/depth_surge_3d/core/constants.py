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

# Model configuration (Video-Depth-Anything only)
DEFAULT_MODEL_PATH = "models/Video-Depth-Anything-Large/video_depth_anything_vitl.pth"
VIDEO_DEPTH_ANYTHING_REPO_DIR = "video_depth_anything_repo"

# Video model configurations
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'num_frames': 32},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'num_frames': 32},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'num_frames': 32},
}

# Default processing settings
DEFAULT_SETTINGS = {
    "baseline": 0.065,  # meters - average human IPD
    "focal_length": 1000,
    "vr_format": "side_by_side",
    "vr_resolution": "auto",
    "fisheye_projection": "stereographic",
    "fisheye_fov": 105,  # degrees
    "crop_factor": 1.5,  # default: 1.5 (matches user preference)
    "fisheye_crop_factor": 0.7,  # default: 0.7 (matches user preference)
    "hole_fill_quality": "fast",
    "super_sample": "auto",
    "target_fps": 60,
    "min_resolution": "1080p",
    "preserve_audio": True,
    "keep_intermediates": True,
    "apply_distortion": True,
    "output_dir": "./output",
    "experimental_frame_interpolation": False,  # Experimental feature with quality warnings
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
PROCESSING_STEPS = [
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
# Numbered to match processing steps: 00=input, 01-07=processing, 99=output
# Optional steps (01, 05, 06) are skipped if not enabled
INTERMEDIATE_DIRS = {
    "frames": "00_original_frames",          # Step 1: Extracted input frames
    "supersampled": "01_supersampled_frames", # Optional: Super sampling (if enabled)
    "depth_maps": "02_depth_maps",           # Step 2: AI-generated depth maps
    "left_frames": "04_left_frames",         # Step 4: Stereo pair - left eye
    "right_frames": "04_right_frames",       # Step 4: Stereo pair - right eye
    "left_distorted": "05_left_distorted",   # Step 5: Fisheye distortion - left (optional)
    "right_distorted": "05_right_distorted", # Step 5: Fisheye distortion - right (optional)
    "left_cropped": "06_left_cropped",       # Optional: Center cropped - left
    "right_cropped": "06_right_cropped",     # Optional: Center cropped - right
    "left_final": "07_left_final",           # Step 6: Final resized - left
    "right_final": "07_right_final",         # Step 6: Final resized - right
    "vr_frames": "99_vr_frames",             # Step 6: Final VR assembled frames (for FFmpeg)
}

# Model download URLs (Video-Depth-Anything)
# Keys match encoder types in MODEL_CONFIGS (vits, vitb, vitl)
# Relative depth models (trained on general scenes)
MODEL_DOWNLOAD_URLS = {
    "vits": "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth",
    "vitb": "https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth",
    "vitl": "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth",
}

# Metric depth models (trained on Virtual KITTI and IRS for accurate depth values)
MODEL_DOWNLOAD_URLS_METRIC = {
    "vits": "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits_metric.pth",
    "vitb": "https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb_metric.pth",
    "vitl": "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl_metric.pth",
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
    "repo_not_found": "Video-Depth-Anything repository not found. Please clone from https://github.com/DepthAnything/Video-Depth-Anything",
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