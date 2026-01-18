"""Inference modules for Depth Surge 3D.

Depth estimation and upscaling inference.
"""

# Depth inference
from .depth import (
    VideoDepthEstimator,
    create_video_depth_estimator,
    create_video_depth_estimator_da3,
)

# Upscaling inference
from .upscaling import create_upscaler

__all__ = [
    # Depth
    "VideoDepthEstimator",
    "create_video_depth_estimator",
    "create_video_depth_estimator_da3",
    # Upscaling
    "create_upscaler",
]
