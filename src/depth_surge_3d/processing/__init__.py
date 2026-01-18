"""Video processing modules for Depth Surge 3D.

Re-exports all processing modules for backward compatibility.
"""

# Orchestration modules
from .orchestration import ProcessingOrchestrator, VideoProcessor

# Frame processing modules
from .frames import (
    DepthMapProcessor,
    StereoPairGenerator,
    DistortionProcessor,
    FrameUpscalerProcessor,
    VRFrameAssembler,
)

# Video processing modules
from .video import VideoEncoder

__all__ = [
    # Orchestration
    "ProcessingOrchestrator",
    "VideoProcessor",
    # Frame processing
    "DepthMapProcessor",
    "StereoPairGenerator",
    "DistortionProcessor",
    "FrameUpscalerProcessor",
    "VRFrameAssembler",
    # Video processing
    "VideoEncoder",
]
