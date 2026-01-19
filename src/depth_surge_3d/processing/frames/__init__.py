"""Frame processing modules.

Specialized processors for frame-level operations.
"""

from .depth_processor import DepthMapProcessor
from .stereo_generator import StereoPairGenerator
from .distortion_processor import DistortionProcessor
from .frame_upscaler import FrameUpscalerProcessor
from .vr_assembler import VRFrameAssembler

__all__ = [
    "DepthMapProcessor",
    "StereoPairGenerator",
    "DistortionProcessor",
    "FrameUpscalerProcessor",
    "VRFrameAssembler",
]
