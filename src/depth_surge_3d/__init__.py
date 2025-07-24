"""
Depth Surge 3D - Convert 2D videos to immersive 3D VR format using advanced AI depth estimation.

This package provides tools for creating stereoscopic VR content from monocular videos
using state-of-the-art depth estimation models.
"""

__version__ = "1.0.0"
__author__ = "Depth Surge 3D Team"
__description__ = "Convert 2D videos to immersive 3D VR format using AI depth estimation"

from .core.stereo_projector import StereoProjector
from .core.constants import *

__all__ = [
    "StereoProjector",
    "DEFAULT_SETTINGS",
    "VR_RESOLUTIONS",
    "MODEL_CONFIGS",
] 