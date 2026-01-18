"""
Depth Surge 3D - Convert 2D videos to immersive 3D VR format using advanced AI depth estimation.

This package provides tools for creating stereoscopic VR content from monocular videos
using state-of-the-art depth estimation models.
"""

__version__ = "0.9.0"
__author__ = "Depth Surge 3D Team"
__description__ = "Convert 2D videos to immersive 3D VR format using AI depth estimation"


# Lazy imports to avoid loading heavy dependencies at package import time
def _lazy_import_stereo_projector():
    """Lazy import StereoProjector to avoid cv2 dependency during package loading."""
    from .rendering import StereoProjector

    return StereoProjector


# Import constants (safe, no cv2 dependency)
from .core.constants import *

# For backwards compatibility, we can add these to __all__ but load them lazily
__all__ = [
    "StereoProjector",
    "DEFAULT_SETTINGS",
    "VR_RESOLUTIONS",
    "MODEL_CONFIGS",
]


# Make StereoProjector available at package level through lazy loading
def __getattr__(name):
    if name == "StereoProjector":
        return _lazy_import_stereo_projector()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
