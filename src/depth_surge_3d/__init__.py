"""
Depth Surge 3D - Convert 2D videos to immersive 3D VR format using advanced AI depth estimation.

This package provides tools for creating stereoscopic VR content from monocular videos
using state-of-the-art depth estimation models.
"""

import sys

__version__ = "0.9.1"
__author__ = "Depth Surge 3D Team"
__description__ = "Convert 2D videos to immersive 3D VR format using AI depth estimation"

# Python version check - must be done early before any heavy imports
if sys.version_info >= (3, 13):
    raise RuntimeError(
        f"Depth Surge 3D requires Python 3.9-3.12, but you are using Python {sys.version_info[0]}.{sys.version_info[1]}.\n"
        f"Python 3.13+ is not yet supported due to dependency limitations (specifically 'open3d' in Depth-Anything V3).\n"
        f"Please use Python 3.12 or earlier. See https://github.com/Tok/depth-surge-3d/issues/11 for updates."
    )

if sys.version_info < (3, 9):
    raise RuntimeError(
        f"Depth Surge 3D requires Python 3.9 or newer, but you are using Python {sys.version_info[0]}.{sys.version_info[1]}.\n"
        f"Please upgrade to Python 3.9-3.12."
    )


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
