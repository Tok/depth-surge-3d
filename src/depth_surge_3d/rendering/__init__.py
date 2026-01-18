"""Rendering modules for Depth Surge 3D.

High-level stereo rendering and projection.
"""

from .stereo_projector import StereoProjector, create_stereo_projector

__all__ = [
    "StereoProjector",
    "create_stereo_projector",
]
