"""Upscaling inference modules.

AI-based image upscalers (Real-ESRGAN, etc.).
"""

from .upscaler import create_upscaler

__all__ = [
    "create_upscaler",
]
