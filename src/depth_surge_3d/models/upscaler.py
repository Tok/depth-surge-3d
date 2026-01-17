"""
AI upscaling models for output enhancement.

This module provides abstraction for various upscaling models including Real-ESRGAN,
with support for style-preserving enhancement without altering video aesthetics.
"""

from __future__ import annotations
from typing import Literal
import torch
import numpy as np
import cv2

UpscaleModel = Literal["none", "x2", "x4", "x4-conservative"]


class ImageUpscaler:
    """Base class for AI upscaling models."""

    def __init__(self, device: str = "auto"):
        """
        Initialize upscaler.

        Args:
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.device = self._determine_device(device)
        self.model = None

    def _determine_device(self, device: str) -> str:
        """Determine best device for inference."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load_model(self) -> bool:
        """
        Load the upscaling model.

        Returns:
            True if model loaded successfully
        """
        raise NotImplementedError

    def upscale_image(self, image: np.ndarray) -> np.ndarray:
        """
        Upscale a single image.

        Args:
            image: Input image in BGR format (OpenCV convention)

        Returns:
            Upscaled image in BGR format
        """
        raise NotImplementedError

    def unload_model(self) -> None:
        """Free model memory."""
        if self.model is not None:
            del self.model
            self.model = None
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()


class LanczosUpscaler(ImageUpscaler):
    """Fallback upscaler using Lanczos interpolation (fast, dependency-free)."""

    def __init__(self, scale: int = 2, device: str = "auto"):
        """
        Initialize Lanczos upscaler.

        Args:
            scale: Upscale factor (2 or 4)
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        super().__init__(device)
        self.scale = scale

    def load_model(self) -> bool:
        """Lanczos doesn't need model loading."""
        print(f"Using Lanczos {self.scale}x upscaling (fast, no AI)")
        print("Note: For AI upscaling, Real-ESRGAN requires compatible dependencies")
        return True

    def upscale_image(self, image: np.ndarray) -> np.ndarray:
        """
        Upscale using Lanczos interpolation (BGR → BGR).

        Args:
            image: Input image in BGR format

        Returns:
            Upscaled image in BGR format
        """
        h, w = image.shape[:2]
        new_size = (w * self.scale, h * self.scale)
        return cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)


def create_upscaler(model_name: str = "none", device: str = "auto") -> ImageUpscaler | None:
    """
    Factory function to create an upscaler.

    Args:
        model_name: Upscale model ('none', 'x2', 'x4', 'x4-conservative')
        device: Device to use ('auto', 'cuda', 'cpu')

    Returns:
        ImageUpscaler instance or None if model_name is 'none'

    Raises:
        ValueError: If model_name is unknown
    """
    if model_name == "none":
        return None

    # Extract scale factor
    if model_name in ["x2", "x4", "x4-conservative"]:
        scale = int(model_name[1])
        # Currently using Lanczos fallback due to Real-ESRGAN dependency issues
        # TODO: Implement Real-ESRGAN when compatible wrapper is available
        return LanczosUpscaler(scale=scale, device=device)

    raise ValueError(f"Unknown upscale model: {model_name}")
