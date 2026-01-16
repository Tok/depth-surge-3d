"""
Video depth estimation using Depth Anything V3.

This module provides a wrapper around Depth Anything V3 for improved
memory efficiency and performance compared to Video-Depth-Anything V2.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

from ..core.constants import DA3_MODEL_NAMES, DEFAULT_DA3_MODEL


class VideoDepthEstimatorDA3:
    """
    Handles video depth estimation using Depth Anything V3 models.

    DA3 offers improved memory efficiency and performance compared to V2,
    making it better suited for GPUs with limited VRAM (e.g., 6GB RTX cards).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_DA3_MODEL,
        device: str = "auto",
        metric: bool = False,
    ):
        """
        Initialize DA3 depth estimator.

        Args:
            model_name: Model size (small, base, large, large-metric, giant, giant-large)
            device: Device to use (auto, cuda, cpu, mps)
            metric: Use metric depth model (returns depth in meters)
        """
        self.model_name = model_name
        self.device = self._determine_device(device)
        self.metric = metric
        self.model = None

    def _determine_device(self, device: str) -> str:
        """Determine the best device to use for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def load_model(self) -> bool:
        """
        Load the Depth Anything V3 model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Import DA3 API
            from depth_anything_3.api import DepthAnything3

            # Resolve model name to Hugging Face ID
            if self.model_name in DA3_MODEL_NAMES:
                hf_model_id = DA3_MODEL_NAMES[self.model_name]
            else:
                # Assume it's a direct HF model ID
                hf_model_id = self.model_name

            # Override with metric model if requested
            if self.metric and "metric" not in self.model_name.lower():
                hf_model_id = DA3_MODEL_NAMES["large-metric"]
                print(f"Using metric depth model: {hf_model_id}")

            # Load model from Hugging Face
            print(f"Loading Depth Anything V3 model: {hf_model_id}")
            self.model = DepthAnything3.from_pretrained(hf_model_id)
            self.model = self.model.to(device=self.device)
            self.model.eval()

            model_variant = "Metric-" if self.metric else ""
            print(
                f"Loaded {model_variant}Depth-Anything-V3 ({self.model_name}) on {self.device}"
            )

            # Check if xformers is available (optional optimization)
            try:
                import xformers  # noqa: F401

                print("xformers detected - using optimized attention")
            except ImportError:
                print(
                    "Warning: xformers not available. "
                    "Install with 'pip install xformers' for better performance "
                    "(optional, may fail on some systems)"
                )

            return True

        except ImportError as e:
            print(f"Error: Depth Anything V3 not installed: {e}")
            print(
                "Install with: pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
            )
            print(
                "Optional (for better performance): pip install xformers (may fail on some systems)"
            )
            return False
        except Exception as e:
            print(f"Error loading DA3 model: {e}")
            return False

    def estimate_depth_batch(
        self,
        frames: np.ndarray,
        target_fps: int = 30,
        input_size: int = 518,
        fp32: bool = False,
    ) -> np.ndarray:
        """
        Estimate depth for a batch of video frames.

        DA3 processes frames more efficiently than V2, with better VRAM usage.

        Args:
            frames: Input frames array (shape: [N, H, W, 3], BGR format)
            target_fps: Target frame rate (unused in DA3, kept for API compatibility)
            input_size: Input size for the model (unused in DA3)
            fp32: Use FP32 instead of FP16 (unused in DA3)

        Returns:
            Depth maps array (shape: [N, H, W], normalized 0-1 range)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Convert BGR to RGB
            frames_rgb = frames[..., ::-1].copy()

            # Save frames temporarily for DA3 inference
            # DA3 API expects file paths, so we need to write frames to disk
            temp_dir = Path("temp_frames_da3")
            temp_dir.mkdir(exist_ok=True)

            frame_paths = []
            for idx, frame in enumerate(frames_rgb):
                frame_path = temp_dir / f"frame_{idx:06d}.png"
                import cv2

                cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_paths.append(str(frame_path))

            # Run DA3 inference
            with torch.no_grad():
                prediction = self.model.inference(frame_paths)

            # Extract depth maps
            depth_maps = prediction.depth  # [N, H, W] float32

            # Clean up temporary frames
            import shutil

            shutil.rmtree(temp_dir)

            # Normalize depth maps to 0-1 range
            return self._normalize_depths(depth_maps)

        except Exception as e:
            raise RuntimeError(f"DA3 depth estimation failed: {e}")

    def _normalize_depths(self, depths: np.ndarray) -> np.ndarray:
        """Normalize depth maps to 0-1 range."""
        # Convert to numpy if torch tensor
        if torch.is_tensor(depths):
            depths = depths.cpu().numpy()

        normalized_depths = []
        for depth in depths:
            if depth.max() == depth.min():
                normalized = np.zeros_like(depth)
            else:
                normalized = (depth - depth.min()) / (depth.max() - depth.min())
            normalized_depths.append(np.clip(normalized, 0.0, 1.0))
        return np.array(normalized_depths)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "model_version": "Depth Anything V3",
            "device": self.device,
            "metric": self.metric,
            "loaded": self.model is not None,
            "temporal_consistency": False,  # DA3 processes frames independently
            "memory_efficient": True,  # Key advantage of DA3
        }

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None

            # Clear GPU cache if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()


def create_video_depth_estimator_da3(
    model_name: Optional[str] = None,
    device: str = "auto",
    metric: bool = False,
) -> VideoDepthEstimatorDA3:
    """
    Factory function to create a Depth Anything V3 depth estimator.

    Args:
        model_name: Model size (uses default if None)
        device: Device to use for inference
        metric: Use metric depth model (true depth values)

    Returns:
        Configured VideoDepthEstimatorDA3 instance
    """
    if model_name is None:
        model_name = DEFAULT_DA3_MODEL

    return VideoDepthEstimatorDA3(model_name, device, metric)
