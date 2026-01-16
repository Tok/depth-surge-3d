"""
Video depth estimation using Depth Anything V3.

This module provides a wrapper around Depth Anything V3 for improved
memory efficiency and performance compared to Video-Depth-Anything V2.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any

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
        verbose: bool = False,
    ):
        """
        Initialize DA3 depth estimator.

        Args:
            model_name: Model size (small, base, large, large-metric, giant, giant-large)
            device: Device to use (auto, cuda, cpu, mps)
            metric: Use metric depth model (returns depth in meters)
            verbose: Print detailed processing information
        """
        self.model_name = model_name
        self.device = self._determine_device(device)
        self.metric = metric
        self.verbose = verbose
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
            import warnings
            import logging

            # Suppress gsplat dependency warning (only needed for giant models with 3DGS)
            # DA3 uses loguru logger which prints directly, need to suppress before import
            warnings.filterwarnings("ignore", message=".*gsplat.*")
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="depth_anything_3"
            )

            # Suppress all depth_anything_3 logging including loguru output
            logging.getLogger("depth_anything_3").setLevel(logging.CRITICAL)

            # Suppress loguru logger used by DA3 (if available)
            try:
                import sys
                from loguru import logger

                logger.remove()  # Remove default handler
                logger.add(sys.stderr, level="ERROR")  # Re-add with ERROR level only
            except ImportError:
                pass  # loguru not installed or not used

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
            except Exception:
                # Catch all exceptions - xformers may fail to import for various reasons
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
            input_size: Processing resolution for DA3 (controls quality vs speed)
            fp32: Use FP32 instead of FP16 (unused in DA3)

        Returns:
            Depth maps array (shape: [N, H, W], normalized 0-1 range)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Convert BGR to RGB for DA3
            frames_rgb = [frame[..., ::-1].copy() for frame in frames]

            # Determine processing resolution
            # Use input_size parameter to control DA3's internal resolution
            # Higher values = better quality but slower & more VRAM
            # DA3 default is 504, but we can go higher for better quality
            original_height, original_width = frames.shape[1:3]

            # Use the larger dimension as process_res, capped at input_size
            process_res = min(max(original_height, original_width), input_size)

            if self.verbose:
                print(
                    f"  DA3 processing: {len(frames)} frames at {process_res}px resolution"
                )
                print(f"  Original frame size: {original_width}x{original_height}")

            # Run DA3 inference directly on numpy arrays (no file I/O needed!)
            with torch.no_grad():
                prediction = self.model.inference(
                    frames_rgb,
                    process_res=process_res,
                    process_res_method="upper_bound_resize",
                )

            # Extract depth maps
            depth_maps = prediction.depth  # [N, H, W] float32

            # Resize depth maps to match original frame resolution if needed
            import cv2

            resized_depth_maps = []
            for depth_map in depth_maps:
                # Convert to numpy if tensor
                if torch.is_tensor(depth_map):
                    depth_map = depth_map.cpu().numpy()

                # Resize to original dimensions if different
                if (
                    depth_map.shape[0] != original_height
                    or depth_map.shape[1] != original_width
                ):
                    resized = cv2.resize(
                        depth_map,
                        (original_width, original_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    resized_depth_maps.append(resized)
                else:
                    resized_depth_maps.append(depth_map)

            # Normalize depth maps to 0-1 range
            return self._normalize_depths(np.array(resized_depth_maps))

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
