"""

from __future__ import annotations
Video depth estimation model management using Video-Depth-Anything.

This module handles loading and interfacing with the Video-Depth-Anything model,
which provides temporal consistency for video depth estimation.
"""

import gc
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any
import torch
import numpy as np

from ...utils.system.console import success as console_success, error as console_error
from ...utils.imaging.image_processing import depth_to_inverse_depth
from ...utils.system.xformers_guard import ensure_attention_backend
from ...utils.domain.depth_alignment import normalize_depth_sequence
from ...core.constants import (
    DEFAULT_MODEL_PATH,
    VIDEO_DEPTH_ANYTHING_REPO_DIR,
    MODEL_CONFIGS,
    MODEL_DOWNLOAD_URLS,
    MODEL_DOWNLOAD_URLS_METRIC,
    DEPTH_MODEL_INPUT_SIZE,
    DEPTH_MODEL_DEFAULT_FPS,
)


class VideoDepthEstimator:
    """Handles video depth estimation using Video-Depth-Anything models."""

    # Output scale/shift is consistent across all frames of one estimate_depth_batch call
    temporally_consistent = True

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        metric: bool = False,
        temporal_window_overlap: int = 10,
    ):
        self.model_path = model_path
        self.device = self._determine_device(device)
        self.metric = metric
        self.temporal_window_overlap = temporal_window_overlap
        self.model = None
        self.model_config: dict[str, Any] | None = None

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
        Load the video depth estimation model.

        Returns:
            True if model loaded successfully
        """
        try:
            # Ensure dependencies are available
            if not self._ensure_dependencies():
                return False

            # Determine model type from path
            model_type = self._get_model_type(self.model_path)
            if not model_type:
                print(f"Cannot determine model type from path: {self.model_path}")
                return False

            self.model_config = MODEL_CONFIGS[model_type]

            # Import and load model
            repo_path = Path(VIDEO_DEPTH_ANYTHING_REPO_DIR)
            if str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))

            # Use torch SDPA if xformers' kernels are missing or don't match this torch
            if ensure_attention_backend() == "sdpa":
                print("  Attention backend: PyTorch SDPA (xformers kernels unavailable)")

            from video_depth_anything.video_depth import VideoDepthAnything

            assert self.model_config is not None
            self.model = VideoDepthAnything(**self.model_config, metric=self.metric)

            # Load state dict and fix key names if needed
            state_dict = torch.load(self.model_path, map_location="cpu")

            # Remap depth_head.* to head.* for compatibility
            fixed_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("depth_head."):
                    new_key = key.replace("depth_head.", "head.")
                    fixed_state_dict[new_key] = value
                else:
                    fixed_state_dict[key] = value

            self.model.load_state_dict(fixed_state_dict, strict=True)  # type: ignore[attr-defined]
            self.model = self.model.to(self.device).eval()  # type: ignore[attr-defined]

            model_variant = "Metric-" if self.metric else ""
            print(f"Loaded {model_variant}Video-Depth-Anything ({model_type}) on {self.device}")
            return True

        except Exception as e:
            print(f"Error loading video model: {e}")
            print("Try downloading the model manually from:")
            print(f"  {self._download_url() or 'Unknown'}")
            return False

    def _ensure_dependencies(self) -> bool:
        """Ensure model file and repository are available."""
        # Check model file
        if not os.path.exists(self.model_path):
            print(f"Model not found at {self.model_path}")
            if not self._auto_download_model():
                return False

        # Check repository
        repo_path = Path(VIDEO_DEPTH_ANYTHING_REPO_DIR)
        if not repo_path.exists():
            print("Video-Depth-Anything repository not found")
            print("Please ensure the repository is cloned to the vendor directory:")
            print(
                f"  mkdir -p vendor && git clone https://github.com/DepthAnything/Video-Depth-Anything.git {VIDEO_DEPTH_ANYTHING_REPO_DIR}"
            )
            return False

        return True

    def _download_url(self) -> str | None:
        """Hugging Face URL for this estimator's model file (relative or metric weights)."""
        model_type = self._get_model_type(self.model_path)
        is_metric = self.metric or "metric" in Path(self.model_path).name.lower()
        urls = MODEL_DOWNLOAD_URLS_METRIC if is_metric else MODEL_DOWNLOAD_URLS
        return urls.get(model_type) if model_type else None

    def _auto_download_model(self) -> bool:
        """Auto-download the model if missing."""
        print("Attempting to download video model automatically...")

        download_url = self._download_url()
        if not download_url:
            print("Cannot determine model download URL")
            return False

        model_file = Path(self.model_path)
        model_file.parent.mkdir(parents=True, exist_ok=True)
        partial = model_file.with_name(model_file.name + ".part")

        try:
            print(f"Downloading video model to {self.model_path}...")
            # Download to a temporary name so an interrupted download isn't mistaken for a model
            urllib.request.urlretrieve(download_url, partial)
            partial.replace(model_file)
            print(console_success("Video model downloaded successfully"))
            return True
        except Exception as e:
            partial.unlink(missing_ok=True)
            print(console_error(f"Auto-download failed: {e}"))
            print(f"Please download manually from: {download_url}")
            return False

    def _get_model_type(self, model_path: str) -> str | None:
        """Determine model type from file path."""
        path_str = str(model_path).lower()

        if "vits" in path_str:
            return "vits"
        elif "vitb" in path_str:
            return "vitb"
        elif "vitl" in path_str:
            return "vitl"

        # Fallback to large model
        return "vitl"

    def estimate_depth_batch(
        self,
        frames: np.ndarray,
        target_fps: int = DEPTH_MODEL_DEFAULT_FPS,
        input_size: int = DEPTH_MODEL_INPUT_SIZE,
        fp32: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Estimate temporally consistent depth for a sequence of frames.

        Video-Depth-Anything processes any number of frames in overlapping 32-frame
        windows with constant VRAM and aligns scale/shift between windows itself, so the
        whole sequence goes through in one call. Long videos are split (with alignment)
        by DepthMapProcessor to bound RAM.

        Args:
            frames: Input frames array (shape: [N, H, W, 3], BGR format)
            target_fps: Target frame rate for processing
            input_size: Input size for the model (default: DEPTH_MODEL_INPUT_SIZE)
            fp32: Use FP32 instead of FP16 (slower but more accurate)
            normalize: Return 0-1 normalized inverse depth (one range for the whole
                sequence). If False, return raw inverse depth (relative scale/shift).

        Returns:
            Inverse depth maps array (shape: [N, H, W], larger = nearer)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        frames_rgb = frames[..., ::-1].copy()
        depths = None
        try:
            depths = self._infer(frames_rgb, target_fps, input_size, fp32)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise RuntimeError(f"Video depth estimation failed: {e}")
        finally:
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if depths is None:
            # Retry outside the except block: while handling the exception its traceback still
            # references the failed attempt's GPU tensors, so that memory couldn't be freed and
            # the retry ran out of memory too
            gc.collect()
            depths = self._retry_chunk_with_reduced_resolution(
                frames_rgb, target_fps, input_size, fp32
            )

        inverse = self._to_inverse_depth(depths)
        return normalize_depth_sequence(inverse) if normalize else inverse

    def _suppress_model_output(self):
        """Context manager to suppress model output streams."""
        import sys
        import os
        import contextlib

        @contextlib.contextmanager
        def suppress():
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = open(os.devnull, "w")
                sys.stderr = open(os.devnull, "w")
                yield
            finally:
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        return suppress()

    def _infer(
        self, frames_rgb: np.ndarray, target_fps: int, input_size: int, fp32: bool
    ) -> np.ndarray:
        """Run the model on RGB frames with its progress output suppressed."""
        with self._suppress_model_output():
            depths, _ = self.model.infer_video_depth(  # type: ignore[attr-defined]
                frames_rgb,
                target_fps,
                input_size=input_size,
                device=self.device,
                fp32=fp32,
            )
        return depths

    def _retry_chunk_with_reduced_resolution(
        self, frames_rgb: np.ndarray, target_fps: int, input_size: int, fp32: bool
    ) -> np.ndarray:
        """Retry depth processing with reduced input size on OOM error."""
        reduced = max(384, input_size // 2)
        print(f"  Out of GPU memory, retrying with depth resolution {reduced}...")
        torch.cuda.empty_cache()
        return self._infer(frames_rgb, target_fps, reduced, fp32)

    def _to_inverse_depth(self, depths: np.ndarray) -> np.ndarray:
        """
        Convert model output to raw inverse depth (float32, larger = nearer).

        Relative models already output inverse depth (disparity); metric models output
        depth in meters and are inverted.
        """
        depths = np.asarray(depths, dtype=np.float32)
        if self.metric:
            return np.stack([depth_to_inverse_depth(d) for d in depths])
        return depths

    def _normalize_depths(self, depths: np.ndarray) -> np.ndarray:
        """Normalize raw model output to 0-1 inverse depth with one range for all frames."""
        return normalize_depth_sequence(self._to_inverse_depth(depths))

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model_config:
            return {}

        return {
            "encoder": self.model_config["encoder"],
            "features": self.model_config["features"],
            "out_channels": self.model_config["out_channels"],
            "num_frames": self.model_config["num_frames"],
            "device": self.device,
            "metric": self.metric,
            "model_path": self.model_path,
            "loaded": self.model is not None,
            "temporal_consistency": True,  # Key feature of video model
        }

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None

            # Clear GPU cache if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()


def create_video_depth_estimator(
    model_path: str | None = None,
    device: str = "auto",
    metric: bool = False,
    temporal_window_overlap: int = 10,
) -> VideoDepthEstimator:
    """
    Factory function to create a video depth estimator.

    Args:
        model_path: Path to model file (uses default if None)
        device: Device to use for inference
        metric: Use metric depth model (true depth values)
        temporal_window_overlap: Unused (chunk overlap is settings["temporal_window_overlap"]
            in DepthMapProcessor); kept for API compatibility

    Returns:
        Configured VideoDepthEstimator instance
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    return VideoDepthEstimator(model_path, device, metric, temporal_window_overlap)
