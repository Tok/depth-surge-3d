"""
Video depth estimation model management using Video-Depth-Anything.

This module handles loading and interfacing with the Video-Depth-Anything model,
which provides temporal consistency for video depth estimation.
"""

import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import numpy as np

from ..utils.console import success as console_success, error as console_error
from ..core.constants import (
    DEFAULT_MODEL_PATH,
    VIDEO_DEPTH_ANYTHING_REPO_DIR,
    MODEL_CONFIGS,
    MODEL_DOWNLOAD_URLS,
    DEPTH_MODEL_INPUT_SIZE,
    DEPTH_MODEL_CHUNK_SIZE,
    DEPTH_MODEL_DEFAULT_FPS,
)


class VideoDepthEstimator:
    """Handles video depth estimation using Video-Depth-Anything models."""

    def __init__(self, model_path: str, device: str = "auto", metric: bool = False):
        self.model_path = model_path
        self.device = self._determine_device(device)
        self.metric = metric
        self.model = None
        self.model_config = None

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

            from video_depth_anything.video_depth import VideoDepthAnything

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

            self.model.load_state_dict(fixed_state_dict, strict=True)
            self.model = self.model.to(self.device).eval()

            model_variant = "Metric-" if self.metric else ""
            print(
                f"Loaded {model_variant}Video-Depth-Anything ({model_type}) on {self.device}"
            )
            return True

        except Exception as e:
            print(f"Error loading video model: {e}")
            print("Try downloading the model manually from:")
            print(f"  {MODEL_DOWNLOAD_URLS.get(model_type, 'Unknown')}")
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
            print("Please ensure the repository is cloned:")
            print(
                f"  git clone https://github.com/DepthAnything/Video-Depth-Anything.git {VIDEO_DEPTH_ANYTHING_REPO_DIR}"
            )
            return False

        return True

    def _auto_download_model(self) -> bool:
        """Auto-download the model if missing."""
        print("Attempting to download video model automatically...")

        # Create model directory
        model_dir = Path(self.model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        # Determine download URL
        model_type = self._get_model_type(self.model_path)
        if not model_type or model_type not in MODEL_DOWNLOAD_URLS:
            print("Cannot determine model download URL")
            return False

        download_url = MODEL_DOWNLOAD_URLS[model_type]

        try:
            print(f"Downloading video model to {self.model_path}...")
            urllib.request.urlretrieve(download_url, self.model_path)
            print(console_success("Video model downloaded successfully"))
            return True
        except Exception as e:
            print(console_error(f"Auto-download failed: {e}"))
            print(f"Please download manually from: {download_url}")
            return False

    def _get_model_type(self, model_path: str) -> Optional[str]:
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
    ) -> np.ndarray:
        """
        Estimate depth for a batch of video frames with temporal consistency.

        Automatically chunks large videos to avoid OOM errors.

        Args:
            frames: Input frames array (shape: [N, H, W, 3], BGR format)
            target_fps: Target frame rate for processing
            input_size: Input size for the model (default: DEPTH_MODEL_INPUT_SIZE)
            fp32: Use FP32 instead of FP16 (slower but more accurate)

        Returns:
            Depth maps array (shape: [N, H, W], normalized 0-1 range)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Check available memory and determine if we need to chunk
        num_frames = len(frames)
        frame_h, frame_w = frames[0].shape[:2]

        # Estimate memory usage per frame (rough heuristic)
        # High-res videos (>2K) need chunking on GPUs with <16GB VRAM
        needs_chunking = (
            self.device == "cuda"
            and torch.cuda.is_available()
            and (frame_h * frame_w > 2000 * 2000 or num_frames > 60)
        )

        if needs_chunking and num_frames > DEPTH_MODEL_CHUNK_SIZE:
            # Process in overlapping chunks to maintain temporal consistency
            print(f"Using memory-efficient chunked processing for {num_frames} frames")
            return self._estimate_depth_chunked(frames, target_fps, input_size, fp32)
        else:
            # Process all at once (original behavior)
            return self._estimate_depth_single_batch(
                frames, target_fps, input_size, fp32
            )

    def _estimate_depth_single_batch(
        self, frames: np.ndarray, target_fps: int, input_size: int, fp32: bool
    ) -> np.ndarray:
        """Process all frames in a single batch."""
        try:
            # Convert BGR to RGB
            frames_rgb = frames[..., ::-1].copy()

            # Suppress tqdm output from Video-Depth-Anything
            import sys
            import os

            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = open(os.devnull, "w")
                sys.stderr = open(os.devnull, "w")

                # Call the video depth inference method
                depths, _ = self.model.infer_video_depth(
                    frames_rgb,
                    target_fps,
                    input_size=input_size,
                    device=self.device,
                    fp32=fp32,
                )
            finally:
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            # Normalize depth maps
            return self._normalize_depths(depths)

        except Exception as e:
            raise RuntimeError(f"Video depth estimation failed: {e}")

    def _estimate_depth_chunked(
        self, frames: np.ndarray, target_fps: int, input_size: int, fp32: bool
    ) -> np.ndarray:
        """Process frames in overlapping chunks to save memory."""
        chunk_size = DEPTH_MODEL_CHUNK_SIZE
        overlap = 4  # Overlap frames for smooth transitions

        all_depths = []
        num_frames = len(frames)

        for chunk_start in range(0, num_frames, chunk_size - overlap):
            chunk_end = min(chunk_start + chunk_size, num_frames)
            chunk_frames = frames[chunk_start:chunk_end]

            print(f"  Processing frames {chunk_start + 1}-{chunk_end}/{num_frames}")

            # Convert BGR to RGB
            frames_rgb = chunk_frames[..., ::-1].copy()

            try:
                # Process chunk with output suppression
                depths = self._process_depth_chunk(
                    frames_rgb, target_fps, input_size, fp32
                )

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Retry with reduced resolution
                    depths = self._retry_chunk_with_reduced_resolution(
                        frames_rgb, target_fps, input_size, fp32
                    )
                else:
                    raise

            # Determine which frames to keep (handle overlap)
            keep_depths = self._determine_chunk_overlap(
                chunk_start, chunk_end, num_frames, overlap, depths
            )
            all_depths.extend(keep_depths)

            # Clear CUDA cache between chunks
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Normalize all depth maps
        return self._normalize_depths(np.array(all_depths))

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

    def _determine_chunk_overlap(
        self, chunk_start: int, chunk_end: int, num_frames: int, overlap: int, depths
    ) -> np.ndarray:
        """Determine which frames to keep from chunk based on overlap."""
        if chunk_start == 0:
            return depths  # First chunk: keep all
        elif chunk_end == num_frames:
            return depths[overlap:]  # Last chunk: skip overlap frames
        else:
            return depths[overlap:]  # Middle chunks: skip overlap frames

    def _process_depth_chunk(
        self, frames_rgb: np.ndarray, target_fps: int, input_size: int, fp32: bool
    ) -> np.ndarray:
        """Process a single chunk for depth estimation with output suppression."""
        with self._suppress_model_output():
            depths, _ = self.model.infer_video_depth(
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
        print("  OOM error, retrying with reduced resolution...")
        torch.cuda.empty_cache()

        with self._suppress_model_output():
            depths, _ = self.model.infer_video_depth(
                frames_rgb,
                target_fps,
                input_size=max(384, input_size // 2),
                device=self.device,
                fp32=fp32,
            )
        return depths

    def _normalize_depths(self, depths: np.ndarray) -> np.ndarray:
        """Normalize depth maps to 0-1 range."""
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
    model_path: Optional[str] = None, device: str = "auto", metric: bool = False
) -> VideoDepthEstimator:
    """
    Factory function to create a video depth estimator.

    Args:
        model_path: Path to model file (uses default if None)
        device: Device to use for inference
        metric: Use metric depth model (true depth values)

    Returns:
        Configured VideoDepthEstimator instance
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    return VideoDepthEstimator(model_path, device, metric)
