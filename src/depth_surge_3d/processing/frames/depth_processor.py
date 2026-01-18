"""
Depth map processing module.

Handles depth map generation with caching, VRAM management, and chunking strategies.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Any

from ...core.constants import (
    DEPTH_MAP_SCALE,
    DEPTH_MAP_SCALE_FLOAT,
    DEFAULT_FALLBACK_FPS,
    RESOLUTION_4K,
    RESOLUTION_1440P,
    RESOLUTION_1080P,
    RESOLUTION_720P,
    RESOLUTION_SD,
    MEGAPIXELS_4K,
    MEGAPIXELS_1080P,
    MEGAPIXELS_720P,
    CHUNK_SIZE_4K,
    CHUNK_SIZE_1440P,
    CHUNK_SIZE_1080P_MANUAL,
    CHUNK_SIZE_720P,
    CHUNK_SIZE_SMALL,
    PREVIEW_FRAME_SAMPLE_RATE,
)
from ...utils import (
    get_cached_depth_maps,
    save_depth_maps_to_cache,
    get_cache_size,
)
from ...utils import calculate_optimal_chunk_size, get_vram_info
from ...utils import resize_image


class DepthMapProcessor:
    """
    Depth map generation with caching and memory management.

    Responsibilities:
    - Depth map generation with model inference
    - VRAM-based chunk sizing
    - Global and local depth cache management
    - GPU memory management
    - Batch and chunked processing strategies
    """

    def __init__(self, depth_estimator, verbose: bool = False):
        """
        Initialize depth map processor.

        Args:
            depth_estimator: Depth estimation model instance
            verbose: Enable verbose output
        """
        self.depth_estimator = depth_estimator
        self.verbose = verbose

    def generate_depth_maps(
        self,
        frame_files: list[Path],
        settings: dict[str, Any],
        directories: dict[str, Path],
        progress_tracker,
    ) -> np.ndarray | None:
        """
        Main entry point - generates depth maps with caching.

        Tries cache first, then generates if needed.

        Args:
            frame_files: List of frame file paths
            settings: Processing settings with depth parameters
            directories: Dictionary of processing directories
            progress_tracker: Optional progress tracker

        Returns:
            Numpy array of depth maps, or None if failed

        Side effects:
            - GPU memory operations
            - Filesystem I/O (cache reads/writes)
            - Depth map image writes
        """
        # Check if depth maps already exist (only if keep_intermediates is enabled)
        if settings.get("keep_intermediates") and "depth_maps" in directories:
            existing = self._try_load_existing_depth_maps(
                frame_files, directories, progress_tracker
            )
            if existing is not None:
                return existing

        # Check global depth cache (works across different output batches)
        video_path = settings.get("video_path")
        if video_path:
            cached = self._try_load_cached_depth_maps(
                video_path, settings, len(frame_files), progress_tracker
            )
            if cached is not None:
                return cached

        print("Step 2/7: Generating depth maps (temporal consistency enabled)...")
        print("  Using memory-efficient chunked processing...")
        progress_tracker.update_progress(
            "Generating depth maps",
            phase="depth_estimation",
            frame_num=0,
            step_name="Depth Map Generation",
            step_progress=0,
            step_total=len(frame_files),
        )

        depth_maps = self._generate_depth_maps_chunked(
            frame_files, settings, directories, progress_tracker
        )
        if depth_maps is None:
            return None

        # Save to global cache for future runs
        if video_path and depth_maps is not None:
            self._save_to_depth_cache(video_path, settings, depth_maps)

        return depth_maps

    def _determine_chunk_params(
        self, frame_w: int, frame_h: int, depth_resolution: str = "auto"
    ) -> tuple[int, int]:
        """
        Determine chunk size and input size based on frame resolution, VRAM, and model.

        Uses smart VRAM-based sizing to maximize throughput without OOM errors.

        Args:
            frame_w: Frame width in pixels
            frame_h: Frame height in pixels
            depth_resolution: Either "auto" or specific resolution like "1080", "720", etc.

        Returns:
            Tuple of (chunk_size, input_size)
        """
        megapixels = (frame_h * frame_w) / 1_000_000
        print(f"  Frame resolution: {frame_w}x{frame_h} ({megapixels:.1f}MP)")

        # Get VRAM info for smart sizing
        vram_info = get_vram_info()
        if vram_info["total"] > 0:
            print(
                f"  GPU VRAM: {vram_info['available']:.1f}GB available / {vram_info['total']:.1f}GB total"
            )

        # Determine input size (depth resolution)
        if depth_resolution != "auto":
            try:
                input_size = int(depth_resolution)
                print(f"  Using manual depth resolution: {input_size}px")
            except (ValueError, TypeError):
                print(f"  Warning: Invalid depth_resolution '{depth_resolution}', using auto")
                input_size = self._auto_determine_input_size(frame_w, frame_h, megapixels)
        else:
            input_size = self._auto_determine_input_size(frame_w, frame_h, megapixels)

        # Get model information
        model_version = "v3" if hasattr(self.depth_estimator, "model_type") else "v2"
        model_size = (
            self.depth_estimator.get_model_size()
            if hasattr(self.depth_estimator, "get_model_size")
            else "base"
        )

        # Calculate optimal chunk size based on VRAM
        if vram_info["total"] > 0:
            # Use smart VRAM-based sizing
            chunk_size = calculate_optimal_chunk_size(
                frame_w, frame_h, input_size, model_version, model_size
            )
            print(
                f"  Smart VRAM sizing: {chunk_size} frames/chunk (model: {model_version}/{model_size})"
            )
        else:
            # Fallback to fixed sizing (CPU or no CUDA)
            chunk_size = self._get_chunk_size_for_resolution(input_size)
            print(f"  CPU mode: {chunk_size} frames/chunk")

        return chunk_size, input_size

    def _auto_determine_input_size(self, frame_w: int, frame_h: int, megapixels: float) -> int:
        """
        Determine input size automatically based on frame resolution.

        Args:
            frame_w: Frame width
            frame_h: Frame height
            megapixels: Frame megapixels

        Returns:
            Optimal input size for depth estimation
        """
        # Auto mode: Match depth resolution to actual frame size
        # Never exceed source frame resolution - upscaling depth is pointless
        if megapixels > MEGAPIXELS_4K:  # >8MP (4K is ~8.3MP)
            input_size = min(max(frame_w, frame_h), RESOLUTION_4K)
        elif megapixels > MEGAPIXELS_1080P:  # >2MP (1080p is 2.1MP)
            input_size = min(max(frame_w, frame_h), RESOLUTION_1080P)
        elif megapixels > MEGAPIXELS_720P:  # >1MP (720p is 0.9MP)
            input_size = min(max(frame_w, frame_h), RESOLUTION_720P)
        else:
            input_size = min(max(frame_w, frame_h), RESOLUTION_SD)

        print(f"  Auto depth resolution: {input_size}px")
        return input_size

    def _get_chunk_size_for_resolution(self, input_size: int) -> int:
        """
        Get appropriate chunk size based on depth map resolution.

        Args:
            input_size: Depth map resolution in pixels

        Returns:
            Chunk size for processing
        """
        if input_size >= RESOLUTION_4K:
            return CHUNK_SIZE_4K
        elif input_size >= RESOLUTION_1440P:
            return CHUNK_SIZE_1440P
        elif input_size >= RESOLUTION_1080P:
            return CHUNK_SIZE_1080P_MANUAL
        elif input_size >= RESOLUTION_720P:
            return CHUNK_SIZE_720P
        else:
            return CHUNK_SIZE_SMALL

    def _clear_gpu_memory(self) -> None:
        """
        Clear GPU memory and cache.

        Side effects:
            - Clears CUDA cache
            - Frees GPU memory
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            mem_free = torch.cuda.mem_get_info()[0] / (1024**3)  # Convert to GB
            print(f"  GPU memory freed: {mem_free:.2f} GB available")

    def _load_chunk_frames(self, chunk_files: list[Path], settings: dict[str, Any]) -> list | None:
        """
        Load chunk of frames into memory.

        Args:
            chunk_files: List of frame file paths for this chunk
            settings: Processing settings

        Returns:
            List of loaded frame images

        Side effects:
            - Reads images from disk
        """
        chunk_frames = []
        for frame_file in chunk_files:
            frame = cv2.imread(str(frame_file))
            if frame is None:
                print(f"Warning: Could not load {frame_file}")
                continue

            # Apply super sampling if needed
            if settings["super_sample"] != "none":
                target_width = max(frame.shape[1], settings["per_eye_width"] * 2)
                target_height = max(frame.shape[0], settings["per_eye_height"] * 2)
                frame = resize_image(frame, target_width, target_height)

            chunk_frames.append(frame)

        return chunk_frames if chunk_frames else None

    def _process_chunk_depth(
        self,
        chunk_frames: list,
        chunk_files: list[Path],
        settings: dict[str, Any],
        directories: dict[str, Path],
        input_size: int,
        progress_tracker=None,
    ) -> np.ndarray | None:
        """
        Process chunk with depth estimation.

        Args:
            chunk_frames: List of frame images
            chunk_files: List of frame file paths
            settings: Processing settings
            directories: Dictionary of processing directories
            input_size: Depth map resolution
            progress_tracker: Optional progress tracker

        Returns:
            Numpy array of depth maps

        Side effects:
            - GPU inference
            - Progress updates
        """
        # Normalize target_fps
        target_fps = settings.get("target_fps", DEFAULT_FALLBACK_FPS)
        if target_fps is None or str(target_fps) == "None" or target_fps == "original":
            target_fps = 30

        # Estimate depth
        chunk_frames_array = np.array(chunk_frames)
        chunk_depth_maps = self.depth_estimator.estimate_depth_batch(
            chunk_frames_array, target_fps=target_fps, input_size=input_size, fp32=False
        )

        # Save depth maps immediately to free memory
        if settings["keep_intermediates"] and "depth_maps" in directories:
            self._save_depth_maps(
                chunk_depth_maps, chunk_files, directories["depth_maps"], progress_tracker
            )

        return chunk_depth_maps

    def _generate_depth_maps_chunked(
        self,
        frame_files: list[Path],
        settings: dict[str, Any],
        directories: dict[str, Path],
        progress_tracker,
    ) -> np.ndarray | None:
        """
        Memory-efficient chunked depth generation.

        Processes frames in small batches to avoid CUDA OOM errors.

        Args:
            frame_files: List of frame file paths
            settings: Processing settings
            directories: Dictionary of processing directories
            progress_tracker: Optional progress tracker

        Returns:
            Numpy array of depth maps, or None if failed

        Side effects:
            - GPU memory operations
            - Filesystem I/O
        """
        # Determine chunk parameters based on resolution
        sample_frame = cv2.imread(str(frame_files[0]))
        if sample_frame is None:
            return None

        frame_h, frame_w = sample_frame.shape[:2]
        depth_resolution = settings.get("depth_resolution", "auto")
        chunk_size, input_size = self._determine_chunk_params(frame_w, frame_h, depth_resolution)

        print(f"  Processing in chunks of {chunk_size} frames (input_size={input_size})...")

        # Clear GPU cache before processing
        self._clear_gpu_memory()

        # Process all chunks
        all_depth_maps = []
        num_frames = len(frame_files)
        total_chunks = (num_frames + chunk_size - 1) // chunk_size

        for chunk_start in range(0, num_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_frames)
            chunk_files = frame_files[chunk_start:chunk_end]
            chunk_num = chunk_start // chunk_size + 1

            # Load chunk frames
            chunk_frames = self._load_chunk_frames(chunk_files, settings)
            if not chunk_frames:
                print("Error: No frames loaded in chunk")
                return None

            # Process chunk for depth
            try:
                chunk_depth_maps = self._process_chunk_depth(
                    chunk_frames, chunk_files, settings, directories, input_size, progress_tracker
                )
                all_depth_maps.extend(chunk_depth_maps)

                # Clear references and GPU cache
                del chunk_frames
                del chunk_depth_maps
                self._clear_gpu_memory()

                # Update progress
                progress_tracker.update_progress(
                    f"Chunk {chunk_num}/{total_chunks}: Depth maps {chunk_end}/{num_frames}",
                    phase="depth_estimation",
                    frame_num=chunk_end,
                    step_name="Depth Map Generation",
                    step_progress=chunk_end,
                    step_total=num_frames,
                )

            except Exception as e:
                print(f"Error processing chunk {chunk_start}-{chunk_end}: {e}")
                return None

        return np.array(all_depth_maps)

    def _generate_depth_maps_batch(
        self, frames: np.ndarray, settings: dict[str, Any], progress_tracker
    ) -> np.ndarray | None:
        """
        Full batch depth generation (no chunking).

        Generate depth maps for all frames with temporal consistency.

        Args:
            frames: Numpy array of frame images
            settings: Processing settings
            progress_tracker: Optional progress tracker

        Returns:
            Numpy array of depth maps, or None if failed

        Side effects:
            - GPU memory operations
            - Filesystem I/O
        """
        try:
            # Use Video-Depth-Anything for temporal consistency
            target_fps = settings.get("target_fps", DEFAULT_FALLBACK_FPS)
            if target_fps is None or str(target_fps) == "None" or target_fps == "original":
                target_fps = 30

            # Use depth resolution from settings (default: auto/1080px)
            depth_resolution = settings.get("depth_resolution", "auto")
            if depth_resolution == "auto":
                input_size = 1080  # Match typical 1080p video resolution
            else:
                try:
                    input_size = int(depth_resolution)
                except (ValueError, TypeError):
                    input_size = 1080

            depth_maps = self.depth_estimator.estimate_depth_batch(
                frames, target_fps=target_fps, input_size=input_size, fp32=False
            )

            return depth_maps

        except Exception as e:
            print(f"Error generating depth maps: {e}")
            return None

    def _save_depth_maps(
        self,
        depth_maps: np.ndarray,
        frame_files: list[Path],
        depth_dir: Path,
        progress_tracker=None,
    ) -> None:
        """
        Save depth maps to disk.

        Args:
            depth_maps: Numpy array of depth maps
            frame_files: List of frame files (for naming)
            depth_dir: Output directory
            progress_tracker: Optional progress tracker

        Side effects:
            - Writes depth map images to disk
        """
        for i, (depth_map, frame_file) in enumerate(zip(depth_maps, frame_files)):
            depth_vis = (depth_map * DEPTH_MAP_SCALE).astype("uint8")
            frame_name = frame_file.stem
            depth_path = depth_dir / f"{frame_name}.png"
            cv2.imwrite(str(depth_path), depth_vis)

            # Send preview frame
            if progress_tracker and hasattr(progress_tracker, "send_preview_frame"):
                if i % PREVIEW_FRAME_SAMPLE_RATE == 0 or i == len(depth_maps) - 1:
                    progress_tracker.send_preview_frame(depth_path, "depth_map", i + 1)

    def _try_load_existing_depth_maps(
        self, frame_files: list[Path], directories: dict[str, Path], progress_tracker
    ) -> np.ndarray | None:
        """
        Try to load existing depth maps from output directory.

        Args:
            frame_files: List of frame file paths
            directories: Dictionary of processing directories
            progress_tracker: Optional progress tracker

        Returns:
            Numpy array of depth maps, or None if not found

        Side effects:
            - Filesystem I/O
        """
        depth_maps_dir = directories.get("depth_maps")
        if not depth_maps_dir or not depth_maps_dir.exists():
            return None

        existing_depth_maps = sorted(list(depth_maps_dir.glob("*.png")))
        if not existing_depth_maps or len(existing_depth_maps) < len(frame_files):
            return None

        print("Step 2/7: Skipping depth map generation (depth maps already exist)")
        print(f"  Found {len(existing_depth_maps):04d} existing depth maps")
        print(f"  Location: {depth_maps_dir}\n")

        # Load existing depth maps
        depth_maps = []
        for depth_file in existing_depth_maps[: len(frame_files)]:
            depth_img = cv2.imread(str(depth_file), cv2.IMREAD_GRAYSCALE)
            if depth_img is not None:
                depth_maps.append(depth_img.astype(float) / DEPTH_MAP_SCALE_FLOAT)

        if len(depth_maps) == len(frame_files):
            if progress_tracker:
                progress_tracker.update_progress(
                    "Skipped depth map generation (already exists)",
                    phase="depth_estimation",
                    frame_num=len(depth_maps),
                    step_name="Depth Map Generation",
                    step_progress=len(depth_maps),
                    step_total=len(depth_maps),
                )
            return np.array(depth_maps)
        return None

    def _try_load_cached_depth_maps(
        self, video_path: str, settings: dict[str, Any], num_frames: int, progress_tracker
    ) -> np.ndarray | None:
        """
        Try to load from global depth cache.

        Args:
            video_path: Path to video file (for cache key)
            settings: Processing settings for cache key
            num_frames: Expected number of frames
            progress_tracker: Optional progress tracker

        Returns:
            Numpy array of depth maps, or None if cache miss

        Side effects:
            - Filesystem I/O (cache reads)
        """
        cached_depths = get_cached_depth_maps(video_path, settings, num_frames)
        if cached_depths is None:
            return None

        print("Step 2/7: Loading depth maps from global cache")
        print(f"  Loaded {len(cached_depths):04d} cached depth maps")
        cache_entries, cache_size_bytes = get_cache_size()
        cache_size_mb = cache_size_bytes / (1024 * 1024)
        print(f"  Cache: {cache_entries} entries, {cache_size_mb:.1f} MB total\n")

        if progress_tracker:
            progress_tracker.update_progress(
                "Loaded depth maps from cache",
                phase="depth_estimation",
                frame_num=len(cached_depths),
                step_name="Depth Map Generation",
                step_progress=len(cached_depths),
                step_total=len(cached_depths),
            )
        return cached_depths

    def _save_to_depth_cache(
        self, video_path: str, settings: dict[str, Any], depth_maps: np.ndarray
    ):
        """
        Save depth maps to global cache.

        Args:
            video_path: Path to video file (for cache key)
            settings: Processing settings for cache key
            depth_maps: Numpy array of depth maps

        Side effects:
            - Filesystem I/O (cache writes)
        """
        if save_depth_maps_to_cache(video_path, settings, depth_maps):
            cache_entries, cache_size_bytes = get_cache_size()
            cache_size_mb = cache_size_bytes / (1024 * 1024)
            print("  Cached depth maps for future use")
            print(f"  Cache: {cache_entries} entries, {cache_size_mb:.1f} MB total\n")
