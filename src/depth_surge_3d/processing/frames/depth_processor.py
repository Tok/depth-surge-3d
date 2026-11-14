"""
Depth map processing module.

Handles depth map generation with caching, VRAM management, and chunking strategies.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ...core.constants import (
    DEPTH_MAP_SCALE,
    DEFAULT_FALLBACK_FPS,
    DEPTH_RANGE_HALF_LIFE_FRAMES,
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
    DEPTH_MODEL_CHUNK_OVERLAP,
    DEPTH_CHUNK_RAM_BUDGET_BYTES,
    DEPTH_CHUNK_BYTES_PER_PIXEL,
    DEPTH_CHUNK_MIN_FRAMES,
    DEPTH_CHUNK_MAX_FRAMES,
)
from ...io.operations import save_depth_map
from ...utils.domain.depth_alignment import DepthSequenceAligner, RunningDepthRange
from ...utils import calculate_optimal_chunk_size, get_vram_info


class DepthMapProcessor:
    """
    Streaming depth map generation with bounded memory.

    Responsibilities:
    - Depth map generation with model inference, chunk by chunk
    - Temporal alignment of the chunks and running normalization
    - Writing finished depth maps to disk as soon as they are final
    - Saving the stream state so an interrupted run continues where it stopped
    - VRAM/RAM-based chunk sizing and GPU memory management
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
            if self.verbose:
                mem_free = torch.cuda.mem_get_info()[0] / (1024**3)  # Convert to GB
                print(f"  GPU memory freed: {mem_free:.2f} GB available")

    def _load_chunk_frames(self, chunk_files: list[Path], settings: dict[str, Any]) -> list | None:
        """
        Load chunk of frames into memory at source resolution.

        Args:
            chunk_files: List of frame file paths for this chunk
            settings: Processing settings (unused, kept for API compatibility)

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

            # Frames go to the depth model at source resolution. The model resizes to
            # input_size internally, so upscaling here only wastes memory; super sampling
            # is applied later, during stereo generation (see render_stereo_pair).
            chunk_frames.append(frame)

        return chunk_frames if chunk_frames else None

    def _estimate_chunk(
        self, chunk_frames: list, settings: dict[str, Any], input_size: int
    ) -> np.ndarray:
        """
        Run depth estimation on a chunk and return raw inverse depth.

        Normalization happens once for the whole video after the chunks are aligned
        (see DepthSequenceAligner), so the estimator is asked for unnormalized output.

        Args:
            chunk_frames: List of frame images
            settings: Processing settings
            input_size: Depth model input resolution

        Returns:
            Raw inverse depth maps [N, H, W]

        Side effects:
            - GPU inference
        """
        target_fps = settings.get("target_fps", DEFAULT_FALLBACK_FPS)
        if target_fps is None or str(target_fps) == "None" or target_fps == "original":
            target_fps = 30

        return self.depth_estimator.estimate_depth_batch(
            np.array(chunk_frames),
            target_fps=target_fps,
            input_size=input_size,
            fp32=False,
            normalize=False,
        )

    @staticmethod
    def _temporal_chunk_size(frame_w: int, frame_h: int) -> int:
        """
        PURE: Chunk length for temporally consistent models, bounded by system RAM.

        Their VRAM use doesn't depend on the chunk length (fixed internal windows), but
        full-resolution frames and depth maps for the whole chunk are held in RAM.
        """
        frames_in_budget = DEPTH_CHUNK_RAM_BUDGET_BYTES // max(
            1, frame_w * frame_h * DEPTH_CHUNK_BYTES_PER_PIXEL
        )
        return int(min(max(frames_in_budget, DEPTH_CHUNK_MIN_FRAMES), DEPTH_CHUNK_MAX_FRAMES))

    @staticmethod
    def _chunk_overlap(settings: dict[str, Any], chunk_size: int) -> int:
        """PURE: Frames shared by consecutive chunks for scale/shift alignment."""
        requested = settings.get("temporal_window_overlap") or DEPTH_MODEL_CHUNK_OVERLAP
        return int(min(max(int(requested), 4), chunk_size // 2))

    def _send_chunk_preview(self, normalized: np.ndarray, frame_num: int, progress_tracker):
        """Send a live preview of the newest finished depth map."""
        if not progress_tracker or not hasattr(progress_tracker, "send_preview_frame_from_array"):
            return
        if len(normalized) == 0:
            return
        preview_bgr = cv2.cvtColor(
            (normalized[-1] * DEPTH_MAP_SCALE).astype("uint8"), cv2.COLOR_GRAY2BGR
        )
        progress_tracker.send_preview_frame_from_array(preview_bgr, "depth_map", frame_num)

    def stream_depth_maps(  # noqa: C901
        self,
        frame_files: list[Path],
        depth_dir: Path,
        state_path: Path,
        settings: dict[str, Any],
        progress_tracker=None,
    ) -> Iterator[int]:
        """
        Generate depth maps chunk by chunk, writing each one to disk once it is final.

        Continues from ``state_path`` if it exists. After every chunk the finished depth
        maps are written to ``depth_dir`` (16-bit PNG, named like the frames), the state
        is saved and the number of finished depth maps is yielded, so the caller can run
        later pipeline steps on them while the generator is suspended.

        Chunks of temporally consistent models overlap: the overlapping frames are held
        back until the next chunk has been aligned and crossfaded with them.

        Args:
            frame_files: All frame files of the video, in order
            depth_dir: Directory for the depth map PNGs
            state_path: File holding the stream state (created/updated here)
            settings: Processing settings with depth parameters
            progress_tracker: Optional progress tracker

        Yields:
            Number of depth maps finished so far (always increasing, ends at len(frame_files))

        Raises:
            RuntimeError: If frames can't be read or depth estimation fails

        Side effects:
            - GPU inference
            - Writes depth map images and the state file
        """
        num_frames = len(frame_files)
        temporal = getattr(self.depth_estimator, "temporally_consistent", False) is True
        emitted, aligner, depth_range = self._load_stream_state(state_path, temporal)
        if emitted >= num_frames:
            return

        # Frames before `emitted` may be deleted already (runs without intermediates)
        sample_frame = cv2.imread(str(frame_files[emitted]))
        if sample_frame is None:
            raise RuntimeError(f"Could not read frame {frame_files[emitted]}")
        frame_h, frame_w = sample_frame.shape[:2]
        chunk_size, input_size = self._determine_chunk_params(
            frame_w, frame_h, settings.get("depth_resolution", "auto")
        )

        # Temporally consistent models (VDA): long chunks, aligned on overlapping frames.
        # Per-frame models (DA3): VRAM-sized chunks, every frame aligned to its predecessors.
        overlap = 0
        if temporal:
            chunk_size = self._temporal_chunk_size(frame_w, frame_h)
            if settings.get("processing_mode", "batched") != "sequential":
                # Batched mode: don't estimate far beyond the batch being worked on
                try:
                    batch_size = int(settings.get("batch_size") or chunk_size)
                except (TypeError, ValueError):
                    batch_size = chunk_size
                chunk_size = min(chunk_size, max(DEPTH_CHUNK_MIN_FRAMES, batch_size))
            overlap = self._chunk_overlap(settings, chunk_size)
        print(
            f"  Depth chunks of {chunk_size} frames (input_size={input_size}, overlap={overlap})"
            + (f", continuing at frame {emitted + 1}" if emitted else "")
        )
        self._clear_gpu_memory()

        while emitted < num_frames:
            held = len(aligner)  # frames estimated before but not final yet
            chunk_end = min(emitted + held + chunk_size, num_frames)
            chunk_files = frame_files[emitted:chunk_end]
            chunk_frames = self._load_chunk_frames(chunk_files, settings)
            if not chunk_frames or len(chunk_frames) != len(chunk_files):
                raise RuntimeError("Could not load all frames of a depth chunk")

            raw_depths = self._estimate_chunk(chunk_frames, settings, input_size)
            del chunk_frames
            aligner.add(raw_depths, overlap=held)
            del raw_depths
            self._clear_gpu_memory()

            is_last = chunk_end >= num_frames
            normalized = depth_range.normalize(aligner.take(0 if is_last else overlap))
            for offset, depth_map in enumerate(normalized):
                depth_path = depth_dir / f"{frame_files[emitted + offset].stem}.png"
                if not save_depth_map(depth_path, depth_map):
                    raise RuntimeError(f"Could not write depth map {depth_path}")
            emitted += len(normalized)
            self._save_stream_state(state_path, emitted, aligner, depth_range)

            self._send_chunk_preview(normalized, emitted, progress_tracker)
            del normalized
            if progress_tracker:
                progress_tracker.update_progress(
                    f"Depth maps {emitted}/{num_frames}",
                    phase="depth_estimation",
                    frame_num=emitted,
                    step_name="Depth Map Generation",
                    step_progress=emitted,
                    step_total=num_frames,
                )
            yield emitted

    @staticmethod
    def finished_depth_maps(state_path: Path) -> int:
        """
        Number of depth maps a saved stream state has finished (0 without a state).

        Side effects:
            - Reads the state file
        """
        try:
            with np.load(state_path) as data:
                return int(data["emitted"])
        except (OSError, ValueError, KeyError):
            return 0

    @staticmethod
    def mark_stream_complete(state_path: Path, num_frames: int) -> None:
        """
        Record that all depth maps exist (e.g. restored from the global cache).

        Side effects:
            - Writes the state file
        """
        DepthMapProcessor._write_state(state_path, {"emitted": np.array(num_frames)})

    @staticmethod
    def _load_stream_state(
        state_path: Path, temporal: bool
    ) -> tuple[int, DepthSequenceAligner, RunningDepthRange]:
        """
        Load the stream state, or start a new stream.

        Returns:
            Tuple of (finished depth maps, aligner, running depth range)

        Side effects:
            - Reads the state file
        """
        aligner = DepthSequenceAligner(per_frame=not temporal)
        depth_range = RunningDepthRange(half_life_frames=DEPTH_RANGE_HALF_LIFE_FRAMES)
        try:
            with np.load(state_path) as data:
                emitted = int(data["emitted"])
                aligner.restore({key: data[key] for key in ("tail", "reference") if key in data})
                if "range" in data:
                    depth_range.low, depth_range.high = (float(v) for v in data["range"])
            return emitted, aligner, depth_range
        except (OSError, ValueError, KeyError):
            return 0, DepthSequenceAligner(per_frame=not temporal), depth_range

    @staticmethod
    def _save_stream_state(
        state_path: Path,
        emitted: int,
        aligner: DepthSequenceAligner,
        depth_range: RunningDepthRange,
    ) -> None:
        """
        Save what's needed to continue the stream after the frames finished so far.

        Side effects:
            - Writes the state file
        """
        arrays: dict[str, np.ndarray] = {"emitted": np.array(emitted)}
        arrays.update(aligner.state())
        if depth_range.low is not None and depth_range.high is not None:
            arrays["range"] = np.array([depth_range.low, depth_range.high], dtype=np.float64)
        DepthMapProcessor._write_state(state_path, arrays)

    @staticmethod
    def _write_state(state_path: Path, arrays: dict[str, np.ndarray]) -> None:
        """Write the state file atomically (an interrupted write keeps the old state)."""
        temp_path = state_path.with_name(state_path.stem + ".tmp.npz")
        np.savez(temp_path, **arrays)  # type: ignore[arg-type]
        os.replace(temp_path, state_path)
