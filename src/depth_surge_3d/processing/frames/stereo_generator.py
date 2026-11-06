"""
Stereo pair generation module.

Converts frames and depth maps into stereo (left/right) pairs using disparity mapping.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ...utils import render_stereo_pair
from ...core.constants import MAX_STEREO_WORKERS, PREVIEW_FRAME_SAMPLE_RATE


def _init_stereo_worker() -> None:
    """
    Worker initializer: keep each process single-threaded.

    Without this every worker spins up one OpenCV thread per core, which on many-core
    machines (or containers with a pids limit) can exhaust thread/process limits.
    """
    cv2.setNumThreads(1)


def _process_single_stereo_pair(
    args: tuple[np.ndarray, np.ndarray, str, str | None, str | None, dict[str, Any]],
) -> str:
    """
    Worker function to process a single stereo pair.

    Args:
        args: Tuple of (frame, depth_map, frame_name, left_path, right_path, settings)

    Returns:
        The frame name

    Side effects:
        - Writes to left_path and right_path if provided
    """
    frame, depth_map, frame_name, left_path, right_path, settings = args

    left_img, right_img = render_stereo_pair(frame, depth_map, settings)

    if left_path and not cv2.imwrite(left_path, left_img):
        raise IOError(f"Failed to write {left_path}")
    if right_path and not cv2.imwrite(right_path, right_img):
        raise IOError(f"Failed to write {right_path}")

    return frame_name


def get_stereo_worker_count(num_tasks: int) -> int:
    """
    PURE-ish: Number of worker processes to use for stereo generation.

    Leaves two cores for the system and caps the pool, since every worker holds a full
    frame and depth map in memory and more workers rarely help (the work is memory bound).
    """
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count - 2, MAX_STEREO_WORKERS, num_tasks))


class StereoPairGenerator:
    """
    Generates stereo pairs from frames and depth maps.

    Responsibilities:
    - Convert depth maps to disparity maps
    - Create left/right shifted images
    - Apply hole filling
    - Parallel processing orchestration
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize stereo pair generator.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose

    def create_stereo_pairs(
        self,
        frames: np.ndarray,
        depth_maps: np.ndarray,
        frame_files: list[Path],
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker=None,
    ) -> bool:
        """
        Generate stereo pairs using a process pool.

        Args:
            frames: Array of frame images
            depth_maps: Array of depth map images
            frame_files: List of frame file paths
            directories: Dictionary of processing directories
            settings: Processing settings with baseline, focal_length, etc.
            progress_tracker: Optional progress tracker

        Returns:
            True if successful, False otherwise

        Side effects:
            - Parallel processing via multiprocessing
            - Writes stereo pair images to disk
        """
        try:
            if len(frames) != len(depth_maps):
                print(
                    f"Error: {len(frames)} frames but {len(depth_maps)} depth maps "
                    "(stale depth cache? try --clear-cache)"
                )
                return False

            args_list = self._build_args(frames, depth_maps, frame_files, directories, settings)
            if not args_list:
                print("Error: No frames to process")
                return False

            num_workers = get_stereo_worker_count(len(args_list))
            if num_workers <= 1:
                print("  Generating stereo pairs in-process...")
                self._run_sequential(args_list, 0, progress_tracker)
                return True

            print(f"  Using {num_workers} parallel workers for stereo generation...")
            try:
                self._run_parallel(args_list, num_workers, progress_tracker)
            except BrokenProcessPool as e:
                # A worker died without reporting (OOM killer, process limit, segfault).
                # Redo the work in-process so the job finishes or surfaces the real error.
                print(f"  Warning: stereo worker pool crashed ({e}); retrying in-process...")
                self._run_sequential(args_list, 0, progress_tracker)

            return True

        except Exception as e:
            print(f"Error creating stereo pairs: {e}")
            traceback.print_exc()
            return False

    @staticmethod
    def _build_args(
        frames: np.ndarray,
        depth_maps: np.ndarray,
        frame_files: list[Path],
        directories: dict[str, Path],
        settings: dict[str, Any],
    ) -> list[tuple]:
        """PURE: Build worker argument tuples."""
        # Stereo pairs are always written: the distortion/crop steps read them from disk
        args_list = []
        for frame, depth_map, frame_file in zip(frames, depth_maps, frame_files):
            frame_name = frame_file.stem
            left_path = (
                str(directories["left_frames"] / f"{frame_name}.png")
                if "left_frames" in directories
                else None
            )
            right_path = (
                str(directories["right_frames"] / f"{frame_name}.png")
                if "right_frames" in directories
                else None
            )
            args_list.append((frame, depth_map, frame_name, left_path, right_path, settings))
        return args_list

    def _run_parallel(self, args_list: list[tuple], num_workers: int, progress_tracker) -> None:
        """
        Process pairs in a pool. Raises BrokenProcessPool if a worker dies.

        Workers are started with "spawn", not "fork": by this point the parent has torch,
        CUDA and OpenCV worker threads running, and a forked child inherits any lock one
        of those threads held at fork time, then blocks on it forever (#17).
        """
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_stereo_worker,
        ) as executor:
            results = executor.map(_process_single_stereo_pair, args_list, chunksize=1)
            for i, _ in enumerate(results):
                self._report_progress(i, args_list, progress_tracker)

    def _run_sequential(self, args_list: list[tuple], start: int, progress_tracker) -> None:
        """Process pairs in the current process."""
        for i in range(start, len(args_list)):
            _process_single_stereo_pair(args_list[i])
            self._report_progress(i, args_list, progress_tracker)

    @staticmethod
    def _report_progress(i: int, args_list: list[tuple], progress_tracker) -> None:
        """Send progress and preview updates for the i-th finished pair."""
        if not progress_tracker:
            return
        total = len(args_list)
        is_last = i == total - 1

        if i % 5 == 0 or is_last:
            progress_tracker.update_progress(
                "Creating stereo pairs",
                phase="stereo_generation",
                frame_num=i + 1,
                step_name="Stereo Pair Creation",
                step_progress=i + 1,
                step_total=total,
            )

        if hasattr(progress_tracker, "send_preview_frame"):
            if i % PREVIEW_FRAME_SAMPLE_RATE == 0 or is_last:
                left_path = args_list[i][3]
                if left_path:
                    progress_tracker.send_preview_frame(Path(left_path), "stereo_left", i + 1)
