"""
Stereo pair generation module.

Converts frames and depth maps into stereo (left/right) pairs using disparity mapping.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import cv2

from ...io.operations import load_depth_map
from ...utils import render_stereo_pair
from ...core.constants import MAX_STEREO_WORKERS


def _init_stereo_worker() -> None:
    """
    Worker initializer: keep each process single-threaded.

    Without this every worker spins up one OpenCV thread per core, which on many-core
    machines (or containers with a pids limit) can exhaust thread/process limits.
    """
    cv2.setNumThreads(1)
    # Ctrl+C is handled by the parent, which cancels the queued work and stops the pool
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _process_single_stereo_pair(
    args: tuple[str, str, str, str | None, str | None, dict[str, Any]],
) -> str:
    """
    Worker function to process a single stereo pair.

    Frame and depth map are read here, in the worker: passing paths instead of arrays keeps
    the parent's memory flat and avoids pickling full frames into every worker.

    Args:
        args: Tuple of (frame_path, depth_path, frame_name, left_path, right_path, settings)

    Returns:
        The frame name

    Side effects:
        - Reads the frame and its depth map
        - Writes to left_path and right_path if provided
    """
    frame_path, depth_path, frame_name, left_path, right_path, settings = args

    frame = cv2.imread(frame_path)
    if frame is None:
        raise IOError(f"Failed to read {frame_path}")
    depth_map = load_depth_map(depth_path)
    if depth_map is None:
        raise IOError(f"Failed to read {depth_path}")

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
        self._executor: ProcessPoolExecutor | None = None
        self._executor_workers = 0

    def create_stereo_pairs(
        self,
        frame_files: list[Path],
        depth_files: list[Path],
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker=None,
    ) -> bool:
        """
        Generate stereo pairs for a batch of frames using a process pool.

        The pool is kept for the next batch; call ``close()`` when done.

        Args:
            frame_files: Frame files of the batch
            depth_files: Depth map files of the same frames
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
            if len(frame_files) != len(depth_files):
                print(f"Error: {len(frame_files)} frames but {len(depth_files)} depth maps")
                return False

            args_list = self._build_args(frame_files, depth_files, directories, settings)
            if not args_list:
                print("Error: No frames to process")
                return False

            num_workers = get_stereo_worker_count(len(args_list))
            if num_workers <= 1:
                self._run_sequential(args_list, 0, progress_tracker)
                return True

            try:
                self._run_parallel(args_list, num_workers, progress_tracker)
            except BrokenProcessPool as e:
                # A worker died without reporting (OOM killer, process limit, segfault).
                # Redo the work in-process so the job finishes or surfaces the real error.
                print(f"  Warning: stereo worker pool crashed ({e}); retrying in-process...")
                self.close()
                self._run_sequential(args_list, 0, progress_tracker)

            return True

        except (InterruptedError, KeyboardInterrupt):
            self.close()  # stop requested: drop the queued pairs instead of finishing them
            raise
        except Exception as e:
            print(f"Error creating stereo pairs: {e}")
            traceback.print_exc()
            return False

    def close(self) -> None:
        """
        Shut down the worker pool (queued work is cancelled).

        Side effects:
            - Stops worker processes
        """
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
            self._executor_workers = 0

    @staticmethod
    def _build_args(
        frame_files: list[Path],
        depth_files: list[Path],
        directories: dict[str, Path],
        settings: dict[str, Any],
    ) -> list[tuple]:
        """PURE: Build worker argument tuples."""
        # Stereo pairs are always written: the distortion/crop steps read them from disk
        args_list = []
        for frame_file, depth_file in zip(frame_files, depth_files):
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
            args_list.append(
                (str(frame_file), str(depth_file), frame_name, left_path, right_path, settings)
            )
        return args_list

    def _get_executor(self, num_workers: int) -> ProcessPoolExecutor:
        """
        Worker pool, created on first use and reused for later batches.

        Workers are started with "spawn", not "fork": by this point the parent has torch,
        CUDA and OpenCV worker threads running, and a forked child inherits any lock one
        of those threads held at fork time, then blocks on it forever (#17).
        """
        if self._executor is None or self._executor_workers < num_workers:
            self.close()
            self._executor = ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=mp.get_context("spawn"),
                initializer=_init_stereo_worker,
            )
            self._executor_workers = num_workers
        return self._executor

    def _run_parallel(self, args_list: list[tuple], num_workers: int, progress_tracker) -> None:
        """Process pairs in the pool. Raises BrokenProcessPool if a worker dies."""
        executor = self._get_executor(num_workers)
        results = executor.map(_process_single_stereo_pair, args_list, chunksize=1)
        for i, _ in enumerate(results):
            self._report_progress(i, args_list, progress_tracker)

    def map_in_pool(self, function, tasks: list, on_result=None) -> None:
        """
        Run a picklable function over tasks in the worker pool, in order.

        Uses the same pool as stereo generation. With a single worker, or after the pool
        crashed, the tasks run in this process instead.

        Args:
            function: Module-level function taking one task
            tasks: Task arguments
            on_result: Called with the task index after each task finished (in order).
                An exception it raises (e.g. a stop request) propagates.

        Side effects:
            - Parallel processing via multiprocessing
        """

        def sequential() -> None:
            for index, task in enumerate(tasks):
                function(task)
                if on_result:
                    on_result(index)

        num_workers = get_stereo_worker_count(len(tasks))
        if num_workers <= 1:
            sequential()
            return
        try:
            results = self._get_executor(num_workers).map(function, tasks, chunksize=1)
            for index, _ in enumerate(results):
                if on_result:
                    on_result(index)
        except BrokenProcessPool as e:
            # A worker died without reporting (OOM killer, process limit, segfault)
            print(f"  Warning: worker pool crashed ({e}); retrying in-process...")
            self.close()
            sequential()

    def _run_sequential(self, args_list: list[tuple], start: int, progress_tracker) -> None:
        """Process pairs in the current process."""
        for i in range(start, len(args_list)):
            _process_single_stereo_pair(args_list[i])
            self._report_progress(i, args_list, progress_tracker)

    @staticmethod
    def _report_progress(i: int, args_list: list[tuple], progress_tracker) -> None:
        """Send a progress update for the i-th finished pair."""
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
