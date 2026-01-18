"""
Stereo pair generation module.

Converts frames and depth maps into stereo (left/right) pairs using disparity mapping.
"""

from __future__ import annotations

import cv2
import multiprocessing as mp
import numpy as np
from pathlib import Path
from typing import Any

from ...utils import (
    depth_to_disparity,
    create_shifted_image,
    hole_fill_image,
)
from ...core.constants import PREVIEW_FRAME_SAMPLE_RATE


def _process_single_stereo_pair(
    args: tuple[np.ndarray, np.ndarray, str, str | None, str | None, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    PURE worker function to process a single stereo pair in parallel.

    Args:
        args: Tuple of (frame, depth_map, frame_name, left_path, right_path, settings)

    Returns:
        Tuple of (left_img, right_img, frame_name)

    Side effects:
        - Writes to left_path and right_path if provided
    """
    frame, depth_map, frame_name, left_path, right_path, settings = args

    # Create stereo pair
    disparity_map = depth_to_disparity(depth_map, settings["baseline"], settings["focal_length"])

    left_img = create_shifted_image(frame, disparity_map, "left")
    right_img = create_shifted_image(frame, disparity_map, "right")

    # Apply hole filling
    if settings["hole_fill_quality"] in ["fast", "advanced"]:
        left_img = hole_fill_image(left_img, method=settings["hole_fill_quality"])
        right_img = hole_fill_image(right_img, method=settings["hole_fill_quality"])

    # Save if paths provided
    if left_path:
        cv2.imwrite(left_path, left_img)
    if right_path:
        cv2.imwrite(right_path, right_img)

    return left_img, right_img, frame_name


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
        Generate stereo pairs using multiprocessing.

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
            # Determine number of worker processes (leave 1-2 cores for system)
            num_workers = max(1, mp.cpu_count() - 2)
            print(f"  Using {num_workers} parallel workers for stereo generation...")

            # Prepare arguments for parallel processing
            args_list = []
            for frame, depth_map, frame_file in zip(frames, depth_maps, frame_files):
                frame_name = frame_file.stem

                # Determine save paths
                left_path = (
                    str(directories["left_frames"] / f"{frame_name}.png")
                    if settings["keep_intermediates"] and "left_frames" in directories
                    else None
                )
                right_path = (
                    str(directories["right_frames"] / f"{frame_name}.png")
                    if settings["keep_intermediates"] and "right_frames" in directories
                    else None
                )

                args_list.append((frame, depth_map, frame_name, left_path, right_path, settings))

            # Process stereo pairs in parallel
            with mp.Pool(processes=num_workers) as pool:
                # Use imap for progress tracking (processes in order, yields results as ready)
                results = []
                for i, result in enumerate(pool.imap(_process_single_stereo_pair, args_list)):
                    results.append(result)

                    # Update progress
                    if i % 5 == 0 or i == len(args_list) - 1:
                        progress_tracker.update_progress(
                            "Creating stereo pairs",
                            phase="stereo_generation",
                            frame_num=i + 1,
                            step_name="Stereo Pair Creation",
                            step_progress=i + 1,
                            step_total=len(frames),
                        )

                    # Send preview frame for left eye
                    if progress_tracker and hasattr(progress_tracker, "send_preview_frame"):
                        if i % PREVIEW_FRAME_SAMPLE_RATE == 0 or i == len(args_list) - 1:
                            left_path = args_list[i][3]  # left_path from args
                            if left_path:
                                progress_tracker.send_preview_frame(
                                    Path(left_path), "stereo_left", i + 1
                                )

            return True

        except Exception as e:
            print(f"Error creating stereo pairs: {e}")
            import traceback

            traceback.print_exc()
            return False
