"""
VR distortion processing module.

Applies fisheye distortion and cropping to stereo frames for VR viewing.
"""

from __future__ import annotations

import cv2
from pathlib import Path
from typing import Any

from ...utils import (
    apply_fisheye_distortion,
    apply_fisheye_square_crop,
    apply_center_crop,
)


class DistortionProcessor:
    """
    Applies VR fisheye distortion and cropping to stereo frames.

    Responsibilities:
    - Apply fisheye distortion for VR headsets
    - Crop frames to VR specifications
    - Batch process frame pairs
    - Source directory resolution
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize distortion processor.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose

    def apply_distortion(
        self,
        left_files: list[Path],
        right_files: list[Path],
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker=None,
    ) -> bool:
        """
        Apply fisheye distortion to stereo frames.

        Args:
            left_files: List of left frame file paths
            right_files: List of right frame file paths
            directories: Dictionary of processing directories
            settings: Processing settings with distortion parameters
            progress_tracker: Optional progress tracker

        Returns:
            True if successful, False otherwise

        Side effects:
            - Reads frames from source directories
            - Writes distorted frames to output directories
        """
        try:
            for i, (left_file, right_file) in enumerate(zip(left_files, right_files)):
                # Load images
                left_img = cv2.imread(str(left_file))
                right_img = cv2.imread(str(right_file))

                if left_img is None or right_img is None:
                    print(f"Warning: Could not load {left_file} or {right_file}")
                    continue

                # Apply fisheye distortion
                left_distorted = apply_fisheye_distortion(
                    left_img, settings["fisheye_fov"], settings["fisheye_projection"]
                )
                right_distorted = apply_fisheye_distortion(
                    right_img, settings["fisheye_fov"], settings["fisheye_projection"]
                )

                # Save distorted frames if keeping intermediates
                if settings["keep_intermediates"]:
                    frame_name = left_file.stem
                    if "left_distorted" in directories:
                        cv2.imwrite(
                            str(directories["left_distorted"] / f"{frame_name}.png"),
                            left_distorted,
                        )
                    if "right_distorted" in directories:
                        cv2.imwrite(
                            str(directories["right_distorted"] / f"{frame_name}.png"),
                            right_distorted,
                        )

                # Update progress
                if i % 5 == 0 or i == len(left_files) - 1:
                    progress_tracker.update_progress(
                        "Applying distortion",
                        phase="distortion",
                        frame_num=i + 1,
                        step_name="Fisheye Distortion",
                        step_progress=i + 1,
                        step_total=len(left_files),
                    )

            return True

        except Exception as e:
            print(f"Error applying distortion: {e}")
            return False

    def crop_frames(
        self,
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker=None,
        total_frames: int = 0,
    ) -> bool:
        """
        Crop frames to VR specifications.

        Args:
            directories: Dictionary of processing directories
            settings: Processing settings with crop parameters
            progress_tracker: Optional progress tracker
            total_frames: Total number of frames (for progress tracking)

        Returns:
            True if successful, False otherwise

        Side effects:
            - Reads frames from source directories
            - Writes cropped frames to output directories
        """
        try:
            # Determine source directories (distorted or original stereo)
            stereo_dirs = self._get_stereo_source_dirs(directories, settings)
            if not stereo_dirs:
                return False

            left_dir, right_dir = stereo_dirs

            # Get frame files
            left_files = sorted(left_dir.glob("*.png"))
            right_files = sorted(right_dir.glob("*.png"))

            if len(left_files) != len(right_files):
                print(
                    f"Warning: Mismatched frame count: {len(left_files)} left, {len(right_files)} right"
                )

            # Process each frame pair
            for i, (left_file, right_file) in enumerate(zip(left_files, right_files)):
                self._crop_single_frame_pair(left_file, right_file, directories, settings)

                # Update progress more frequently (every frame for slow operations)
                if i % 1 == 0 or i == len(left_files) - 1:
                    progress_tracker.update_progress(
                        f"Cropping frame {i + 1}/{len(left_files)}",
                        phase="cropping",
                        frame_num=i + 1,
                        step_name="Frame Cropping",
                        step_progress=i + 1,
                        step_total=len(left_files),
                    )

            return True

        except Exception as e:
            print(f"Error cropping frames: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _crop_single_frame_pair(
        self,
        left_file: Path,
        right_file: Path,
        directories: dict[str, Path],
        settings: dict[str, Any],
    ) -> bool:
        """
        Crop a single stereo frame pair.

        Args:
            left_file: Left frame file path
            right_file: Right frame file path
            directories: Dictionary of processing directories
            settings: Processing settings with crop parameters

        Returns:
            True if successful, False otherwise

        Side effects:
            - Reads images from disk
            - Writes cropped images to disk
        """
        # Load images
        left_img = cv2.imread(str(left_file))
        right_img = cv2.imread(str(right_file))

        if left_img is None or right_img is None:
            print(f"Warning: Could not load {left_file} or {right_file}")
            return False

        # Crop based on distortion setting
        crop_factor = (
            max(0.5, min(2.0, float(settings.get("fisheye_crop_factor", 0.7))))
            if settings["apply_distortion"]
            else max(0.5, min(1.0, float(settings.get("crop_factor", 1.0))))
        )

        if settings["apply_distortion"]:
            left_cropped = apply_fisheye_square_crop(
                left_img,
                settings["per_eye_width"],
                settings["per_eye_height"],
                crop_factor,
            )
            right_cropped = apply_fisheye_square_crop(
                right_img,
                settings["per_eye_width"],
                settings["per_eye_height"],
                crop_factor,
            )
        else:
            left_cropped = apply_center_crop(left_img, crop_factor)
            right_cropped = apply_center_crop(right_img, crop_factor)

        # Save cropped frames (always - needed for upscaling or VR assembly)
        frame_name = left_file.stem
        if "left_cropped" in directories:
            cv2.imwrite(str(directories["left_cropped"] / f"{frame_name}.png"), left_cropped)
        if "right_cropped" in directories:
            cv2.imwrite(str(directories["right_cropped"] / f"{frame_name}.png"), right_cropped)

        return True

    @staticmethod
    def _get_stereo_source_dirs(
        directories: dict[str, Path], settings: dict[str, Any]
    ) -> tuple[Path, Path] | None:
        """
        PURE: Determine source directories for cropping.

        Args:
            directories: Dictionary of processing directories
            settings: Processing settings with apply_distortion flag

        Returns:
            Tuple of (left_source_dir, right_source_dir) or None if not found
        """
        # Priority: distorted (if enabled) > original stereo
        if settings.get("apply_distortion") and "left_distorted" in directories:
            left_dir = directories["left_distorted"]
            right_dir = directories["right_distorted"]
            if left_dir.exists() and right_dir.exists():
                left_files = list(left_dir.glob("*.png"))
                if left_files:  # Verify directory has frames
                    return left_dir, right_dir

        if "left_frames" in directories:
            left_dir = directories["left_frames"]
            right_dir = directories["right_frames"]
            if left_dir.exists() and right_dir.exists():
                left_files = list(left_dir.glob("*.png"))
                if left_files:  # Verify directory has frames
                    return left_dir, right_dir

        print("Error: No stereo frames found")
        return None
