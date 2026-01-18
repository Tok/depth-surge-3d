"""
VR frame assembly module.

Assembles stereo pairs into final VR format (side-by-side or over-under).
"""

from __future__ import annotations

import cv2
from pathlib import Path
from typing import Any

from ...utils import (
    resize_image,
    create_vr_frame,
)


class VRFrameAssembler:
    """
    Assembles stereo frames into final VR format.

    Responsibilities:
    - Combine left/right frames into VR format
    - Support side-by-side and over-under layouts
    - Directory resolution for source frames
    - Batch frame processing
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize VR frame assembler.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose

    def assemble_vr_frames(
        self,
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker=None,
        total_frames: int = 0,
    ) -> bool:
        """
        Assemble stereo frames into final VR format.

        Args:
            directories: Dictionary of processing directories
            settings: Processing settings with vr_format parameter
            progress_tracker: Optional progress tracker
            total_frames: Total number of frames (for progress tracking)

        Returns:
            True if successful, False otherwise

        Side effects:
            - Reads stereo frame pairs from disk
            - Writes assembled VR frames to disk
        """
        try:
            # Determine source directories (upscaled if available, otherwise cropped)
            source_dirs = self._get_vr_assembly_source_dirs(directories, settings)
            if not source_dirs:
                return False

            left_dir, right_dir = source_dirs

            # Get frame files
            left_files = sorted(left_dir.glob("*.png"))
            right_files = sorted(right_dir.glob("*.png"))

            if len(left_files) != len(right_files):
                print(
                    f"Warning: Mismatched frame count: {len(left_files)} left, {len(right_files)} right"
                )

            # Process each frame pair
            for i, (left_file, right_file) in enumerate(zip(left_files, right_files)):
                self._assemble_single_vr_frame(
                    left_file, right_file, directories, settings, progress_tracker, i
                )

                # Update progress more frequently (every frame for slow operations)
                if i % 1 == 0 or i == len(left_files) - 1:
                    progress_tracker.update_progress(
                        f"Assembling VR frame {i + 1}/{len(left_files)}",
                        phase="vr_assembly",
                        frame_num=i + 1,
                        step_name="VR Assembly",
                        step_progress=i + 1,
                        step_total=len(left_files),
                    )

            return True

        except Exception as e:
            print(f"Error assembling VR frames: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _assemble_single_vr_frame(
        self,
        left_file: Path,
        right_file: Path,
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker=None,
        frame_idx: int = 0,
    ) -> bool:
        """
        Assemble a single VR frame from cropped/upscaled frames.

        Args:
            left_file: Left frame file path
            right_file: Right frame file path
            directories: Dictionary of processing directories
            settings: Processing settings with vr_format, per_eye_width, per_eye_height
            progress_tracker: Optional progress tracker
            frame_idx: Frame index for progress tracking

        Returns:
            True if successful, False otherwise

        Side effects:
            - Reads images from disk
            - Writes VR frame to disk
            - Sends preview frame if progress_tracker supports it
        """
        # Load images
        left_img = cv2.imread(str(left_file))
        right_img = cv2.imread(str(right_file))

        if left_img is None or right_img is None:
            print(f"Warning: Could not load {left_file} or {right_file}")
            return False

        # Resize to final target resolution
        left_final = resize_image(left_img, settings["per_eye_width"], settings["per_eye_height"])
        right_final = resize_image(right_img, settings["per_eye_width"], settings["per_eye_height"])

        # Create and save final VR frame
        vr_frame = create_vr_frame(left_final, right_final, settings["vr_format"])
        frame_name = left_file.stem
        if "vr_frames" in directories:
            vr_path = directories["vr_frames"] / f"{frame_name}.png"
            cv2.imwrite(str(vr_path), vr_frame)

            # Send preview frame
            if progress_tracker and hasattr(progress_tracker, "send_preview_frame"):
                # Send every VR frame preview (already throttled by time in send_preview_frame)
                progress_tracker.send_preview_frame(vr_path, "vr_frame", frame_idx + 1)

        return True

    @staticmethod
    def _get_vr_assembly_source_dirs(
        directories: dict[str, Path], settings: dict[str, Any]
    ) -> tuple[Path, Path] | None:
        """
        PURE: Determine source directories for VR assembly.

        Args:
            directories: Dictionary of processing directories
            settings: Processing settings with upscale_model parameter

        Returns:
            Tuple of (left_source_dir, right_source_dir) or None if not found
        """
        # Priority: upscaled (if enabled) > cropped
        if settings.get("upscale_model", "none") != "none" and "left_upscaled" in directories:
            left_dir = directories["left_upscaled"]
            right_dir = directories["right_upscaled"]
            if left_dir.exists() and right_dir.exists():
                left_files = list(left_dir.glob("*.png"))
                if left_files:  # Verify directory has frames
                    return left_dir, right_dir

        # Fallback to cropped frames
        if "left_cropped" in directories and "right_cropped" in directories:
            left_dir = directories["left_cropped"]
            right_dir = directories["right_cropped"]
            if left_dir.exists() and right_dir.exists():
                left_files = list(left_dir.glob("*.png"))
                if left_files:  # Verify directory has frames
                    return left_dir, right_dir

        print("Error: No cropped or upscaled frames found for VR assembly")
        return None
