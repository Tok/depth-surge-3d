"""
AI upscaling module.

Handles AI-based frame upscaling using Real-ESRGAN or similar models.
"""

from __future__ import annotations

import cv2
import traceback
from pathlib import Path
from typing import Any

from ...core.constants import INTERMEDIATE_DIRS, PREVIEW_FRAME_SAMPLE_RATE


class FrameUpscalerProcessor:
    """
    AI upscaling orchestrator for stereo frames.

    Responsibilities:
    - Model loading and management
    - Batch frame processing
    - GPU memory management during upscaling
    - Directory resolution for source frames
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize frame upscaler processor.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.upscaler: Any = None  # loaded on first use, kept for later batches

    def apply_upscaling(
        self,
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker=None,
        frame_names: list[str] | None = None,
    ) -> bool:
        """
        Apply AI upscaling to stereo frames.

        The model stays loaded for the next batch; call ``close()`` when done.

        Args:
            directories: Dictionary of processing directories
            settings: Processing settings with upscale_model parameter
            progress_tracker: Optional progress tracker
            frame_names: Frame names (file stems) to upscale; all frames if None

        Returns:
            True if successful, False otherwise

        Side effects:
            - Loads upscaling model into GPU memory
            - Reads and writes frame images
            - Modifies GPU memory state
        """
        try:
            # Get source directories (cropped frames)
            source_left, source_right = self._get_upscaling_source_dirs(directories, settings)  # type: ignore[misc]
            if source_left is None or source_right is None:
                return False

            if self.upscaler is None:
                from ...inference import create_upscaler

                upscaler = create_upscaler(
                    model_name=settings["upscale_model"],
                    device=settings.get("device", "auto"),
                )
                if upscaler is None:
                    return True  # No upscaling
                if not upscaler.load_model():
                    print("Failed to load upscaling model")
                    return False
                self.upscaler = upscaler

            return self._process_upscaling_frames(
                self.upscaler,
                source_left,
                source_right,
                directories,
                settings,
                progress_tracker,
                frame_names,
            )

        except InterruptedError:
            raise
        except Exception as e:
            print(f"Error applying upscaling: {e}")

            traceback.print_exc()
            return False

    def close(self) -> None:
        """
        Unload the upscaling model.

        Side effects:
            - Frees GPU memory
        """
        if self.upscaler is not None:
            self.upscaler.unload_model()
            self.upscaler = None

    def _process_upscaling_frames(
        self,
        upscaler,
        left_dir,
        right_dir,
        directories,
        settings,
        progress_tracker,
        frame_names: list[str] | None = None,
    ) -> bool:
        """
        Process frames in batches with model management.

        Args:
            upscaler: Upscaler instance
            left_dir: Input directory for left frames
            right_dir: Input directory for right frames
            directories: Dictionary of processing directories
            settings: Processing settings
            progress_tracker: Optional progress tracker
            frame_names: Frame names (file stems) to upscale; all frames if None

        Returns:
            True if successful, False otherwise

        Side effects:
            - GPU memory operations
            - Filesystem I/O
        """
        if frame_names is not None:
            left_files = [left_dir / f"{name}.png" for name in frame_names]
            right_files = [right_dir / f"{name}.png" for name in frame_names]
        else:
            left_files = sorted(list(left_dir.glob("*.png")))
            right_files = sorted(list(right_dir.glob("*.png")))

        if len(left_files) != len(right_files):
            print(f"Frame count mismatch: {len(left_files)} left, {len(right_files)} right")
            return False

        # Upscaled frames are always written: VR assembly reads them from disk (without them it
        # silently fell back to the non-upscaled frames when intermediates weren't kept)
        left_upscaled = directories.get("left_upscaled") or (
            directories["base"] / INTERMEDIATE_DIRS["left_upscaled"]
        )
        right_upscaled = directories.get("right_upscaled") or (
            directories["base"] / INTERMEDIATE_DIRS["right_upscaled"]
        )
        left_upscaled.mkdir(exist_ok=True)
        right_upscaled.mkdir(exist_ok=True)
        directories["left_upscaled"] = left_upscaled
        directories["right_upscaled"] = right_upscaled

        # Process frames
        for i, (left_file, right_file) in enumerate(zip(left_files, right_files)):
            self._upscale_frame_pair(
                upscaler,
                left_file,
                right_file,
                left_upscaled,
                right_upscaled,
                settings,
                i,
                len(left_files),
                progress_tracker,
            )

        return True

    def _upscale_frame_pair(
        self,
        upscaler,
        left_file,
        right_file,
        left_upscaled,
        right_upscaled,
        settings,
        frame_idx,
        total_frames,
        progress_tracker,
    ):
        """
        Upscale single frame pair.

        Args:
            upscaler: Upscaler instance
            left_file: Input left frame path
            right_file: Input right frame path
            left_upscaled: Output left directory
            right_upscaled: Output right directory
            settings: Processing settings
            frame_idx: Current frame index
            total_frames: Total number of frames
            progress_tracker: Optional progress tracker

        Side effects:
            - GPU inference
            - Filesystem I/O
        """
        left_img = cv2.imread(str(left_file))
        right_img = cv2.imread(str(right_file))

        if left_img is None or right_img is None:
            print(f"Warning: Could not load {left_file} or {right_file}")
            return

        # Upscale
        left_upscaled_img = upscaler.upscale_image(left_img)
        right_upscaled_img = upscaler.upscale_image(right_img)

        frame_name = left_file.stem

        # Determine if we should send preview for this frame
        should_send_preview = (
            progress_tracker
            and hasattr(progress_tracker, "send_preview_frame_from_array")
            and (frame_idx % PREVIEW_FRAME_SAMPLE_RATE == 0 or frame_idx == total_frames - 1)
        )

        # Always save (VR assembly reads these from disk)
        if left_upscaled:
            cv2.imwrite(str(left_upscaled / f"{frame_name}.png"), left_upscaled_img)
        if right_upscaled:
            cv2.imwrite(str(right_upscaled / f"{frame_name}.png"), right_upscaled_img)

        if should_send_preview:
            progress_tracker.send_preview_frame_from_array(
                left_upscaled_img, "upscaled_left", frame_idx + 1
            )

        # Progress update (every frame since upscaling is slow)
        if frame_idx % 1 == 0 or frame_idx == total_frames - 1:
            progress_tracker.update_progress(
                f"Upscaling frame {frame_idx+1}/{total_frames}",
                phase="upscaling",
                frame_num=frame_idx + 1,
                step_name="AI Upscaling",
                step_progress=frame_idx + 1,
                step_total=total_frames,
            )

    @staticmethod
    def _get_upscaling_source_dirs(
        directories: dict[str, Path], settings: dict[str, Any]
    ) -> tuple[Path, Path] | None:
        """
        PURE: Determine source directories for upscaling.

        Args:
            directories: Dictionary of processing directories
            settings: Processing settings with apply_distortion flag

        Returns:
            Tuple of (left_source_dir, right_source_dir)
        """
        # Upscaling happens at Step 6.5, after VR assembly/cropping (Step 6)
        if "left_cropped" in directories and "right_cropped" in directories:
            return directories["left_cropped"], directories["right_cropped"]
        else:
            return None, None  # type: ignore[return-value]
