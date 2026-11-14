"""
Fused per-frame pipeline.

Runs stereo generation, fisheye distortion, cropping and (optionally) VR assembly for a
frame in one worker, passing the images in memory. The separate step processors do the
same work but write and re-read a PNG between every step; they remain for redoing single
steps of frames whose intermediates are on disk.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import cv2

from ...core.constants import PREVIEW_FRAME_SAMPLE_RATE
from ...io.operations import load_depth_map
from ...utils import (
    apply_center_crop,
    apply_fisheye_distortion,
    apply_fisheye_square_crop,
    create_vr_frame,
    fit_to_canvas,
    render_stereo_pair,
)
from .distortion_processor import DistortionProcessor

# (frame path, depth map path, frame name, output directories by key, settings, assemble)
FrameTask = tuple[str, str, str, dict[str, str], dict[str, Any], bool]


def _write_pair(outputs: dict[str, str], key: str, name: str, left: Any, right: Any) -> None:
    """Write a left/right image pair if its directories are requested."""
    for side, image in (("left", left), ("right", right)):
        directory = outputs.get(f"{side}_{key}")
        if directory:
            path = f"{directory}/{name}.png"
            if not cv2.imwrite(path, image):
                raise IOError(f"Failed to write {path}")


def process_frame(task: FrameTask) -> str:
    """
    Worker function: turn one frame and its depth map into cropped eyes or a VR frame.

    Args:
        task: (frame_path, depth_path, frame_name, outputs, settings, assemble).
            ``outputs`` maps directory keys (left_frames, right_frames, left_distorted,
            right_distorted, left_cropped, right_cropped, vr_frames) to directories; only
            the listed ones are written. ``assemble`` adds the VR assembly step.

    Returns:
        The frame name

    Side effects:
        - Reads the frame and its depth map
        - Writes the requested images
    """
    frame_path, depth_path, name, outputs, settings, assemble = task

    frame = cv2.imread(frame_path)
    if frame is None:
        raise IOError(f"Failed to read {frame_path}")
    depth_map = load_depth_map(depth_path)
    if depth_map is None:
        raise IOError(f"Failed to read {depth_path}")

    left, right = render_stereo_pair(frame, depth_map, settings)
    _write_pair(outputs, "frames", name, left, right)

    eye_w, eye_h = settings["per_eye_width"], settings["per_eye_height"]
    if settings.get("apply_distortion", True):
        fit_radius = DistortionProcessor._fisheye_crop_factor(settings)
        fov, projection = settings["fisheye_fov"], settings["fisheye_projection"]
        left = apply_fisheye_distortion(left, fov, projection, fit_radius)
        right = apply_fisheye_distortion(right, fov, projection, fit_radius)
        _write_pair(outputs, "distorted", name, left, right)
        left = apply_fisheye_square_crop(left, eye_w, eye_h, fit_radius)
        right = apply_fisheye_square_crop(right, eye_w, eye_h, fit_radius)
    else:
        crop_factor = max(0.5, min(1.0, float(settings.get("crop_factor", 1.0))))
        left = apply_center_crop(left, crop_factor)
        right = apply_center_crop(right, crop_factor)
    _write_pair(outputs, "cropped", name, left, right)

    if assemble:
        vr_frame = create_vr_frame(
            fit_to_canvas(left, eye_w, eye_h),
            fit_to_canvas(right, eye_w, eye_h),
            settings["vr_format"],
        )
        vr_path = f"{outputs['vr_frames']}/{name}.png"
        if not cv2.imwrite(vr_path, vr_frame):
            raise IOError(f"Failed to write {vr_path}")

    return name


class FusedFrameProcessor:
    """
    Runs the fused per-frame pipeline for batches of frames in the stereo worker pool.

    Responsibilities:
    - Decide which images a batch has to write (intermediates only when they are kept)
    - Run the frames in parallel
    - Report progress for every step the fused pipeline covers
    """

    def __init__(self, stereo_generator, verbose: bool = False):
        """
        Args:
            stereo_generator: StereoPairGenerator whose worker pool is used
            verbose: Enable verbose output
        """
        self.stereo_generator = stereo_generator
        self.verbose = verbose

    @staticmethod
    def step_names(settings: dict[str, Any], assemble: bool) -> list[str]:
        """PURE: `step_name` of every pipeline step the fused pipeline performs."""
        names = ["Stereo Pair Creation"]
        if settings.get("apply_distortion", True):
            names.append("Fisheye Distortion")
        names.append("Crop Frames")
        if assemble:
            names.append("VR Assembly")
        return names

    @staticmethod
    def output_directories(
        directories: dict[str, Path], settings: dict[str, Any], assemble: bool
    ) -> dict[str, str]:
        """
        PURE: Directories a batch writes to.

        Intermediates are written only if they are kept. Without assembly (AI upscaling
        runs next) the cropped eyes are the result and always written.
        """
        keys: list[str] = []
        if settings.get("keep_intermediates", True):
            keys += ["left_frames", "right_frames", "left_cropped", "right_cropped"]
            if settings.get("apply_distortion", True):
                keys += ["left_distorted", "right_distorted"]
        if assemble:
            keys.append("vr_frames")
        else:
            keys += ["left_cropped", "right_cropped"]
        return {key: str(directories[key]) for key in keys if key in directories}

    def process_frames(
        self,
        frame_files: list[Path],
        depth_files: list[Path],
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker=None,
        assemble: bool = True,
    ) -> bool:
        """
        Run stereo generation, distortion, cropping and (optionally) VR assembly for a batch.

        Args:
            frame_files: Frame files of the batch
            depth_files: Depth map files of the same frames
            directories: Dictionary of processing directories
            settings: Processing settings
            progress_tracker: Optional progress tracker
            assemble: Also assemble the VR frames (False when AI upscaling comes first)

        Returns:
            True if successful, False otherwise

        Side effects:
            - Parallel processing via multiprocessing
            - Writes VR frames (or cropped eyes) and, if kept, the intermediate images
        """
        try:
            outputs = self.output_directories(directories, settings, assemble)
            tasks: list[FrameTask] = [
                (str(frame), str(depth), frame.stem, outputs, settings, assemble)
                for frame, depth in zip(frame_files, depth_files)
            ]
            if not tasks or len(frame_files) != len(depth_files):
                print(f"Error: {len(frame_files)} frames but {len(depth_files)} depth maps")
                return False

            names = self.step_names(settings, assemble)
            total = len(tasks)

            def report(index: int) -> None:
                if not progress_tracker:
                    return
                for step_name in names:
                    progress_tracker.update_progress(
                        f"Processing frame {index + 1}/{total}",
                        phase="frame_processing",
                        frame_num=index + 1,
                        step_name=step_name,
                        step_progress=index + 1,
                        step_total=total,
                    )
                self._send_previews(progress_tracker, outputs, tasks[index][2], index, total)

            self.stereo_generator.map_in_pool(process_frame, tasks, report)
            return True

        except (InterruptedError, KeyboardInterrupt):
            self.stereo_generator.close()
            raise
        except Exception as e:
            print(f"Error processing frames: {e}")
            traceback.print_exc()
            return False

    @staticmethod
    def _send_previews(tracker, outputs: dict[str, str], name: str, index: int, total: int) -> None:
        """Send previews of the images this frame wrote (if the tracker shows previews)."""
        if not hasattr(tracker, "send_preview_frame"):
            return
        if index % PREVIEW_FRAME_SAMPLE_RATE == 0 or index == total - 1:
            if "left_frames" in outputs:
                left = Path(outputs["left_frames"]) / f"{name}.png"
                tracker.send_preview_frame(left, "stereo_left", index + 1)
        if "vr_frames" in outputs:
            # throttled by time in the tracker
            vr_path = Path(outputs["vr_frames"]) / f"{name}.png"
            tracker.send_preview_frame(vr_path, "vr_frame", index + 1)
