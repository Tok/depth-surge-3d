"""
Processing pipeline orchestrator.

Coordinates the complete video processing pipeline across all specialized processors.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core.constants import BATCH_SIZE_RANGE, DEFAULT_SETTINGS, DEPTH_STATE_FILE
from ...io.operations import (
    cleanup_intermediate_files,
    get_available_space,
    create_output_directories,
    find_settings_file,
    remove_intermediate_directories,
    save_processing_settings,
    update_processing_status,
)
from ...utils.path_utils import calculate_frame_range, format_file_size, generate_output_filename
from ...utils.domain.disk_estimate import estimate_disk_usage
from ...utils.domain.resolution import fisheye_cut_fraction
from ..frames.distortion_processor import DistortionProcessor
from ...utils.domain.progress import BatchOffsetProgress, NullProgressTracker
from .pipeline_state import PipelineState, step_fingerprints
from ...utils import (
    step_complete,
    saved_to,
    title_bar,
    completion_banner,
    warning,
)


# `step_name` each step reports to the progress tracker
_STEP_NAMES = {
    "depth": "Depth Map Generation",
    "stereo": "Stereo Pair Creation",
    "distortion": "Fisheye Distortion",
    "crop": "Crop Frames",
    "upscale": "AI Upscaling",
    "assemble": "VR Assembly",
    "encode": "Video Creation",
}


def _segment_name(start: int, end: int) -> str:
    """PURE: File name of the video segment for frames [start, end)."""
    return f"segment_{start + 1:06d}-{end:06d}.mp4"


_FRAME_STEP_NAMES = [key for key in _STEP_NAMES if key != "depth"]


@dataclass
class _FrameStep:
    """A pipeline step that turns a range of frames into its output files."""

    key: str
    # (first frame index, end index, progress tracker) -> success
    run: Callable[[int, int, Any], bool]
    # directories whose files for those frames only this step reads
    consumes: tuple[str, ...]
    # directories this step writes to (created when the step runs on its own)
    writes: tuple[str, ...] = ()


class ProcessingOrchestrator:
    """
    High-level pipeline control and step sequencing.

    Responsibilities:
    - Pipeline execution flow
    - Step sequencing
    - Progress tracking coordination
    - Error handling
    - Settings management
    - Console output formatting
    """

    def __init__(
        self,
        depth_processor,
        stereo_generator,
        distortion_processor,
        upscaler,
        vr_assembler,
        video_encoder,
        verbose: bool = False,
        frame_pipeline=None,
    ):
        """
        Initialize processing orchestrator.

        Args:
            depth_processor: DepthMapProcessor instance
            stereo_generator: StereoPairGenerator instance
            distortion_processor: DistortionProcessor instance
            upscaler: FrameUpscalerProcessor instance
            vr_assembler: VRFrameAssembler instance
            video_encoder: VideoEncoder instance
            verbose: Enable verbose output
            frame_pipeline: FusedFrameProcessor instance (optional). Without it every
                step runs separately, writing its frames to disk.
        """
        self.frame_pipeline = frame_pipeline
        self.depth_processor = depth_processor
        self.stereo_generator = stereo_generator
        self.distortion_processor = distortion_processor
        self.upscaler = upscaler
        self.vr_assembler = vr_assembler
        self.video_encoder = video_encoder
        self.verbose = verbose
        self._settings_file: Path | None = None  # Track settings file for error handling
        self._total_steps = 7  # Updated dynamically based on settings
        self._start_time: float = 0.0  # Track processing start time

    def process(
        self,
        video_path: Path,
        output_dir: Path,
        video_properties: dict[str, Any],
        settings: dict[str, Any],
        progress_tracker=None,
    ) -> bool:
        """
        Main processing pipeline entry point.

        Continues an earlier run in ``output_dir`` where it stopped (see PipelineState).

        Args:
            video_path: Input video path
            output_dir: Output directory
            video_properties: Video metadata (frame_count, fps, etc.)
            settings: Processing settings
            progress_tracker: Optional progress tracker

        Returns:
            True if successful, False otherwise

        Raises:
            InterruptedError, KeyboardInterrupt: When the run is stopped; the batch is
                marked as paused and can be resumed

        Side effects:
            - Console output
            - Filesystem state management
            - Delegates to all processor modules
        """
        # CLI runs have no tracker; stages call it unconditionally (#14, #17)
        if progress_tracker is None:
            progress_tracker = NullProgressTracker()

        try:
            # Start timer
            self._start_time = time.time()

            # Setup processing environment
            output_path, directories, self._settings_file = self._setup_processing(
                str(video_path), str(output_dir), settings, video_properties
            )

            # Execute processing pipeline
            return self._execute_pipeline(
                str(video_path),
                output_path,
                directories,
                video_properties,
                settings,
                progress_tracker,
            )

        except (InterruptedError, KeyboardInterrupt):
            if self._settings_file:
                update_processing_status(self._settings_file, "paused")
            raise
        except Exception as e:
            print(f"Error in video processing: {e}")
            if self.verbose:
                traceback.print_exc()
            if self._settings_file:
                update_processing_status(self._settings_file, "failed", {"error": str(e)})
            return False
        finally:
            self.stereo_generator.close()
            self.upscaler.close()

    def _execute_pipeline(
        self,
        video_path: str,
        output_path: Path,
        directories: dict[str, Path],
        video_properties: dict[str, Any],
        settings: dict[str, Any],
        progress_tracker=None,
    ) -> bool:
        """
        Execute the pipeline, skipping what an earlier run in this directory finished.

        Args:
            video_path: Input video path
            output_path: Output directory path
            directories: Dictionary of processing directories
            video_properties: Video metadata (frame_count, fps, etc.)
            settings: Processing settings
            progress_tracker: Optional progress tracker

        Returns:
            True if all steps successful, False otherwise

        Side effects:
            - Executes all pipeline steps
            - Reads/writes the resume state
            - Progress updates, console output
        """
        self._total_steps = self._get_total_steps(settings)
        depth_state_path = output_path / DEPTH_STATE_FILE
        self._report_disk_estimate(output_path, video_properties, settings)
        self._warn_about_cut_off_sides(video_properties, settings)

        # What can be kept from an earlier run in this directory?
        state = PipelineState.load(output_path)
        restart = state.reconcile(
            step_fingerprints(settings, self._source_identity(video_path), self._model_identity())
        )
        if restart in ("extract", "depth"):
            depth_state_path.unlink(missing_ok=True)
        state.save()

        # Step 1: Extract frames (delegated to video_encoder)
        frame_files = self._extract_or_reuse_frames(
            video_path, directories, video_properties, settings, state, progress_tracker
        )
        if not frame_files:
            return self._handle_step_error("Frame extraction failed")
        num_frames = len(frame_files)

        # Steps 2-7: depth maps to VR frames
        depth_done = self._prepare_depth(frame_files, directories, settings, depth_state_path)
        self._report_restored_progress(state, depth_done, num_frames, progress_tracker)
        self._run_frame_steps(
            frame_files, directories, settings, state, depth_state_path, progress_tracker
        )
        # Step 8: Create final video (delegated to video_encoder)
        if self._uses_segments(settings):
            # The video was encoded batch by batch; join the segments and add the audio
            segments_dir = directories["video_segments"]
            progress_tracker.update_progress(
                "Joining video segments and adding audio",
                phase="video_creation",
                step_name="Video Creation",
            )
            success = self.video_encoder.concat_segments(
                [segments_dir / _segment_name(start, end) for start, end in state.segments],
                directories["base"],
                video_path,
                settings,
                num_frames,
            )
            progress_tracker.update_progress(
                "Created final video",
                phase="video_creation",
                step_name="Video Creation",
                step_progress=num_frames,
                step_total=num_frames,
            )
        else:
            success = self.video_encoder.create_video(
                directories["vr_frames"],
                directories["base"],
                video_path,
                settings,
                progress_tracker,
                max_frames=num_frames,
            )

        if success:
            output_filename = generate_output_filename(
                Path(video_path).name,
                settings["vr_format"],
                settings["vr_resolution"],
            )
            print(step_complete("Created final video"))
            self._print_saved_to(directories["base"], f"Final output: {output_filename}")

        # Finalize and cleanup
        if progress_tracker and hasattr(progress_tracker, "finish"):
            progress_tracker.finish("Video processing complete")
        if success and not settings.get("keep_intermediates", True):
            removed = remove_intermediate_directories(directories["base"])
            depth_state_path.unlink(missing_ok=True)
            state.path.unlink(missing_ok=True)
            if removed:
                print(saved_to(f"Removed {removed} intermediate directories"))
        self._finalize_processing(success, output_path, video_path, settings, num_frames)
        return success

    @staticmethod
    def _warn_about_cut_off_sides(
        video_properties: dict[str, Any], settings: dict[str, Any]
    ) -> None:
        """
        Say how much of a video that isn't square the fisheye view shows.

        Side effects:
            - Console output
        """
        if not settings.get("apply_distortion", True):
            return
        width = int(video_properties.get("width") or 0)
        height = int(video_properties.get("height") or 0)
        fit = DistortionProcessor._fisheye_frame_fit(settings)
        cut = fisheye_cut_fraction(width, height, fit)
        if cut >= 0.1:
            side = "width" if width >= height else "height"
            print(
                saved_to(
                    f"Fisheye view: about {cut:.0%} of the video's {side} is outside the square "
                    f"view (frame fit {fit:g}; 1 shows the whole frame with black bars)"
                )
            )

    @staticmethod
    def _report_disk_estimate(
        output_path: Path, video_properties: dict[str, Any], settings: dict[str, Any]
    ) -> None:
        """
        Print the estimated disk space of the run and warn if the disk looks too small.

        Side effects:
            - Reads free disk space, console output
        """
        try:
            start_frame, end_frame = calculate_frame_range(
                video_properties["frame_count"],
                video_properties["fps"],
                settings.get("start_time"),
                settings.get("end_time"),
            )
            estimate = estimate_disk_usage(
                end_frame - start_frame,
                int(video_properties.get("width") or settings.get("source_width") or 0),
                int(video_properties.get("height") or settings.get("source_height") or 0),
                settings,
            )["total"]
        except (KeyError, TypeError, ValueError):
            return
        free = get_available_space(output_path)
        print(f"Estimated disk space: {format_file_size(estimate)} ({format_file_size(free)} free)")
        if 0 < free < estimate:
            hint = (
                ""
                if not settings.get("keep_intermediates", True)
                else (" Without keeping intermediate files far less is needed.")
            )
            print(warning(f"Warning: this may not fit on the disk.{hint}"))

    def _extract_or_reuse_frames(
        self,
        video_path: str,
        directories: dict[str, Path],
        video_properties: dict[str, Any],
        settings: dict[str, Any],
        state: PipelineState,
        progress_tracker,
    ) -> list[Path]:
        """
        Step 1: extract the frames, unless an earlier run already did.

        Returns:
            Frame files in order (empty if extraction failed)

        Side effects:
            - May run FFmpeg and clear intermediates of an earlier, different extraction
            - Updates the resume state
        """
        # Frames are numbered from 1; without intermediates the ones every step is done
        # with have been deleted
        frame_files = [
            directories["frames"] / f"frame_{index + 1:06d}.png"
            for index in range(state.total_frames)
        ]
        if (
            state.extracted
            and state.total_frames > 0
            and all(frame.exists() for frame in frame_files[state.frames_pruned :])
        ):
            print(step_complete(f"Step 1: Using {len(frame_files)} frames extracted earlier"))
            progress_tracker.update_progress(
                "Frames already extracted",
                phase="resume",
                step_name="Frame Extraction",
                step_progress=len(frame_files),
                step_total=len(frame_files),
            )
            return frame_files

        # New extraction: nothing from an earlier run in this directory is valid any more
        cleanup_intermediate_files(directories["base"])
        (directories["base"] / DEPTH_STATE_FILE).unlink(missing_ok=True)
        state.done = {}
        state.segments = []
        state.pruned = False
        state.frames_pruned = 0
        state.extracted = False
        state.save()

        frame_files = self.video_encoder.extract_frames(
            video_path, directories, video_properties, settings, progress_tracker
        )
        if frame_files:
            state.extracted = True
            state.total_frames = len(frame_files)
            state.save()
            print(step_complete(f"Step 1: Extracted {len(frame_files)} frames"))
            self._print_saved_to(directories.get("frames"), "Extracted frames")
        return frame_files

    def _prepare_depth(
        self,
        frame_files: list[Path],
        directories: dict[str, Path],
        settings: dict[str, Any],
        depth_state_path: Path,
    ) -> int:
        """
        Find out how many depth maps an earlier run in this directory finished.

        Returns:
            Number of finished depth maps

        Side effects:
            - Reads the depth stream state
        """
        depth_done = min(
            self.depth_processor.finished_depth_maps(depth_state_path), len(frame_files)
        )
        return depth_done

    def _report_restored_progress(
        self, state: PipelineState, depth_done: int, num_frames: int, progress_tracker
    ) -> None:
        """
        Tell the progress tracker what an earlier run already finished.

        Side effects:
            - Progress updates, console output
        """
        restored = {"depth": depth_done}
        restored.update({key: state.frames_done(key) for key in _FRAME_STEP_NAMES})
        for key, done in restored.items():
            if done > 0:
                progress_tracker.update_progress(
                    "Resuming",
                    phase="resume",
                    step_name=_STEP_NAMES[key],
                    step_progress=min(done, num_frames),
                    step_total=num_frames,
                )
        finished = state.frames_done("assemble")
        if finished > 0 or depth_done > 0:
            print(
                saved_to(
                    f"Resuming: {finished}/{num_frames} VR frames and "
                    f"{depth_done}/{num_frames} depth maps are already done"
                )
            )

    def _frame_steps(
        self,
        frame_files: list[Path],
        directories: dict[str, Path],
        settings: dict[str, Any],
    ) -> list[_FrameStep]:
        """
        Steps 3-7 as functions of a frame range, with the directories each one consumes.

        Returns:
            Active steps in pipeline order
        """
        names = [f.stem for f in frame_files]
        distort = bool(settings.get("apply_distortion", True))
        upscale = settings.get("upscale_model", "none") != "none"

        def stereo(start: int, end: int, tracker) -> bool:
            depth_files = [directories["depth_maps"] / f"{n}.png" for n in names[start:end]]
            return self.stereo_generator.create_stereo_pairs(
                frame_files[start:end], depth_files, directories, settings, tracker
            )

        def distortion(start: int, end: int, tracker) -> bool:
            left = [directories["left_frames"] / f"{n}.png" for n in names[start:end]]
            right = [directories["right_frames"] / f"{n}.png" for n in names[start:end]]
            return self.distortion_processor.apply_distortion(
                left, right, directories, settings, tracker
            )

        def crop(start: int, end: int, tracker) -> bool:
            return self.distortion_processor.crop_frames(
                directories, settings, tracker, end - start, frame_names=names[start:end]
            )

        def upscaling(start: int, end: int, tracker) -> bool:
            return self.upscaler.apply_upscaling(
                directories, settings, tracker, frame_names=names[start:end]
            )

        def assemble(start: int, end: int, tracker) -> bool:
            return self.vr_assembler.assemble_vr_frames(
                directories, settings, tracker, end - start, frame_names=names[start:end]
            )

        stereo_dirs = ("left_frames", "right_frames")
        distorted_dirs = ("left_distorted", "right_distorted")
        cropped_dirs = ("left_cropped", "right_cropped")
        steps = [_FrameStep("stereo", stereo, ("depth_maps",), stereo_dirs)]
        if distort:
            steps.append(_FrameStep("distortion", distortion, stereo_dirs, distorted_dirs))
        steps.append(
            _FrameStep("crop", crop, distorted_dirs if distort else stereo_dirs, cropped_dirs)
        )
        if upscale:
            steps.append(
                _FrameStep("upscale", upscaling, cropped_dirs, ("left_upscaled", "right_upscaled"))
            )
        steps.append(
            _FrameStep(
                "assemble",
                assemble,
                ("left_upscaled", "right_upscaled") if upscale else cropped_dirs,
            )
        )
        return steps

    def _run_frame_steps(
        self,
        frame_files: list[Path],
        directories: dict[str, Path],
        settings: dict[str, Any],
        state: PipelineState,
        depth_state_path: Path,
        progress_tracker,
    ) -> None:
        """
        Steps 2-7 (and the video segments of step 8) for all frames.

        See ``_FrameRun``.

        Raises:
            RuntimeError: If a step fails

        Side effects:
            - Runs the depth model and all frame steps
            - Updates the resume state
        """
        _FrameRun(
            self, frame_files, directories, settings, state, depth_state_path, progress_tracker
        ).run()

    def _needed_directories(self, settings: dict[str, Any]) -> set[str]:
        """
        Directories a run with these settings writes to (others would stay empty).

        The fused frame pipeline writes stereo, distorted and cropped images only when
        intermediates are kept; cropped and upscaled eyes are needed for AI upscaling.
        """
        needed = {"frames", "depth_maps", "vr_frames"}
        if self._uses_segments(settings):
            needed.add("video_segments")
        if settings.get("keep_intermediates", True):
            needed |= {"left_frames", "right_frames", "left_cropped", "right_cropped"}
            if settings.get("apply_distortion", True):
                needed |= {"left_distorted", "right_distorted"}
        if settings.get("upscale_model", "none") != "none":
            needed |= {"left_cropped", "right_cropped", "left_upscaled", "right_upscaled"}
        return needed

    def _uses_segments(self, settings: dict[str, Any]) -> bool:
        """
        Whether the video is encoded in segments, batch by batch.

        Motion interpolation needs neighbouring frames across batch borders, so it is
        encoded in one pass at the end instead.
        """
        return hasattr(self.video_encoder, "encode_segment") and not settings.get(
            "experimental_frame_interpolation"
        )

    @staticmethod
    def _remove_frame_files(
        directories: dict[str, Path], dir_keys: tuple[str, ...], frame_files: list[Path]
    ) -> None:
        """
        Delete the intermediate files of frames a step has consumed.

        Side effects:
            - Deletes files
        """
        for key in dir_keys:
            directory = directories.get(key)
            if directory is None:
                continue
            for frame_file in frame_files:
                (directory / f"{frame_file.stem}.png").unlink(missing_ok=True)

    @staticmethod
    def _batch_size(settings: dict[str, Any]) -> int:
        """PURE: Frames per batch, clamped to the supported range."""
        default = int(DEFAULT_SETTINGS["batch_size"])  # type: ignore[call-overload]
        try:
            size = int(settings.get("batch_size") or default)
        except (TypeError, ValueError):
            size = default
        return max(BATCH_SIZE_RANGE[0], min(BATCH_SIZE_RANGE[1], size))

    def _model_identity(self) -> str:
        """Identify the depth model (a different model invalidates existing depth maps)."""
        estimator = getattr(self.depth_processor, "depth_estimator", None)
        parts = [type(estimator).__name__]
        parts.extend(
            str(getattr(estimator, name, "")) for name in ("model_path", "model_name", "metric")
        )
        return "|".join(parts)

    @staticmethod
    def _source_identity(video_path: str) -> str:
        """
        Identify the source video by name and size.

        Side effects:
            - Reads file metadata
        """
        try:
            return f"{Path(video_path).name}|{Path(video_path).stat().st_size}"
        except OSError:
            return Path(video_path).name

    def _setup_processing(
        self,
        video_path: str,
        output_dir: str,
        settings: dict[str, Any],
        video_properties: dict[str, Any],
    ) -> tuple[Path, dict[str, Path], Path | None]:
        """
        Setup processing directories and settings file.

        Args:
            video_path: Input video path
            output_dir: Output directory
            settings: Processing settings
            video_properties: Video metadata

        Returns:
            Tuple of (output_path, directories, settings_file)

        Side effects:
            - Creates directories
            - Writes settings file
            - Console output
        """
        output_path = Path(output_dir)
        # All pipeline directories get a name, but only the ones this run writes to are created
        directories = create_output_directories(
            output_path, False, needed=self._needed_directories(settings)
        )
        # A resumed run keeps the batch's settings file instead of adding another one
        existing = find_settings_file(output_path)
        if existing is not None:
            batch_name = existing.name[: -len("-settings.json")]
        else:
            batch_name = f"{Path(video_path).stem}_{int(time.time())}"
        settings_file = save_processing_settings(
            output_path, batch_name, settings, video_properties, video_path
        )

        print(f"\n{title_bar('=== Depth Surge 3D Video Processing ===')}")
        print(f"Input: {video_path}")
        print(f"Output: {output_path}")
        mode = settings.get("processing_mode", "batched")
        print(f"Mode: {mode} (batches of {self._batch_size(settings)} frames)\n")

        return output_path, directories, settings_file

    def _finalize_processing(
        self,
        success: bool,
        output_path: Path,
        video_path: str,
        settings: dict[str, Any],
        num_frames: int,
    ) -> None:
        """
        Finalize processing and update settings file.

        Args:
            success: Whether processing succeeded
            output_path: Output directory path
            video_path: Input video path
            settings: Processing settings
            num_frames: Number of frames processed

        Side effects:
            - Updates settings file with completion status
            - Console output
        """
        if success:
            # Calculate processing time
            elapsed_time = time.time() - self._start_time
            formatted_time = self._format_processing_time(elapsed_time)

            # Generate output filename
            output_filename = generate_output_filename(
                Path(video_path).name,
                settings["vr_format"],
                settings["vr_resolution"],
            )
            output_file_path = str(output_path / output_filename)

            # Display colored completion banner
            completion_banner(
                output_file=output_file_path,
                processing_time=formatted_time,
                num_frames=num_frames,
                vr_format=settings["vr_format"],
            )

            # Update settings file
            if self._settings_file:
                update_processing_status(
                    self._settings_file,
                    "completed",
                    {
                        "final_output": output_file_path,
                        "frames_processed": num_frames,
                        "processing_time_seconds": elapsed_time,
                    },
                )
        elif self._settings_file:
            update_processing_status(
                self._settings_file, "failed", {"error": "Video creation failed"}
            )

    @staticmethod
    def _get_total_steps(settings: dict[str, Any]) -> int:
        """
        PURE: Calculate total steps based on settings.

        Args:
            settings: Processing settings

        Returns:
            Total number of pipeline steps (6-8)
        """
        total = 6  # Base: Extract, Depth, Stereo, Crop, Assemble, Video
        if settings.get("apply_distortion", True):
            total += 1  # Add Step 4: Distortion
        if settings.get("upscale_model", "none") != "none":
            total += 1  # Add Step 6: Upscaling
        return total  # 6-8 steps total

    @staticmethod
    def _format_processing_time(seconds: float) -> str:
        """
        PURE: Format processing time as human-readable string.

        Args:
            seconds: Processing time in seconds

        Returns:
            Formatted time string (e.g., "1h 23m 45s", "5m 30s", "45s")
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:  # Always show seconds if no other parts
            parts.append(f"{secs}s")

        return " ".join(parts)

    def _update_step_progress(
        self,
        progress_tracker,
        message: str,
        step_name: str,
        progress: int,
        total: int,
    ) -> None:
        """
        Update progress for a processing step.

        Args:
            progress_tracker: Progress tracker instance
            message: Progress message
            step_name: Name of current step
            progress: Current progress value
            total: Total progress value

        Side effects:
            - Updates progress tracker state
        """
        if progress_tracker:
            progress_tracker.update_progress(
                message,
                phase="processing",
                frame_num=progress,
                step_name=step_name,
                step_progress=progress,
                step_total=total,
            )

    def _print_step_complete(
        self, num_items: int, duration: float, item_type: str = "frames"
    ) -> None:
        """
        Print step completion message.

        Args:
            num_items: Number of items processed
            duration: Processing duration in seconds
            item_type: Type of items processed

        Side effects:
            - Console output
        """
        print(step_complete(f"Processed {num_items:04d} {item_type} in {duration:.2f}s"))

    def _print_saved_to(self, directory: Path | None, message_prefix: str = "Saved to") -> None:
        """
        Print save location message.

        Args:
            directory: Directory path or None
            message_prefix: Message prefix text

        Side effects:
            - Console output
        """
        if directory:
            print(saved_to(f"{message_prefix}: {directory}"))

    def _handle_step_error(self, error_msg: str) -> bool:
        """
        Handle step failure and update settings file.

        Args:
            error_msg: Error message

        Returns:
            False (always returns False to indicate failure)

        Side effects:
            - Console output
            - Updates settings file with error status
        """
        print(f"Error: {error_msg}")
        if self._settings_file:
            update_processing_status(self._settings_file, "failed", {"error": error_msg})
        return False


class _FrameRun:
    """
    One run of steps 2-7 over all frames of a video.

    Batched mode runs every step for one batch of frames before starting the next batch,
    so finished VR frames (and video segments) appear early. Sequential mode finishes the
    depth maps for the whole video first. Both work through the video in batches and
    save their progress after each one.

    Stereo generation, distortion, cropping and VR assembly normally run fused: one
    worker per frame, images passed in memory. The separate step processors are used
    when those steps aren't equally far along (after a settings change that only
    invalidated some of them) and for AI upscaling.

    Without intermediates, files are deleted as soon as the last step that reads them
    has finished the frames: disk use stays at about one batch plus the video.
    """

    def __init__(
        self,
        orchestrator: ProcessingOrchestrator,
        frame_files: list[Path],
        directories: dict[str, Path],
        settings: dict[str, Any],
        state: PipelineState,
        depth_state_path: Path,
        progress_tracker: Any,
    ):
        self.orchestrator = orchestrator
        self.frame_files = frame_files
        self.directories = directories
        self.settings = settings
        self.state = state
        self.depth_state_path = depth_state_path
        self.tracker = progress_tracker

        self.num_frames = len(frame_files)
        self.batch_size = orchestrator._batch_size(settings)
        self.prune = not settings.get("keep_intermediates", True)
        self._segmented = orchestrator._uses_segments(settings)
        self.upscale = settings.get("upscale_model", "none") != "none"
        self.steps = orchestrator._frame_steps(frame_files, directories, settings)
        # Steps the fused pipeline covers (assembly only if no upscaling comes before it)
        self.fused_keys = [
            step.key for step in self.steps if step.key in ("stereo", "distortion", "crop")
        ] + ([] if self.upscale else ["assemble"])
        self.depth_stream = orchestrator.depth_processor.stream_depth_maps(
            frame_files, directories["depth_maps"], depth_state_path, settings, progress_tracker
        )
        self.depth_done = min(
            orchestrator.depth_processor.finished_depth_maps(depth_state_path), self.num_frames
        )

    def run(self) -> None:
        """
        Run all frame steps.

        Raises:
            RuntimeError: If a step fails

        Side effects:
            - Runs the depth model and all frame steps, updates the resume state
        """
        num_frames = self.num_frames
        if self.settings.get("processing_mode", "batched") == "sequential":
            self.depth_until(num_frames)
            print(step_complete(f"Step 2: {num_frames} depth maps"))
            self.orchestrator._print_saved_to(self.directories.get("depth_maps"), "Depth maps")
            self.advance_frames(num_frames)
            print(step_complete(f"Steps 3-7: {num_frames} VR frames"))
            return

        batch_count = (num_frames + self.batch_size - 1) // self.batch_size
        first = self._frames_finished() // self.batch_size
        for batch in range(first, batch_count):
            start = batch * self.batch_size
            end = min(start + self.batch_size, num_frames)
            if hasattr(self.tracker, "set_batch_info"):
                self.tracker.set_batch_info(batch, batch_count, start, end - start)
            self.depth_until(end)
            self.advance_frames(end)
            done = "encoded into a video segment" if self._segmented else "finished"
            print(
                step_complete(
                    f"Batch {batch + 1}/{batch_count}: VR frames {start + 1}-{end} "
                    f"of {num_frames} {done}"
                )
            )
        if self._segmented:
            self.orchestrator._print_saved_to(
                self.directories.get("video_segments"), "Video segments"
            )
        else:
            self.orchestrator._print_saved_to(self.directories.get("vr_frames"), "VR frames")

    def _frames_finished(self) -> int:
        """Frames that are through every step (including their video segment, if used)."""
        key = "encode" if self.orchestrator._uses_segments(self.settings) else "assemble"
        return self.state.frames_done(key)

    def depth_until(self, target: int) -> None:
        """Run the depth stream until at least ``target`` depth maps are finished."""
        while self.depth_done < target:
            emitted = next(self.depth_stream, None)
            if emitted is None:
                raise RuntimeError("Depth map generation ended early")
            self.depth_done = emitted
            self.state.done["depth"] = emitted
            self.state.save()

    def advance_frames(self, target: int) -> None:
        """Bring steps 3-7, the video segments and the cleanup up to frame ``target``."""
        fused = self.orchestrator.frame_pipeline is not None
        aligned = len({self.state.frames_done(key) for key in self.fused_keys}) == 1
        if fused and aligned:
            self._advance_fused(target)
        for step in self.steps:
            # after the fused pipeline this only leaves upscaling and its assembly
            self._advance_step(step, target)
        if self.orchestrator._uses_segments(self.settings):
            self._encode_segments(target)
        if self.prune:
            self._prune_original_frames()

    def _advance_fused(self, target: int) -> None:
        """Stereo, distortion, crop (and assembly) for the frames up to ``target``, fused."""
        state, names = self.state, [f.stem for f in self.frame_files]
        start = state.frames_done(self.fused_keys[0])
        while start < target:
            end = min(start + self.batch_size, target)
            depth_files = [self.directories["depth_maps"] / f"{n}.png" for n in names[start:end]]
            if not self.orchestrator.frame_pipeline.process_frames(
                self.frame_files[start:end],
                depth_files,
                self.directories,
                self.settings,
                BatchOffsetProgress(self.tracker, start, self.num_frames),
                assemble=not self.upscale,
            ):
                raise RuntimeError("Frame processing failed")
            for key in self.fused_keys:
                state.done[key] = end
            if self.prune:
                self.orchestrator._remove_frame_files(
                    self.directories, ("depth_maps",), self.frame_files[start:end]
                )
                state.pruned = True
            state.save()
            start = end

    def _advance_step(self, step: _FrameStep, target: int) -> None:
        """One separate step for the frames up to ``target``."""
        state = self.state
        start = state.frames_done(step.key)
        if start < target:
            for key in step.writes:
                if key in self.directories:
                    self.directories[key].mkdir(exist_ok=True)
        while start < target:
            end = min(start + self.batch_size, target)
            tracker = BatchOffsetProgress(self.tracker, start, self.num_frames)
            if not step.run(start, end, tracker):
                raise RuntimeError(f"{_STEP_NAMES[step.key]} failed")
            state.done[step.key] = end
            if self.prune:
                self.orchestrator._remove_frame_files(
                    self.directories, step.consumes, self.frame_files[start:end]
                )
                state.pruned = True
            state.save()
            start = end

    def _encode_segments(self, target: int) -> None:
        """Encode the assembled VR frames up to ``target`` into video segments."""
        state = self.state
        start = state.frames_done("encode")
        while start < target:
            end = min(start + self.batch_size, target)

            def report(frames: int, offset: int = start) -> None:
                self.tracker.update_progress(
                    f"Encoding video {offset + frames}/{self.num_frames}",
                    phase="video_creation",
                    step_name="Video Creation",
                    step_progress=min(offset + frames, end),
                    step_total=self.num_frames,
                )

            segment = self.directories["video_segments"] / _segment_name(start, end)
            if not self.orchestrator.video_encoder.encode_segment(
                self.directories["vr_frames"], segment, start, end - start, self.settings, report
            ):
                raise RuntimeError("Video encoding failed")
            report(end - start)
            state.segments = [s for s in state.segments if s[1] <= start] + [[start, end]]
            state.done["encode"] = end
            if self.prune:
                self.orchestrator._remove_frame_files(
                    self.directories, ("vr_frames",), self.frame_files[start:end]
                )
                state.pruned = True
            state.save()
            start = end

    def _prune_original_frames(self) -> None:
        """Delete original frames that neither depth estimation nor stereo needs any more."""
        state = self.state
        upto = min(self.depth_done, state.frames_done("stereo"))
        if upto > state.frames_pruned:
            self.orchestrator._remove_frame_files(
                self.directories, ("frames",), self.frame_files[state.frames_pruned : upto]
            )
            state.frames_pruned = upto
            state.save()
