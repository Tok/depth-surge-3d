"""
Video processor with temporal consistency.

This module implements video processing using Video-Depth-Anything
for temporal consistency across video frames.

REFACTORED: This is now a thin orchestrator that delegates to specialized
processor modules for improved maintainability and testability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..frames.depth_processor import DepthMapProcessor
from ..frames.stereo_generator import StereoPairGenerator
from ..frames.distortion_processor import DistortionProcessor
from ..frames.frame_upscaler import FrameUpscalerProcessor
from ..frames.vr_assembler import VRFrameAssembler
from ..video.video_encoder import VideoEncoder
from .pipeline_orchestrator import ProcessingOrchestrator


class VideoProcessor:
    """
    Handles video processing with temporal consistency.

    Uses Video-Depth-Anything model to process entire videos with
    temporal consistency for superior depth estimation quality.

    This is a thin orchestrator that delegates to specialized processors:
    - DepthMapProcessor: Depth map generation with caching
    - StereoPairGenerator: Stereo pair creation
    - DistortionProcessor: Fisheye distortion and cropping
    - FrameUpscalerProcessor: AI upscaling
    - VRFrameAssembler: VR frame assembly
    - VideoEncoder: Video encoding with FFmpeg
    - ProcessingOrchestrator: Pipeline coordination
    """

    def __init__(self, depth_estimator: Any, verbose: bool = False):
        """
        Initialize video processor with specialized modules.

        Args:
            depth_estimator: Depth estimation model instance (VideoDepthEstimator or VideoDepthEstimatorDA3)
            verbose: Enable verbose output
        """
        self.depth_estimator = depth_estimator
        self.verbose = verbose
        self._settings_file = None  # Backward compatibility for tests

        # Initialize specialized processor modules
        self.depth_processor = DepthMapProcessor(depth_estimator, verbose=verbose)
        self.stereo_generator = StereoPairGenerator(verbose=verbose)
        self.distortion_processor = DistortionProcessor(verbose=verbose)
        self.upscaler = FrameUpscalerProcessor(verbose=verbose)
        self.vr_assembler = VRFrameAssembler(verbose=verbose)
        self.video_encoder = VideoEncoder(verbose=verbose)

        # Initialize pipeline orchestrator with all processors
        self.orchestrator = ProcessingOrchestrator(
            depth_processor=self.depth_processor,
            stereo_generator=self.stereo_generator,
            distortion_processor=self.distortion_processor,
            upscaler=self.upscaler,
            vr_assembler=self.vr_assembler,
            video_encoder=self.video_encoder,
            verbose=verbose,
        )

    def process(
        self,
        video_path: str,
        output_dir: str,
        video_properties: dict[str, Any],
        settings: dict[str, Any],
        progress_callback=None,
    ) -> bool:
        """
        Process video in batch mode with temporal consistency.

        Args:
            video_path: Path to input video
            output_dir: Output directory path
            video_properties: Video metadata (frame_count, fps, etc.)
            settings: Processing settings
            progress_callback: Optional progress callback for web UI

        Returns:
            True if processing completed successfully

        Side effects:
            - Delegates to ProcessingOrchestrator which executes full pipeline
            - Creates output directory structure
            - Generates intermediate and final output files
        """
        return self.orchestrator.process(
            Path(video_path),
            Path(output_dir),
            video_properties,
            settings,
            progress_tracker=progress_callback,
        )

    # Backward compatibility methods for existing tests - delegate to specialized modules
    def _get_total_steps(self, settings):
        """Delegate to ProcessingOrchestrator."""
        return ProcessingOrchestrator._get_total_steps(settings)

    def _update_step_progress(self, progress_tracker, message, step_name, progress, total):
        """Delegate to orchestrator."""
        return self.orchestrator._update_step_progress(
            progress_tracker, message, step_name, progress, total
        )

    def _handle_step_error(self, error_msg):
        """Delegate to orchestrator."""
        return self.orchestrator._handle_step_error(error_msg)

    def _setup_processing(self, video_path, output_dir, settings, video_properties):
        """Delegate to orchestrator."""
        return self.orchestrator._setup_processing(
            video_path, output_dir, settings, video_properties
        )

    def _finalize_processing(self, success, output_path, video_path, settings, num_frames):
        """Delegate to orchestrator."""
        return self.orchestrator._finalize_processing(
            success, output_path, video_path, settings, num_frames
        )

    def _crop_frames(self, directories, settings, progress_tracker, total_frames):
        """Delegate to distortion processor."""
        return self.distortion_processor.crop_frames(
            directories, settings, progress_tracker, total_frames
        )

    def _apply_upscaling(self, left_dir, right_dir, directories, settings, progress_tracker):
        """Delegate to upscaler."""
        return self.upscaler.apply_upscaling(directories, settings, progress_tracker)

    def _process_upscaling_frames(
        self, upscaler, left_dir, right_dir, directories, settings, progress_tracker
    ):
        """Delegate to upscaler."""
        return self.upscaler._process_upscaling_frames(
            upscaler, left_dir, right_dir, directories, settings, progress_tracker
        )

    def _upscale_frame_pair(self, *args):
        """Delegate to upscaler."""
        return self.upscaler._upscale_frame_pair(*args)
