"""
Progress tracking utilities for processing operations.

This module provides progress tracking classes and utilities with minimal
side effects and clear separation of concerns.
"""

import time
from typing import Optional, Callable
from abc import ABC, abstractmethod

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from ..core.constants import (
    PROGRESS_UPDATE_INTERVAL,
    PROCESSING_STEPS,
    PROGRESS_DECIMAL_PLACES,
)

# Processing phases with weight distribution
PROCESSING_PHASES = {
    "extraction": {"weight": 15, "description": "Extracting frames"},
    "super_sampling": {"weight": 10, "description": "Super sampling frames"},
    "depth_estimation": {"weight": 30, "description": "Estimating depth"},
    "stereo_generation": {"weight": 25, "description": "Creating stereo pairs"},
    "distortion": {"weight": 10, "description": "Applying fisheye distortion"},
    "vr_assembly": {"weight": 5, "description": "Assembling VR frames"},
    "video_creation": {"weight": 5, "description": "Creating final video"},
}


class ProgressReporter(ABC):
    """Abstract base class for progress reporting."""

    @abstractmethod
    def report_progress(self, current: int, total: int, message: str = "") -> None:
        """Report progress update."""
        pass

    @abstractmethod
    def report_completion(self, message: str = "") -> None:
        """Report completion."""
        pass


class ConsoleProgressReporter(ProgressReporter):
    """Console-based progress reporter with tqdm support."""

    def __init__(self, show_eta: bool = True, use_tqdm: bool = True):
        self.show_eta = show_eta
        self.use_tqdm = use_tqdm and HAS_TQDM
        self.start_time = time.time()
        self.last_update_time = 0
        self.current_pbar = None
        self.current_phase = None

    def _get_phase_description(self, phase: str) -> str:
        """Get a user-friendly description for a phase."""
        return PROCESSING_PHASES.get(phase, {}).get(
            "description", phase.replace("_", " ").title()
        )

    def report_progress(self, current: int, total: int, message: str = "") -> None:
        """Report progress to console with optional tqdm support."""
        current_time = time.time()

        # Throttle updates for non-tqdm mode
        if (
            not self.use_tqdm
            and current_time - self.last_update_time < PROGRESS_UPDATE_INTERVAL
        ):
            return

        self.last_update_time = current_time

        if self.use_tqdm and total > 0:
            # Use tqdm for better progress display
            if self.current_pbar is None:
                self.current_pbar = tqdm(
                    total=total, desc=message, unit="frame", position=0, leave=True
                )

            # Update progress bar
            self.current_pbar.n = current
            self.current_pbar.refresh()

            if current >= total:
                self.current_pbar.close()
                self.current_pbar = None
        else:
            # Fallback to simple console output
            if total > 0:
                percentage = (current / total) * 100
                elapsed = current_time - self.start_time

                if self.show_eta and current > 0:
                    eta = (elapsed / current) * (total - current)
                    eta_str = f" - ETA: {eta:.1f}s"
                else:
                    eta_str = ""

                progress_line = (
                    f"\r{message} {percentage:.1f}% ({current}/{total}){eta_str}"
                )
                print(progress_line, end="", flush=True)

    def report_completion(self, message: str = "") -> None:
        """Report completion to console."""
        if self.current_pbar:
            self.current_pbar.close()
            self.current_pbar = None

        elapsed = time.time() - self.start_time
        print(f"\r{message} - Completed in {elapsed:.1f}s")
        print()  # New line


class ProgressTracker:
    """Enhanced progress tracking for both serial and batch processing modes."""

    def __init__(
        self,
        total_frames: int,
        processing_mode: str = "serial",
        reporter: Optional[ProgressReporter] = None,
    ):
        self.total_frames = total_frames
        self.processing_mode = processing_mode
        self.reporter = reporter or ConsoleProgressReporter()
        self.start_time = time.time()

        self.steps = PROCESSING_STEPS.copy()
        self.current_step_index = 0
        self.step_progress = 0
        self.step_total = 0

    def update_serial(self, frame_num: int, step_description: str) -> None:
        """Update progress for serial processing."""
        self.current_frame = frame_num
        self.current_step = step_description

        message = f"[SERIAL] Frame {frame_num}/{self.total_frames} - {step_description}"
        self.reporter.report_progress(frame_num, self.total_frames, message)

    def update_batch_step(
        self, step_name: str, progress: int = 0, total: int = 0
    ) -> None:
        """Update progress for batch processing step."""
        if step_name in self.steps:
            self.current_step_index = self.steps.index(step_name)
        self.step_progress = progress
        self.step_total = total
        self._display_batch()

    def update_batch_progress(self, progress: int, total: Optional[int] = None) -> None:
        """Update progress within current batch step."""
        self.step_progress = progress
        if total is not None:
            self.step_total = total
        self._display_batch()

    def _display_batch(self) -> None:
        """Display batch mode progress."""
        step_name = (
            self.steps[self.current_step_index]
            if self.current_step_index < len(self.steps)
            else "Processing"
        )

        # Calculate overall progress
        step_progress_ratio = (
            (self.step_progress / max(self.step_total, 1)) if self.step_total > 0 else 0
        )
        overall_progress = (
            (self.current_step_index + step_progress_ratio) / len(self.steps)
        ) * 100

        if self.step_total > 0:
            step_percentage = (self.step_progress / self.step_total) * 100
            message = f"[BATCH] Step {self.current_step_index + 1}/{len(self.steps)}: {step_name} ({step_percentage:.1f}%)"
        else:
            message = f"[BATCH] Step {self.current_step_index + 1}/{len(self.steps)}: {step_name}"

        # Report as if total progress is 100 (for percentage calculation)
        self.reporter.report_progress(int(overall_progress), 100, message)

    def update_progress(
        self,
        stage: str,
        frame_num: Optional[int] = None,
        phase: Optional[str] = None,
        step_name: Optional[str] = None,
        step_progress: Optional[int] = None,
        step_total: Optional[int] = None,
    ) -> None:
        """Update progress for video processing."""
        if step_name:
            self.update_batch_step(step_name, step_progress or 0, step_total or 0)
        elif step_progress is not None:
            self.update_batch_progress(step_progress, step_total)

    def finish(self, message: str = "Processing complete") -> None:
        """Finish progress tracking."""
        self.reporter.report_completion(message)


class ProgressCallback:
    """Callback-based progress tracking for integration with external systems."""

    def __init__(
        self,
        session_id: str,
        total_frames: int,
        processing_mode: str = "serial",
        callback_func: Optional[Callable] = None,
    ):
        self.session_id = session_id
        self.total_frames = total_frames
        self.processing_mode = processing_mode
        self.callback_func = callback_func
        self.start_time = time.time()
        self.last_update_time = 0
        self.current_phase = "extraction"

        self.steps = PROCESSING_STEPS.copy()
        self.current_step_index = 0
        self.step_progress = 0
        self.step_total = 0

    def update_progress(
        self,
        stage: str,
        frame_num: Optional[int] = None,
        phase: Optional[str] = None,
        step_name: Optional[str] = None,
        step_progress: Optional[int] = None,
        step_total: Optional[int] = None,
    ) -> None:
        """Update progress with callback notification."""
        current_time = time.time()

        # Throttle updates
        if current_time - self.last_update_time < PROGRESS_UPDATE_INTERVAL:
            return

        self.last_update_time = current_time

        # Update phase if provided
        if phase:
            self.current_phase = phase

        # Calculate progress based on processing mode
        if self.processing_mode == "batch":
            progress = self._calculate_batch_progress(
                step_name, step_progress, step_total
            )
            progress_data = self._create_batch_progress_data(stage, step_name, progress)
        else:
            progress = self._calculate_serial_progress(frame_num)
            progress_data = self._create_serial_progress_data(
                stage, frame_num, progress
            )

        # Execute callback if provided
        if self.callback_func:
            self.callback_func(progress_data)

    def _calculate_batch_progress(
        self,
        step_name: Optional[str],
        step_progress: Optional[int],
        step_total: Optional[int],
    ) -> float:
        """Calculate batch mode progress percentage."""
        if step_name and step_name in self.steps:
            self.current_step_index = self.steps.index(step_name)
        if step_progress is not None:
            self.step_progress = step_progress
        if step_total is not None:
            self.step_total = step_total

        step_progress_ratio = (
            (self.step_progress / max(self.step_total, 1)) if self.step_total > 0 else 0
        )
        overall_progress = (
            (self.current_step_index + step_progress_ratio) / len(self.steps)
        ) * 100

        return round(overall_progress, PROGRESS_DECIMAL_PLACES)

    def _calculate_serial_progress(self, frame_num: Optional[int]) -> float:
        """Calculate serial mode progress percentage."""
        if frame_num is None:
            return 0.0

        # Phase-based progress calculation
        if self.current_phase == "extraction":
            return (frame_num / self.total_frames * 20) if self.total_frames > 0 else 0
        elif self.current_phase == "processing":
            frame_progress = (
                (frame_num / self.total_frames * 65) if self.total_frames > 0 else 0
            )
            return 20 + frame_progress
        elif self.current_phase == "video":
            return 85 + 15  # Set to 100% for video phase
        else:
            return 0.0

    def _create_batch_progress_data(
        self, stage: str, step_name: Optional[str], progress: float
    ) -> dict:
        """Create progress data dictionary for batch mode."""
        return {
            "session_id": self.session_id,
            "progress": progress,
            "stage": stage,
            "total_frames": self.total_frames,
            "phase": self.current_phase,
            "processing_mode": self.processing_mode,
            "step_name": (
                step_name or self.steps[self.current_step_index]
                if self.current_step_index < len(self.steps)
                else "Processing"
            ),
            "step_progress": self.step_progress,
            "step_total": self.step_total,
            "step_index": self.current_step_index,
            "total_steps": len(self.steps),
        }

    def _create_serial_progress_data(
        self, stage: str, frame_num: Optional[int], progress: float
    ) -> dict:
        """Create progress data dictionary for serial mode."""
        return {
            "session_id": self.session_id,
            "progress": progress,
            "stage": stage,
            "current_frame": frame_num or 0,
            "total_frames": self.total_frames,
            "phase": self.current_phase,
            "processing_mode": self.processing_mode,
        }


def create_progress_tracker(
    total_frames: int, processing_mode: str = "serial", reporter_type: str = "console"
) -> ProgressTracker:
    """
    Factory function to create progress tracker with specified reporter.

    Args:
        total_frames: Total number of frames to process
        processing_mode: 'serial' or 'batch'
        reporter_type: Type of reporter to use

    Returns:
        Configured ProgressTracker instance
    """
    if reporter_type == "console":
        reporter = ConsoleProgressReporter()
    else:
        reporter = ConsoleProgressReporter()  # Default fallback

    return ProgressTracker(total_frames, processing_mode, reporter)


def calculate_eta(start_time: float, current: int, total: int) -> float:
    """
    Calculate estimated time to completion.

    Args:
        start_time: Process start time
        current: Current progress count
        total: Total items to process

    Returns:
        ETA in seconds, or 0 if cannot calculate
    """
    if current <= 0 or total <= 0:
        return 0.0

    elapsed = time.time() - start_time
    rate = current / elapsed
    remaining = total - current

    return remaining / rate if rate > 0 else 0.0


def format_time_duration(seconds: float) -> str:
    """
    Format time duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
