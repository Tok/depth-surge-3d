"""
Progress tracking utilities for processing operations.

This module provides progress tracking classes and utilities with minimal
side effects and clear separation of concerns.
"""

from __future__ import annotations

import time
from typing import Any
from collections.abc import Callable
from abc import ABC, abstractmethod

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from ...core.constants import (
    PROGRESS_UPDATE_INTERVAL,
    PROCESSING_STEPS,
    PROGRESS_DECIMAL_PLACES,
    PROGRESS_STEP_WEIGHTS,
    PROGRESS_EMIT_INTERVAL,
    PIPELINE_STEP_NAMES,
    PIPELINE_LOOP_STEP_NAMES,
)

# Processing phases with weight distribution
PROCESSING_PHASES = {
    "extraction": {"weight": 15, "description": "Extracting frames"},
    "super_sampling": {"weight": 10, "description": "Super sampling frames"},
    "depth_estimation": {"weight": 30, "description": "Estimating depth"},
    "stereo_generation": {"weight": 20, "description": "Creating stereo pairs"},
    "distortion": {"weight": 10, "description": "Applying fisheye distortion"},
    "upscaling": {"weight": 5, "description": "Upscaling frames"},
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
        self.last_update_time: float = 0.0
        self.current_pbar = None
        self.current_phase: str | None = None

    def _get_phase_description(self, phase: str) -> str:
        """Get a user-friendly description for a phase."""
        phase_info = PROCESSING_PHASES.get(phase, {})
        if isinstance(phase_info, dict):
            return str(phase_info.get("description", phase.replace("_", " ").title()))
        return phase.replace("_", " ").title()

    def report_progress(self, current: int, total: int, message: str = "") -> None:
        """Report progress to console with optional tqdm support."""
        current_time = time.time()

        # Throttle updates for non-tqdm mode
        if not self.use_tqdm and current_time - self.last_update_time < PROGRESS_UPDATE_INTERVAL:
            return

        self.last_update_time = current_time

        if self.use_tqdm and total > 0:
            # Use tqdm for better progress display
            if self.current_pbar is None:
                self.current_pbar = tqdm(
                    total=total, desc=message, unit="frame", position=0, leave=True
                )

            # Update progress bar
            self.current_pbar.n = current  # type: ignore[attr-defined]
            self.current_pbar.refresh()  # type: ignore[attr-defined]

            if current >= total:
                self.current_pbar.close()  # type: ignore[attr-defined]
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

                progress_line = f"\r{message} {percentage:.1f}% ({current}/{total}){eta_str}"
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
        reporter: ProgressReporter | None = None,
    ):
        self.total_frames = total_frames
        self.processing_mode = processing_mode
        self.reporter = reporter or ConsoleProgressReporter()
        self.start_time = time.time()
        self.step_start_time = time.time()

        self.steps = PROCESSING_STEPS.copy()
        self.current_step_index = 0
        self.step_progress = 0
        self.step_total = 0

        # ETA tracking
        self.last_progress = 0
        self.last_progress_time = time.time()

    def _calculate_eta(self, current_progress: int, total_progress: int) -> str | None:
        """
        Calculate estimated time remaining.

        Args:
            current_progress: Current progress value
            total_progress: Total progress value

        Returns:
            Formatted ETA string (e.g., "5m 23s") or None if not enough data
        """
        if current_progress <= 0 or total_progress <= 0:
            return None

        current_time = time.time()
        elapsed = current_time - self.start_time

        # Need at least 5 seconds of data for reasonable estimate
        if elapsed < 5:
            return None

        # Calculate time per unit
        progress_ratio = current_progress / total_progress
        if progress_ratio <= 0:
            return None

        estimated_total_time = elapsed / progress_ratio
        remaining_time = estimated_total_time - elapsed

        if remaining_time < 0:
            return None

        return self._format_time(remaining_time)

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time (e.g., '5m 23s' or '2h 15m')."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:  # Less than 1 hour
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:  # 1 hour or more
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def update_serial(self, frame_num: int, step_description: str) -> None:
        """Update progress for serial processing."""
        self.current_frame = frame_num
        self.current_step = step_description

        # Calculate ETA
        eta_str = self._calculate_eta(frame_num, self.total_frames)
        eta_suffix = f" (ETA: {eta_str})" if eta_str else ""

        message = f"[SERIAL] Frame {frame_num}/{self.total_frames} - {step_description}{eta_suffix}"
        self.reporter.report_progress(frame_num, self.total_frames, message)

    def update_batch_step(self, step_name: str, progress: int = 0, total: int = 0) -> None:
        """Update progress for batch processing step."""
        if step_name in self.steps:
            self.current_step_index = self.steps.index(step_name)
        self.step_progress = progress
        self.step_total = total
        self._display_batch()

    def update_batch_progress(self, progress: int, total: int | None = None) -> None:
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
        overall_progress = ((self.current_step_index + step_progress_ratio) / len(self.steps)) * 100

        # Calculate ETA based on overall progress
        eta_str = self._calculate_eta(int(overall_progress), 100)
        eta_suffix = f" - ETA: {eta_str}" if eta_str else ""

        if self.step_total > 0:
            step_percentage = (self.step_progress / self.step_total) * 100
            message = (
                f"[BATCH] Step {self.current_step_index + 1}/{len(self.steps)}: "
                f"{step_name} ({step_percentage:.1f}%){eta_suffix}"
            )
        else:
            message = (
                f"[BATCH] Step {self.current_step_index + 1}/{len(self.steps)}: "
                f"{step_name}{eta_suffix}"
            )

        # Report as if total progress is 100 (for percentage calculation)
        self.reporter.report_progress(int(overall_progress), 100, message)

    def update_progress(
        self,
        stage: str,
        frame_num: int | None = None,
        phase: str | None = None,
        step_name: str | None = None,
        step_progress: int | None = None,
        step_total: int | None = None,
    ) -> None:
        """Update progress for video processing."""
        if step_name:
            self.update_batch_step(step_name, step_progress or 0, step_total or 0)
        elif step_progress is not None:
            self.update_batch_progress(step_progress, step_total)

    def finish(self, message: str = "Processing complete") -> None:
        """Finish progress tracking."""
        self.reporter.report_completion(message)


class NullProgressTracker:
    """
    No-op progress tracker for runs without a UI (e.g. the CLI).

    Lets pipeline stages call update_progress()/finish() unconditionally. It has no
    send_preview_* methods, so stages that check hasattr() skip preview generation.
    """

    def update_progress(self, *args: Any, **kwargs: Any) -> None:
        """Ignore progress updates."""

    def finish(self, *args: Any, **kwargs: Any) -> None:
        """Ignore completion."""


class ProgressCallback:
    """Callback-based progress tracking for integration with external systems."""

    def __init__(
        self,
        session_id: str,
        total_frames: int,
        processing_mode: str = "serial",
        callback_func: Callable | None = None,
    ):
        self.session_id = session_id
        self.total_frames = total_frames
        self.processing_mode = processing_mode
        self.callback_func = callback_func
        self.start_time = time.time()
        self.last_update_time: float = 0.0
        self.current_phase = "extraction"

        self.steps = PROCESSING_STEPS.copy()
        self.current_step_index = 0
        self.step_progress = 0
        self.step_total = 0

    def update_progress(
        self,
        stage: str,
        frame_num: int | None = None,
        phase: str | None = None,
        step_name: str | None = None,
        step_progress: int | None = None,
        step_total: int | None = None,
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
            progress = self._calculate_batch_progress(step_name, step_progress, step_total)
            progress_data = self._create_batch_progress_data(stage, step_name, progress)
        else:
            progress = self._calculate_serial_progress(frame_num)
            progress_data = self._create_serial_progress_data(stage, frame_num, progress)

        # Execute callback if provided
        if self.callback_func:
            self.callback_func(progress_data)

    def _calculate_batch_progress(
        self,
        step_name: str | None,
        step_progress: int | None,
        step_total: int | None,
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
        overall_progress = ((self.current_step_index + step_progress_ratio) / len(self.steps)) * 100

        return round(overall_progress, PROGRESS_DECIMAL_PLACES)

    def _calculate_serial_progress(self, frame_num: int | None) -> float:
        """Calculate serial mode progress percentage."""
        if frame_num is None:
            return 0.0

        # Phase-based progress calculation
        if self.current_phase == "extraction":
            return (frame_num / self.total_frames * 20) if self.total_frames > 0 else 0
        elif self.current_phase == "processing":
            frame_progress = (frame_num / self.total_frames * 65) if self.total_frames > 0 else 0
            return 20 + frame_progress
        elif self.current_phase == "video":
            return 85 + 15  # Set to 100% for video phase
        else:
            return 0.0

    def _create_batch_progress_data(
        self, stage: str, step_name: str | None, progress: float
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
        self, stage: str, frame_num: int | None, progress: float
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


def effective_step_weights(
    steps: list[str], skipped_steps: set[str] | frozenset[str]
) -> list[float]:
    """
    PURE: Progress weight per step, with disabled optional steps weighted 0.

    The remaining weights are rescaled to sum to 1, so overall progress doesn't jump (e.g. by
    18% when AI upscaling is off) when a skipped step is passed.
    """
    weights = [0.0 if name in skipped_steps else w for name, w in zip(steps, PROGRESS_STEP_WEIGHTS)]
    total = sum(weights)
    return [w / total for w in weights] if total > 0 else list(PROGRESS_STEP_WEIGHTS)


class StepProgressState:
    """
    Progress of every pipeline step, shared by the web UI and the console progress bars.

    Steps report how many frames of the whole video they have finished. In batched mode
    the steps of the inner loop (depth maps to VR assembly) are additionally shown relative
    to the current batch, next to a batch counter.
    """

    def __init__(self, skipped_steps: set[str] | frozenset[str] = frozenset()):
        self.steps = list(PIPELINE_STEP_NAMES)
        self.skipped = frozenset(skipped_steps)
        self.weights = effective_step_weights(self.steps, self.skipped)
        self.done = [0] * len(self.steps)
        self.totals = [0] * len(self.steps)
        self.current_step_index = 0
        self.batch: dict[str, int] | None = None
        self._baseline_time = time.time()
        self._baseline_progress = 0.0

    def record(
        self,
        step_name: str | None,
        step_progress: int | None,
        step_total: int | None,
        phase: str | None = None,
    ) -> None:
        """
        Record a step's progress (frames of the whole video).

        ``phase="resume"`` marks progress restored from an earlier run: it counts towards
        the overall progress but not towards the speed the ETA is based on.
        """
        if step_name in self.steps:
            index = self.steps.index(step_name)
            if phase != "resume":
                self.current_step_index = index
            if step_progress is not None:
                self.done[index] = max(0, int(step_progress))
            if step_total is not None:
                self.totals[index] = max(0, int(step_total))
        if phase == "resume":
            self._baseline_time = time.time()
            self._baseline_progress = self.overall()

    def set_batch(self, index: int, count: int, start: int, size: int) -> None:
        """Set the batch the inner loop is working on (index is 0-based)."""
        self.batch = {"index": index, "count": count, "start": start, "size": size}

    def step_ratio(self, index: int) -> float:
        """PURE: Finished fraction (0-1) of a step over the whole video."""
        total = self.totals[index]
        return min(1.0, self.done[index] / total) if total > 0 else 0.0

    def overall(self) -> float:
        """PURE: Overall progress in percent (weighted by typical step duration)."""
        return 100.0 * sum(w * self.step_ratio(i) for i, w in enumerate(self.weights))

    def batch_ratio(self) -> float:
        """PURE: Finished fraction (0-1) of the current batch across the inner loop's steps."""
        if not self.batch:
            return 0.0
        loop = [i for i, name in enumerate(self.steps) if name in PIPELINE_LOOP_STEP_NAMES]
        weight = sum(self.weights[i] for i in loop)
        if weight <= 0:
            return 0.0
        return sum(self.weights[i] * self._batch_local(i)[0] for i in loop) / (
            weight * max(1, self.batch["size"])
        )

    def _batch_local(self, index: int) -> tuple[int, int]:
        """PURE: A loop step's (progress, total) within the current batch."""
        assert self.batch is not None
        size = self.batch["size"]
        return max(0, min(size, self.done[index] - self.batch["start"])), size

    def display_steps(self) -> list[dict[str, Any]]:
        """
        Per-step progress as shown to the user.

        Returns:
            One dict per step: name, progress, total, skipped, in_batch (True if the
            numbers count frames of the current batch instead of the whole video)
        """
        rows = []
        for i, name in enumerate(self.steps):
            in_batch = self.batch is not None and name in PIPELINE_LOOP_STEP_NAMES
            progress, total = self._batch_local(i) if in_batch else (self.done[i], self.totals[i])
            rows.append(
                {
                    "name": name,
                    "progress": progress,
                    "total": total,
                    "skipped": name in self.skipped,
                    "in_batch": in_batch,
                }
            )
        return rows

    def eta_seconds(self) -> float | None:
        """Estimated seconds remaining, from the speed since this run started (or resumed)."""
        elapsed = time.time() - self._baseline_time
        gained = self.overall() - self._baseline_progress
        if elapsed < 5 or gained < 0.5:
            return None
        return max(0.0, (100.0 - self.overall()) * elapsed / gained)


class BatchOffsetProgress:
    """
    Turns a stage's progress within a batch into progress over the whole video.

    Stages count the frames of the batch they were given; the tracker behind this wrapper
    expects frames of the whole video.
    """

    def __init__(self, tracker: Any, offset: int, total_frames: int):
        self._tracker = tracker
        self._offset = offset
        self._total_frames = total_frames

    def update_progress(
        self,
        stage: str,
        frame_num: int | None = None,
        phase: str | None = None,
        step_name: str | None = None,
        step_progress: int | None = None,
        step_total: int | None = None,
    ) -> None:
        """Forward a progress update with batch-local counts shifted by the batch start."""
        self._tracker.update_progress(
            stage,
            frame_num=None if frame_num is None else frame_num + self._offset,
            phase=phase,
            step_name=step_name,
            step_progress=None if step_progress is None else step_progress + self._offset,
            step_total=self._total_frames,
        )

    def __getattr__(self, name: str) -> Any:
        # Preview methods exist only on trackers that support previews (stages check hasattr)
        target = getattr(self._tracker, name)
        if name in ("send_preview_frame", "send_preview_frame_from_array"):
            offset = self._offset

            def shifted(frame: Any, frame_type: str, frame_number: int) -> Any:
                return target(frame, frame_type, frame_number + offset)

            return shifted
        return target


# Console bar colours: the web UI's theme
_COLOUR_LIME = "#39ff14"  # --accent-lime: overall progress, finished steps
_COLOUR_GREEN = "#00ff41"  # --accent-green: batch counter
_COLOUR_BLUE = "#00d9ff"  # running step (the UI's .progress-bar.bg-info)

# Short labels for the console progress bars (same order as PIPELINE_STEP_NAMES)
_CONSOLE_STEP_LABELS = [
    "Extract frames",
    "Depth maps",
    "Stereo pairs",
    "Distortion",
    "Crop frames",
    "AI upscaling",
    "Assemble VR",
    "Create video",
]


class _TqdmStream:
    """File-like object that prints through tqdm.write so output doesn't break the bars."""

    def __init__(self, stream: Any):
        self._stream = stream
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            tqdm.write(line, file=self._stream)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._stream, "isatty", lambda: False)())


class TqdmProgressTracker:
    """
    Console progress for the CLI: one bar per pipeline step plus an overall bar.

    Mirrors the web UI. In batched mode a batch counter bar is shown and the bars of the
    inner loop (indented below it) count the frames of the current batch.
    Without a terminal (piped output) it prints a status line every few seconds instead.
    """

    def __init__(
        self,
        skipped_steps: set[str] | frozenset[str] = frozenset(),
        batched: bool = True,
        use_bars: bool | None = None,
    ):
        """
        Args:
            skipped_steps: Names of disabled optional steps (no bar, no progress weight)
            batched: Show the batch counter and batch-relative inner bars
            use_bars: Force bars on/off (default: on if stderr is a terminal and tqdm exists)
        """
        import sys

        self.state = StepProgressState(skipped_steps)
        self.batched = batched
        if use_bars is None:
            use_bars = HAS_TQDM and sys.stderr.isatty()
        self.use_bars = bool(use_bars and HAS_TQDM)
        self._bars: dict[str, Any] = {}
        self._old_stdout: Any = None
        self._last_refresh = 0.0
        self._last_line = 0.0
        self._closed = False

    def _open(self) -> None:
        """Create the bars and route print() output around them."""
        import sys

        if self._bars or not self.use_bars:
            return
        position = 0

        def bar(key: str, desc: str, total: int, fmt: str, colour: str) -> None:
            nonlocal position
            self._bars[key] = tqdm(
                total=total,
                desc=desc,
                position=position,
                leave=True,
                bar_format=fmt,
                colour=colour,
                dynamic_ncols=True,
                file=sys.stderr,
            )
            position += 1

        # `unit` carries free text (ETA, frame range): unlike postfix it has no ", " prefix.
        # Bars of the inner loop start further right, so the loop reads as a nested block.
        overall_fmt = "{desc:<18}|{bar}| {percentage:5.1f}% [{elapsed}<{unit}]"
        step_fmt = "{desc:<18}|{bar}| {n_fmt}/{total_fmt}"
        inner_fmt = "  {desc:<22}|{bar}| {n_fmt}/{total_fmt}"
        bar("overall", "Overall", 100, overall_fmt, _COLOUR_LIME)
        self._bars["overall"].unit = "?"
        active = [n for n in self.state.steps if n not in self.state.skipped]
        loop = [n for n in active if self.batched and n in PIPELINE_LOOP_STEP_NAMES]
        for name in active:
            i = self.state.steps.index(name)
            label = f"{i + 1} {_CONSOLE_STEP_LABELS[i]}"
            if name not in loop:
                bar(name, label, 1, step_fmt, _COLOUR_BLUE)
                continue
            if "batch" not in self._bars:
                bar("batch", "Batch", 1, "{desc:<18}|{bar}| {unit}", _COLOUR_GREEN)
                self._bars["batch"].unit = ""
            branch = "└" if name == loop[-1] else "├"
            bar(name, f"{branch} {label}", 1, inner_fmt, _COLOUR_BLUE)

        self._old_stdout = sys.stdout
        sys.stdout = _TqdmStream(self._old_stdout)

    def set_batch_info(self, index: int, count: int, start: int, size: int) -> None:
        """Set the batch the inner loop is working on (index is 0-based)."""
        self.state.set_batch(index, count, start, size)
        self._refresh(force=True)

    def update_progress(
        self,
        stage: str,
        frame_num: int | None = None,
        phase: str | None = None,
        step_name: str | None = None,
        step_progress: int | None = None,
        step_total: int | None = None,
    ) -> None:
        """Record a step's progress (frames of the whole video) and redraw."""
        self.state.record(step_name, step_progress, step_total, phase)
        finished = step_total is not None and step_progress == step_total
        self._refresh(force=finished)

    @staticmethod
    def _set(bar: Any, n: float, total: float) -> None:
        bar.total = total
        bar.n = min(n, total)
        bar.refresh()

    def _refresh(self, force: bool = False) -> None:
        """Redraw the bars (throttled), or print a status line without a terminal."""
        if self._closed:
            return
        now = time.time()
        if not self.use_bars:
            if force or now - self._last_line >= 10:
                self._last_line = now
                print(self._status_line(), flush=True)
            return
        if not force and now - self._last_refresh < PROGRESS_EMIT_INTERVAL:
            return
        self._last_refresh = now
        self._open()

        eta = self.state.eta_seconds()
        overall = self._bars["overall"]
        overall.unit = format_time_duration(eta) if eta is not None else "?"
        self._set(overall, self.state.overall(), 100)
        for row in self.state.display_steps():
            if row["name"] in self._bars:
                bar = self._bars[row["name"]]
                # Like the UI: blue while a step runs, lime once it is finished
                finished = row["total"] > 0 and row["progress"] >= row["total"]
                bar.colour = _COLOUR_LIME if finished else _COLOUR_BLUE
                self._set(bar, row["progress"], max(1, row["total"]))
        batch = self.state.batch
        if batch and "batch" in self._bars:
            bar = self._bars["batch"]
            bar.set_description_str(f"Batch {batch['index'] + 1}/{batch['count']}", False)
            first, last = batch["start"] + 1, batch["start"] + batch["size"]
            bar.unit = f"frames {first}-{last}"
            self._set(bar, batch["index"] + self.state.batch_ratio(), batch["count"])

    def _status_line(self) -> str:
        """PURE: One-line progress summary for output that isn't a terminal."""
        state = self.state
        index = state.current_step_index
        line = (
            f"Overall {state.overall():5.1f}% | Step {index + 1}: {_CONSOLE_STEP_LABELS[index]} "
            f"{state.done[index]}/{state.totals[index]}"
        )
        if state.batch:
            line += f" | Batch {state.batch['index'] + 1}/{state.batch['count']}"
        eta = state.eta_seconds()
        return line + (f" | ETA {format_time_duration(eta)}" if eta is not None else "")

    def finish(self, message: str = "Processing complete") -> None:
        """Draw the final state and release the terminal."""
        self._refresh(force=True)
        self.close()

    def close(self) -> None:
        """Close the bars and restore print() (safe to call more than once)."""
        import sys

        if self._closed:
            return
        self._closed = True
        # Bars at fixed positions leave gaps when closed; print their final state as text
        final_lines = [str(bar) for bar in self._bars.values()]
        for bar in reversed(list(self._bars.values())):
            bar.leave = False
            bar.close()
        self._bars = {}
        if final_lines:
            print("\n".join(final_lines), file=sys.stderr, flush=True)
        if self._old_stdout is not None:
            sys.stdout = self._old_stdout
            self._old_stdout = None
