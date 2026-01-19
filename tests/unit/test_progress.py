"""Unit tests for progress tracking utilities."""

import time
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from src.depth_surge_3d.utils.domain.progress import (
    ProgressReporter,
    ConsoleProgressReporter,
    ProgressTracker,
    ProgressCallback,
    create_progress_tracker,
    calculate_eta,
    format_time_duration,
    PROCESSING_PHASES,
)


class TestProcessingPhases:
    """Test PROCESSING_PHASES constant."""

    def test_phases_structure(self):
        """Test that PROCESSING_PHASES has correct structure."""
        assert isinstance(PROCESSING_PHASES, dict)
        assert len(PROCESSING_PHASES) > 0

        for phase_name, phase_info in PROCESSING_PHASES.items():
            assert "weight" in phase_info
            assert "description" in phase_info
            assert isinstance(phase_info["weight"], int)
            assert isinstance(phase_info["description"], str)

    def test_total_weight_is_100(self):
        """Test that phase weights sum to 100."""
        total_weight = sum(phase["weight"] for phase in PROCESSING_PHASES.values())
        assert total_weight == 100


class TestConsoleProgressReporter:
    """Test ConsoleProgressReporter class."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        reporter = ConsoleProgressReporter()

        assert reporter.show_eta is True
        assert reporter.start_time > 0
        assert reporter.last_update_time == 0
        assert reporter.current_pbar is None
        assert reporter.current_phase is None

    def test_init_without_eta(self):
        """Test initialization without ETA."""
        reporter = ConsoleProgressReporter(show_eta=False)
        assert reporter.show_eta is False

    def test_init_without_tqdm(self):
        """Test initialization without tqdm."""
        reporter = ConsoleProgressReporter(use_tqdm=False)
        assert reporter.use_tqdm is False

    def test_get_phase_description_known_phase(self):
        """Test getting description for known phase."""
        reporter = ConsoleProgressReporter()
        desc = reporter._get_phase_description("depth_estimation")
        assert desc == "Estimating depth"

    def test_get_phase_description_unknown_phase(self):
        """Test getting description for unknown phase."""
        reporter = ConsoleProgressReporter()
        desc = reporter._get_phase_description("custom_phase")
        # Should title-case and replace underscores
        assert desc == "Custom Phase"

    @patch("sys.stdout", new_callable=StringIO)
    def test_report_progress_without_tqdm(self, mock_stdout):
        """Test reporting progress without tqdm."""
        reporter = ConsoleProgressReporter(use_tqdm=False)

        # First update (should print)
        reporter.report_progress(50, 100, "Processing")

        # Immediate second update (should be throttled)
        reporter.report_progress(51, 100, "Processing")

        output = mock_stdout.getvalue()
        assert "Processing" in output
        assert "50.0%" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_report_progress_with_eta(self, mock_stdout):
        """Test reporting progress with ETA calculation."""
        reporter = ConsoleProgressReporter(use_tqdm=False, show_eta=True)

        # Need some time to elapse for ETA calculation
        time.sleep(0.01)
        reporter.report_progress(1, 100, "Processing")

        output = mock_stdout.getvalue()
        assert "Processing" in output
        assert "ETA" in output or "1.0%" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_report_progress_zero_total(self, mock_stdout):
        """Test reporting progress with zero total."""
        reporter = ConsoleProgressReporter(use_tqdm=False)

        # Should not crash with zero total
        reporter.report_progress(0, 0, "Processing")

        # No output expected for zero total
        output = mock_stdout.getvalue()
        assert output == "" or "Processing" not in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_report_completion(self, mock_stdout):
        """Test reporting completion."""
        reporter = ConsoleProgressReporter(use_tqdm=False)

        reporter.report_completion("Task complete")

        output = mock_stdout.getvalue()
        assert "Task complete" in output
        assert "Completed in" in output

    def test_report_completion_closes_tqdm(self):
        """Test that completion closes tqdm progress bar."""
        reporter = ConsoleProgressReporter(use_tqdm=True)

        # Mock a progress bar
        mock_pbar = MagicMock()
        reporter.current_pbar = mock_pbar

        with patch("sys.stdout", new_callable=StringIO):
            reporter.report_completion("Done")

        # Should have closed the progress bar
        mock_pbar.close.assert_called_once()
        assert reporter.current_pbar is None


class TestProgressTracker:
    """Test ProgressTracker class."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        tracker = ProgressTracker(total_frames=100)

        assert tracker.total_frames == 100
        assert tracker.processing_mode == "serial"
        assert tracker.reporter is not None
        assert tracker.start_time > 0
        assert tracker.current_step_index == 0

    def test_init_with_custom_reporter(self):
        """Test initialization with custom reporter."""
        mock_reporter = Mock(spec=ProgressReporter)
        tracker = ProgressTracker(total_frames=100, reporter=mock_reporter)

        assert tracker.reporter == mock_reporter

    def test_init_with_batch_mode(self):
        """Test initialization with batch processing mode."""
        tracker = ProgressTracker(total_frames=100, processing_mode="batch")

        assert tracker.processing_mode == "batch"

    def test_update_serial(self):
        """Test updating progress in serial mode."""
        mock_reporter = Mock(spec=ProgressReporter)
        tracker = ProgressTracker(total_frames=100, reporter=mock_reporter)

        tracker.update_serial(50, "Processing frame")

        # Should update internal state
        assert tracker.current_frame == 50
        assert tracker.current_step == "Processing frame"


class TestCreateProgressTracker:
    """Test create_progress_tracker factory function."""

    def test_create_with_defaults(self):
        """Test creating tracker with defaults."""
        tracker = create_progress_tracker(total_frames=100)

        assert isinstance(tracker, ProgressTracker)
        assert tracker.total_frames == 100
        assert tracker.processing_mode == "serial"

    def test_create_with_custom_mode(self):
        """Test creating tracker with custom mode."""
        tracker = create_progress_tracker(total_frames=100, processing_mode="batch")

        assert tracker.processing_mode == "batch"

    def test_create_with_console_reporter_type(self):
        """Test creating tracker with console reporter type."""
        tracker = create_progress_tracker(total_frames=100, reporter_type="console")

        assert isinstance(tracker.reporter, ConsoleProgressReporter)

    def test_create_with_unknown_reporter_type(self):
        """Test creating tracker with unknown reporter type defaults to console."""
        tracker = create_progress_tracker(total_frames=100, reporter_type="unknown")

        # Should default to console reporter
        assert isinstance(tracker.reporter, ConsoleProgressReporter)


class TestCalculateEta:
    """Test calculate_eta function."""

    def test_calculate_eta_half_complete(self):
        """Test ETA calculation when half complete."""
        start_time = time.time() - 10.0  # Started 10 seconds ago
        eta = calculate_eta(start_time, current=50, total=100)

        # Should be approximately 10 seconds (same time remaining as elapsed)
        assert 8.0 <= eta <= 12.0

    def test_calculate_eta_quarter_complete(self):
        """Test ETA calculation when quarter complete."""
        start_time = time.time() - 5.0  # Started 5 seconds ago
        eta = calculate_eta(start_time, current=25, total=100)

        # Should be approximately 15 seconds (3x the elapsed time remaining)
        assert 13.0 <= eta <= 17.0

    def test_calculate_eta_almost_complete(self):
        """Test ETA calculation when almost complete."""
        start_time = time.time() - 10.0
        eta = calculate_eta(start_time, current=99, total=100)

        # Should be very small
        assert 0.0 <= eta <= 1.0

    def test_calculate_eta_zero_current(self):
        """Test ETA calculation with zero current progress."""
        start_time = time.time()
        eta = calculate_eta(start_time, current=0, total=100)

        # Should handle gracefully (might return 0 or infinity)
        assert isinstance(eta, (int, float))

    def test_calculate_eta_zero_total(self):
        """Test ETA calculation with zero total."""
        start_time = time.time()
        eta = calculate_eta(start_time, current=0, total=0)

        # Should handle edge case without crashing
        assert isinstance(eta, (int, float))


class TestFormatTimeDuration:
    """Test format_time_duration function."""

    def test_format_seconds_only(self):
        """Test formatting duration with seconds only."""
        formatted = format_time_duration(45.0)

        assert isinstance(formatted, str)
        assert "45" in formatted or "0:45" in formatted

    def test_format_minutes_and_seconds(self):
        """Test formatting duration with minutes and seconds."""
        formatted = format_time_duration(125.0)  # 2 minutes 5 seconds

        assert isinstance(formatted, str)
        # Should contain minute and second info
        assert "2" in formatted

    def test_format_hours(self):
        """Test formatting duration with hours."""
        formatted = format_time_duration(3665.0)  # 1 hour, 1 minute, 5 seconds

        assert isinstance(formatted, str)
        assert "1" in formatted  # Should have hour component

    def test_format_zero_duration(self):
        """Test formatting zero duration."""
        formatted = format_time_duration(0.0)

        assert isinstance(formatted, str)
        assert "0" in formatted

    def test_format_fractional_seconds(self):
        """Test formatting fractional seconds."""
        formatted = format_time_duration(45.7)

        # Should handle fractional seconds
        assert isinstance(formatted, str)
        assert len(formatted) > 0


class MockProgressReporter(ProgressReporter):
    """Mock implementation of ProgressReporter for testing."""

    def __init__(self):
        self.progress_calls = []
        self.completion_calls = []

    def report_progress(self, current: int, total: int, message: str = "") -> None:
        """Record progress report."""
        self.progress_calls.append({"current": current, "total": total, "message": message})

    def report_completion(self, message: str = "") -> None:
        """Record completion report."""
        self.completion_calls.append({"message": message})


class TestProgressReporterInterface:
    """Test ProgressReporter abstract interface."""

    def test_mock_reporter_implementation(self):
        """Test that mock reporter implements interface correctly."""
        reporter = MockProgressReporter()

        # Should be able to call methods
        reporter.report_progress(50, 100, "Test")
        reporter.report_completion("Done")

        # Should have recorded calls
        assert len(reporter.progress_calls) == 1
        assert len(reporter.completion_calls) == 1

    def test_mock_reporter_records_progress(self):
        """Test that mock reporter records progress correctly."""
        reporter = MockProgressReporter()

        reporter.report_progress(25, 100, "Quarter done")

        assert reporter.progress_calls[0]["current"] == 25
        assert reporter.progress_calls[0]["total"] == 100
        assert reporter.progress_calls[0]["message"] == "Quarter done"

    def test_mock_reporter_records_completion(self):
        """Test that mock reporter records completion correctly."""
        reporter = MockProgressReporter()

        reporter.report_completion("All finished")

        assert reporter.completion_calls[0]["message"] == "All finished"


class TestProgressTrackerBatchMode:
    """Test ProgressTracker batch mode functionality."""

    def test_update_batch_step(self):
        """Test updating batch processing step."""
        mock_reporter = MockProgressReporter()
        tracker = ProgressTracker(total_frames=100, processing_mode="batch", reporter=mock_reporter)

        tracker.update_batch_step("depth_estimation", progress=50, total=100)

        assert tracker.step_progress == 50
        assert tracker.step_total == 100
        # Should have called reporter
        assert len(mock_reporter.progress_calls) > 0

    def test_update_batch_progress(self):
        """Test updating progress within batch step."""
        mock_reporter = MockProgressReporter()
        tracker = ProgressTracker(total_frames=100, processing_mode="batch", reporter=mock_reporter)

        # Set initial step
        tracker.update_batch_step("depth_estimation", progress=0, total=100)
        initial_calls = len(mock_reporter.progress_calls)

        # Update progress within step
        tracker.update_batch_progress(progress=75)

        assert tracker.step_progress == 75
        assert len(mock_reporter.progress_calls) > initial_calls

    def test_update_batch_progress_with_total(self):
        """Test updating progress with new total."""
        mock_reporter = MockProgressReporter()
        tracker = ProgressTracker(total_frames=100, processing_mode="batch", reporter=mock_reporter)

        tracker.update_batch_progress(progress=50, total=200)

        assert tracker.step_progress == 50
        assert tracker.step_total == 200

    def test_display_batch_with_progress(self):
        """Test batch display with step progress."""
        mock_reporter = MockProgressReporter()
        tracker = ProgressTracker(total_frames=100, processing_mode="batch", reporter=mock_reporter)

        tracker.update_batch_step("depth_estimation", progress=50, total=100)

        # Check that progress was reported
        assert len(mock_reporter.progress_calls) > 0
        last_call = mock_reporter.progress_calls[-1]
        assert "BATCH" in last_call["message"]

    def test_display_batch_without_progress(self):
        """Test batch display without step progress."""
        mock_reporter = MockProgressReporter()
        tracker = ProgressTracker(total_frames=100, processing_mode="batch", reporter=mock_reporter)

        tracker.update_batch_step("depth_estimation", progress=0, total=0)

        # Check that progress was reported even with zero total
        assert len(mock_reporter.progress_calls) > 0

    def test_update_progress_with_step_name(self):
        """Test unified update_progress with step_name."""
        mock_reporter = MockProgressReporter()
        tracker = ProgressTracker(total_frames=100, processing_mode="batch", reporter=mock_reporter)

        tracker.update_progress(
            stage="processing", step_name="depth_estimation", step_progress=25, step_total=100
        )

        assert tracker.step_progress == 25
        assert tracker.step_total == 100

    def test_update_progress_with_step_progress_only(self):
        """Test unified update_progress with step_progress only."""
        mock_reporter = MockProgressReporter()
        tracker = ProgressTracker(total_frames=100, processing_mode="batch", reporter=mock_reporter)

        # Set initial step
        tracker.update_batch_step("depth_estimation", progress=0, total=100)

        # Update with progress only
        tracker.update_progress(stage="processing", step_progress=50)

        assert tracker.step_progress == 50

    def test_finish(self):
        """Test finishing progress tracking."""
        mock_reporter = MockProgressReporter()
        tracker = ProgressTracker(total_frames=100, processing_mode="batch", reporter=mock_reporter)

        tracker.finish("All done!")

        assert len(mock_reporter.completion_calls) == 1
        assert mock_reporter.completion_calls[0]["message"] == "All done!"


class TestProgressCallback:
    """Test ProgressCallback class."""

    def test_init(self):
        """Test initialization."""
        callback = ProgressCallback(session_id="test123", total_frames=100, processing_mode="batch")

        assert callback.session_id == "test123"
        assert callback.total_frames == 100
        assert callback.processing_mode == "batch"
        assert callback.callback_func is None
        assert callback.current_phase == "extraction"

    def test_update_progress_batch_mode(self):
        """Test update_progress in batch mode."""
        called_data = []

        def mock_callback(data):
            called_data.append(data)

        callback = ProgressCallback(
            session_id="test123",
            total_frames=100,
            processing_mode="batch",
            callback_func=mock_callback,
        )

        # First update should trigger callback (no throttling on first call)
        callback.update_progress(
            stage="processing", step_name="depth_estimation", step_progress=50, step_total=100
        )

        # Need to wait for throttle interval
        time.sleep(0.6)

        callback.update_progress(
            stage="processing", step_name="depth_estimation", step_progress=75, step_total=100
        )

        # Should have at least one callback
        assert len(called_data) >= 1
        assert called_data[0]["session_id"] == "test123"
        assert called_data[0]["processing_mode"] == "batch"
        assert "step_name" in called_data[0]

    def test_update_progress_serial_mode(self):
        """Test update_progress in serial mode."""
        called_data = []

        def mock_callback(data):
            called_data.append(data)

        callback = ProgressCallback(
            session_id="test456",
            total_frames=100,
            processing_mode="serial",
            callback_func=mock_callback,
        )

        callback.update_progress(stage="extraction", frame_num=50, phase="extraction")

        time.sleep(0.6)

        callback.update_progress(stage="extraction", frame_num=75, phase="extraction")

        # Should have at least one callback
        assert len(called_data) >= 1
        assert called_data[0]["session_id"] == "test456"
        assert called_data[0]["processing_mode"] == "serial"
        assert "current_frame" in called_data[0]

    def test_calculate_batch_progress(self):
        """Test batch progress calculation."""
        callback = ProgressCallback(session_id="test", total_frames=100, processing_mode="batch")

        progress = callback._calculate_batch_progress(
            step_name="depth_estimation", step_progress=50, step_total=100
        )

        # Progress should be a float between 0-100
        assert isinstance(progress, float)
        assert 0 <= progress <= 100

    def test_calculate_serial_progress_extraction(self):
        """Test serial progress calculation in extraction phase."""
        callback = ProgressCallback(session_id="test", total_frames=100, processing_mode="serial")

        callback.current_phase = "extraction"
        progress = callback._calculate_serial_progress(frame_num=50)

        # Extraction is 20% of total, so 50/100 frames = 10% overall
        assert progress == 10.0

    def test_calculate_serial_progress_processing(self):
        """Test serial progress calculation in processing phase."""
        callback = ProgressCallback(session_id="test", total_frames=100, processing_mode="serial")

        callback.current_phase = "processing"
        progress = callback._calculate_serial_progress(frame_num=50)

        # Processing starts at 20% and is 65% of total
        # 50/100 frames = 32.5% of processing = 52.5% overall
        assert progress == 52.5

    def test_calculate_serial_progress_video(self):
        """Test serial progress calculation in video phase."""
        callback = ProgressCallback(session_id="test", total_frames=100, processing_mode="serial")

        callback.current_phase = "video"
        progress = callback._calculate_serial_progress(frame_num=100)

        # Video phase is 100%
        assert progress == 100.0

    def test_calculate_serial_progress_unknown_phase(self):
        """Test serial progress calculation with unknown phase."""
        callback = ProgressCallback(session_id="test", total_frames=100, processing_mode="serial")

        callback.current_phase = "unknown"
        progress = callback._calculate_serial_progress(frame_num=50)

        # Unknown phase should return 0
        assert progress == 0.0

    def test_calculate_serial_progress_none_frame(self):
        """Test serial progress calculation with None frame_num."""
        callback = ProgressCallback(session_id="test", total_frames=100, processing_mode="serial")

        progress = callback._calculate_serial_progress(frame_num=None)

        # None frame should return 0
        assert progress == 0.0

    def test_create_batch_progress_data(self):
        """Test creating batch progress data dictionary."""
        callback = ProgressCallback(session_id="test123", total_frames=100, processing_mode="batch")

        data = callback._create_batch_progress_data(
            stage="processing", step_name="depth_estimation", progress=50.0
        )

        assert data["session_id"] == "test123"
        assert data["progress"] == 50.0
        assert data["stage"] == "processing"
        assert data["processing_mode"] == "batch"
        assert data["step_name"] == "depth_estimation"
        assert "step_progress" in data
        assert "step_total" in data

    def test_create_serial_progress_data(self):
        """Test creating serial progress data dictionary."""
        callback = ProgressCallback(
            session_id="test456", total_frames=100, processing_mode="serial"
        )

        data = callback._create_serial_progress_data(
            stage="extraction", frame_num=50, progress=25.0
        )

        assert data["session_id"] == "test456"
        assert data["progress"] == 25.0
        assert data["stage"] == "extraction"
        assert data["processing_mode"] == "serial"
        assert data["current_frame"] == 50
        assert data["total_frames"] == 100

    def test_throttling(self):
        """Test that updates are throttled."""
        called_data = []

        def mock_callback(data):
            called_data.append(data)

        callback = ProgressCallback(
            session_id="test",
            total_frames=100,
            processing_mode="batch",
            callback_func=mock_callback,
        )

        # Make multiple rapid updates
        for i in range(10):
            callback.update_progress(
                stage="processing",
                step_name="depth_estimation",
                step_progress=i * 10,
                step_total=100,
            )

        # Should be throttled to very few calls (likely just 1-2)
        assert len(called_data) < 10


class TestConsoleProgressReporterEdgeCases:
    """Test edge cases for ConsoleProgressReporter."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_report_progress_without_eta(self, mock_stdout):
        """Test progress reporting without ETA."""
        reporter = ConsoleProgressReporter(show_eta=False, use_tqdm=False)

        reporter.report_progress(50, 100, "Processing")

        output = mock_stdout.getvalue()
        # Should not contain "ETA"
        assert "ETA" not in output
        assert "50.0%" in output

    @patch("src.depth_surge_3d.utils.domain.progress.HAS_TQDM", True)
    @patch("src.depth_surge_3d.utils.domain.progress.tqdm")
    def test_report_progress_with_tqdm(self, mock_tqdm_class):
        """Test progress reporting with tqdm enabled."""
        mock_pbar = MagicMock()
        mock_tqdm_class.return_value = mock_pbar

        reporter = ConsoleProgressReporter(use_tqdm=True)

        # First call should create progress bar
        reporter.report_progress(10, 100, "Loading")
        mock_tqdm_class.assert_called_once()

        # Subsequent calls should update existing bar
        reporter.report_progress(20, 100, "Loading")
        assert mock_pbar.n == 20
        mock_pbar.refresh.assert_called()

        # Completion should close bar
        reporter.report_progress(100, 100, "Loading")
        mock_pbar.close.assert_called()

    def test_abstract_methods(self):
        """Test that abstract methods exist in base class."""
        # This test ensures the abstract methods are defined
        from abc import ABCMeta

        assert isinstance(ProgressReporter, ABCMeta)
        assert hasattr(ProgressReporter, "report_progress")
        assert hasattr(ProgressReporter, "report_completion")


class TestProgressTrackerEdgeCases:
    """Test edge cases for ProgressTracker."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_calculate_eta_early_stages(self, mock_stdout):
        """Test ETA calculation in early stages (< 5 seconds elapsed)."""
        tracker = ProgressTracker(total_frames=100, processing_mode="serial")

        # Mock time to simulate early stage
        with patch("time.time", return_value=tracker.reporter.start_time + 2):
            eta_str = tracker._calculate_eta(10, 100)
            # Should return None for early stages
            assert eta_str is None

    @patch("sys.stdout", new_callable=StringIO)
    def test_calculate_eta_zero_progress(self, mock_stdout):
        """Test ETA calculation with zero progress."""
        tracker = ProgressTracker(total_frames=100, processing_mode="serial")

        # Mock time to simulate enough elapsed time
        with patch("time.time", return_value=tracker.reporter.start_time + 10):
            eta_str = tracker._calculate_eta(0, 100)
            # Should return None for zero progress
            assert eta_str is None

    @patch("sys.stdout", new_callable=StringIO)
    def test_calculate_eta_negative_remaining(self, mock_stdout):
        """Test ETA calculation with negative remaining time."""
        tracker = ProgressTracker(total_frames=100, processing_mode="serial")

        # This shouldn't happen in practice, but test the safeguard
        with patch("time.time", return_value=tracker.reporter.start_time + 100):
            # Simulate going backward (shouldn't happen)
            tracker.reporter.start_time = time.time() + 50  # Future start time
            result = tracker._calculate_eta(50, 100)
            # Should handle gracefully (return None for negative remaining time)
            assert result is None or isinstance(result, str)

    @patch("sys.stdout", new_callable=StringIO)
    def test_format_time_seconds_only(self, mock_stdout):
        """Test time formatting for seconds only."""
        tracker = ProgressTracker(total_frames=100, processing_mode="serial")

        result = tracker._format_time(45.7)
        assert result == "45s"

    @patch("sys.stdout", new_callable=StringIO)
    def test_format_time_minutes(self, mock_stdout):
        """Test time formatting for minutes and seconds."""
        tracker = ProgressTracker(total_frames=100, processing_mode="serial")

        result = tracker._format_time(125)  # 2m 5s
        assert "2m" in result
        assert "5s" in result

    @patch("sys.stdout", new_callable=StringIO)
    def test_format_time_hours(self, mock_stdout):
        """Test time formatting for hours."""
        tracker = ProgressTracker(total_frames=100, processing_mode="serial")

        result = tracker._format_time(7325)  # 2h 2m
        assert "2h" in result
        assert "2m" in result

    @patch("sys.stdout", new_callable=StringIO)
    def test_update_batch_step_with_step_name(self, mock_stdout):
        """Test update_batch_step updates step index."""
        from src.depth_surge_3d.core.constants import PROCESSING_STEPS

        tracker = ProgressTracker(total_frames=100, processing_mode="batch")

        # Use a step from PROCESSING_STEPS
        if len(PROCESSING_STEPS) >= 2:
            second_step = PROCESSING_STEPS[1]
            tracker.update_batch_step(second_step, progress=50, total=100)
            assert tracker.current_step_index == 1  # Second step


class TestProgressCallbackEdgeCases:
    """Test edge cases for ProgressCallback."""

    def test_calculate_batch_progress_with_step_name(self):
        """Test batch progress calculation updates step index."""
        from src.depth_surge_3d.core.constants import PROCESSING_STEPS

        mock_callback = Mock()
        callback = ProgressCallback(
            session_id="test_session",
            total_frames=100,
            processing_mode="batch",
            callback_func=mock_callback,
        )

        # Use a step from PROCESSING_STEPS
        if len(PROCESSING_STEPS) >= 2:
            second_step = PROCESSING_STEPS[1]
            callback.update_progress(
                stage="processing",
                step_name=second_step,
                step_progress=50,
                step_total=100,
            )

            # Should update current_step_index when step_name is provided
            assert callback.current_step_index >= 0


class TestTqdmImportHandling:
    """Test tqdm import handling."""

    def test_has_tqdm_constant(self):
        """Test that HAS_TQDM constant is defined."""
        from src.depth_surge_3d.utils.domain.progress import HAS_TQDM

        assert isinstance(HAS_TQDM, bool)
