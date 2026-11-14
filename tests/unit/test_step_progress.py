"""Tests for step/batch progress tracking shared by the web UI and the console bars."""

import io
import sys
from unittest.mock import Mock, patch

import pytest

from src.depth_surge_3d.core.constants import PIPELINE_STEP_NAMES
from src.depth_surge_3d.utils.domain.progress import (
    BatchOffsetProgress,
    StepProgressState,
    TqdmProgressTracker,
    effective_step_weights,
)

MODULE = "src.depth_surge_3d.utils.domain.progress"
SKIPPED = frozenset({"AI Upscaling"})


class TestEffectiveStepWeights:
    def test_skipped_steps_weigh_nothing(self):
        weights = effective_step_weights(PIPELINE_STEP_NAMES, SKIPPED)
        assert weights[PIPELINE_STEP_NAMES.index("AI Upscaling")] == 0
        assert sum(weights) == pytest.approx(1.0)

    def test_everything_skipped_falls_back(self):
        weights = effective_step_weights(PIPELINE_STEP_NAMES, set(PIPELINE_STEP_NAMES))
        assert sum(weights) == pytest.approx(1.0)


class TestStepProgressState:
    def test_overall_is_weighted_sum_of_all_steps(self):
        state = StepProgressState(SKIPPED)
        assert state.overall() == 0.0

        for name in PIPELINE_STEP_NAMES:
            state.record(name, 50, 100)
        assert state.overall() == pytest.approx(50.0)

        for name in PIPELINE_STEP_NAMES:
            state.record(name, 100, 100)
        assert state.overall() == pytest.approx(100.0)

    def test_steps_progress_independently(self):
        """Batched mode: later steps advance while earlier ones aren't finished."""
        state = StepProgressState()
        state.record("Depth Map Generation", 40, 100)
        state.record("VR Assembly", 20, 100)

        assert state.done[1] == 40 and state.done[6] == 20
        assert state.current_step_index == 6
        expected = 100 * (state.weights[1] * 0.4 + state.weights[6] * 0.2)
        assert state.overall() == pytest.approx(expected)

    def test_unknown_step_and_overshoot_are_harmless(self):
        state = StepProgressState()
        state.record("Something Else", 5, 10)
        state.record(None, 5, 10)
        state.record("Crop Frames", 20, 10)
        assert state.step_ratio(4) == 1.0

    def test_display_counts_frames_of_the_batch_for_inner_steps(self):
        state = StepProgressState(SKIPPED)
        state.record("Frame Extraction", 250, 250)
        state.record("Depth Map Generation", 190, 250)
        state.record("Stereo Pair Creation", 140, 250)
        state.record("Crop Frames", 100, 250)
        state.set_batch(index=1, count=3, start=100, size=100)

        rows = {row["name"]: row for row in state.display_steps()}

        assert (rows["Frame Extraction"]["progress"], rows["Frame Extraction"]["total"]) == (
            250,
            250,
        )
        assert not rows["Frame Extraction"]["in_batch"] and not rows["Video Creation"]["in_batch"]
        # depth is ahead of the batch, stereo in the middle, crop hasn't started this batch
        assert (
            rows["Depth Map Generation"]["progress"],
            rows["Depth Map Generation"]["total"],
        ) == (90, 100)
        assert rows["Stereo Pair Creation"]["progress"] == 40
        assert rows["Crop Frames"]["progress"] == 0
        assert rows["AI Upscaling"]["skipped"] and rows["Stereo Pair Creation"]["in_batch"]

    def test_display_without_batches_counts_whole_video(self):
        state = StepProgressState()
        state.record("Stereo Pair Creation", 140, 250)
        row = state.display_steps()[2]
        assert (row["progress"], row["total"], row["in_batch"]) == (140, 250, False)

    def test_batch_ratio(self):
        state = StepProgressState(SKIPPED)
        assert state.batch_ratio() == 0.0
        state.set_batch(0, 2, 0, 100)
        assert state.batch_ratio() == 0.0
        for name in PIPELINE_STEP_NAMES[1:7]:
            state.record(name, 50, 200)
        assert state.batch_ratio() == pytest.approx(0.5)
        for name in PIPELINE_STEP_NAMES[1:7]:
            state.record(name, 100, 200)
        assert state.batch_ratio() == pytest.approx(1.0)

    def test_eta_ignores_progress_restored_from_an_earlier_run(self):
        with patch(f"{MODULE}.time.time", return_value=1000.0):
            state = StepProgressState()
            state.record("Frame Extraction", 100, 100, phase="resume")
            state.record("Depth Map Generation", 50, 100, phase="resume")
            restored = state.overall()
            assert state.current_step_index == 0
        with patch(f"{MODULE}.time.time", return_value=1002.0):
            assert state.eta_seconds() is None  # too early to tell
        with patch(f"{MODULE}.time.time", return_value=1100.0):
            assert state.eta_seconds() is None  # nothing gained since resuming
            state.record("Depth Map Generation", 100, 100)
            gained = state.overall() - restored
            assert state.eta_seconds() == pytest.approx((100 - state.overall()) * 100 / gained)


class TestBatchOffsetProgress:
    def test_shifts_batch_counts_to_the_whole_video(self):
        tracker = Mock()
        wrapped = BatchOffsetProgress(tracker, offset=200, total_frames=1000)

        wrapped.update_progress(
            "Cropping",
            frame_num=5,
            phase="cropping",
            step_name="Crop Frames",
            step_progress=5,
            step_total=100,
        )

        tracker.update_progress.assert_called_once_with(
            "Cropping",
            frame_num=205,
            phase="cropping",
            step_name="Crop Frames",
            step_progress=205,
            step_total=1000,
        )

    def test_missing_counts_stay_missing(self):
        tracker = Mock()
        BatchOffsetProgress(tracker, 200, 1000).update_progress("Working")
        assert tracker.update_progress.call_args[1]["step_progress"] is None

    def test_preview_frame_numbers_are_shifted(self):
        tracker = Mock()
        wrapped = BatchOffsetProgress(tracker, 200, 1000)

        wrapped.send_preview_frame("path.png", "vr_frame", 7)
        wrapped.send_preview_frame_from_array("array", "depth_map", 8)

        tracker.send_preview_frame.assert_called_once_with("path.png", "vr_frame", 207)
        tracker.send_preview_frame_from_array.assert_called_once_with("array", "depth_map", 208)

    def test_preview_support_follows_the_tracker(self):
        class NoPreviews:
            def update_progress(self, *args, **kwargs):
                pass

            def finish(self):
                return "done"

        wrapped = BatchOffsetProgress(NoPreviews(), 0, 10)
        assert not hasattr(wrapped, "send_preview_frame")
        assert wrapped.finish() == "done"


def _plain(lines):
    """Lines without colour codes."""
    import re

    return [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in lines]


class TestTqdmProgressTracker:
    def _run(self, tracker):
        tracker.update_progress("x", step_name="Frame Extraction", step_progress=20, step_total=20)
        tracker.set_batch_info(0, 2, 0, 10)
        tracker.update_progress(
            "x", step_name="Depth Map Generation", step_progress=10, step_total=20
        )
        tracker.update_progress(
            "x", step_name="Stereo Pair Creation", step_progress=5, step_total=20
        )

    def test_block_mirrors_the_ui(self):
        tracker = TqdmProgressTracker(SKIPPED, batched=True, use_bars=False)
        self._run(tracker)

        coloured = tracker.render_lines(100)
        lines = _plain(coloured)

        assert [line.split("|")[0].strip() for line in lines] == [
            "Overall",
            "1 Extract frames",
            "Batch 1/2",
            "├ 2 Depth maps",
            "├ 3 Stereo pairs",
            "├ 4 Distortion",
            "├ 5 Crop frames",
            "└ 7 Assemble VR",
            "8 Create video",
        ]
        # inner bars count frames of the batch and start further right than the outer ones
        assert lines[3].endswith("| 10/10") and lines[4].endswith("| 5/10")
        assert lines[2].endswith("| frames 1-10")
        assert lines[3].index("|") > lines[1].index("|")
        assert lines[8].endswith("| -")  # nothing known about the last step yet
        # UI colours: finished lime, running blue, batch counter green
        assert "38;2;57;255;20m" in coloured[3] and "38;2;0;217;255m" in coloured[4]
        assert "38;2;0;255;65m" in coloured[2]
        assert all(len(line) <= 100 for line in lines)

    def test_loop_is_shown_before_the_first_batch(self):
        """Batched mode shows the batch bar and nested steps from the start."""
        tracker = TqdmProgressTracker(batched=True, use_bars=False)
        tracker.update_progress("x", step_name="Frame Extraction", step_progress=5, step_total=20)

        lines = _plain(tracker.render_lines(100))

        assert lines[2].startswith("Batch ") and lines[2].endswith("| waiting")
        assert lines[3].lstrip().startswith("├ 2 Depth maps") and lines[3].endswith("| -")

    def test_sequential_mode_has_no_batch_bar(self):
        tracker = TqdmProgressTracker(batched=False, use_bars=False)
        tracker.update_progress(
            "x", step_name="Depth Map Generation", step_progress=10, step_total=20
        )

        lines = _plain(tracker.render_lines(100))

        assert len(lines) == 9 and not any(line.startswith("Batch") for line in lines)
        assert lines[2].startswith("2 Depth maps") and lines[2].endswith("| 10/20")

    def test_block_is_redrawn_in_place_and_messages_go_above_it(self):
        """Regression: bars leaked into the scrollback whenever a message was printed."""
        stderr = io.StringIO()
        stdout = sys.stdout
        with patch("sys.stderr", stderr):
            tracker = TqdmProgressTracker(SKIPPED, batched=True, use_bars=True)
            self._run(tracker)
            assert sys.stdout is not stdout
            print("a message while the bars are shown")
            tracker.finish()
            tracker.close()
        assert sys.stdout is stdout

        output = stderr.getvalue()
        first_draw, later = output.split("\x1b[9A", 1)
        assert first_draw.count("\n") == 9  # the block: nine lines
        # every redraw goes back to the block's first line and clears it before writing
        assert later.count("\x1b[9A") == output.count("\r\x1b[J") - 2
        message_at = output.index("a message while the bars are shown")
        assert output[:message_at].endswith("\x1b[9A\r\x1b[J")
        assert output[message_at:].count("\n") >= 10  # message line, then the whole block again
        assert output.rstrip().endswith("m")  # ends with the block, cursor below it

    def test_status_lines_without_a_terminal(self, capsys):
        tracker = TqdmProgressTracker(SKIPPED, use_bars=False)
        self._run(tracker)
        stereo = {"step_name": "Stereo Pair Creation", "step_total": 20}
        tracker.update_progress("x", step_progress=6, **stereo)
        lines = capsys.readouterr().out.splitlines()
        assert lines[0].startswith("Overall") and "Step 1: Extract frames 20/20" in lines[0]
        assert "Batch 1/2" in lines[-1]
        assert not any("6/20" in line for line in lines)  # throttled

        tracker.finish()
        assert "Stereo pairs 6/20" in capsys.readouterr().out  # the final state is always shown
        tracker.update_progress("x", step_progress=20, **stereo)
        assert capsys.readouterr().out == ""  # closed

    def test_messages_after_closing_go_to_stdout(self, capsys):
        tracker = TqdmProgressTracker(use_bars=False)
        tracker.close()
        tracker.write_line("plain line")
        assert "plain line" in capsys.readouterr().out
