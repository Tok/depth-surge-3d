"""Tests for StereoPairGenerator module."""

from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.io.operations import save_depth_map
from src.depth_surge_3d.processing.frames.stereo_generator import (
    StereoPairGenerator,
    _process_single_stereo_pair,
    get_stereo_worker_count,
)

MODULE = "src.depth_surge_3d.processing.frames.stereo_generator"


class _InlineExecutor:
    """Stand-in for ProcessPoolExecutor that runs work in-process."""

    def __init__(self, *args, initializer=None, **kwargs):
        if initializer:
            initializer()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, iterable, chunksize=1):
        return map(fn, iterable)

    def shutdown(self, wait=True, cancel_futures=False):
        self.was_shut_down = True


class _BrokenExecutor(_InlineExecutor):
    """Executor whose workers all die (e.g. OOM killer)."""

    def map(self, fn, iterable, chunksize=1):
        raise BrokenProcessPool("A process in the process pool was terminated abruptly")


BASE_SETTINGS = {
    "baseline": 0.065,
    "focal_length": 1000,
    "hole_fill_quality": "fast",
    "keep_intermediates": True,
}


def _write_inputs(directory, count=1, h=100, w=100, depth_shape=None):
    """Write frame and 16-bit depth map files; returns (frame_files, depth_files)."""
    frames, depths = [], []
    for i in range(count):
        frame = directory / f"frame_{i:04d}.png"
        depth = directory / f"depth_{i:04d}.png"
        cv2.imwrite(str(frame), np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
        save_depth_map(depth, np.random.rand(*(depth_shape or (h, w))))
        frames.append(frame)
        depths.append(depth)
    return frames, depths


class TestProcessSingleStereoPair:
    """Test _process_single_stereo_pair worker function."""

    def _args(self, tmp_path, settings=None, with_outputs=True, **kwargs):
        frames, depths = _write_inputs(tmp_path, **kwargs)
        left = str(tmp_path / "left.png") if with_outputs else None
        right = str(tmp_path / "right.png") if with_outputs else None
        return (
            str(frames[0]),
            str(depths[0]),
            "frame_0000",
            left,
            right,
            settings or BASE_SETTINGS,
        )

    def test_process_pair_basic(self, tmp_path):
        """Worker reads frame and depth map, writes both eyes and returns the frame name."""
        args = self._args(tmp_path)

        assert _process_single_stereo_pair(args) == "frame_0000"
        assert cv2.imread(args[3]).shape == (100, 100, 3)
        assert cv2.imread(args[4]).shape == (100, 100, 3)

    def test_process_pair_no_paths(self, tmp_path):
        assert _process_single_stereo_pair(self._args(tmp_path, with_outputs=False)) == "frame_0000"

    def test_depth_map_resolution_mismatch(self, tmp_path):
        """Depth maps smaller than the frame are resized, not an error."""
        args = self._args(tmp_path, depth_shape=(50, 50))
        _process_single_stereo_pair(args)
        assert cv2.imread(args[3]).shape == (100, 100, 3)

    def test_super_sampling_upscales_output(self, tmp_path):
        settings = {**BASE_SETTINGS, "super_sample": "2x"}
        args = self._args(tmp_path, settings=settings, h=60, w=80)
        _process_single_stereo_pair(args)
        assert cv2.imread(args[3]).shape == (120, 160, 3)

    def test_missing_frame(self, tmp_path):
        args = list(self._args(tmp_path))
        args[0] = str(tmp_path / "missing.png")
        with pytest.raises(IOError, match="Failed to read"):
            _process_single_stereo_pair(tuple(args))

    def test_missing_depth_map(self, tmp_path):
        args = list(self._args(tmp_path))
        args[1] = str(tmp_path / "missing.png")
        with pytest.raises(IOError, match="Failed to read"):
            _process_single_stereo_pair(tuple(args))

    def test_unwritable_output(self, tmp_path):
        args = list(self._args(tmp_path))
        args[3] = str(tmp_path / "no_such_dir" / "left.png")
        with pytest.raises(IOError, match="Failed to write"):
            _process_single_stereo_pair(tuple(args))


class TestStereoWorkerCount:
    """Test worker count selection."""

    def test_capped(self):
        with patch(f"{MODULE}.os.cpu_count", return_value=128):
            assert get_stereo_worker_count(1000) == 16

    def test_small_machine(self):
        with patch(f"{MODULE}.os.cpu_count", return_value=2):
            assert get_stereo_worker_count(1000) == 1

    def test_few_tasks(self):
        with patch(f"{MODULE}.os.cpu_count", return_value=32):
            assert get_stereo_worker_count(3) == 3

    def test_unknown_cpu_count(self):
        with patch(f"{MODULE}.os.cpu_count", return_value=None):
            assert get_stereo_worker_count(10) == 1


class TestStereoPairGeneratorInit:
    """Test StereoPairGenerator initialization."""

    def test_init_default(self):
        assert StereoPairGenerator().verbose is False

    def test_init_verbose(self):
        assert StereoPairGenerator(verbose=True).verbose is True


class TestCreateStereoPairs:
    """Test create_stereo_pairs method."""

    @pytest.fixture
    def mock_progress_tracker(self):
        tracker = Mock()
        tracker.update_progress = Mock()
        tracker.send_preview_frame = Mock()
        return tracker

    @pytest.fixture
    def temp_frames(self, tmp_path):
        left_dir = tmp_path / "left_frames"
        right_dir = tmp_path / "right_frames"
        left_dir.mkdir()
        right_dir.mkdir()
        frame_files, depth_files = _write_inputs(tmp_path, count=3, h=60, w=80)
        return {
            "frame_files": frame_files,
            "depth_files": depth_files,
            "directories": {"left_frames": left_dir, "right_frames": right_dir},
        }

    def _run(self, temp_frames, settings, tracker=None, generator=None):
        generator = generator or StereoPairGenerator()
        try:
            return generator.create_stereo_pairs(
                temp_frames["frame_files"],
                temp_frames["depth_files"],
                temp_frames["directories"],
                settings,
                tracker,
            )
        finally:
            generator.close()

    def test_create_stereo_pairs_parallel(self, temp_frames, mock_progress_tracker):
        """Pool path writes all pairs and reports progress."""
        with (
            patch(f"{MODULE}.ProcessPoolExecutor", _InlineExecutor),
            patch(f"{MODULE}.get_stereo_worker_count", return_value=4),
        ):
            assert self._run(temp_frames, BASE_SETTINGS, mock_progress_tracker) is True

        assert len(list(temp_frames["directories"]["left_frames"].glob("*.png"))) == 3
        assert len(list(temp_frames["directories"]["right_frames"].glob("*.png"))) == 3
        mock_progress_tracker.update_progress.assert_called()
        mock_progress_tracker.send_preview_frame.assert_called()

    def test_create_stereo_pairs_without_progress_tracker(self, temp_frames):
        """CLI runs pass no progress tracker (regression: NoneType.update_progress)."""
        with patch(f"{MODULE}.get_stereo_worker_count", return_value=1):
            assert self._run(temp_frames, BASE_SETTINGS, None) is True

    def test_create_stereo_pairs_without_intermediates(self, temp_frames, mock_progress_tracker):
        """Regression: pairs must be written even without intermediates (later steps read them)."""
        settings = {**BASE_SETTINGS, "keep_intermediates": False}
        with patch(f"{MODULE}.get_stereo_worker_count", return_value=1):
            assert self._run(temp_frames, settings, mock_progress_tracker) is True

        assert len(list(temp_frames["directories"]["left_frames"].glob("*.png"))) == 3
        assert len(list(temp_frames["directories"]["right_frames"].glob("*.png"))) == 3

    def test_broken_pool_falls_back_to_sequential(self, temp_frames):
        """Regression #17: dead workers must not hang the pipeline."""
        with (
            patch(f"{MODULE}.ProcessPoolExecutor", _BrokenExecutor),
            patch(f"{MODULE}.get_stereo_worker_count", return_value=8),
        ):
            assert self._run(temp_frames, BASE_SETTINGS) is True

        assert len(list(temp_frames["directories"]["left_frames"].glob("*.png"))) == 3

    def test_worker_exception_reported(self, temp_frames, mock_progress_tracker):
        """Exceptions from the worker are surfaced as failure."""
        with (
            patch(f"{MODULE}.get_stereo_worker_count", return_value=1),
            patch(f"{MODULE}._process_single_stereo_pair", side_effect=RuntimeError("boom")),
        ):
            assert self._run(temp_frames, BASE_SETTINGS, mock_progress_tracker) is False

    def test_frame_depth_count_mismatch(self, temp_frames):
        """Mismatched frame/depth counts fail fast with a clear error."""
        temp_frames["depth_files"] = temp_frames["depth_files"][:2]
        assert self._run(temp_frames, BASE_SETTINGS) is False

    def test_real_process_pool(self, temp_frames):
        """Smoke test with a real process pool."""
        with patch(f"{MODULE}.get_stereo_worker_count", return_value=2):
            assert self._run(temp_frames, BASE_SETTINGS) is True

        assert len(list(Path(temp_frames["directories"]["left_frames"]).glob("*.png"))) == 3

    def test_pool_is_reused_across_batches(self, temp_frames):
        """Batches share one worker pool instead of starting a new one each time."""
        generator = StereoPairGenerator()
        with (
            patch(f"{MODULE}.ProcessPoolExecutor", _InlineExecutor),
            patch(f"{MODULE}.get_stereo_worker_count", return_value=4),
        ):
            args = (temp_frames["frame_files"], temp_frames["depth_files"])
            assert generator.create_stereo_pairs(*args, temp_frames["directories"], BASE_SETTINGS)
            pool = generator._executor
            assert generator.create_stereo_pairs(*args, temp_frames["directories"], BASE_SETTINGS)
            assert generator._executor is pool

        generator.close()
        assert generator._executor is None and pool.was_shut_down

    def test_stop_request_propagates_and_closes_pool(self, temp_frames):
        """A stop request (InterruptedError from the tracker) isn't reported as a failure."""
        tracker = Mock()
        tracker.update_progress = Mock(side_effect=InterruptedError("stop"))
        generator = StereoPairGenerator()
        with (
            patch(f"{MODULE}.ProcessPoolExecutor", _InlineExecutor),
            patch(f"{MODULE}.get_stereo_worker_count", return_value=4),
            pytest.raises(InterruptedError),
        ):
            self._run(temp_frames, BASE_SETTINGS, tracker, generator)
        assert generator._executor is None

    def test_no_frames(self, temp_frames):
        temp_frames["frame_files"] = []
        temp_frames["depth_files"] = []
        assert self._run(temp_frames, BASE_SETTINGS) is False
