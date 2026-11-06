"""Tests for StereoPairGenerator module."""

from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

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


class TestProcessSingleStereoPair:
    """Test _process_single_stereo_pair worker function."""

    def test_process_pair_basic(self, tmp_path):
        """Worker writes both eyes and returns the frame name."""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        depth_map = np.random.rand(100, 100)
        left_path = str(tmp_path / "left.png")
        right_path = str(tmp_path / "right.png")

        name = _process_single_stereo_pair(
            (frame, depth_map, "frame_0000", left_path, right_path, BASE_SETTINGS)
        )

        assert name == "frame_0000"
        assert cv2.imread(left_path).shape == (100, 100, 3)
        assert cv2.imread(right_path).shape == (100, 100, 3)

    def test_process_pair_no_paths(self):
        """Worker without paths writes nothing and doesn't fail."""
        frame = np.random.randint(0, 255, (50, 80, 3), dtype=np.uint8)
        depth_map = np.random.rand(50, 80)

        assert (
            _process_single_stereo_pair((frame, depth_map, "f", None, None, BASE_SETTINGS)) == "f"
        )

    def test_depth_map_resolution_mismatch(self, tmp_path):
        """Regression #16/#17: depth maps at a different resolution than the frame."""
        frame = np.random.randint(0, 255, (192, 108, 3), dtype=np.uint8)
        depth_map = np.random.rand(307, 307)  # e.g. stale 2x-upscaled cached depth
        left_path = str(tmp_path / "left.png")

        _process_single_stereo_pair((frame, depth_map, "f", left_path, None, BASE_SETTINGS))

        assert cv2.imread(left_path).shape == (192, 108, 3)

    def test_super_sampling_upscales_output(self, tmp_path):
        """Super sampling renders at a higher resolution and preserves aspect ratio."""
        frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        depth_map = np.random.rand(100, 200)
        left_path = str(tmp_path / "left.png")
        settings = {
            **BASE_SETTINGS,
            "super_sample": "auto",
            "per_eye_width": 400,
            "per_eye_height": 400,
        }

        _process_single_stereo_pair((frame, depth_map, "f", left_path, None, settings))

        # scale capped at 2x
        assert cv2.imread(left_path).shape == (200, 400, 3)

    def test_hole_fill_skipped_when_disabled(self):
        """hole_fill_quality='none' skips hole filling."""
        frame = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        settings = {**BASE_SETTINGS, "hole_fill_quality": "none"}

        with patch("src.depth_surge_3d.utils.imaging.image_processing.hole_fill_image") as fill:
            _process_single_stereo_pair((frame, np.random.rand(20, 20), "f", None, None, settings))

        fill.assert_not_called()


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

        return {
            "frames": np.random.randint(0, 255, (3, 60, 80, 3), dtype=np.uint8),
            "depth_maps": np.random.rand(3, 60, 80),
            "frame_files": [tmp_path / f"frame_{i:04d}.png" for i in range(3)],
            "directories": {"left_frames": left_dir, "right_frames": right_dir},
        }

    def _run(self, temp_frames, settings, tracker=None):
        return StereoPairGenerator().create_stereo_pairs(
            temp_frames["frames"],
            temp_frames["depth_maps"],
            temp_frames["frame_files"],
            temp_frames["directories"],
            settings,
            tracker,
        )

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
        """No files are written when keep_intermediates is False."""
        settings = {**BASE_SETTINGS, "keep_intermediates": False}
        with patch(f"{MODULE}.get_stereo_worker_count", return_value=1):
            assert self._run(temp_frames, settings, mock_progress_tracker) is True

        assert not list(temp_frames["directories"]["left_frames"].glob("*.png"))
        mock_progress_tracker.send_preview_frame.assert_not_called()

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
        temp_frames["depth_maps"] = temp_frames["depth_maps"][:2]
        assert self._run(temp_frames, BASE_SETTINGS) is False

    def test_real_process_pool(self, temp_frames):
        """Smoke test with a real process pool."""
        with patch(f"{MODULE}.get_stereo_worker_count", return_value=2):
            assert self._run(temp_frames, BASE_SETTINGS) is True

        assert len(list(Path(temp_frames["directories"]["left_frames"]).glob("*.png"))) == 3
