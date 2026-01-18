"""Tests for StereoPairGenerator module."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.depth_surge_3d.processing.frames.stereo_generator import (
    StereoPairGenerator,
    _process_single_stereo_pair,
)


class TestProcessSingleStereoPair:
    """Test _process_single_stereo_pair worker function."""

    def test_process_pair_basic(self, tmp_path):
        """Test basic stereo pair processing."""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        depth_map = np.random.rand(100, 100)
        frame_name = "frame_0000"

        left_path = str(tmp_path / "left.png")
        right_path = str(tmp_path / "right.png")

        settings = {
            "baseline": 0.065,
            "focal_length": 1000,
            "hole_fill_quality": "fast",
        }

        with patch(
            "src.depth_surge_3d.processing.frames.stereo_generator.depth_to_disparity"
        ) as mock_disp:
            with patch(
                "src.depth_surge_3d.processing.frames.stereo_generator.create_shifted_image"
            ) as mock_shift:
                with patch(
                    "src.depth_surge_3d.processing.frames.stereo_generator.hole_fill_image"
                ) as mock_fill:
                    mock_disp.return_value = np.zeros((100, 100))
                    mock_shift.return_value = frame
                    mock_fill.return_value = frame

                    left_img, right_img, name = _process_single_stereo_pair(
                        (frame, depth_map, frame_name, left_path, right_path, settings)
                    )

        assert name == frame_name
        assert Path(left_path).exists()
        assert Path(right_path).exists()

    def test_process_pair_no_hole_fill(self, tmp_path):
        """Test stereo pair processing without hole filling."""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        depth_map = np.random.rand(100, 100)

        settings = {
            "baseline": 0.065,
            "focal_length": 1000,
            "hole_fill_quality": "none",
        }

        with patch(
            "src.depth_surge_3d.processing.frames.stereo_generator.depth_to_disparity"
        ) as mock_disp:
            with patch(
                "src.depth_surge_3d.processing.frames.stereo_generator.create_shifted_image"
            ) as mock_shift:
                with patch(
                    "src.depth_surge_3d.processing.frames.stereo_generator.hole_fill_image"
                ) as mock_fill:
                    mock_disp.return_value = np.zeros((100, 100))
                    mock_shift.return_value = frame

                    _process_single_stereo_pair(
                        (frame, depth_map, "frame_0000", None, None, settings)
                    )

                    # Hole fill should not be called
                    mock_fill.assert_not_called()


class TestStereoPairGeneratorInit:
    """Test StereoPairGenerator initialization."""

    def test_init_default(self):
        """Test default initialization."""
        generator = StereoPairGenerator()
        assert generator.verbose is False

    def test_init_verbose(self):
        """Test initialization with verbose."""
        generator = StereoPairGenerator(verbose=True)
        assert generator.verbose is True


class TestCreateStereoPairs:
    """Test create_stereo_pairs method."""

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create mock progress tracker."""
        tracker = Mock()
        tracker.update_progress = Mock()
        tracker.send_preview_frame = Mock()
        return tracker

    @pytest.fixture
    def temp_frames(self, tmp_path):
        """Create temporary frame data."""
        left_dir = tmp_path / "left_frames"
        right_dir = tmp_path / "right_frames"
        left_dir.mkdir()
        right_dir.mkdir()

        frames = np.random.randint(0, 255, (3, 100, 100, 3), dtype=np.uint8)
        depth_maps = np.random.rand(3, 100, 100)
        frame_files = [tmp_path / f"frame_{i:04d}.png" for i in range(3)]

        directories = {
            "left_frames": left_dir,
            "right_frames": right_dir,
        }

        return {
            "frames": frames,
            "depth_maps": depth_maps,
            "frame_files": frame_files,
            "directories": directories,
        }

    def test_create_stereo_pairs_success(self, temp_frames, mock_progress_tracker):
        """Test successful stereo pair creation."""
        generator = StereoPairGenerator()

        settings = {
            "baseline": 0.065,
            "focal_length": 1000,
            "hole_fill_quality": "fast",
            "keep_intermediates": True,
        }

        # Mock the multiprocessing Pool
        mock_pool = Mock()
        mock_imap_result = [
            (temp_frames["frames"][i], temp_frames["frames"][i], f"frame_{i:04d}") for i in range(3)
        ]
        mock_pool.__enter__ = Mock(return_value=mock_pool)
        mock_pool.__exit__ = Mock(return_value=False)
        mock_pool.imap = Mock(return_value=iter(mock_imap_result))

        with patch("multiprocessing.Pool", return_value=mock_pool):
            result = generator.create_stereo_pairs(
                temp_frames["frames"],
                temp_frames["depth_maps"],
                temp_frames["frame_files"],
                temp_frames["directories"],
                settings,
                mock_progress_tracker,
            )

        assert result is True
        mock_pool.imap.assert_called_once()

    def test_create_stereo_pairs_without_intermediates(self, temp_frames, mock_progress_tracker):
        """Test stereo pair creation without saving intermediates."""
        generator = StereoPairGenerator()

        settings = {
            "baseline": 0.065,
            "focal_length": 1000,
            "hole_fill_quality": "fast",
            "keep_intermediates": False,
        }

        # Mock the multiprocessing Pool
        mock_pool = Mock()
        mock_imap_result = [
            (temp_frames["frames"][i], temp_frames["frames"][i], f"frame_{i:04d}") for i in range(3)
        ]
        mock_pool.__enter__ = Mock(return_value=mock_pool)
        mock_pool.__exit__ = Mock(return_value=False)
        mock_pool.imap = Mock(return_value=iter(mock_imap_result))

        with patch("multiprocessing.Pool", return_value=mock_pool):
            result = generator.create_stereo_pairs(
                temp_frames["frames"],
                temp_frames["depth_maps"],
                temp_frames["frame_files"],
                temp_frames["directories"],
                settings,
                mock_progress_tracker,
            )

        assert result is True

    def test_create_stereo_pairs_exception_handling(self, temp_frames, mock_progress_tracker):
        """Test exception handling during stereo pair creation."""
        generator = StereoPairGenerator()

        settings = {
            "baseline": 0.065,
            "focal_length": 1000,
            "hole_fill_quality": "fast",
            "keep_intermediates": True,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.stereo_generator._process_single_stereo_pair",
            side_effect=RuntimeError("Test error"),
        ):
            result = generator.create_stereo_pairs(
                temp_frames["frames"],
                temp_frames["depth_maps"],
                temp_frames["frame_files"],
                temp_frames["directories"],
                settings,
                mock_progress_tracker,
            )

        assert result is False
