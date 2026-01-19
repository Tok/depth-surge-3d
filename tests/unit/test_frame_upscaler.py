"""Tests for FrameUpscalerProcessor module."""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch

from src.depth_surge_3d.processing.frames.frame_upscaler import FrameUpscalerProcessor


class TestFrameUpscalerInit:
    """Test FrameUpscalerProcessor initialization."""

    def test_init_default(self):
        """Test default initialization."""
        processor = FrameUpscalerProcessor()

        assert processor.verbose is False
        assert processor.upscaler is None

    def test_init_verbose(self):
        """Test initialization with verbose enabled."""
        processor = FrameUpscalerProcessor(verbose=True)

        assert processor.verbose is True


class TestApplyUpscaling:
    """Test apply_upscaling main entry point."""

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create mock progress tracker."""
        tracker = Mock()
        tracker.update_progress = Mock()
        return tracker

    @pytest.fixture
    def temp_frames(self, tmp_path):
        """Create temporary frame directories with test frames."""
        left_dir = tmp_path / "left_cropped"
        right_dir = tmp_path / "right_cropped"
        left_dir.mkdir()
        right_dir.mkdir()

        # Create test frames
        for i in range(3):
            left_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            right_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(left_dir / f"frame_{i:04d}.png"), left_frame)
            cv2.imwrite(str(right_dir / f"frame_{i:04d}.png"), right_frame)

        return {
            "base": tmp_path,
            "left_cropped": left_dir,
            "right_cropped": right_dir,
        }

    def test_apply_upscaling_disabled(self, temp_frames, mock_progress_tracker):
        """Test upscaling when disabled (model='none')."""
        processor = FrameUpscalerProcessor()

        settings = {"upscale_model": "none", "device": "cpu"}

        with patch(
            "src.depth_surge_3d.inference.create_upscaler",
            return_value=None,
        ):
            result = processor.apply_upscaling(temp_frames, settings, mock_progress_tracker)

        assert result is True

    def test_apply_upscaling_model_load_failure(self, temp_frames, mock_progress_tracker):
        """Test upscaling when model fails to load."""
        processor = FrameUpscalerProcessor()

        settings = {"upscale_model": "x4", "device": "cpu", "keep_intermediates": True}

        mock_upscaler = Mock()
        mock_upscaler.load_model.return_value = False

        with patch(
            "src.depth_surge_3d.inference.create_upscaler",
            return_value=mock_upscaler,
        ):
            result = processor.apply_upscaling(temp_frames, settings, mock_progress_tracker)

        assert result is False
        mock_upscaler.load_model.assert_called_once()

    def test_apply_upscaling_success(self, temp_frames, mock_progress_tracker):
        """Test successful upscaling."""
        processor = FrameUpscalerProcessor()

        settings = {"upscale_model": "x4", "device": "cpu", "keep_intermediates": True}

        mock_upscaler = Mock()
        mock_upscaler.load_model.return_value = True
        mock_upscaler.unload_model = Mock()
        mock_upscaler.upscale_image = Mock(
            side_effect=lambda img: np.zeros(
                (img.shape[0] * 2, img.shape[1] * 2, 3), dtype=np.uint8
            )
        )

        with patch(
            "src.depth_surge_3d.inference.create_upscaler",
            return_value=mock_upscaler,
        ):
            result = processor.apply_upscaling(temp_frames, settings, mock_progress_tracker)

        assert result is True
        mock_upscaler.load_model.assert_called_once()
        mock_upscaler.unload_model.assert_called_once()

    def test_apply_upscaling_missing_directories(self, mock_progress_tracker, tmp_path):
        """Test upscaling with missing source directories."""
        processor = FrameUpscalerProcessor()

        directories = {"base": tmp_path}  # Missing left_cropped and right_cropped
        settings = {"upscale_model": "x4", "device": "cpu", "keep_intermediates": True}

        result = processor.apply_upscaling(directories, settings, mock_progress_tracker)

        assert result is False

    def test_apply_upscaling_exception_handling(self, temp_frames, mock_progress_tracker):
        """Test exception handling during upscaling."""
        processor = FrameUpscalerProcessor()

        settings = {"upscale_model": "x4", "device": "cpu", "keep_intermediates": True}

        with patch(
            "src.depth_surge_3d.inference.create_upscaler",
            side_effect=RuntimeError("GPU error"),
        ):
            result = processor.apply_upscaling(temp_frames, settings, mock_progress_tracker)

        assert result is False


class TestProcessUpscalingFrames:
    """Test _process_upscaling_frames method."""

    @pytest.fixture
    def mock_upscaler(self):
        """Create mock upscaler."""
        upscaler = Mock()
        upscaler.upscale_image = Mock(
            side_effect=lambda img: np.zeros(
                (img.shape[0] * 2, img.shape[1] * 2, 3), dtype=np.uint8
            )
        )
        return upscaler

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create mock progress tracker."""
        tracker = Mock()
        tracker.update_progress = Mock()
        tracker.send_preview_frame = Mock()
        tracker.send_preview_frame_from_array = Mock()
        return tracker

    @pytest.fixture
    def temp_frames(self, tmp_path):
        """Create temporary frame directories."""
        left_dir = tmp_path / "left_cropped"
        right_dir = tmp_path / "right_cropped"
        left_dir.mkdir()
        right_dir.mkdir()

        # Create test frames
        for i in range(3):
            left_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            right_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(left_dir / f"frame_{i:04d}.png"), left_frame)
            cv2.imwrite(str(right_dir / f"frame_{i:04d}.png"), right_frame)

        return {
            "base": tmp_path,
            "left_cropped": left_dir,
            "right_cropped": right_dir,
        }

    def test_process_frames_with_intermediates(
        self, mock_upscaler, mock_progress_tracker, temp_frames
    ):
        """Test processing with keep_intermediates=True."""
        processor = FrameUpscalerProcessor()

        settings = {"keep_intermediates": True}

        result = processor._process_upscaling_frames(
            mock_upscaler,
            temp_frames["left_cropped"],
            temp_frames["right_cropped"],
            temp_frames,
            settings,
            mock_progress_tracker,
        )

        assert result is True
        assert mock_upscaler.upscale_image.call_count == 6  # 3 left + 3 right

        # Check output directories were created
        left_upscaled = temp_frames["base"] / "07_left_upscaled"
        right_upscaled = temp_frames["base"] / "07_right_upscaled"
        assert left_upscaled.exists()
        assert right_upscaled.exists()

        # Check output files were created
        assert len(list(left_upscaled.glob("*.png"))) == 3
        assert len(list(right_upscaled.glob("*.png"))) == 3

    def test_process_frames_without_intermediates(
        self, mock_upscaler, mock_progress_tracker, temp_frames
    ):
        """Test processing with keep_intermediates=False."""
        processor = FrameUpscalerProcessor()

        settings = {"keep_intermediates": False}

        result = processor._process_upscaling_frames(
            mock_upscaler,
            temp_frames["left_cropped"],
            temp_frames["right_cropped"],
            temp_frames,
            settings,
            mock_progress_tracker,
        )

        assert result is True
        assert mock_upscaler.upscale_image.call_count == 6

        # Check output directories were NOT created
        left_upscaled = temp_frames["base"] / "07_left_upscaled"
        right_upscaled = temp_frames["base"] / "07_right_upscaled"
        assert not left_upscaled.exists()
        assert not right_upscaled.exists()

    def test_process_frames_mismatched_count(
        self, mock_upscaler, mock_progress_tracker, temp_frames
    ):
        """Test processing with mismatched frame counts."""
        processor = FrameUpscalerProcessor()

        # Remove one right frame
        right_frames = list(temp_frames["right_cropped"].glob("*.png"))
        right_frames[0].unlink()

        settings = {"keep_intermediates": False}

        result = processor._process_upscaling_frames(
            mock_upscaler,
            temp_frames["left_cropped"],
            temp_frames["right_cropped"],
            temp_frames,
            settings,
            mock_progress_tracker,
        )

        assert result is False

    def test_process_frames_with_existing_output_dirs(
        self, mock_upscaler, mock_progress_tracker, temp_frames
    ):
        """Test processing when output directories already exist."""
        processor = FrameUpscalerProcessor()

        left_upscaled = temp_frames["base"] / "left_upscaled"
        right_upscaled = temp_frames["base"] / "right_upscaled"
        left_upscaled.mkdir()
        right_upscaled.mkdir()

        directories = {
            **temp_frames,
            "left_upscaled": left_upscaled,
            "right_upscaled": right_upscaled,
        }

        settings = {"keep_intermediates": True}

        result = processor._process_upscaling_frames(
            mock_upscaler,
            directories["left_cropped"],
            directories["right_cropped"],
            directories,
            settings,
            mock_progress_tracker,
        )

        assert result is True


class TestUpscaleFramePair:
    """Test _upscale_frame_pair method."""

    @pytest.fixture
    def mock_upscaler(self):
        """Create mock upscaler."""
        upscaler = Mock()
        upscaler.upscale_image = Mock(
            side_effect=lambda img: np.zeros(
                (img.shape[0] * 2, img.shape[1] * 2, 3), dtype=np.uint8
            )
        )
        return upscaler

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create mock progress tracker."""
        tracker = Mock()
        tracker.update_progress = Mock()
        tracker.send_preview_frame = Mock()
        tracker.send_preview_frame_from_array = Mock()
        return tracker

    @pytest.fixture
    def temp_frames(self, tmp_path):
        """Create temporary frame files."""
        left_dir = tmp_path / "left"
        right_dir = tmp_path / "right"
        left_dir.mkdir()
        right_dir.mkdir()

        left_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        right_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        left_path = left_dir / "frame_0000.png"
        right_path = right_dir / "frame_0000.png"

        cv2.imwrite(str(left_path), left_frame)
        cv2.imwrite(str(right_path), right_frame)

        return {
            "left_path": left_path,
            "right_path": right_path,
            "left_dir": left_dir,
            "right_dir": right_dir,
        }

    def test_upscale_pair_with_intermediates(
        self, mock_upscaler, mock_progress_tracker, temp_frames, tmp_path
    ):
        """Test upscaling single frame pair with saving."""
        processor = FrameUpscalerProcessor()

        left_upscaled = tmp_path / "left_upscaled"
        right_upscaled = tmp_path / "right_upscaled"
        left_upscaled.mkdir()
        right_upscaled.mkdir()

        settings = {"keep_intermediates": True}

        processor._upscale_frame_pair(
            mock_upscaler,
            temp_frames["left_path"],
            temp_frames["right_path"],
            left_upscaled,
            right_upscaled,
            settings,
            frame_idx=0,
            total_frames=3,
            progress_tracker=mock_progress_tracker,
        )

        # Check upscaler was called
        assert mock_upscaler.upscale_image.call_count == 2

        # Check files were saved
        assert (left_upscaled / "frame_0000.png").exists()
        assert (right_upscaled / "frame_0000.png").exists()

        # Check progress was updated
        mock_progress_tracker.update_progress.assert_called()

    def test_upscale_pair_without_intermediates(
        self, mock_upscaler, mock_progress_tracker, temp_frames, tmp_path
    ):
        """Test upscaling without saving (keep_intermediates=False)."""
        processor = FrameUpscalerProcessor()

        left_upscaled = tmp_path / "left_upscaled"
        right_upscaled = tmp_path / "right_upscaled"
        left_upscaled.mkdir()
        right_upscaled.mkdir()

        settings = {"keep_intermediates": False}

        processor._upscale_frame_pair(
            mock_upscaler,
            temp_frames["left_path"],
            temp_frames["right_path"],
            left_upscaled,
            right_upscaled,
            settings,
            frame_idx=1,  # Not first or last, no preview
            total_frames=100,
            progress_tracker=mock_progress_tracker,
        )

        # Check upscaler was called
        assert mock_upscaler.upscale_image.call_count == 2

        # Check files were NOT saved
        assert not (left_upscaled / "frame_0000.png").exists()
        assert not (right_upscaled / "frame_0000.png").exists()

    def test_upscale_pair_with_preview_first_frame(
        self, mock_upscaler, mock_progress_tracker, temp_frames, tmp_path
    ):
        """Test preview sending for first frame."""
        processor = FrameUpscalerProcessor()

        left_upscaled = tmp_path / "left_upscaled"
        right_upscaled = tmp_path / "right_upscaled"
        left_upscaled.mkdir()
        right_upscaled.mkdir()

        settings = {"keep_intermediates": True}

        processor._upscale_frame_pair(
            mock_upscaler,
            temp_frames["left_path"],
            temp_frames["right_path"],
            left_upscaled,
            right_upscaled,
            settings,
            frame_idx=0,  # First frame
            total_frames=100,
            progress_tracker=mock_progress_tracker,
        )

        # Check preview was sent (from file since keep_intermediates=True)
        mock_progress_tracker.send_preview_frame.assert_called()

    def test_upscale_pair_with_preview_last_frame(
        self, mock_upscaler, mock_progress_tracker, temp_frames, tmp_path
    ):
        """Test preview sending for last frame."""
        processor = FrameUpscalerProcessor()

        left_upscaled = tmp_path / "left_upscaled"
        right_upscaled = tmp_path / "right_upscaled"
        left_upscaled.mkdir()
        right_upscaled.mkdir()

        settings = {"keep_intermediates": False}

        processor._upscale_frame_pair(
            mock_upscaler,
            temp_frames["left_path"],
            temp_frames["right_path"],
            left_upscaled,
            right_upscaled,
            settings,
            frame_idx=99,  # Last frame
            total_frames=100,
            progress_tracker=mock_progress_tracker,
        )

        # Check preview was sent (from array since keep_intermediates=False)
        mock_progress_tracker.send_preview_frame_from_array.assert_called()

    def test_upscale_pair_missing_image(self, mock_upscaler, mock_progress_tracker, tmp_path):
        """Test upscaling when image file is missing."""
        processor = FrameUpscalerProcessor()

        fake_left = tmp_path / "nonexistent_left.png"
        fake_right = tmp_path / "nonexistent_right.png"

        left_upscaled = tmp_path / "left_upscaled"
        right_upscaled = tmp_path / "right_upscaled"
        left_upscaled.mkdir()
        right_upscaled.mkdir()

        settings = {"keep_intermediates": True}

        # Should not raise, just print warning
        processor._upscale_frame_pair(
            mock_upscaler,
            fake_left,
            fake_right,
            left_upscaled,
            right_upscaled,
            settings,
            frame_idx=0,
            total_frames=3,
            progress_tracker=mock_progress_tracker,
        )

        # Upscaler should not have been called
        mock_upscaler.upscale_image.assert_not_called()

    def test_upscale_pair_no_output_dirs(self, mock_upscaler, mock_progress_tracker, temp_frames):
        """Test upscaling without output directories (None)."""
        processor = FrameUpscalerProcessor()

        settings = {"keep_intermediates": True}

        # Pass None for output directories
        processor._upscale_frame_pair(
            mock_upscaler,
            temp_frames["left_path"],
            temp_frames["right_path"],
            None,  # left_upscaled
            None,  # right_upscaled
            settings,
            frame_idx=0,
            total_frames=3,
            progress_tracker=mock_progress_tracker,
        )

        # Should still upscale
        assert mock_upscaler.upscale_image.call_count == 2


class TestGetUpscalingSourceDirs:
    """Test _get_upscaling_source_dirs static method."""

    def test_get_source_dirs_with_cropped(self, tmp_path):
        """Test getting source directories when cropped dirs exist."""
        left_cropped = tmp_path / "left_cropped"
        right_cropped = tmp_path / "right_cropped"
        left_cropped.mkdir()
        right_cropped.mkdir()

        directories = {
            "left_cropped": left_cropped,
            "right_cropped": right_cropped,
        }

        settings = {"apply_distortion": False}

        left, right = FrameUpscalerProcessor._get_upscaling_source_dirs(directories, settings)

        assert left == left_cropped
        assert right == right_cropped

    def test_get_source_dirs_missing_cropped(self):
        """Test getting source directories when cropped dirs don't exist."""
        directories = {"base": Path("/tmp")}
        settings = {"apply_distortion": False}

        left, right = FrameUpscalerProcessor._get_upscaling_source_dirs(directories, settings)

        assert left is None
        assert right is None

    def test_get_source_dirs_partial_cropped(self, tmp_path):
        """Test getting source directories when only one cropped dir exists."""
        left_cropped = tmp_path / "left_cropped"
        left_cropped.mkdir()

        directories = {
            "left_cropped": left_cropped,
            # Missing right_cropped
        }

        settings = {"apply_distortion": False}

        left, right = FrameUpscalerProcessor._get_upscaling_source_dirs(directories, settings)

        # Should return None for both if incomplete
        assert left is None
        assert right is None
