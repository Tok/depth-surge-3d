"""Tests for video_processor upscaling functionality."""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch

from src.depth_surge_3d.processing.video_processor import VideoProcessor


class TestUpscaling:
    """Test upscaling functionality in VideoProcessor."""

    @pytest.fixture
    def mock_depth_estimator(self):
        """Create a mock depth estimator."""
        estimator = Mock()
        return estimator

    @pytest.fixture
    def mock_upscaler(self):
        """Create a mock upscaler."""
        upscaler = Mock()
        upscaler.load_model.return_value = True
        upscaler.unload_model.return_value = None
        # Mock upscale_image to return 2x larger image
        upscaler.upscale_image = Mock(
            side_effect=lambda img: np.zeros(
                (img.shape[0] * 2, img.shape[1] * 2, 3), dtype=np.uint8
            )
        )
        return upscaler

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create a mock progress tracker."""
        tracker = Mock()
        tracker.update_progress = Mock()
        tracker.send_preview_frame = Mock()
        return tracker

    @pytest.fixture
    def temp_frame_dirs(self, tmp_path):
        """Create temporary directories with test frames."""
        left_dir = tmp_path / "left"
        right_dir = tmp_path / "right"
        left_dir.mkdir()
        right_dir.mkdir()

        # Create test frames
        for i in range(3):
            left_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            right_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(left_dir / f"frame_{i:04d}.png"), left_frame)
            cv2.imwrite(str(right_dir / f"frame_{i:04d}.png"), right_frame)

        return {"left": left_dir, "right": right_dir, "base": tmp_path}

    def test_apply_upscaling_disabled(
        self, temp_frame_dirs, mock_depth_estimator, mock_progress_tracker
    ):
        """Test upscaling when disabled (model='none')."""
        processor = VideoProcessor(mock_depth_estimator)
        settings = {"upscale_model": "none", "device": "cpu"}
        directories = temp_frame_dirs

        with patch("src.depth_surge_3d.models.upscaler.create_upscaler") as mock_create:
            mock_create.return_value = None
            result = processor._apply_upscaling(
                directories["left"],
                directories["right"],
                directories,
                settings,
                mock_progress_tracker,
            )

        assert result is True

    def test_apply_upscaling_load_model_failure(
        self, temp_frame_dirs, mock_depth_estimator, mock_progress_tracker
    ):
        """Test upscaling when model fails to load."""
        processor = VideoProcessor(mock_depth_estimator)
        settings = {"upscale_model": "x4", "device": "cpu"}
        directories = temp_frame_dirs

        mock_upscaler = Mock()
        mock_upscaler.load_model.return_value = False

        with patch("src.depth_surge_3d.models.upscaler.create_upscaler") as mock_create:
            mock_create.return_value = mock_upscaler
            result = processor._apply_upscaling(
                directories["left"],
                directories["right"],
                directories,
                settings,
                mock_progress_tracker,
            )

        assert result is False
        mock_upscaler.load_model.assert_called_once()

    def test_apply_upscaling_success(
        self, temp_frame_dirs, mock_depth_estimator, mock_upscaler, mock_progress_tracker
    ):
        """Test successful upscaling."""
        processor = VideoProcessor(mock_depth_estimator)
        settings = {
            "upscale_model": "x4",
            "device": "cpu",
            "keep_intermediates": True,
        }
        directories = {
            **temp_frame_dirs,
            "left_upscaled": temp_frame_dirs["base"] / "left_upscaled",
            "right_upscaled": temp_frame_dirs["base"] / "right_upscaled",
        }
        directories["left_upscaled"].mkdir()
        directories["right_upscaled"].mkdir()

        with patch("src.depth_surge_3d.models.upscaler.create_upscaler") as mock_create:
            mock_create.return_value = mock_upscaler
            result = processor._apply_upscaling(
                directories["left"],
                directories["right"],
                directories,
                settings,
                mock_progress_tracker,
            )

        assert result is True
        mock_upscaler.load_model.assert_called_once()
        mock_upscaler.unload_model.assert_called_once()
        assert mock_upscaler.upscale_image.call_count >= 6  # 3 left + 3 right frames

    def test_process_upscaling_frames_mismatch(
        self, temp_frame_dirs, mock_depth_estimator, mock_upscaler, mock_progress_tracker
    ):
        """Test upscaling with mismatched frame counts."""
        # Remove one right frame to create mismatch
        right_frames = list(temp_frame_dirs["right"].glob("*.png"))
        right_frames[0].unlink()

        processor = VideoProcessor(mock_depth_estimator)
        settings = {"keep_intermediates": False}
        directories = temp_frame_dirs

        result = processor._process_upscaling_frames(
            mock_upscaler,
            directories["left"],
            directories["right"],
            directories,
            settings,
            mock_progress_tracker,
        )

        assert result is False

    def test_process_upscaling_frames_no_intermediates(
        self, temp_frame_dirs, mock_depth_estimator, mock_upscaler, mock_progress_tracker
    ):
        """Test that temporary directories are NOT created when keep_intermediates=False."""
        processor = VideoProcessor(mock_depth_estimator)
        settings = {"keep_intermediates": False}
        directories = temp_frame_dirs  # No upscaled dirs

        # Mock send_preview_frame_from_array to avoid AttributeError
        mock_progress_tracker.send_preview_frame_from_array = Mock()

        result = processor._process_upscaling_frames(
            mock_upscaler,
            directories["left"],
            directories["right"],
            directories,
            settings,
            mock_progress_tracker,
        )

        assert result is True
        # Check that temp directories were NOT created
        left_upscaled = directories["base"] / "07_left_upscaled"
        right_upscaled = directories["base"] / "07_right_upscaled"
        assert not left_upscaled.exists()
        assert not right_upscaled.exists()

    def test_upscale_frame_pair_success(
        self, temp_frame_dirs, mock_depth_estimator, mock_upscaler, mock_progress_tracker
    ):
        """Test upscaling a single frame pair."""
        processor = VideoProcessor(mock_depth_estimator)

        left_files = sorted(temp_frame_dirs["left"].glob("*.png"))
        right_files = sorted(temp_frame_dirs["right"].glob("*.png"))

        left_upscaled = temp_frame_dirs["base"] / "left_upscaled"
        right_upscaled = temp_frame_dirs["base"] / "right_upscaled"
        left_upscaled.mkdir()
        right_upscaled.mkdir()

        settings = {"keep_intermediates": True}

        processor._upscale_frame_pair(
            mock_upscaler,
            left_files[0],
            right_files[0],
            left_upscaled,
            right_upscaled,
            settings,
            frame_idx=0,
            total_frames=len(left_files),
            progress_tracker=mock_progress_tracker,
        )

        # Check that upscale_image was called
        assert mock_upscaler.upscale_image.call_count == 2  # left and right

        # Check that progress was updated
        mock_progress_tracker.update_progress.assert_called()

    def test_upscale_frame_pair_with_preview(
        self, temp_frame_dirs, mock_depth_estimator, mock_upscaler, mock_progress_tracker
    ):
        """Test upscaling with preview frame sending."""
        processor = VideoProcessor(mock_depth_estimator)

        left_files = sorted(temp_frame_dirs["left"].glob("*.png"))
        right_files = sorted(temp_frame_dirs["right"].glob("*.png"))

        left_upscaled = temp_frame_dirs["base"] / "left_upscaled"
        right_upscaled = temp_frame_dirs["base"] / "right_upscaled"
        left_upscaled.mkdir()
        right_upscaled.mkdir()

        settings = {"keep_intermediates": True}

        # Use frame_idx that triggers preview (divisible by PREVIEW_FRAME_SAMPLE_RATE=30)
        processor._upscale_frame_pair(
            mock_upscaler,
            left_files[0],
            right_files[0],
            left_upscaled,
            right_upscaled,
            settings,
            frame_idx=0,  # First frame should trigger preview
            total_frames=len(left_files),
            progress_tracker=mock_progress_tracker,
        )

        # Check that upscale_image was called
        assert mock_upscaler.upscale_image.call_count == 2

        # Check that preview was sent (frame_idx=0 should trigger it)
        if hasattr(mock_progress_tracker, "send_preview_frame"):
            mock_progress_tracker.send_preview_frame.assert_called()

    def test_upscale_frame_pair_missing_image(
        self, temp_frame_dirs, mock_depth_estimator, mock_upscaler, mock_progress_tracker
    ):
        """Test upscaling when image file is missing."""
        processor = VideoProcessor(mock_depth_estimator)

        left_upscaled = temp_frame_dirs["base"] / "left_upscaled"
        right_upscaled = temp_frame_dirs["base"] / "right_upscaled"
        left_upscaled.mkdir()
        right_upscaled.mkdir()

        settings = {"keep_intermediates": True}

        # Use non-existent files
        fake_left = temp_frame_dirs["left"] / "nonexistent_left.png"
        fake_right = temp_frame_dirs["right"] / "nonexistent_right.png"

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

    def test_upscale_frame_pair_no_intermediates_no_preview(
        self, temp_frame_dirs, mock_depth_estimator, mock_upscaler, mock_progress_tracker
    ):
        """Test that frames are not saved when keep_intermediates=False and no preview needed."""
        processor = VideoProcessor(mock_depth_estimator)

        left_files = sorted(temp_frame_dirs["left"].glob("*.png"))
        right_files = sorted(temp_frame_dirs["right"].glob("*.png"))

        left_upscaled = temp_frame_dirs["base"] / "left_upscaled"
        right_upscaled = temp_frame_dirs["base"] / "right_upscaled"
        left_upscaled.mkdir()
        right_upscaled.mkdir()

        settings = {"keep_intermediates": False}

        # Use frame_idx that doesn't trigger preview (not divisible by 30)
        processor._upscale_frame_pair(
            mock_upscaler,
            left_files[0],
            right_files[0],
            left_upscaled,
            right_upscaled,
            settings,
            frame_idx=1,  # Not first, not last, not divisible by 30
            total_frames=100,  # Large enough that frame 1 isn't last
            progress_tracker=mock_progress_tracker,
        )

        # Check that upscale_image was called
        assert mock_upscaler.upscale_image.call_count == 2

        # Check that no files were saved (since keep_intermediates=False and no preview)
        saved_files = list(left_upscaled.glob("*.png"))
        assert len(saved_files) == 0
