"""Tests for DistortionProcessor module."""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch

from src.depth_surge_3d.processing.frames.distortion_processor import DistortionProcessor


class TestDistortionProcessorInit:
    """Test DistortionProcessor initialization."""

    def test_init_default(self):
        """Test default initialization."""
        processor = DistortionProcessor()
        assert processor.verbose is False

    def test_init_verbose(self):
        """Test initialization with verbose enabled."""
        processor = DistortionProcessor(verbose=True)
        assert processor.verbose is True


class TestApplyDistortion:
    """Test apply_distortion method."""

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create mock progress tracker."""
        tracker = Mock()
        tracker.update_progress = Mock()
        return tracker

    @pytest.fixture
    def temp_frames(self, tmp_path):
        """Create temporary frame files."""
        left_dir = tmp_path / "left"
        right_dir = tmp_path / "right"
        left_dir.mkdir()
        right_dir.mkdir()

        left_files = []
        right_files = []

        for i in range(3):
            left_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            right_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

            left_path = left_dir / f"frame_{i:04d}.png"
            right_path = right_dir / f"frame_{i:04d}.png"

            cv2.imwrite(str(left_path), left_frame)
            cv2.imwrite(str(right_path), right_frame)

            left_files.append(left_path)
            right_files.append(right_path)

        return {
            "left_files": left_files,
            "right_files": right_files,
            "base": tmp_path,
        }

    def test_apply_distortion_success(self, temp_frames, mock_progress_tracker, tmp_path):
        """Test successful distortion application."""
        processor = DistortionProcessor()

        left_distorted = tmp_path / "left_distorted"
        right_distorted = tmp_path / "right_distorted"
        left_distorted.mkdir()
        right_distorted.mkdir()

        directories = {
            "left_distorted": left_distorted,
            "right_distorted": right_distorted,
        }

        settings = {
            "keep_intermediates": True,
            "fisheye_fov": 90,
            "fisheye_projection": "equidistant",
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.apply_fisheye_distortion",
            side_effect=lambda img, fov, proj: img,  # Pass through
        ):
            result = processor.apply_distortion(
                temp_frames["left_files"],
                temp_frames["right_files"],
                directories,
                settings,
                mock_progress_tracker,
            )

        assert result is True

        # Check output files were created
        assert len(list(left_distorted.glob("*.png"))) == 3
        assert len(list(right_distorted.glob("*.png"))) == 3

    def test_apply_distortion_without_intermediates(
        self, temp_frames, mock_progress_tracker, tmp_path
    ):
        """Test distortion without saving intermediates."""
        processor = DistortionProcessor()

        directories = {}

        settings = {
            "keep_intermediates": False,
            "fisheye_fov": 90,
            "fisheye_projection": "equidistant",
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.apply_fisheye_distortion",
            side_effect=lambda img, fov, proj: img,
        ):
            result = processor.apply_distortion(
                temp_frames["left_files"],
                temp_frames["right_files"],
                directories,
                settings,
                mock_progress_tracker,
            )

        assert result is True

    def test_apply_distortion_missing_file(self, mock_progress_tracker, tmp_path):
        """Test distortion with missing frame file."""
        processor = DistortionProcessor()

        fake_left = [tmp_path / "nonexistent_left.png"]
        fake_right = [tmp_path / "nonexistent_right.png"]

        directories = {}
        settings = {
            "keep_intermediates": False,
            "fisheye_fov": 90,
            "fisheye_projection": "equidistant",
        }

        result = processor.apply_distortion(
            fake_left, fake_right, directories, settings, mock_progress_tracker
        )

        # Should still return True (just skips bad frames with warning)
        assert result is True

    def test_apply_distortion_exception_handling(self, temp_frames, mock_progress_tracker):
        """Test exception handling during distortion."""
        processor = DistortionProcessor()

        directories = {}
        settings = {
            "keep_intermediates": False,
            "fisheye_fov": 90,
            "fisheye_projection": "equidistant",
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.apply_fisheye_distortion",
            side_effect=RuntimeError("GPU error"),
        ):
            result = processor.apply_distortion(
                temp_frames["left_files"],
                temp_frames["right_files"],
                directories,
                settings,
                mock_progress_tracker,
            )

        assert result is False


class TestCropFrames:
    """Test crop_frames method."""

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create mock progress tracker."""
        tracker = Mock()
        tracker.update_progress = Mock()
        return tracker

    @pytest.fixture
    def temp_frames(self, tmp_path):
        """Create temporary frame directories with test frames."""
        left_dir = tmp_path / "left_frames"
        right_dir = tmp_path / "right_frames"
        left_dir.mkdir()
        right_dir.mkdir()

        # Create test frames (large enough for cropping)
        for i in range(3):
            left_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            right_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            cv2.imwrite(str(left_dir / f"frame_{i:04d}.png"), left_frame)
            cv2.imwrite(str(right_dir / f"frame_{i:04d}.png"), right_frame)

        left_cropped = tmp_path / "left_cropped"
        right_cropped = tmp_path / "right_cropped"
        left_cropped.mkdir()
        right_cropped.mkdir()

        return {
            "base": tmp_path,
            "left_frames": left_dir,
            "right_frames": right_dir,
            "left_cropped": left_cropped,
            "right_cropped": right_cropped,
        }

    def test_crop_frames_success(self, temp_frames, mock_progress_tracker):
        """Test successful frame cropping."""
        processor = DistortionProcessor()

        settings = {
            "apply_distortion": False,
            "per_eye_width": 1920,
            "per_eye_height": 1080,
            "crop_factor": 1.0,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.apply_center_crop",
            side_effect=lambda img, factor: img,
        ):
            result = processor.crop_frames(
                temp_frames, settings, mock_progress_tracker, total_frames=3
            )

        assert result is True

        # Check output files were created
        assert len(list(temp_frames["left_cropped"].glob("*.png"))) == 3
        assert len(list(temp_frames["right_cropped"].glob("*.png"))) == 3

    def test_crop_frames_with_distortion(self, temp_frames, mock_progress_tracker, tmp_path):
        """Test cropping with fisheye distortion enabled."""
        processor = DistortionProcessor()

        # Create distorted directories
        left_distorted = tmp_path / "left_distorted"
        right_distorted = tmp_path / "right_distorted"
        left_distorted.mkdir()
        right_distorted.mkdir()

        # Add distorted frames
        for i in range(3):
            frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            cv2.imwrite(str(left_distorted / f"frame_{i:04d}.png"), frame)
            cv2.imwrite(str(right_distorted / f"frame_{i:04d}.png"), frame)

        directories = {
            **temp_frames,
            "left_distorted": left_distorted,
            "right_distorted": right_distorted,
        }

        settings = {
            "apply_distortion": True,
            "per_eye_width": 1920,
            "per_eye_height": 1080,
            "fisheye_crop_factor": 0.8,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.apply_fisheye_square_crop",
            side_effect=lambda img, w, h, factor: img,
        ):
            result = processor.crop_frames(
                directories, settings, mock_progress_tracker, total_frames=3
            )

        assert result is True

    def test_crop_frames_mismatched_count(self, temp_frames, mock_progress_tracker):
        """Test cropping with mismatched frame counts."""
        processor = DistortionProcessor()

        # Remove one right frame
        right_frames = list(temp_frames["right_frames"].glob("*.png"))
        right_frames[0].unlink()

        settings = {
            "apply_distortion": False,
            "per_eye_width": 1920,
            "per_eye_height": 1080,
            "crop_factor": 1.0,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.apply_center_crop",
            side_effect=lambda img, factor: img,
        ):
            result = processor.crop_frames(
                temp_frames, settings, mock_progress_tracker, total_frames=3
            )

        # Should still succeed, just processes matched pairs
        assert result is True

    def test_crop_frames_no_source(self, mock_progress_tracker, tmp_path):
        """Test cropping with no source directories."""
        processor = DistortionProcessor()

        directories = {"base": tmp_path}
        settings = {"apply_distortion": False}

        result = processor.crop_frames(directories, settings, mock_progress_tracker, total_frames=0)

        assert result is False

    def test_crop_frames_exception_handling(self, temp_frames, mock_progress_tracker):
        """Test exception handling during cropping."""
        processor = DistortionProcessor()

        settings = {
            "apply_distortion": False,
            "per_eye_width": 1920,
            "per_eye_height": 1080,
            "crop_factor": 1.0,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.apply_center_crop",
            side_effect=RuntimeError("Crop error"),
        ):
            result = processor.crop_frames(
                temp_frames, settings, mock_progress_tracker, total_frames=3
            )

        assert result is False


class TestCropSingleFramePair:
    """Test _crop_single_frame_pair method."""

    @pytest.fixture
    def temp_frames(self, tmp_path):
        """Create temporary frame files."""
        left_dir = tmp_path / "left"
        right_dir = tmp_path / "right"
        left_dir.mkdir()
        right_dir.mkdir()

        left_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        right_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        left_path = left_dir / "frame_0000.png"
        right_path = right_dir / "frame_0000.png"

        cv2.imwrite(str(left_path), left_frame)
        cv2.imwrite(str(right_path), right_frame)

        left_cropped = tmp_path / "left_cropped"
        right_cropped = tmp_path / "right_cropped"
        left_cropped.mkdir()
        right_cropped.mkdir()

        return {
            "left_path": left_path,
            "right_path": right_path,
            "left_cropped": left_cropped,
            "right_cropped": right_cropped,
        }

    def test_crop_single_pair_without_distortion(self, temp_frames):
        """Test cropping single pair without distortion."""
        processor = DistortionProcessor()

        directories = {
            "left_cropped": temp_frames["left_cropped"],
            "right_cropped": temp_frames["right_cropped"],
        }

        settings = {
            "apply_distortion": False,
            "per_eye_width": 1920,
            "per_eye_height": 1080,
            "crop_factor": 0.9,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.apply_center_crop",
            side_effect=lambda img, factor: img,
        ):
            result = processor._crop_single_frame_pair(
                temp_frames["left_path"],
                temp_frames["right_path"],
                directories,
                settings,
            )

        assert result is True
        assert (temp_frames["left_cropped"] / "frame_0000.png").exists()
        assert (temp_frames["right_cropped"] / "frame_0000.png").exists()

    def test_crop_single_pair_with_distortion(self, temp_frames):
        """Test cropping single pair with distortion."""
        processor = DistortionProcessor()

        directories = {
            "left_cropped": temp_frames["left_cropped"],
            "right_cropped": temp_frames["right_cropped"],
        }

        settings = {
            "apply_distortion": True,
            "per_eye_width": 1920,
            "per_eye_height": 1080,
            "fisheye_crop_factor": 0.8,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.apply_fisheye_square_crop",
            side_effect=lambda img, w, h, factor: img,
        ):
            result = processor._crop_single_frame_pair(
                temp_frames["left_path"],
                temp_frames["right_path"],
                directories,
                settings,
            )

        assert result is True

    def test_crop_single_pair_missing_file(self, tmp_path):
        """Test cropping with missing file."""
        processor = DistortionProcessor()

        fake_left = tmp_path / "nonexistent_left.png"
        fake_right = tmp_path / "nonexistent_right.png"

        directories = {}
        settings = {"apply_distortion": False}

        result = processor._crop_single_frame_pair(fake_left, fake_right, directories, settings)

        assert result is False

    def test_crop_single_pair_crop_factor_clamping(self, temp_frames):
        """Test that crop factor is clamped to valid range."""
        processor = DistortionProcessor()

        directories = {
            "left_cropped": temp_frames["left_cropped"],
            "right_cropped": temp_frames["right_cropped"],
        }

        # Test with out-of-range crop factor
        settings = {
            "apply_distortion": False,
            "per_eye_width": 1920,
            "per_eye_height": 1080,
            "crop_factor": 5.0,  # Will be clamped to 1.0
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.apply_center_crop",
            side_effect=lambda img, factor: img,
        ) as mock_crop:
            processor._crop_single_frame_pair(
                temp_frames["left_path"],
                temp_frames["right_path"],
                directories,
                settings,
            )

            # Check that clamped value was used (1.0 max for non-distortion)
            call_args = mock_crop.call_args_list[0][0]
            assert call_args[1] == 1.0  # Clamped from 5.0 to 1.0

    def test_crop_single_pair_fisheye_factor_clamping(self, temp_frames):
        """Test that fisheye crop factor is clamped to valid range."""
        processor = DistortionProcessor()

        directories = {
            "left_cropped": temp_frames["left_cropped"],
            "right_cropped": temp_frames["right_cropped"],
        }

        # Test with out-of-range fisheye crop factor
        settings = {
            "apply_distortion": True,
            "per_eye_width": 1920,
            "per_eye_height": 1080,
            "fisheye_crop_factor": 3.0,  # Will be clamped to 2.0
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.apply_fisheye_square_crop",
            side_effect=lambda img, w, h, factor: img,
        ) as mock_crop:
            processor._crop_single_frame_pair(
                temp_frames["left_path"],
                temp_frames["right_path"],
                directories,
                settings,
            )

            # Check that clamped value was used (2.0 max for fisheye)
            call_args = mock_crop.call_args_list[0][0]
            assert call_args[3] == 2.0  # Clamped from 3.0 to 2.0


class TestGetStereoSourceDirs:
    """Test _get_stereo_source_dirs static method."""

    def test_get_source_with_distortion(self, tmp_path):
        """Test getting source directories when distortion is enabled."""
        left_distorted = tmp_path / "left_distorted"
        right_distorted = tmp_path / "right_distorted"
        left_distorted.mkdir()
        right_distorted.mkdir()

        # Create frame in distorted directory
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(left_distorted / "frame_0000.png"), frame)

        directories = {
            "left_distorted": left_distorted,
            "right_distorted": right_distorted,
        }

        settings = {"apply_distortion": True}

        result = DistortionProcessor._get_stereo_source_dirs(directories, settings)

        assert result == (left_distorted, right_distorted)

    def test_get_source_without_distortion(self, tmp_path):
        """Test getting source directories when distortion is disabled."""
        left_frames = tmp_path / "left_frames"
        right_frames = tmp_path / "right_frames"
        left_frames.mkdir()
        right_frames.mkdir()

        # Create frame in frames directory
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(left_frames / "frame_0000.png"), frame)

        directories = {
            "left_frames": left_frames,
            "right_frames": right_frames,
        }

        settings = {"apply_distortion": False}

        result = DistortionProcessor._get_stereo_source_dirs(directories, settings)

        assert result == (left_frames, right_frames)

    def test_get_source_fallback_to_frames(self, tmp_path):
        """Test fallback to frames when distorted not available."""
        left_frames = tmp_path / "left_frames"
        right_frames = tmp_path / "right_frames"
        left_frames.mkdir()
        right_frames.mkdir()

        # Create frame in frames directory
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(left_frames / "frame_0000.png"), frame)

        directories = {
            "left_frames": left_frames,
            "right_frames": right_frames,
        }

        # Request distortion but directories don't exist
        settings = {"apply_distortion": True}

        result = DistortionProcessor._get_stereo_source_dirs(directories, settings)

        # Should fallback to frames
        assert result == (left_frames, right_frames)

    def test_get_source_empty_distorted_dirs(self, tmp_path):
        """Test handling empty distorted directories."""
        left_distorted = tmp_path / "left_distorted"
        right_distorted = tmp_path / "right_distorted"
        left_frames = tmp_path / "left_frames"
        right_frames = tmp_path / "right_frames"

        left_distorted.mkdir()
        right_distorted.mkdir()
        left_frames.mkdir()
        right_frames.mkdir()

        # Create frame only in frames directory (distorted is empty)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(left_frames / "frame_0000.png"), frame)

        directories = {
            "left_distorted": left_distorted,
            "right_distorted": right_distorted,
            "left_frames": left_frames,
            "right_frames": right_frames,
        }

        settings = {"apply_distortion": True}

        result = DistortionProcessor._get_stereo_source_dirs(directories, settings)

        # Should fallback to frames since distorted is empty
        assert result == (left_frames, right_frames)

    def test_get_source_no_directories(self):
        """Test handling when no source directories exist."""
        directories = {}
        settings = {"apply_distortion": False}

        result = DistortionProcessor._get_stereo_source_dirs(directories, settings)

        assert result is None
