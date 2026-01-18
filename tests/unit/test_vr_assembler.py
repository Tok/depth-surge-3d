"""Tests for VRFrameAssembler module."""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch

from src.depth_surge_3d.processing.frames.vr_assembler import VRFrameAssembler


class TestVRFrameAssemblerInit:
    """Test VRFrameAssembler initialization."""

    def test_init_default(self):
        """Test default initialization."""
        assembler = VRFrameAssembler()
        assert assembler.verbose is False

    def test_init_verbose(self):
        """Test initialization with verbose."""
        assembler = VRFrameAssembler(verbose=True)
        assert assembler.verbose is True


class TestAssembleVRFrames:
    """Test assemble_vr_frames method."""

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create mock progress tracker."""
        tracker = Mock()
        tracker.update_progress = Mock()
        tracker.send_preview_frame = Mock()
        return tracker

    @pytest.fixture
    def temp_frames(self, tmp_path):
        """Create temporary frame directories."""
        left_dir = tmp_path / "left_cropped"
        right_dir = tmp_path / "right_cropped"
        vr_dir = tmp_path / "vr_frames"

        left_dir.mkdir()
        right_dir.mkdir()
        vr_dir.mkdir()

        # Create test frames
        for i in range(3):
            frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
            cv2.imwrite(str(left_dir / f"frame_{i:04d}.png"), frame)
            cv2.imwrite(str(right_dir / f"frame_{i:04d}.png"), frame)

        return {
            "left_cropped": left_dir,
            "right_cropped": right_dir,
            "vr_frames": vr_dir,
            "base": tmp_path,
        }

    def test_assemble_vr_frames_side_by_side(self, temp_frames, mock_progress_tracker):
        """Test side-by-side VR frame assembly."""
        assembler = VRFrameAssembler()

        settings = {
            "vr_format": "side_by_side",
            "keep_intermediates": True,
            "per_eye_width": 200,
            "per_eye_height": 100,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.vr_assembler.create_vr_frame"
        ) as mock_create:
            mock_create.return_value = np.zeros((100, 400, 3), dtype=np.uint8)

            result = assembler.assemble_vr_frames(temp_frames, settings, mock_progress_tracker)

        assert result is True
        assert mock_create.call_count == 3

        # Check output files
        vr_files = list(temp_frames["vr_frames"].glob("*.png"))
        assert len(vr_files) == 3

    def test_assemble_vr_frames_over_under(self, temp_frames, mock_progress_tracker):
        """Test over-under VR frame assembly."""
        assembler = VRFrameAssembler()

        settings = {
            "vr_format": "over_under",
            "keep_intermediates": True,
            "per_eye_width": 200,
            "per_eye_height": 100,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.vr_assembler.create_vr_frame"
        ) as mock_create:
            mock_create.return_value = np.zeros((200, 200, 3), dtype=np.uint8)

            result = assembler.assemble_vr_frames(temp_frames, settings, mock_progress_tracker)

        assert result is True

    def test_assemble_vr_frames_without_intermediates(self, temp_frames, mock_progress_tracker):
        """Test VR assembly without saving intermediates."""
        assembler = VRFrameAssembler()

        settings = {
            "vr_format": "side_by_side",
            "keep_intermediates": False,
            "per_eye_width": 200,
            "per_eye_height": 100,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.vr_assembler.create_vr_frame"
        ) as mock_create:
            mock_create.return_value = np.zeros((100, 400, 3), dtype=np.uint8)

            result = assembler.assemble_vr_frames(temp_frames, settings, mock_progress_tracker)

        assert result is True

        # VR frames are still saved even with keep_intermediates=False
        # because the VR frames directory is in temp_frames dict
        vr_files = list(temp_frames["vr_frames"].glob("*.png"))
        assert len(vr_files) == 3

    def test_assemble_vr_frames_mismatched_count(self, temp_frames, mock_progress_tracker):
        """Test VR assembly with mismatched frame counts."""
        assembler = VRFrameAssembler()

        # Remove one right frame
        right_frames = list(temp_frames["right_cropped"].glob("*.png"))
        right_frames[0].unlink()

        settings = {
            "vr_format": "side_by_side",
            "keep_intermediates": True,
            "per_eye_width": 200,
            "per_eye_height": 100,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.vr_assembler.create_vr_frame"
        ) as mock_create:
            mock_create.return_value = np.zeros((100, 400, 3), dtype=np.uint8)

            result = assembler.assemble_vr_frames(temp_frames, settings, mock_progress_tracker)

        # Should still succeed, just processes matched pairs (2 remaining)
        assert result is True
        assert mock_create.call_count == 2

    def test_assemble_vr_frames_no_source(self, mock_progress_tracker, tmp_path):
        """Test VR assembly with no source frames."""
        assembler = VRFrameAssembler()

        directories = {"base": tmp_path}
        settings = {"vr_format": "side_by_side", "keep_intermediates": True}

        result = assembler.assemble_vr_frames(directories, settings, mock_progress_tracker)

        assert result is False

    def test_assemble_vr_frames_exception_handling(self, temp_frames, mock_progress_tracker):
        """Test exception handling during VR assembly."""
        assembler = VRFrameAssembler()

        settings = {
            "vr_format": "side_by_side",
            "keep_intermediates": True,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.vr_assembler.create_vr_frame",
            side_effect=RuntimeError("Test error"),
        ):
            result = assembler.assemble_vr_frames(temp_frames, settings, mock_progress_tracker)

        assert result is False


class TestGetVRAssemblySourceDirs:
    """Test _get_vr_assembly_source_dirs static method."""

    def test_get_source_with_upscaled(self, tmp_path):
        """Test getting source with upscaled frames."""
        left_upscaled = tmp_path / "left_upscaled"
        right_upscaled = tmp_path / "right_upscaled"
        left_upscaled.mkdir()
        right_upscaled.mkdir()

        # Create frame
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(left_upscaled / "frame_0000.png"), frame)

        directories = {
            "left_upscaled": left_upscaled,
            "right_upscaled": right_upscaled,
        }

        # Must set upscale_model to something other than "none" for upscaled frames to be used
        settings = {"upscale_model": "x4"}

        result = VRFrameAssembler._get_vr_assembly_source_dirs(directories, settings)

        assert result == (left_upscaled, right_upscaled)

    def test_get_source_with_cropped(self, tmp_path):
        """Test getting source with cropped frames."""
        left_cropped = tmp_path / "left_cropped"
        right_cropped = tmp_path / "right_cropped"
        left_cropped.mkdir()
        right_cropped.mkdir()

        # Create frame
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(left_cropped / "frame_0000.png"), frame)

        directories = {
            "left_cropped": left_cropped,
            "right_cropped": right_cropped,
        }

        settings = {}

        result = VRFrameAssembler._get_vr_assembly_source_dirs(directories, settings)

        assert result == (left_cropped, right_cropped)

    def test_get_source_no_directories(self):
        """Test handling when no source directories exist."""
        directories = {}
        settings = {}

        result = VRFrameAssembler._get_vr_assembly_source_dirs(directories, settings)

        assert result is None
