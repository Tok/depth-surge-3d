"""Unit tests for batch_analysis.py"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.depth_surge_3d.utils.batch_analysis import (
    _get_cv2,
    _get_stage_number,
    _detect_highest_stage,
    _detect_vr_format_and_resolution,
    _load_settings_summary,
    _detect_audio_availability,
    _summarize_settings,
    analyze_batch_directory,
    create_video_from_batch,
)


class TestGetStageNumber:
    """Test _get_stage_number function."""

    def test_get_stage_number_with_prefix(self):
        """Test extracting stage number from directory with prefix."""
        result = _get_stage_number("99_vr_frames")
        assert result == 99

    def test_get_stage_number_with_two_digit(self):
        """Test extracting two-digit stage number."""
        result = _get_stage_number("04_left_frames")
        assert result == 4

    def test_get_stage_number_with_single_digit(self):
        """Test extracting single-digit stage number."""
        result = _get_stage_number("2_depth_maps")
        assert result == 2

    def test_get_stage_number_invalid_format(self):
        """Test invalid format returns 0."""
        result = _get_stage_number("invalid_format")
        assert result == 0

    def test_get_stage_number_no_number(self):
        """Test directory with no number returns 0."""
        result = _get_stage_number("frames")
        assert result == 0


class TestDetectHighestStage:
    """Test _detect_highest_stage function."""

    def test_detect_highest_stage_no_frames(self):
        """Test detecting when no frames exist."""
        mock_batch_path = MagicMock(spec=Path)

        mock_dir = MagicMock(spec=Path)
        mock_dir.exists.return_value = False

        mock_batch_path.__truediv__ = MagicMock(return_value=mock_dir)

        stages = {"99_vr_frames": "VR frames"}

        stage_num, stage_name, frame_count = _detect_highest_stage(mock_batch_path, stages)

        assert stage_num == 0
        assert stage_name == "none"
        assert frame_count == 0


class TestDetectVRFormatAndResolution:
    """Test _detect_vr_format_and_resolution function."""

    def test_detect_vr_format_low_stage(self):
        """Test detection returns unknown for low stage number."""
        mock_batch_path = MagicMock(spec=Path)

        vr_format, resolution = _detect_vr_format_and_resolution(mock_batch_path, 10)

        assert vr_format == "unknown"
        assert resolution == "unknown"


class TestLoadSettingsSummary:
    """Test _load_settings_summary function."""

    def test_load_settings_summary_success(self):
        """Test loading settings summary successfully."""
        mock_batch_path = MagicMock(spec=Path)
        mock_settings_file = MagicMock(spec=Path)
        mock_batch_path.glob.return_value = [mock_settings_file]

        settings_data = {
            "vr_format": "side_by_side",
            "vr_resolution": "1080p",
        }

        with patch("builtins.open", MagicMock()):
            with patch("json.load", return_value=settings_data):
                result = _load_settings_summary(mock_batch_path)

        assert "side_by_side" in result
        assert "1080p" in result

    def test_load_settings_summary_no_file(self):
        """Test when no settings file exists."""
        mock_batch_path = MagicMock(spec=Path)
        mock_batch_path.glob.return_value = []

        result = _load_settings_summary(mock_batch_path)

        assert result == "unknown"

    def test_load_settings_summary_exception(self):
        """Test handling of exception during load."""
        mock_batch_path = MagicMock(spec=Path)
        mock_settings_file = MagicMock(spec=Path)
        mock_batch_path.glob.return_value = [mock_settings_file]

        with patch("builtins.open", side_effect=OSError("Read error")):
            result = _load_settings_summary(mock_batch_path)

        assert result == "unknown"


class TestDetectAudioAvailability:
    """Test _detect_audio_availability function."""

    def test_detect_audio_not_available(self):
        """Test when no audio files exist."""
        mock_batch_path = MagicMock(spec=Path)
        mock_parent = MagicMock(spec=Path)
        mock_parent.parent.glob.return_value = []
        mock_batch_path.parent = mock_parent

        result = _detect_audio_availability(mock_batch_path)

        assert result is False


class TestAnalyzeBatchDirectory:
    """Test analyze_batch_directory function."""

    def test_analyze_batch_directory_success(self):
        """Test analyzing batch directory successfully."""
        mock_batch_path = Path("/tmp/batch")

        with patch(
            "src.depth_surge_3d.utils.batch_analysis._detect_highest_stage",
            return_value=(99, "VR frames", 50),
        ):
            with patch(
                "src.depth_surge_3d.utils.batch_analysis._detect_vr_format_and_resolution",
                return_value=("side_by_side", "3840x1080"),
            ):
                with patch(
                    "src.depth_surge_3d.utils.batch_analysis._load_settings_summary",
                    return_value="Settings loaded",
                ):
                    with patch(
                        "src.depth_surge_3d.utils.batch_analysis._detect_audio_availability",
                        return_value=True,
                    ):
                        result = analyze_batch_directory(mock_batch_path)

        assert result["frame_count"] == 50
        assert result["vr_format"] == "side_by_side"
        assert result["resolution"] == "3840x1080"
        assert result["highest_stage"] == "VR frames"
        assert result["has_audio"] is True
        assert result["settings_summary"] == "Settings loaded"


class TestGetCv2:
    """Test _get_cv2 function."""

    def test_get_cv2_import_error(self):
        """Test cv2 import error handling."""
        with patch.dict("sys.modules", {"cv2": None}):
            with pytest.raises(ImportError, match="opencv-python is required"):
                _get_cv2()


class TestDetectHighestStageWithFrames:
    """Test _detect_highest_stage with actual frames."""

    def test_detect_highest_stage_with_frames(self):
        """Test detecting highest stage when frames exist."""
        mock_batch_path = MagicMock(spec=Path)

        # Mock VR frames directory with 100 frames
        mock_vr_dir = MagicMock(spec=Path)
        mock_vr_dir.exists.return_value = True
        mock_vr_frames = [MagicMock(spec=Path) for _ in range(100)]
        mock_vr_dir.glob.return_value = mock_vr_frames

        # Mock depth maps directory with 50 frames (lower stage)
        mock_depth_dir = MagicMock(spec=Path)
        mock_depth_dir.exists.return_value = True
        mock_depth_frames = [MagicMock(spec=Path) for _ in range(50)]
        mock_depth_dir.glob.return_value = mock_depth_frames

        def truediv_handler(dirname):
            if "99_vr_frames" in str(dirname):
                return mock_vr_dir
            elif "02_depth_maps" in str(dirname):
                return mock_depth_dir
            return MagicMock(spec=Path, exists=lambda: False)

        mock_batch_path.__truediv__.side_effect = truediv_handler

        stages = {"99_vr_frames": "VR frames", "02_depth_maps": "Depth maps"}

        stage_num, stage_name, frame_count = _detect_highest_stage(mock_batch_path, stages)

        assert stage_num == 99
        assert stage_name == "VR frames"
        assert frame_count == 100


class TestDetectVRFormatAndResolutionWithFrames:
    """Test _detect_vr_format_and_resolution with actual frames."""

    @patch("src.depth_surge_3d.utils.batch_analysis._get_cv2")
    def test_detect_vr_format_side_by_side(self, mock_get_cv2):
        """Test detecting side-by-side VR format."""
        import numpy as np
        from src.depth_surge_3d.core.constants import INTERMEDIATE_DIRS

        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_batch_path = MagicMock(spec=Path)

        # Mock VR frames directory
        mock_vr_dir = MagicMock(spec=Path)
        mock_vr_dir.exists.return_value = True

        mock_frame = MagicMock(spec=Path)
        mock_vr_dir.glob.return_value = [mock_frame]

        # Mock __truediv__ to return the VR directory for the correct path
        def truediv_handler(dirname):
            if dirname == INTERMEDIATE_DIRS["vr_frames"]:
                return mock_vr_dir
            mock_other = MagicMock(spec=Path)
            mock_other.exists.return_value = False
            return mock_other

        mock_batch_path.__truediv__.side_effect = truediv_handler

        # Mock side-by-side image (width > height * 1.5)
        mock_image = np.zeros((1080, 3840, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = mock_image

        vr_format, resolution = _detect_vr_format_and_resolution(mock_batch_path, 99)

        assert vr_format == "side_by_side"
        assert resolution == "3840x1080"

    @patch("src.depth_surge_3d.utils.batch_analysis._get_cv2")
    def test_detect_vr_format_over_under(self, mock_get_cv2):
        """Test detecting over-under VR format."""
        import numpy as np
        from src.depth_surge_3d.core.constants import INTERMEDIATE_DIRS

        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_batch_path = MagicMock(spec=Path)

        mock_vr_dir = MagicMock(spec=Path)
        mock_vr_dir.exists.return_value = True

        mock_frame = MagicMock(spec=Path)
        mock_vr_dir.glob.return_value = [mock_frame]

        # Mock __truediv__ to return the VR directory for the correct path
        def truediv_handler(dirname):
            if dirname == INTERMEDIATE_DIRS["vr_frames"]:
                return mock_vr_dir
            mock_other = MagicMock(spec=Path)
            mock_other.exists.return_value = False
            return mock_other

        mock_batch_path.__truediv__.side_effect = truediv_handler

        # Mock over-under image (width < height * 1.5)
        mock_image = np.zeros((2160, 1920, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = mock_image

        vr_format, resolution = _detect_vr_format_and_resolution(mock_batch_path, 99)

        assert vr_format == "over_under"
        assert resolution == "1920x2160"

    @patch("src.depth_surge_3d.utils.batch_analysis._get_cv2")
    def test_detect_vr_format_exception(self, mock_get_cv2):
        """Test handling exception during format detection."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2
        mock_cv2.imread.side_effect = Exception("Read error")

        mock_batch_path = MagicMock(spec=Path)

        mock_vr_dir = MagicMock(spec=Path)
        mock_vr_dir.exists.return_value = True
        mock_vr_dir.glob.return_value = [MagicMock(spec=Path)]

        mock_batch_path.__truediv__.return_value = mock_vr_dir

        vr_format, resolution = _detect_vr_format_and_resolution(mock_batch_path, 99)

        assert vr_format == "unknown"
        assert resolution == "unknown"


class TestDetectAudioAvailabilityWithAudio:
    """Test _detect_audio_availability with audio files."""

    def test_detect_audio_available(self):
        """Test when audio files exist."""
        mock_batch_path = MagicMock(spec=Path)
        mock_parent = MagicMock(spec=Path)

        # Mock finding an mp4 file
        mock_video = MagicMock(spec=Path)
        mock_parent.parent.glob.return_value = [mock_video]
        mock_batch_path.parent = mock_parent

        result = _detect_audio_availability(mock_batch_path)

        assert result is True


class TestSummarizeSettings:
    """Test _summarize_settings function."""

    def test_summarize_settings_full(self):
        """Test summary with all settings."""
        settings = {
            "vr_format": "side_by_side",
            "vr_resolution": "3840x1080",
            "processing_mode": "batch",
            "super_sample": "2x",
            "fisheye_enabled": True,
        }

        result = _summarize_settings(settings)

        assert "side_by_side" in result
        assert "3840x1080" in result
        assert "batch" in result
        assert "2x" in result
        assert "Fisheye: enabled" in result

    def test_summarize_settings_minimal(self):
        """Test summary with minimal settings."""
        settings = {"vr_format": "over_under"}

        result = _summarize_settings(settings)

        assert "over_under" in result
        assert "Resolution" not in result

    def test_summarize_settings_empty(self):
        """Test summary with empty settings."""
        settings = {}

        result = _summarize_settings(settings)

        assert result == "Standard processing"

    def test_summarize_settings_super_sample_none(self):
        """Test that super_sample='none' is not included."""
        settings = {"super_sample": "none", "vr_format": "side_by_side"}

        result = _summarize_settings(settings)

        assert "Super-sample" not in result
        assert "side_by_side" in result

    def test_summarize_settings_fisheye_disabled(self):
        """Test that disabled fisheye is not included."""
        settings = {"fisheye_enabled": False, "vr_format": "side_by_side"}

        result = _summarize_settings(settings)

        assert "Fisheye" not in result
        assert "side_by_side" in result


class TestCreateVideoFromBatch:
    """Test create_video_from_batch function."""

    @patch("subprocess.run")
    def test_create_video_auto_frame_source(self, mock_run):
        """Test creating video with auto frame source detection."""
        mock_batch_path = Path("/tmp/batch")

        # Mock VR frames directory with frames
        mock_vr_dir = MagicMock(spec=Path)
        mock_vr_dir.exists.return_value = True
        mock_vr_dir.glob.return_value = [MagicMock(spec=Path)]

        settings = {"frame_source": "auto", "quality": "medium", "fps": 30}

        # Mock successful subprocess run
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "glob") as mock_glob:
                mock_glob.return_value = [MagicMock(spec=Path)]

                result = create_video_from_batch(mock_batch_path, settings)

        assert result is not None
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_create_video_specific_frame_source(self, mock_run):
        """Test creating video with specific frame source."""
        mock_batch_path = Path("/tmp/batch")

        settings = {
            "frame_source": "left_right_basic",
            "quality": "high",
            "fps": "original",
            "output_filename": "test_output.mp4",
        }

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        with patch.object(Path, "exists", return_value=True):
            result = create_video_from_batch(mock_batch_path, settings)

        assert result is not None
        # Check that high quality settings were used
        call_args = mock_run.call_args[0][0]
        assert "-crf" in call_args
        assert "18" in call_args

    @patch("subprocess.run")
    def test_create_video_ffmpeg_error(self, mock_run):
        """Test handling of FFmpeg error."""
        mock_batch_path = Path("/tmp/batch")

        settings = {"frame_source": "auto", "quality": "low", "fps": 24}

        # Mock failed subprocess run
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "FFmpeg error output"
        mock_run.return_value = mock_result

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "glob", return_value=[MagicMock(spec=Path)]):
                result = create_video_from_batch(mock_batch_path, settings)

        assert result is None

    def test_create_video_no_frames_found(self):
        """Test error when no frames found."""
        mock_batch_path = Path("/tmp/batch")

        settings = {"frame_source": "auto", "quality": "medium", "fps": 30}

        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(ValueError, match="No frames found"):
                create_video_from_batch(mock_batch_path, settings)

    @patch("subprocess.run")
    def test_create_video_lossless_quality(self, mock_run):
        """Test creating video with lossless quality."""
        mock_batch_path = Path("/tmp/batch")

        settings = {"frame_source": "vr_frames", "quality": "lossless", "fps": 60}

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        with patch.object(Path, "exists", return_value=True):
            result = create_video_from_batch(mock_batch_path, settings)

        assert result is not None
        # Check that lossless settings were used (crf 0)
        call_args = mock_run.call_args[0][0]
        assert "-crf" in call_args
        assert "0" in call_args
