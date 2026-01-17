"""Unit tests for batch_analysis.py"""

from pathlib import Path
from unittest.mock import patch, MagicMock
from src.depth_surge_3d.utils.batch_analysis import (
    _get_stage_number,
    _detect_highest_stage,
    _detect_vr_format_and_resolution,
    _load_settings_summary,
    _detect_audio_availability,
    analyze_batch_directory,
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
