"""Unit tests for io_operations.py pure helper functions."""

from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from src.depth_surge_3d.processing.io_operations import (
    _should_keep_file,
    _remove_file_safe,
    get_frame_files,
    create_output_directories,
)


class TestShouldKeepFile:
    """Test _should_keep_file pattern matching."""

    def test_should_keep_file_matches_pattern(self):
        """Test file matches a keep pattern."""
        file_path = Path("output/frames/frame_001.png")
        keep_patterns = ["*.png", "*.jpg"]

        result = _should_keep_file(file_path, keep_patterns)

        assert result is True

    def test_should_keep_file_matches_wildcard_pattern(self):
        """Test file matches wildcard pattern."""
        file_path = Path("output/frames/frame_001.png")
        keep_patterns = ["frame_*"]

        result = _should_keep_file(file_path, keep_patterns)

        assert result is True

    def test_should_keep_file_no_match(self):
        """Test file doesn't match any pattern."""
        file_path = Path("output/temp/temp_file.tmp")
        keep_patterns = ["*.png", "*.jpg", "frame_*"]

        result = _should_keep_file(file_path, keep_patterns)

        assert result is False

    def test_should_keep_file_empty_patterns(self):
        """Test with empty keep patterns list."""
        file_path = Path("output/any_file.txt")
        keep_patterns = []

        result = _should_keep_file(file_path, keep_patterns)

        assert result is False

    def test_should_keep_file_matches_first_pattern(self):
        """Test file matches first pattern in list."""
        file_path = Path("settings.json")
        keep_patterns = ["settings.json", "*.mp4", "*.png"]

        result = _should_keep_file(file_path, keep_patterns)

        assert result is True


class TestRemoveFileSafe:
    """Test _remove_file_safe error handling."""

    def test_remove_file_safe_success(self):
        """Test successful file removal."""
        mock_path = MagicMock(spec=Path)
        mock_path.unlink = MagicMock()

        result = _remove_file_safe(mock_path)

        assert result is True
        mock_path.unlink.assert_called_once()

    def test_remove_file_safe_os_error(self):
        """Test file removal with OSError."""
        mock_path = MagicMock(spec=Path)
        mock_path.unlink.side_effect = OSError("File not found")

        result = _remove_file_safe(mock_path)

        assert result is False
        mock_path.unlink.assert_called_once()

    def test_remove_file_safe_permission_error(self):
        """Test file removal with permission error."""
        mock_path = MagicMock(spec=Path)
        mock_path.unlink.side_effect = OSError("Permission denied")

        result = _remove_file_safe(mock_path)

        assert result is False


class TestGetFrameFiles:
    """Test get_frame_files function."""

    def test_get_frame_files_with_frames(self):
        """Test getting frame files from directory."""
        mock_frames_dir = MagicMock(spec=Path)
        mock_frames_dir.exists.return_value = True

        # Mock glob to return different files for different extensions
        def glob_side_effect(pattern):
            if pattern == "*.png":
                return iter([Path("frames/frame_001.png"), Path("frames/frame_003.png")])
            elif pattern == "*.jpg":
                return iter([Path("frames/frame_002.jpg")])
            else:
                return iter([])

        mock_frames_dir.glob.side_effect = glob_side_effect

        result = get_frame_files(mock_frames_dir)

        # Should be sorted by frame number
        assert len(result) == 3
        assert result[0] == Path("frames/frame_001.png")
        assert result[1] == Path("frames/frame_002.jpg")
        assert result[2] == Path("frames/frame_003.png")

    def test_get_frame_files_empty_directory(self):
        """Test getting frame files from empty directory."""
        mock_frames_dir = MagicMock(spec=Path)
        mock_frames_dir.exists.return_value = True
        mock_frames_dir.glob.return_value = iter([])

        result = get_frame_files(mock_frames_dir)

        assert len(result) == 0
        assert isinstance(result, list)

    def test_get_frame_files_nonexistent_directory(self):
        """Test getting frame files from non-existent directory."""
        mock_frames_dir = MagicMock(spec=Path)
        mock_frames_dir.exists.return_value = False

        result = get_frame_files(mock_frames_dir)

        assert len(result) == 0
        assert isinstance(result, list)


class TestCreateOutputDirectories:
    """Test create_output_directories function."""

    def test_create_output_directories_with_intermediates(self):
        """Test creating output directories with intermediates."""
        base_path = Path("/tmp/output")

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            result = create_output_directories(base_path, keep_intermediates=True)

        # Should create all directories from INTERMEDIATE_DIRS
        assert "base" in result
        assert "frames" in result
        assert "depth_maps" in result
        assert "left_frames" in result
        assert "right_frames" in result
        assert "vr_frames" in result

        assert result["base"] == base_path
        assert result["frames"] == base_path / "00_original_frames"
        assert result["depth_maps"] == base_path / "02_depth_maps"
        assert result["left_frames"] == base_path / "04_left_frames"
        assert result["right_frames"] == base_path / "04_right_frames"
        assert result["vr_frames"] == base_path / "99_vr_frames"

        # Verify mkdir was called
        assert mock_mkdir.called

    def test_create_output_directories_without_intermediates(self):
        """Test creating output directories without intermediates."""
        base_path = Path("/tmp/output")

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            result = create_output_directories(base_path, keep_intermediates=False)

        # Should only create base directory
        assert "base" in result
        assert result["base"] == base_path

        # Should not create intermediate directories
        assert "frames" not in result
        assert "depth_maps" not in result
        assert "vr_frames" not in result

        # Only base mkdir should be called
        assert mock_mkdir.called

    def test_create_output_directories_default_intermediates(self):
        """Test creating output directories with default keep_intermediates."""
        base_path = Path("/tmp/output")

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            # Default should be True
            result = create_output_directories(base_path)

        # Should create all directories (default is True)
        assert "base" in result
        assert "frames" in result
        assert "depth_maps" in result
        assert "vr_frames" in result
