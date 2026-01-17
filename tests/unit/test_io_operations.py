"""Unit tests for io_operations.py pure helper functions."""

from pathlib import Path
from unittest.mock import patch, MagicMock
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

        with patch("pathlib.Path.mkdir"):
            # Default should be True
            result = create_output_directories(base_path)

        # Should create all directories (default is True)
        assert "base" in result
        assert "frames" in result
        assert "depth_maps" in result
        assert "vr_frames" in result


class TestValidateVideoFile:
    """Test validate_video_file function."""

    def test_validate_video_file_valid_mp4(self):
        """Test validation of valid MP4 file."""
        from src.depth_surge_3d.processing.io_operations import validate_video_file

        with patch("os.path.exists", return_value=True):
            result = validate_video_file("test.mp4")

        assert result is True

    def test_validate_video_file_valid_mkv(self):
        """Test validation of valid MKV file."""
        from src.depth_surge_3d.processing.io_operations import validate_video_file

        with patch("os.path.exists", return_value=True):
            result = validate_video_file("test.mkv")

        assert result is True

    def test_validate_video_file_invalid_extension(self):
        """Test validation of file with invalid extension."""
        from src.depth_surge_3d.processing.io_operations import validate_video_file

        with patch("os.path.exists", return_value=True):
            result = validate_video_file("test.txt")

        assert result is False

    def test_validate_video_file_nonexistent_file(self):
        """Test validation of non-existent file."""
        from src.depth_surge_3d.processing.io_operations import validate_video_file

        with patch("os.path.exists", return_value=False):
            result = validate_video_file("nonexistent.mp4")

        assert result is False

    def test_validate_video_file_case_insensitive(self):
        """Test validation is case insensitive for extensions."""
        from src.depth_surge_3d.processing.io_operations import validate_video_file

        with patch("os.path.exists", return_value=True):
            result = validate_video_file("test.MP4")

        assert result is True


class TestValidateImageFile:
    """Test validate_image_file function."""

    def test_validate_image_file_valid_png(self):
        """Test validation of valid PNG file."""
        from src.depth_surge_3d.processing.io_operations import validate_image_file

        with patch("os.path.exists", return_value=True):
            result = validate_image_file("test.png")

        assert result is True

    def test_validate_image_file_valid_jpg(self):
        """Test validation of valid JPG file."""
        from src.depth_surge_3d.processing.io_operations import validate_image_file

        with patch("os.path.exists", return_value=True):
            result = validate_image_file("test.jpg")

        assert result is True

    def test_validate_image_file_invalid_extension(self):
        """Test validation of file with invalid extension."""
        from src.depth_surge_3d.processing.io_operations import validate_image_file

        with patch("os.path.exists", return_value=True):
            result = validate_image_file("test.mp4")

        assert result is False

    def test_validate_image_file_nonexistent_file(self):
        """Test validation of non-existent file."""
        from src.depth_surge_3d.processing.io_operations import validate_image_file

        with patch("os.path.exists", return_value=False):
            result = validate_image_file("nonexistent.png")

        assert result is False

    def test_validate_image_file_case_insensitive(self):
        """Test validation is case insensitive for extensions."""
        from src.depth_surge_3d.processing.io_operations import validate_image_file

        with patch("os.path.exists", return_value=True):
            result = validate_image_file("test.PNG")

        assert result is True


class TestVerifyFFmpegInstallation:
    """Test verify_ffmpeg_installation function."""

    def test_verify_ffmpeg_installation_success(self):
        """Test FFmpeg verification when installed."""
        from src.depth_surge_3d.processing.io_operations import verify_ffmpeg_installation

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            result = verify_ffmpeg_installation()

        assert result is True

    def test_verify_ffmpeg_installation_not_found(self):
        """Test FFmpeg verification when not installed."""
        from src.depth_surge_3d.processing.io_operations import verify_ffmpeg_installation

        with patch("subprocess.run", side_effect=FileNotFoundError()):
            result = verify_ffmpeg_installation()

        assert result is False

    def test_verify_ffmpeg_installation_subprocess_error(self):
        """Test FFmpeg verification with subprocess error."""
        from src.depth_surge_3d.processing.io_operations import verify_ffmpeg_installation
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.SubprocessError()):
            result = verify_ffmpeg_installation()

        assert result is False

    def test_verify_ffmpeg_installation_os_error(self):
        """Test FFmpeg verification with OS error."""
        from src.depth_surge_3d.processing.io_operations import verify_ffmpeg_installation

        with patch("subprocess.run", side_effect=OSError()):
            result = verify_ffmpeg_installation()

        assert result is False


class TestGetAvailableSpace:
    """Test get_available_space function."""

    def test_get_available_space_with_statvfs(self):
        """Test getting available space using statvfs."""
        from src.depth_surge_3d.processing.io_operations import get_available_space

        mock_stat = MagicMock()
        mock_stat.f_bavail = 1000
        mock_stat.f_frsize = 4096

        with patch("os.statvfs", return_value=mock_stat):
            result = get_available_space(Path("/tmp"))

        assert result == 1000 * 4096

    def test_get_available_space_fallback_to_shutil(self):
        """Test getting available space with shutil fallback."""
        from src.depth_surge_3d.processing.io_operations import get_available_space

        with patch("os.statvfs", side_effect=AttributeError()):
            with patch("shutil.disk_usage", return_value=(0, 0, 5000000)):
                result = get_available_space(Path("/tmp"))

        assert result == 5000000

    def test_get_available_space_all_fallbacks_fail(self):
        """Test getting available space when all methods fail."""
        from src.depth_surge_3d.processing.io_operations import get_available_space

        with patch("os.statvfs", side_effect=OSError()):
            with patch("shutil.disk_usage", side_effect=OSError()):
                result = get_available_space(Path("/tmp"))

        assert result == 0


class TestCalculateDirectorySize:
    """Test calculate_directory_size function."""

    def test_calculate_directory_size_with_files(self):
        """Test calculating directory size with files."""
        from src.depth_surge_3d.processing.io_operations import calculate_directory_size

        # Mock os.walk to return directory structure
        mock_walk_data = [
            ("/tmp", [], ["file1.txt", "file2.txt"]),
        ]

        with patch("os.walk", return_value=mock_walk_data):
            with patch("os.path.exists", return_value=True):
                with patch("os.path.getsize", side_effect=[1024, 2048]):
                    result = calculate_directory_size(Path("/tmp"))

        assert result == 3072

    def test_calculate_directory_size_empty_directory(self):
        """Test calculating size of empty directory."""
        from src.depth_surge_3d.processing.io_operations import calculate_directory_size

        with patch("os.walk", return_value=[("/tmp", [], [])]):
            result = calculate_directory_size(Path("/tmp"))

        assert result == 0

    def test_calculate_directory_size_with_permission_error(self):
        """Test calculating size with permission errors."""
        from src.depth_surge_3d.processing.io_operations import calculate_directory_size

        with patch("os.walk", side_effect=PermissionError()):
            result = calculate_directory_size(Path("/tmp"))

        # Should handle permission errors gracefully
        assert result == 0


class TestGetVideoProperties:
    """Test get_video_properties function."""

    def test_get_video_properties_success(self):
        """Test getting video properties with cv2."""
        from src.depth_surge_3d.processing.io_operations import get_video_properties

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # Order: fps, frame_count, width, height, codec
        mock_cap.get.side_effect = [30.0, 900, 1920, 1080, 1196444237]

        with patch("cv2.VideoCapture", return_value=mock_cap):
            result = get_video_properties("test.mp4")

        assert result["width"] == 1920
        assert result["height"] == 1080
        assert result["fps"] == 30.0
        assert result["frame_count"] == 900
        assert result["duration"] == 30.0  # 900 / 30
        mock_cap.release.assert_called_once()

    def test_get_video_properties_not_opened(self):
        """Test video properties when video cannot be opened."""
        from src.depth_surge_3d.processing.io_operations import get_video_properties

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch("cv2.VideoCapture", return_value=mock_cap):
            result = get_video_properties("invalid.mp4")

        assert result == {}

    def test_get_video_properties_zero_fps(self):
        """Test video properties with zero FPS."""
        from src.depth_surge_3d.processing.io_operations import get_video_properties

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # Order: fps, frame_count, width, height, codec
        mock_cap.get.side_effect = [0.0, 0, 1920, 1080, 0]  # fps=0

        with patch("cv2.VideoCapture", return_value=mock_cap):
            result = get_video_properties("test.mp4")

        assert result["duration"] == 0.0

    def test_get_video_properties_exception(self):
        """Test video properties with exception."""
        from src.depth_surge_3d.processing.io_operations import get_video_properties

        with patch("cv2.VideoCapture", side_effect=Exception("OpenCV error")):
            result = get_video_properties("test.mp4")

        assert result == {}


class TestGetVideoInfoFFprobe:
    """Test get_video_info_ffprobe function."""

    def test_get_video_info_ffprobe_success(self):
        """Test getting video info with ffprobe."""
        from src.depth_surge_3d.processing.io_operations import get_video_info_ffprobe
        import json

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"format": {"duration": "30.0"}, "streams": []})

        with patch("subprocess.run", return_value=mock_result):
            result = get_video_info_ffprobe("test.mp4")

        assert "format" in result
        assert result["format"]["duration"] == "30.0"

    def test_get_video_info_ffprobe_failure(self):
        """Test ffprobe with non-zero return code."""
        from src.depth_surge_3d.processing.io_operations import get_video_info_ffprobe

        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            result = get_video_info_ffprobe("test.mp4")

        assert result == {}

    def test_get_video_info_ffprobe_exception(self):
        """Test ffprobe with exception."""
        from src.depth_surge_3d.processing.io_operations import get_video_info_ffprobe

        with patch("subprocess.run", side_effect=Exception("ffprobe not found")):
            result = get_video_info_ffprobe("test.mp4")

        assert result == {}


class TestCleanupDirectory:
    """Test _cleanup_directory function."""

    def test_cleanup_directory_with_keep_patterns(self):
        """Test cleanup directory with keep patterns."""
        from src.depth_surge_3d.processing.io_operations import _cleanup_directory

        # Create MagicMock files
        mock_file1 = MagicMock(spec=Path)
        mock_file1.is_file.return_value = True
        mock_file1.match.return_value = True  # keep this file

        mock_file2 = MagicMock(spec=Path)
        mock_file2.is_file.return_value = True
        mock_file2.match.return_value = False  # remove this file
        mock_file2.unlink.return_value = None

        mock_dir = MagicMock(spec=Path)
        mock_dir.exists.return_value = True
        mock_dir.rglob.return_value = [mock_file1, mock_file2]

        keep_patterns = ["*.png", "*.jpg"]

        result = _cleanup_directory(mock_dir, keep_patterns)

        # Should delete the file that doesn't match
        assert result >= 0
        mock_file2.unlink.assert_called()

    def test_cleanup_directory_nonexistent(self):
        """Test cleanup of non-existent directory."""
        from src.depth_surge_3d.processing.io_operations import _cleanup_directory

        mock_dir = MagicMock(spec=Path)
        mock_dir.exists.return_value = False

        result = _cleanup_directory(mock_dir, [])

        assert result == 0


class TestSaveProcessingSettings:
    """Test save_processing_settings function."""

    def test_save_processing_settings_success(self):
        """Test saving processing settings to JSON."""
        from src.depth_surge_3d.processing.io_operations import save_processing_settings

        settings_data = {
            "depth_resolution": 1080,
            "vr_format": "side_by_side",
            "vr_resolution": "1080p",
        }
        video_properties = {"width": 1920, "height": 1080}

        with patch("builtins.open", MagicMock()):
            with patch("json.dump"):
                with patch(
                    "src.depth_surge_3d.processing.io_operations.generate_output_filename",
                    return_value="output.mp4",
                ):
                    result = save_processing_settings(
                        Path("/tmp"),
                        "batch1",
                        settings_data,
                        video_properties,
                        "test.mp4",
                    )

        assert result == Path("/tmp/batch1-settings.json")

    def test_save_processing_settings_exception_with_fallback(self):
        """Test save settings with exception and fallback."""
        from src.depth_surge_3d.processing.io_operations import save_processing_settings

        settings_data = {
            "depth_resolution": 1080,
            "vr_format": "side_by_side",
            "vr_resolution": "1080p",
        }
        video_properties = {"width": 1920, "height": 1080}

        # First open() fails, second open() for fallback succeeds
        mock_open_context = MagicMock()
        with patch("builtins.open", side_effect=[OSError("Write error"), mock_open_context]):
            with patch("json.dump"):
                with patch(
                    "src.depth_surge_3d.processing.io_operations.generate_output_filename",
                    return_value="output.mp4",
                ):
                    result = save_processing_settings(
                        Path("/tmp"),
                        "batch1",
                        settings_data,
                        video_properties,
                        "test.mp4",
                    )

        # Should still return the path even if initial save failed
        assert result == Path("/tmp/batch1-settings.json")


class TestLoadProcessingSettings:
    """Test load_processing_settings function."""

    def test_load_processing_settings_success(self):
        """Test loading processing settings from JSON."""
        from src.depth_surge_3d.processing.io_operations import load_processing_settings
        import json

        settings_data = {"input_video": "test.mp4", "depth_resolution": 1080}

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", MagicMock()):
                with patch("json.load", return_value=settings_data):
                    result = load_processing_settings(Path("/tmp/settings.json"))

        assert result == settings_data

    def test_load_processing_settings_file_not_exists(self):
        """Test loading settings when file doesn't exist."""
        from src.depth_surge_3d.processing.io_operations import load_processing_settings

        with patch("pathlib.Path.exists", return_value=False):
            result = load_processing_settings(Path("/tmp/settings.json"))

        assert result is None

    def test_load_processing_settings_exception(self):
        """Test loading settings with exception."""
        from src.depth_surge_3d.processing.io_operations import load_processing_settings

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", side_effect=OSError("Read error")):
                result = load_processing_settings(Path("/tmp/settings.json"))

        assert result is None


class TestUpdateProcessingStatus:
    """Test update_processing_status function."""

    def test_update_processing_status_success(self):
        """Test updating processing status."""
        from src.depth_surge_3d.processing.io_operations import update_processing_status

        existing_settings = {
            "metadata": {
                "processing_status": "in_progress",
                "created_timestamp": 1000.0,
            }
        }

        with patch(
            "src.depth_surge_3d.processing.io_operations.load_processing_settings",
            return_value=existing_settings,
        ):
            with patch("builtins.open", MagicMock()):
                with patch("json.dump"):
                    with patch(
                        "src.depth_surge_3d.processing.io_operations.format_time_duration",
                        return_value="1h 30m",
                    ):
                        result = update_processing_status(
                            Path("/tmp/settings.json"), "completed"
                        )

        assert result is True

    def test_update_processing_status_load_failure(self):
        """Test update status when loading fails."""
        from src.depth_surge_3d.processing.io_operations import update_processing_status

        with patch(
            "src.depth_surge_3d.processing.io_operations.load_processing_settings",
            return_value=None,
        ):
            result = update_processing_status(Path("/tmp/settings.json"), "completed")

        assert result is False
