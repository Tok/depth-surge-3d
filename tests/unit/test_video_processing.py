"""Unit tests for video_processing.py"""

import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from src.depth_surge_3d.utils.video_processing import (
    _get_cv2,
    _create_intermediate_directories,
    _process_supersample_frame,
    _process_depth_frame,
    _process_stereo_frame,
    _process_fisheye_frame,
    _process_vr_assembly_frame,
    _process_single_frame_complete,
    process_video_serial,
    process_video_batch,
)


class TestGetCv2:
    """Test _get_cv2 function."""

    def test_get_cv2_success(self):
        """Test successful cv2 import."""
        with patch.dict("sys.modules", {"cv2": MagicMock()}):
            import cv2 as mock_cv2

            with patch("src.depth_surge_3d.utils.video_processing._get_cv2", return_value=mock_cv2):
                result = _get_cv2()
                assert result is not None

    def test_get_cv2_import_error(self):
        """Test cv2 import error handling."""
        # Remove cv2 from sys.modules to simulate it not being installed
        with patch.dict("sys.modules", {"cv2": None}):
            with pytest.raises(ImportError, match="opencv-python is required"):
                _get_cv2()


class TestCreateIntermediateDirectories:
    """Test _create_intermediate_directories function."""

    def test_create_directories_with_intermediates(self):
        """Test creating all intermediate directories."""
        mock_output_path = MagicMock(spec=Path)
        mock_vr_dir = MagicMock(spec=Path)
        mock_other_dir = MagicMock(spec=Path)

        def truediv_handler(dirname):
            if "vr_frames" in str(dirname):
                return mock_vr_dir
            return mock_other_dir

        mock_output_path.__truediv__.side_effect = truediv_handler

        result = _create_intermediate_directories(mock_output_path, keep_intermediates=True)

        assert "vr_frames" in result
        mock_vr_dir.mkdir.assert_called_with(exist_ok=True)

    def test_create_directories_without_intermediates(self):
        """Test creating only VR frames directory."""
        mock_output_path = MagicMock(spec=Path)
        mock_vr_dir = MagicMock(spec=Path)
        mock_output_path.__truediv__.return_value = mock_vr_dir

        result = _create_intermediate_directories(mock_output_path, keep_intermediates=False)

        assert "vr_frames" in result
        mock_vr_dir.mkdir.assert_called_with(exist_ok=True)


class TestProcessSupersampleFrame:
    """Test _process_supersample_frame function."""

    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    def test_process_supersample_frame_success(self, mock_get_cv2):
        """Test successful super sampling."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_projector = MagicMock()
        mock_frame_file = MagicMock(spec=Path)
        mock_frame_file.stem = "frame_001"

        # Mock cv2.imread to return a valid image
        mock_image = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_cv2.imread.return_value = mock_image

        # Mock projector.apply_super_sampling
        mock_supersampled = np.random.rand(2160, 3840, 3).astype(np.uint8)
        mock_projector.apply_super_sampling.return_value = mock_supersampled

        directories = {"supersampled": Path("/tmp/supersampled")}

        _process_supersample_frame(mock_projector, mock_frame_file, directories, 3840, 2160)

        mock_projector.apply_super_sampling.assert_called_once_with(mock_image, 3840, 2160)
        mock_cv2.imwrite.assert_called_once()

    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    def test_process_supersample_frame_no_image(self, mock_get_cv2):
        """Test handling of None image."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2
        mock_cv2.imread.return_value = None

        mock_projector = MagicMock()
        mock_frame_file = MagicMock(spec=Path)
        directories = {}

        # Should return early without error
        _process_supersample_frame(mock_projector, mock_frame_file, directories, 1920, 1080)

        mock_projector.apply_super_sampling.assert_not_called()


class TestProcessDepthFrame:
    """Test _process_depth_frame function."""

    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    def test_process_depth_frame_with_supersampled(self, mock_get_cv2):
        """Test depth processing with supersampled frame."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_projector = MagicMock()
        mock_frame_file = MagicMock(spec=Path)
        mock_frame_file.stem = "frame_001"

        # Mock supersampled directory exists
        mock_supersample_path = MagicMock(spec=Path)
        mock_supersample_path.exists.return_value = True

        mock_supersampled_dir = MagicMock(spec=Path)
        mock_supersampled_dir.__truediv__.return_value = mock_supersample_path

        directories = {"supersampled": mock_supersampled_dir, "depth_maps": Path("/tmp/depth")}

        mock_image = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_cv2.imread.return_value = mock_image

        mock_depth = np.random.rand(1080, 1920).astype(np.float32)
        mock_projector.generate_depth_map_from_array.return_value = mock_depth

        _process_depth_frame(mock_projector, mock_frame_file, directories)

        mock_projector.generate_depth_map_from_array.assert_called_once_with(mock_image)
        mock_cv2.imwrite.assert_called_once()

    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    def test_process_depth_frame_without_supersampled(self, mock_get_cv2):
        """Test depth processing without supersampled frame."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_projector = MagicMock()
        mock_frame_file = MagicMock(spec=Path)
        mock_frame_file.stem = "frame_001"

        directories = {"depth_maps": Path("/tmp/depth")}

        mock_image = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_cv2.imread.return_value = mock_image

        mock_depth = np.random.rand(1080, 1920).astype(np.float32)
        mock_projector.generate_depth_map_from_array.return_value = mock_depth

        _process_depth_frame(mock_projector, mock_frame_file, directories)

        mock_projector.generate_depth_map_from_array.assert_called_once()


class TestProcessStereoFrame:
    """Test _process_stereo_frame function."""

    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    def test_process_stereo_frame_success(self, mock_get_cv2):
        """Test successful stereo pair creation."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_projector = MagicMock()
        mock_frame_file = MagicMock(spec=Path)
        mock_frame_file.stem = "frame_001"

        # Mock depth maps directory
        mock_depth_dir = MagicMock(spec=Path)
        mock_depth_path = MagicMock(spec=Path)
        mock_depth_dir.__truediv__.return_value = mock_depth_path

        directories = {
            "depth_maps": mock_depth_dir,
            "left_frames": Path("/tmp/left"),
            "right_frames": Path("/tmp/right"),
        }

        mock_image = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_depth_vis = np.random.rand(1080, 1920).astype(np.uint8)

        # cv2.imread is called twice: once for image, once for depth
        mock_cv2.imread.side_effect = [mock_image, mock_depth_vis]

        mock_left = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_right = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_projector.create_stereo_pair_from_depth.return_value = (mock_left, mock_right)

        _process_stereo_frame(mock_projector, mock_frame_file, directories)

        mock_projector.create_stereo_pair_from_depth.assert_called_once()
        # Should write both left and right frames
        assert mock_cv2.imwrite.call_count == 2


class TestProcessFisheyeFrame:
    """Test _process_fisheye_frame function."""

    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    def test_process_fisheye_frame_success(self, mock_get_cv2):
        """Test successful fisheye distortion."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_projector = MagicMock()
        mock_frame_file = MagicMock(spec=Path)
        mock_frame_file.stem = "frame_001"

        # Mock directories
        mock_left_dir = MagicMock(spec=Path)
        mock_right_dir = MagicMock(spec=Path)
        mock_left_path = MagicMock(spec=Path)
        mock_right_path = MagicMock(spec=Path)

        mock_left_dir.__truediv__.return_value = mock_left_path
        mock_right_dir.__truediv__.return_value = mock_right_path

        directories = {
            "left_frames": mock_left_dir,
            "right_frames": mock_right_dir,
            "left_distorted": Path("/tmp/left_distorted"),
            "right_distorted": Path("/tmp/right_distorted"),
        }

        mock_left = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_right = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_cv2.imread.side_effect = [mock_left, mock_right]

        mock_left_distorted = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_right_distorted = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_projector.apply_fisheye_distortion.side_effect = [
            mock_left_distorted,
            mock_right_distorted,
        ]

        _process_fisheye_frame(mock_projector, mock_frame_file, directories)

        # Should apply fisheye to both left and right
        assert mock_projector.apply_fisheye_distortion.call_count == 2
        # Should write both distorted frames
        assert mock_cv2.imwrite.call_count == 2

    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    def test_process_fisheye_frame_missing_frames(self, mock_get_cv2):
        """Test handling of missing stereo frames."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_projector = MagicMock()
        mock_frame_file = MagicMock(spec=Path)

        mock_left_dir = MagicMock(spec=Path)
        mock_right_dir = MagicMock(spec=Path)

        directories = {
            "left_frames": mock_left_dir,
            "right_frames": mock_right_dir,
        }

        # Return None for one of the frames
        mock_cv2.imread.side_effect = [None, np.random.rand(1080, 1920, 3).astype(np.uint8)]

        _process_fisheye_frame(mock_projector, mock_frame_file, directories)

        # Should not process if frames are missing
        mock_projector.apply_fisheye_distortion.assert_not_called()


class TestProcessVRAssemblyFrame:
    """Test _process_vr_assembly_frame function."""

    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    def test_process_vr_assembly_with_distortion(self, mock_get_cv2):
        """Test VR assembly with distorted frames."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_projector = MagicMock()
        mock_frame_file = MagicMock(spec=Path)
        mock_frame_file.stem = "frame_001"

        # Mock distorted directories
        mock_left_dir = MagicMock(spec=Path)
        mock_right_dir = MagicMock(spec=Path)

        directories = {
            "left_distorted": mock_left_dir,
            "right_distorted": mock_right_dir,
            "vr_frames": Path("/tmp/vr"),
        }

        mock_left = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_right = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_cv2.imread.side_effect = [mock_left, mock_right]

        mock_vr_frame = np.random.rand(1080, 3840, 3).astype(np.uint8)
        mock_projector.create_vr_format.return_value = mock_vr_frame

        _process_vr_assembly_frame(
            mock_projector, mock_frame_file, directories, apply_distortion=True
        )

        mock_projector.create_vr_format.assert_called_once()
        mock_cv2.imwrite.assert_called_once()

    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    def test_process_vr_assembly_without_distortion(self, mock_get_cv2):
        """Test VR assembly without distortion."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_projector = MagicMock()
        mock_frame_file = MagicMock(spec=Path)
        mock_frame_file.stem = "frame_001"

        # Mock regular stereo directories
        mock_left_dir = MagicMock(spec=Path)
        mock_right_dir = MagicMock(spec=Path)

        directories = {
            "left_frames": mock_left_dir,
            "right_frames": mock_right_dir,
            "vr_frames": Path("/tmp/vr"),
        }

        mock_left = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_right = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_cv2.imread.side_effect = [mock_left, mock_right]

        mock_vr_frame = np.random.rand(1080, 3840, 3).astype(np.uint8)
        mock_projector.create_vr_format.return_value = mock_vr_frame

        _process_vr_assembly_frame(
            mock_projector, mock_frame_file, directories, apply_distortion=False
        )

        mock_projector.create_vr_format.assert_called_once()


class TestProcessSingleFrameComplete:
    """Test _process_single_frame_complete function."""

    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    def test_process_single_frame_complete_with_distortion(self, mock_get_cv2):
        """Test complete frame processing with distortion."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_projector = MagicMock()
        mock_callback = MagicMock()

        mock_image = np.random.rand(1080, 1920, 3).astype(np.uint8)
        directories = {}

        # Mock projector methods
        mock_depth = np.random.rand(1080, 1920).astype(np.float32)
        mock_projector.generate_depth_map_from_array.return_value = mock_depth

        mock_left = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_right = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_projector.create_stereo_pair_from_depth.return_value = (mock_left, mock_right)

        mock_left_distorted = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_right_distorted = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_projector.apply_fisheye_distortion.side_effect = [
            mock_left_distorted,
            mock_right_distorted,
        ]

        mock_vr = np.random.rand(1080, 3840, 3).astype(np.uint8)
        mock_projector.create_vr_format.return_value = mock_vr

        result = _process_single_frame_complete(
            mock_projector,
            mock_callback,
            mock_image,
            0,
            10,
            directories,
            "frame_001",
            1920,
            1080,
            apply_distortion=True,
        )

        assert result is not None
        mock_projector.generate_depth_map_from_array.assert_called_once()
        mock_projector.create_stereo_pair_from_depth.assert_called_once()
        assert mock_projector.apply_fisheye_distortion.call_count == 2
        mock_projector.create_vr_format.assert_called_once()

    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    def test_process_single_frame_complete_without_distortion(self, mock_get_cv2):
        """Test complete frame processing without distortion."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_projector = MagicMock()
        mock_callback = MagicMock()

        mock_image = np.random.rand(1080, 1920, 3).astype(np.uint8)
        directories = {}

        # Mock projector methods
        mock_depth = np.random.rand(1080, 1920).astype(np.float32)
        mock_projector.generate_depth_map_from_array.return_value = mock_depth

        mock_left = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_right = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_projector.create_stereo_pair_from_depth.return_value = (mock_left, mock_right)

        mock_vr = np.random.rand(1080, 3840, 3).astype(np.uint8)
        mock_projector.create_vr_format.return_value = mock_vr

        result = _process_single_frame_complete(
            mock_projector,
            mock_callback,
            mock_image,
            0,
            10,
            directories,
            "frame_001",
            1920,
            1080,
            apply_distortion=False,
        )

        assert result is not None
        mock_projector.apply_fisheye_distortion.assert_not_called()


class TestProcessVideoSerial:
    """Test process_video_serial function."""

    @patch("src.depth_surge_3d.utils.video_processing._create_intermediate_directories")
    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    @patch("src.depth_surge_3d.utils.video_processing._process_single_frame_complete")
    def test_process_video_serial_success(self, mock_process_frame, mock_get_cv2, mock_create_dirs):
        """Test successful serial video processing."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_projector = MagicMock()
        mock_callback = MagicMock()

        # Mock frame files
        mock_frame1 = MagicMock(spec=Path)
        mock_frame1.stem = "frame_001"
        mock_frame2 = MagicMock(spec=Path)
        mock_frame2.stem = "frame_002"

        frame_files = [mock_frame1, mock_frame2]
        output_path = Path("/tmp/output")

        # Mock directories
        mock_vr_dir = MagicMock(spec=Path)
        mock_vr_frame1 = MagicMock(spec=Path)
        mock_vr_frame1.exists.return_value = False
        mock_vr_frame2 = MagicMock(spec=Path)
        mock_vr_frame2.exists.return_value = False

        mock_vr_dir.__truediv__.side_effect = [mock_vr_frame1, mock_vr_frame2]

        directories = {"vr_frames": mock_vr_dir}
        mock_create_dirs.return_value = directories

        # Mock image loading
        mock_image = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_cv2.imread.return_value = mock_image

        # Mock frame processing
        mock_vr_frame = np.random.rand(1080, 3840, 3).astype(np.uint8)
        mock_process_frame.return_value = mock_vr_frame

        result = process_video_serial(mock_projector, mock_callback, frame_files, output_path)

        assert result is True
        assert mock_process_frame.call_count == 2
        assert mock_cv2.imwrite.call_count == 2

    @patch("src.depth_surge_3d.utils.video_processing._create_intermediate_directories")
    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    def test_process_video_serial_skip_existing(self, mock_get_cv2, mock_create_dirs):
        """Test serial processing skips existing frames."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_projector = MagicMock()
        mock_callback = MagicMock()

        mock_frame = MagicMock(spec=Path)
        mock_frame.stem = "frame_001"
        frame_files = [mock_frame]
        output_path = Path("/tmp/output")

        # Mock VR frame already exists
        mock_vr_dir = MagicMock(spec=Path)
        mock_vr_frame = MagicMock(spec=Path)
        mock_vr_frame.exists.return_value = True
        mock_vr_dir.__truediv__.return_value = mock_vr_frame

        directories = {"vr_frames": mock_vr_dir}
        mock_create_dirs.return_value = directories

        result = process_video_serial(mock_projector, mock_callback, frame_files, output_path)

        assert result is True
        # Should not process frame that already exists
        mock_cv2.imread.assert_not_called()

    @patch("src.depth_surge_3d.utils.video_processing._create_intermediate_directories")
    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    @patch("src.depth_surge_3d.utils.video_processing._process_single_frame_complete")
    def test_process_video_serial_none_image(self, mock_process, mock_get_cv2, mock_create_dirs):
        """Test serial processing handles None image (imread failure)."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        # imread returns None (file read failure)
        mock_cv2.imread.return_value = None

        mock_projector = MagicMock()
        mock_callback = MagicMock()

        mock_frame = MagicMock(spec=Path)
        mock_frame.stem = "frame_001"
        frame_files = [mock_frame]
        output_path = Path("/tmp/output")

        mock_vr_dir = MagicMock(spec=Path)
        mock_vr_frame = MagicMock(spec=Path)
        mock_vr_frame.exists.return_value = False
        mock_vr_dir.__truediv__.return_value = mock_vr_frame

        directories = {"vr_frames": mock_vr_dir}
        mock_create_dirs.return_value = directories

        result = process_video_serial(mock_projector, mock_callback, frame_files, output_path)

        assert result is True
        # Should not process frame with None image
        mock_process.assert_not_called()

    @patch("src.depth_surge_3d.utils.video_processing._create_intermediate_directories")
    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    @patch("src.depth_surge_3d.utils.video_processing._process_single_frame_complete")
    def test_process_video_serial_exception_handling(
        self, mock_process, mock_get_cv2, mock_create_dirs
    ):
        """Test serial processing handles exceptions gracefully."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_image = np.random.rand(480, 640, 3).astype(np.uint8)
        mock_cv2.imread.return_value = mock_image

        # Simulate exception during processing
        mock_process.side_effect = RuntimeError("Processing failed")

        mock_projector = MagicMock()
        mock_callback = MagicMock()

        mock_frame = MagicMock(spec=Path)
        mock_frame.stem = "frame_001"
        frame_files = [mock_frame]
        output_path = Path("/tmp/output")

        mock_vr_dir = MagicMock(spec=Path)
        mock_vr_frame = MagicMock(spec=Path)
        mock_vr_frame.exists.return_value = False
        mock_vr_dir.__truediv__.return_value = mock_vr_frame

        directories = {"vr_frames": mock_vr_dir}
        mock_create_dirs.return_value = directories

        # Should handle exception and continue
        result = process_video_serial(mock_projector, mock_callback, frame_files, output_path)

        assert result is True

    @patch("src.depth_surge_3d.utils.video_processing._create_intermediate_directories")
    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    @patch("src.depth_surge_3d.utils.video_processing._process_single_frame_complete")
    def test_process_video_serial_with_super_sampling(
        self, mock_process, mock_get_cv2, mock_create_dirs
    ):
        """Test serial processing with super sampling enabled."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_image = np.random.rand(480, 640, 3).astype(np.uint8)
        mock_cv2.imread.return_value = mock_image

        # Mock imwrite
        mock_cv2.imwrite.return_value = True

        mock_projector = MagicMock()
        # Return different sized image for super sampling
        mock_projector.apply_super_sampling.return_value = np.random.rand(1080, 1920, 3).astype(
            np.uint8
        )

        mock_callback = MagicMock()

        mock_frame = MagicMock(spec=Path)
        mock_frame.stem = "frame_001"
        frame_files = [mock_frame]
        output_path = Path("/tmp/output")

        mock_vr_dir = MagicMock(spec=Path)
        mock_vr_frame = MagicMock(spec=Path)
        mock_vr_frame.exists.return_value = False
        mock_vr_dir.__truediv__.return_value = mock_vr_frame

        mock_ss_dir = MagicMock(spec=Path)
        directories = {"vr_frames": mock_vr_dir, "supersampled": mock_ss_dir}
        mock_create_dirs.return_value = directories

        # Enable super sampling
        result = process_video_serial(
            mock_projector,
            mock_callback,
            frame_files,
            output_path,
            super_sample_width=1920,
            super_sample_height=1080,
        )

        assert result is True
        # Should call _process_single_frame_complete with super sampling dimensions
        mock_process.assert_called_once()
        call_kwargs = mock_process.call_args[1]
        assert call_kwargs["super_sample_width"] == 1920
        assert call_kwargs["super_sample_height"] == 1080


class TestProcessVideoBatch:
    """Test process_video_batch function."""

    @patch("src.depth_surge_3d.utils.video_processing._create_intermediate_directories")
    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    @patch("src.depth_surge_3d.utils.video_processing._process_depth_frame")
    @patch("src.depth_surge_3d.utils.video_processing._process_stereo_frame")
    @patch("src.depth_surge_3d.utils.video_processing._process_vr_assembly_frame")
    def test_process_video_batch_success(
        self,
        mock_vr_assembly,
        mock_stereo,
        mock_depth,
        mock_get_cv2,
        mock_create_dirs,
    ):
        """Test successful batch video processing."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_projector = MagicMock()
        mock_callback = MagicMock()

        mock_frame1 = MagicMock(spec=Path)
        mock_frame2 = MagicMock(spec=Path)
        frame_files = [mock_frame1, mock_frame2]
        output_path = Path("/tmp/output")

        directories = {"vr_frames": Path("/tmp/vr")}
        mock_create_dirs.return_value = directories

        result = process_video_batch(
            mock_projector, mock_callback, frame_files, output_path, apply_distortion=False
        )

        assert result is True
        # Each phase should process all frames
        assert mock_depth.call_count == 2
        assert mock_stereo.call_count == 2
        assert mock_vr_assembly.call_count == 2

    @patch("src.depth_surge_3d.utils.video_processing._create_intermediate_directories")
    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    @patch("src.depth_surge_3d.utils.video_processing._process_depth_frame")
    @patch("src.depth_surge_3d.utils.video_processing._process_stereo_frame")
    @patch("src.depth_surge_3d.utils.video_processing._process_fisheye_frame")
    @patch("src.depth_surge_3d.utils.video_processing._process_vr_assembly_frame")
    def test_process_video_batch_with_distortion(
        self,
        mock_vr_assembly,
        mock_fisheye,
        mock_stereo,
        mock_depth,
        mock_get_cv2,
        mock_create_dirs,
    ):
        """Test batch processing with fisheye distortion."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_projector = MagicMock()
        mock_callback = MagicMock()

        mock_frame = MagicMock(spec=Path)
        frame_files = [mock_frame]
        output_path = Path("/tmp/output")

        directories = {"vr_frames": Path("/tmp/vr")}
        mock_create_dirs.return_value = directories

        result = process_video_batch(
            mock_projector, mock_callback, frame_files, output_path, apply_distortion=True
        )

        assert result is True
        # Should include fisheye phase
        mock_fisheye.assert_called_once()

    @patch("src.depth_surge_3d.utils.video_processing._create_intermediate_directories")
    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    @patch("src.depth_surge_3d.utils.video_processing._process_supersample_frame")
    @patch("src.depth_surge_3d.utils.video_processing._process_depth_frame")
    @patch("src.depth_surge_3d.utils.video_processing._process_stereo_frame")
    @patch("src.depth_surge_3d.utils.video_processing._process_fisheye_frame")
    @patch("src.depth_surge_3d.utils.video_processing._process_vr_assembly_frame")
    def test_process_video_batch_with_super_sampling(
        self,
        mock_vr_assembly,
        mock_fisheye,
        mock_stereo,
        mock_depth,
        mock_supersample,
        mock_get_cv2,
        mock_create_dirs,
    ):
        """Test batch processing with super sampling."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_projector = MagicMock()
        mock_callback = MagicMock()

        mock_frame = MagicMock(spec=Path)
        frame_files = [mock_frame]
        output_path = Path("/tmp/output")

        directories = {
            "vr_frames": Path("/tmp/vr"),
            "left_frames": Path("/tmp/left"),
            "right_frames": Path("/tmp/right"),
            "supersampled_frames": Path("/tmp/ss"),
        }
        mock_create_dirs.return_value = directories

        # Set super_sample dimensions different from defaults
        result = process_video_batch(
            mock_projector,
            mock_callback,
            frame_files,
            output_path,
            super_sample_width=3840,
            super_sample_height=2160,
        )

        assert result is True
        # Should include super sampling phase
        mock_supersample.assert_called_once()


class TestProcessSingleFrameCompleteEdgeCases:
    """Test _process_single_frame_complete edge cases."""

    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    def test_process_single_frame_save_intermediates(self, mock_get_cv2):
        """Test saving intermediate outputs."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_projector = MagicMock()
        mock_callback = MagicMock()

        mock_image = np.random.rand(1080, 1920, 3).astype(np.uint8)

        # Mock directories for all intermediate outputs
        directories = {
            "supersampled": Path("/tmp/supersampled"),
            "depth_maps": Path("/tmp/depth"),
            "left_frames": Path("/tmp/left"),
            "right_frames": Path("/tmp/right"),
            "left_distorted": Path("/tmp/left_dist"),
            "right_distorted": Path("/tmp/right_dist"),
        }

        mock_depth = np.random.rand(1080, 1920).astype(np.float32)
        mock_projector.generate_depth_map_from_array.return_value = mock_depth

        mock_left = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_right = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_projector.create_stereo_pair_from_depth.return_value = (mock_left, mock_right)
        mock_projector.apply_fisheye_distortion.side_effect = [mock_left, mock_right]
        mock_projector.create_vr_format.return_value = np.random.rand(1080, 3840, 3).astype(
            np.uint8
        )

        result = _process_single_frame_complete(
            mock_projector,
            mock_callback,
            mock_image,
            0,
            10,
            directories,
            "frame_001",
            1920,
            1080,
            apply_distortion=True,
        )

        assert result is not None
        # Should save all intermediates
        assert mock_cv2.imwrite.call_count >= 4

    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    def test_process_single_frame_with_super_sampling(self, mock_get_cv2):
        """Test super sampling with intermediate save."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_projector = MagicMock()
        mock_callback = MagicMock()

        # Original image is smaller than target super sample dimensions
        mock_image = np.random.rand(720, 1280, 3).astype(np.uint8)

        # Mock super sampling result
        mock_ss_image = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_projector.apply_super_sampling.return_value = mock_ss_image

        # Mock directories including supersampled
        directories = {
            "supersampled": Path("/tmp/supersampled"),
            "depth_maps": Path("/tmp/depth"),
            "left_frames": Path("/tmp/left"),
            "right_frames": Path("/tmp/right"),
        }

        mock_depth = np.random.rand(1080, 1920).astype(np.float32)
        mock_projector.generate_depth_map_from_array.return_value = mock_depth

        mock_left = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_right = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_projector.create_stereo_pair_from_depth.return_value = (mock_left, mock_right)
        mock_projector.create_vr_format.return_value = np.random.rand(1080, 3840, 3).astype(
            np.uint8
        )

        # Enable super sampling with different dimensions
        result = _process_single_frame_complete(
            mock_projector,
            mock_callback,
            mock_image,
            0,
            10,
            directories,
            "frame_001",
            super_sample_width=1920,
            super_sample_height=1080,
            apply_distortion=False,
        )

        assert result is not None
        # Should call super sampling
        mock_projector.apply_super_sampling.assert_called_once_with(mock_image, 1920, 1080)
        # Should save supersampled intermediate
        assert mock_cv2.imwrite.call_count >= 1


class TestProcessDepthFrameEdgeCases:
    """Test _process_depth_frame edge cases."""

    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    def test_process_depth_frame_none_image(self, mock_get_cv2):
        """Test depth processing with None image."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2
        mock_cv2.imread.return_value = None

        mock_projector = MagicMock()
        mock_frame_file = MagicMock(spec=Path)
        mock_frame_file.stem = "frame_001"

        directories = {"depth_maps": Path("/tmp/depth")}

        # Should return early without error
        _process_depth_frame(mock_projector, mock_frame_file, directories)

        mock_projector.generate_depth_map_from_array.assert_not_called()

    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    def test_process_depth_frame_with_supersampled_missing(self, mock_get_cv2):
        """Test depth processing when supersampled file doesn't exist."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_image = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_cv2.imread.return_value = mock_image

        mock_projector = MagicMock()
        mock_depth = np.random.rand(1080, 1920).astype(np.float32)
        mock_projector.generate_depth_map_from_array.return_value = mock_depth

        mock_frame_file = MagicMock(spec=Path)
        mock_frame_file.stem = "frame_001"

        mock_ss_path = MagicMock(spec=Path)
        mock_ss_path.exists.return_value = False
        mock_ss_dir = MagicMock(spec=Path)
        mock_ss_dir.__truediv__.return_value = mock_ss_path

        directories = {"depth_maps": Path("/tmp/depth"), "supersampled": mock_ss_dir}

        # Should load from original frame when supersampled doesn't exist
        _process_depth_frame(mock_projector, mock_frame_file, directories)

        mock_projector.generate_depth_map_from_array.assert_called_once()


class TestProcessStereoFrameEdgeCases:
    """Test _process_stereo_frame edge cases."""

    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    def test_process_stereo_frame_none_image(self, mock_get_cv2):
        """Test stereo processing with None image."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2
        mock_cv2.imread.return_value = None

        mock_projector = MagicMock()
        mock_frame_file = MagicMock(spec=Path)

        mock_depth_dir = MagicMock(spec=Path)
        directories = {"depth_maps": mock_depth_dir}

        # Should return early without error
        _process_stereo_frame(mock_projector, mock_frame_file, directories)

        mock_projector.create_stereo_pair_from_depth.assert_not_called()

    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    def test_process_stereo_frame_with_supersampled_exists(self, mock_get_cv2):
        """Test stereo processing when supersampled file exists."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_image = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_depth = np.random.rand(1080, 1920).astype(np.float32)
        mock_cv2.imread.side_effect = [mock_image, mock_depth]

        mock_projector = MagicMock()
        mock_left = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_right = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_projector.create_stereo_pair_from_depth.return_value = (mock_left, mock_right)

        mock_frame_file = MagicMock(spec=Path)
        mock_frame_file.stem = "frame_001"

        mock_ss_path = MagicMock(spec=Path)
        mock_ss_path.exists.return_value = True
        mock_ss_dir = MagicMock(spec=Path)
        mock_ss_dir.__truediv__.return_value = mock_ss_path

        mock_depth_dir = MagicMock(spec=Path)

        directories = {
            "depth_maps": mock_depth_dir,
            "left_frames": Path("/tmp/left"),
            "right_frames": Path("/tmp/right"),
            "supersampled": mock_ss_dir,
        }

        # Should load from supersampled when it exists
        _process_stereo_frame(mock_projector, mock_frame_file, directories)

        mock_projector.create_stereo_pair_from_depth.assert_called_once()

    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    def test_process_stereo_frame_with_supersampled_missing(self, mock_get_cv2):
        """Test stereo processing when supersampled file doesn't exist."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_image = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_depth = np.random.rand(1080, 1920).astype(np.float32)
        mock_cv2.imread.side_effect = [mock_image, mock_depth]

        mock_projector = MagicMock()
        mock_left = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_right = np.random.rand(1080, 1920, 3).astype(np.uint8)
        mock_projector.create_stereo_pair_from_depth.return_value = (mock_left, mock_right)

        mock_frame_file = MagicMock(spec=Path)
        mock_frame_file.stem = "frame_001"

        mock_ss_path = MagicMock(spec=Path)
        mock_ss_path.exists.return_value = False
        mock_ss_dir = MagicMock(spec=Path)
        mock_ss_dir.__truediv__.return_value = mock_ss_path

        mock_depth_dir = MagicMock(spec=Path)

        directories = {
            "depth_maps": mock_depth_dir,
            "left_frames": Path("/tmp/left"),
            "right_frames": Path("/tmp/right"),
            "supersampled": mock_ss_dir,
        }

        # Should load from original when supersampled doesn't exist
        _process_stereo_frame(mock_projector, mock_frame_file, directories)

        mock_projector.create_stereo_pair_from_depth.assert_called_once()


class TestProcessVRAssemblyEdgeCases:
    """Test _process_vr_assembly_frame edge cases."""

    @patch("src.depth_surge_3d.utils.video_processing._get_cv2")
    def test_process_vr_assembly_none_images(self, mock_get_cv2):
        """Test VR assembly with None images."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2
        mock_cv2.imread.side_effect = [None, np.random.rand(1080, 1920, 3).astype(np.uint8)]

        mock_projector = MagicMock()
        mock_frame_file = MagicMock(spec=Path)

        mock_left_dir = MagicMock(spec=Path)
        mock_right_dir = MagicMock(spec=Path)
        directories = {"left_frames": mock_left_dir, "right_frames": mock_right_dir}

        # Should return early without error
        _process_vr_assembly_frame(
            mock_projector, mock_frame_file, directories, apply_distortion=False
        )

        mock_projector.create_vr_format.assert_not_called()
