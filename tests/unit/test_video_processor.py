"""Unit tests for VideoProcessor."""

import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.depth_surge_3d.processing.video_processor import VideoProcessor
from src.depth_surge_3d.models.video_depth_estimator import VideoDepthEstimator


# Default settings for tests
DEFAULT_TEST_SETTINGS = {
    "depth_resolution": "auto",
    "super_sample": "none",
    "apply_distortion": False,
    "vr_format": "side_by_side",
    "vr_resolution": "1080p",
    "disparity_shift": 2.0,
    "video_encoder": "libx264",
    "keep_intermediates": True,
    "baseline": 0.065,  # Required for stereo pair creation
    "focal_length": 1000.0,  # Required for stereo pair creation
    "hole_fill_quality": "none",  # Required for stereo pair creation
    "fisheye_fov": 190.0,  # Required for distortion
    "fisheye_projection": "equidistant",  # Required for distortion
    "per_eye_width": 1920,  # Required for VR frame creation
    "per_eye_height": 1080,  # Required for VR frame creation
}


def get_test_settings(**overrides):
    """Get test settings with optional overrides."""
    settings = DEFAULT_TEST_SETTINGS.copy()
    settings.update(overrides)
    return settings


class TestVideoProcessorInit:
    """Test VideoProcessor initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator)

        assert processor.depth_estimator == mock_estimator
        assert processor.verbose is False
        assert processor._settings_file is None

    def test_init_with_verbose(self):
        """Test initialization with verbose enabled."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator, verbose=True)

        assert processor.verbose is True


class TestVideoProcessorHelpers:
    """Test VideoProcessor helper methods."""

    def test_update_step_progress_with_tracker(self):
        """Test progress update with tracker."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator)

        mock_tracker = MagicMock()
        processor._update_step_progress(mock_tracker, "Processing...", "step1", 50, 100)

        mock_tracker.update_progress.assert_called_once_with(
            "Processing...",
            phase="processing",
            frame_num=50,
            step_name="step1",
            step_progress=50,
            step_total=100,
        )

    def test_update_step_progress_without_tracker(self):
        """Test progress update without tracker (no-op)."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator)

        # Should not raise exception
        processor._update_step_progress(None, "Processing...", "step1", 50, 100)

    def test_handle_step_error_with_settings_file(self):
        """Test error handling with settings file."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator)
        processor._settings_file = Path("/tmp/settings.json")

        with patch(
            "src.depth_surge_3d.processing.video_processor.update_processing_status"
        ) as mock_update:
            result = processor._handle_step_error("Test error")

        assert result is False
        mock_update.assert_called_once_with(
            Path("/tmp/settings.json"), "failed", {"error": "Test error"}
        )

    def test_handle_step_error_without_settings_file(self):
        """Test error handling without settings file."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator)
        processor._settings_file = None

        result = processor._handle_step_error("Test error")

        assert result is False


class TestSetupProcessing:
    """Test _setup_processing method."""

    @patch("src.depth_surge_3d.processing.video_processor.save_processing_settings")
    @patch("src.depth_surge_3d.processing.video_processor.create_output_directories")
    @patch("src.depth_surge_3d.processing.video_processor.title_bar")
    @patch("time.time", return_value=1234567890.0)
    def test_setup_processing_success(
        self, mock_time, mock_title, mock_create_dirs, mock_save_settings
    ):
        """Test successful processing setup."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator)

        mock_directories = {
            "base": Path("/tmp/output"),
            "frames": Path("/tmp/output/frames"),
        }
        mock_create_dirs.return_value = mock_directories
        mock_save_settings.return_value = Path("/tmp/settings.json")
        mock_title.return_value = "=== Title ==="

        video_properties = {"width": 1920, "height": 1080}
        settings = {"vr_format": "side_by_side", "keep_intermediates": True}

        output_path, directories, settings_file = processor._setup_processing(
            "/tmp/input.mp4", "/tmp/output", settings, video_properties
        )

        assert output_path == Path("/tmp/output")
        assert directories == mock_directories
        assert settings_file == Path("/tmp/settings.json")
        mock_create_dirs.assert_called_once_with(Path("/tmp/output"), True)
        mock_save_settings.assert_called_once()


class TestFinalizeProcessing:
    """Test _finalize_processing method."""

    @patch("src.depth_surge_3d.processing.video_processor.update_processing_status")
    @patch("src.depth_surge_3d.processing.video_processor.generate_output_filename")
    @patch("src.depth_surge_3d.processing.video_processor.console_success")
    def test_finalize_processing_success(self, mock_console, mock_gen_filename, mock_update_status):
        """Test successful processing finalization."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator)
        processor._settings_file = Path("/tmp/settings.json")

        mock_gen_filename.return_value = "output_3D_side_by_side.mp4"
        mock_console.return_value = "Success!"

        settings = {"vr_format": "side_by_side", "vr_resolution": "1080p"}

        processor._finalize_processing(
            success=True,
            output_path=Path("/tmp/output"),
            video_path="/tmp/input.mp4",
            settings=settings,
            num_frames=100,
        )

        mock_update_status.assert_called_once_with(
            Path("/tmp/settings.json"),
            "completed",
            {
                "final_output": str(Path("/tmp/output/output_3D_side_by_side.mp4")),
                "frames_processed": 100,
            },
        )

    @patch("src.depth_surge_3d.processing.video_processor.update_processing_status")
    def test_finalize_processing_failure(self, mock_update_status):
        """Test failed processing finalization."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator)
        processor._settings_file = Path("/tmp/settings.json")

        settings = {"vr_format": "side_by_side", "vr_resolution": "1080p"}

        processor._finalize_processing(
            success=False,
            output_path=Path("/tmp/output"),
            video_path="/tmp/input.mp4",
            settings=settings,
            num_frames=0,
        )

        mock_update_status.assert_called_once_with(
            Path("/tmp/settings.json"), "failed", {"error": "Video creation failed"}
        )

    def test_finalize_processing_success_no_settings_file(self):
        """Test finalization without settings file."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator)
        processor._settings_file = None

        settings = {"vr_format": "side_by_side", "vr_resolution": "1080p"}

        # Should not raise exception
        with patch("src.depth_surge_3d.processing.video_processor.console_success"):
            processor._finalize_processing(
                success=True,
                output_path=Path("/tmp/output"),
                video_path="/tmp/input.mp4",
                settings=settings,
                num_frames=100,
            )


class TestStepExtractFrames:
    """Test _step_extract_frames method."""

    @patch("src.depth_surge_3d.processing.video_processor.get_frame_files")
    def test_extract_frames_skip_existing(self, mock_get_files):
        """Test frame extraction skips when frames already exist."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator)

        mock_frames_dir = MagicMock(spec=Path)
        mock_frames_dir.exists.return_value = True
        mock_get_files.return_value = [
            Path("/tmp/frame_001.png"),
            Path("/tmp/frame_002.png"),
        ]

        directories = {"frames": mock_frames_dir}
        settings = get_test_settings()
        video_properties = {}
        mock_callback = MagicMock()

        result = processor._step_extract_frames(
            "/tmp/input.mp4", directories, video_properties, settings, mock_callback
        )

        assert len(result) == 2
        mock_callback.update_progress.assert_called_once()


class TestMainProcess:
    """Test main process() method."""

    @patch.object(VideoProcessor, "_step_create_final_video")
    @patch.object(VideoProcessor, "_step_create_vr_frames")
    @patch.object(VideoProcessor, "_step_apply_distortion")
    @patch.object(VideoProcessor, "_step_create_stereo_pairs")
    @patch.object(VideoProcessor, "_step_load_frames")
    @patch.object(VideoProcessor, "_step_generate_depth_maps")
    @patch.object(VideoProcessor, "_step_extract_frames")
    @patch.object(VideoProcessor, "_setup_processing")
    @patch("src.depth_surge_3d.processing.video_processor.create_progress_tracker")
    def test_process_success_all_steps(
        self,
        mock_create_tracker,
        mock_setup,
        mock_extract,
        mock_depth,
        mock_load,
        mock_stereo,
        mock_distortion,
        mock_vr,
        mock_video,
    ):
        """Test successful full video processing workflow."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator)

        # Setup mocks
        mock_setup.return_value = (
            Path("/tmp/output"),
            {"frames": Path("/tmp/frames")},
            Path("/tmp/settings.json"),
        )
        mock_frame_files = [Path(f"/tmp/frame_{i:03d}.png") for i in range(10)]
        mock_extract.return_value = mock_frame_files
        mock_depth.return_value = np.random.rand(10, 480, 640)
        mock_load.return_value = np.random.rand(10, 480, 640, 3).astype(np.uint8)
        mock_stereo.return_value = True
        mock_distortion.return_value = True
        mock_vr.return_value = True
        mock_video.return_value = True

        mock_tracker = MagicMock()
        mock_create_tracker.return_value = mock_tracker

        video_properties = {"width": 1920, "height": 1080, "fps": 30}
        settings = {"vr_format": "side_by_side", "vr_resolution": "1080p"}

        with patch.object(processor, "_finalize_processing"):
            result = processor.process("/tmp/input.mp4", "/tmp/output", video_properties, settings)

        assert result is True
        # Verify all steps were called
        mock_setup.assert_called_once()
        mock_extract.assert_called_once()
        mock_depth.assert_called_once()
        mock_load.assert_called_once()
        mock_stereo.assert_called_once()
        mock_distortion.assert_called_once()
        mock_vr.assert_called_once()
        mock_video.assert_called_once()

    @patch.object(VideoProcessor, "_step_extract_frames")
    @patch.object(VideoProcessor, "_setup_processing")
    def test_process_fails_on_extract_frames(self, mock_setup, mock_extract):
        """Test process fails when frame extraction fails."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator)

        mock_setup.return_value = (
            Path("/tmp/output"),
            {"frames": Path("/tmp/frames")},
            Path("/tmp/settings.json"),
        )
        mock_extract.return_value = None  # Extraction failed

        video_properties = {"width": 1920, "height": 1080}
        settings = get_test_settings()

        result = processor.process("/tmp/input.mp4", "/tmp/output", video_properties, settings)

        assert result is False

    @patch.object(VideoProcessor, "_step_generate_depth_maps")
    @patch.object(VideoProcessor, "_step_extract_frames")
    @patch.object(VideoProcessor, "_setup_processing")
    @patch("src.depth_surge_3d.processing.video_processor.create_progress_tracker")
    def test_process_fails_on_depth_maps(self, mock_tracker, mock_setup, mock_extract, mock_depth):
        """Test process fails when depth map generation fails."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator)

        mock_setup.return_value = (
            Path("/tmp/output"),
            {"frames": Path("/tmp/frames")},
            None,
        )
        mock_extract.return_value = [Path("/tmp/frame_001.png")]
        mock_depth.return_value = None  # Depth map generation failed
        mock_tracker.return_value = MagicMock()

        video_properties = {"width": 1920, "height": 1080}
        settings = get_test_settings()

        result = processor.process("/tmp/input.mp4", "/tmp/output", video_properties, settings)

        assert result is False


class TestStepGenerateDepthMaps:
    """Test _step_generate_depth_maps method."""

    @patch("src.depth_surge_3d.processing.video_processor.saved_to")
    @patch("src.depth_surge_3d.processing.video_processor.step_complete")
    @patch("time.time", side_effect=[1000.0, 1005.0])
    def test_generate_depth_maps_success(self, mock_time, mock_step, mock_saved):
        """Test successful depth map generation."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        mock_estimator.estimate_depth_batch.return_value = np.random.rand(10, 480, 640)
        processor = VideoProcessor(mock_estimator)

        frame_files = [Path(f"/tmp/frame_{i:03d}.png") for i in range(10)]
        settings = get_test_settings()
        directories = {}
        mock_tracker = MagicMock()
        mock_tracker.get_step_duration.return_value = 5.0

        with patch("cv2.imread", return_value=np.random.rand(480, 640, 3).astype(np.uint8)):
            result = processor._step_generate_depth_maps(
                frame_files, settings, directories, mock_tracker
            )

        assert result is not None
        assert result.shape == (10, 480, 640)
        mock_estimator.estimate_depth_batch.assert_called_once()

    @patch("cv2.imread", return_value=None)
    def test_generate_depth_maps_imread_failure(self, mock_imread):
        """Test depth map generation with imread failure."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator)

        frame_files = [Path("/tmp/frame_001.png")]
        settings = get_test_settings()
        directories = {}
        mock_tracker = MagicMock()

        with patch.object(processor, "_handle_step_error") as mock_error:
            result = processor._step_generate_depth_maps(
                frame_files, settings, directories, mock_tracker
            )

        assert result is None
        mock_error.assert_called_once()

    @patch("cv2.imread", return_value=np.random.rand(480, 640, 3).astype(np.uint8))
    def test_generate_depth_maps_estimation_exception(self, mock_imread):
        """Test depth map generation with estimation exception."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        mock_estimator.estimate_depth_batch.side_effect = RuntimeError("GPU OOM")
        processor = VideoProcessor(mock_estimator)

        frame_files = [Path("/tmp/frame_001.png")]
        settings = get_test_settings()
        directories = {}
        mock_tracker = MagicMock()

        with patch.object(processor, "_handle_step_error") as mock_error:
            result = processor._step_generate_depth_maps(
                frame_files, settings, directories, mock_tracker
            )

        assert result is None
        mock_error.assert_called_once()


class TestStepLoadFrames:
    """Test _step_load_frames method."""

    @patch("src.depth_surge_3d.processing.video_processor.step_complete")
    @patch("time.time", side_effect=[1000.0, 1002.0])
    @patch("cv2.imread")
    def test_load_frames_success(self, mock_imread, mock_time, mock_step):
        """Test successful frame loading."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator)

        mock_imread.return_value = np.random.rand(480, 640, 3).astype(np.uint8)

        frame_files = [Path(f"/tmp/frame_{i:03d}.png") for i in range(5)]
        settings = get_test_settings()
        mock_tracker = MagicMock()
        mock_tracker.get_step_duration.return_value = 2.0

        result = processor._step_load_frames(frame_files, settings, mock_tracker)

        assert result is not None
        assert len(result) == 5
        assert mock_imread.call_count == 5

    @patch("cv2.imread", return_value=None)
    def test_load_frames_imread_failure(self, mock_imread):
        """Test frame loading with imread failure."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator)

        frame_files = [Path("/tmp/frame_001.png")]
        settings = get_test_settings()
        mock_tracker = MagicMock()

        with patch.object(processor, "_handle_step_error") as mock_error:
            result = processor._step_load_frames(frame_files, settings, mock_tracker)

        assert result is None
        mock_error.assert_called_once()


class TestStepCreateStereoPairs:
    """Test _step_create_stereo_pairs method."""

    @patch("src.depth_surge_3d.processing.video_processor.create_shifted_image")
    @patch("src.depth_surge_3d.processing.video_processor.depth_to_disparity")
    @patch("src.depth_surge_3d.processing.video_processor.resize_image")
    @patch("src.depth_surge_3d.processing.video_processor.step_complete")
    @patch("time.time", side_effect=[1000.0, 1010.0])
    @patch("cv2.imwrite", return_value=True)
    def test_create_stereo_pairs_success(
        self,
        mock_imwrite,
        mock_time,
        mock_step,
        mock_resize,
        mock_depth_disp,
        mock_shift,
    ):
        """Test successful stereo pair creation."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator)

        frames = np.random.rand(3, 480, 640, 3).astype(np.uint8)
        depth_maps = np.random.rand(3, 480, 640)
        frame_files = [Path(f"/tmp/frame_{i:03d}.png") for i in range(3)]

        mock_left_dir = MagicMock(spec=Path)
        mock_right_dir = MagicMock(spec=Path)
        directories = {
            "left_frames": mock_left_dir,
            "right_frames": mock_right_dir,
        }

        settings = get_test_settings()
        mock_tracker = MagicMock()
        mock_tracker.get_step_duration.return_value = 10.0

        mock_resize.side_effect = lambda x, *args: x
        mock_depth_disp.return_value = np.random.rand(480, 640)
        # Need 6 returns: left+right for each of 3 frames
        mock_shift.return_value = np.random.rand(480, 640, 3).astype(np.uint8)

        result = processor._step_create_stereo_pairs(
            frames, depth_maps, frame_files, directories, settings, mock_tracker
        )

        assert result is True
        assert mock_imwrite.call_count >= 6  # 3 left + 3 right


class TestStepApplyDistortion:
    """Test _step_apply_distortion method."""

    @patch("src.depth_surge_3d.processing.video_processor.apply_fisheye_distortion")
    @patch("src.depth_surge_3d.processing.video_processor.get_frame_files")
    @patch("src.depth_surge_3d.processing.video_processor.step_complete")
    @patch("time.time", side_effect=[1000.0, 1005.0])
    @patch("cv2.imread")
    @patch("cv2.imwrite", return_value=True)
    def test_apply_distortion_enabled(
        self,
        mock_imwrite,
        mock_imread,
        mock_time,
        mock_step,
        mock_get_files,
        mock_fisheye,
    ):
        """Test fisheye distortion application when enabled."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator)

        mock_imread.return_value = np.random.rand(480, 640, 3).astype(np.uint8)
        mock_fisheye.return_value = np.random.rand(480, 640, 3).astype(np.uint8)
        mock_get_files.return_value = [
            Path("/tmp/left_001.png"),
            Path("/tmp/right_001.png"),
        ]

        mock_left_dist_dir = MagicMock(spec=Path)
        mock_left_dist_dir.exists.return_value = False  # Prevent skipping
        mock_right_dist_dir = MagicMock(spec=Path)
        mock_right_dist_dir.exists.return_value = False  # Prevent skipping

        mock_left_frames_dir = MagicMock(spec=Path)
        mock_left_frames_dir.glob.return_value = [Path("/tmp/left/frame_001.png")]
        mock_right_frames_dir = MagicMock(spec=Path)
        mock_right_frames_dir.glob.return_value = [Path("/tmp/right/frame_001.png")]

        directories = {
            "left_frames": mock_left_frames_dir,
            "right_frames": mock_right_frames_dir,
            "left_distorted": mock_left_dist_dir,
            "right_distorted": mock_right_dist_dir,
        }

        settings = get_test_settings(apply_distortion=True)
        mock_tracker = MagicMock()
        mock_tracker.get_step_duration.return_value = 5.0

        result = processor._step_apply_distortion(directories, settings, mock_tracker)

        assert result is True
        assert mock_fisheye.call_count >= 2

    def test_apply_distortion_disabled(self):
        """Test skipping distortion when disabled."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator)

        directories = {}
        settings = get_test_settings()
        mock_tracker = MagicMock()

        result = processor._step_apply_distortion(directories, settings, mock_tracker)

        assert result is True


class TestStepCreateVRFrames:
    """Test _step_create_vr_frames method."""

    @patch("src.depth_surge_3d.processing.video_processor.create_vr_frame")
    @patch("src.depth_surge_3d.processing.video_processor.get_frame_files")
    @patch("src.depth_surge_3d.processing.video_processor.step_complete")
    @patch("time.time", side_effect=[1000.0, 1010.0])
    @patch("cv2.imread")
    @patch("cv2.imwrite", return_value=True)
    def test_create_vr_frames_side_by_side(
        self,
        mock_imwrite,
        mock_imread,
        mock_time,
        mock_step,
        mock_get_files,
        mock_create_vr,
    ):
        """Test VR frame creation in side-by-side format."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator)

        mock_imread.return_value = np.random.rand(480, 640, 3).astype(np.uint8)
        mock_create_vr.return_value = np.random.rand(480, 1280, 3).astype(np.uint8)
        mock_get_files.return_value = [Path("/tmp/frame_001.png")]

        mock_vr_dir = MagicMock(spec=Path)
        mock_vr_dir.exists.return_value = False  # Prevent skipping

        # Mock left/right directories to return frame files via .glob()
        mock_left_dir = MagicMock(spec=Path)
        mock_left_dir.glob.return_value = [Path("/tmp/left/frame_001.png")]
        mock_right_dir = MagicMock(spec=Path)
        mock_right_dir.glob.return_value = [Path("/tmp/right/frame_001.png")]

        directories = {
            "left_frames": mock_left_dir,
            "right_frames": mock_right_dir,
            "vr_frames": mock_vr_dir,
        }

        settings = get_test_settings()
        mock_tracker = MagicMock()
        mock_tracker.get_step_duration.return_value = 10.0

        result = processor._step_create_vr_frames(directories, settings, mock_tracker, num_frames=1)

        assert result is True
        mock_create_vr.assert_called_once()


class TestStepCreateFinalVideo:
    """Test _step_create_final_video method."""

    @patch("subprocess.run")
    @patch("src.depth_surge_3d.processing.video_processor.get_frame_files")
    @patch("src.depth_surge_3d.processing.video_processor.generate_output_filename")
    @patch("src.depth_surge_3d.processing.video_processor.step_complete")
    @patch("src.depth_surge_3d.processing.video_processor.saved_to")
    @patch("time.time", side_effect=[1000.0, 1020.0])
    def test_create_final_video_success(
        self,
        mock_time,
        mock_saved,
        mock_step,
        mock_gen_filename,
        mock_get_files,
        mock_subprocess,
    ):
        """Test successful final video creation."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator)

        mock_gen_filename.return_value = "output_3D_side_by_side.mp4"
        mock_get_files.return_value = [Path(f"/tmp/vr_{i:03d}.png") for i in range(10)]
        mock_subprocess.return_value = MagicMock(returncode=0)

        mock_vr_dir = MagicMock(spec=Path)
        mock_vr_dir.__truediv__.return_value = Path("/tmp/vr/frame_%03d.png")
        directories = {"vr_frames": mock_vr_dir}

        settings = get_test_settings()
        mock_tracker = MagicMock()
        mock_tracker.get_step_duration.return_value = 20.0

        result = processor._step_create_final_video(
            directories,
            Path("/tmp/output"),
            "/tmp/input.mp4",
            settings,
            mock_tracker,
            progress_callback=None,
        )

        assert result is True
        mock_subprocess.assert_called()

    @patch("subprocess.run")
    def test_create_final_video_ffmpeg_failure(self, mock_subprocess):
        """Test final video creation fails when FFmpeg returns error."""
        mock_estimator = MagicMock(spec=VideoDepthEstimator)
        processor = VideoProcessor(mock_estimator)

        # Simulate FFmpeg failure
        mock_subprocess.return_value = MagicMock(returncode=1, stderr="FFmpeg error")

        directories = {"vr_frames": Path("/tmp/vr")}
        settings = get_test_settings()
        mock_tracker = MagicMock()

        result = processor._step_create_final_video(
            directories,
            Path("/tmp/output"),
            "/tmp/input.mp4",
            settings,
            mock_tracker,
            progress_callback=None,
        )

        assert result is False
        mock_subprocess.assert_called_once()
