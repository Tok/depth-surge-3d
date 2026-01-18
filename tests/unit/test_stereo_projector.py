"""Unit tests for StereoProjector."""

from pathlib import Path
from unittest.mock import patch, MagicMock

from src.depth_surge_3d.rendering import (
    StereoProjector,
    create_stereo_projector,
)


class TestStereoProjector:
    """Test StereoProjector class."""

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_init_with_v2_default(self, mock_create_v2):
        """Test initialization with V2 (default)."""
        mock_estimator = MagicMock()
        mock_create_v2.return_value = mock_estimator

        projector = StereoProjector(
            model_path="models/test.pth",
            device="cpu",
            metric=False,
            depth_model_version="v2",
        )

        assert projector.depth_model_version == "v2"
        assert projector.depth_estimator == mock_estimator
        assert projector._model_loaded is False
        mock_create_v2.assert_called_once_with("models/test.pth", "cpu", False, 10)

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator_da3")
    def test_init_with_v3(self, mock_create_v3):
        """Test initialization with V3."""
        mock_estimator = MagicMock()
        mock_create_v3.return_value = mock_estimator

        projector = StereoProjector(
            model_path="large",
            device="cpu",
            metric=False,
            depth_model_version="v3",
        )

        assert projector.depth_model_version == "v3"
        assert projector.depth_estimator == mock_estimator
        mock_create_v3.assert_called_once_with("large", "cpu", False)

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_init_with_none_model_path_v2(self, mock_create_v2):
        """Test initialization with None model path for V2."""
        mock_estimator = MagicMock()
        mock_create_v2.return_value = mock_estimator

        projector = StereoProjector(
            model_path=None,
            device="cpu",
            depth_model_version="v2",
        )

        assert projector.depth_model_version == "v2"
        mock_create_v2.assert_called_once_with(None, "cpu", False, 10)

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator_da3")
    def test_init_with_none_model_path_v3(self, mock_create_v3):
        """Test initialization with None model path for V3."""
        mock_estimator = MagicMock()
        mock_create_v3.return_value = mock_estimator

        projector = StereoProjector(
            model_path=None,
            device="cpu",
            depth_model_version="v3",
        )

        assert projector.depth_model_version == "v3"
        mock_create_v3.assert_called_once_with(None, "cpu", False)

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_apply_default_settings(self, mock_create_v2):
        """Test default settings application."""
        mock_estimator = MagicMock()
        mock_create_v2.return_value = mock_estimator

        projector = StereoProjector(device="cpu")

        # Create a locals dict similar to what process_video receives
        test_locals = {
            "self": projector,
            "video_path": "test.mp4",
            "output_dir": "output",
            "vr_format": None,
            "baseline": None,
        }

        settings = projector._apply_default_settings(test_locals)

        # Should have defaults applied
        assert "vr_format" in settings
        assert "baseline" in settings
        assert settings["video_path"] == "test.mp4"
        assert settings["output_dir"] == "output"


class TestCreateStereoProjector:
    """Test factory function for StereoProjector."""

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_create_with_defaults(self, mock_create_v2):
        """Test factory function with defaults."""
        mock_estimator = MagicMock()
        mock_create_v2.return_value = mock_estimator

        projector = create_stereo_projector()

        assert isinstance(projector, StereoProjector)
        mock_create_v2.assert_called_once_with(None, "auto", False, 10)

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_create_with_v2(self, mock_create_v2):
        """Test factory function with V2."""
        mock_estimator = MagicMock()
        mock_create_v2.return_value = mock_estimator

        projector = create_stereo_projector(
            model_path="models/test.pth",
            device="cpu",
            metric=True,
            depth_model_version="v2",
        )

        assert isinstance(projector, StereoProjector)
        assert projector.depth_model_version == "v2"
        mock_create_v2.assert_called_once_with("models/test.pth", "cpu", True, 10)

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator_da3")
    def test_create_with_v3(self, mock_create_v3):
        """Test factory function with V3."""
        mock_estimator = MagicMock()
        mock_create_v3.return_value = mock_estimator

        projector = create_stereo_projector(
            model_path="large",
            device="cuda",
            metric=False,
            depth_model_version="v3",
        )

        assert isinstance(projector, StereoProjector)
        assert projector.depth_model_version == "v3"
        mock_create_v3.assert_called_once_with("large", "cuda", False)


class TestStereoProjectorHelpers:
    """Test helper methods of StereoProjector."""

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("subprocess.run")
    def test_check_nvenc_available_true(self, mock_run, mock_create):
        """Test NVENC availability check when NVENC is available."""
        mock_create.return_value = MagicMock()
        mock_result = MagicMock()
        mock_result.stdout = "encoders:\n  hevc_nvenc   NVIDIA NVENC H.265 encoder"
        mock_run.return_value = mock_result

        projector = StereoProjector(device="cpu")
        result = projector._check_nvenc_available()

        assert result is True
        mock_run.assert_called_once()

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("subprocess.run")
    def test_check_nvenc_available_false(self, mock_run, mock_create):
        """Test NVENC availability check when NVENC is not available."""
        mock_create.return_value = MagicMock()
        mock_result = MagicMock()
        mock_result.stdout = "encoders:\n  libx264   H.264 encoder"
        mock_run.return_value = mock_result

        projector = StereoProjector(device="cpu")
        result = projector._check_nvenc_available()

        assert result is False

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("subprocess.run")
    def test_check_nvenc_available_exception(self, mock_run, mock_create):
        """Test NVENC availability check when subprocess fails."""
        mock_create.return_value = MagicMock()
        mock_run.side_effect = Exception("FFmpeg not found")

        projector = StereoProjector(device="cpu")
        result = projector._check_nvenc_available()

        assert result is False

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_add_video_encoder_options_nvenc(self, mock_create):
        """Test adding NVENC encoder options."""
        mock_create.return_value = MagicMock()
        projector = StereoProjector(device="cpu")

        # Mock NVENC as available
        with patch.object(projector, "_check_nvenc_available", return_value=True):
            cmd = ["ffmpeg", "-y"]
            projector._add_video_encoder_options(cmd)

            assert "-c:v" in cmd
            assert "hevc_nvenc" in cmd
            assert "-preset" in cmd
            assert "p7" in cmd

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_add_video_encoder_options_software(self, mock_create):
        """Test adding software encoder options."""
        mock_create.return_value = MagicMock()
        projector = StereoProjector(device="cpu")

        # Mock NVENC as unavailable
        with patch.object(projector, "_check_nvenc_available", return_value=False):
            cmd = ["ffmpeg", "-y"]
            projector._add_video_encoder_options(cmd)

            assert "-c:v" in cmd
            assert "libx264" in cmd
            assert "-crf" in cmd
            assert "18" in cmd

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_ensure_model_loaded_success(self, mock_create):
        """Test model loading success."""
        mock_estimator = MagicMock()
        mock_estimator.load_model.return_value = True
        mock_create.return_value = mock_estimator

        projector = StereoProjector(device="cpu")
        assert projector._model_loaded is False

        result = projector._ensure_model_loaded()

        assert result is True
        assert projector._model_loaded is True
        mock_estimator.load_model.assert_called_once()

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_ensure_model_loaded_already_loaded(self, mock_create):
        """Test model loading when already loaded."""
        mock_estimator = MagicMock()
        mock_create.return_value = mock_estimator

        projector = StereoProjector(device="cpu")
        projector._model_loaded = True

        result = projector._ensure_model_loaded()

        assert result is True
        # Should not call load_model again
        mock_estimator.load_model.assert_not_called()

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_ensure_model_loaded_failure(self, mock_create):
        """Test model loading failure."""
        mock_estimator = MagicMock()
        mock_estimator.load_model.return_value = False
        mock_create.return_value = mock_estimator

        projector = StereoProjector(device="cpu")

        result = projector._ensure_model_loaded()

        assert result is False
        assert projector._model_loaded is False

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("src.depth_surge_3d.rendering.stereo_projector.validate_video_file")
    def test_validate_inputs_valid(self, mock_validate, mock_create):
        """Test input validation with valid inputs."""
        mock_create.return_value = MagicMock()
        mock_validate.return_value = True

        projector = StereoProjector(device="cpu")

        with patch("pathlib.Path.mkdir"):
            result = projector._validate_inputs("video.mp4", "/tmp/output", {})

        assert result is True
        mock_validate.assert_called_once_with("video.mp4")

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("src.depth_surge_3d.rendering.stereo_projector.validate_video_file")
    def test_validate_inputs_invalid_video(self, mock_validate, mock_create):
        """Test input validation with invalid video."""
        mock_create.return_value = MagicMock()
        mock_validate.return_value = False

        projector = StereoProjector(device="cpu")

        result = projector._validate_inputs("invalid.txt", "/tmp/output", {})

        assert result is False


class TestResolveSettings:
    """Test _resolve_settings method."""

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("src.depth_surge_3d.rendering.stereo_projector.auto_detect_resolution")
    @patch("src.depth_surge_3d.rendering.stereo_projector.validate_resolution_settings")
    @patch("src.depth_surge_3d.rendering.stereo_projector.get_resolution_dimensions")
    @patch("src.depth_surge_3d.rendering.stereo_projector.calculate_vr_output_dimensions")
    def test_resolve_settings_with_auto_resolution(
        self,
        mock_calc_vr,
        mock_get_dims,
        mock_validate,
        mock_auto_detect,
        mock_create,
    ):
        """Test settings resolution with auto resolution detection."""
        mock_create.return_value = MagicMock()

        # Mock auto-detection
        mock_auto_detect.return_value = "16x9-1080p"

        # Mock validation
        mock_validate.return_value = {
            "valid": True,
            "warnings": [],
            "recommendations": ["Use this resolution"],
        }

        # Mock dimensions
        mock_get_dims.return_value = (1920, 1080)
        mock_calc_vr.return_value = (3840, 1080)

        projector = StereoProjector(device="cpu")

        settings = {
            "vr_resolution": "auto",
            "vr_format": "side_by_side",
        }
        video_props = {"width": 1920, "height": 1080, "fps": 30}

        resolved = projector._resolve_settings(settings, video_props)

        # Should have auto-detected resolution
        assert resolved["vr_resolution"] == "16x9-1080p"
        assert resolved["per_eye_width"] == 1920
        assert resolved["per_eye_height"] == 1080
        assert resolved["vr_output_width"] == 3840
        assert resolved["vr_output_height"] == 1080
        assert resolved["source_width"] == 1920
        assert resolved["source_height"] == 1080
        assert resolved["source_fps"] == 30

        mock_auto_detect.assert_called_once_with(1920, 1080, "side_by_side")

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("src.depth_surge_3d.rendering.stereo_projector.validate_resolution_settings")
    @patch("src.depth_surge_3d.rendering.stereo_projector.get_resolution_dimensions")
    @patch("src.depth_surge_3d.rendering.stereo_projector.calculate_vr_output_dimensions")
    def test_resolve_settings_with_manual_resolution(
        self,
        mock_calc_vr,
        mock_get_dims,
        mock_validate,
        mock_create,
    ):
        """Test settings resolution with manual resolution."""
        mock_create.return_value = MagicMock()

        # Mock validation
        mock_validate.return_value = {
            "valid": True,
            "warnings": [],
            "recommendations": [],
        }

        # Mock dimensions
        mock_get_dims.return_value = (2048, 2048)
        mock_calc_vr.return_value = (4096, 2048)

        projector = StereoProjector(device="cpu")

        settings = {
            "vr_resolution": "square-2k",
            "vr_format": "side_by_side",
        }
        video_props = {"width": 1920, "height": 1080, "fps": 60}

        resolved = projector._resolve_settings(settings, video_props)

        # Should keep manual resolution
        assert resolved["vr_resolution"] == "square-2k"
        assert resolved["per_eye_width"] == 2048
        assert resolved["per_eye_height"] == 2048
        assert resolved["vr_output_width"] == 4096
        assert resolved["vr_output_height"] == 2048

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("src.depth_surge_3d.rendering.stereo_projector.auto_detect_resolution")
    @patch("src.depth_surge_3d.rendering.stereo_projector.validate_resolution_settings")
    @patch("src.depth_surge_3d.rendering.stereo_projector.get_resolution_dimensions")
    @patch("src.depth_surge_3d.rendering.stereo_projector.calculate_vr_output_dimensions")
    def test_resolve_settings_with_validation_warnings(
        self,
        mock_calc_vr,
        mock_get_dims,
        mock_validate,
        mock_auto_detect,
        mock_create,
    ):
        """Test settings resolution with validation warnings."""
        mock_create.return_value = MagicMock()

        mock_auto_detect.return_value = "cinema-4k"

        # Mock validation with warnings
        mock_validate.return_value = {
            "valid": False,
            "warnings": ["Resolution too high for source", "Consider downscaling"],
            "recommendations": ["Use 16x9-1080p instead"],
        }

        mock_get_dims.return_value = (4096, 2160)
        mock_calc_vr.return_value = (8192, 2160)

        projector = StereoProjector(device="cpu")

        settings = {
            "vr_resolution": "auto",
            "vr_format": "side_by_side",
        }
        video_props = {"width": 1280, "height": 720, "fps": 30}

        # Should not raise error even with invalid validation
        resolved = projector._resolve_settings(settings, video_props)

        assert resolved is not None
        assert "per_eye_width" in resolved


class TestSuperSampleResolution:
    """Test determine_super_sample_resolution method."""

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_super_sample_none(self, mock_create):
        """Test super sample with 'none' mode."""
        mock_create.return_value = MagicMock()
        projector = StereoProjector(device="cpu")

        width, height = projector.determine_super_sample_resolution(1280, 720, "none")

        assert width == 1280
        assert height == 720

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_super_sample_1080p(self, mock_create):
        """Test super sample with '1080p' mode."""
        mock_create.return_value = MagicMock()
        projector = StereoProjector(device="cpu")

        width, height = projector.determine_super_sample_resolution(1280, 720, "1080p")

        assert width == 1920
        assert height == 1080

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_super_sample_4k(self, mock_create):
        """Test super sample with '4k' mode."""
        mock_create.return_value = MagicMock()
        projector = StereoProjector(device="cpu")

        width, height = projector.determine_super_sample_resolution(1280, 720, "4k")

        assert width == 3840
        assert height == 2160

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_super_sample_auto_720p_source(self, mock_create):
        """Test super sample auto mode with 720p source."""
        mock_create.return_value = MagicMock()
        projector = StereoProjector(device="cpu")

        width, height = projector.determine_super_sample_resolution(1280, 720, "auto")

        # 720p should upscale to 1080p
        assert width == 1920
        assert height == 1080

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_super_sample_auto_1080p_source(self, mock_create):
        """Test super sample auto mode with 1080p source."""
        mock_create.return_value = MagicMock()
        projector = StereoProjector(device="cpu")

        width, height = projector.determine_super_sample_resolution(1920, 1080, "auto")

        # 1080p should upscale to 4K
        assert width == 3840
        assert height == 2160

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_super_sample_auto_4k_source(self, mock_create):
        """Test super sample auto mode with 4K source."""
        mock_create.return_value = MagicMock()
        projector = StereoProjector(device="cpu")

        width, height = projector.determine_super_sample_resolution(3840, 2160, "auto")

        # 4K should keep original resolution
        assert width == 3840
        assert height == 2160

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_super_sample_invalid_mode(self, mock_create):
        """Test super sample with invalid mode."""
        mock_create.return_value = MagicMock()
        projector = StereoProjector(device="cpu")

        width, height = projector.determine_super_sample_resolution(1280, 720, "invalid")

        # Invalid mode should keep original resolution
        assert width == 1280
        assert height == 720


class TestVROutputResolution:
    """Test determine_vr_output_resolution method."""

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_vr_output_auto_side_by_side(self, mock_create):
        """Test VR output resolution with auto and side-by-side format."""
        mock_create.return_value = MagicMock()
        projector = StereoProjector(device="cpu")

        width, height = projector.determine_vr_output_resolution(1920, 1080, "auto", "side_by_side")

        # Auto should use original as per-eye, side-by-side doubles width
        assert width == 3840
        assert height == 1080

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_vr_output_auto_over_under(self, mock_create):
        """Test VR output resolution with auto and over-under format."""
        mock_create.return_value = MagicMock()
        projector = StereoProjector(device="cpu")

        width, height = projector.determine_vr_output_resolution(1920, 1080, "auto", "over_under")

        # Auto should use original as per-eye, over-under doubles height
        assert width == 1920
        assert height == 2160

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_vr_output_manual_resolution(self, mock_create):
        """Test VR output resolution with manual resolution."""
        mock_create.return_value = MagicMock()

        projector = StereoProjector(device="cpu")

        # Use a known resolution preset
        width, height = projector.determine_vr_output_resolution(
            1920, 1080, "16x9-1080p", "side_by_side"
        )

        # 16x9-1080p is 1920x1080 per eye, side-by-side doubles width
        assert width == 3840
        assert height == 1080


class TestModelDelegation:
    """Test model delegation methods."""

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_get_model_info(self, mock_create):
        """Test get_model_info delegation."""
        mock_estimator = MagicMock()
        mock_estimator.get_model_info.return_value = {
            "loaded": True,
            "encoder": "vitl",
        }
        mock_create.return_value = mock_estimator

        projector = StereoProjector(device="cpu")
        info = projector.get_model_info()

        assert info["loaded"] is True
        assert info["encoder"] == "vitl"
        mock_estimator.get_model_info.assert_called_once()

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_unload_model(self, mock_create):
        """Test unload_model delegation."""
        mock_estimator = MagicMock()
        mock_create.return_value = mock_estimator

        projector = StereoProjector(device="cpu")
        projector._model_loaded = True

        projector.unload_model()

        assert projector._model_loaded is False
        mock_estimator.unload_model.assert_called_once()


class TestProcessVideoErrorPaths:
    """Test error handling in process_video method."""

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("src.depth_surge_3d.rendering.stereo_projector.validate_video_file")
    def test_process_video_invalid_input(self, mock_validate, mock_create):
        """Test process_video with invalid video input."""
        mock_create.return_value = MagicMock()
        mock_validate.return_value = False

        projector = StereoProjector(device="cpu")
        result = projector.process_video("invalid.txt", "/tmp/output")

        assert result is False
        mock_validate.assert_called_once_with("invalid.txt")

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("src.depth_surge_3d.rendering.stereo_projector.validate_video_file")
    def test_process_video_model_load_failure(self, mock_validate, mock_create):
        """Test process_video when model fails to load."""
        mock_estimator = MagicMock()
        mock_estimator.load_model.return_value = False
        mock_create.return_value = mock_estimator
        mock_validate.return_value = True

        projector = StereoProjector(device="cpu")

        with patch("pathlib.Path.mkdir"):
            result = projector.process_video("test.mp4", "/tmp/output")

        assert result is False
        mock_estimator.load_model.assert_called_once()

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("src.depth_surge_3d.rendering.stereo_projector.validate_video_file")
    @patch("src.depth_surge_3d.rendering.stereo_projector.get_video_properties")
    def test_process_video_invalid_video_properties(
        self, mock_get_props, mock_validate, mock_create
    ):
        """Test process_video when video properties cannot be read."""
        mock_estimator = MagicMock()
        mock_estimator.load_model.return_value = True
        mock_create.return_value = mock_estimator
        mock_validate.return_value = True
        mock_get_props.return_value = None

        projector = StereoProjector(device="cpu")

        with patch("pathlib.Path.mkdir"):
            result = projector.process_video("test.mp4", "/tmp/output")

        assert result is False
        mock_get_props.assert_called_once_with("test.mp4")

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("src.depth_surge_3d.rendering.stereo_projector.validate_video_file")
    @patch("src.depth_surge_3d.io.operations.get_video_properties")
    def test_process_video_exception_handling(self, mock_get_props, mock_validate, mock_create):
        """Test process_video handles exceptions gracefully."""
        mock_estimator = MagicMock()
        mock_estimator.load_model.side_effect = RuntimeError("Model error")
        mock_create.return_value = mock_estimator
        mock_validate.return_value = True

        projector = StereoProjector(device="cpu")

        with patch("pathlib.Path.mkdir"):
            result = projector.process_video("test.mp4", "/tmp/output")

        assert result is False


class TestExtractFrames:
    """Test extract_frames method."""

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("src.depth_surge_3d.io.operations.get_video_properties")
    @patch("subprocess.run")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.glob")
    def test_extract_frames_success_with_cuda(
        self, mock_glob, mock_mkdir, mock_run, mock_get_props, mock_create
    ):
        """Test successful frame extraction with CUDA."""
        mock_create.return_value = MagicMock()
        mock_get_props.return_value = {"width": 1920, "height": 1080, "fps": 30}

        # Mock successful CUDA extraction
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        # Mock glob to return frame files
        mock_glob.return_value = [
            Path("frames/frame_000001.png"),
            Path("frames/frame_000002.png"),
        ]

        projector = StereoProjector(device="cpu")
        frames = projector.extract_frames("test.mp4", "/tmp/output")

        assert len(frames) == 2
        assert mock_run.called
        # Should try CUDA command first
        assert "-hwaccel" in mock_run.call_args[0][0]
        assert "cuda" in mock_run.call_args[0][0]

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("src.depth_surge_3d.io.operations.get_video_properties")
    @patch("subprocess.run")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.glob")
    def test_extract_frames_cuda_fallback(
        self, mock_glob, mock_mkdir, mock_run, mock_get_props, mock_create
    ):
        """Test frame extraction with CUDA fallback to CPU."""
        mock_create.return_value = MagicMock()
        mock_get_props.return_value = {"width": 1920, "height": 1080, "fps": 30}

        # First call (CUDA) fails, second call (CPU) succeeds
        mock_result_cuda = MagicMock()
        mock_result_cuda.returncode = 1
        mock_result_cpu = MagicMock()
        mock_result_cpu.returncode = 0

        mock_run.side_effect = [mock_result_cuda, mock_result_cpu]

        mock_glob.return_value = [Path("frames/frame_000001.png")]

        projector = StereoProjector(device="cpu")
        frames = projector.extract_frames("test.mp4", "/tmp/output")

        assert len(frames) == 1
        assert mock_run.call_count == 2  # CUDA + CPU fallback

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("src.depth_surge_3d.io.operations.get_video_properties")
    def test_extract_frames_invalid_video(self, mock_get_props, mock_create):
        """Test frame extraction with invalid video properties."""
        mock_create.return_value = MagicMock()
        mock_get_props.return_value = None

        projector = StereoProjector(device="cpu")

        import pytest

        with pytest.raises(ValueError, match="Could not read video properties"):
            projector.extract_frames("invalid.mp4", "/tmp/output")

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("src.depth_surge_3d.io.operations.get_video_properties")
    @patch("subprocess.run")
    @patch("pathlib.Path.mkdir")
    def test_extract_frames_with_time_range(
        self, mock_mkdir, mock_run, mock_get_props, mock_create
    ):
        """Test frame extraction with start/end time."""
        mock_create.return_value = MagicMock()
        mock_get_props.return_value = {"width": 1920, "height": 1080, "fps": 30}

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        projector = StereoProjector(device="cpu")

        with patch("pathlib.Path.glob", return_value=[]):
            projector.extract_frames(
                "test.mp4", "/tmp/output", start_time="00:10", end_time="00:20"
            )

        # Verify time range args were included
        cmd_args = mock_run.call_args[0][0]
        assert "-ss" in cmd_args
        assert "00:10" in cmd_args
        assert "-to" in cmd_args
        assert "00:20" in cmd_args


class TestCreateOutputVideo:
    """Test create_output_video method."""

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("subprocess.run")
    @patch("pathlib.Path.glob")
    def test_create_output_video_with_nvenc(self, mock_glob, mock_run, mock_create):
        """Test output video creation with NVENC."""
        mock_create.return_value = MagicMock()

        mock_glob.return_value = [
            Path("vr/frame_000001.png"),
            Path("vr/frame_000002.png"),
        ]

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.side_effect = [
            # First call: check NVENC availability
            MagicMock(stdout="encoders:\n  hevc_nvenc   NVIDIA NVENC"),
            # Second call: create video
            mock_result,
        ]

        projector = StereoProjector(device="cpu")
        result = projector.create_output_video(
            "/tmp/vr_frames", "/tmp/output.mp4", "/tmp/original.mp4"
        )

        assert result is True
        # Verify NVENC was used
        create_cmd = mock_run.call_args_list[1][0][0]
        assert "hevc_nvenc" in create_cmd

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("subprocess.run")
    @patch("pathlib.Path.glob")
    def test_create_output_video_with_software_encoder(self, mock_glob, mock_run, mock_create):
        """Test output video creation with software encoder."""
        mock_create.return_value = MagicMock()

        mock_glob.return_value = [Path("vr/frame_000001.png")]

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.side_effect = [
            # First call: NVENC not available
            MagicMock(stdout="encoders:\n  libx264   H.264"),
            # Second call: create video
            mock_result,
        ]

        projector = StereoProjector(device="cpu")
        result = projector.create_output_video(
            "/tmp/vr_frames", "/tmp/output.mp4", "/tmp/original.mp4", preserve_audio=False
        )

        assert result is True
        # Verify software encoder was used
        create_cmd = mock_run.call_args_list[1][0][0]
        assert "libx264" in create_cmd

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("pathlib.Path.glob")
    def test_create_output_video_no_frames(self, mock_glob, mock_create):
        """Test output video creation with no frames."""
        mock_create.return_value = MagicMock()
        mock_glob.return_value = []

        projector = StereoProjector(device="cpu")

        import pytest

        with pytest.raises(ValueError, match="No VR frames found"):
            projector.create_output_video("/tmp/empty", "/tmp/output.mp4", "/tmp/original.mp4")

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("subprocess.run")
    @patch("pathlib.Path.glob")
    def test_create_output_video_with_audio_and_time_range(self, mock_glob, mock_run, mock_create):
        """Test output video creation with audio and time range."""
        mock_create.return_value = MagicMock()

        mock_glob.return_value = [Path("vr/frame_000001.png")]

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.side_effect = [
            MagicMock(stdout="encoders:\n  libx264"),
            mock_result,
        ]

        projector = StereoProjector(device="cpu")
        result = projector.create_output_video(
            "/tmp/vr_frames",
            "/tmp/output.mp4",
            "/tmp/original.mp4",
            start_time="00:05",
            end_time="00:15",
            preserve_audio=True,
        )

        assert result is True
        # Verify audio and time range args
        create_cmd = mock_run.call_args_list[1][0][0]
        assert "-ss" in create_cmd
        assert "00:05" in create_cmd
        assert "-to" in create_cmd
        assert "00:15" in create_cmd
        assert "-c:a" in create_cmd

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("subprocess.run")
    @patch("pathlib.Path.glob")
    def test_create_output_video_ffmpeg_error(self, mock_glob, mock_run, mock_create):
        """Test output video creation with FFmpeg error."""
        mock_create.return_value = MagicMock()

        mock_glob.return_value = [Path("vr/frame_000001.png")]

        # First call succeeds (NVENC check), second fails (video creation)
        import subprocess

        mock_run.side_effect = [
            MagicMock(stdout="encoders:\n  libx264"),
            subprocess.CalledProcessError(1, "ffmpeg", stderr="FFmpeg error"),
        ]

        projector = StereoProjector(device="cpu")
        result = projector.create_output_video(
            "/tmp/vr_frames", "/tmp/output.mp4", "/tmp/original.mp4"
        )

        assert result is False


class TestProcessImage:
    """Test process_image method."""

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("cv2.imread")
    @patch("cv2.imwrite")
    @patch("pathlib.Path.mkdir")
    def test_process_image_success(self, mock_mkdir, mock_imwrite, mock_imread, mock_create):
        """Test successful image processing."""
        import numpy as np

        mock_estimator = MagicMock()
        mock_estimator.load_model.return_value = True
        mock_estimator.estimate_depth_batch.return_value = np.array([np.random.rand(480, 640)])
        mock_create.return_value = mock_estimator

        # Mock imread to return a valid image
        mock_imread.return_value = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_imwrite.return_value = True

        projector = StereoProjector(device="cpu")
        result = projector.process_image("test.jpg", "/tmp/output")

        assert result is True
        mock_estimator.load_model.assert_called_once()
        mock_estimator.estimate_depth_batch.assert_called_once()
        # Should save 4 images: left, right, vr, depth
        assert mock_imwrite.call_count == 4

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("cv2.imread")
    def test_process_image_model_load_failure(self, mock_imread, mock_create):
        """Test process_image when model fails to load."""
        mock_estimator = MagicMock()
        mock_estimator.load_model.return_value = False
        mock_create.return_value = mock_estimator

        projector = StereoProjector(device="cpu")
        result = projector.process_image("test.jpg", "/tmp/output")

        assert result is False

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("cv2.imread")
    def test_process_image_invalid_image(self, mock_imread, mock_create):
        """Test process_image with invalid image file."""
        mock_estimator = MagicMock()
        mock_estimator.load_model.return_value = True
        mock_create.return_value = mock_estimator

        mock_imread.return_value = None

        projector = StereoProjector(device="cpu")
        result = projector.process_image("invalid.jpg", "/tmp/output")

        assert result is False

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("cv2.imread")
    def test_process_image_depth_estimation_failure(self, mock_imread, mock_create):
        """Test process_image when depth estimation fails."""
        import numpy as np

        mock_estimator = MagicMock()
        mock_estimator.load_model.return_value = True
        mock_estimator.estimate_depth_batch.return_value = None
        mock_create.return_value = mock_estimator

        mock_imread.return_value = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        projector = StereoProjector(device="cpu")
        result = projector.process_image("test.jpg", "/tmp/output")

        assert result is False

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("cv2.imread")
    def test_process_image_with_custom_settings(self, mock_imread, mock_create):
        """Test process_image with custom settings."""
        import numpy as np

        mock_estimator = MagicMock()
        mock_estimator.load_model.return_value = True
        mock_estimator.estimate_depth_batch.return_value = np.array([np.random.rand(480, 640)])
        mock_create.return_value = mock_estimator

        mock_imread.return_value = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        projector = StereoProjector(device="cpu")

        with patch("cv2.imwrite", return_value=True):
            with patch("pathlib.Path.mkdir"):
                result = projector.process_image(
                    "test.jpg",
                    "/tmp/output",
                    baseline=0.065,
                    focal_length=800,
                    depth_resolution="720",
                )

        assert result is True
        # Verify custom depth resolution was used (batch estimation called)
        mock_estimator.estimate_depth_batch.assert_called_once()

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    @patch("cv2.imread")
    def test_process_image_exception_handling(self, mock_imread, mock_create):
        """Test process_image handles exceptions gracefully."""
        import numpy as np

        mock_estimator = MagicMock()
        mock_estimator.load_model.return_value = True
        mock_estimator.estimate_depth_batch.side_effect = RuntimeError("Processing error")
        mock_create.return_value = mock_estimator

        mock_imread.return_value = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        projector = StereoProjector(device="cpu")
        result = projector.process_image("test.jpg", "/tmp/output")

        assert result is False
