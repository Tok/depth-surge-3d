"""Unit tests for StereoProjector."""

from unittest.mock import patch, MagicMock

from src.depth_surge_3d.core.stereo_projector import (
    StereoProjector,
    create_stereo_projector,
)


class TestStereoProjector:
    """Test StereoProjector class."""

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator")
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
        mock_create_v2.assert_called_once_with("models/test.pth", "cpu", False)

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator_da3")
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

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator")
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
        mock_create_v2.assert_called_once_with(None, "cpu", False)

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator_da3")
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

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator")
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

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator")
    def test_create_with_defaults(self, mock_create_v2):
        """Test factory function with defaults."""
        mock_estimator = MagicMock()
        mock_create_v2.return_value = mock_estimator

        projector = create_stereo_projector()

        assert isinstance(projector, StereoProjector)
        mock_create_v2.assert_called_once_with(None, "auto", False)

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator")
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
        mock_create_v2.assert_called_once_with("models/test.pth", "cpu", True)

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator_da3")
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

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator")
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

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator")
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

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator")
    @patch("subprocess.run")
    def test_check_nvenc_available_exception(self, mock_run, mock_create):
        """Test NVENC availability check when subprocess fails."""
        mock_create.return_value = MagicMock()
        mock_run.side_effect = Exception("FFmpeg not found")

        projector = StereoProjector(device="cpu")
        result = projector._check_nvenc_available()

        assert result is False

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator")
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

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator")
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

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator")
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

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator")
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

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator")
    def test_ensure_model_loaded_failure(self, mock_create):
        """Test model loading failure."""
        mock_estimator = MagicMock()
        mock_estimator.load_model.return_value = False
        mock_create.return_value = mock_estimator

        projector = StereoProjector(device="cpu")

        result = projector._ensure_model_loaded()

        assert result is False
        assert projector._model_loaded is False

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator")
    @patch("src.depth_surge_3d.core.stereo_projector.validate_video_file")
    def test_validate_inputs_valid(self, mock_validate, mock_create):
        """Test input validation with valid inputs."""
        mock_create.return_value = MagicMock()
        mock_validate.return_value = True

        projector = StereoProjector(device="cpu")

        with patch("pathlib.Path.mkdir"):
            result = projector._validate_inputs("video.mp4", "/tmp/output", {})

        assert result is True
        mock_validate.assert_called_once_with("video.mp4")

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator")
    @patch("src.depth_surge_3d.core.stereo_projector.validate_video_file")
    def test_validate_inputs_invalid_video(self, mock_validate, mock_create):
        """Test input validation with invalid video."""
        mock_create.return_value = MagicMock()
        mock_validate.return_value = False

        projector = StereoProjector(device="cpu")

        result = projector._validate_inputs("invalid.txt", "/tmp/output", {})

        assert result is False


class TestResolveSettings:
    """Test _resolve_settings method."""

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator")
    @patch("src.depth_surge_3d.core.stereo_projector.auto_detect_resolution")
    @patch("src.depth_surge_3d.core.stereo_projector.validate_resolution_settings")
    @patch("src.depth_surge_3d.core.stereo_projector.get_resolution_dimensions")
    @patch("src.depth_surge_3d.core.stereo_projector.calculate_vr_output_dimensions")
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

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator")
    @patch("src.depth_surge_3d.core.stereo_projector.validate_resolution_settings")
    @patch("src.depth_surge_3d.core.stereo_projector.get_resolution_dimensions")
    @patch("src.depth_surge_3d.core.stereo_projector.calculate_vr_output_dimensions")
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

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator")
    @patch("src.depth_surge_3d.core.stereo_projector.auto_detect_resolution")
    @patch("src.depth_surge_3d.core.stereo_projector.validate_resolution_settings")
    @patch("src.depth_surge_3d.core.stereo_projector.get_resolution_dimensions")
    @patch("src.depth_surge_3d.core.stereo_projector.calculate_vr_output_dimensions")
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
