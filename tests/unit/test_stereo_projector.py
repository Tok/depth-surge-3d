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
