"""Integration tests for end-to-end processing."""

import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.integration
class TestEndToEnd:
    """Integration tests for complete processing pipeline."""

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator")
    def test_projector_creation_v2(self, mock_create):
        """Test complete projector creation with V2."""
        from src.depth_surge_3d.core.stereo_projector import create_stereo_projector

        mock_estimator = MagicMock()
        mock_create.return_value = mock_estimator

        projector = create_stereo_projector(
            model_path="models/test.pth",
            device="cpu",
            depth_model_version="v2",
        )

        assert projector is not None
        assert projector.depth_model_version == "v2"

    @patch("src.depth_surge_3d.core.stereo_projector.create_video_depth_estimator_da3")
    def test_projector_creation_v3(self, mock_create):
        """Test complete projector creation with V3."""
        from src.depth_surge_3d.core.stereo_projector import create_stereo_projector

        mock_estimator = MagicMock()
        mock_create.return_value = mock_estimator

        projector = create_stereo_projector(
            model_path="large",
            device="cpu",
            depth_model_version="v3",
        )

        assert projector is not None
        assert projector.depth_model_version == "v3"

    def test_constants_consistency(self):
        """Test that constants are consistent across modules."""
        from src.depth_surge_3d.core.constants import (
            DA3_MODEL_NAMES,
            DEFAULT_DA3_MODEL,
            VR_RESOLUTIONS,
            DEFAULT_SETTINGS,
        )

        # Check DA3 default model exists
        assert DEFAULT_DA3_MODEL in DA3_MODEL_NAMES

        # Check VR resolutions are valid
        for res_name, (width, height) in VR_RESOLUTIONS.items():
            assert width > 0
            assert height > 0

        # Check default settings completeness
        required_keys = ["baseline", "focal_length", "vr_format"]
        for key in required_keys:
            assert key in DEFAULT_SETTINGS

    def test_model_factory_functions(self):
        """Test model factory functions are accessible."""
        from src.depth_surge_3d.models.video_depth_estimator import (
            create_video_depth_estimator,
        )
        from src.depth_surge_3d.models.video_depth_estimator_da3 import (
            create_video_depth_estimator_da3,
        )

        # Functions should be callable
        assert callable(create_video_depth_estimator)
        assert callable(create_video_depth_estimator_da3)
