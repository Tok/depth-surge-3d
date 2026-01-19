"""Unit tests for package __init__.py"""

import sys
from unittest.mock import patch, MagicMock
import pytest


class TestPackageMetadata:
    """Test package metadata attributes."""

    def test_version_attribute(self):
        """Test that __version__ is defined."""
        from src import depth_surge_3d

        assert hasattr(depth_surge_3d, "__version__")
        assert isinstance(depth_surge_3d.__version__, str)
        assert depth_surge_3d.__version__ == "0.9.1"

    def test_author_attribute(self):
        """Test that __author__ is defined."""
        from src import depth_surge_3d

        assert hasattr(depth_surge_3d, "__author__")
        assert isinstance(depth_surge_3d.__author__, str)

    def test_description_attribute(self):
        """Test that __description__ is defined."""
        from src import depth_surge_3d

        assert hasattr(depth_surge_3d, "__description__")
        assert isinstance(depth_surge_3d.__description__, str)


class TestLazyImport:
    """Test lazy import functionality."""

    def test_lazy_import_stereo_projector(self):
        """Test lazy import of StereoProjector."""
        from src import depth_surge_3d

        # Access StereoProjector through lazy loading
        StereoProjector = depth_surge_3d.StereoProjector

        # Should return the class
        assert StereoProjector is not None
        assert hasattr(StereoProjector, "__init__")

    def test_getattr_stereo_projector(self):
        """Test __getattr__ returns StereoProjector correctly."""
        from src import depth_surge_3d

        # This should trigger __getattr__
        stereo_projector_class = getattr(depth_surge_3d, "StereoProjector")

        assert stereo_projector_class is not None

    def test_getattr_invalid_attribute(self):
        """Test __getattr__ raises AttributeError for invalid attributes."""
        from src import depth_surge_3d
        import pytest

        with pytest.raises(AttributeError, match="has no attribute 'InvalidAttribute'"):
            getattr(depth_surge_3d, "InvalidAttribute")

    def test_getattr_invalid_attribute_direct(self):
        """Test accessing invalid attribute directly raises AttributeError."""
        from src import depth_surge_3d
        import pytest

        with pytest.raises(AttributeError):
            _ = depth_surge_3d.NonExistentAttribute


class TestExportedConstants:
    """Test that constants are properly exported."""

    def test_default_settings_exported(self):
        """Test that DEFAULT_SETTINGS is exported."""
        from src import depth_surge_3d

        assert hasattr(depth_surge_3d, "DEFAULT_SETTINGS")
        assert isinstance(depth_surge_3d.DEFAULT_SETTINGS, dict)

    def test_vr_resolutions_exported(self):
        """Test that VR_RESOLUTIONS is exported."""
        from src import depth_surge_3d

        assert hasattr(depth_surge_3d, "VR_RESOLUTIONS")
        assert isinstance(depth_surge_3d.VR_RESOLUTIONS, dict)

    def test_model_configs_exported(self):
        """Test that MODEL_CONFIGS is exported."""
        from src import depth_surge_3d

        assert hasattr(depth_surge_3d, "MODEL_CONFIGS")
        assert isinstance(depth_surge_3d.MODEL_CONFIGS, dict)

    def test_all_attribute(self):
        """Test that __all__ is defined and contains expected exports."""
        from src import depth_surge_3d

        assert hasattr(depth_surge_3d, "__all__")
        assert "StereoProjector" in depth_surge_3d.__all__
        assert "DEFAULT_SETTINGS" in depth_surge_3d.__all__
        assert "VR_RESOLUTIONS" in depth_surge_3d.__all__
        assert "MODEL_CONFIGS" in depth_surge_3d.__all__


class TestStereoProjectorInstantiation:
    """Test that StereoProjector can be instantiated through package import."""

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_video_depth_estimator")
    def test_create_stereo_projector_from_package(self, mock_create):
        """Test creating StereoProjector instance from package level."""
        mock_create.return_value = MagicMock()

        from src import depth_surge_3d

        # Should be able to instantiate StereoProjector
        projector = depth_surge_3d.StereoProjector(device="cpu")

        assert projector is not None
        assert hasattr(projector, "process_video")
        assert hasattr(projector, "process_image")


class TestPythonVersionCheck:
    """Test Python version compatibility checks."""

    def test_python_version_too_new(self):
        """Test that Python 3.13+ raises RuntimeError."""
        # Mock sys.version_info to simulate Python 3.13
        with patch.object(sys, "version_info", (3, 13, 0, "final", 0)):
            with pytest.raises(
                RuntimeError,
                match=r"(?s)Depth Surge 3D requires Python 3\.9-3\.12.*Python 3\.13.*not yet supported",
            ):
                # Force reimport to trigger version check
                import importlib
                import src.depth_surge_3d

                importlib.reload(src.depth_surge_3d)

    def test_python_version_too_old(self):
        """Test that Python < 3.9 raises RuntimeError."""
        # Mock sys.version_info to simulate Python 3.8
        with patch.object(sys, "version_info", (3, 8, 10, "final", 0)):
            with pytest.raises(
                RuntimeError,
                match=r"(?s)Depth Surge 3D requires Python 3\.9 or newer.*Python 3\.8",
            ):
                # Force reimport to trigger version check
                import importlib
                import src.depth_surge_3d

                importlib.reload(src.depth_surge_3d)

    def test_python_version_supported(self):
        """Test that supported Python versions (3.9-3.12) work correctly."""
        # Test with various supported versions
        for minor_version in [9, 10, 11, 12]:
            with patch.object(sys, "version_info", (3, minor_version, 0, "final", 0)):
                # Should not raise any errors
                import importlib
                import src.depth_surge_3d

                importlib.reload(src.depth_surge_3d)
                assert src.depth_surge_3d.__version__ == "0.9.1"
