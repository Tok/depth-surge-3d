"""Test that all entry points can import correctly."""

import sys
from pathlib import Path


class TestEntryPointImports:
    """Test that all entry point files can import successfully."""

    def test_cli_imports(self):
        """Test that CLI script (depth_surge_3d.py) imports work."""
        # Add project root to path
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root / "src"))

        # Test CLI imports (without actually running the script)
        from depth_surge_3d.utils.domain.depth_cache import (
            get_cache_size,
            get_cache_dir,
            clear_cache,
        )

        assert callable(get_cache_size)
        assert callable(get_cache_dir)
        assert callable(clear_cache)

    def test_app_imports(self):
        """Test that web UI (app.py) imports work."""
        # Add project root to path
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root / "src"))

        # Test all imports used in app.py
        from src.depth_surge_3d.utils.system.console import warning
        from depth_surge_3d.processing import VideoProcessor
        from depth_surge_3d.rendering import create_stereo_projector
        from depth_surge_3d.utils.domain.resolution import (
            get_resolution_dimensions,
            calculate_vr_output_dimensions,
            auto_detect_resolution,
        )
        from depth_surge_3d.utils.path_utils import sanitize_filename

        assert callable(warning)
        assert VideoProcessor is not None
        assert callable(create_stereo_projector)
        assert callable(get_resolution_dimensions)
        assert callable(calculate_vr_output_dimensions)
        assert callable(auto_detect_resolution)
        assert callable(sanitize_filename)

    def test_all_public_modules_importable(self):
        """Test that all public modules can be imported."""
        # Add project root to path
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root / "src"))

        # Test all major module imports
        from depth_surge_3d import __version__
        from depth_surge_3d.core import constants
        from depth_surge_3d.inference import (
            create_video_depth_estimator,
            create_video_depth_estimator_da3,
            create_upscaler,
        )
        from depth_surge_3d.io import operations
        from depth_surge_3d.processing import (  # noqa: F401
            VideoProcessor,
            ProcessingOrchestrator,
            DepthMapProcessor,
            StereoPairGenerator,
            DistortionProcessor,
            FrameUpscalerProcessor,
            VRFrameAssembler,
            VideoEncoder,
        )
        from depth_surge_3d.rendering import create_stereo_projector
        from depth_surge_3d.utils import (  # noqa: F401
            batch_analysis,
            path_utils,
        )
        from depth_surge_3d.utils.domain import (  # noqa: F401
            depth_cache,
            progress,
            resolution,
        )
        from depth_surge_3d.utils.imaging import (  # noqa: F401
            image_processing,
            video_processing,
        )
        from depth_surge_3d.utils.system import (  # noqa: F401
            check_cuda,
            console,
            vram_manager,
        )

        # Basic sanity checks
        assert __version__ is not None
        assert constants is not None
        assert callable(create_video_depth_estimator)
        assert callable(create_video_depth_estimator_da3)
        assert callable(create_upscaler)
        assert operations is not None
        assert VideoProcessor is not None
        assert ProcessingOrchestrator is not None
        assert callable(create_stereo_projector)


class TestNoOrphanedImports:
    """Test that no code uses old import paths."""

    def test_no_old_console_imports(self):
        """Verify no code uses old utils.console path."""
        project_root = Path(__file__).parent.parent.parent

        # Check source files
        bad_imports = []
        for py_file in project_root.glob("**/*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            content = py_file.read_text()
            # Check for old import pattern (but not the new one)
            if "utils.console" in content and "utils.system.console" not in content:
                bad_imports.append(str(py_file.relative_to(project_root)))

        assert bad_imports == [], f"Found old console imports in: {bad_imports}"

    def test_no_old_resolution_imports(self):
        """Verify no code uses old utils.resolution path."""
        project_root = Path(__file__).parent.parent.parent

        bad_imports = []
        for py_file in project_root.glob("**/*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            content = py_file.read_text()
            # Check for old import pattern (but not the new one)
            if "utils.resolution" in content and "utils.domain.resolution" not in content:
                bad_imports.append(str(py_file.relative_to(project_root)))

        assert bad_imports == [], f"Found old resolution imports in: {bad_imports}"

    def test_no_old_depth_cache_imports(self):
        """Verify no code uses old utils.depth_cache path."""
        project_root = Path(__file__).parent.parent.parent

        bad_imports = []
        for py_file in project_root.glob("**/*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            content = py_file.read_text()
            # Check for old import pattern (but not the new one)
            if "utils.depth_cache" in content and "utils.domain.depth_cache" not in content:
                bad_imports.append(str(py_file.relative_to(project_root)))

        assert bad_imports == [], f"Found old depth_cache imports in: {bad_imports}"

    def test_no_old_video_processor_imports(self):
        """Verify no code uses old processing.video_processor path."""
        project_root = Path(__file__).parent.parent.parent

        bad_imports = []
        for py_file in project_root.glob("**/*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue
            if "processing/orchestration/video_processor.py" in str(py_file):
                continue  # Skip the actual module file
            if "test_imports.py" in str(py_file):
                continue  # Skip this test file itself

            content = py_file.read_text()
            # Check for old import pattern
            if "processing.video_processor import VideoProcessor" in content:
                bad_imports.append(str(py_file.relative_to(project_root)))

        assert bad_imports == [], f"Found old imports in: {bad_imports}"
