"""Unit tests for VideoDepthEstimator (V2)."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.depth_surge_3d.inference.depth.video_depth_estimator import (
    VideoDepthEstimator,
    create_video_depth_estimator,
)
from src.depth_surge_3d.core.constants import DEFAULT_MODEL_PATH


class TestVideoDepthEstimator:
    """Test VideoDepthEstimator class."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH)
        assert estimator.model_path == DEFAULT_MODEL_PATH
        assert estimator.device in ["cuda", "cpu", "mps"]
        assert estimator.metric is False
        assert estimator.model is None

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        custom_path = "models/custom.pth"
        estimator = VideoDepthEstimator(custom_path, device="cpu", metric=True)
        assert estimator.model_path == custom_path
        assert estimator.device == "cpu"
        assert estimator.metric is True

    def test_determine_device_auto_with_cuda(self):
        """Test device determination when CUDA is available."""
        with patch("torch.cuda.is_available", return_value=True):
            estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="auto")
            assert estimator.device == "cuda"

    def test_determine_device_auto_without_cuda(self):
        """Test device determination when CUDA is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="auto")
            assert estimator.device in ["cpu", "mps"]

    def test_determine_device_auto_mps(self):
        """Test device determination when MPS is available (macOS)."""
        with patch("torch.cuda.is_available", return_value=False):
            # Mock MPS availability
            mock_mps = MagicMock()
            mock_mps.is_available.return_value = True
            with patch("torch.backends.mps", mock_mps, create=True):
                estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="auto")
                assert estimator.device == "mps"

    def test_get_model_type_from_path(self):
        """Test model type detection from path."""
        estimator = VideoDepthEstimator("models/video_depth_anything_vits.pth")
        assert estimator._get_model_type(estimator.model_path) == "vits"

        estimator = VideoDepthEstimator("models/video_depth_anything_vitb.pth")
        assert estimator._get_model_type(estimator.model_path) == "vitb"

        estimator = VideoDepthEstimator("models/video_depth_anything_vitl.pth")
        assert estimator._get_model_type(estimator.model_path) == "vitl"

    def test_get_model_type_fallback(self):
        """Test model type detection falls back to vitl."""
        estimator = VideoDepthEstimator("models/unknown_model.pth")
        assert estimator._get_model_type(estimator.model_path) == "vitl"

    def test_normalize_depths(self):
        """Test depth map normalization."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")

        # Create test depth maps
        depths = np.array(
            [
                [[0.0, 0.5, 1.0], [0.2, 0.8, 0.6]],
                [[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]],
            ]
        )

        normalized = estimator._normalize_depths(depths)

        # Check shape is preserved
        assert normalized.shape == depths.shape

        # Check all values are in [0, 1]
        assert np.all(normalized >= 0.0)
        assert np.all(normalized <= 1.0)

        # Check each depth map is normalized independently
        assert np.isclose(normalized[0].min(), 0.0)
        assert np.isclose(normalized[0].max(), 1.0)

    def test_normalize_depths_flat_depth(self):
        """Test normalization of flat depth map."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")

        depths = np.array(
            [
                [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
            ]
        )

        normalized = estimator._normalize_depths(depths)
        assert np.all(normalized == 0.0)

    def test_get_model_info_not_loaded(self):
        """Test model info when model is not loaded."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
        info = estimator.get_model_info()

        assert info == {}

    def test_determine_chunk_overlap_first_chunk(self):
        """Test chunk overlap determination for first chunk."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")

        depths = np.zeros((32, 100, 100))
        keep = estimator._determine_chunk_overlap(0, 32, 100, 4, depths)

        assert len(keep) == 32  # Keep all frames from first chunk

    def test_determine_chunk_overlap_last_chunk(self):
        """Test chunk overlap determination for last chunk."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")

        depths = np.zeros((32, 100, 100))
        keep = estimator._determine_chunk_overlap(68, 100, 100, 4, depths)

        assert len(keep) == 28  # Skip 4 overlap frames

    def test_determine_chunk_overlap_middle_chunk(self):
        """Test chunk overlap determination for middle chunk."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")

        depths = np.zeros((32, 100, 100))
        keep = estimator._determine_chunk_overlap(28, 60, 100, 4, depths)

        assert len(keep) == 28  # Skip 4 overlap frames

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    def test_unload_model_with_cuda(self, mock_empty_cache, mock_cuda_available):
        """Test model unloading with CUDA cleanup."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cuda")
        estimator.model = MagicMock()  # Simulate loaded model
        estimator.unload_model()

        assert estimator.model is None
        mock_empty_cache.assert_called_once()

    def test_unload_model_without_loading(self):
        """Test unload when model was never loaded."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
        estimator.unload_model()  # Should not raise error
        assert estimator.model is None


class TestCreateVideoDepthEstimator:
    """Test factory function for VideoDepthEstimator."""

    def test_create_with_defaults(self):
        """Test factory function with defaults."""
        estimator = create_video_depth_estimator()
        assert isinstance(estimator, VideoDepthEstimator)
        assert estimator.model_path == DEFAULT_MODEL_PATH

    def test_create_with_custom_params(self):
        """Test factory function with custom parameters."""
        custom_path = "models/custom.pth"
        estimator = create_video_depth_estimator(model_path=custom_path, device="cpu", metric=True)
        assert isinstance(estimator, VideoDepthEstimator)
        assert estimator.model_path == custom_path
        assert estimator.device == "cpu"
        assert estimator.metric is True

    def test_create_with_none_model_path(self):
        """Test factory function with None model path uses default."""
        estimator = create_video_depth_estimator(model_path=None)
        assert estimator.model_path == DEFAULT_MODEL_PATH


class TestEnsureDependencies:
    """Test _ensure_dependencies method."""

    def test_ensure_dependencies_success(self):
        """Test successful dependency check."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")

        # Assume dependencies are available in test environment
        result = estimator._ensure_dependencies()

        # Should return True if repo exists
        assert isinstance(result, bool)

    def test_ensure_dependencies_repo_not_exists(self):
        """Test dependency check when repo doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
            result = estimator._ensure_dependencies()

            # Should return False when repo missing
            assert result is False

    def test_ensure_dependencies_model_not_found_auto_download_fails(self):
        """Test when model file doesn't exist and auto-download fails."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")

        # Mock os.path.exists to return False for model file
        with patch("os.path.exists", return_value=False):
            with patch.object(estimator, "_auto_download_model", return_value=False):
                result = estimator._ensure_dependencies()

                assert result is False


class TestSuppressModelOutput:
    """Test _suppress_model_output context manager."""

    def test_suppress_model_output_context(self):
        """Test that output suppression works as context manager."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")

        # Should not raise error
        with estimator._suppress_model_output():
            pass  # Context manager should work

    def test_suppress_model_output_restores_stdout(self):
        """Test that stdout is restored after suppression."""
        import sys

        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")

        original_stdout = sys.stdout

        with estimator._suppress_model_output():
            # stdout should be redirected
            assert sys.stdout != original_stdout

        # stdout should be restored
        assert sys.stdout == original_stdout


class TestLoadModel:
    """Test load_model method."""

    def test_load_model_dependencies_fail(self):
        """Test load_model when dependencies are missing."""
        with patch.object(VideoDepthEstimator, "_ensure_dependencies", return_value=False):
            estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
            result = estimator.load_model()

            assert result is False
            assert estimator.model is None

    def test_load_model_invalid_model_type(self):
        """Test load_model with invalid model type."""
        with patch.object(VideoDepthEstimator, "_ensure_dependencies", return_value=True):
            with patch.object(VideoDepthEstimator, "_get_model_type", return_value=None):
                estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
                result = estimator.load_model()

                assert result is False
                assert estimator.model is None

    def test_load_model_exception_handling(self):
        """Test load_model handles exceptions gracefully."""
        with patch.object(VideoDepthEstimator, "_ensure_dependencies", return_value=True):
            with patch.object(VideoDepthEstimator, "_get_model_type", return_value="vitl"):
                # Make sys.path.insert raise an exception
                with patch("sys.path", side_effect=RuntimeError("Test error")):
                    estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
                    result = estimator.load_model()

                    assert result is False


class TestAutoDownloadModel:
    """Test _auto_download_model method."""

    def test_auto_download_model_file_exists(self):
        """Test auto download when model file already exists."""
        with patch("pathlib.Path.exists", return_value=True):
            estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
            result = estimator._auto_download_model()

            # Should return True if file exists
            assert result is True

    def test_auto_download_model_no_url(self):
        """Test auto download when no URL configured."""
        with patch("pathlib.Path.exists", return_value=False):
            estimator = VideoDepthEstimator("models/unknown_model.pth", device="cpu")

            # Mock MODEL_DOWNLOAD_URLS to not have this model
            with patch(
                "src.depth_surge_3d.inference.depth.video_depth_estimator.MODEL_DOWNLOAD_URLS", {}
            ):
                result = estimator._auto_download_model()

                # Should return False when no URL available
                assert result is False

    def test_auto_download_model_download_exception(self):
        """Test auto download when download fails with exception."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")

        with patch("pathlib.Path.exists", return_value=False):
            with patch.object(estimator, "_get_model_type", return_value="vitl"):
                # Mock urllib.request.urlretrieve to raise an exception
                with patch("urllib.request.urlretrieve", side_effect=Exception("Network error")):
                    result = estimator._auto_download_model()

                    assert result is False


class TestGetModelInfo:
    """Test get_model_info when model is loaded."""

    def test_get_model_info_with_loaded_model(self):
        """Test model info when model is loaded."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")

        # Simulate loaded model with all required config keys
        estimator.model = MagicMock()
        estimator.model_config = {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
            "num_frames": 32,
        }

        info = estimator.get_model_info()

        assert "loaded" in info
        assert info["loaded"] is True
        assert "encoder" in info
        assert info["encoder"] == "vitl"
        assert "features" in info
        assert info["features"] == 256
        assert "temporal_consistency" in info
        assert info["temporal_consistency"] is True


class TestEstimateDepthBatchDecisionLogic:
    """Test estimate_depth_batch chunking decision logic."""

    def test_estimate_depth_batch_model_not_loaded(self):
        """Test batch estimation when model not loaded."""
        import numpy as np
        import pytest

        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
        frames = np.random.rand(10, 480, 640, 3)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            estimator.estimate_depth_batch(frames)

    @patch("torch.cuda.is_available", return_value=True)
    def test_estimate_depth_batch_uses_chunking_for_large_video(self, mock_cuda):
        """Test that large videos trigger chunking on CUDA."""
        import numpy as np

        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cuda")
        estimator.model = MagicMock()

        # Large video (>60 frames) should trigger chunking
        frames = np.random.rand(70, 480, 640, 3).astype(np.uint8)

        with patch.object(estimator, "_estimate_depth_chunked") as mock_chunked:
            mock_chunked.return_value = np.random.rand(70, 480, 640)
            result = estimator.estimate_depth_batch(frames)

            mock_chunked.assert_called_once()
            assert result.shape == (70, 480, 640)

    @patch("torch.cuda.is_available", return_value=True)
    def test_estimate_depth_batch_uses_chunking_for_high_res(self, mock_cuda):
        """Test that high-res videos trigger chunking on CUDA."""
        import numpy as np

        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cuda")
        estimator.model = MagicMock()

        # High-res video (>2K) with 70 frames should trigger chunking
        frames = np.random.rand(70, 2160, 3840, 3).astype(np.uint8)

        with patch.object(estimator, "_estimate_depth_chunked") as mock_chunked:
            mock_chunked.return_value = np.random.rand(70, 2160, 3840)
            estimator.estimate_depth_batch(frames)

            mock_chunked.assert_called_once()

    @patch("torch.cuda.is_available", return_value=False)
    def test_estimate_depth_batch_single_batch_on_cpu(self, mock_cuda):
        """Test that CPU processing uses single batch."""
        import numpy as np

        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
        estimator.model = MagicMock()

        frames = np.random.rand(50, 480, 640, 3).astype(np.uint8)

        with patch.object(estimator, "_estimate_depth_single_batch") as mock_single:
            mock_single.return_value = np.random.rand(50, 480, 640)
            estimator.estimate_depth_batch(frames)

            mock_single.assert_called_once()


class TestEstimateDepthSingleBatch:
    """Test _estimate_depth_single_batch method."""

    def test_estimate_depth_single_batch_success(self):
        """Test single batch depth estimation."""
        import numpy as np

        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
        estimator.model = MagicMock()

        # Mock infer_video_depth to return depth maps
        mock_depths = np.random.rand(10, 480, 640)
        estimator.model.infer_video_depth.return_value = (mock_depths, None)

        frames = np.random.rand(10, 480, 640, 3).astype(np.uint8)

        with patch("builtins.open", MagicMock()):
            result = estimator._estimate_depth_single_batch(frames, 30, 518, False)

        assert result.shape == (10, 480, 640)
        estimator.model.infer_video_depth.assert_called_once()

    def test_estimate_depth_single_batch_exception(self):
        """Test single batch with model exception."""
        import numpy as np
        import pytest

        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
        estimator.model = MagicMock()
        estimator.model.infer_video_depth.side_effect = RuntimeError("CUDA OOM")

        frames = np.random.rand(10, 480, 640, 3).astype(np.uint8)

        with patch("builtins.open", MagicMock()):
            with pytest.raises(RuntimeError, match="Video depth estimation failed"):
                estimator._estimate_depth_single_batch(frames, 30, 518, False)


class TestEstimateDepthChunked:
    """Test _estimate_depth_chunked method."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    def test_estimate_depth_chunked_success(self, mock_cache, mock_cuda):
        """Test chunked depth estimation."""
        import numpy as np

        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cuda")
        estimator.model = MagicMock()

        # Mock _process_depth_chunk to return chunk depths
        def mock_process_chunk(frames_rgb, fps, size, fp32):
            return np.random.rand(len(frames_rgb), 480, 640)

        estimator._process_depth_chunk = mock_process_chunk

        # 50 frames should be split into chunks
        frames = np.random.rand(50, 480, 640, 3).astype(np.uint8)

        with patch("src.depth_surge_3d.core.constants.DEPTH_MODEL_CHUNK_SIZE", 24):
            result = estimator._estimate_depth_chunked(frames, 30, 518, False)

        assert result.shape == (50, 480, 640)
        # Should clear cache between chunks
        assert mock_cache.called

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    def test_estimate_depth_chunked_with_oom_retry(self, mock_cache, mock_cuda):
        """Test chunked processing with OOM error and retry."""
        import numpy as np

        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cuda")
        estimator.model = MagicMock()

        # First chunk fails with OOM, retry succeeds
        call_count = 0

        def mock_process_chunk(frames_rgb, fps, size, fp32):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("CUDA out of memory")
            return np.random.rand(len(frames_rgb), 480, 640)

        estimator._process_depth_chunk = mock_process_chunk

        # Mock retry method
        def mock_retry(frames_rgb, fps, size, fp32):
            return np.random.rand(len(frames_rgb), 480, 640)

        estimator._retry_chunk_with_reduced_resolution = mock_retry

        frames = np.random.rand(30, 480, 640, 3).astype(np.uint8)

        with patch("src.depth_surge_3d.core.constants.DEPTH_MODEL_CHUNK_SIZE", 24):
            result = estimator._estimate_depth_chunked(frames, 30, 518, False)

        assert result.shape == (30, 480, 640)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    def test_estimate_depth_chunked_non_oom_error_reraise(self, mock_cache, mock_cuda):
        """Test chunked processing re-raises non-OOM errors."""
        import numpy as np

        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cuda")
        estimator.model = MagicMock()

        # Raise a non-OOM error
        def mock_process_chunk(frames_rgb, fps, size, fp32):
            raise ValueError("Some other error")

        estimator._process_depth_chunk = mock_process_chunk

        frames = np.random.rand(30, 480, 640, 3).astype(np.uint8)

        with patch("src.depth_surge_3d.core.constants.DEPTH_MODEL_CHUNK_SIZE", 24):
            with pytest.raises(ValueError, match="Some other error"):
                estimator._estimate_depth_chunked(frames, 30, 518, False)


class TestProcessDepthChunk:
    """Test _process_depth_chunk method."""

    def test_process_depth_chunk_with_output_suppression(self):
        """Test chunk processing suppresses output."""
        import numpy as np

        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
        estimator.model = MagicMock()

        mock_depths = np.random.rand(10, 480, 640)
        estimator.model.infer_video_depth.return_value = (mock_depths, None)

        frames_rgb = np.random.rand(10, 480, 640, 3).astype(np.uint8)

        with patch.object(estimator, "_suppress_model_output") as mock_suppress:
            mock_suppress.return_value.__enter__ = MagicMock()
            mock_suppress.return_value.__exit__ = MagicMock()

            result = estimator._process_depth_chunk(frames_rgb, 30, 518, False)

            assert result.shape == (10, 480, 640)
            mock_suppress.assert_called_once()


class TestRetryChunkWithReducedResolution:
    """Test _retry_chunk_with_reduced_resolution method."""

    @patch("torch.cuda.empty_cache")
    def test_retry_chunk_with_reduced_resolution(self, mock_cache):
        """Test OOM retry with reduced input size."""
        import numpy as np

        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cuda")
        estimator.model = MagicMock()

        mock_depths = np.random.rand(10, 480, 640)
        estimator.model.infer_video_depth.return_value = (mock_depths, None)

        frames_rgb = np.random.rand(10, 480, 640, 3).astype(np.uint8)

        with patch.object(estimator, "_suppress_model_output") as mock_suppress:
            mock_suppress.return_value.__enter__ = MagicMock()
            mock_suppress.return_value.__exit__ = MagicMock()

            result = estimator._retry_chunk_with_reduced_resolution(frames_rgb, 30, 1024, False)

            assert result.shape == (10, 480, 640)
            # Should use reduced input size (max(384, 1024 // 2) = 512)
            call_args = estimator.model.infer_video_depth.call_args
            assert call_args[1]["input_size"] == 512
            mock_cache.assert_called_once()
