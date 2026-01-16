"""Unit tests for VideoDepthEstimator (V2)."""

import numpy as np
from unittest.mock import patch, MagicMock

from src.depth_surge_3d.models.video_depth_estimator import (
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
        estimator = create_video_depth_estimator(
            model_path=custom_path, device="cpu", metric=True
        )
        assert isinstance(estimator, VideoDepthEstimator)
        assert estimator.model_path == custom_path
        assert estimator.device == "cpu"
        assert estimator.metric is True

    def test_create_with_none_model_path(self):
        """Test factory function with None model path uses default."""
        estimator = create_video_depth_estimator(model_path=None)
        assert estimator.model_path == DEFAULT_MODEL_PATH
