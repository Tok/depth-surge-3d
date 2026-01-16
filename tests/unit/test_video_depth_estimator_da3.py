"""Unit tests for VideoDepthEstimatorDA3."""

import numpy as np
import torch
from unittest.mock import patch, MagicMock

from src.depth_surge_3d.models.video_depth_estimator_da3 import (
    VideoDepthEstimatorDA3,
    create_video_depth_estimator_da3,
)
from src.depth_surge_3d.core.constants import DA3_MODEL_NAMES, DEFAULT_DA3_MODEL


class TestVideoDepthEstimatorDA3:
    """Test VideoDepthEstimatorDA3 class."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        estimator = VideoDepthEstimatorDA3()
        assert estimator.model_name == DEFAULT_DA3_MODEL
        assert estimator.device in ["cuda", "cpu", "mps"]
        assert estimator.metric is False
        assert estimator.model is None

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        estimator = VideoDepthEstimatorDA3(model_name="base", device="cpu", metric=True)
        assert estimator.model_name == "base"
        assert estimator.device == "cpu"
        assert estimator.metric is True

    def test_determine_device_auto_with_cuda(self):
        """Test device determination when CUDA is available."""
        with patch("torch.cuda.is_available", return_value=True):
            estimator = VideoDepthEstimatorDA3(device="auto")
            assert estimator.device == "cuda"

    def test_determine_device_auto_without_cuda(self):
        """Test device determination when CUDA is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            estimator = VideoDepthEstimatorDA3(device="auto")
            assert estimator.device in ["cpu", "mps"]

    def test_determine_device_explicit(self):
        """Test explicit device specification."""
        estimator = VideoDepthEstimatorDA3(device="cpu")
        assert estimator.device == "cpu"

    def test_load_model_success(self):
        """Test successful model loading."""
        with patch.dict(
            "sys.modules",
            {"depth_anything_3": MagicMock(), "depth_anything_3.api": MagicMock()},
        ):
            mock_da3 = MagicMock()
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model  # Return self for chaining
            mock_model.eval.return_value = mock_model  # Return self for chaining
            mock_da3.from_pretrained.return_value = mock_model

            with patch("depth_anything_3.api.DepthAnything3", mock_da3):
                estimator = VideoDepthEstimatorDA3(device="cpu")
                result = estimator.load_model()

                assert result is True
                assert estimator.model is not None
                mock_da3.from_pretrained.assert_called_once()
                mock_model.to.assert_called_once_with(device="cpu")
                mock_model.eval.assert_called_once()

    def test_load_model_metric_override(self):
        """Test metric model override."""
        with patch.dict(
            "sys.modules",
            {"depth_anything_3": MagicMock(), "depth_anything_3.api": MagicMock()},
        ):
            mock_da3 = MagicMock()
            mock_model = MagicMock()
            mock_da3.from_pretrained.return_value = mock_model

            with patch("depth_anything_3.api.DepthAnything3", mock_da3):
                estimator = VideoDepthEstimatorDA3(
                    model_name="base", device="cpu", metric=True
                )
                result = estimator.load_model()

                assert result is True
                # Should use large-metric model when metric=True
                call_args = mock_da3.from_pretrained.call_args
                assert (
                    "metric" in call_args[0][0].lower()
                    or call_args[0][0] == DA3_MODEL_NAMES["large-metric"]
                )

    def test_load_model_import_error(self):
        """Test model loading when DA3 is not installed."""
        import sys

        estimator = VideoDepthEstimatorDA3(device="cpu")

        # Remove depth_anything_3 from sys.modules to simulate it not being installed
        modules_to_remove = [
            key for key in sys.modules.keys() if key.startswith("depth_anything_3")
        ]
        saved_modules = {key: sys.modules[key] for key in modules_to_remove}

        try:
            # Remove the modules
            for key in modules_to_remove:
                del sys.modules[key]

            # Mock sys.modules to prevent reimport
            with patch.dict("sys.modules", {"depth_anything_3": None}):
                result = estimator.load_model()
                assert result is False
        finally:
            # Restore the modules
            sys.modules.update(saved_modules)

    def test_normalize_depths_with_numpy_array(self):
        """Test depth normalization with numpy array."""
        estimator = VideoDepthEstimatorDA3(device="cpu")

        # Create test depth maps with known values
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
        assert np.isclose(normalized[1].min(), 0.0)
        assert np.isclose(normalized[1].max(), 1.0)

    def test_normalize_depths_with_torch_tensor(self):
        """Test depth normalization with torch tensor."""
        estimator = VideoDepthEstimatorDA3(device="cpu")

        # Create test depth maps as torch tensor
        depths = torch.tensor(
            [
                [[0.0, 0.5, 1.0], [0.2, 0.8, 0.6]],
                [[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]],
            ]
        )

        normalized = estimator._normalize_depths(depths)

        # Check result is numpy array
        assert isinstance(normalized, np.ndarray)

        # Check all values are in [0, 1]
        assert np.all(normalized >= 0.0)
        assert np.all(normalized <= 1.0)

    def test_normalize_depths_flat_depth(self):
        """Test normalization of flat depth map (all same values)."""
        estimator = VideoDepthEstimatorDA3(device="cpu")

        # All values the same
        depths = np.array(
            [
                [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
            ]
        )

        normalized = estimator._normalize_depths(depths)

        # Should result in all zeros
        assert np.all(normalized == 0.0)

    def test_get_model_info_not_loaded(self):
        """Test model info when model is not loaded."""
        estimator = VideoDepthEstimatorDA3(model_name="base", device="cpu")
        info = estimator.get_model_info()

        assert info["model_name"] == "base"
        assert info["model_version"] == "Depth Anything V3"
        assert info["device"] == "cpu"
        assert info["metric"] is False
        assert info["loaded"] is False
        assert info["temporal_consistency"] is False
        assert info["memory_efficient"] is True

    def test_get_model_info_loaded(self):
        """Test model info when model is loaded."""
        with patch.dict(
            "sys.modules",
            {"depth_anything_3": MagicMock(), "depth_anything_3.api": MagicMock()},
        ):
            mock_da3 = MagicMock()
            mock_model = MagicMock()
            mock_da3.from_pretrained.return_value = mock_model

            with patch("depth_anything_3.api.DepthAnything3", mock_da3):
                estimator = VideoDepthEstimatorDA3(model_name="large", device="cpu")
                estimator.load_model()
                info = estimator.get_model_info()

                assert info["loaded"] is True

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    def test_unload_model_with_cuda(self, mock_empty_cache, mock_cuda_available):
        """Test model unloading with CUDA cleanup."""
        with patch.dict(
            "sys.modules",
            {"depth_anything_3": MagicMock(), "depth_anything_3.api": MagicMock()},
        ):
            mock_da3 = MagicMock()
            mock_model = MagicMock()
            mock_da3.from_pretrained.return_value = mock_model

            with patch("depth_anything_3.api.DepthAnything3", mock_da3):
                estimator = VideoDepthEstimatorDA3(device="cuda")
                estimator.load_model()
                estimator.unload_model()

                assert estimator.model is None
                mock_empty_cache.assert_called_once()

    def test_unload_model_without_loading(self):
        """Test unload when model was never loaded."""
        estimator = VideoDepthEstimatorDA3(device="cpu")
        estimator.unload_model()  # Should not raise error
        assert estimator.model is None


class TestCreateVideoDepthEstimatorDA3:
    """Test factory function for VideoDepthEstimatorDA3."""

    def test_create_with_defaults(self):
        """Test factory function with defaults."""
        estimator = create_video_depth_estimator_da3()
        assert isinstance(estimator, VideoDepthEstimatorDA3)
        assert estimator.model_name == DEFAULT_DA3_MODEL

    def test_create_with_custom_params(self):
        """Test factory function with custom parameters."""
        estimator = create_video_depth_estimator_da3(
            model_name="base", device="cpu", metric=True
        )
        assert isinstance(estimator, VideoDepthEstimatorDA3)
        assert estimator.model_name == "base"
        assert estimator.device == "cpu"
        assert estimator.metric is True

    def test_create_with_none_model_name(self):
        """Test factory function with None model name uses default."""
        estimator = create_video_depth_estimator_da3(model_name=None)
        assert estimator.model_name == DEFAULT_DA3_MODEL
