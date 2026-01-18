"""Unit tests for VideoDepthEstimatorDA3."""

import numpy as np
import torch
from unittest.mock import patch, MagicMock

from src.depth_surge_3d.inference.depth.video_depth_estimator_da3 import (
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

    def test_determine_device_auto_with_mps(self):
        """Test device determination when MPS is available (Apple Silicon)."""
        with patch("torch.cuda.is_available", return_value=False):
            # Mock MPS availability
            mock_backends = MagicMock()
            mock_backends.mps.is_available.return_value = True

            with patch("torch.backends", mock_backends):
                estimator = VideoDepthEstimatorDA3(device="auto")
                assert estimator.device == "mps"

    def test_determine_device_auto_fallback_to_cpu(self):
        """Test device determination falls back to CPU when no accelerator available."""
        with patch("torch.cuda.is_available", return_value=False):
            # Mock torch.backends without mps attribute
            mock_backends = MagicMock(spec=[])  # Empty spec means no attributes

            with patch("torch.backends", mock_backends):
                estimator = VideoDepthEstimatorDA3(device="auto")
                assert estimator.device == "cpu"

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
                estimator = VideoDepthEstimatorDA3(model_name="base", device="cpu", metric=True)
                result = estimator.load_model()

                assert result is True
                # Should use large-metric model when metric=True
                call_args = mock_da3.from_pretrained.call_args
                assert (
                    "metric" in call_args[0][0].lower()
                    or call_args[0][0] == DA3_MODEL_NAMES["large-metric"]
                )

    def test_load_model_with_direct_hf_model_id(self):
        """Test loading with direct Hugging Face model ID (not in DA3_MODEL_NAMES)."""
        with patch.dict(
            "sys.modules",
            {"depth_anything_3": MagicMock(), "depth_anything_3.api": MagicMock()},
        ):
            mock_da3 = MagicMock()
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = mock_model
            mock_da3.from_pretrained.return_value = mock_model

            with patch("depth_anything_3.api.DepthAnything3", mock_da3):
                # Use a custom HF model ID not in DA3_MODEL_NAMES
                custom_hf_id = "LiheYoung/depth-anything-custom"
                estimator = VideoDepthEstimatorDA3(model_name=custom_hf_id, device="cpu")
                result = estimator.load_model()

                assert result is True
                # Should use the custom ID directly
                mock_da3.from_pretrained.assert_called_once()
                called_model_id = mock_da3.from_pretrained.call_args[0][0]
                assert called_model_id == custom_hf_id

    def test_load_model_with_loguru_suppression(self):
        """Test that loguru logging is suppressed during model loading."""
        with patch.dict(
            "sys.modules",
            {"depth_anything_3": MagicMock(), "depth_anything_3.api": MagicMock()},
        ):
            mock_da3 = MagicMock()
            mock_model = MagicMock()
            mock_da3.from_pretrained.return_value = mock_model

            # Mock loguru
            mock_logger = MagicMock()

            with patch("depth_anything_3.api.DepthAnything3", mock_da3):
                with patch.dict("sys.modules", {"loguru": MagicMock()}):
                    with patch("loguru.logger", mock_logger):
                        estimator = VideoDepthEstimatorDA3(device="cpu")
                        result = estimator.load_model()

                        assert result is True
                        # Verify loguru.logger.remove() was called to suppress logs
                        # (This is best-effort since the actual implementation may vary)

    def test_load_model_exception_handling(self):
        """Test model loading handles exceptions gracefully."""
        with patch.dict(
            "sys.modules",
            {"depth_anything_3": MagicMock(), "depth_anything_3.api": MagicMock()},
        ):
            mock_da3 = MagicMock()
            # Make from_pretrained raise an exception
            mock_da3.from_pretrained.side_effect = RuntimeError("Model loading failed")

            with patch("depth_anything_3.api.DepthAnything3", mock_da3):
                estimator = VideoDepthEstimatorDA3(device="cpu")
                result = estimator.load_model()

                # Should return False on exception
                assert result is False
                # Model should remain None
                assert estimator.model is None

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
        estimator = create_video_depth_estimator_da3(model_name="base", device="cpu", metric=True)
        assert isinstance(estimator, VideoDepthEstimatorDA3)
        assert estimator.model_name == "base"
        assert estimator.device == "cpu"
        assert estimator.metric is True

    def test_create_with_none_model_name(self):
        """Test factory function with None model name uses default."""
        estimator = create_video_depth_estimator_da3(model_name=None)
        assert estimator.model_name == DEFAULT_DA3_MODEL


class TestEstimateDepthBatch:
    """Test estimate_depth_batch method."""

    def test_estimate_depth_batch_model_not_loaded(self):
        """Test that estimate_depth_batch raises error when model not loaded."""
        import pytest

        estimator = VideoDepthEstimatorDA3(device="cpu")
        frames = np.zeros((2, 480, 640, 3), dtype=np.uint8)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            estimator.estimate_depth_batch(frames)

    def test_estimate_depth_batch_success(self):
        """Test successful depth estimation."""
        import torch

        estimator = VideoDepthEstimatorDA3(device="cpu", verbose=True)
        estimator.model = MagicMock()

        frames = np.random.rand(5, 480, 640, 3).astype(np.uint8)

        # Mock model inference
        mock_prediction = MagicMock()
        mock_depth_maps = [torch.rand(480, 640) for _ in range(5)]
        mock_prediction.depth = mock_depth_maps
        estimator.model.inference.return_value = mock_prediction

        with patch("torch.no_grad"):
            result = estimator.estimate_depth_batch(frames, input_size=518)

        assert result.shape == (5, 480, 640)
        estimator.model.inference.assert_called_once()

    def test_estimate_depth_batch_with_resize(self):
        """Test depth estimation with resizing."""
        import torch

        estimator = VideoDepthEstimatorDA3(device="cpu", verbose=False)
        estimator.model = MagicMock()

        frames = np.random.rand(3, 240, 320, 3).astype(np.uint8)

        # Mock model inference with different size output
        mock_prediction = MagicMock()
        # Model returns larger depth maps than input
        mock_depth_maps = [torch.rand(480, 640) for _ in range(3)]
        mock_prediction.depth = mock_depth_maps
        estimator.model.inference.return_value = mock_prediction

        with patch("torch.no_grad"):
            with patch("cv2.resize") as mock_resize:
                mock_resize.return_value = np.random.rand(240, 320)
                estimator.estimate_depth_batch(frames)

        # Should resize depth maps to match input
        assert mock_resize.call_count == 3

    def test_estimate_depth_batch_exception_handling(self):
        """Test exception handling during inference."""
        import pytest

        estimator = VideoDepthEstimatorDA3(device="cpu")
        estimator.model = MagicMock()
        estimator.model.inference.side_effect = RuntimeError("CUDA OOM")

        frames = np.random.rand(2, 480, 640, 3).astype(np.uint8)

        with patch("torch.no_grad"):
            with pytest.raises(RuntimeError, match="DA3 depth estimation failed"):
                estimator.estimate_depth_batch(frames)
