"""Tests for upscaler module."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.depth_surge_3d.inference.upscaling.upscaler import (
    ImageUpscaler,
    RealESRGANUpscaler,
    create_upscaler,
    default_init_weights,
    make_layer,
    pixel_unshuffle,
    ResidualDenseBlock,
    RRDB,
    RRDBNet,
)


class TestCreateUpscaler:
    """Test the create_upscaler factory function."""

    def test_create_upscaler_none(self):
        """Test creating upscaler with 'none' returns None."""
        result = create_upscaler("none")
        assert result is None

    def test_create_upscaler_x2(self):
        """Test creating x2 Real-ESRGAN upscaler."""
        result = create_upscaler("x2", "cpu")
        assert isinstance(result, RealESRGANUpscaler)
        assert result.model_name == "x2"
        assert result.scale == 2
        assert result.device == "cpu"

    def test_create_upscaler_x4(self):
        """Test creating x4 Real-ESRGAN upscaler."""
        result = create_upscaler("x4", "cpu")
        assert isinstance(result, RealESRGANUpscaler)
        assert result.model_name == "x4"
        assert result.scale == 4
        assert result.device == "cpu"

    def test_create_upscaler_x4_conservative(self):
        """Test creating x4-conservative Real-ESRGAN upscaler."""
        result = create_upscaler("x4-conservative", "cpu")
        assert isinstance(result, RealESRGANUpscaler)
        assert result.model_name == "x4-conservative"
        assert result.scale == 4
        assert result.device == "cpu"

    def test_create_upscaler_invalid_model(self):
        """Test creating upscaler with invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown upscale model"):
            create_upscaler("invalid_model")

    @patch("torch.cuda.is_available", return_value=True)
    def test_create_upscaler_auto_device_cuda(self, mock_cuda):
        """Test auto device selection chooses CUDA when available."""
        result = create_upscaler("x4", "auto")
        assert result.device == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    def test_create_upscaler_auto_device_cpu(self, mock_cuda):
        """Test auto device selection chooses CPU when CUDA unavailable."""
        result = create_upscaler("x4", "auto")
        assert result.device == "cpu"


class TestImageUpscaler:
    """Test ImageUpscaler base class."""

    @patch("torch.cuda.is_available", return_value=True)
    def test_determine_device_auto_cuda(self, mock_cuda):
        """Test device determination with auto and CUDA available."""
        upscaler = ImageUpscaler(device="auto")
        assert upscaler.device == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    def test_determine_device_auto_cpu(self, mock_cuda):
        """Test device determination with auto and no CUDA."""
        upscaler = ImageUpscaler(device="auto")
        assert upscaler.device == "cpu"

    def test_determine_device_explicit_cuda(self):
        """Test explicit CUDA device selection."""
        upscaler = ImageUpscaler(device="cuda")
        assert upscaler.device == "cuda"

    def test_determine_device_explicit_cpu(self):
        """Test explicit CPU device selection."""
        upscaler = ImageUpscaler(device="cpu")
        assert upscaler.device == "cpu"

    def test_load_model_not_implemented(self):
        """Test load_model raises NotImplementedError."""
        upscaler = ImageUpscaler()
        with pytest.raises(NotImplementedError):
            upscaler.load_model()

    def test_upscale_image_not_implemented(self):
        """Test upscale_image raises NotImplementedError."""
        upscaler = ImageUpscaler()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(NotImplementedError):
            upscaler.upscale_image(image)

    def test_unload_model_no_model(self):
        """Test unloading when no model is loaded."""
        upscaler = ImageUpscaler()
        upscaler.unload_model()  # Should not raise
        assert upscaler.model is None

    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.is_available", return_value=True)
    def test_unload_model_with_cuda(self, mock_cuda, mock_empty_cache):
        """Test unloading model with CUDA clears cache."""
        upscaler = ImageUpscaler(device="cuda")
        upscaler.model = Mock()
        upscaler.unload_model()
        assert upscaler.model is None
        mock_empty_cache.assert_called_once()

    def test_unload_model_with_cpu(self):
        """Test unloading model with CPU."""
        upscaler = ImageUpscaler(device="cpu")
        upscaler.model = Mock()
        upscaler.unload_model()
        assert upscaler.model is None


class TestRealESRGANUpscaler:
    """Test RealESRGANUpscaler class."""

    def test_init_x2(self):
        """Test initialization with x2 model."""
        upscaler = RealESRGANUpscaler("x2", "cpu")
        assert upscaler.model_name == "x2"
        assert upscaler.scale == 2
        assert upscaler.device == "cpu"
        assert upscaler.model is None

    def test_init_x4(self):
        """Test initialization with x4 model."""
        upscaler = RealESRGANUpscaler("x4", "cpu")
        assert upscaler.model_name == "x4"
        assert upscaler.scale == 4
        assert upscaler.device == "cpu"

    def test_init_x4_conservative(self):
        """Test initialization with x4-conservative model."""
        upscaler = RealESRGANUpscaler("x4-conservative", "cpu")
        assert upscaler.model_name == "x4-conservative"
        assert upscaler.scale == 4
        assert upscaler.device == "cpu"

    def test_upscale_image_no_model(self):
        """Test upscaling without loaded model raises RuntimeError."""
        upscaler = RealESRGANUpscaler("x4", "cpu")
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="Model not loaded"):
            upscaler.upscale_image(image)

    @patch("src.depth_surge_3d.inference.upscaling.upscaler.RRDBNet")
    @patch("torch.load")
    def test_load_model_x2_download(self, mock_torch_load, mock_rrdb):
        """Test loading x2 model downloads when not cached."""
        upscaler = RealESRGANUpscaler("x2", "cpu")

        # Mock model weights
        mock_weights = {"params": {}}
        mock_torch_load.return_value = mock_weights

        # Mock RRDBNet instance
        mock_model = MagicMock()
        mock_rrdb.return_value = mock_model

        # Mock cache path doesn't exist and mock helper methods
        with patch.object(Path, "exists", return_value=False):
            with patch.object(Path, "mkdir"):
                with patch.object(upscaler, "_download_model_weights"):
                    with patch.object(upscaler, "_verify_model_checksum"):
                        result = upscaler.load_model()

        assert result is True
        mock_rrdb.assert_called_once_with(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2
        )
        mock_model.load_state_dict.assert_called_once()
        mock_model.eval.assert_called_once()

    @patch("src.depth_surge_3d.inference.upscaling.upscaler.RRDBNet")
    @patch("torch.load")
    def test_load_model_x4_cached(self, mock_torch_load, mock_rrdb):
        """Test loading x4 model uses cache when available."""
        upscaler = RealESRGANUpscaler("x4", "cpu")

        # Mock model weights
        mock_weights = {"params_ema": {}}
        mock_torch_load.return_value = mock_weights

        # Mock RRDBNet instance
        mock_model = MagicMock()
        mock_rrdb.return_value = mock_model

        # Mock cache path exists
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "mkdir"):
                with patch.object(upscaler, "_verify_model_checksum"):
                    result = upscaler.load_model()

        assert result is True
        mock_rrdb.assert_called_once_with(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
        )

    @patch("src.depth_surge_3d.inference.upscaling.upscaler.RRDBNet")
    @patch("torch.load")
    def test_load_model_x4_conservative(self, mock_torch_load, mock_rrdb):
        """Test loading x4-conservative model."""
        upscaler = RealESRGANUpscaler("x4-conservative", "cpu")

        # Mock model weights
        mock_weights = {"params": {}}
        mock_torch_load.return_value = mock_weights

        # Mock RRDBNet instance
        mock_model = MagicMock()
        mock_rrdb.return_value = mock_model

        # Mock cache path exists
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "mkdir"):
                with patch.object(upscaler, "_verify_model_checksum"):
                    result = upscaler.load_model()

        assert result is True
        mock_rrdb.assert_called_once_with(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
        )

    @patch("urllib.request.urlretrieve", side_effect=Exception("Network error"))
    def test_load_model_download_failure(self, mock_retrieve):
        """Test loading model handles download failure."""
        upscaler = RealESRGANUpscaler("x4", "cpu")

        # Mock cache path doesn't exist
        with patch.object(Path, "exists", return_value=False):
            with patch.object(Path, "mkdir"):
                result = upscaler.load_model()

        assert result is False

    def test_upscale_image_with_mock_model(self):
        """Test upscaling image with mocked model."""
        upscaler = RealESRGANUpscaler("x4", "cpu")

        # Create mock model
        mock_model = MagicMock()
        mock_output = torch.rand(1, 3, 400, 400)  # 4x upscaled from 100x100
        mock_model.return_value = mock_output
        upscaler.model = mock_model

        # Create test image (100x100 BGR)
        input_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Upscale
        result = upscaler.upscale_image(input_image)

        # Verify output
        assert isinstance(result, np.ndarray)
        assert result.shape == (400, 400, 3)
        assert result.dtype == np.uint8
        mock_model.assert_called_once()


class TestNeuralNetworkComponents:
    """Test neural network building blocks."""

    def test_pixel_unshuffle_scale2(self):
        """Test pixel unshuffle with scale 2."""
        x = torch.rand(1, 3, 4, 4)
        result = pixel_unshuffle(x, scale=2)
        assert result.shape == (1, 12, 2, 2)  # c * scale^2, h/scale, w/scale

    def test_pixel_unshuffle_scale4(self):
        """Test pixel unshuffle with scale 4."""
        x = torch.rand(1, 3, 8, 8)
        result = pixel_unshuffle(x, scale=4)
        assert result.shape == (1, 48, 2, 2)

    def test_make_layer(self):
        """Test make_layer creates sequential blocks."""

        class DummyBlock(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

        layers = make_layer(DummyBlock, 3, value=42)
        assert isinstance(layers, torch.nn.Sequential)
        assert len(layers) == 3
        assert all(isinstance(layer, DummyBlock) for layer in layers)

    def test_residual_dense_block_forward(self):
        """Test ResidualDenseBlock forward pass."""
        block = ResidualDenseBlock(num_feat=64, num_grow_ch=32)
        x = torch.rand(1, 64, 16, 16)
        output = block(x)
        assert output.shape == x.shape  # Should preserve dimensions

    def test_rrdb_forward(self):
        """Test RRDB forward pass."""
        block = RRDB(num_feat=64, num_grow_ch=32)
        x = torch.rand(1, 64, 16, 16)
        output = block(x)
        assert output.shape == x.shape

    def test_rrdbnet_forward_scale4(self):
        """Test RRDBNet forward pass with scale 4."""
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=2, num_grow_ch=32
        )
        x = torch.rand(1, 3, 16, 16)
        output = model(x)
        assert output.shape == (1, 3, 64, 64)  # 4x upscale

    def test_rrdbnet_forward_scale2(self):
        """Test RRDBNet forward pass with scale 2."""
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=2, num_grow_ch=32
        )
        x = torch.rand(1, 3, 16, 16)
        output = model(x)
        assert output.shape == (1, 3, 32, 32)  # 2x upscale

    def test_default_init_weights_single_module(self):
        """Test default_init_weights with single module."""
        module = torch.nn.Conv2d(3, 64, 3, 1, 1)
        default_init_weights(module, scale=0.1)
        # Just verify it doesn't raise - weight initialization is hard to test

    def test_default_init_weights_module_list(self):
        """Test default_init_weights with list of modules."""
        modules = [torch.nn.Conv2d(3, 64, 3, 1, 1), torch.nn.Conv2d(64, 64, 3, 1, 1)]
        default_init_weights(modules, scale=0.1)
        # Just verify it doesn't raise


class TestIntegration:
    """Integration tests for complete upscaling workflow."""

    def test_full_workflow_creation_and_cleanup(self):
        """Test complete workflow from creation to cleanup."""
        # Create upscaler
        upscaler = create_upscaler("x4", "cpu")
        assert upscaler is not None
        assert isinstance(upscaler, RealESRGANUpscaler)
        assert upscaler.model_name == "x4"
        assert upscaler.scale == 4
        assert upscaler.device == "cpu"
        assert upscaler.model is None

        # Verify unload works when no model loaded
        upscaler.unload_model()
        assert upscaler.model is None

    def test_none_upscaler_returns_none(self):
        """Test that 'none' model returns None."""
        upscaler = create_upscaler("none")
        assert upscaler is None
