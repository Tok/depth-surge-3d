"""Unit tests for VideoDepthEstimator (V2)."""

from pathlib import Path

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
        """One normalization range for the whole sequence (no per-frame rescaling)."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")

        depths = np.array(
            [
                [[0.0, 0.5, 1.0], [0.2, 0.8, 0.6]],
                [[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]],
            ]
        )

        normalized = estimator._normalize_depths(depths)

        assert normalized.shape == depths.shape
        assert np.all(normalized >= 0.0) and np.all(normalized <= 1.0)
        # frame 0 is entirely nearer-is-smaller than frame 1, so it stays darker
        assert normalized[0].max() < normalized[1].max()
        assert normalized[0].mean() < normalized[1].mean()

    def test_temporally_consistent_flag(self):
        assert VideoDepthEstimator.temporally_consistent is True

    def test_metric_output_inverted(self):
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu", metric=True)
        depths = np.array([[[1.0, 2.0], [4.0, 8.0]]], dtype=np.float32)

        inverse = estimator._to_inverse_depth(depths)

        assert inverse[0, 0, 0] > inverse[0, 1, 1]  # 1 m is nearer than 8 m

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

    # These tests must never hit the network or touch models/: always mock urlretrieve and
    # use tmp_path (an earlier version re-downloaded the real 1.5 GB model on every run).

    def test_auto_download_success(self, tmp_path):
        target = tmp_path / "Video-Depth-Anything-Large" / "video_depth_anything_vitl.pth"
        estimator = VideoDepthEstimator(str(target), device="cpu")

        def fake_download(url, path):
            Path(path).write_bytes(b"weights")

        with patch("urllib.request.urlretrieve", side_effect=fake_download) as mock_get:
            assert estimator._auto_download_model() is True

        assert target.read_bytes() == b"weights"
        assert not target.with_name(target.name + ".part").exists()
        assert mock_get.call_args[0][0].endswith(
            "Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth"
        )

    def test_metric_model_uses_metric_repo(self, tmp_path):
        """Regression: metric files were downloaded from the relative-model URL."""
        target = tmp_path / "video_depth_anything_vitb_metric.pth"
        estimator = VideoDepthEstimator(str(target), device="cpu", metric=True)

        assert estimator._download_url() == (
            "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Base/"
            "resolve/main/metric_video_depth_anything_vitb.pth"
        )

    def test_metric_detected_from_filename(self, tmp_path):
        estimator = VideoDepthEstimator(str(tmp_path / "video_depth_anything_vits_metric.pth"))
        assert "Metric-Video-Depth-Anything-Small" in estimator._download_url()

    def test_auto_download_model_no_url(self, tmp_path):
        estimator = VideoDepthEstimator(str(tmp_path / "unknown_model.pth"), device="cpu")

        with patch(
            "src.depth_surge_3d.inference.depth.video_depth_estimator.MODEL_DOWNLOAD_URLS", {}
        ):
            with patch("urllib.request.urlretrieve") as mock_get:
                assert estimator._auto_download_model() is False
        mock_get.assert_not_called()

    def test_auto_download_model_download_exception(self, tmp_path):
        target = tmp_path / "video_depth_anything_vitl.pth"
        estimator = VideoDepthEstimator(str(target), device="cpu")

        def failing_download(url, path):
            Path(path).write_bytes(b"partial")
            raise Exception("Network error")

        with patch("urllib.request.urlretrieve", side_effect=failing_download):
            assert estimator._auto_download_model() is False

        assert not target.exists()  # no half-downloaded file left behind as a "model"
        assert not target.with_name(target.name + ".part").exists()


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


class TestEstimateDepthBatch:
    """Test estimate_depth_batch."""

    def _estimator(self, depths=None):
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
        estimator.model = MagicMock()
        if depths is not None:
            estimator.model.infer_video_depth.return_value = (depths, 24)
        return estimator

    def test_model_not_loaded(self):
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
        with pytest.raises(RuntimeError, match="Model not loaded"):
            estimator.estimate_depth_batch(np.zeros((2, 8, 8, 3), np.uint8))

    def test_whole_sequence_in_one_call(self):
        """VDA aligns its own windows, so long inputs are not split up here."""
        raw = np.random.rand(70, 12, 16).astype(np.float32) * 5
        estimator = self._estimator(raw)

        with patch.object(estimator, "_suppress_model_output", MagicMock()):
            result = estimator.estimate_depth_batch(np.zeros((70, 12, 16, 3), np.uint8))

        estimator.model.infer_video_depth.assert_called_once()
        assert result.shape == (70, 12, 16)
        assert 0.0 <= result.min() and result.max() <= 1.0

    def test_raw_output(self):
        raw = np.random.rand(3, 12, 16).astype(np.float32) * 5
        estimator = self._estimator(raw)

        with patch.object(estimator, "_suppress_model_output", MagicMock()):
            result = estimator.estimate_depth_batch(
                np.zeros((3, 12, 16, 3), np.uint8), normalize=False
            )

        np.testing.assert_allclose(result, raw)

    def test_bgr_to_rgb(self):
        estimator = self._estimator(np.zeros((1, 2, 2), np.float32))
        frames = np.zeros((1, 2, 2, 3), np.uint8)
        frames[..., 0] = 255  # blue in BGR

        with patch.object(estimator, "_suppress_model_output", MagicMock()):
            estimator.estimate_depth_batch(frames)

        sent = estimator.model.infer_video_depth.call_args[0][0]
        assert sent[0, 0, 0, 2] == 255 and sent[0, 0, 0, 0] == 0

    @patch("torch.cuda.empty_cache")
    def test_oom_retries_at_lower_resolution(self, mock_cache):
        estimator = self._estimator()
        estimator.model.infer_video_depth.side_effect = [
            RuntimeError("CUDA out of memory"),
            (np.random.rand(2, 4, 4).astype(np.float32), 24),
        ]

        with patch.object(estimator, "_suppress_model_output", MagicMock()):
            result = estimator.estimate_depth_batch(
                np.zeros((2, 4, 4, 3), np.uint8), input_size=1024
            )

        assert result.shape == (2, 4, 4)
        assert estimator.model.infer_video_depth.call_args[1]["input_size"] == 512

    def test_other_errors_raised(self):
        estimator = self._estimator()
        estimator.model.infer_video_depth.side_effect = RuntimeError("bad input")

        with patch.object(estimator, "_suppress_model_output", MagicMock()):
            with pytest.raises(RuntimeError, match="Video depth estimation failed: bad input"):
                estimator.estimate_depth_batch(np.zeros((2, 4, 4, 3), np.uint8))


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
