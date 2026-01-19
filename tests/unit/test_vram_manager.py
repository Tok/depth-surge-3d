"""
Unit tests for VRAM management utilities.
"""

from unittest.mock import patch

from src.depth_surge_3d.utils.system.vram_manager import (
    get_available_vram,
    get_total_vram,
    estimate_frame_vram_usage,
    calculate_optimal_chunk_size,
    get_vram_info,
)


class TestGetAvailableVRAM:
    """Test get_available_vram function."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.mem_get_info", return_value=(8 * 1024**3, 16 * 1024**3))
    def test_available_vram_cuda(self, mock_mem_info, mock_cuda_available):
        """Test getting available VRAM with CUDA available."""
        vram = get_available_vram()
        assert vram == 8.0  # 8 GB

    @patch("torch.cuda.is_available", return_value=False)
    def test_available_vram_no_cuda(self, mock_cuda_available):
        """Test getting available VRAM without CUDA."""
        vram = get_available_vram()
        assert vram == 0.0


class TestGetTotalVRAM:
    """Test get_total_vram function."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.mem_get_info", return_value=(8 * 1024**3, 16 * 1024**3))
    def test_total_vram_cuda(self, mock_mem_info, mock_cuda_available):
        """Test getting total VRAM with CUDA available."""
        vram = get_total_vram()
        assert vram == 16.0  # 16 GB

    @patch("torch.cuda.is_available", return_value=False)
    def test_total_vram_no_cuda(self, mock_cuda_available):
        """Test getting total VRAM without CUDA."""
        vram = get_total_vram()
        assert vram == 0.0


class TestEstimateFrameVRAMUsage:
    """Test estimate_frame_vram_usage function."""

    def test_estimate_1080p_v3(self):
        """Test VRAM estimation for 1080p V3."""
        usage = estimate_frame_vram_usage(1920, 1080, 518, "v3")
        assert usage > 0
        assert isinstance(usage, float)

    def test_estimate_4k_v3(self):
        """Test VRAM estimation for 4K V3."""
        usage_4k = estimate_frame_vram_usage(3840, 2160, 518, "v3")
        usage_1080p = estimate_frame_vram_usage(1920, 1080, 518, "v3")
        # 4K should use more VRAM than 1080p
        assert usage_4k > usage_1080p

    def test_estimate_v2_vs_v3(self):
        """Test that V2 uses more VRAM than V3."""
        usage_v2 = estimate_frame_vram_usage(1920, 1080, 518, "v2")
        usage_v3 = estimate_frame_vram_usage(1920, 1080, 518, "v3")
        # V2 should use more VRAM
        assert usage_v2 > usage_v3


class TestCalculateOptimalChunkSize:
    """Test calculate_optimal_chunk_size function."""

    @patch("src.depth_surge_3d.utils.system.vram_manager.get_available_vram", return_value=8.0)
    def test_optimal_chunk_size_8gb(self, mock_vram):
        """Test optimal chunk size with 8GB VRAM."""
        chunk_size = calculate_optimal_chunk_size(1920, 1080, 518, "v3", "base")
        assert chunk_size >= 4
        assert chunk_size <= 32

    @patch("src.depth_surge_3d.utils.system.vram_manager.get_available_vram", return_value=16.0)
    def test_optimal_chunk_size_16gb(self, mock_vram):
        """Test optimal chunk size with 16GB VRAM."""
        chunk_size = calculate_optimal_chunk_size(1920, 1080, 518, "v3", "base")
        assert chunk_size >= 4
        assert chunk_size <= 32

    @patch("src.depth_surge_3d.utils.system.vram_manager.get_available_vram", return_value=2.0)
    def test_optimal_chunk_size_low_vram(self, mock_vram):
        """Test optimal chunk size with low VRAM."""
        chunk_size = calculate_optimal_chunk_size(1920, 1080, 518, "v3", "base")
        # Should return minimum chunk size (can be as low as 2)
        assert chunk_size >= 2

    @patch("src.depth_surge_3d.utils.system.vram_manager.get_available_vram", return_value=0.0)
    def test_optimal_chunk_size_no_cuda(self, mock_vram):
        """Test optimal chunk size without CUDA."""
        chunk_size = calculate_optimal_chunk_size(1920, 1080, 518, "v3", "base")
        # Should return minimum chunk size
        assert chunk_size >= 4

    @patch("src.depth_surge_3d.utils.system.vram_manager.get_available_vram", return_value=8.0)
    def test_chunk_size_large_model(self, mock_vram):
        """Test chunk size with large model uses fewer frames."""
        chunk_small = calculate_optimal_chunk_size(1920, 1080, 518, "v3", "small")
        chunk_large = calculate_optimal_chunk_size(1920, 1080, 518, "v3", "large")
        # Large model should process fewer frames per chunk
        assert chunk_large <= chunk_small

    @patch("src.depth_surge_3d.utils.system.vram_manager.get_available_vram", return_value=8.0)
    def test_chunk_size_4k_vs_1080p(self, mock_vram):
        """Test chunk size for 4K vs 1080p."""
        chunk_1080p = calculate_optimal_chunk_size(1920, 1080, 518, "v3", "base")
        chunk_4k = calculate_optimal_chunk_size(3840, 2160, 518, "v3", "base")
        # 4K should process fewer frames per chunk
        assert chunk_4k <= chunk_1080p


class TestGetVRAMInfo:
    """Test get_vram_info function."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.mem_get_info", return_value=(8 * 1024**3, 16 * 1024**3))
    def test_vram_info_with_cuda(self, mock_mem_info, mock_cuda_available):
        """Test VRAM info with CUDA."""
        info = get_vram_info()
        assert info["available"] == 8.0
        assert info["total"] == 16.0
        assert info["used"] == 8.0
        assert info["usage_percent"] == 50.0

    @patch("torch.cuda.is_available", return_value=False)
    def test_vram_info_no_cuda(self, mock_cuda_available):
        """Test VRAM info without CUDA."""
        info = get_vram_info()
        assert info["available"] == 0.0
        assert info["total"] == 0.0
        assert info["used"] == 0.0
        assert info["usage_percent"] == 0.0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.mem_get_info", side_effect=RuntimeError("CUDA error"))
    def test_vram_info_exception(self, mock_mem_info, mock_cuda_available):
        """Test VRAM info handles exceptions gracefully."""
        info = get_vram_info()
        assert info["available"] == 0.0
        assert info["total"] == 0.0
        assert info["used"] == 0.0
        assert info["usage_percent"] == 0.0


class TestExceptionHandling:
    """Test exception handling in VRAM functions."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.mem_get_info", side_effect=RuntimeError("CUDA error"))
    def test_get_available_vram_exception(self, mock_mem_info, mock_cuda_available):
        """Test get_available_vram handles exceptions."""
        vram = get_available_vram()
        assert vram == 0.0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.mem_get_info", side_effect=RuntimeError("CUDA error"))
    def test_get_total_vram_exception(self, mock_mem_info, mock_cuda_available):
        """Test get_total_vram handles exceptions."""
        vram = get_total_vram()
        assert vram == 0.0
