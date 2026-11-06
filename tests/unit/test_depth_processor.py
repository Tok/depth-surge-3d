"""Tests for DepthMapProcessor module."""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch

from src.depth_surge_3d.processing.frames.depth_processor import DepthMapProcessor


class TestDepthMapProcessorInit:
    """Test DepthMapProcessor initialization."""

    def test_init_with_estimator(self):
        """Test initialization with depth estimator."""
        estimator = Mock()
        processor = DepthMapProcessor(estimator, verbose=False)

        assert processor.depth_estimator == estimator
        assert processor.verbose is False

    def test_init_with_verbose(self):
        """Test initialization with verbose enabled."""
        estimator = Mock()
        processor = DepthMapProcessor(estimator, verbose=True)

        assert processor.verbose is True


class TestGenerateDepthMaps:
    """Test generate_depth_maps main entry point."""

    @pytest.fixture
    def mock_estimator(self):
        """Create mock depth estimator."""
        estimator = Mock()
        estimator.estimate_depth_batch = Mock(return_value=np.random.rand(3, 100, 100))
        return estimator

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create mock progress tracker."""
        tracker = Mock()
        tracker.update_progress = Mock()
        return tracker

    @pytest.fixture
    def temp_frames(self, tmp_path):
        """Create temporary frame files."""
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()

        frame_files = []
        for i in range(3):
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            frame_path = frame_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_files.append(frame_path)

        return frame_files

    def test_generate_from_cache_existing_depth_maps(
        self, mock_estimator, mock_progress_tracker, temp_frames, tmp_path
    ):
        """Test loading existing depth maps when keep_intermediates=True."""
        processor = DepthMapProcessor(mock_estimator, verbose=False)

        # Create existing depth maps
        depth_dir = tmp_path / "depth_maps"
        depth_dir.mkdir()
        for i in range(3):
            depth_map = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            cv2.imwrite(str(depth_dir / f"frame_{i:04d}.png"), depth_map)

        settings = {"keep_intermediates": True}
        directories = {"depth_maps": depth_dir}

        result = processor.generate_depth_maps(
            temp_frames, settings, directories, mock_progress_tracker
        )

        assert result is not None
        assert len(result) == 3
        # Should not call estimator if loading from cache
        mock_estimator.estimate_depth_batch.assert_not_called()

    def test_generate_from_global_cache(
        self, mock_estimator, mock_progress_tracker, temp_frames, tmp_path
    ):
        """Test loading from global depth cache."""
        processor = DepthMapProcessor(mock_estimator, verbose=False)

        settings = {"video_path": "/test/video.mp4"}
        directories = {}

        cached_depths = np.random.rand(3, 100, 100)

        with patch(
            "src.depth_surge_3d.processing.frames.depth_processor.get_cached_depth_maps",
            return_value=cached_depths,
        ):
            with patch(
                "src.depth_surge_3d.processing.frames.depth_processor.get_cache_size",
                return_value=(5, 10240),
            ):
                result = processor.generate_depth_maps(
                    temp_frames, settings, directories, mock_progress_tracker
                )

        assert result is not None
        assert len(result) == 3
        mock_estimator.estimate_depth_batch.assert_not_called()

    def test_generate_new_depth_maps(
        self, mock_estimator, mock_progress_tracker, temp_frames, tmp_path
    ):
        """Test generating new depth maps when no cache exists."""
        processor = DepthMapProcessor(mock_estimator, verbose=False)

        settings = {"video_path": "/test/video.mp4", "depth_resolution": "1080"}
        directories = {}

        with patch.object(processor, "_generate_depth_maps_chunked") as mock_chunked:
            mock_chunked.return_value = np.random.rand(3, 100, 100)

            with patch(
                "src.depth_surge_3d.processing.frames.depth_processor.get_cached_depth_maps",
                return_value=None,
            ):
                with patch.object(processor, "_save_to_depth_cache"):
                    result = processor.generate_depth_maps(
                        temp_frames, settings, directories, mock_progress_tracker
                    )

        assert result is not None
        assert len(result) == 3
        mock_chunked.assert_called_once()


class TestDetermineChunkParams:
    """Test chunk parameter determination."""

    @pytest.fixture
    def mock_estimator(self):
        """Create mock depth estimator with model info."""
        estimator = Mock()
        estimator.model_type = "v3"
        estimator.get_model_size = Mock(return_value="large")
        return estimator

    def test_determine_chunk_params_auto_resolution(self, mock_estimator):
        """Test auto resolution detection."""
        processor = DepthMapProcessor(mock_estimator, verbose=False)

        with patch(
            "src.depth_surge_3d.processing.frames.depth_processor.get_vram_info",
            return_value={"total": 8.0, "available": 6.0},
        ):
            with patch(
                "src.depth_surge_3d.processing.frames.depth_processor.calculate_optimal_chunk_size",
                return_value=4,
            ):
                chunk_size, input_size = processor._determine_chunk_params(1920, 1080, "auto")

        assert chunk_size == 4
        assert input_size == 1080  # Auto selected based on resolution

    def test_determine_chunk_params_manual_resolution(self, mock_estimator):
        """Test manual resolution setting."""
        processor = DepthMapProcessor(mock_estimator, verbose=False)

        with patch(
            "src.depth_surge_3d.processing.frames.depth_processor.get_vram_info",
            return_value={"total": 8.0, "available": 6.0},
        ):
            with patch(
                "src.depth_surge_3d.processing.frames.depth_processor.calculate_optimal_chunk_size",
                return_value=2,
            ):
                chunk_size, input_size = processor._determine_chunk_params(1920, 1080, "720")

        assert chunk_size == 2
        assert input_size == 720

    def test_determine_chunk_params_invalid_manual(self, mock_estimator):
        """Test invalid manual resolution falls back to auto."""
        processor = DepthMapProcessor(mock_estimator, verbose=False)

        with patch(
            "src.depth_surge_3d.processing.frames.depth_processor.get_vram_info",
            return_value={"total": 8.0, "available": 6.0},
        ):
            with patch(
                "src.depth_surge_3d.processing.frames.depth_processor.calculate_optimal_chunk_size",
                return_value=4,
            ):
                chunk_size, input_size = processor._determine_chunk_params(1920, 1080, "invalid")

        assert input_size == 1080  # Fell back to auto

    def test_determine_chunk_params_cpu_mode(self, mock_estimator):
        """Test CPU mode without VRAM."""
        processor = DepthMapProcessor(mock_estimator, verbose=False)

        with patch(
            "src.depth_surge_3d.processing.frames.depth_processor.get_vram_info",
            return_value={"total": 0, "available": 0},
        ):
            chunk_size, input_size = processor._determine_chunk_params(1920, 1080, "auto")

        assert chunk_size in [4, 6, 8, 12, 16, 24, 32]  # Fixed size for CPU (actual constants)
        assert input_size == 1080


class TestAutoDetermineInputSize:
    """Test automatic input size determination."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return DepthMapProcessor(Mock(), verbose=False)

    def test_4k_resolution(self, processor):
        """Test 4K resolution input sizing."""
        input_size = processor._auto_determine_input_size(3840, 2160, 8.3)
        assert input_size == 2160  # Should cap at source resolution

    def test_1080p_resolution(self, processor):
        """Test 1080p resolution input sizing."""
        input_size = processor._auto_determine_input_size(1920, 1080, 2.1)
        assert input_size == 1080

    def test_720p_resolution(self, processor):
        """Test 720p resolution input sizing."""
        input_size = processor._auto_determine_input_size(1280, 720, 0.9)
        assert input_size == 640  # Falls to SD since 0.9 is not > MEGAPIXELS_720P (1.0)

    def test_sd_resolution(self, processor):
        """Test SD resolution input sizing."""
        input_size = processor._auto_determine_input_size(640, 480, 0.3)
        assert input_size == 640


class TestGetChunkSizeForResolution:
    """Test chunk size selection based on resolution."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return DepthMapProcessor(Mock(), verbose=False)

    def test_4k_chunk_size(self, processor):
        """Test 4K chunk size."""
        chunk_size = processor._get_chunk_size_for_resolution(2160)
        assert chunk_size == 4  # CHUNK_SIZE_4K

    def test_1440p_chunk_size(self, processor):
        """Test 1440p chunk size."""
        chunk_size = processor._get_chunk_size_for_resolution(1440)
        assert chunk_size == 6  # CHUNK_SIZE_1440P

    def test_1080p_chunk_size(self, processor):
        """Test 1080p chunk size."""
        chunk_size = processor._get_chunk_size_for_resolution(1080)
        assert chunk_size == 12  # CHUNK_SIZE_1080P_MANUAL

    def test_720p_chunk_size(self, processor):
        """Test 720p chunk size."""
        chunk_size = processor._get_chunk_size_for_resolution(720)
        assert chunk_size == 16  # CHUNK_SIZE_720P

    def test_small_chunk_size(self, processor):
        """Test small resolution chunk size."""
        chunk_size = processor._get_chunk_size_for_resolution(480)
        assert chunk_size == 32  # CHUNK_SIZE_SMALL


class TestClearGPUMemory:
    """Test GPU memory clearing."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return DepthMapProcessor(Mock(), verbose=False)

    def test_clear_gpu_memory_cuda_available(self, processor):
        """Test clearing GPU memory when CUDA is available."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.empty_cache") as mock_empty:
                with patch("torch.cuda.synchronize") as mock_sync:
                    with patch("torch.cuda.mem_get_info", return_value=(4 * 1024**3, 8 * 1024**3)):
                        processor._clear_gpu_memory()

        mock_empty.assert_called_once()
        mock_sync.assert_called_once()

    def test_clear_gpu_memory_cpu_mode(self, processor):
        """Test clearing GPU memory in CPU mode (no-op)."""
        with patch("torch.cuda.is_available", return_value=False):
            processor._clear_gpu_memory()
        # Should not raise any errors


class TestLoadChunkFrames:
    """Test loading chunk of frames."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return DepthMapProcessor(Mock(), verbose=False)

    @pytest.fixture
    def temp_frames(self, tmp_path):
        """Create temporary frame files."""
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()

        frame_files = []
        for i in range(3):
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            frame_path = frame_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_files.append(frame_path)

        return frame_files

    def test_load_chunk_frames_success(self, processor, temp_frames):
        """Test successful frame loading."""
        settings = {"super_sample": "none", "per_eye_width": 100, "per_eye_height": 100}

        result = processor._load_chunk_frames(temp_frames, settings)

        assert result is not None
        assert len(result) == 3
        assert all(isinstance(frame, np.ndarray) for frame in result)

    def test_load_chunk_frames_never_upscales(self, processor, temp_frames):
        """Regression #16/#17: frames reach the depth model at source resolution.

        Super sampling used to upscale frames here, producing depth maps that didn't
        match the frames used for stereo generation.
        """
        settings = {"super_sample": "auto", "per_eye_width": 200, "per_eye_height": 200}

        result = processor._load_chunk_frames(temp_frames, settings)

        assert result is not None
        assert all(frame.shape == (100, 100, 3) for frame in result)

    def test_load_chunk_frames_missing_file(self, processor, temp_frames):
        """Test loading with missing file."""
        settings = {"super_sample": "none", "per_eye_width": 100, "per_eye_height": 100}

        # Add non-existent file
        bad_path = Path("/nonexistent/frame.png")
        chunk_files = temp_frames + [bad_path]

        result = processor._load_chunk_frames(chunk_files, settings)

        # Should still load valid frames
        assert result is not None
        assert len(result) == 3

    def test_load_chunk_frames_all_missing(self, processor):
        """Test loading with all files missing."""
        settings = {"super_sample": "none", "per_eye_width": 100, "per_eye_height": 100}
        chunk_files = [Path("/nonexistent/frame1.png"), Path("/nonexistent/frame2.png")]

        result = processor._load_chunk_frames(chunk_files, settings)

        assert result is None


def _sized_estimator(temporal: bool, h: int = 20, w: int = 30) -> Mock:
    """Mock estimator returning one depth map per input frame."""
    estimator = Mock()
    estimator.temporally_consistent = temporal
    estimator.estimate_depth_batch = Mock(
        side_effect=lambda frames, **kw: np.random.rand(len(frames), h, w).astype(np.float32)
    )
    return estimator


class TestEstimateChunk:
    """Test raw chunk depth estimation."""

    def test_requests_raw_output(self):
        estimator = _sized_estimator(temporal=True)
        processor = DepthMapProcessor(estimator, verbose=False)
        frames = [np.zeros((20, 30, 3), np.uint8)] * 3

        result = processor._estimate_chunk(frames, {"target_fps": 24}, input_size=518)

        assert result.shape == (3, 20, 30)
        kwargs = estimator.estimate_depth_batch.call_args[1]
        assert kwargs["normalize"] is False
        assert kwargs["target_fps"] == 24
        assert kwargs["input_size"] == 518

    @pytest.mark.parametrize("fps", [None, "None", "original"])
    def test_fallback_fps(self, fps):
        estimator = _sized_estimator(temporal=True)
        processor = DepthMapProcessor(estimator, verbose=False)

        processor._estimate_chunk([np.zeros((20, 30, 3), np.uint8)], {"target_fps": fps}, 518)

        assert estimator.estimate_depth_batch.call_args[1]["target_fps"] == 30


class TestChunkSizing:
    """Test chunk sizing helpers."""

    def test_temporal_chunk_size_bounded(self):
        assert DepthMapProcessor._temporal_chunk_size(320, 240) == 512  # tiny frames: max
        assert DepthMapProcessor._temporal_chunk_size(7680, 4320) == 32  # 8K: min
        assert 32 < DepthMapProcessor._temporal_chunk_size(1280, 720) < 512

    def test_chunk_overlap(self):
        assert DepthMapProcessor._chunk_overlap({}, 200) == 10  # default
        assert DepthMapProcessor._chunk_overlap({"temporal_window_overlap": 16}, 200) == 16
        assert DepthMapProcessor._chunk_overlap({"temporal_window_overlap": 1}, 200) == 4
        assert DepthMapProcessor._chunk_overlap({"temporal_window_overlap": 40}, 32) == 16


class TestGenerateDepthMapsChunked:
    """Test chunked depth map generation with alignment."""

    @pytest.fixture
    def temp_frames(self, tmp_path):
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        frame_files = []
        for i in range(25):
            frame_path = frame_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(frame_path), np.random.randint(0, 255, (20, 30, 3), dtype=np.uint8))
            frame_files.append(frame_path)
        return frame_files

    SETTINGS = {"depth_resolution": "518", "target_fps": 30, "keep_intermediates": False}

    def _run(self, processor, frames, settings=None, directories=None, tracker=None, chunk=4):
        with (
            patch.object(processor, "_determine_chunk_params", return_value=(chunk, 518)),
            patch.object(processor, "_temporal_chunk_size", return_value=10),
            patch.object(processor, "_clear_gpu_memory"),
        ):
            return processor._generate_depth_maps_chunked(
                frames, settings or self.SETTINGS, directories or {}, tracker
            )

    def test_temporal_model_overlapping_chunks(self, temp_frames):
        """Temporal models get long chunks that re-process the overlap frames."""
        estimator = _sized_estimator(temporal=True)
        settings = {**self.SETTINGS, "temporal_window_overlap": 4}

        result = self._run(DepthMapProcessor(estimator), temp_frames, settings)

        assert result.shape == (25, 20, 30)
        assert 0.0 <= result.min() and result.max() <= 1.0
        sizes = [len(c[0][0]) for c in estimator.estimate_depth_batch.call_args_list]
        # chunks of 10 new frames; later chunks also re-process 4 overlap frames
        assert sizes == [10, 14, 9]

    def test_per_frame_model_no_overlap(self, temp_frames):
        """Per-frame models use the VRAM-based chunk size and no overlap."""
        estimator = _sized_estimator(temporal=False)

        result = self._run(DepthMapProcessor(estimator), temp_frames, chunk=4)

        assert result.shape == (25, 20, 30)
        sizes = [len(c[0][0]) for c in estimator.estimate_depth_batch.call_args_list]
        assert sizes == [4, 4, 4, 4, 4, 4, 1]

    def test_saves_normalized_maps_at_end(self, temp_frames, tmp_path):
        estimator = _sized_estimator(temporal=True)
        processor = DepthMapProcessor(estimator)
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()

        with patch.object(processor, "_save_depth_maps") as mock_save:
            result = self._run(
                processor,
                temp_frames,
                {**self.SETTINGS, "keep_intermediates": True},
                {"depth_maps": depth_dir},
            )

        mock_save.assert_called_once()
        assert mock_save.call_args[0][0] is result

    def test_live_preview_per_chunk(self, temp_frames):
        tracker = Mock()
        result = self._run(DepthMapProcessor(_sized_estimator(True)), temp_frames, tracker=tracker)

        assert result is not None
        assert tracker.send_preview_frame_from_array.call_count == 3
        preview = tracker.send_preview_frame_from_array.call_args[0][0]
        assert preview.shape == (20, 30, 3) and preview.dtype == np.uint8

    def test_generate_chunked_sample_frame_missing(self, tmp_path):
        processor = DepthMapProcessor(_sized_estimator(True), verbose=False)

        result = processor._generate_depth_maps_chunked(
            [tmp_path / "nonexistent.png"], self.SETTINGS, {}, None
        )

        assert result is None

    def test_generate_chunked_error_handling(self, temp_frames):
        processor = DepthMapProcessor(_sized_estimator(True), verbose=False)

        with patch.object(processor, "_load_chunk_frames", return_value=None):
            assert self._run(processor, temp_frames) is None

    def test_estimator_failure(self, temp_frames):
        estimator = _sized_estimator(True)
        estimator.estimate_depth_batch.side_effect = RuntimeError("boom")

        assert self._run(DepthMapProcessor(estimator), temp_frames) is None


class TestSaveDepthMaps:
    """Test depth map saving."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return DepthMapProcessor(Mock(), verbose=False)

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create mock progress tracker."""
        tracker = Mock()
        tracker.send_preview_frame = Mock()
        return tracker

    @pytest.fixture
    def temp_frames(self, tmp_path):
        """Create temporary frame files."""
        frame_files = []
        for i in range(3):
            frame_files.append(tmp_path / f"frame_{i:04d}.png")
        return frame_files

    def test_save_depth_maps(self, processor, mock_progress_tracker, temp_frames, tmp_path):
        """Test saving depth maps to disk."""
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()

        depth_maps = np.random.rand(3, 100, 100)

        processor._save_depth_maps(depth_maps, temp_frames, depth_dir, mock_progress_tracker)

        # Check files were created
        saved_files = list(depth_dir.glob("*.png"))
        assert len(saved_files) == 3

    def test_save_depth_maps_with_preview(
        self, processor, mock_progress_tracker, temp_frames, tmp_path
    ):
        """Test saving with preview frames."""
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()

        depth_maps = np.random.rand(3, 100, 100)

        processor._save_depth_maps(depth_maps, temp_frames, depth_dir, mock_progress_tracker)

        # Should send preview for first and last frame
        assert mock_progress_tracker.send_preview_frame.called


class TestTryLoadExistingDepthMaps:
    """Test loading existing depth maps."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return DepthMapProcessor(Mock(), verbose=False)

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create mock progress tracker."""
        tracker = Mock()
        tracker.update_progress = Mock()
        return tracker

    @pytest.fixture
    def temp_frames(self, tmp_path):
        """Create temporary frame files."""
        return [tmp_path / f"frame_{i:04d}.png" for i in range(3)]

    def test_load_existing_success(self, processor, mock_progress_tracker, temp_frames, tmp_path):
        """Test successful loading of existing depth maps."""
        depth_dir = tmp_path / "depth_maps"
        depth_dir.mkdir()

        # Create existing depth maps
        for i in range(3):
            depth_map = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            cv2.imwrite(str(depth_dir / f"frame_{i:04d}.png"), depth_map)

        directories = {"depth_maps": depth_dir}

        result = processor._try_load_existing_depth_maps(
            temp_frames, directories, mock_progress_tracker
        )

        assert result is not None
        assert len(result) == 3

    def test_load_existing_missing_directory(self, processor, mock_progress_tracker, temp_frames):
        """Test loading when directory doesn't exist."""
        directories = {}

        result = processor._try_load_existing_depth_maps(
            temp_frames, directories, mock_progress_tracker
        )

        assert result is None

    def test_load_existing_insufficient_maps(
        self, processor, mock_progress_tracker, temp_frames, tmp_path
    ):
        """Test loading when not enough depth maps exist."""
        depth_dir = tmp_path / "depth_maps"
        depth_dir.mkdir()

        # Only create 1 depth map when 3 are needed
        depth_map = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        cv2.imwrite(str(depth_dir / "frame_0000.png"), depth_map)

        directories = {"depth_maps": depth_dir}

        result = processor._try_load_existing_depth_maps(
            temp_frames, directories, mock_progress_tracker
        )

        assert result is None


class TestTryLoadCachedDepthMaps:
    """Test loading from global cache."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return DepthMapProcessor(Mock(), verbose=False)

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create mock progress tracker."""
        tracker = Mock()
        tracker.update_progress = Mock()
        return tracker

    def test_load_cached_success(self, processor, mock_progress_tracker):
        """Test successful cache loading."""
        cached_depths = np.random.rand(5, 100, 100)

        with patch(
            "src.depth_surge_3d.processing.frames.depth_processor.get_cached_depth_maps",
            return_value=cached_depths,
        ):
            with patch(
                "src.depth_surge_3d.processing.frames.depth_processor.get_cache_size",
                return_value=(10, 20480),
            ):
                result = processor._try_load_cached_depth_maps(
                    "/test/video.mp4", {}, 5, mock_progress_tracker
                )

        assert result is not None
        assert len(result) == 5

    def test_load_cached_miss(self, processor, mock_progress_tracker):
        """Test cache miss."""
        with patch(
            "src.depth_surge_3d.processing.frames.depth_processor.get_cached_depth_maps",
            return_value=None,
        ):
            result = processor._try_load_cached_depth_maps(
                "/test/video.mp4", {}, 5, mock_progress_tracker
            )

        assert result is None


class TestSaveToDepthCache:
    """Test saving to global cache."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return DepthMapProcessor(Mock(), verbose=False)

    def test_save_to_cache_success(self, processor):
        """Test successful cache saving."""
        depth_maps = np.random.rand(5, 100, 100)

        with patch(
            "src.depth_surge_3d.processing.frames.depth_processor.save_depth_maps_to_cache",
            return_value=True,
        ):
            with patch(
                "src.depth_surge_3d.processing.frames.depth_processor.get_cache_size",
                return_value=(11, 30720),
            ):
                processor._save_to_depth_cache("/test/video.mp4", {}, depth_maps)
        # Should not raise any errors

    def test_save_to_cache_failure(self, processor):
        """Test cache save failure."""
        depth_maps = np.random.rand(5, 100, 100)

        with patch(
            "src.depth_surge_3d.processing.frames.depth_processor.save_depth_maps_to_cache",
            return_value=False,
        ):
            processor._save_to_depth_cache("/test/video.mp4", {}, depth_maps)
        # Should not raise any errors


class TestGenerateDepthMapsBatch:
    """Test batch depth generation (legacy method)."""

    @pytest.fixture
    def mock_estimator(self):
        """Create mock depth estimator."""
        estimator = Mock()
        estimator.estimate_depth_batch = Mock(return_value=np.random.rand(5, 100, 100))
        return estimator

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create mock progress tracker."""
        return Mock()

    def test_batch_generation_success(self, mock_estimator, mock_progress_tracker):
        """Test successful batch generation."""
        processor = DepthMapProcessor(mock_estimator, verbose=False)

        frames = np.random.randint(0, 255, (5, 100, 100, 3), dtype=np.uint8)
        settings = {"target_fps": 30, "depth_resolution": "1080"}

        result = processor._generate_depth_maps_batch(frames, settings, mock_progress_tracker)

        assert result is not None
        assert len(result) == 5
        mock_estimator.estimate_depth_batch.assert_called_once()

    def test_batch_generation_auto_resolution(self, mock_estimator, mock_progress_tracker):
        """Test batch generation with auto resolution."""
        processor = DepthMapProcessor(mock_estimator, verbose=False)

        frames = np.random.randint(0, 255, (5, 100, 100, 3), dtype=np.uint8)
        settings = {"target_fps": 30, "depth_resolution": "auto"}

        result = processor._generate_depth_maps_batch(frames, settings, mock_progress_tracker)

        assert result is not None
        call_args = mock_estimator.estimate_depth_batch.call_args
        assert call_args[1]["input_size"] == 1080

    def test_batch_generation_error(self, mock_estimator, mock_progress_tracker):
        """Test error handling in batch generation."""
        processor = DepthMapProcessor(mock_estimator, verbose=False)

        mock_estimator.estimate_depth_batch.side_effect = RuntimeError("GPU OOM")

        frames = np.random.randint(0, 255, (5, 100, 100, 3), dtype=np.uint8)
        settings = {"target_fps": 30, "depth_resolution": "1080"}

        result = processor._generate_depth_maps_batch(frames, settings, mock_progress_tracker)

        assert result is None
