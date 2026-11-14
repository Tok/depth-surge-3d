"""Tests for DepthMapProcessor module."""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch

from src.depth_surge_3d.processing.frames.depth_processor import DepthMapProcessor

MODULE = "src.depth_surge_3d.processing.frames.depth_processor"


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


class TestOutOfMemoryRetry:
    """Free VRAM can shrink while a job runs; a chunk that no longer fits is split."""

    def _estimator(self, max_frames, temporal=False):
        estimator = Mock()
        estimator.temporally_consistent = temporal
        sizes = []

        def estimate(frames, **kwargs):
            sizes.append(len(frames))
            if len(frames) > max_frames:
                raise RuntimeError("DA3 depth estimation failed: CUDA out of memory. Tried to ...")
            return frames[..., 0].astype(np.float32)

        estimator.estimate_depth_batch = Mock(side_effect=estimate)
        return estimator, sizes

    def _frames(self, count=8):
        return [np.full((4, 6, 3), i, np.uint8) for i in range(count)]

    def test_chunk_is_split_until_it_fits_and_order_is_kept(self, capsys):
        estimator, sizes = self._estimator(max_frames=2)
        processor = DepthMapProcessor(estimator)

        with patch.object(processor, "_clear_gpu_memory"):
            result = processor._estimate_chunk(self._frames(8), {"target_fps": 30}, 518)

        assert result.shape == (8, 4, 6)
        assert result[:, 0, 0].tolist() == list(range(8))
        assert sizes == [8, 4, 2, 2, 4, 2, 2]
        assert "retrying in chunks of 4" in capsys.readouterr().out

    def test_single_frame_that_does_not_fit_fails(self):
        estimator, _ = self._estimator(max_frames=0)
        processor = DepthMapProcessor(estimator)
        with (
            patch.object(processor, "_clear_gpu_memory"),
            pytest.raises(RuntimeError, match="out of memory"),
        ):
            processor._estimate_chunk(self._frames(2), {"target_fps": 30}, 518)

    def test_other_errors_and_temporal_models_are_not_retried(self):
        estimator = Mock()
        estimator.temporally_consistent = False
        estimator.estimate_depth_batch = Mock(side_effect=RuntimeError("bad input"))
        with pytest.raises(RuntimeError, match="bad input"):
            DepthMapProcessor(estimator)._estimate_chunk(self._frames(4), {"target_fps": 30}, 518)
        assert estimator.estimate_depth_batch.call_count == 1

        temporal, sizes = self._estimator(max_frames=2, temporal=True)
        with pytest.raises(RuntimeError, match="out of memory"):
            DepthMapProcessor(temporal)._estimate_chunk(self._frames(4), {"target_fps": 30}, 518)
        assert sizes == [4]


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


class TestStreamDepthMaps:
    """Streaming depth generation: chunked, written as it goes, resumable."""

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

    SETTINGS = {"depth_resolution": "518", "target_fps": 30, "processing_mode": "sequential"}

    def _stream(self, processor, frames, tmp_path, settings=None, tracker=None, chunk=4):
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir(exist_ok=True)
        with (
            patch.object(processor, "_determine_chunk_params", return_value=(chunk, 518)),
            patch.object(processor, "_temporal_chunk_size", return_value=10),
            patch.object(processor, "_clear_gpu_memory"),
        ):
            yield from processor.stream_depth_maps(
                frames, depth_dir, tmp_path / "state.npz", settings or self.SETTINGS, tracker
            )

    def test_temporal_model_holds_back_overlap(self, temp_frames, tmp_path):
        """Overlap frames are re-estimated by the next chunk and only then written."""
        estimator = _sized_estimator(temporal=True)
        settings = {**self.SETTINGS, "temporal_window_overlap": 4}

        emitted = list(self._stream(DepthMapProcessor(estimator), temp_frames, tmp_path, settings))

        # chunks of 10 new frames; all but the last hold back the 4 overlap frames
        assert emitted == [6, 16, 25]
        sizes = [len(c[0][0]) for c in estimator.estimate_depth_batch.call_args_list]
        assert sizes == [10, 14, 9]
        depth_files = sorted((tmp_path / "depth").glob("*.png"))
        assert [f.stem for f in depth_files] == [f.stem for f in temp_frames]
        first = cv2.imread(str(depth_files[0]), cv2.IMREAD_UNCHANGED)
        assert first.dtype == np.uint16 and first.shape == (20, 30)

    def test_files_exist_when_progress_is_yielded(self, temp_frames, tmp_path):
        """Depth maps are on disk as soon as their count is reported (not at the end)."""
        processor = DepthMapProcessor(_sized_estimator(temporal=False))

        for emitted in self._stream(processor, temp_frames, tmp_path, chunk=4):
            assert len(list((tmp_path / "depth").glob("*.png"))) == emitted
            assert DepthMapProcessor.finished_depth_maps(tmp_path / "state.npz") == emitted

    def test_per_frame_model_no_overlap(self, temp_frames, tmp_path):
        estimator = _sized_estimator(temporal=False)

        emitted = list(self._stream(DepthMapProcessor(estimator), temp_frames, tmp_path, chunk=4))

        assert emitted == [4, 8, 12, 16, 20, 24, 25]
        sizes = [len(c[0][0]) for c in estimator.estimate_depth_batch.call_args_list]
        assert sizes == [4, 4, 4, 4, 4, 4, 1]

    @pytest.mark.parametrize("temporal", [True, False])
    def test_resumed_stream_matches_uninterrupted(self, temp_frames, tmp_path, temporal):
        """Stopping after any chunk and continuing gives exactly the same depth maps."""

        def estimator():
            mock = Mock()
            mock.temporally_consistent = temporal
            # Deterministic per frame content, with a different scale/shift per call
            calls = {"n": 0}

            def estimate(frames, **kw):
                calls["n"] += 1
                base = frames[..., 0].astype(np.float32)
                return base * (1.0 + 0.1 * calls["n"]) + calls["n"]

            mock.estimate_depth_batch = Mock(side_effect=estimate)
            return mock

        settings = {**self.SETTINGS, "temporal_window_overlap": 4}
        straight = tmp_path / "straight"
        resumed = tmp_path / "resumed"
        straight.mkdir()
        resumed.mkdir()

        list(self._stream(DepthMapProcessor(estimator()), temp_frames, straight, settings))

        stream = self._stream(DepthMapProcessor(estimator()), temp_frames, resumed, settings)
        next(stream)
        stream.close()  # interrupted after the first chunk
        first_run = DepthMapProcessor.finished_depth_maps(resumed / "state.npz")
        assert 0 < first_run < 25
        rest = list(self._stream(DepthMapProcessor(estimator()), temp_frames, resumed, settings))
        assert rest[-1] == 25 and rest[0] > first_run

        for frame in temp_frames:
            a = cv2.imread(str(straight / "depth" / f"{frame.stem}.png"), cv2.IMREAD_UNCHANGED)
            b = cv2.imread(str(resumed / "depth" / f"{frame.stem}.png"), cv2.IMREAD_UNCHANGED)
            # same content; only the per-call scale differs, which alignment removes
            assert np.abs(a.astype(int) - b.astype(int)).max() <= 2

    def test_resume_does_not_need_frames_that_are_done(self, temp_frames, tmp_path):
        """Regression: without intermediates, frames of finished batches are deleted."""
        processor = DepthMapProcessor(_sized_estimator(temporal=False))
        stream = self._stream(processor, temp_frames, tmp_path, chunk=5)
        assert next(stream) == 5
        stream.close()
        for frame in temp_frames[:5]:
            frame.unlink()

        assert list(self._stream(processor, temp_frames, tmp_path, chunk=5))[-1] == 25

    def test_finished_stream_yields_nothing(self, temp_frames, tmp_path):
        estimator = _sized_estimator(temporal=True)
        DepthMapProcessor.mark_stream_complete(tmp_path / "state.npz", len(temp_frames))

        assert list(self._stream(DepthMapProcessor(estimator), temp_frames, tmp_path)) == []
        estimator.estimate_depth_batch.assert_not_called()

    def test_batched_mode_caps_chunk_at_batch_size(self, temp_frames, tmp_path):
        """Batched mode doesn't estimate far beyond the batch being worked on."""
        estimator = _sized_estimator(temporal=True)
        settings = {**self.SETTINGS, "processing_mode": "batched", "batch_size": 5}
        processor = DepthMapProcessor(estimator)

        with patch(f"{MODULE}.DEPTH_CHUNK_MIN_FRAMES", 6):
            list(self._stream(processor, temp_frames, tmp_path, settings))

        assert len(estimator.estimate_depth_batch.call_args_list[0][0][0]) == 6

    def test_live_preview_and_progress_per_chunk(self, temp_frames, tmp_path):
        tracker = Mock()
        processor = DepthMapProcessor(_sized_estimator(temporal=False))

        list(self._stream(processor, temp_frames, tmp_path, tracker=tracker, chunk=5))

        assert tracker.send_preview_frame_from_array.call_count == 5
        last = tracker.update_progress.call_args[1]
        assert last["step_name"] == "Depth Map Generation"
        assert (last["step_progress"], last["step_total"]) == (25, 25)

    def test_sample_frame_missing(self, tmp_path):
        processor = DepthMapProcessor(Mock())
        with pytest.raises(RuntimeError, match="Could not read frame"):
            list(self._stream(processor, [tmp_path / "missing.png"], tmp_path))

    def test_unreadable_chunk_frame(self, temp_frames, tmp_path):
        temp_frames[2].unlink()
        processor = DepthMapProcessor(_sized_estimator(temporal=False))
        with pytest.raises(RuntimeError, match="Could not load all frames"):
            list(self._stream(processor, temp_frames, tmp_path))

    def test_estimator_failure_propagates(self, temp_frames, tmp_path):
        estimator = Mock()
        estimator.temporally_consistent = True
        estimator.estimate_depth_batch = Mock(side_effect=RuntimeError("Depth error"))
        with pytest.raises(RuntimeError, match="Depth error"):
            list(self._stream(DepthMapProcessor(estimator), temp_frames, tmp_path))

    def test_unwritable_depth_map(self, temp_frames, tmp_path):
        processor = DepthMapProcessor(_sized_estimator(temporal=False))
        with patch(f"{MODULE}.save_depth_map", return_value=False):
            with pytest.raises(RuntimeError, match="Could not write depth map"):
                list(self._stream(processor, temp_frames, tmp_path))

    def test_state_helpers_without_state(self, tmp_path):
        assert DepthMapProcessor.finished_depth_maps(tmp_path / "none.npz") == 0
        emitted, aligner, depth_range = DepthMapProcessor._load_stream_state(
            tmp_path / "none.npz", temporal=True
        )
        assert (emitted, len(aligner), depth_range.low) == (0, 0, None)
