"""Tests for the fused per-frame pipeline and the thread map used by the separate steps."""

from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.io.operations import save_depth_map
from src.depth_surge_3d.processing.frames import parallel
from src.depth_surge_3d.processing.frames.distortion_processor import DistortionProcessor
from src.depth_surge_3d.processing.frames.frame_pipeline import FusedFrameProcessor, process_frame
from src.depth_surge_3d.processing.frames.stereo_generator import StereoPairGenerator
from src.depth_surge_3d.processing.frames.vr_assembler import VRFrameAssembler

STEREO_MODULE = "src.depth_surge_3d.processing.frames.stereo_generator"

SETTINGS = {
    "baseline": 0.065,
    "focal_length": 1000,
    "hole_fill_quality": "fast",
    "super_sample": "auto",
    "apply_distortion": True,
    "fisheye_projection": "stereographic",
    "fisheye_fov": 180,
    "fisheye_crop_factor": 0.7,
    "crop_factor": 0.9,
    "per_eye_width": 96,
    "per_eye_height": 96,
    "vr_format": "side_by_side",
    "keep_intermediates": True,
    "upscale_model": "none",
}

DIR_KEYS = [
    "frames",
    "depth_maps",
    "left_frames",
    "right_frames",
    "left_distorted",
    "right_distorted",
    "left_cropped",
    "right_cropped",
    "vr_frames",
]


def _batch(tmp_path, count=3):
    """Directories with frames and depth maps; returns (directories, frame files, depth files)."""
    rng = np.random.default_rng(0)
    directories = {key: tmp_path / key for key in DIR_KEYS}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    frames, depths = [], []
    for i in range(count):
        frame = directories["frames"] / f"frame_{i + 1:06d}.png"
        depth = directories["depth_maps"] / f"frame_{i + 1:06d}.png"
        cv2.imwrite(str(frame), rng.integers(0, 255, (72, 128, 3), dtype=np.uint8))
        save_depth_map(depth, rng.random((72, 128)))
        frames.append(frame)
        depths.append(depth)
    return directories, frames, depths


def _run_fused(directories, frames, depths, settings, tracker=None, assemble=True):
    generator = StereoPairGenerator()
    with patch(f"{STEREO_MODULE}.get_stereo_worker_count", return_value=1):
        return FusedFrameProcessor(generator).process_frames(
            frames, depths, directories, settings, tracker, assemble=assemble
        )


def _run_separate(directories, frames, depths, settings):
    """The same frames through the separate step processors."""
    tracker = Mock()
    names = [f.stem for f in frames]
    generator = StereoPairGenerator()
    with patch(f"{STEREO_MODULE}.get_stereo_worker_count", return_value=1):
        assert generator.create_stereo_pairs(frames, depths, directories, settings, tracker)
    distortion = DistortionProcessor()
    if settings["apply_distortion"]:
        left = [directories["left_frames"] / f"{n}.png" for n in names]
        right = [directories["right_frames"] / f"{n}.png" for n in names]
        assert distortion.apply_distortion(left, right, directories, settings, tracker)
    assert distortion.crop_frames(directories, settings, tracker, len(names), frame_names=names)
    assert VRFrameAssembler().assemble_vr_frames(
        directories, settings, tracker, len(names), frame_names=names
    )


class TestFusedMatchesSeparateSteps:
    @pytest.mark.parametrize("distort", [True, False])
    def test_identical_output_in_every_directory(self, tmp_path, distort):
        """Passing images in memory must give exactly what the PNG round trips gave."""
        settings = {**SETTINGS, "apply_distortion": distort}
        fused_dirs, frames, depths = _batch(tmp_path / "fused")
        separate_dirs, frames2, depths2 = _batch(tmp_path / "separate")

        assert _run_fused(fused_dirs, frames, depths, settings) is True
        _run_separate(separate_dirs, frames2, depths2, settings)

        compared = 0
        for key in DIR_KEYS[2:]:
            for separate_file in sorted(separate_dirs[key].glob("*.png")):
                fused = cv2.imread(str(fused_dirs[key] / separate_file.name))
                assert np.array_equal(fused, cv2.imread(str(separate_file))), key
                compared += 1
        assert compared == (21 if distort else 15)
        assert cv2.imread(str(fused_dirs["vr_frames"] / "frame_000001.png")).shape == (96, 192, 3)


class TestWhatGetsWritten:
    def test_without_intermediates_only_vr_frames(self, tmp_path):
        directories, frames, depths = _batch(tmp_path)
        settings = {**SETTINGS, "keep_intermediates": False}

        assert _run_fused(directories, frames, depths, settings) is True

        written = {key: len(list(directories[key].glob("*.png"))) for key in DIR_KEYS[2:]}
        assert written.pop("vr_frames") == 3
        assert set(written.values()) == {0}

    def test_before_upscaling_cropped_eyes_are_the_result(self, tmp_path):
        directories, frames, depths = _batch(tmp_path)
        settings = {**SETTINGS, "keep_intermediates": False, "upscale_model": "x2"}

        assert _run_fused(directories, frames, depths, settings, assemble=False) is True

        assert len(list(directories["left_cropped"].glob("*.png"))) == 3
        assert len(list(directories["right_cropped"].glob("*.png"))) == 3
        assert not list(directories["vr_frames"].glob("*.png"))
        assert not list(directories["left_frames"].glob("*.png"))

    def test_step_names(self):
        names = FusedFrameProcessor.step_names
        assert names(SETTINGS, True) == [
            "Stereo Pair Creation",
            "Fisheye Distortion",
            "Crop Frames",
            "VR Assembly",
        ]
        assert names({"apply_distortion": False}, False) == ["Stereo Pair Creation", "Crop Frames"]


class TestProgressAndFailures:
    def test_progress_for_every_covered_step_and_previews(self, tmp_path):
        directories, frames, depths = _batch(tmp_path)
        tracker = Mock()

        _run_fused(directories, frames, depths, SETTINGS, tracker)

        last = {}
        for call in tracker.update_progress.call_args_list:
            last[call[1]["step_name"]] = (call[1]["step_progress"], call[1]["step_total"])
        assert last == {name: (3, 3) for name in FusedFrameProcessor.step_names(SETTINGS, True)}
        kinds = {call[0][1] for call in tracker.send_preview_frame.call_args_list}
        assert kinds == {"stereo_left", "vr_frame"}

    def test_tracker_without_previews(self, tmp_path):
        directories, frames, depths = _batch(tmp_path, count=1)
        tracker = Mock(spec=["update_progress"])
        assert _run_fused(directories, frames, depths, SETTINGS, tracker) is True

    def test_unreadable_inputs_fail_the_batch(self, tmp_path, capsys):
        directories, frames, depths = _batch(tmp_path)
        depths[1].unlink()
        assert _run_fused(directories, frames, depths, SETTINGS) is False
        assert "Failed to read" in capsys.readouterr().out
        frames[0].unlink()
        assert _run_fused(directories, frames, depths, SETTINGS) is False

    def test_unwritable_output(self, tmp_path):
        directories, frames, depths = _batch(tmp_path, count=1)
        task = (
            str(frames[0]),
            str(depths[0]),
            "frame_000001",
            {"vr_frames": "/no/such/dir"},
            SETTINGS,
            True,
        )
        with pytest.raises(IOError, match="Failed to write"):
            process_frame(task)
        task = (
            str(frames[0]),
            str(depths[0]),
            "frame_000001",
            {"left_frames": "/no/such/dir"},
            SETTINGS,
            True,
        )
        with pytest.raises(IOError, match="Failed to write"):
            process_frame(task)

    def test_count_mismatch_and_empty_batch(self, tmp_path, capsys):
        directories, frames, depths = _batch(tmp_path)
        assert _run_fused(directories, frames, depths[:2], SETTINGS) is False
        assert _run_fused(directories, [], [], SETTINGS) is False

    def test_stop_request_propagates(self, tmp_path):
        directories, frames, depths = _batch(tmp_path)
        tracker = Mock()
        tracker.update_progress.side_effect = InterruptedError("stop")
        with pytest.raises(InterruptedError):
            _run_fused(directories, frames, depths, SETTINGS, tracker)


class TestMapInPool:
    def test_parallel_results_in_order(self):
        from concurrent.futures import ThreadPoolExecutor

        generator = StereoPairGenerator()
        seen = []
        with (
            patch(f"{STEREO_MODULE}.get_stereo_worker_count", return_value=3),
            patch(f"{STEREO_MODULE}.ProcessPoolExecutor", _threads_as_pool(ThreadPoolExecutor)),
        ):
            generator.map_in_pool(str, list(range(20)), seen.append)
        generator.close()
        assert seen == list(range(20))

    def test_crashed_pool_falls_back_to_this_process(self, capsys):
        from concurrent.futures.process import BrokenProcessPool

        generator = StereoPairGenerator()
        executor = Mock()
        executor.map.side_effect = BrokenProcessPool("worker died")
        done = []
        with (
            patch(f"{STEREO_MODULE}.get_stereo_worker_count", return_value=3),
            patch.object(generator, "_get_executor", return_value=executor),
        ):
            generator.map_in_pool(done.append, [1, 2, 3])
        assert done == [1, 2, 3]
        assert "retrying in-process" in capsys.readouterr().out


def _threads_as_pool(thread_pool):
    """ProcessPoolExecutor stand-in that runs the work in threads of this process."""

    def create(max_workers=None, mp_context=None, initializer=None):
        return thread_pool(max_workers=max_workers)

    return create


class TestMapFrames:
    def test_results_keep_input_order(self):
        with patch.object(parallel.os, "cpu_count", return_value=8):
            assert list(parallel.map_frames(lambda x: x * 2, range(50))) == [
                x * 2 for x in range(50)
            ]

    def test_single_thread_on_small_machines(self):
        with patch.object(parallel.os, "cpu_count", return_value=2):
            assert parallel.frame_thread_count(10) == 1
            assert list(parallel.map_frames(lambda x: x + 1, [1, 2])) == [2, 3]
        with patch.object(parallel.os, "cpu_count", return_value=None):
            assert parallel.frame_thread_count(10) == 1

    def test_thread_count_is_capped(self):
        with patch.object(parallel.os, "cpu_count", return_value=64):
            assert parallel.frame_thread_count(1000) == parallel.MAX_FRAME_THREADS
            assert parallel.frame_thread_count(3) == 3

    def test_stopping_early_drops_remaining_work(self):
        started = []

        def work(item):
            started.append(item)
            return item

        with patch.object(parallel.os, "cpu_count", return_value=4):
            results = parallel.map_frames(work, range(200))
            assert next(results) == 0
            results.close()
        assert len(started) < 200
