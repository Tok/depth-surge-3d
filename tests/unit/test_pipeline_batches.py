"""Tests for the orchestrator's batched/sequential loop, resume and cleanup."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.depth_surge_3d.processing.frames.depth_processor import DepthMapProcessor
from src.depth_surge_3d.processing.orchestration.pipeline_orchestrator import (
    ProcessingOrchestrator,
)

MODULE = "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator"
NUM_FRAMES = 10


class _Estimator:
    model_path = "model.pth"
    metric = False


class _Pipeline:
    """Fake processors that touch the files the real ones would write and log their work."""

    def __init__(self, depth_chunk=3):
        self.log: list[tuple] = []
        self.depth_chunk = depth_chunk
        self.fail: dict[str, int] = {}  # step -> number of calls before it fails
        self.interrupt_at: str | None = None
        self.depth_estimator = _Estimator()
        self.created_video: dict | None = None

    # --- video encoder
    def extract_frames(self, video_path, directories, props, settings, tracker=None):
        self.log.append(("extract",))
        files = []
        for i in range(NUM_FRAMES):
            path = directories["frames"] / f"frame_{i + 1:06d}.png"
            path.touch()
            files.append(path)
        return files

    def create_video(self, vr_dir, output_dir, video, settings, tracker=None, max_frames=None):
        self.log.append(("video", max_frames))
        self.created_video = {"frames": len(list(vr_dir.glob("*.png"))), "max": max_frames}
        return True

    # --- depth processor
    finished_depth_maps = staticmethod(DepthMapProcessor.finished_depth_maps)
    mark_stream_complete = staticmethod(DepthMapProcessor.mark_stream_complete)

    def stream_depth_maps(self, frame_files, depth_dir, state_path, settings, tracker=None):
        emitted = self.finished_depth_maps(state_path)
        while emitted < len(frame_files):
            end = min(emitted + self.depth_chunk, len(frame_files))
            for frame in frame_files[emitted:end]:
                (depth_dir / f"{frame.stem}.png").touch()
            self.log.append(("depth", emitted, end))
            emitted = end
            self.mark_stream_complete(state_path, emitted)
            yield emitted

    # --- frame steps
    def _step(self, name, sources, outputs, names, tracker):
        if self.interrupt_at == name:
            raise InterruptedError("stop")
        if name in self.fail:
            if self.fail[name] == 0:
                return False
            self.fail[name] -= 1
        for source in sources:
            missing = [n for n in names if not (source / f"{n}.png").exists()]
            assert not missing, f"{name}: missing inputs {missing} in {source.name}"
        for output in outputs:
            for n in names:
                (output / f"{n}.png").touch()
        self.log.append((name, names[0], names[-1]))
        if tracker:
            tracker.update_progress(
                name, step_name=STEP_NAMES[name], step_progress=len(names), step_total=len(names)
            )
        return True

    def create_stereo_pairs(self, frame_files, depth_files, dirs, settings, tracker=None):
        assert all(f.exists() for f in depth_files)
        outputs = [dirs["left_frames"], dirs["right_frames"]]
        return self._step("stereo", [], outputs, [f.stem for f in frame_files], tracker)

    def apply_distortion(self, left, right, dirs, settings, tracker=None):
        outputs = [dirs["left_distorted"], dirs["right_distorted"]]
        return self._step("distortion", [left[0].parent], outputs, [f.stem for f in left], tracker)

    def crop_frames(self, dirs, settings, tracker=None, total=0, frame_names=None):
        key = "left_distorted" if settings.get("apply_distortion", True) else "left_frames"
        outputs = [dirs["left_cropped"], dirs["right_cropped"]]
        return self._step("crop", [dirs[key]], outputs, frame_names, tracker)

    def apply_upscaling(self, dirs, settings, tracker=None, frame_names=None):
        outputs = [dirs["left_upscaled"], dirs["right_upscaled"]]
        return self._step("upscale", [dirs["left_cropped"]], outputs, frame_names, tracker)

    def assemble_vr_frames(self, dirs, settings, tracker=None, total=0, frame_names=None):
        upscaled = settings.get("upscale_model", "none") != "none"
        source = dirs["left_upscaled" if upscaled else "left_cropped"]
        return self._step("assemble", [source], [dirs["vr_frames"]], frame_names, tracker)

    def close(self):
        self.log.append(("close",))


STEP_NAMES = {
    "stereo": "Stereo Pair Creation",
    "distortion": "Fisheye Distortion",
    "crop": "Crop Frames",
    "upscale": "AI Upscaling",
    "assemble": "VR Assembly",
}


class _Tracker:
    def __init__(self):
        self.updates: list[dict] = []
        self.batches: list[tuple] = []
        self.finished = False

    def update_progress(self, stage, **kwargs):
        self.updates.append(kwargs)

    def set_batch_info(self, index, count, start, size):
        self.batches.append((index, count, start, size))

    def finish(self, message=""):
        self.finished = True


def _settings(**changes):
    settings = {
        "vr_format": "side_by_side",
        "vr_resolution": "square-480",
        "per_eye_width": 480,
        "per_eye_height": 480,
        "vr_output_width": 960,
        "vr_output_height": 480,
        "keep_intermediates": True,
        "apply_distortion": True,
        "fisheye_projection": "stereographic",
        "fisheye_fov": 180,
        "upscale_model": "none",
        "processing_mode": "batched",
        "batch_size": 4,
    }
    settings.update(changes)
    return settings


class _FusedPipeline(_Pipeline):
    """Fake with the fused frame pipeline and segment encoding of the real processors."""

    def process_frames(self, frame_files, depth_files, dirs, settings, tracker=None, assemble=True):
        assert all(f.exists() for f in frame_files + depth_files)
        keep = settings.get("keep_intermediates", True)
        keys = ["left_frames", "left_distorted", "left_cropped"] if keep else []
        keys += ["vr_frames"] if assemble else ["left_cropped", "right_cropped"]
        outputs = [dirs[key] for key in dict.fromkeys(keys)]
        return self._step("fused", [], outputs, [f.stem for f in frame_files], None)

    def encode_segment(self, vr_dir, segment_path, start, count, settings, on_frame=None):
        if "encode" in self.fail:
            if self.fail["encode"] == 0:
                return False
            self.fail["encode"] -= 1
        names = [f"frame_{i + 1:06d}.png" for i in range(start, start + count)]
        assert all((vr_dir / name).exists() for name in names)
        segment_path.touch()
        on_frame(count)
        self.log.append(("segment", start, start + count))
        return True

    def concat_segments(self, segments, output_dir, video, settings, num_frames, **kwargs):
        assert all(path.exists() for path in segments)
        self.log.append(("concat", [path.name for path in segments], num_frames))
        return True


def _run(tmp_path, pipeline, tracker=None, **changes):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    orchestrator = ProcessingOrchestrator(
        depth_processor=pipeline,
        stereo_generator=pipeline,
        distortion_processor=pipeline,
        upscaler=pipeline,
        vr_assembler=pipeline,
        video_encoder=pipeline,
        frame_pipeline=pipeline if isinstance(pipeline, _FusedPipeline) else None,
    )
    with patch(f"{MODULE}.BATCH_SIZE_RANGE", (1, 2000)):
        return orchestrator.process(
            video,
            tmp_path / "out",
            {"frame_count": NUM_FRAMES, "fps": 30.0},
            _settings(**changes),
            tracker,
        )


def _work(pipeline, step=None):
    """Logged work without bookkeeping entries (optionally of one step)."""
    return [e for e in pipeline.log if e[0] != "close" and (step is None or e[0] == step)]


def _status(tmp_path):
    settings_file = next((tmp_path / "out").glob("*-settings.json"))
    return json.loads(settings_file.read_text())["metadata"]["processing_status"]


class TestBatchedMode:
    def test_steps_run_batch_by_batch(self, tmp_path, capsys):
        pipeline = _Pipeline(depth_chunk=3)

        assert _run(tmp_path, pipeline) is True

        f = "frame_{:06d}".format
        assert _work(pipeline) == [
            ("extract",),
            ("depth", 0, 3),
            ("depth", 3, 6),  # depth runs until the batch (frames 1-4) is covered
            ("stereo", f(1), f(4)),
            ("distortion", f(1), f(4)),
            ("crop", f(1), f(4)),
            ("assemble", f(1), f(4)),
            ("depth", 6, 9),
            ("stereo", f(5), f(8)),
            ("distortion", f(5), f(8)),
            ("crop", f(5), f(8)),
            ("assemble", f(5), f(8)),
            ("depth", 9, 10),
            ("stereo", f(9), f(10)),
            ("distortion", f(9), f(10)),
            ("crop", f(9), f(10)),
            ("assemble", f(9), f(10)),
            ("video", NUM_FRAMES),
        ]
        assert pipeline.created_video == {"frames": NUM_FRAMES, "max": NUM_FRAMES}
        assert _status(tmp_path) == "completed"
        assert pipeline.log[-2:] == [("close",), ("close",)]

    def test_tracker_gets_batches_and_whole_video_progress(self, tmp_path, capsys):
        tracker = _Tracker()

        _run(tmp_path, _Pipeline(), tracker)

        assert tracker.batches == [(0, 3, 0, 4), (1, 3, 4, 4), (2, 3, 8, 2)]
        assemble = [u for u in tracker.updates if u.get("step_name") == "VR Assembly"]
        # stages count frames of their batch; the tracker sees frames of the whole video
        assert [(u["step_progress"], u["step_total"]) for u in assemble] == [
            (4, 10),
            (8, 10),
            (10, 10),
        ]
        assert tracker.finished

    def test_optional_steps(self, tmp_path, capsys):
        pipeline = _Pipeline()

        assert _run(tmp_path, pipeline, apply_distortion=False, upscale_model="x2", batch_size=10)

        steps = [e[0] for e in _work(pipeline) if e[0] not in ("depth", "extract")]
        assert steps == ["stereo", "crop", "upscale", "assemble", "video"]


class TestFusedPipeline:
    """Frames go through stereo-to-assembly in one pass; the video is encoded per batch."""

    def test_one_pass_per_batch_and_segment_encoding(self, tmp_path, capsys):
        pipeline = _FusedPipeline(depth_chunk=5)
        tracker = _Tracker()

        assert _run(tmp_path, pipeline, tracker) is True

        f = "frame_{:06d}".format
        segments = [
            "segment_000001-000004.mp4",
            "segment_000005-000008.mp4",
            "segment_000009-000010.mp4",
        ]
        assert _work(pipeline) == [
            ("extract",),
            ("depth", 0, 5),
            ("fused", f(1), f(4)),
            ("segment", 0, 4),
            ("depth", 5, 10),
            ("fused", f(5), f(8)),
            ("segment", 4, 8),
            ("fused", f(9), f(10)),
            ("segment", 8, 10),
            ("concat", segments, NUM_FRAMES),
        ]
        video = [u for u in tracker.updates if u.get("step_name") == "Video Creation"]
        assert [u["step_progress"] for u in video] == [4, 4, 8, 8, 10, 10, 10]
        assert all(u["step_total"] == NUM_FRAMES for u in video)
        state = json.loads((tmp_path / "out" / "pipeline_state.json").read_text())
        assert state["segments"] == [[0, 4], [4, 8], [8, 10]]
        assert state["done"]["encode"] == state["done"]["crop"] == NUM_FRAMES

    def test_no_intermediates_are_written_or_left(self, tmp_path, capsys):
        pipeline = _FusedPipeline()
        pipeline.fail["encode"] = 2  # third batch fails while encoding

        assert _run(tmp_path, pipeline, keep_intermediates=False) is False

        out = tmp_path / "out"
        counts = {d.name: len(list(d.glob("*"))) for d in out.iterdir() if d.is_dir()}
        assert (
            counts["04_left_frames"]
            == counts["05_left_distorted"]
            == counts["06_left_cropped"]
            == 0
        )
        assert counts["02_depth_maps"] == 0
        assert counts["99_vr_frames"] == 2  # only the batch whose segment is missing
        assert counts["98_video_segments"] == 2
        assert counts["00_original_frames"] == 2

        again = _FusedPipeline()
        assert _run(tmp_path, again, keep_intermediates=False) is True
        assert _work(again) == [
            ("segment", 8, 10),
            ("concat", [f"segment_{a:06d}-{b:06d}.mp4" for a, b in ((1, 4), (5, 8), (9, 10))], 10),
        ]
        assert not (out / "98_video_segments").exists()

    def test_sequential_mode(self, tmp_path, capsys):
        pipeline = _FusedPipeline(depth_chunk=5)

        assert _run(tmp_path, pipeline, processing_mode="sequential") is True

        order = [e[0] for e in _work(pipeline)]
        assert order == ["extract"] + ["depth"] * 2 + ["fused"] * 3 + ["segment"] * 3 + ["concat"]

    def test_upscaling_runs_between_fused_pass_and_assembly(self, tmp_path, capsys):
        pipeline = _FusedPipeline()

        assert _run(tmp_path, pipeline, upscale_model="x2", batch_size=10) is True

        order = [e[0] for e in _work(pipeline) if e[0] not in ("extract", "depth")]
        assert order == ["fused", "upscale", "assemble", "segment", "concat"]

    def test_changed_setting_redoes_single_steps_from_kept_intermediates(self, tmp_path, capsys):
        """A new fisheye FOV keeps the stereo pairs: only the later steps run again."""
        _run(tmp_path, _FusedPipeline())
        again = _FusedPipeline()

        assert _run(tmp_path, again, fisheye_fov=120) is True

        steps = [e[0] for e in _work(again)]
        assert "fused" not in steps and "stereo" not in steps and "depth" not in steps
        assert steps[:3] == ["distortion", "crop", "assemble"]
        assert steps[-1] == "concat" and steps.count("segment") == 3

    def test_changed_encoder_only_encodes_again(self, tmp_path, capsys):
        _run(tmp_path, _FusedPipeline())
        again = _FusedPipeline()

        assert _run(tmp_path, again, video_encoder="libx265") is True
        assert {e[0] for e in _work(again)} == {"segment", "concat"}

    def test_frame_interpolation_encodes_in_one_pass(self, tmp_path, capsys):
        pipeline = _FusedPipeline()

        assert _run(tmp_path, pipeline, experimental_frame_interpolation=True) is True

        assert _work(pipeline)[-1] == ("video", NUM_FRAMES)
        assert _work(pipeline, "segment") == []

    def test_failed_frame_processing(self, tmp_path, capsys):
        pipeline = _FusedPipeline()
        pipeline.fail["fused"] = 0

        assert _run(tmp_path, pipeline) is False
        assert "Frame processing failed" in capsys.readouterr().out


class TestSequentialMode:
    def test_each_step_finishes_before_the_next(self, tmp_path, capsys):
        pipeline = _Pipeline(depth_chunk=5)
        tracker = _Tracker()

        assert _run(tmp_path, pipeline, tracker, processing_mode="sequential") is True

        order = [e[0] for e in _work(pipeline)]
        assert order == (
            ["extract"]
            + ["depth"] * 2
            + ["stereo"] * 3
            + ["distortion"] * 3
            + ["crop"] * 3
            + ["assemble"] * 3
            + ["video"]
        )
        assert tracker.batches == []


class TestResume:
    def test_failed_run_continues_where_it_stopped(self, tmp_path, capsys):
        first = _Pipeline()
        first.fail["assemble"] = 1  # second batch fails in the last step

        assert _run(tmp_path, first) is False
        assert _status(tmp_path) == "failed"

        second = _Pipeline()
        tracker = _Tracker()
        assert _run(tmp_path, second, tracker) is True

        f = "frame_{:06d}".format
        assert _work(second, "extract") == []
        assert _work(second, "stereo") == [("stereo", f(9), f(10))]  # batch 2 was done up to crop
        assert _work(second, "assemble") == [("assemble", f(5), f(8)), ("assemble", f(9), f(10))]
        assert _work(second, "depth") == [("depth", 9, 10)]
        restored = {
            u["step_name"]: u["step_progress"]
            for u in tracker.updates
            if u.get("phase") == "resume"
        }
        assert restored["Frame Extraction"] == 10
        assert restored["Depth Map Generation"] == 9
        assert restored["Crop Frames"] == 8
        assert restored["VR Assembly"] == 4
        assert tracker.batches[0] == (1, 3, 4, 4)
        assert len(list((tmp_path / "out").glob("*-settings.json"))) == 1

    def test_stop_marks_batch_paused_and_can_resume(self, tmp_path, capsys):
        first = _Pipeline()
        first.interrupt_at = "crop"

        with pytest.raises(InterruptedError):
            _run(tmp_path, first)
        assert _status(tmp_path) == "paused"
        assert first.log[-2:] == [("close",), ("close",)]

        second = _Pipeline()
        assert _run(tmp_path, second) is True
        assert _work(second, "extract") == []
        assert _work(second, "distortion")[0] == ("distortion", "frame_000005", "frame_000008")

    def test_finished_run_is_not_repeated(self, tmp_path, capsys):
        _run(tmp_path, _Pipeline())
        again = _Pipeline()

        assert _run(tmp_path, again) is True
        assert _work(again) == [("video", NUM_FRAMES)]

    def test_changed_setting_redoes_only_affected_steps(self, tmp_path, capsys):
        _run(tmp_path, _Pipeline())
        again = _Pipeline()

        assert _run(tmp_path, again, fisheye_fov=120) is True

        steps = {e[0] for e in _work(again)}
        assert steps == {"distortion", "crop", "assemble", "video"}

    def test_changed_time_range_extracts_again(self, tmp_path, capsys):
        _run(tmp_path, _Pipeline())
        again = _Pipeline()

        assert _run(tmp_path, again, start_time="00:01") is True

        assert _work(again)[0] == ("extract",)
        assert len(_work(again, "depth")) > 0

    def test_mode_can_change_between_runs(self, tmp_path, capsys):
        first = _Pipeline()
        first.fail["assemble"] = 1
        _run(tmp_path, first)

        second = _Pipeline()
        assert _run(tmp_path, second, processing_mode="sequential") is True
        assert _work(second, "assemble")[0][1] == "frame_000005"


class TestWithoutIntermediates:
    def test_consumed_files_are_deleted_per_batch(self, tmp_path, capsys):
        pipeline = _Pipeline()
        pipeline.fail["assemble"] = 1

        assert _run(tmp_path, pipeline, keep_intermediates=False) is False

        out = tmp_path / "out"
        assert len(list((out / "99_vr_frames").glob("*.png"))) == 4
        # batch 1 is finished: its intermediates are gone; batch 2 still has its inputs
        assert len(list((out / "04_left_frames").glob("*.png"))) == 0
        assert len(list((out / "06_left_cropped").glob("*.png"))) == 4
        # original frames go too, once depth estimation and stereo are past them
        assert len(list((out / "00_original_frames").glob("*.png"))) == NUM_FRAMES - 4
        assert json.loads((out / "pipeline_state.json").read_text())["pruned"] is True

    def test_everything_is_removed_after_success(self, tmp_path, capsys):
        assert _run(tmp_path, _Pipeline(), keep_intermediates=False) is True

        out = tmp_path / "out"
        assert not (out / "99_vr_frames").exists()
        assert not (out / "pipeline_state.json").exists()
        assert not (out / "depth_stream_state.npz").exists()

    def test_resume_works_with_deleted_frames(self, tmp_path, capsys):
        first = _Pipeline()
        first.fail["assemble"] = 1
        _run(tmp_path, first, keep_intermediates=False)

        again = _Pipeline()
        assert _run(tmp_path, again, keep_intermediates=False) is True
        assert _work(again, "extract") == []
        assert _work(again, "assemble")[0] == ("assemble", "frame_000005", "frame_000008")

    def test_changed_setting_starts_over(self, tmp_path, capsys):
        """Nothing a changed step could start from is left, so the frames are extracted again."""
        first = _Pipeline()
        first.fail["assemble"] = 1
        _run(tmp_path, first, keep_intermediates=False)

        again = _Pipeline()
        assert _run(tmp_path, again, keep_intermediates=False, fisheye_fov=120) is True
        assert _work(again)[0] == ("extract",)
        assert _work(again, "depth")[0] == ("depth", 0, 3)


class TestFailures:
    def test_extraction_failure(self, tmp_path, capsys):
        pipeline = _Pipeline()
        pipeline.extract_frames = lambda *args, **kwargs: []

        assert _run(tmp_path, pipeline) is False
        assert _status(tmp_path) == "failed"

    def test_depth_ending_early(self, tmp_path, capsys):
        pipeline = _Pipeline()
        pipeline.stream_depth_maps = lambda *args, **kwargs: iter([3])

        assert _run(tmp_path, pipeline) is False
        assert "ended early" in capsys.readouterr().out


class TestBatchSize:
    @pytest.mark.parametrize(
        "value, expected", [(100, 100), ("60", 60), (1, 10), (99999, 2000), (None, 120), ("x", 120)]
    )
    def test_clamped(self, value, expected):
        assert ProcessingOrchestrator._batch_size({"batch_size": value}) == expected

    def test_source_identity_of_missing_file(self):
        assert ProcessingOrchestrator._source_identity("/no/such/video.mp4") == "video.mp4"

    def test_remove_frame_files_ignores_unknown_dirs(self, tmp_path):
        (tmp_path / "frame_000001.png").touch()
        ProcessingOrchestrator._remove_frame_files(
            {"a": tmp_path}, ("a", "missing"), [Path("frame_000001.png")]
        )
        assert not (tmp_path / "frame_000001.png").exists()


class TestDiskEstimate:
    PROPERTIES = {"frame_count": 3000, "fps": 30.0, "width": 1920, "height": 1080}

    def _report(self, tmp_path, free, capsys, **changes):
        with patch(f"{MODULE}.get_available_space", return_value=free):
            ProcessingOrchestrator._report_disk_estimate(
                tmp_path, self.PROPERTIES, _settings(**changes)
            )
        return capsys.readouterr().out

    def test_prints_estimate_and_free_space(self, tmp_path, capsys):
        out = self._report(tmp_path, 500 * 1024**3, capsys)
        assert "Estimated disk space:" in out and "500.0 GB free" in out
        assert "may not fit" not in out

    def test_warns_when_the_disk_looks_too_small(self, tmp_path, capsys):
        out = self._report(tmp_path, 1024**3, capsys)
        assert "may not fit" in out and "Without keeping intermediate files" in out
        out = self._report(tmp_path, 1024**2, capsys, keep_intermediates=False)
        assert "may not fit" in out and "Without keeping" not in out

    def test_unknown_video_properties_are_skipped(self, tmp_path, capsys):
        ProcessingOrchestrator._report_disk_estimate(tmp_path, {}, _settings())
        assert capsys.readouterr().out == ""


class TestFisheyeViewNote:
    def _note(self, capsys, width, height, **changes):
        ProcessingOrchestrator._warn_about_cut_off_sides(
            {"width": width, "height": height}, _settings(**{"apply_distortion": True, **changes})
        )
        return capsys.readouterr().out

    def test_says_how_much_is_outside_the_view(self, capsys):
        assert "about 44% of the video's width" in self._note(
            capsys, 1920, 1080, fisheye_frame_fit=0
        )
        assert "about 22% of the video's width" in self._note(
            capsys, 1920, 1080, fisheye_frame_fit=0.5
        )
        assert "about 44% of the video's height" in self._note(
            capsys, 1080, 1920, fisheye_frame_fit=0
        )

    def test_quiet_when_little_or_nothing_is_cut(self, capsys):
        assert self._note(capsys, 1920, 1080, fisheye_frame_fit=1) == ""
        assert self._note(capsys, 1080, 1080) == ""
        assert self._note(capsys, 0, 0) == ""
        assert self._note(capsys, 1920, 1080, apply_distortion=False) == ""
