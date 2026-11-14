"""Tests for the pipeline resume state."""

import json

from src.depth_surge_3d.processing.orchestration.pipeline_state import (
    PipelineState,
    active_step_keys,
    step_fingerprints,
)

SETTINGS = {
    "start_time": None,
    "end_time": None,
    "baseline": 0.065,
    "focal_length": 1000,
    "apply_distortion": True,
    "fisheye_fov": 180,
    "upscale_model": "none",
    "vr_format": "side_by_side",
}


def _fingerprints(**changes):
    return step_fingerprints({**SETTINGS, **changes}, "video.mp4|100", "model")


class TestActiveSteps:
    def test_optional_steps(self):
        assert active_step_keys(SETTINGS) == [
            "extract",
            "depth",
            "stereo",
            "distortion",
            "crop",
            "assemble",
            "encode",
        ]
        keys = active_step_keys({"apply_distortion": False, "upscale_model": "x2"})
        assert keys == ["extract", "depth", "stereo", "crop", "upscale", "assemble", "encode"]


class TestFingerprints:
    def test_stable(self):
        assert _fingerprints() == _fingerprints()

    def test_change_affects_step_and_everything_after(self):
        base, changed = _fingerprints(), _fingerprints(fisheye_fov=120)
        same = [key for key in base if base[key] == changed[key]]
        assert same == ["extract", "depth", "stereo"]

    def test_source_and_model_identity(self):
        base = _fingerprints()
        other_video = step_fingerprints(SETTINGS, "other.mp4|5", "model")
        other_model = step_fingerprints(SETTINGS, "video.mp4|100", "other model")
        assert other_video["extract"] != base["extract"]
        assert other_model["extract"] == base["extract"]
        assert other_model["depth"] != base["depth"]

    def test_irrelevant_setting_changes_nothing(self):
        assert _fingerprints(keep_intermediates=False, batch_size=7) == _fingerprints()


class TestPipelineState:
    def _state(self, tmp_path, done=None):
        state = PipelineState.load(tmp_path)
        state.reconcile(_fingerprints())
        state.total_frames = 100
        state.extracted = True
        state.done = dict(done or {"depth": 60, "stereo": 50, "distortion": 50, "assemble": 50})
        state.save()
        return state

    def test_fresh_state(self, tmp_path):
        state = PipelineState.load(tmp_path)
        assert (state.total_frames, state.extracted, state.done) == (0, False, {})
        assert state.frames_done("stereo") == 0
        assert state.reconcile(_fingerprints()) == "extract"

    def test_save_and_load(self, tmp_path):
        self._state(tmp_path)
        loaded = PipelineState.load(tmp_path)
        assert loaded.total_frames == 100 and loaded.extracted
        assert loaded.frames_done("assemble") == 50
        assert not list(tmp_path.glob("*.tmp"))

    def test_unreadable_or_foreign_state_starts_fresh(self, tmp_path):
        (tmp_path / "pipeline_state.json").write_text("{broken")
        assert PipelineState.load(tmp_path).done == {}
        (tmp_path / "pipeline_state.json").write_text(
            json.dumps({"version": 999, "done": {"a": 1}})
        )
        assert PipelineState.load(tmp_path).done == {}

    def test_same_settings_keep_progress(self, tmp_path):
        state = self._state(tmp_path)
        assert state.reconcile(_fingerprints()) is None
        assert state.frames_done("assemble") == 50

    def test_changed_step_and_later_steps_restart(self, tmp_path):
        state = self._state(tmp_path)
        assert state.reconcile(_fingerprints(fisheye_fov=120)) == "distortion"
        assert state.done == {"depth": 60, "stereo": 50}
        assert state.extracted

    def test_changed_time_range_restarts_everything(self, tmp_path):
        state = self._state(tmp_path)
        assert state.reconcile(_fingerprints(start_time="00:05")) == "extract"
        assert state.done == {} and not state.extracted and state.total_frames == 0

    def test_pruned_intermediates_restart_from_depth(self, tmp_path):
        """Without intermediates a later step can't be redone on its own."""
        state = self._state(tmp_path)
        state.pruned = True
        assert state.reconcile(_fingerprints(fisheye_fov=120)) == "depth"
        assert state.done == {} and state.extracted and not state.pruned

    def test_deleted_frames_mean_extracting_again(self, tmp_path):
        """Without intermediates the original frames of finished batches are gone too."""
        state = self._state(tmp_path)
        state.pruned = True
        state.frames_pruned = 50
        assert state.reconcile(_fingerprints(fisheye_fov=120)) == "extract"
        assert not state.extracted and state.frames_pruned == 0

    def test_encoder_change_only_redoes_the_video(self, tmp_path):
        state = self._state(tmp_path)
        state.done["encode"] = 50
        state.segments = [[0, 50]]
        assert state.reconcile(_fingerprints(video_encoder="libx265")) == "encode"
        assert state.frames_done("assemble") == 50
        assert state.segments == [] and state.frames_done("encode") == 0

    def test_segments_and_pruned_frames_survive_save(self, tmp_path):
        state = self._state(tmp_path)
        state.segments = [[0, 50], [50, 100]]
        state.frames_pruned = 40
        state.save()
        loaded = PipelineState.load(tmp_path)
        assert loaded.segments == [[0, 50], [50, 100]] and loaded.frames_pruned == 40

    def test_disabled_step_is_dropped(self, tmp_path):
        state = self._state(tmp_path)
        state.reconcile(_fingerprints(apply_distortion=False))
        assert "distortion" not in state.done and "distortion" not in state.fingerprints

    def test_restart_from_a_step(self, tmp_path):
        state = self._state(tmp_path)
        state.done["encode"] = 50
        state.segments = [[0, 50]]
        assert state.restart_from("encode") == "encode"
        assert state.segments == [] and state.frames_done("assemble") == 50

        state.pruned = True
        assert state.restart_from("encode") == "depth"
        assert state.done == {}
