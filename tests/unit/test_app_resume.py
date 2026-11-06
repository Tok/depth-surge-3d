"""Tests for web UI resume settings (app.detect_resume_settings)."""

import json

import pytest

app_module = pytest.importorskip("app")


def test_uses_saved_batch_settings(tmp_path):
    saved = {
        "metadata": {"batch_name": "clip_1"},
        "processing_settings": {
            "depth_model_version": "v2",
            "fisheye_fov": 150,
            "hole_fill_quality": "advanced",
            "per_eye_width": 1234,  # derived: must be recomputed, not reused
        },
    }
    (tmp_path / "clip_1-settings.json").write_text(json.dumps(saved))

    settings = app_module.detect_resume_settings(tmp_path)

    assert settings["depth_model_version"] == "v2"
    assert settings["fisheye_fov"] == 150
    assert settings["hole_fill_quality"] == "advanced"
    assert "per_eye_width" not in settings


def test_falls_back_to_current_defaults(tmp_path):
    from src.depth_surge_3d.core.constants import DEFAULT_SETTINGS

    settings = app_module.detect_resume_settings(tmp_path)

    # Regression: resume used stale hardcoded values (fisheye FOV 105, crop factor 1.0)
    assert settings["fisheye_fov"] == DEFAULT_SETTINGS["fisheye_fov"]
    assert settings["fisheye_crop_factor"] == DEFAULT_SETTINGS["fisheye_crop_factor"]
    assert settings["super_sample"] == DEFAULT_SETTINGS["super_sample"]


def test_requests_do_not_kill_running_jobs():
    """Regression: a per-request teardown hook ran `pkill ffmpeg` after every HTTP request,
    killing frame extraction/encoding of a running job whenever the browser polled."""
    from unittest.mock import patch

    client = app_module.app.test_client()
    with patch.object(app_module.subprocess, "run") as mock_run:
        client.get("/status")
        client.get("/system_info")
    assert not any("pkill" in str(call) for call in mock_run.call_args_list)


def test_shutdown_cleanup_targets_own_children_only():
    import os
    from unittest.mock import patch

    with patch.object(app_module.subprocess, "run") as mock_run:
        app_module.cleanup_processes()
    args = mock_run.call_args[0][0]
    assert args[:4] == ["pkill", "-TERM", "-P", str(os.getpid())]


def test_status_while_job_running():
    """Regression: /status returned 500 while a job ran (thread object in the response)."""
    import threading

    app_module.current_processing["thread"] = threading.Thread(target=lambda: None)
    try:
        response = app_module.app.test_client().get("/status")
    finally:
        app_module.current_processing["thread"] = None
    assert response.status_code == 200
    assert "thread" not in response.get_json()


def test_new_job_starts_with_fresh_progress():
    """Regression: a second job's UI started at the previous job's final progress."""
    state = app_module.current_processing
    state.update({"progress": 59.7, "stage": "Applying distortion", "step_index": 3})

    app_module.reset_progress_state("new-session")

    assert state["progress"] == 0
    assert state["step_index"] == 0
    assert state["stage"] == "Initializing..."
    assert state["session_id"] == "new-session"
    state["session_id"] = None


def test_join_sends_initial_progress_only_for_running_session():
    state = app_module.current_processing
    state.update({"progress": 59.7, "stage": "Applying distortion"})
    app_module.reset_progress_state("running")
    client = app_module.socketio.test_client(app_module.app)
    try:
        client.emit("join_session", {"session_id": "stale-session"})
        assert not [m for m in client.get_received() if m["name"] == "progress_update"]

        client.emit("join_session", {"session_id": "running"})
        updates = [m for m in client.get_received() if m["name"] == "progress_update"]
        assert updates and updates[0]["args"][0]["progress"] == 0
    finally:
        client.disconnect()
        state["session_id"] = None


def test_skipped_steps_carry_no_progress_weight():
    """Regression: disabled upscaling kept its 18% weight, so progress jumped past it."""
    steps = app_module.ProgressCallback("s", 10).steps
    weights = app_module._effective_step_weights(steps, {"AI Upscaling"})
    assert weights[steps.index("AI Upscaling")] == 0
    assert abs(sum(weights) - 1.0) < 1e-9


def test_crop_step_name_matches_progress_steps():
    """Regression: the crop step reported "Frame Cropping", which isn't in the step list, so
    overall progress fell back to the start of the distortion step."""
    import inspect

    from src.depth_surge_3d.processing.frames import distortion_processor

    steps = app_module.ProgressCallback("s", 10).steps
    names = set(
        __import__("re").findall(r'step_name="([^"]+)"', inspect.getsource(distortion_processor))
    )
    assert names and names <= set(steps)
