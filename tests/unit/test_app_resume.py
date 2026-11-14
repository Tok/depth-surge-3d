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


def _callback(monkeypatch, **kwargs):
    """ProgressCallback whose emitted progress is collected instead of sent."""
    emitted = []
    monkeypatch.setattr(
        app_module.socketio, "emit", lambda name, data, room=None: emitted.append((name, data))
    )
    monkeypatch.setattr(app_module.socketio, "sleep", lambda seconds: None)
    monkeypatch.setitem(app_module.current_processing, "stop_requested", False)
    return app_module.ProgressCallback("session", 200, "batched", **kwargs), emitted


def test_progress_reports_every_step_and_the_batch(monkeypatch, capsys):
    callback, emitted = _callback(monkeypatch, skipped_steps={"AI Upscaling"})

    callback.update_progress("x", step_name="Frame Extraction", step_progress=200, step_total=200)
    callback.set_batch_info(1, 2, 100, 100)
    callback.update_progress(
        "x", step_name="Depth Map Generation", step_progress=200, step_total=200
    )
    callback.update_progress(
        "Assembling", step_name="VR Assembly", step_progress=150, step_total=200
    )
    callback.last_update_time = 0  # let the next update through the throttle
    callback.update_progress(
        "Assembling", step_name="VR Assembly", step_progress=160, step_total=200
    )

    data = emitted[-1][1]
    steps = {step["name"]: step for step in data["steps"]}
    assert (steps["Frame Extraction"]["progress"], steps["Frame Extraction"]["total"]) == (200, 200)
    assert (steps["VR Assembly"]["progress"], steps["VR Assembly"]["total"]) == (60, 100)
    assert steps["VR Assembly"]["in_batch"] and steps["AI Upscaling"]["skipped"]
    assert data["batch"]["index"] == 1 and data["batch"]["count"] == 2
    assert 0 < data["batch"]["progress"] < 1
    assert data["step_index"] == 6 and 0 < data["progress"] < 100
    assert app_module.current_processing["last_progress"] is data
    app_module.current_processing["last_progress"] = None


def test_progress_updates_are_throttled_but_never_lost(monkeypatch, capsys):
    """Regression: throttled updates were dropped, so finished steps stayed below 100%."""
    callback, emitted = _callback(monkeypatch)

    for frame in range(1, 51):
        callback.update_progress("x", step_name="Crop Frames", step_progress=frame, step_total=50)

    assert len(emitted) == 2  # first update and the final one
    assert emitted[-1][1]["steps"][4]["progress"] == 50
    app_module.current_processing["last_progress"] = None


def test_stop_request_interrupts_processing(monkeypatch):
    callback, _ = _callback(monkeypatch)
    monkeypatch.setitem(app_module.current_processing, "stop_requested", True)
    with pytest.raises(InterruptedError):
        callback.update_progress("x", step_name="Crop Frames", step_progress=1, step_total=50)


def test_join_sends_full_progress_of_running_job():
    """A reloaded page gets the step and batch state, not just a percentage."""
    state = app_module.current_processing
    app_module.reset_progress_state("running")
    state["last_progress"] = {"progress": 42.0, "steps": [], "batch": {"index": 1, "count": 3}}
    client = app_module.socketio.test_client(app_module.app)
    try:
        client.emit("join_session", {"session_id": "running"})
        updates = [m for m in client.get_received() if m["name"] == "progress_update"]
        assert updates[0]["args"][0]["batch"] == {"index": 1, "count": 3}
        assert (
            app_module.app.test_client().get("/status").get_json()["last_progress"]["progress"]
            == 42.0
        )
    finally:
        client.disconnect()
        state["session_id"] = None
        state["last_progress"] = None


def test_assemble_partial_endpoint(tmp_path, monkeypatch):
    client = app_module.app.test_client()

    missing = client.post("/assemble_partial", json={"output_dir": str(tmp_path / "nope")})
    assert missing.get_json()["success"] is False

    monkeypatch.setattr(app_module, "assemble_partial_video", lambda path: None)
    assert (
        client.post("/assemble_partial", json={"output_dir": str(tmp_path)}).get_json()["success"]
        is False
    )

    monkeypatch.setattr(app_module, "assemble_partial_video", lambda path: path / "partial.mp4")
    monkeypatch.setattr(app_module, "count_finished_vr_frames", lambda path: 120)
    data = client.post("/assemble_partial", json={"output_dir": str(tmp_path)}).get_json()
    assert data["success"] and data["frames"] == 120
    assert data["output_path"].endswith("partial.mp4")

    def broken(path):
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr(app_module, "assemble_partial_video", broken)
    data = client.post("/assemble_partial", json={"output_dir": str(tmp_path)}).get_json()
    assert data == {"success": False, "error": "ffmpeg exploded"}


def test_resumable_jobs_report_progress_and_ignore_partial_videos(tmp_path, monkeypatch):
    job = tmp_path / "job"
    (job / "00_original_frames").mkdir(parents=True)
    (job / "clip.mp4").touch()
    (job / "clip_3D_side-by-side_auto_partial_120frames.mp4").touch()
    (job / "pipeline_state.json").write_text(
        json.dumps({"total_frames": 240, "done": {"assemble": 120}})
    )
    done = tmp_path / "done"
    (done / "00_original_frames").mkdir(parents=True)
    (done / "clip.mp4").touch()
    (done / "clip_3D_side-by-side_auto.mp4").touch()
    monkeypatch.setitem(app_module.app.config, "OUTPUT_FOLDER", str(tmp_path))

    jobs = app_module.app.test_client().get("/detect_resumable").get_json()["jobs"]

    assert [j["name"] for j in jobs] == ["job"]
    assert (jobs[0]["finished_frames"], jobs[0]["total_frames"]) == (120, 240)


@pytest.fixture
def job_starts(monkeypatch):
    """Record job launches instead of running them; leaves the app idle afterwards."""
    started = []
    state = app_module.current_processing
    monkeypatch.setattr(
        app_module.socketio,
        "start_background_task",
        lambda target, *args: started.append(args) or "thread",
    )
    monkeypatch.setattr(app_module, "SHUTDOWN_FLAG", False)  # set by the shutdown tests
    state["active"] = False
    app_module.job_queue.clear()
    yield started
    app_module.job_queue.clear()
    state.update({"active": False, "session_id": None, "thread": None, "output_dir": None})


def test_second_job_is_queued_and_started_when_the_first_ends(tmp_path, job_starts):
    first = app_module.submit_job(tmp_path / "a.mp4", {"x": 1}, tmp_path / "a")
    second = app_module.submit_job(tmp_path / "b.mp4", {"x": 2}, tmp_path / "b")
    third = app_module.submit_job(tmp_path / "c.mp4", {}, tmp_path / "c")

    assert first["queued"] is False and len(job_starts) == 1
    assert (second["queued"], second["position"]) == (True, 1)
    assert third["position"] == 2
    assert app_module.current_processing["session_id"] == first["session_id"]

    app_module._finish_job()

    assert len(job_starts) == 2 and job_starts[1][0] == second["session_id"]
    assert app_module.current_processing["active"] is True
    assert app_module.current_processing["output_dir"] == str(tmp_path / "b")
    assert [job["session_id"] for job in app_module.job_queue] == [third["session_id"]]

    app_module._finish_job()
    app_module._finish_job()
    assert app_module.current_processing["active"] is False and len(job_starts) == 3


def test_same_directory_cannot_run_or_queue_twice(tmp_path, job_starts):
    app_module.submit_job(tmp_path / "a.mp4", {}, tmp_path / "a")
    app_module.submit_job(tmp_path / "b.mp4", {}, tmp_path / "b")

    for directory in ("a", "b"):
        again = app_module.submit_job(tmp_path / "x.mp4", {}, tmp_path / directory)
        assert again == {"success": False, "error": "This job is already running or queued"}
    assert len(app_module.job_queue) == 1


def test_process_endpoint_queues_instead_of_refusing(tmp_path, monkeypatch, job_starts):
    monkeypatch.setitem(app_module.app.config, "OUTPUT_FOLDER", str(tmp_path))
    for name in ("one", "two"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "clip.mp4").touch()
    client = app_module.app.test_client()

    first = client.post("/process", json={"output_dir": str(tmp_path / "one"), "settings": {}})
    second = client.post("/process", json={"output_dir": str(tmp_path / "two"), "settings": {}})
    duplicate = client.post("/process", json={"output_dir": str(tmp_path / "two"), "settings": {}})

    assert first.get_json()["queued"] is False
    assert second.get_json()["queued"] is True
    assert duplicate.status_code == 409

    jobs = client.get("/jobs").get_json()
    assert jobs["running"]["name"] == "one" and jobs["running"]["session_id"]
    assert [job["name"] for job in jobs["queued"]] == ["two"]
    assert jobs["incomplete"] == []

    removed = client.post("/queue/remove", json={"session_id": jobs["queued"][0]["session_id"]})
    assert removed.get_json()["success"] is True
    assert (
        client.post("/queue/remove", json={"session_id": "unknown"}).get_json()["success"] is False
    )
    assert client.get("/jobs").get_json()["queued"] == []


def test_resume_endpoint_queues_too(tmp_path, job_starts):
    job = tmp_path / "job"
    job.mkdir()
    (job / "clip.mp4").touch()
    app_module.submit_job(tmp_path / "a.mp4", {}, tmp_path / "running")

    data = app_module.app.test_client().post("/resume", json={"output_dir": str(job)}).get_json()

    assert data["success"] and data["queued"] and data["output_dir"] == str(job.resolve())


def test_jobs_lists_unfinished_batches_with_status(tmp_path, monkeypatch, job_starts):
    monkeypatch.setitem(app_module.app.config, "OUTPUT_FOLDER", str(tmp_path))

    def batch(name, status=None, state=None, final=False):
        directory = tmp_path / name
        (directory / "00_original_frames").mkdir(parents=True)
        (directory / "clip.mp4").touch()
        if status:
            (directory / "clip_1-settings.json").write_text(
                json.dumps({"metadata": {"processing_status": status}})
            )
        if state:
            (directory / "pipeline_state.json").write_text(json.dumps(state))
        if final:
            (directory / "clip_3D_side-by-side_auto.mp4").touch()

    batch("stopped", "paused", {"total_frames": 240, "done": {"assemble": 120}})
    batch(
        "crashed",
        "in_progress",
        {"total_frames": 100, "done": {"encode": 40}, "segments": [[0, 40]]},
    )
    batch("broken", "failed")
    batch("old")
    batch("done", "completed", final=True)
    (tmp_path / "not_a_job").mkdir()

    jobs = {
        j["name"]: j for j in app_module.app.test_client().get("/jobs").get_json()["incomplete"]
    }

    assert set(jobs) == {"stopped", "crashed", "broken", "old"}
    assert (jobs["stopped"]["status"], jobs["stopped"]["finished_frames"]) == ("paused", 120)
    assert (jobs["crashed"]["status"], jobs["crashed"]["finished_frames"]) == ("interrupted", 40)
    assert jobs["broken"]["status"] == "failed" and jobs["broken"]["total_frames"] is None
    assert jobs["old"]["status"] == "interrupted"


def test_disk_estimate_endpoint(tmp_path, monkeypatch):
    job = tmp_path / "job"
    job.mkdir()
    (job / "clip.mp4").touch()
    info = {"width": 1920, "height": 1080, "fps": 30.0, "frame_count": 900}
    monkeypatch.setattr(app_module, "get_video_info", lambda path: info)
    monkeypatch.setattr(app_module, "get_available_space", lambda path: 5 * 1024**3)
    client = app_module.app.test_client()

    def estimate(**settings):
        body = {"output_dir": str(job), "settings": {"vr_resolution": "auto", **settings}}
        return client.post("/estimate", json=body).get_json()

    # keeping intermediates is a server option; a value sent by the page is ignored
    monkeypatch.setattr(app_module, "KEEP_INTERMEDIATES", True)
    full = estimate(keep_intermediates=False)
    ranged = estimate(start_time="00:10", end_time="00:20")
    monkeypatch.setattr(app_module, "KEEP_INTERMEDIATES", False)
    lean = estimate(keep_intermediates=True)

    assert full["success"] and full["frames"] == 900 and full["free_bytes"] == 5 * 1024**3
    assert ranged["frames"] == 300 and ranged["estimated_bytes"] < full["estimated_bytes"]
    assert lean["estimated_bytes"] < full["estimated_bytes"] / 3
    assert (
        client.post("/estimate", json={"output_dir": str(tmp_path / "nope")}).get_json()["success"]
        is False
    )
    bad = estimate(vr_resolution="custom:axb")
    assert bad["success"] is False


def test_resolve_output_dimensions():
    settings = {"vr_resolution": "square-1k", "vr_format": "over_under"}
    app_module.resolve_output_dimensions(settings, {"width": 1280, "height": 720, "fps": 25.0})
    assert (settings["per_eye_width"], settings["per_eye_height"]) == (1080, 1080)
    assert (settings["vr_output_width"], settings["vr_output_height"]) == (1080, 2160)
    assert settings["source_fps"] == 25.0

    auto = {"vr_resolution": "auto"}
    app_module.resolve_output_dimensions(auto, {"width": 1280, "height": 720, "fps": 25.0})
    assert auto["vr_resolution"] != "auto"


def test_server_console_shows_the_same_progress_as_the_browser(monkeypatch, capsys):
    """The server's terminal gets the CLI's progress bars instead of plain status lines."""
    callback, _ = _callback(monkeypatch)
    console = callback.console

    callback.set_batch_info(0, 2, 0, 50)
    callback.update_progress("x", step_name="Crop Frames", step_progress=25, step_total=100)

    assert console.batched is True
    assert console.state.batch == callback.state.batch
    assert console.state.done == callback.state.done
    callback.finish()
    callback.close()
    assert console._closed
    assert "Overall:" not in capsys.readouterr().out
    assert app_module.ProgressCallback("s", 10, "sequential").console.batched is False
    app_module.current_processing["last_progress"] = None


def test_processing_mode_is_known_before_the_first_progress_update(tmp_path, job_starts):
    """The UI shows the batch loop from the start of a batched job."""
    app_module.submit_job(tmp_path / "a.mp4", {"processing_mode": "sequential"}, tmp_path / "a")
    assert app_module.current_processing["processing_mode"] == "sequential"
    jobs = app_module.app.test_client().get("/jobs").get_json()
    assert jobs["running"]["processing_mode"] == "sequential"

    app_module._finish_job()
    app_module.submit_job(tmp_path / "b.mp4", {}, tmp_path / "b")
    assert app_module.current_processing["processing_mode"] == "batched"
    client = app_module.socketio.test_client(app_module.app)
    try:
        client.emit("join_session", {"session_id": app_module.current_processing["session_id"]})
        update = [m for m in client.get_received() if m["name"] == "progress_update"][0]
        assert update["args"][0]["processing_mode"] == "batched"
    finally:
        client.disconnect()


def test_keeping_intermediates_is_a_server_option(monkeypatch):
    """The page has no per-job switch; it says how the server was started."""
    client = app_module.app.test_client()

    monkeypatch.setattr(app_module, "KEEP_INTERMEDIATES", False)
    page = client.get("/").get_data(as_text=True)
    assert 'id="keepIntermediates"' not in page
    assert "const KEEP_INTERMEDIATES = false;" in page
    assert "Intermediate files are deleted as the job goes" in page

    monkeypatch.setattr(app_module, "KEEP_INTERMEDIATES", True)
    page = client.get("/").get_data(as_text=True)
    assert "const KEEP_INTERMEDIATES = true;" in page
    assert "Intermediate files are kept" in page
