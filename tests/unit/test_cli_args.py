"""Tests for CLI argument parsing (depth_surge_3d.py)."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("depth_surge_3d_cli", ROOT / "depth_surge_3d.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "src"))
    spec.loader.exec_module(module)
    return module


class TestVrResolutionArg:
    """Regression #17: --vr-resolution custom:WxH was rejected by argparse."""

    @pytest.mark.parametrize("value", ["auto", "custom:1376x768", "custom:2560x1080"])
    def test_accepts(self, cli, value):
        args = cli.create_argument_parser().parse_args(["video.mp4", "--vr-resolution", value])
        assert args.vr_resolution == value

    def test_accepts_every_preset(self, cli):
        parser = cli.create_argument_parser()
        for preset in cli.VR_RESOLUTIONS:
            assert parser.parse_args(["v.mp4", "--vr-resolution", preset]).vr_resolution == preset

    @pytest.mark.parametrize(
        "value", ["custom", "custom:abc", "custom:0x100", "custom:1920", "16x9-9000p"]
    )
    def test_rejects(self, cli, value, capsys):
        with pytest.raises(SystemExit):
            cli.create_argument_parser().parse_args(["video.mp4", "--vr-resolution", value])
        assert "--vr-resolution" in capsys.readouterr().err

    def test_default(self, cli):
        assert cli.create_argument_parser().parse_args(["video.mp4"]).vr_resolution == "auto"


class TestBatchArgs:
    def test_defaults(self, cli):
        args = cli.create_argument_parser().parse_args(["video.mp4"])
        assert (args.mode, args.batch_size) == ("batched", 120)

    def test_mode_and_batch_size(self, cli):
        args = cli.create_argument_parser().parse_args(
            ["video.mp4", "--mode", "sequential", "--batch-size", "60"]
        )
        assert (args.mode, args.batch_size) == ("sequential", 60)

    @pytest.mark.parametrize("value", ["5", "5000", "many"])
    def test_rejects_bad_batch_size(self, cli, value, capsys):
        with pytest.raises(SystemExit):
            cli.create_argument_parser().parse_args(["video.mp4", "--batch-size", value])

    def test_progress_tracker_matches_settings(self, cli):
        tracker = cli.create_progress_tracker(
            {"apply_distortion": False, "upscale_model": "none", "processing_mode": "sequential"}
        )
        assert tracker.state.skipped == {"Fisheye Distortion", "AI Upscaling"}
        assert tracker.batched is False
        assert cli.create_progress_tracker({"upscale_model": "x2"}).state.skipped == set()


class TestAssemblePartialCommand:
    def test_missing_directory(self, cli, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["prog", "--assemble-partial", str(tmp_path / "nope")])
        assert cli.main() == 1

    def test_success_and_failure(self, cli, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["prog", "--assemble-partial", str(tmp_path)])
        monkeypatch.setattr(cli, "assemble_partial_video", lambda path: tmp_path / "partial.mp4")
        assert cli.main() == 0
        assert "partial.mp4" in capsys.readouterr().out

        monkeypatch.setattr(cli, "assemble_partial_video", lambda path: None)
        assert cli.main() == 1


class TestResumeCommand:
    """Regression: --resume passed unknown settings to process_video and used the default model."""

    def _batch(self, tmp_path, status="paused"):
        import json

        settings = {
            "metadata": {
                "batch_name": "clip_1",
                "source_video": "/videos/clip.mp4",
                "processing_status": status,
            },
            "processing_settings": {
                "vr_format": "over_under",
                "upscale_model": "x2",
                "temporal_window_overlap": 8,
                "processing_mode": "sequential",
                "batch_size": 50,
                "device": "cpu",
                "per_eye_width": 1080,
                "source_fps": 24.0,
                "depth_model_version": "v3",
                "depth_model_path": "large",
                "use_metric_depth": True,
            },
        }
        (tmp_path / "clip_1-settings.json").write_text(json.dumps(settings))
        return tmp_path

    def test_resumes_with_saved_settings_and_model(self, cli, tmp_path, monkeypatch, capsys):
        from unittest.mock import Mock

        projector = Mock()
        projector.process_video.return_value = True
        factory = Mock(return_value=projector)
        monkeypatch.setattr(cli, "create_stereo_projector", factory)
        monkeypatch.setattr(sys, "argv", ["prog", "--resume", str(self._batch(tmp_path))])

        assert cli.main() == 0

        factory.assert_called_once_with("large", "cpu", True, "v3")
        kwargs = projector.process_video.call_args[1]
        assert kwargs["video_path"] == "/videos/clip.mp4" and kwargs["output_dir"] == str(tmp_path)
        assert kwargs["vr_format"] == "over_under" and kwargs["upscale_model"] == "x2"
        assert kwargs["batch_size"] == 50 and kwargs["temporal_window_overlap"] == 8
        assert not {"per_eye_width", "source_fps", "device", "depth_model_path"} & set(kwargs)
        assert kwargs["progress_tracker"].batched is False
        projector.unload_model.assert_called_once()

    def test_interrupted_resume_explains_how_to_continue(self, cli, tmp_path, monkeypatch, capsys):
        from unittest.mock import Mock

        projector = Mock()
        projector.process_video.side_effect = KeyboardInterrupt
        monkeypatch.setattr(cli, "create_stereo_projector", Mock(return_value=projector))
        monkeypatch.setattr(sys, "argv", ["prog", "--resume", str(self._batch(tmp_path))])

        assert cli.main() == 1
        assert "--assemble-partial" in capsys.readouterr().out
        projector.unload_model.assert_called_once()

    def test_completed_batch_cannot_resume(self, cli, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            sys, "argv", ["prog", "--resume", str(self._batch(tmp_path, "completed"))]
        )
        assert cli.main() == 1


class TestQualityFlags:
    """Options the web UI had but the CLI didn't."""

    def test_defaults(self, cli):
        args = cli.create_argument_parser().parse_args(["video.mp4"])
        assert (args.depth_resolution, args.super_sample, args.video_encoder) == (
            "auto",
            "auto",
            "auto",
        )
        assert args.reuse is None

    def test_values(self, cli):
        args = cli.create_argument_parser().parse_args(
            [
                "video.mp4",
                "--depth-resolution",
                "720",
                "--super-sample",
                "2x",
                "--video-encoder",
                "libx265",
            ]
        )
        assert (args.depth_resolution, args.super_sample, args.video_encoder) == (
            "720",
            "2x",
            "libx265",
        )

    @pytest.mark.parametrize(
        "flag, value", [("--depth-resolution", "999"), ("--video-encoder", "av1")]
    )
    def test_rejects_unknown_values(self, cli, flag, value, capsys):
        with pytest.raises(SystemExit):
            cli.create_argument_parser().parse_args(["video.mp4", flag, value])

    def test_settings_reach_the_pipeline_and_reuse_sets_the_directory(
        self, cli, tmp_path, monkeypatch, capsys
    ):
        from unittest.mock import Mock

        video = tmp_path / "video.mp4"
        video.touch()
        projector = Mock()
        projector.process_video.return_value = True
        monkeypatch.setattr(cli, "create_stereo_projector", Mock(return_value=projector))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                str(video),
                "--reuse",
                str(tmp_path),
                "--depth-resolution",
                "720",
                "--super-sample",
                "none",
                "--video-encoder",
                "libx264",
                "--upscale-model",
                "x2",
            ],
        )

        assert cli.main() == 0

        kwargs = projector.process_video.call_args[1]
        assert kwargs["output_dir"] == str(tmp_path)
        assert (kwargs["depth_resolution"], kwargs["super_sample"]) == ("720", "none")
        assert (kwargs["video_encoder"], kwargs["upscale_model"]) == ("libx264", "x2")

    def test_reuse_needs_an_existing_directory(self, cli, tmp_path, monkeypatch, capsys):
        video = tmp_path / "video.mp4"
        video.touch()
        monkeypatch.setattr(sys, "argv", ["prog", str(video), "--reuse", str(tmp_path / "nope")])
        assert cli.main() == 1


class TestIntermediatesFlag:
    """Intermediate files are only kept on request."""

    def _settings(self, cli, tmp_path, monkeypatch, *flags):
        from unittest.mock import Mock

        video = tmp_path / "video.mp4"
        video.touch()
        projector = Mock()
        projector.process_video.return_value = True
        monkeypatch.setattr(cli, "create_stereo_projector", Mock(return_value=projector))
        monkeypatch.setattr(sys, "argv", ["prog", str(video), "-o", str(tmp_path), *flags])
        assert cli.main() == 0
        return projector.process_video.call_args[1]

    def test_off_by_default(self, cli, tmp_path, monkeypatch, capsys):
        assert self._settings(cli, tmp_path, monkeypatch)["keep_intermediates"] is False

    def test_keep_flag(self, cli, tmp_path, monkeypatch, capsys):
        kwargs = self._settings(cli, tmp_path, monkeypatch, "--keep-intermediates")
        assert kwargs["keep_intermediates"] is True

    def test_old_flag_is_still_accepted(self, cli, tmp_path, monkeypatch, capsys):
        kwargs = self._settings(cli, tmp_path, monkeypatch, "--no-intermediates")
        assert kwargs["keep_intermediates"] is False
