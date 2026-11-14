"""Tests for the FFmpeg-based video encoder."""

from unittest.mock import patch

import pytest

from src.depth_surge_3d.processing.video.video_encoder import VideoEncoder, run_ffmpeg

MODULE = "src.depth_surge_3d.processing.video.video_encoder"


class _Process:
    """Stand-in for subprocess.Popen producing FFmpeg's -progress output."""

    def __init__(self, lines, returncode=0):
        self.stdout = _Lines(lines)
        self._returncode = returncode
        self.finished = False
        self.killed = False

    def wait(self):
        self.finished = True
        return self._returncode

    def poll(self):
        return self._returncode if self.finished else None

    def kill(self):
        self.killed = True


class _Lines(list):
    closed = False

    def close(self):
        self.closed = True


class TestRunFfmpeg:
    def test_reports_frames_and_adds_progress_options(self):
        process = _Process(["frame=5\n", "fps=30\n", "progress=continue\n", "frame=9\n"])
        frames = []
        with patch(f"{MODULE}.subprocess.Popen", return_value=process) as popen:
            returncode, output = run_ffmpeg(
                ["ffmpeg", "-y", "-i", "in.mp4", "out.mp4"], frames.append
            )

        assert (returncode, output, frames) == (0, "", [5, 9])
        command = popen.call_args[0][0]
        assert command[0] == "ffmpeg" and command[-4:] == ["-y", "-i", "in.mp4", "out.mp4"]
        assert command[command.index("-progress") + 1] == "pipe:1"
        assert process.stdout.closed and not process.killed

    def test_returns_error_output(self):
        process = _Process(["frame=1\n", "No such file or directory\n"], returncode=1)
        with patch(f"{MODULE}.subprocess.Popen", return_value=process):
            assert run_ffmpeg(["ffmpeg"]) == (1, "No such file or directory")

    def test_stop_request_kills_ffmpeg(self):
        process = _Process(["frame=1\n", "frame=2\n"])

        def stop(frame):
            raise InterruptedError("stop")

        with patch(f"{MODULE}.subprocess.Popen", return_value=process):
            with pytest.raises(InterruptedError):
                run_ffmpeg(["ffmpeg"], stop)
        assert process.killed


SETTINGS = {
    "vr_format": "side_by_side",
    "vr_resolution": "square-480",
    "source_fps": 24.0,
    "preserve_audio": True,
    "video_encoder": "libx264",
}


class TestCreateVideo:
    @pytest.fixture
    def vr_dir(self, tmp_path):
        directory = tmp_path / "99_vr_frames"
        directory.mkdir()
        for i in range(48):
            (directory / f"frame_{i + 1:06d}.png").touch()
        return directory

    def _create(self, vr_dir, tmp_path, tracker=None, returncode=0, **kwargs):
        settings = {**SETTINGS, **kwargs.pop("settings", {})}
        with (
            patch(f"{MODULE}.verify_ffmpeg_installation", return_value=True),
            patch(f"{MODULE}.run_ffmpeg", return_value=(returncode, "boom")) as run,
        ):
            result = VideoEncoder().create_video(
                vr_dir, tmp_path, "/videos/clip.mp4", settings, tracker, **kwargs
            )
        return result, (run.call_args[0][0] if run.called else None)

    def test_uses_source_frame_rate_and_matching_audio(self, vr_dir, tmp_path):
        """Regression: videos were encoded at 30 fps whatever the source frame rate."""
        settings = {"start_time": "00:02"}
        result, cmd = self._create(vr_dir, tmp_path, settings=settings)

        assert result is True
        assert cmd[cmd.index("-framerate") + 1] == "24"
        # audio: starts at the selected range and lasts as long as the frames (48 / 24 fps)
        audio = cmd.index("/videos/clip.mp4")
        assert cmd[audio - 5 : audio] == ["-ss", "2.000000", "-t", "2.000000", "-i"]
        assert cmd[cmd.index("-map") : cmd.index("-map") + 4] == ["-map", "0:v:0", "-map", "1:a:0?"]
        assert cmd[cmd.index("-frames:v") + 1] == "48"
        assert "-r" not in cmd
        assert cmd[-1].endswith("clip_3D_side-by-side_square-480.mp4")

    def test_prefers_extracted_audio_file(self, vr_dir, tmp_path):
        (tmp_path / "original_audio.flac").touch()
        _, cmd = self._create(vr_dir, tmp_path)
        assert str(tmp_path / "original_audio.flac") in cmd

    def test_without_audio(self, vr_dir, tmp_path):
        _, cmd = self._create(vr_dir, tmp_path, settings={"preserve_audio": False})
        assert cmd.count("-i") == 1 and "1:a:0?" not in cmd

    def test_partial_video(self, vr_dir, tmp_path):
        _, cmd = self._create(vr_dir, tmp_path, max_frames=12, output_filename="partial.mp4")
        assert cmd[cmd.index("-frames:v") + 1] == "12"
        assert cmd[cmd.index("-t") + 1] == "0.500000"
        assert cmd[-1] == str(tmp_path / "partial.mp4")

    def test_progress_is_reported(self, vr_dir, tmp_path):
        updates = []

        class Tracker:
            def update_progress(self, stage, **kwargs):
                updates.append((kwargs["step_name"], kwargs["step_progress"], kwargs["step_total"]))

        self._create(vr_dir, tmp_path, Tracker())
        assert updates == [("Video Creation", 0, 48), ("Video Creation", 48, 48)]

    def test_failures(self, vr_dir, tmp_path, capsys):
        assert self._create(vr_dir, tmp_path, returncode=1)[0] is False
        assert "boom" in capsys.readouterr().out
        empty = tmp_path / "empty"
        empty.mkdir()
        assert self._create(empty, tmp_path) == (False, None)
        with patch(f"{MODULE}.verify_ffmpeg_installation", return_value=False):
            assert VideoEncoder().create_video(vr_dir, tmp_path, "clip.mp4", SETTINGS) is False


class TestFrameRateOptions:
    @pytest.mark.parametrize(
        "settings, expected",
        [
            ({}, []),
            ({"target_fps": None}, []),
            ({"target_fps": "original"}, []),
            ({"target_fps": 24}, []),
            ({"target_fps": 60}, ["-r", "60"]),
            ({"target_fps": "bad"}, []),
            (
                {"experimental_frame_interpolation": True},
                ["-vf", "minterpolate=fps=48:mi_mode=mci"],
            ),
            (
                {"experimental_frame_interpolation": True, "target_fps": 60},
                ["-vf", "minterpolate=fps=60:mi_mode=mci"],
            ),
        ],
    )
    def test_fps_output_args(self, settings, expected):
        assert VideoEncoder._fps_output_args(settings, 24.0) == expected

    @pytest.mark.parametrize(
        "value, expected", [(24.0, 24.0), ("25", 25.0), (None, 30.0), ("x", 30.0), (0, 30.0)]
    )
    def test_source_fps(self, value, expected):
        assert VideoEncoder._source_fps({"source_fps": value}) == expected

    def test_range_start_is_snapped_to_a_frame(self):
        assert VideoEncoder._range_start_seconds({}, 24.0) == 0.0
        assert VideoEncoder._range_start_seconds({"start_time": "01:00"}, 24.0) == 60.0
        assert VideoEncoder._range_start_seconds({"start_time": "bad"}, 24.0) == 0.0


class TestExtractFrames:
    def _extract(self, tmp_path, results, tracker=None):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        (frames_dir / "frame_000099.png").touch()  # left over from an earlier, longer extraction
        calls = []

        def fake_run(cmd, on_frame=None):
            calls.append(cmd)
            returncode = results[len(calls) - 1]
            if returncode == 0:
                for i in range(3):
                    (frames_dir / f"frame_{i + 1:06d}.png").touch()
                on_frame(3)
            return returncode, "error text"

        with patch(f"{MODULE}.run_ffmpeg", side_effect=fake_run):
            files = VideoEncoder().extract_frames(
                "clip.mp4",
                {"frames": frames_dir, "base": tmp_path},
                {"frame_count": 240, "fps": 24.0},
                {"start_time": "00:01", "end_time": "00:02"},
                tracker,
            )
        return files, calls

    def test_extracts_selected_range_and_clears_old_frames(self, tmp_path):
        files, calls = self._extract(tmp_path, [0])

        assert [f.name for f in files] == [
            "frame_000001.png",
            "frame_000002.png",
            "frame_000003.png",
        ]
        assert len(calls) == 1 and "cuda" in calls[0]
        assert calls[0][calls[0].index("-ss") + 1] == "1.0"
        assert calls[0][calls[0].index("-t") + 1] == "1.0"

    def test_falls_back_to_cpu(self, tmp_path, capsys):
        files, calls = self._extract(tmp_path, [1, 0])
        assert len(files) == 3 and "cuda" not in calls[1]

    def test_failure(self, tmp_path, capsys):
        files, _ = self._extract(tmp_path, [1, 1])
        assert files == []
        assert "error text" in capsys.readouterr().out

    def test_progress(self, tmp_path):
        updates = []

        class Tracker:
            def update_progress(self, stage, **kwargs):
                updates.append((kwargs["step_progress"], kwargs["step_total"]))

        self._extract(tmp_path, [0], Tracker())
        assert updates[0] == (0, 24) and updates[-1] == (3, 3)

    def test_creates_frames_dir_when_missing(self, tmp_path):
        with patch(f"{MODULE}.run_ffmpeg", return_value=(1, "")):
            VideoEncoder().extract_frames(
                "clip.mp4", {"base": tmp_path}, {"frame_count": 10, "fps": 10.0}, {}
            )
        assert (tmp_path / "00_original_frames").is_dir()


class TestEncoderSelection:
    def test_nvenc_when_available(self, tmp_path, capsys):
        encoder = VideoEncoder()
        with patch.object(encoder, "_check_nvenc_available", return_value=True):
            args, nvenc = encoder._build_encoder_cmd("auto", tmp_path / "out.mp4")
        assert nvenc and "hevc_nvenc" in args

    @pytest.mark.parametrize(
        "name, codec", [("nvenc", "libx264"), ("libx265", "libx265"), ("weird", "libx264")]
    )
    def test_software_fallbacks(self, tmp_path, capsys, name, codec):
        encoder = VideoEncoder()
        with patch.object(encoder, "_check_nvenc_available", return_value=False):
            args, nvenc = encoder._build_encoder_cmd(name, tmp_path / "out.mp4")
        assert not nvenc and args[1] == codec

    def test_nvenc_detection(self):
        with patch(f"{MODULE}.subprocess.run") as run:
            run.return_value.stdout = "V..... hevc_nvenc"
            assert VideoEncoder()._check_nvenc_available() is True


class TestSegments:
    """The video is encoded per batch and joined without re-encoding."""

    def test_encode_segment_reads_its_frame_range(self, tmp_path):
        frames = []
        with patch(f"{MODULE}.run_ffmpeg", return_value=(0, "")) as run:
            result = VideoEncoder().encode_segment(
                tmp_path / "vr", tmp_path / "segment.mp4", 100, 50, SETTINGS, frames.append
            )

        cmd = run.call_args[0][0]
        assert result is True
        assert cmd[cmd.index("-start_number") + 1] == "101"  # frame files are numbered from 1
        assert cmd[cmd.index("-frames:v") + 1] == "50"
        assert cmd[cmd.index("-framerate") + 1] == "24"
        assert cmd.count("-i") == 1 and cmd[-1] == str(tmp_path / "segment.mp4")
        assert run.call_args[0][1] == frames.append

    def test_encode_segment_failure(self, tmp_path, capsys):
        with patch(f"{MODULE}.run_ffmpeg", return_value=(1, "encoder broke")):
            assert not VideoEncoder().encode_segment(tmp_path, tmp_path / "s.mp4", 0, 5, SETTINGS)
        assert "encoder broke" in capsys.readouterr().out

    def test_concat_copies_video_and_adds_matching_audio(self, tmp_path):
        segments = [tmp_path / "it's a.mp4", tmp_path / "b.mp4"]
        settings = {**SETTINGS, "start_time": "00:02"}
        with patch(f"{MODULE}.run_ffmpeg", return_value=(0, "")) as run:
            result = VideoEncoder().concat_segments(
                segments, tmp_path, "/videos/clip.mp4", settings, 48
            )

        cmd = run.call_args[0][0]
        assert result is True
        assert cmd[cmd.index("-f") + 1] == "concat"
        assert cmd[cmd.index("-c:v") + 1] == "copy"
        audio = cmd.index("/videos/clip.mp4")
        assert cmd[audio - 5 : audio] == ["-ss", "2.000000", "-t", "2.000000", "-i"]
        assert "1:a:0?" in cmd and cmd[-1].endswith("clip_3D_side-by-side_square-480.mp4")
        listed = (tmp_path / "segments.txt").read_text().splitlines()
        assert listed[1] == f"file '{tmp_path / 'b.mp4'}'"
        assert "'\\''" in listed[0]  # quotes in paths are escaped

    def test_concat_without_audio_and_custom_name(self, tmp_path):
        settings = {**SETTINGS, "preserve_audio": False}
        with patch(f"{MODULE}.run_ffmpeg", return_value=(0, "")) as run:
            VideoEncoder().concat_segments(
                [tmp_path / "a.mp4"], tmp_path, "clip.mp4", settings, 10, output_filename="part.mp4"
            )
        cmd = run.call_args[0][0]
        assert cmd.count("-i") == 1 and "-c:a" not in cmd and cmd[-1] == str(tmp_path / "part.mp4")

    def test_concat_failures(self, tmp_path, capsys):
        assert VideoEncoder().concat_segments([], tmp_path, "clip.mp4", SETTINGS, 10) is False
        with patch(f"{MODULE}.run_ffmpeg", return_value=(1, "bad segment")):
            assert not VideoEncoder().concat_segments(
                [tmp_path / "a.mp4"], tmp_path, "clip.mp4", SETTINGS, 10
            )
        assert "bad segment" in capsys.readouterr().out

    def test_encoder_choice_is_checked_and_announced_once(self, tmp_path, capsys):
        encoder = VideoEncoder()
        with patch(f"{MODULE}.subprocess.run") as run:
            run.return_value.stdout = "hevc_nvenc"
            for _ in range(3):
                encoder._build_encoder_cmd("auto", tmp_path / "out.mp4")
        assert run.call_count == 1
        assert capsys.readouterr().out.count("NVENC") == 1
