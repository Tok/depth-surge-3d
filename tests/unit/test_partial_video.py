"""Tests for assembling a video from an unfinished run."""

import json
from unittest.mock import patch

from src.depth_surge_3d.processing.orchestration.partial_video import (
    assemble_partial_video,
    count_finished_vr_frames,
)

MODULE = "src.depth_surge_3d.processing.orchestration.partial_video"


def _batch(tmp_path, vr_frames=6, assembled=None, with_settings=True, source_exists=True):
    vr_dir = tmp_path / "99_vr_frames"
    vr_dir.mkdir()
    for i in range(vr_frames):
        (vr_dir / f"frame_{i + 1:06d}.png").touch()
    source = tmp_path / "clip.mp4"
    if source_exists:
        source.touch()
    if assembled is not None:
        (tmp_path / "pipeline_state.json").write_text(
            json.dumps({"total_frames": 10, "done": {"assemble": assembled}})
        )
    if with_settings:
        (tmp_path / "clip_1-settings.json").write_text(
            json.dumps(
                {
                    "metadata": {"source_video": str(source)},
                    "processing_settings": {
                        "vr_format": "over_under",
                        "vr_resolution": "square-1k",
                        "preserve_audio": True,
                    },
                }
            )
        )
    return tmp_path


class TestCountFinishedFrames:
    def test_uses_pipeline_state(self, tmp_path):
        """Frames of a batch that was interrupted mid-way don't count."""
        assert count_finished_vr_frames(_batch(tmp_path, vr_frames=6, assembled=4)) == 4

    def test_counts_gapless_frames_without_state(self, tmp_path):
        batch = _batch(tmp_path, vr_frames=6)
        (batch / "99_vr_frames" / "frame_000004.png").unlink()
        assert count_finished_vr_frames(batch) == 3


class TestAssemblePartialVideo:
    def test_encodes_finished_frames(self, tmp_path, capsys):
        batch = _batch(tmp_path, assembled=4)
        with patch(f"{MODULE}.VideoEncoder") as encoder:
            encoder.return_value.create_video.return_value = True
            output = assemble_partial_video(batch)

        assert output == batch / "clip_3D_over-under_square-1k_partial_4frames.mp4"
        args, kwargs = encoder.return_value.create_video.call_args
        assert args[0] == batch / "99_vr_frames" and args[2] == str(batch / "clip.mp4")
        assert kwargs["max_frames"] == 4 and kwargs["output_filename"] == output.name
        assert args[3]["preserve_audio"] is True

    def test_without_audio_source(self, tmp_path, capsys):
        batch = _batch(tmp_path, assembled=4, source_exists=False)
        with patch(f"{MODULE}.VideoEncoder") as encoder:
            encoder.return_value.create_video.return_value = True
            assert assemble_partial_video(batch) is not None
        assert encoder.return_value.create_video.call_args[0][3]["preserve_audio"] is False

    def test_nothing_to_assemble(self, tmp_path, capsys):
        assert assemble_partial_video(_batch(tmp_path, assembled=0)) is None
        assert "No finished VR frames" in capsys.readouterr().out

    def test_no_settings_file(self, tmp_path, capsys):
        assert assemble_partial_video(_batch(tmp_path, with_settings=False)) is None

    def test_encoding_failure(self, tmp_path, capsys):
        batch = _batch(tmp_path, assembled=4)
        with patch(f"{MODULE}.VideoEncoder") as encoder:
            encoder.return_value.create_video.return_value = False
            assert assemble_partial_video(batch) is None


class TestWithVideoSegments:
    """Runs that encode per batch: the finished segments are joined."""

    def _batch(self, tmp_path):
        batch = _batch(tmp_path, vr_frames=2, assembled=None)
        (batch / "pipeline_state.json").write_text(
            json.dumps(
                {
                    "total_frames": 300,
                    "done": {"assemble": 250, "encode": 200},
                    "segments": [[0, 100], [100, 200]],
                }
            )
        )
        return batch

    def test_finished_frames_are_the_encoded_ones(self, tmp_path):
        """VR frame images may be deleted already; the segments are what counts."""
        assert count_finished_vr_frames(self._batch(tmp_path)) == 200

    def test_joins_segments_instead_of_encoding_frames(self, tmp_path, capsys):
        batch = self._batch(tmp_path)
        with patch(f"{MODULE}.VideoEncoder") as encoder:
            encoder.return_value.concat_segments.return_value = True
            output = assemble_partial_video(batch)

        assert output.name == "clip_3D_over-under_square-1k_partial_200frames.mp4"
        args, kwargs = encoder.return_value.concat_segments.call_args
        assert [path.name for path in args[0]] == [
            "segment_000001-000100.mp4",
            "segment_000101-000200.mp4",
        ]
        assert args[4] == 200 and kwargs["output_filename"] == output.name
        encoder.return_value.create_video.assert_not_called()
