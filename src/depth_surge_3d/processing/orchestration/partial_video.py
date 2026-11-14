"""
Video from the finished part of an unfinished run.

When a run is interrupted, the VR frames finished so far can still be turned into a
video (with the matching part of the audio), without resuming the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...core.constants import INTERMEDIATE_DIRS
from ...io.operations import find_settings_file, load_processing_settings, read_pipeline_progress
from ...utils.path_utils import generate_output_filename
from ..video.video_encoder import VideoEncoder


def count_finished_vr_frames(output_dir: Path) -> int:
    """
    Number of VR frames, counted from the start of the video, that are finished.

    Uses the pipeline's resume state; for directories without one (older versions) it
    counts the frames present without a gap.

    Args:
        output_dir: Batch output directory

    Returns:
        Number of finished VR frames

    Side effects:
        - Reads the state file / the VR frames directory
    """
    vr_dir = output_dir / INTERMEDIATE_DIRS["vr_frames"]
    progress = read_pipeline_progress(output_dir)
    if progress is not None:
        # With segment encoding, a frame is finished once its segment exists (its VR
        # image may already be deleted)
        if progress["segments"]:
            return int(progress["done"].get("encode", 0))
        return int(progress["done"].get("assemble", 0))

    count = 0
    while (vr_dir / f"frame_{count + 1:06d}.png").exists():
        count += 1
    return count


def assemble_partial_video(output_dir: Path | str, progress_tracker: Any = None) -> Path | None:
    """
    Encode the VR frames finished so far into a video.

    Args:
        output_dir: Batch output directory of an interrupted (or running) job
        progress_tracker: Optional progress tracker

    Returns:
        Path of the created video, or None if there is nothing to encode or encoding failed

    Side effects:
        - Reads the batch's settings and state, runs FFmpeg, writes the video
    """
    output_path = Path(output_dir)
    settings_file = find_settings_file(output_path)
    settings_data = load_processing_settings(settings_file) if settings_file else None
    if not settings_data:
        print(f"Error: No settings file found in {output_path}")
        return None

    settings = dict(settings_data.get("processing_settings", {}))
    source_video = str(settings_data.get("metadata", {}).get("source_video", ""))
    num_frames = count_finished_vr_frames(output_path)
    if num_frames <= 0:
        print("Error: No finished VR frames to assemble yet")
        return None

    has_audio_source = (output_path / "original_audio.flac").exists() or Path(source_video).exists()
    if not has_audio_source:
        settings["preserve_audio"] = False

    full_name = generate_output_filename(
        Path(source_video).name or output_path.name,
        settings.get("vr_format", "side_by_side"),
        settings.get("vr_resolution"),
    )
    filename = f"{Path(full_name).stem}_partial_{num_frames}frames.mp4"
    encoder = VideoEncoder()
    progress = read_pipeline_progress(output_path)
    if progress and progress["segments"]:
        # Encoded batch by batch already: join the finished segments
        segments_dir = output_path / INTERMEDIATE_DIRS["video_segments"]
        segments = [
            segments_dir / f"segment_{start + 1:06d}-{end:06d}.mp4"
            for start, end in progress["segments"]
            if end <= num_frames
        ]
        created = encoder.concat_segments(
            segments, output_path, source_video, settings, num_frames, output_filename=filename
        )
    else:
        created = encoder.create_video(
            output_path / INTERMEDIATE_DIRS["vr_frames"],
            output_path,
            source_video,
            settings,
            progress_tracker,
            max_frames=num_frames,
            output_filename=filename,
        )
    if not created:
        return None
    return output_path / filename
