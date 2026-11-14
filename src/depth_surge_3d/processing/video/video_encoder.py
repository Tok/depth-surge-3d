"""
Video encoding module.

Handles FFmpeg-based video encoding with hardware acceleration support (NVENC).
"""

from __future__ import annotations

import subprocess
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...io.operations import (
    get_frame_files,
    verify_ffmpeg_installation,
)
from ...utils.path_utils import (
    calculate_frame_range,
    generate_output_filename,
    parse_time_string,
)
from ...core.constants import (
    INTERMEDIATE_DIRS,
    DEFAULT_FALLBACK_FPS,
    NVENC_CONSTANT_QUALITY,
)


def run_ffmpeg(cmd: list[str], on_frame: Callable[[int], None] | None = None) -> tuple[int, str]:
    """
    Run an FFmpeg command, reporting the number of frames written while it runs.

    Args:
        cmd: FFmpeg command; must start with the ffmpeg executable
        on_frame: Called with the frame count whenever FFmpeg reports progress. If it
            raises (e.g. a stop request), FFmpeg is killed and the exception propagates.

    Returns:
        Tuple of (return code, last lines of FFmpeg's error output)

    Side effects:
        - Executes FFmpeg subprocess
    """
    # Machine-readable progress on stdout; errors (only) share the pipe
    full_cmd = [cmd[0], "-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:1"]
    full_cmd.extend(cmd[1:])
    messages: deque[str] = deque(maxlen=40)

    process = subprocess.Popen(
        full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            key, _, value = line.partition("=")
            if key == "frame" and value.isdigit():
                if on_frame:
                    on_frame(int(value))
            elif line and key not in _FFMPEG_PROGRESS_KEYS:
                messages.append(line)
        return process.wait(), "\n".join(messages)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if process.stdout is not None:
            process.stdout.close()


_FFMPEG_PROGRESS_KEYS = {
    "fps",
    "stream_0_0_q",
    "bitrate",
    "total_size",
    "out_time_us",
    "out_time_ms",
    "out_time",
    "dup_frames",
    "drop_frames",
    "speed",
    "progress",
}


class VideoEncoder:
    """
    Handles video encoding using FFmpeg.

    Responsibilities:
    - Video creation with FFmpeg
    - Hardware encoder detection (NVENC)
    - Frame extraction from videos
    - Audio integration
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize video encoder.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self._nvenc_available: bool | None = None
        self._announced_encoder: str | None = None

    def encode_segment(
        self,
        vr_frames_dir: Path,
        segment_path: Path,
        start_index: int,
        num_frames: int,
        settings: dict[str, Any],
        on_frame: Callable[[int], None] | None = None,
    ) -> bool:
        """
        Encode a range of VR frames into a video segment without audio.

        Segments of consecutive ranges are joined by ``concat_segments`` without
        re-encoding, so the frames of a finished batch don't have to be kept as images.

        Args:
            vr_frames_dir: Directory containing VR frames
            segment_path: Output file of the segment
            start_index: Index of the first frame (0-based; frame files are numbered from 1)
            num_frames: Number of frames in the segment
            settings: Processing settings
            on_frame: Called with the number of frames encoded so far

        Returns:
            True if successful, False otherwise

        Side effects:
            - Executes FFmpeg subprocess
            - Writes the segment file
        """
        source_fps = self._source_fps(settings)
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{source_fps:g}",
            "-start_number",
            str(start_index + 1),
            "-i",
            str(vr_frames_dir / "frame_%06d.png"),
            "-frames:v",
            str(num_frames),
        ]
        cmd.extend(self._fps_output_args(settings, source_fps))
        encoder_args, _ = self._build_encoder_cmd(
            settings.get("video_encoder", "auto"), segment_path
        )
        cmd.extend(encoder_args)

        returncode, output = run_ffmpeg(cmd, on_frame)
        if returncode != 0:
            print(f"FFmpeg error: {output}")
            return False
        return True

    def concat_segments(
        self,
        segment_paths: list[Path],
        output_dir: Path,
        original_video: str,
        settings: dict[str, Any],
        num_frames: int,
        output_filename: str | None = None,
    ) -> bool:
        """
        Join video segments into the output video and add the matching part of the audio.

        The video stream is copied, not re-encoded.

        Args:
            segment_paths: Segment files in playback order
            output_dir: Output directory
            original_video: Source video path for audio extraction
            settings: Processing settings
            num_frames: Number of source frames the segments cover (for the audio length)
            output_filename: Output file name (default: generated from the settings)

        Returns:
            True if successful, False otherwise

        Side effects:
            - Executes FFmpeg subprocess
            - Writes a segment list file and the video file
        """
        if not segment_paths:
            print("Error: No video segments to join")
            return False
        if not output_filename:
            output_filename = generate_output_filename(
                Path(original_video).name, settings["vr_format"], settings["vr_resolution"]
            )

        list_file = segment_paths[0].parent / "segments.txt"
        lines = [
            f"file '{str(path.resolve()).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'"
            for path in segment_paths
        ]
        list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file)]
        audio_args = self._audio_input_args(output_dir, original_video, settings, num_frames)
        cmd.extend(audio_args)
        cmd.extend(["-map", "0:v:0", "-c:v", "copy"])
        if audio_args:
            cmd.extend(["-map", "1:a:0?", "-c:a", "aac"])
        cmd.append(str(output_dir / output_filename))

        returncode, output = run_ffmpeg(cmd)
        if returncode != 0:
            print(f"FFmpeg error: {output}")
            return False
        return True

    def _audio_input_args(
        self, output_dir: Path, original_video: str, settings: dict[str, Any], num_frames: int
    ) -> list[str]:
        """
        FFmpeg input options for the part of the audio that belongs to the frames.

        Returns:
            Input options, or an empty list if audio isn't preserved

        Side effects:
            - Checks for the pre-extracted audio file
        """
        if not settings.get("preserve_audio", True):
            return []
        source_fps = self._source_fps(settings)
        audio_file = output_dir / "original_audio.flac"
        audio_source = str(audio_file) if audio_file.exists() else original_video
        return [
            "-ss",
            f"{self._range_start_seconds(settings, source_fps):.6f}",
            "-t",
            f"{num_frames / source_fps:.6f}",
            "-i",
            audio_source,
        ]

    def create_video(
        self,
        vr_frames_dir: Path,
        output_dir: Path,
        original_video: str,
        settings: dict[str, Any],
        progress_tracker=None,
        max_frames: int | None = None,
        output_filename: str | None = None,
    ) -> bool:
        """
        Create the output video from VR frames, with the matching part of the audio.

        Frames are read at the source frame rate (they were extracted one to one), so the
        video keeps its speed; a different target FPS is produced by ffmpeg on output.

        Args:
            vr_frames_dir: Directory containing VR frames
            output_dir: Output directory
            original_video: Source video path for audio extraction
            settings: Processing settings
            progress_tracker: Optional progress tracker
            max_frames: Only use the first N frames (partial video of an unfinished run)
            output_filename: Output file name (default: generated from the settings)

        Returns:
            True if successful, False otherwise

        Side effects:
            - Executes FFmpeg subprocess
            - Writes video file to disk
        """
        if not verify_ffmpeg_installation():
            print("Error: FFmpeg not found. Cannot create output video.")
            return False

        available = len(list(vr_frames_dir.glob("frame_*.png")))
        num_frames = min(available, max_frames) if max_frames else available
        if num_frames <= 0:
            print(f"Error: No VR frames found in {vr_frames_dir}")
            return False

        if not output_filename:
            output_filename = generate_output_filename(
                Path(original_video).name, settings["vr_format"], settings["vr_resolution"]
            )
        output_path = output_dir / output_filename

        source_fps = self._source_fps(settings)
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{source_fps:g}",
            "-i",
            str(vr_frames_dir / "frame_%06d.png"),
        ]

        # Audio: the part of the source that belongs to these frames
        audio_args = self._audio_input_args(output_dir, original_video, settings, num_frames)
        cmd.extend(audio_args)
        has_audio_input = bool(audio_args)

        # Explicit stream choice: without it ffmpeg may pick the source's video stream
        cmd.extend(["-map", "0:v:0"])
        if has_audio_input:
            cmd.extend(["-map", "1:a:0?", "-c:a", "aac"])
        cmd.extend(["-frames:v", str(num_frames)])
        cmd.extend(self._fps_output_args(settings, source_fps))

        encoder = settings.get("video_encoder", "auto")
        encoder_args, _ = self._build_encoder_cmd(encoder, output_path)
        cmd.extend(encoder_args)

        def report(frame: int) -> None:
            if progress_tracker:
                progress_tracker.update_progress(
                    f"Encoding video {min(frame, num_frames)}/{num_frames}",
                    phase="video_creation",
                    frame_num=min(frame, num_frames),
                    step_name="Video Creation",
                    step_progress=min(frame, num_frames),
                    step_total=num_frames,
                )

        report(0)
        returncode, output = run_ffmpeg(cmd, report)
        if returncode != 0:
            print(f"FFmpeg error: {output}")
            return False
        report(num_frames)
        return True

    @staticmethod
    def _source_fps(settings: dict[str, Any]) -> float:
        """PURE: Frame rate of the extracted frames."""
        try:
            fps = float(settings.get("source_fps") or 0)
        except (TypeError, ValueError):
            fps = 0.0
        return fps if fps > 0 else float(DEFAULT_FALLBACK_FPS)

    @staticmethod
    def _range_start_seconds(settings: dict[str, Any], source_fps: float) -> float:
        """PURE: Time of the first extracted frame in the source video."""
        start_time = settings.get("start_time")
        start_seconds = parse_time_string(str(start_time)) if start_time else None
        if not start_seconds:
            return 0.0
        return int(start_seconds * source_fps) / source_fps

    @staticmethod
    def _fps_output_args(settings: dict[str, Any], source_fps: float) -> list[str]:
        """
        PURE: FFmpeg output options for the requested frame rate.

        Nothing for "keep original". Otherwise frames are duplicated/dropped to the target
        rate, or motion-interpolated when the experimental interpolation is enabled (to
        the target rate, or to double the source rate if none is set).
        """
        target = settings.get("target_fps")
        try:
            target_fps = 0.0 if target in (None, "None", "original") else float(str(target))
        except (TypeError, ValueError):
            target_fps = 0.0

        if settings.get("experimental_frame_interpolation"):
            fps = target_fps if target_fps > source_fps else source_fps * 2
            return ["-vf", f"minterpolate=fps={fps:g}:mi_mode=mci"]
        if target_fps > 0 and abs(target_fps - source_fps) > 0.01:
            return ["-r", f"{target_fps:g}"]
        return []

    def extract_frames(
        self,
        video_path: str,
        directories: dict[str, Path],
        video_properties: dict[str, Any],
        settings: dict[str, Any],
        progress_tracker=None,
    ) -> list[Path]:
        """
        Extract frames from video using FFmpeg.

        Args:
            video_path: Input video path
            directories: Dictionary of processing directories
            video_properties: Video metadata (frame_count, fps)
            settings: Processing settings with frame range info
            progress_tracker: Optional progress tracker

        Returns:
            List of extracted frame file paths

        Side effects:
            - Executes FFmpeg subprocess
            - Deletes frames of an earlier extraction, writes frame images to disk
        """
        frames_dir = directories.get("frames")
        if not frames_dir:
            frames_dir = directories["base"] / INTERMEDIATE_DIRS["frames"]
            frames_dir.mkdir(exist_ok=True)
        for stale in frames_dir.glob("frame_*.png"):
            stale.unlink()

        # Calculate frame range and timing
        total_frames = video_properties["frame_count"]
        fps = video_properties["fps"]
        start_frame, end_frame = calculate_frame_range(
            total_frames, fps, settings.get("start_time"), settings.get("end_time")
        )
        expected_frames = end_frame - start_frame

        # Convert frame numbers to timestamps for more efficient seeking
        start_time = start_frame / fps if fps > 0 else 0
        duration = expected_frames / fps if fps > 0 else 0
        output_pattern = str(frames_dir / "frame_%06d.png")

        def report(frame: int) -> None:
            if progress_tracker:
                progress_tracker.update_progress(
                    f"Extracting frames {min(frame, expected_frames)}/{expected_frames}",
                    phase="extraction",
                    frame_num=min(frame, expected_frames),
                    step_name="Frame Extraction",
                    step_progress=min(frame, expected_frames),
                    step_total=expected_frames,
                )

        # Seek before decoding (much faster) and limit by duration
        seek = ["-ss", str(start_time), "-i", video_path, "-t", str(duration)]
        cmd_cuda = [
            "ffmpeg",
            "-y",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            *seek,
            "-vf",
            "hwdownload,format=nv12,format=rgb24",
            "-pix_fmt",
            "rgb24",
            "-vsync",
            "0",  # Pass through original timestamps
            output_pattern,
        ]
        cmd_cpu = ["ffmpeg", "-y", *seek, "-pix_fmt", "rgb24", "-vsync", "0", output_pattern]

        report(0)
        returncode, _ = run_ffmpeg(cmd_cuda, report)
        if returncode != 0:
            print("  CUDA frame extraction failed, falling back to CPU")
            returncode, output = run_ffmpeg(cmd_cpu, report)
            if returncode != 0:
                print(f"FFmpeg error: {output}")
                return []

        frame_files = get_frame_files(frames_dir)
        if frame_files and progress_tracker:
            progress_tracker.update_progress(
                f"Extracted {len(frame_files)} frames",
                phase="extraction",
                frame_num=len(frame_files),
                step_name="Frame Extraction",
                step_progress=len(frame_files),
                step_total=len(frame_files),
            )
        return frame_files

    def _announce(self, message: str) -> None:
        """Print the encoder choice once, not for every segment."""
        if self._announced_encoder != message:
            self._announced_encoder = message
            print(message)

    def _check_nvenc_available(self) -> bool:
        """
        Check if NVIDIA NVENC hardware encoder is available.

        Returns:
            True if NVENC is available, False otherwise

        Side effects:
            - Executes FFmpeg subprocess for encoder detection
        """
        if self._nvenc_available is None:
            test_result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
            )
            self._nvenc_available = "hevc_nvenc" in test_result.stdout
        return self._nvenc_available

    def _build_encoder_cmd(self, encoder: str, output_path: Path) -> tuple[list[str], bool]:
        """
        Build FFmpeg encoder arguments.

        Args:
            encoder: Encoder name (auto, nvenc, libx264, libx265)
            output_path: Output video file path

        Returns:
            Tuple of (encoder_args, is_nvenc_used)

        Side effects:
            - May print console output for encoder selection
            - Checks NVENC availability
        """
        # Try NVENC for auto or explicit nvenc
        if encoder in ["auto", "nvenc"]:
            if self._check_nvenc_available():
                self._announce("  Using NVENC hardware encoding (H.265)")
                return (
                    [
                        "-c:v",
                        "hevc_nvenc",
                        "-pix_fmt",
                        "yuv420p",
                        "-preset",
                        "p7",
                        "-tune",
                        "hq",
                        # Constant quality; without it NVENC uses a low default bitrate
                        "-rc",
                        "vbr",
                        "-cq",
                        str(NVENC_CONSTANT_QUALITY),
                        "-b:v",
                        "0",
                        str(output_path),
                    ],
                    True,
                )
            elif encoder == "nvenc":
                self._announce("  Warning: NVENC not available, falling back to software encoding")
                self._announced_encoder = None
            # Fall through to software encoding

        # Software encoding (default or explicit)
        if encoder in ["libx264", "libx265"]:
            codec = encoder
        else:
            # Unknown encoder or auto fallback
            codec = "libx264"
            if encoder not in ["auto", "nvenc"]:
                print(f"  Warning: Unknown encoder '{encoder}', using libx264")

        self._announce(f"  Using software encoding ({codec})")
        return (
            [
                "-c:v",
                codec,
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",  # High quality
                "-preset",
                "medium",
                str(output_path),
            ],
            False,
        )
