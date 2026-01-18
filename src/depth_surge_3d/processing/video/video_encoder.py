"""
Video encoding module.

Handles FFmpeg-based video encoding with hardware acceleration support (NVENC).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ...io.operations import (
    get_frame_files,
    verify_ffmpeg_installation,
)
from ...utils.path_utils import (
    calculate_frame_range,
    generate_output_filename,
)
from ...core.constants import (
    INTERMEDIATE_DIRS,
    DEFAULT_FALLBACK_FPS,
)


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

    def create_video(
        self,
        vr_frames_dir: Path,
        output_dir: Path,
        original_video: str,
        settings: dict[str, Any],
    ) -> bool:
        """
        Create final video with audio and encoding.

        Args:
            vr_frames_dir: Directory containing VR frames
            output_dir: Output directory
            original_video: Source video path for audio extraction
            settings: Processing settings

        Returns:
            True if successful, False otherwise

        Side effects:
            - Executes FFmpeg subprocess
            - Writes video file to disk
        """
        if not verify_ffmpeg_installation():
            print("Error: FFmpeg not found. Cannot create output video.")
            return False

        # Generate output filename
        output_filename = generate_output_filename(
            Path(original_video).name, settings["vr_format"], settings["vr_resolution"]
        )
        output_path = output_dir / output_filename

        # Build base FFmpeg command
        base_fps = settings.get("target_fps", DEFAULT_FALLBACK_FPS)
        if base_fps is None or str(base_fps) == "None" or base_fps == "original":
            base_fps = 30

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(base_fps),
            "-i",
            str(vr_frames_dir / "frame_%06d.png"),
        ]

        # Add audio if preserving - use pre-extracted FLAC file
        if settings.get("preserve_audio", True):
            audio_file = output_dir / "original_audio.flac"
            print(f"Looking for pre-extracted audio at: {audio_file}")
            if audio_file.exists():
                print(f"Using pre-extracted audio: {audio_file}")
                cmd.extend(["-i", str(audio_file), "-c:a", "aac", "-shortest"])
            else:
                print(f"Warning: Pre-extracted audio not found at {audio_file}")
                print(f"Extracting audio from original video: {original_video}")
                cmd.extend(["-i", original_video, "-c:a", "aac", "-shortest"])

        # Add video encoding settings
        encoder = settings.get("video_encoder", "auto")
        encoder_args, _ = self._build_encoder_cmd(encoder, output_path)
        cmd.extend(encoder_args)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return False
            return True
        except Exception as e:
            print(f"Error creating output video: {e}")
            return False

    def extract_frames(
        self,
        video_path: str,
        directories: dict[str, Path],
        video_properties: dict[str, Any],
        settings: dict[str, Any],
    ) -> list[Path]:
        """
        Extract frames from video using FFmpeg.

        Args:
            video_path: Input video path
            directories: Dictionary of processing directories
            video_properties: Video metadata (frame_count, fps)
            settings: Processing settings with frame range info

        Returns:
            List of extracted frame file paths

        Side effects:
            - Executes FFmpeg subprocess
            - Writes frame images to disk
        """
        frames_dir = directories.get("frames")
        if not frames_dir:
            frames_dir = directories["base"] / INTERMEDIATE_DIRS["frames"]
            frames_dir.mkdir(exist_ok=True)

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

        # Try CUDA hardware decoding with optimized seeking
        cmd_cuda = [
            "ffmpeg",
            "-y",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-ss",
            str(start_time),  # Seek before decoding (much faster)
            "-i",
            video_path,
            "-t",
            str(duration),  # Duration limit (more efficient than select filter)
            "-vf",
            "hwdownload,format=nv12,format=rgb24",
            "-pix_fmt",
            "rgb24",
            "-vsync",
            "0",  # Pass through original timestamps
            str(frames_dir / "frame_%06d.png"),
        ]

        try:
            result = subprocess.run(cmd_cuda, capture_output=True, text=True)
            if result.returncode != 0:
                # CUDA failed, try CPU fallback with optimized seeking
                print("  CUDA frame extraction failed, falling back to CPU")
                cmd_cpu = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(start_time),  # Seek before decoding
                    "-i",
                    video_path,
                    "-t",
                    str(duration),  # Duration limit
                    "-pix_fmt",
                    "rgb24",
                    "-vsync",
                    "0",
                    "-threads",
                    "0",  # Auto-detect optimal thread count
                    str(frames_dir / "frame_%06d.png"),
                ]
                result_cpu = subprocess.run(cmd_cpu, capture_output=True, text=True)
                if result_cpu.returncode != 0:
                    print(f"FFmpeg error: {result_cpu.stderr}")
                    return []
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return []

        return get_frame_files(frames_dir)

    def _check_nvenc_available(self) -> bool:
        """
        Check if NVIDIA NVENC hardware encoder is available.

        Returns:
            True if NVENC is available, False otherwise

        Side effects:
            - Executes FFmpeg subprocess for encoder detection
        """
        test_result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
        )
        return "hevc_nvenc" in test_result.stdout

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
                print("  Using NVENC hardware encoding (H.265)")
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
                        str(output_path),
                    ],
                    True,
                )
            elif encoder == "nvenc":
                print("  Warning: NVENC not available, falling back to software encoding")
            # Fall through to software encoding

        # Software encoding (default or explicit)
        if encoder in ["libx264", "libx265"]:
            codec = encoder
        else:
            # Unknown encoder or auto fallback
            codec = "libx264"
            if encoder not in ["auto", "nvenc"]:
                print(f"  Warning: Unknown encoder '{encoder}', using libx264")

        print(f"  Using software encoding ({codec})")
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
