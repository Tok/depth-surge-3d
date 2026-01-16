"""
Video processor with temporal consistency.

This module implements video processing using Video-Depth-Anything
for temporal consistency across video frames.
"""

import cv2
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import torch

from ..models.video_depth_estimator import VideoDepthEstimator
from ..utils.progress import create_progress_tracker
from ..utils.file_operations import (
    create_output_directories,
    get_frame_files,
    calculate_frame_range,
    generate_output_filename,
    verify_ffmpeg_installation,
    save_processing_settings,
    update_processing_status,
)
from ..utils.image_processing import (
    resize_image,
    depth_to_disparity,
    create_shifted_image,
    apply_center_crop,
    apply_fisheye_distortion,
    apply_fisheye_square_crop,
    create_vr_frame,
    hole_fill_image,
)
from ..core.constants import INTERMEDIATE_DIRS
from ..utils.console import (
    step_complete,
    saved_to,
    title_bar,
    success as console_success,
)


class VideoProcessor:
    """
    Handles video processing with temporal consistency.

    Uses Video-Depth-Anything model to process entire videos with
    temporal consistency for superior depth estimation quality.
    """

    def __init__(self, depth_estimator: VideoDepthEstimator, verbose: bool = False):
        self.depth_estimator = depth_estimator
        self.verbose = verbose
        self._settings_file = None  # Track settings file for error handling

    def _update_step_progress(
        self, progress_tracker, message: str, step_name: str, progress: int, total: int
    ) -> None:
        """Update progress for a processing step."""
        if progress_tracker:
            progress_tracker.update_progress(
                message,
                phase="processing",
                frame_num=progress,
                step_name=step_name,
                step_progress=progress,
                step_total=total,
            )

    def _print_step_complete(
        self, num_items: int, duration: float, item_type: str = "frames"
    ) -> None:
        """Print step completion message."""
        print(
            step_complete(f"Processed {num_items:04d} {item_type} in {duration:.2f}s")
        )

    def _print_saved_to(
        self, directory: Path, message_prefix: str = "Saved to"
    ) -> None:
        """Print save location message."""
        if directory:
            print(saved_to(f"{message_prefix}: {directory}\n"))
        else:
            print()

    def _handle_step_error(self, error_msg: str) -> bool:
        """Handle step failure and update settings file."""
        print(f"Error: {error_msg}")
        if self._settings_file:
            update_processing_status(
                self._settings_file, "failed", {"error": error_msg}
            )
        return False

    def _setup_processing(
        self,
        video_path: str,
        output_dir: str,
        settings: Dict[str, Any],
        video_properties: Dict[str, Any],
    ) -> tuple[Path, Dict[str, Path], Optional[Path]]:
        """Setup processing directories and settings file."""
        output_path = Path(output_dir)
        directories = create_output_directories(
            output_path, settings["keep_intermediates"]
        )
        batch_name = f"{Path(video_path).stem}_{int(time.time())}"
        settings_file = save_processing_settings(
            output_path, batch_name, settings, video_properties, video_path
        )

        print(f"\n{title_bar('=== Depth Surge 3D Video Processing ===')}")
        print(f"Input: {video_path}")
        print(f"Output: {output_path}")
        print("Using Video-Depth-Anything for temporal consistency\n")

        return output_path, directories, settings_file

    def _finalize_processing(
        self,
        success: bool,
        output_path: Path,
        video_path: str,
        settings: Dict[str, Any],
        num_frames: int,
    ) -> None:
        """Finalize processing and update settings file."""
        if success:
            print(console_success("Processing complete!"))
            if self._settings_file:
                output_filename = generate_output_filename(
                    Path(video_path).name,
                    settings["vr_format"],
                    settings["vr_resolution"],
                )
                update_processing_status(
                    self._settings_file,
                    "completed",
                    {
                        "final_output": str(output_path / output_filename),
                        "frames_processed": num_frames,
                    },
                )
        elif self._settings_file:
            update_processing_status(
                self._settings_file, "failed", {"error": "Video creation failed"}
            )

    def _step_extract_frames(
        self,
        video_path: str,
        directories: Dict[str, Path],
        video_properties: Dict[str, Any],
        settings: Dict[str, Any],
        progress_callback,
    ) -> Optional[List[Path]]:
        """Execute Step 1: Extract frames from video."""
        # Check if frames already exist (resume functionality)
        frames_dir = directories.get("frames")
        if frames_dir and frames_dir.exists():
            existing_frames = get_frame_files(frames_dir)
            if existing_frames:
                print("Step 1/7: Skipping frame extraction (frames already exist)")
                print(f"  Found {len(existing_frames):04d} existing frames")
                print(saved_to(f"  Location: {frames_dir}\n"))
                if progress_callback:
                    progress_callback.update_progress(
                        "Skipped frame extraction (already exists)",
                        phase="extraction",
                        frame_num=len(existing_frames),
                        step_name="Frame Extraction",
                        step_progress=1,
                        step_total=1,
                    )
                return existing_frames

        print("Step 1/7: Extracting frames from video...")
        if progress_callback:
            progress_callback.update_progress(
                "Extracting frames from video",
                phase="extraction",
                frame_num=0,
                step_name="Frame Extraction",
                step_progress=0,
                step_total=1,
            )

        frame_files = self._extract_frames(
            video_path, directories, video_properties, settings
        )
        if not frame_files:
            self._handle_step_error("No frames extracted from video")
            return None

        duration = progress_callback.get_step_duration() if progress_callback else 0
        print(
            step_complete(f"Extracted {len(frame_files):04d} frames in {duration:.2f}s")
        )
        print(saved_to(f"Saved to: {directories.get('frames', 'N/A')}\n"))
        return frame_files

    def _step_generate_depth_maps(
        self,
        frame_files: List[Path],
        settings: Dict[str, Any],
        directories: Dict[str, Path],
        progress_tracker,
    ) -> Optional[np.ndarray]:
        """Execute Step 2: Generate depth maps."""
        # Check if depth maps already exist (only if keep_intermediates is enabled)
        if settings.get("keep_intermediates") and "depth_maps" in directories:
            depth_maps_dir = directories["depth_maps"]
            if depth_maps_dir.exists():
                existing_depth_maps = sorted(list(depth_maps_dir.glob("*.png")))
                if existing_depth_maps and len(existing_depth_maps) >= len(frame_files):
                    print(
                        "Step 2/7: Skipping depth map generation (depth maps already exist)"
                    )
                    print(f"  Found {len(existing_depth_maps):04d} existing depth maps")
                    print(saved_to(f"  Location: {depth_maps_dir}\n"))
                    # Load existing depth maps
                    depth_maps = []
                    for depth_file in existing_depth_maps[: len(frame_files)]:
                        depth_img = cv2.imread(str(depth_file), cv2.IMREAD_GRAYSCALE)
                        if depth_img is not None:
                            depth_maps.append(depth_img.astype(float) / 255.0)
                    if len(depth_maps) == len(frame_files):
                        if progress_tracker:
                            progress_tracker.update_progress(
                                "Skipped depth map generation (already exists)",
                                phase="depth_estimation",
                                frame_num=len(depth_maps),
                                step_name="Depth Map Generation",
                                step_progress=len(depth_maps),
                                step_total=len(depth_maps),
                            )
                        return np.array(depth_maps)

        print("Step 2/7: Generating depth maps (temporal consistency enabled)...")
        print("  Using memory-efficient chunked processing...")
        progress_tracker.update_progress(
            "Generating depth maps",
            phase="depth_estimation",
            frame_num=0,
            step_name="Depth Map Generation",
            step_progress=0,
            step_total=len(frame_files),
        )

        depth_maps = self._generate_depth_maps_chunked(
            frame_files, settings, directories, progress_tracker
        )
        if depth_maps is None:
            self._handle_step_error("Depth map generation failed")
            return None

        duration = (
            progress_tracker.get_step_duration()
            if hasattr(progress_tracker, "get_step_duration")
            else 0
        )
        print(
            step_complete(
                f"Generated {len(depth_maps):04d} depth maps in {duration:.2f}s"
            )
        )
        if settings["keep_intermediates"] and "depth_maps" in directories:
            print(saved_to(f"Saved to: {directories['depth_maps']}\n"))
        else:
            print()
        return depth_maps

    def _step_load_frames(
        self, frame_files: List[Path], settings: Dict[str, Any], progress_tracker
    ) -> Optional[np.ndarray]:
        """Execute Step 3: Load frames for stereo processing."""
        print("Step 3/7: Loading frames for stereo processing...")
        progress_tracker.update_progress(
            "Loading frames",
            phase="depth_estimation",
            frame_num=0,
            step_name="Frame Loading",
            step_progress=0,
            step_total=len(frame_files),
        )

        frames = self._load_frames(frame_files, settings, progress_tracker)
        if frames is None or len(frames) == 0:
            self._handle_step_error("Failed to load frames")
            return None

        duration = (
            progress_tracker.get_step_duration()
            if hasattr(progress_tracker, "get_step_duration")
            else 0
        )
        print(step_complete(f"Loaded {len(frames):04d} frames in {duration:.2f}s"))
        print()
        return frames

    def _step_create_stereo_pairs(
        self,
        frames: np.ndarray,
        depth_maps: np.ndarray,
        frame_files: List[Path],
        directories: Dict[str, Path],
        settings: Dict[str, Any],
        progress_tracker,
    ) -> bool:
        """Execute Step 4: Create stereo pairs."""
        # Check if stereo pairs already exist (only if keep_intermediates is enabled)
        if (
            settings.get("keep_intermediates")
            and "left_frames" in directories
            and "right_frames" in directories
        ):
            left_dir = directories["left_frames"]
            right_dir = directories["right_frames"]
            if left_dir.exists() and right_dir.exists():
                existing_left = sorted(list(left_dir.glob("*.png")))
                existing_right = sorted(list(right_dir.glob("*.png")))
                if (
                    existing_left
                    and existing_right
                    and len(existing_left) >= len(frames)
                    and len(existing_right) >= len(frames)
                ):
                    print(
                        "Step 4/7: Skipping stereo pair creation (stereo pairs already exist)"
                    )
                    print(
                        f"  Found {len(existing_left):04d} left and {len(existing_right):04d} right frames"
                    )
                    print(saved_to(f"  Location: {left_dir} & {right_dir}\n"))
                    if progress_tracker:
                        progress_tracker.update_progress(
                            "Skipped stereo pair creation (already exists)",
                            phase="stereo_generation",
                            frame_num=len(existing_left),
                            step_name="Stereo Pair Creation",
                            step_progress=len(existing_left),
                            step_total=len(existing_left),
                        )
                    return True

        print("Step 4/7: Creating stereo pairs...")
        success = self._create_stereo_pairs(
            frames, depth_maps, frame_files, directories, settings, progress_tracker
        )
        if not success:
            self._handle_step_error("Stereo pair creation failed")
            return False

        duration = (
            progress_tracker.get_step_duration()
            if hasattr(progress_tracker, "get_step_duration")
            else 0
        )
        print(
            step_complete(
                f"Created stereo pairs for {len(frames):04d} frames in {duration:.2f}s"
            )
        )
        if settings["keep_intermediates"] and "left_frames" in directories:
            print(
                saved_to(
                    f"Saved to: {directories['left_frames']} & {directories['right_frames']}\n"
                )
            )
        else:
            print()
        return True

    def _step_apply_distortion(
        self, directories: Dict[str, Path], settings: Dict[str, Any], progress_tracker
    ) -> bool:
        """Execute Step 5: Apply fisheye distortion (if enabled)."""
        if not settings["apply_distortion"]:
            print("Step 5/7: Skipping fisheye distortion (disabled)\n")
            return True

        # Check if distorted frames already exist (only if keep_intermediates is enabled)
        if (
            settings.get("keep_intermediates")
            and "left_distorted" in directories
            and "right_distorted" in directories
        ):
            left_distorted_dir = directories["left_distorted"]
            right_distorted_dir = directories["right_distorted"]
            if left_distorted_dir.exists() and right_distorted_dir.exists():
                existing_left = sorted(list(left_distorted_dir.glob("*.png")))
                existing_right = sorted(list(right_distorted_dir.glob("*.png")))
                # Count source frames for comparison
                left_files = (
                    sorted(directories["left_frames"].glob("*.png"))
                    if "left_frames" in directories
                    else []
                )
                if (
                    existing_left
                    and existing_right
                    and len(existing_left) >= len(left_files)
                    and len(existing_right) >= len(left_files)
                ):
                    print(
                        "Step 5/7: Skipping fisheye distortion (distorted frames already exist)"
                    )
                    print(
                        f"  Found {len(existing_left):04d} distorted left and {len(existing_right):04d} distorted right frames"
                    )
                    print(
                        saved_to(
                            f"  Location: {left_distorted_dir} & {right_distorted_dir}\n"
                        )
                    )
                    if progress_tracker:
                        progress_tracker.update_progress(
                            "Skipped fisheye distortion (already exists)",
                            phase="distortion",
                            frame_num=len(existing_left),
                            step_name="Fisheye Distortion",
                            step_progress=len(existing_left),
                            step_total=len(existing_left),
                        )
                    return True

        print("Step 5/7: Applying fisheye distortion...")
        left_files = (
            sorted(directories["left_frames"].glob("*.png"))
            if "left_frames" in directories
            else []
        )
        right_files = (
            sorted(directories["right_frames"].glob("*.png"))
            if "right_frames" in directories
            else []
        )

        success = self._apply_distortion(
            left_files, right_files, directories, settings, progress_tracker
        )
        if not success:
            self._handle_step_error("Distortion failed")
            return False

        duration = (
            progress_tracker.get_step_duration()
            if hasattr(progress_tracker, "get_step_duration")
            else 0
        )
        print(
            step_complete(
                f"Applied fisheye distortion to {len(left_files):04d} frames in {duration:.2f}s"
            )
        )
        if settings["keep_intermediates"] and "left_distorted" in directories:
            print(
                saved_to(
                    f"Saved to: {directories['left_distorted']} & {directories['right_distorted']}\n"
                )
            )
        else:
            print()
        return True

    def _step_create_vr_frames(
        self,
        directories: Dict[str, Path],
        settings: Dict[str, Any],
        progress_tracker,
        num_frames: int,
    ) -> bool:
        """Execute Step 6: Create final VR frames."""
        # Check if VR frames already exist (resume functionality)
        vr_frames_dir = directories.get("vr_frames")
        if vr_frames_dir and vr_frames_dir.exists():
            existing_vr_frames = sorted(list(vr_frames_dir.glob("*.png")))
            if existing_vr_frames and len(existing_vr_frames) >= num_frames:
                print("Step 6/7: Skipping VR frame creation (VR frames already exist)")
                print(f"  Found {len(existing_vr_frames):04d} existing VR frames")
                print(saved_to(f"  Location: {vr_frames_dir}\n"))
                if progress_tracker:
                    progress_tracker.update_progress(
                        "Skipped VR frame creation (already exists)",
                        phase="vr_assembly",
                        frame_num=len(existing_vr_frames),
                        step_name="Final Processing",
                        step_progress=len(existing_vr_frames),
                        step_total=len(existing_vr_frames),
                    )
                return True

        print("Step 6/7: Creating final VR frames...")
        success = self._create_vr_frames(
            directories, settings, progress_tracker, num_frames
        )
        if not success:
            self._handle_step_error("VR frame creation failed")
            return False

        duration = (
            progress_tracker.get_step_duration()
            if hasattr(progress_tracker, "get_step_duration")
            else 0
        )
        print(step_complete(f"Created {num_frames:04d} VR frames in {duration:.2f}s"))
        if "vr_frames" in directories:
            print(saved_to(f"Saved to: {directories['vr_frames']}\n"))
        else:
            print()
        return True

    def _step_create_final_video(
        self,
        directories: Dict[str, Path],
        output_path: Path,
        video_path: str,
        settings: Dict[str, Any],
        progress_tracker,
        progress_callback,
    ) -> bool:
        """Execute Step 7: Create final video with audio."""
        print("Step 7/7: Creating final video with audio...")
        if progress_callback:
            progress_callback.update_progress(
                "Creating final video",
                phase="video_creation",
                frame_num=0,
                step_name="Video Creation",
                step_progress=0,
                step_total=1,
            )

        success = self._create_output_video(
            directories["vr_frames"], output_path, video_path, settings
        )
        if success:
            output_filename = generate_output_filename(
                Path(video_path).name, settings["vr_format"], settings["vr_resolution"]
            )
            duration = (
                progress_tracker.get_step_duration()
                if hasattr(progress_tracker, "get_step_duration")
                else 0
            )
            print(step_complete(f"Created final video in {duration:.2f}s"))
            print(saved_to(f"Saved to: {output_path / output_filename}\n"))

        return success

    def process(
        self,
        video_path: str,
        output_dir: str,
        video_properties: Dict[str, Any],
        settings: Dict[str, Any],
        progress_callback=None,
    ) -> bool:
        """
        Process video in batch mode with temporal consistency.

        Args:
            video_path: Path to input video
            output_dir: Output directory path
            video_properties: Video metadata
            settings: Processing settings
            progress_callback: Optional progress callback for web UI

        Returns:
            True if processing completed successfully
        """
        try:
            # Setup processing environment
            output_path, directories, self._settings_file = self._setup_processing(
                video_path, output_dir, settings, video_properties
            )

            # Step 1: Extract frames
            frame_files = self._step_extract_frames(
                video_path, directories, video_properties, settings, progress_callback
            )
            if not frame_files:
                return False

            # Initialize progress tracker
            progress_tracker = (
                progress_callback
                if progress_callback
                else create_progress_tracker(len(frame_files), "batch")
            )

            # Step 2: Generate depth maps
            depth_maps = self._step_generate_depth_maps(
                frame_files, settings, directories, progress_tracker
            )
            if depth_maps is None:
                return False

            # Step 3: Load frames for stereo processing
            frames = self._step_load_frames(frame_files, settings, progress_tracker)
            if frames is None:
                return False

            # Step 4: Create stereo pairs
            if not self._step_create_stereo_pairs(
                frames, depth_maps, frame_files, directories, settings, progress_tracker
            ):
                return False

            # Step 5: Apply fisheye distortion (if enabled)
            if not self._step_apply_distortion(directories, settings, progress_tracker):
                return False

            # Step 6: Create final VR frames
            if not self._step_create_vr_frames(
                directories, settings, progress_tracker, len(frames)
            ):
                return False

            # Step 7: Create final video
            success = self._step_create_final_video(
                directories,
                output_path,
                video_path,
                settings,
                progress_tracker,
                progress_callback,
            )

            # Finalize and cleanup
            progress_tracker.finish("Video processing complete")
            self._finalize_processing(
                success, output_path, video_path, settings, len(frames)
            )
            return success

        except Exception as e:
            print(f"Error in video processing: {e}")
            if self._settings_file:
                update_processing_status(
                    self._settings_file, "failed", {"error": str(e)}
                )
            return False

    def _extract_frames(
        self,
        video_path: str,
        directories: Dict[str, Path],
        video_properties: Dict[str, Any],
        settings: Dict[str, Any],
    ) -> List[Path]:
        """Extract frames from video."""
        frames_dir = directories.get("frames")
        if not frames_dir:
            frames_dir = directories["base"] / INTERMEDIATE_DIRS["frames"]
            frames_dir.mkdir(exist_ok=True)

        # Calculate frame range
        total_frames = video_properties["frame_count"]
        fps = video_properties["fps"]
        start_frame, end_frame = calculate_frame_range(
            total_frames, fps, settings.get("start_time"), settings.get("end_time")
        )
        expected_frames = end_frame - start_frame

        # Try CUDA acceleration first, fall back to CPU if unavailable
        cmd_cuda = [
            "ffmpeg",
            "-y",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-i",
            video_path,
            "-vf",
            f"select=between(n\\,{start_frame}\\,{end_frame - 1}),hwdownload,format=nv12,format=rgb24",
            "-pix_fmt",
            "rgb24",
            "-frames:v",
            str(expected_frames),
            "-fps_mode",
            "passthrough",
            str(frames_dir / "frame_%06d.png"),
        ]

        try:
            result = subprocess.run(cmd_cuda, capture_output=True, text=True)
            if result.returncode != 0:
                # CUDA failed, try CPU fallback
                print("  CUDA frame extraction failed, falling back to CPU")
                cmd_cpu = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                    "-vf",
                    f"select=between(n\\,{start_frame}\\,{end_frame - 1})",
                    "-pix_fmt",
                    "rgb24",
                    "-frames:v",
                    str(expected_frames),
                    "-fps_mode",
                    "passthrough",
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

    def _load_frames(
        self, frame_files: List[Path], settings: Dict[str, Any], progress_tracker
    ) -> Optional[np.ndarray]:
        """Load all frames into memory as numpy array."""
        frames_list = []

        for i, frame_file in enumerate(frame_files):
            try:
                image = cv2.imread(str(frame_file))
                if image is None:
                    print(f"Warning: Could not load frame {frame_file}")
                    continue

                # Apply super sampling if needed
                if settings["super_sample"] != "none":
                    target_width = max(image.shape[1], settings["per_eye_width"] * 2)
                    target_height = max(image.shape[0], settings["per_eye_height"] * 2)
                    image = resize_image(image, target_width, target_height)

                frames_list.append(image)

                # Update progress
                if i % 10 == 0 or i == len(frame_files) - 1:
                    progress_tracker.update_progress(
                        "Loading frames",
                        phase="depth_estimation",
                        frame_num=i + 1,
                        step_name="Frame Loading",
                        step_progress=i + 1,
                        step_total=len(frame_files),
                    )

            except Exception as e:
                print(f"Error loading frame {frame_file}: {e}")
                return None

        if not frames_list:
            return None

        return np.array(frames_list)

    def _get_chunk_size_for_resolution(self, input_size: int) -> int:
        """Get appropriate chunk size based on depth map resolution.

        Args:
            input_size: Depth map resolution in pixels

        Returns:
            Chunk size for processing
        """
        if input_size >= 2160:
            return 4
        elif input_size >= 1440:
            return 6
        elif input_size >= 1080:
            return 12
        elif input_size >= 720:
            return 16
        else:
            return 32

    def _determine_chunk_params(
        self, frame_w: int, frame_h: int, depth_resolution: str = "auto"
    ) -> tuple[int, int]:
        """Determine chunk size and input size based on frame resolution and settings.

        Args:
            frame_w: Frame width in pixels
            frame_h: Frame height in pixels
            depth_resolution: Either "auto" or specific resolution like "1080", "720", etc.

        Returns:
            Tuple of (chunk_size, input_size)
        """
        megapixels = (frame_h * frame_w) / 1_000_000
        print(f"  Frame resolution: {frame_w}x{frame_h} ({megapixels:.1f}MP)")

        # If specific resolution is requested, use it
        if depth_resolution != "auto":
            try:
                input_size = int(depth_resolution)
                chunk_size = self._get_chunk_size_for_resolution(input_size)
                print(
                    f"  Using manual depth resolution: {input_size}px (chunk size: {chunk_size})"
                )
                return chunk_size, input_size
            except (ValueError, TypeError):
                print(
                    f"  Warning: Invalid depth_resolution '{depth_resolution}', using auto"
                )

        # Auto mode: Match depth resolution to actual frame size (or slightly higher for quality)
        # Never exceed source frame resolution - upscaling depth beyond that is pointless
        if megapixels > 8.0:  # >8MP (4K is ~8.3MP)
            # For 4K, use full resolution - DA3 can handle it
            return 4, min(max(frame_w, frame_h), 2160)
        elif megapixels > 2.0:  # >2MP (1080p is 2.1MP)
            # For 1080p, match the resolution exactly
            return 8, min(max(frame_w, frame_h), 1080)
        elif megapixels > 1.0:  # >1MP (720p is 0.9MP)
            # For 720p, match the resolution
            return 12, min(max(frame_w, frame_h), 720)
        else:
            # For SD, match actual resolution (usually 480p or 576p)
            return 24, min(max(frame_w, frame_h), 640)

    def _clear_gpu_memory(self) -> None:
        """Clear GPU cache and print available memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            mem_free = torch.cuda.mem_get_info()[0] / (1024**3)  # Convert to GB
            print(f"  GPU memory freed: {mem_free:.2f} GB available")

    def _load_chunk_frames(
        self, chunk_files: List[Path], settings: Dict[str, Any]
    ) -> Optional[List]:
        """Load and optionally supersample frames for a chunk."""
        chunk_frames = []
        for frame_file in chunk_files:
            frame = cv2.imread(str(frame_file))
            if frame is None:
                print(f"Warning: Could not load {frame_file}")
                continue

            # Apply super sampling if needed
            if settings["super_sample"] != "none":
                target_width = max(frame.shape[1], settings["per_eye_width"] * 2)
                target_height = max(frame.shape[0], settings["per_eye_height"] * 2)
                frame = resize_image(frame, target_width, target_height)

            chunk_frames.append(frame)

        return chunk_frames if chunk_frames else None

    def _process_chunk_depth(
        self,
        chunk_frames: List,
        chunk_files: List[Path],
        settings: Dict[str, Any],
        directories: Dict[str, Path],
        input_size: int,
    ) -> Optional[np.ndarray]:
        """Process depth for a chunk and optionally save results."""
        # Normalize target_fps
        target_fps = settings.get("target_fps", 30)
        if target_fps is None or str(target_fps) == "None" or target_fps == "original":
            target_fps = 30

        # Estimate depth
        chunk_frames_array = np.array(chunk_frames)
        chunk_depth_maps = self.depth_estimator.estimate_depth_batch(
            chunk_frames_array, target_fps=target_fps, input_size=input_size, fp32=False
        )

        # Save depth maps immediately to free memory
        if settings["keep_intermediates"] and "depth_maps" in directories:
            self._save_depth_maps(
                chunk_depth_maps, chunk_files, directories["depth_maps"]
            )

        return chunk_depth_maps

    def _generate_depth_maps_chunked(
        self,
        frame_files: List[Path],
        settings: Dict[str, Any],
        directories: Dict[str, Path],
        progress_tracker,
    ) -> Optional[np.ndarray]:
        """
        Generate depth maps in memory-efficient chunks.

        Processes frames in small batches to avoid CUDA OOM errors.
        """
        # Determine chunk parameters based on resolution
        sample_frame = cv2.imread(str(frame_files[0]))
        if sample_frame is None:
            return None

        frame_h, frame_w = sample_frame.shape[:2]
        depth_resolution = settings.get("depth_resolution", "auto")
        chunk_size, input_size = self._determine_chunk_params(
            frame_w, frame_h, depth_resolution
        )

        print(
            f"  Processing in chunks of {chunk_size} frames (input_size={input_size})..."
        )

        # Clear GPU cache before processing
        self._clear_gpu_memory()

        # Process all chunks
        all_depth_maps = []
        num_frames = len(frame_files)
        total_chunks = (num_frames + chunk_size - 1) // chunk_size

        for chunk_start in range(0, num_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_frames)
            chunk_files = frame_files[chunk_start:chunk_end]
            chunk_num = chunk_start // chunk_size + 1

            # Load chunk frames
            chunk_frames = self._load_chunk_frames(chunk_files, settings)
            if not chunk_frames:
                print("Error: No frames loaded in chunk")
                return None

            # Process chunk for depth
            try:
                chunk_depth_maps = self._process_chunk_depth(
                    chunk_frames, chunk_files, settings, directories, input_size
                )
                all_depth_maps.extend(chunk_depth_maps)

                # Clear references and GPU cache
                del chunk_frames
                del chunk_depth_maps
                self._clear_gpu_memory()

                # Update progress
                progress_tracker.update_progress(
                    f"Chunk {chunk_num}/{total_chunks}: Depth maps {chunk_end}/{num_frames}",
                    phase="depth_estimation",
                    frame_num=chunk_end,
                    step_name="Depth Map Generation",
                    step_progress=chunk_end,
                    step_total=num_frames,
                )

            except Exception as e:
                print(f"Error processing chunk {chunk_start}-{chunk_end}: {e}")
                return None

        return np.array(all_depth_maps)

    def _generate_depth_maps_batch(
        self, frames: np.ndarray, settings: Dict[str, Any], progress_tracker
    ) -> Optional[np.ndarray]:
        """Generate depth maps for all frames with temporal consistency."""
        try:
            # Use Video-Depth-Anything for temporal consistency
            target_fps = settings.get("target_fps", 30)
            if (
                target_fps is None
                or str(target_fps) == "None"
                or target_fps == "original"
            ):
                target_fps = 30

            # Use depth resolution from settings (default: auto/1080px)
            depth_resolution = settings.get("depth_resolution", "auto")
            if depth_resolution == "auto":
                input_size = 1080  # Match typical 1080p video resolution
            else:
                try:
                    input_size = int(depth_resolution)
                except (ValueError, TypeError):
                    input_size = 1080

            depth_maps = self.depth_estimator.estimate_depth_batch(
                frames, target_fps=target_fps, input_size=input_size, fp32=False
            )

            return depth_maps

        except Exception as e:
            print(f"Error generating depth maps: {e}")
            return None

    def _save_depth_maps(
        self, depth_maps: np.ndarray, frame_files: List[Path], depth_dir: Path
    ) -> None:
        """Save depth maps to disk."""
        for i, (depth_map, frame_file) in enumerate(zip(depth_maps, frame_files)):
            depth_vis = (depth_map * 255).astype("uint8")
            frame_name = frame_file.stem
            cv2.imwrite(str(depth_dir / f"{frame_name}.png"), depth_vis)

    def _create_stereo_pairs(
        self,
        frames: np.ndarray,
        depth_maps: np.ndarray,
        frame_files: List[Path],
        directories: Dict[str, Path],
        settings: Dict[str, Any],
        progress_tracker,
    ) -> bool:
        """Create stereo pairs from frames and depth maps."""
        try:
            for i, (frame, depth_map, frame_file) in enumerate(
                zip(frames, depth_maps, frame_files)
            ):
                frame_name = frame_file.stem

                # Create stereo pair
                disparity_map = depth_to_disparity(
                    depth_map, settings["baseline"], settings["focal_length"]
                )

                left_img = create_shifted_image(frame, disparity_map, "left")
                right_img = create_shifted_image(frame, disparity_map, "right")

                # Apply hole filling
                if settings["hole_fill_quality"] in ["fast", "advanced"]:
                    left_img = hole_fill_image(
                        left_img, method=settings["hole_fill_quality"]
                    )
                    right_img = hole_fill_image(
                        right_img, method=settings["hole_fill_quality"]
                    )

                # Save stereo pair if keeping intermediates
                if settings["keep_intermediates"]:
                    if "left_frames" in directories:
                        cv2.imwrite(
                            str(directories["left_frames"] / f"{frame_name}.png"),
                            left_img,
                        )
                    if "right_frames" in directories:
                        cv2.imwrite(
                            str(directories["right_frames"] / f"{frame_name}.png"),
                            right_img,
                        )

                # Update progress
                if i % 5 == 0 or i == len(frames) - 1:
                    progress_tracker.update_progress(
                        "Creating stereo pairs",
                        phase="stereo_generation",
                        frame_num=i + 1,
                        step_name="Stereo Pair Creation",
                        step_progress=i + 1,
                        step_total=len(frames),
                    )

            return True

        except Exception as e:
            print(f"Error creating stereo pairs: {e}")
            return False

    def _apply_distortion(
        self,
        left_files: List[Path],
        right_files: List[Path],
        directories: Dict[str, Path],
        settings: Dict[str, Any],
        progress_tracker,
    ) -> bool:
        """Apply fisheye distortion to stereo pairs."""
        try:
            for i, (left_file, right_file) in enumerate(zip(left_files, right_files)):
                # Load images
                left_img = cv2.imread(str(left_file))
                right_img = cv2.imread(str(right_file))

                if left_img is None or right_img is None:
                    print(f"Warning: Could not load {left_file} or {right_file}")
                    continue

                # Apply fisheye distortion
                left_distorted = apply_fisheye_distortion(
                    left_img, settings["fisheye_fov"], settings["fisheye_projection"]
                )
                right_distorted = apply_fisheye_distortion(
                    right_img, settings["fisheye_fov"], settings["fisheye_projection"]
                )

                # Save distorted frames if keeping intermediates
                if settings["keep_intermediates"]:
                    frame_name = left_file.stem
                    if "left_distorted" in directories:
                        cv2.imwrite(
                            str(directories["left_distorted"] / f"{frame_name}.png"),
                            left_distorted,
                        )
                    if "right_distorted" in directories:
                        cv2.imwrite(
                            str(directories["right_distorted"] / f"{frame_name}.png"),
                            right_distorted,
                        )

                # Update progress
                if i % 5 == 0 or i == len(left_files) - 1:
                    progress_tracker.update_progress(
                        "Applying distortion",
                        phase="distortion",
                        frame_num=i + 1,
                        step_name="Fisheye Distortion",
                        step_progress=i + 1,
                        step_total=len(left_files),
                    )

            return True

        except Exception as e:
            print(f"Error applying distortion: {e}")
            return False

    def _get_stereo_source_dirs(
        self, directories: Dict[str, Path], settings: Dict[str, Any]
    ) -> Optional[tuple[Path, Path]]:
        """Determine source directories for stereo frames."""
        if settings["apply_distortion"] and "left_distorted" in directories:
            return directories["left_distorted"], directories["right_distorted"]
        elif "left_frames" in directories:
            return directories["left_frames"], directories["right_frames"]
        else:
            print("Error: No stereo frames found")
            return None

    def _process_fisheye_frame_pair(
        self, left_img, right_img, settings: Dict[str, Any]
    ) -> tuple:
        """Process frame pair with fisheye distortion applied."""
        fisheye_crop_factor = max(
            0.5, min(2.0, float(settings.get("fisheye_crop_factor", 0.7)))
        )

        left_cropped = apply_fisheye_square_crop(
            left_img,
            settings["per_eye_width"],
            settings["per_eye_height"],
            fisheye_crop_factor,
        )
        right_cropped = apply_fisheye_square_crop(
            right_img,
            settings["per_eye_width"],
            settings["per_eye_height"],
            fisheye_crop_factor,
        )

        left_final = resize_image(
            left_cropped, settings["per_eye_width"], settings["per_eye_height"]
        )
        right_final = resize_image(
            right_cropped, settings["per_eye_width"], settings["per_eye_height"]
        )

        return left_cropped, right_cropped, left_final, right_final

    def _process_regular_frame_pair(
        self, left_img, right_img, settings: Dict[str, Any]
    ) -> tuple:
        """Process frame pair without fisheye distortion."""
        crop_factor = max(0.5, min(1.0, float(settings.get("crop_factor", 1.0))))

        left_cropped = apply_center_crop(left_img, crop_factor)
        right_cropped = apply_center_crop(right_img, crop_factor)

        left_final = resize_image(
            left_cropped, settings["per_eye_width"], settings["per_eye_height"]
        )
        right_final = resize_image(
            right_cropped, settings["per_eye_width"], settings["per_eye_height"]
        )

        return left_cropped, right_cropped, left_final, right_final

    def _save_vr_intermediate_frames(
        self,
        directories: Dict[str, Path],
        frame_name: str,
        left_cropped,
        right_cropped,
        left_final,
        right_final,
    ) -> None:
        """Save intermediate cropped and final frames."""
        if "left_cropped" in directories:
            cv2.imwrite(
                str(directories["left_cropped"] / f"{frame_name}.png"), left_cropped
            )
        if "right_cropped" in directories:
            cv2.imwrite(
                str(directories["right_cropped"] / f"{frame_name}.png"), right_cropped
            )
        if "left_final" in directories:
            cv2.imwrite(
                str(directories["left_final"] / f"{frame_name}.png"), left_final
            )
        if "right_final" in directories:
            cv2.imwrite(
                str(directories["right_final"] / f"{frame_name}.png"), right_final
            )

    def _process_single_vr_frame(
        self,
        left_file: Path,
        right_file: Path,
        directories: Dict[str, Path],
        settings: Dict[str, Any],
    ) -> bool:
        """Process a single VR frame pair."""
        # Load images
        left_img = cv2.imread(str(left_file))
        right_img = cv2.imread(str(right_file))

        if left_img is None or right_img is None:
            print(f"Warning: Could not load {left_file} or {right_file}")
            return False

        # Process frame pair based on distortion setting
        if settings["apply_distortion"]:
            left_cropped, right_cropped, left_final, right_final = (
                self._process_fisheye_frame_pair(left_img, right_img, settings)
            )
        else:
            left_cropped, right_cropped, left_final, right_final = (
                self._process_regular_frame_pair(left_img, right_img, settings)
            )

        # Save intermediate frames if requested
        frame_name = left_file.stem
        if settings["keep_intermediates"]:
            self._save_vr_intermediate_frames(
                directories,
                frame_name,
                left_cropped,
                right_cropped,
                left_final,
                right_final,
            )

        # Create and save final VR frame
        vr_frame = create_vr_frame(left_final, right_final, settings["vr_format"])
        if "vr_frames" in directories:
            cv2.imwrite(str(directories["vr_frames"] / f"{frame_name}.png"), vr_frame)

        return True

    def _create_vr_frames(
        self,
        directories: Dict[str, Path],
        settings: Dict[str, Any],
        progress_tracker,
        total_frames: int,
    ) -> bool:
        """Create final VR frames from processed left/right frames."""
        try:
            # Determine source directories
            stereo_dirs = self._get_stereo_source_dirs(directories, settings)
            if not stereo_dirs:
                return False

            left_dir, right_dir = stereo_dirs

            # Get frame files
            left_files = sorted(left_dir.glob("*.png"))
            right_files = sorted(right_dir.glob("*.png"))

            if len(left_files) != len(right_files):
                print(
                    f"Warning: Mismatched frame count: {len(left_files)} left, {len(right_files)} right"
                )

            # Process each frame pair
            for i, (left_file, right_file) in enumerate(zip(left_files, right_files)):
                self._process_single_vr_frame(
                    left_file, right_file, directories, settings
                )

                # Update progress periodically
                if i % 5 == 0 or i == len(left_files) - 1:
                    progress_tracker.update_progress(
                        "Creating VR frames",
                        phase="vr_assembly",
                        frame_num=i + 1,
                        step_name="Final Processing",
                        step_progress=i + 1,
                        step_total=len(left_files),
                    )

            return True

        except Exception as e:
            print(f"Error creating VR frames: {e}")
            return False

    def _check_nvenc_available(self) -> bool:
        """Check if NVENC hardware encoding is available."""
        test_result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
        )
        return "hevc_nvenc" in test_result.stdout

    def _build_encoder_cmd(
        self, encoder: str, output_path: Path
    ) -> tuple[list[str], bool]:
        """Build FFmpeg encoder arguments.

        Args:
            encoder: Encoder name (auto, nvenc, libx264, libx265)
            output_path: Output video file path

        Returns:
            Tuple of (encoder_args, is_nvenc_used)
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
                print(
                    "  Warning: NVENC not available, falling back to software encoding"
                )
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

    def _create_output_video(
        self,
        vr_frames_dir: Path,
        output_dir: Path,
        original_video: str,
        settings: Dict[str, Any],
    ) -> bool:
        """Create final output video with audio."""

        if not verify_ffmpeg_installation():
            print("Error: FFmpeg not found. Cannot create output video.")
            return False

        # Generate output filename
        output_filename = generate_output_filename(
            Path(original_video).name, settings["vr_format"], settings["vr_resolution"]
        )
        output_path = output_dir / output_filename

        # Build base FFmpeg command
        base_fps = settings.get("target_fps", 30)
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
            if audio_file.exists():
                cmd.extend(["-i", str(audio_file), "-c:a", "aac", "-shortest"])
            else:
                print(
                    "Warning: Pre-extracted audio not found, extracting from original video"
                )
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
