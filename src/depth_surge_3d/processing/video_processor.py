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
    create_output_directories, get_frame_files, calculate_frame_range,
    generate_output_filename, verify_ffmpeg_installation,
    save_processing_settings, update_processing_status
)
from ..utils.image_processing import (
    resize_image, normalize_depth_map, depth_to_disparity,
    create_shifted_image, apply_center_crop, apply_fisheye_distortion,
    apply_fisheye_square_crop, create_vr_frame, hole_fill_image
)
from ..core.constants import INTERMEDIATE_DIRS


class VideoProcessor:
    """
    Handles video processing with temporal consistency.

    Uses Video-Depth-Anything model to process entire videos with
    temporal consistency for superior depth estimation quality.
    """

    def __init__(self, depth_estimator: VideoDepthEstimator, verbose: bool = False):
        self.depth_estimator = depth_estimator
        self.verbose = verbose

    def process(
        self,
        video_path: str,
        output_dir: str,
        video_properties: Dict[str, Any],
        settings: Dict[str, Any],
        progress_callback=None
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
        settings_file = None

        try:
            output_path = Path(output_dir)

            # Create output directories
            directories = create_output_directories(output_path, settings['keep_intermediates'])

            # Generate batch name
            batch_name = f"{Path(video_path).stem}_{int(time.time())}"

            # Save processing settings
            settings_file = save_processing_settings(
                output_path, batch_name, settings, video_properties, video_path
            )

            print(f"\n=== Depth Surge 3D Video Processing ===")
            print(f"Input: {video_path}")
            print(f"Output: {output_path}")
            print(f"Using Video-Depth-Anything for temporal consistency\n")

            # Step 1: Extract frames
            print("Step 1/7: Extracting frames from video...")
            frame_files = self._extract_frames(video_path, directories, video_properties, settings)
            if not frame_files:
                print("Error: No frames extracted from video")
                if settings_file:
                    update_processing_status(settings_file, "failed", {
                        "error": "No frames extracted from video"
                    })
                return False

            print(f"  -> Extracted {len(frame_files)} frames")
            print(f"  -> Saved to: {directories.get('frames', 'N/A')}\n")

            # Initialize progress tracker
            if progress_callback:
                progress_tracker = progress_callback
            else:
                progress_tracker = create_progress_tracker(len(frame_files), 'batch')

            # Step 2: Generate depth maps (memory-efficient chunked processing)
            print("Step 2/7: Generating depth maps (temporal consistency enabled)...")
            print("  Using memory-efficient chunked processing...")
            progress_tracker.update_progress(
                "Generating depth maps",
                phase="depth_estimation",
                frame_num=0,
                step_name="Depth Map Generation",
                step_progress=0,
                step_total=len(frame_files)
            )

            depth_maps = self._generate_depth_maps_chunked(frame_files, settings, directories, progress_tracker)
            if depth_maps is None:
                print("Error: Failed to generate depth maps")
                if settings_file:
                    update_processing_status(settings_file, "failed", {"error": "Depth map generation failed"})
                return False

            print(f"  -> Generated {len(depth_maps)} depth maps")
            if settings['keep_intermediates'] and 'depth_maps' in directories:
                print(f"  -> Saved to: {directories['depth_maps']}\n")
            else:
                print()

            # Step 3: Load frames for stereo processing
            print("Step 3/7: Loading frames for stereo processing...")
            progress_tracker.update_progress(
                "Loading frames",
                phase="depth_estimation",
                frame_num=0,
                step_name="Frame Extraction",
                step_progress=0,
                step_total=len(frame_files)
            )

            frames = self._load_frames(frame_files, settings, progress_tracker)
            if frames is None or len(frames) == 0:
                print("Error: Failed to load frames")
                if settings_file:
                    update_processing_status(settings_file, "failed", {"error": "Failed to load frames"})
                return False

            print(f"  -> Loaded {len(frames)} frames\n")

            # Step 4: Create stereo pairs
            print("Step 4/7: Creating stereo pairs...")
            success = self._create_stereo_pairs(
                frames, depth_maps, frame_files, directories, settings, progress_tracker
            )
            if not success:
                if settings_file:
                    update_processing_status(settings_file, "failed", {"error": "Stereo pair creation failed"})
                return False
            print(f"  -> Created stereo pairs for {len(frames)} frames")
            if settings['keep_intermediates'] and 'left_frames' in directories:
                print(f"  -> Saved to: {directories['left_frames']} & {directories['right_frames']}\n")
            else:
                print()

            # Step 5: Apply fisheye distortion (if enabled)
            if settings['apply_distortion']:
                print("Step 5/7: Applying fisheye distortion...")
                left_files = sorted(directories['left_frames'].glob('*.png')) if 'left_frames' in directories else []
                right_files = sorted(directories['right_frames'].glob('*.png')) if 'right_frames' in directories else []
                success = self._apply_distortion(
                    left_files, right_files, directories, settings, progress_tracker
                )
                if not success:
                    if settings_file:
                        update_processing_status(settings_file, "failed", {"error": "Distortion failed"})
                    return False
                print(f"  -> Applied fisheye distortion to {len(left_files)} frames")
                if settings['keep_intermediates'] and 'left_distorted' in directories:
                    print(f"  -> Saved to: {directories['left_distorted']} & {directories['right_distorted']}\n")
                else:
                    print()
            else:
                print("Step 5/7: Skipping fisheye distortion (disabled)\n")

            # Step 6: Create final VR frames
            print("Step 6/7: Creating final VR frames...")
            success = self._create_vr_frames(
                directories, settings, progress_tracker, len(frames)
            )
            if not success:
                if settings_file:
                    update_processing_status(settings_file, "failed", {"error": "VR frame creation failed"})
                return False
            print(f"  -> Created {len(frames)} VR frames")
            if 'vr_frames' in directories:
                print(f"  -> Saved to: {directories['vr_frames']}\n")
            else:
                print()

            # Step 7: Create final video
            print("Step 7/7: Creating final video with audio...")
            success = self._create_output_video(
                directories['vr_frames'], output_path, video_path, settings
            )

            if success:
                output_filename = generate_output_filename(
                    Path(video_path).name,
                    settings['vr_format'],
                    settings['vr_resolution']
                )
                print(f"  -> Created final video: {output_filename}")
                print(f"  -> Saved to: {output_path / output_filename}\n")

            progress_tracker.finish("Video processing complete")

            if success:
                print(f"✓ Processing complete!")
                if settings_file:
                    update_processing_status(settings_file, "completed", {
                        "final_output": str(output_path / generate_output_filename(
                            Path(video_path).name,
                            settings['vr_format'],
                            settings['vr_resolution']
                        )),
                        "frames_processed": len(frames)
                    })
            else:
                if settings_file:
                    update_processing_status(settings_file, "failed", {"error": "Video creation failed"})

            return success

        except Exception as e:
            print(f"Error in video processing: {e}")
            if settings_file:
                update_processing_status(settings_file, "failed", {"error": str(e)})
            return False

    def _extract_frames(
        self,
        video_path: str,
        directories: Dict[str, Path],
        video_properties: Dict[str, Any],
        settings: Dict[str, Any]
    ) -> List[Path]:
        """Extract frames from video."""
        frames_dir = directories.get('frames')
        if not frames_dir:
            frames_dir = directories['base'] / INTERMEDIATE_DIRS['frames']
            frames_dir.mkdir(exist_ok=True)

        # Calculate frame range
        total_frames = video_properties['frame_count']
        fps = video_properties['fps']
        start_frame, end_frame = calculate_frame_range(
            total_frames, fps, settings.get('start_time'), settings.get('end_time')
        )

        # Build FFmpeg command
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vf', f'select=between(n\\,{start_frame}\\,{end_frame-1})',
            '-vsync', '0',
            str(frames_dir / 'frame_%06d.png')
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return []
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return []

        return get_frame_files(frames_dir)

    def _load_frames(
        self,
        frame_files: List[Path],
        settings: Dict[str, Any],
        progress_tracker
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
                if settings['super_sample'] != 'none':
                    target_width = max(image.shape[1], settings['per_eye_width'] * 2)
                    target_height = max(image.shape[0], settings['per_eye_height'] * 2)
                    image = resize_image(image, target_width, target_height)

                frames_list.append(image)

                # Update progress
                if i % 10 == 0 or i == len(frame_files) - 1:
                    progress_tracker.update_progress(
                        "Loading frames",
                        phase="depth_estimation",
                        frame_num=i + 1,
                        step_name="Frame Extraction",
                        step_progress=i + 1,
                        step_total=len(frame_files)
                    )

            except Exception as e:
                print(f"Error loading frame {frame_file}: {e}")
                return None

        if not frames_list:
            return None

        return np.array(frames_list)

    def _generate_depth_maps_chunked(
        self,
        frame_files: List[Path],
        settings: Dict[str, Any],
        directories: Dict[str, Path],
        progress_tracker
    ) -> Optional[np.ndarray]:
        """
        Generate depth maps in memory-efficient chunks.

        Processes frames in small batches to avoid CUDA OOM errors.
        """
        # Determine chunk size based on resolution
        sample_frame = cv2.imread(str(frame_files[0]))
        if sample_frame is None:
            return None

        frame_h, frame_w = sample_frame.shape[:2]
        megapixels = (frame_h * frame_w) / 1_000_000

        print(f"  Frame resolution: {frame_w}x{frame_h} ({megapixels:.1f}MP)")

        # Very aggressive chunking for limited VRAM
        # Model uses ~9GB, only ~6GB free on typical 16GB GPU
        if megapixels > 8.0:  # >8MP (4K is ~8.3MP)
            chunk_size = 4   # Ultra-tiny chunks for 4K
            input_size = 384  # Reduce depth model input resolution
        elif megapixels > 2.0:  # >2MP (1080p is 2.1MP)
            chunk_size = 8  # Very small chunks for HD
            input_size = 384  # Also reduce for HD
        elif megapixels > 1.0:  # >1MP (720p is 0.9MP)
            chunk_size = 16  # Small chunks for 720p
            input_size = 448
        else:
            chunk_size = 32  # Standard chunks for SD
            input_size = 518

        print(f"  Processing in chunks of {chunk_size} frames (input_size={input_size})...")

        # Clear GPU cache before processing to maximize available VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            mem_free = torch.cuda.mem_get_info()[0] / (1024**3)  # Convert to GB
            print(f"  GPU memory freed: {mem_free:.2f} GB available")

        all_depth_maps = []
        num_frames = len(frame_files)

        for chunk_start in range(0, num_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_frames)
            chunk_files = frame_files[chunk_start:chunk_end]

            print(f"  Chunk {chunk_start//chunk_size + 1}: frames {chunk_start+1}-{chunk_end}/{num_frames}")

            # Load chunk of frames
            chunk_frames = []
            for frame_file in chunk_files:
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    print(f"Warning: Could not load {frame_file}")
                    continue

                # Apply super sampling if needed
                if settings['super_sample'] != 'none':
                    target_width = max(frame.shape[1], settings['per_eye_width'] * 2)
                    target_height = max(frame.shape[0], settings['per_eye_height'] * 2)
                    frame = resize_image(frame, target_width, target_height)

                chunk_frames.append(frame)

            if not chunk_frames:
                print(f"Error: No frames loaded in chunk")
                return None

            # Process chunk for depth
            try:
                target_fps = settings.get('target_fps', 30)
                if target_fps is None or str(target_fps) == 'None' or target_fps == 'original':
                    target_fps = 30

                chunk_frames_array = np.array(chunk_frames)
                chunk_depth_maps = self.depth_estimator.estimate_depth_batch(
                    chunk_frames_array,
                    target_fps=target_fps,
                    input_size=input_size,
                    fp32=False
                )

                all_depth_maps.extend(chunk_depth_maps)

                # Save depth maps immediately to free memory
                if settings['keep_intermediates'] and 'depth_maps' in directories:
                    depth_dir = directories['depth_maps']
                    for i, (depth_map, frame_file) in enumerate(zip(chunk_depth_maps, chunk_files)):
                        depth_vis = (depth_map * 255).astype('uint8')
                        frame_name = frame_file.stem
                        cv2.imwrite(str(depth_dir / f"{frame_name}.png"), depth_vis)

                # Clear references to free memory
                del chunk_frames
                del chunk_frames_array
                del chunk_depth_maps

                # Clear GPU cache between chunks
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Update progress
                progress_tracker.update_progress(
                    f"Depth maps: {chunk_end}/{num_frames}",
                    phase="depth_estimation",
                    frame_num=chunk_end,
                    step_name="Depth Map Generation",
                    step_progress=chunk_end,
                    step_total=num_frames
                )

            except Exception as e:
                print(f"Error processing chunk {chunk_start}-{chunk_end}: {e}")
                return None

        return np.array(all_depth_maps)

    def _generate_depth_maps_batch(
        self,
        frames: np.ndarray,
        settings: Dict[str, Any],
        progress_tracker
    ) -> Optional[np.ndarray]:
        """Generate depth maps for all frames with temporal consistency."""
        try:
            # Use Video-Depth-Anything for temporal consistency
            target_fps = settings.get('target_fps', 30)
            if target_fps is None or str(target_fps) == 'None' or target_fps == 'original':
                target_fps = 30

            depth_maps = self.depth_estimator.estimate_depth_batch(
                frames,
                target_fps=target_fps,
                input_size=518,
                fp32=False
            )

            return depth_maps

        except Exception as e:
            print(f"Error generating depth maps: {e}")
            return None

    def _save_depth_maps(
        self,
        depth_maps: np.ndarray,
        frame_files: List[Path],
        depth_dir: Path
    ):
        """Save depth maps to disk."""
        for i, (depth_map, frame_file) in enumerate(zip(depth_maps, frame_files)):
            depth_vis = (depth_map * 255).astype('uint8')
            frame_name = frame_file.stem
            cv2.imwrite(str(depth_dir / f"{frame_name}.png"), depth_vis)

    def _create_stereo_pairs(
        self,
        frames: np.ndarray,
        depth_maps: np.ndarray,
        frame_files: List[Path],
        directories: Dict[str, Path],
        settings: Dict[str, Any],
        progress_tracker
    ) -> bool:
        """Create stereo pairs from frames and depth maps."""
        try:
            for i, (frame, depth_map, frame_file) in enumerate(zip(frames, depth_maps, frame_files)):
                frame_name = frame_file.stem

                # Create stereo pair
                disparity_map = depth_to_disparity(
                    depth_map, settings['baseline'], settings['focal_length']
                )

                left_img = create_shifted_image(frame, disparity_map, "left")
                right_img = create_shifted_image(frame, disparity_map, "right")

                # Apply hole filling
                if settings['hole_fill_quality'] in ['fast', 'advanced']:
                    left_img = hole_fill_image(left_img, method=settings['hole_fill_quality'])
                    right_img = hole_fill_image(right_img, method=settings['hole_fill_quality'])

                # Save stereo pair if keeping intermediates
                if settings['keep_intermediates']:
                    if 'left_frames' in directories:
                        cv2.imwrite(str(directories['left_frames'] / f"{frame_name}.png"), left_img)
                    if 'right_frames' in directories:
                        cv2.imwrite(str(directories['right_frames'] / f"{frame_name}.png"), right_img)

                # Update progress
                if i % 5 == 0 or i == len(frames) - 1:
                    progress_tracker.update_progress(
                        "Creating stereo pairs",
                        phase="stereo_generation",
                        frame_num=i + 1,
                        step_name="Stereo Pair Creation",
                        step_progress=i + 1,
                        step_total=len(frames)
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
        progress_tracker
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
                    left_img, settings['fisheye_fov'], settings['fisheye_projection']
                )
                right_distorted = apply_fisheye_distortion(
                    right_img, settings['fisheye_fov'], settings['fisheye_projection']
                )

                # Save distorted frames if keeping intermediates
                if settings['keep_intermediates']:
                    frame_name = left_file.stem
                    if 'left_distorted' in directories:
                        cv2.imwrite(str(directories['left_distorted'] / f"{frame_name}.png"), left_distorted)
                    if 'right_distorted' in directories:
                        cv2.imwrite(str(directories['right_distorted'] / f"{frame_name}.png"), right_distorted)

                # Update progress
                if i % 5 == 0 or i == len(left_files) - 1:
                    progress_tracker.update_progress(
                        "Applying distortion",
                        phase="distortion",
                        frame_num=i + 1,
                        step_name="Fisheye Distortion",
                        step_progress=i + 1,
                        step_total=len(left_files)
                    )

            return True

        except Exception as e:
            print(f"Error applying distortion: {e}")
            return False

    def _create_vr_frames(
        self,
        directories: Dict[str, Path],
        settings: Dict[str, Any],
        progress_tracker,
        total_frames: int
    ) -> bool:
        """Create final VR frames from processed left/right frames."""
        try:
            # Determine source directory based on whether distortion was applied
            if settings['apply_distortion'] and 'left_distorted' in directories:
                left_dir = directories['left_distorted']
                right_dir = directories['right_distorted']
            elif 'left_frames' in directories:
                left_dir = directories['left_frames']
                right_dir = directories['right_frames']
            else:
                print("Error: No stereo frames found")
                return False

            # Get frame files
            left_files = sorted(left_dir.glob('*.png'))
            right_files = sorted(right_dir.glob('*.png'))

            if len(left_files) != len(right_files):
                print(f"Warning: Mismatched frame count: {len(left_files)} left, {len(right_files)} right")

            for i, (left_file, right_file) in enumerate(zip(left_files, right_files)):
                # Load images
                left_img = cv2.imread(str(left_file))
                right_img = cv2.imread(str(right_file))

                if left_img is None or right_img is None:
                    print(f"Warning: Could not load {left_file} or {right_file}")
                    continue

                # Apply cropping and resizing
                if settings['apply_distortion']:
                    fisheye_crop_factor = float(settings.get('fisheye_crop_factor', 1.0))
                    fisheye_crop_factor = max(0.7, min(1.5, fisheye_crop_factor))

                    left_cropped = apply_fisheye_square_crop(
                        left_img, settings['per_eye_width'], settings['per_eye_height'], fisheye_crop_factor
                    )
                    right_cropped = apply_fisheye_square_crop(
                        right_img, settings['per_eye_width'], settings['per_eye_height'], fisheye_crop_factor
                    )

                    left_final = resize_image(left_cropped, settings['per_eye_width'], settings['per_eye_height'])
                    right_final = resize_image(right_cropped, settings['per_eye_width'], settings['per_eye_height'])
                else:
                    crop_factor = float(settings.get('crop_factor', 1.0))
                    crop_factor = max(0.5, min(1.0, crop_factor))

                    left_cropped = apply_center_crop(left_img, crop_factor)
                    right_cropped = apply_center_crop(right_img, crop_factor)

                    left_final = resize_image(left_cropped, settings['per_eye_width'], settings['per_eye_height'])
                    right_final = resize_image(right_cropped, settings['per_eye_width'], settings['per_eye_height'])

                # Create final VR frame
                vr_frame = create_vr_frame(left_final, right_final, settings['vr_format'])

                # Save VR frame
                if 'vr_frames' in directories:
                    frame_name = left_file.stem
                    cv2.imwrite(str(directories['vr_frames'] / f"{frame_name}.png"), vr_frame)

                # Update progress
                if i % 5 == 0 or i == len(left_files) - 1:
                    progress_tracker.update_progress(
                        "Creating VR frames",
                        phase="vr_assembly",
                        frame_num=i + 1,
                        step_name="Final Processing",
                        step_progress=i + 1,
                        step_total=len(left_files)
                    )

            return True

        except Exception as e:
            print(f"Error creating VR frames: {e}")
            return False

    def _create_output_video(
        self,
        vr_frames_dir: Path,
        output_dir: Path,
        original_video: str,
        settings: Dict[str, Any]
    ) -> bool:
        """Create final output video with audio."""

        if not verify_ffmpeg_installation():
            print("Error: FFmpeg not found. Cannot create output video.")
            return False

        # Generate output filename
        output_filename = generate_output_filename(
            Path(original_video).name,
            settings['vr_format'],
            settings['vr_resolution']
        )

        output_path = output_dir / output_filename

        # Build FFmpeg command
        base_fps = settings.get('target_fps', 30)
        if base_fps is None or str(base_fps) == 'None' or base_fps == 'original':
            base_fps = 30

        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(base_fps),
            '-i', str(vr_frames_dir / 'frame_%06d.png'),
        ]

        # Add audio if preserving
        if settings.get('preserve_audio', True):
            cmd.extend(['-i', original_video, '-c:a', 'aac', '-shortest'])

        # Video encoding settings
        cmd.extend([
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',  # High quality
            str(output_path)
        ])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return False
            return True
        except Exception as e:
            print(f"Error creating output video: {e}")
            return False
