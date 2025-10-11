"""
Video processor for serial (frame-by-frame) processing mode.

This module handles the traditional frame-by-frame processing approach
with proper progress tracking and error handling.
"""

import cv2
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import time

from ..models.depth_estimator import DepthEstimator
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
    """Handles serial video processing (frame-by-frame)."""
    
    def __init__(self, depth_estimator: DepthEstimator, verbose: bool = False):
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
        Process video in serial mode.
        
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
            
            # Generate batch name from video file and current time
            batch_name = f"{Path(video_path).stem}_{int(time.time())}"
            
            # Save processing settings at the start
            settings_file = save_processing_settings(
                output_path, batch_name, settings, video_properties, video_path
            )
            
            # Extract frames
            frame_files = self._extract_frames(video_path, directories, video_properties, settings)
            if not frame_files:
                print("Error: No frames extracted from video")
                if settings_file:
                    update_processing_status(settings_file, "failed", {
                        "error": "No frames extracted from video"
                    })
                return False
            
            print(f"\n=== Depth Surge 3D Processing Pipeline ===")
            print(f"Input: {video_path}")
            print(f"Output: {output_path}")
            print(f"Total frames: {len(frame_files)}\n")
            print(f"Step 1/7: Extracting frames...")
            print(f"  -> Saved to: {directories.get('frames', 'N/A')}")
            
            # Initialize progress tracker
            if progress_callback:
                # Use the provided callback (from Flask web app)
                progress_tracker = progress_callback
            else:
                # Create default progress tracker for CLI usage
                progress_tracker = create_progress_tracker(len(frame_files), 'serial')
            
            # Step 2/7: Super Sampling (if enabled)
            if settings['super_sample'] != 'none':
                print(f"Step 2/7: Super sampling frames...")
                print(f"  -> Will generate upscaled frames if needed.")
            # Step 3/7: Depth Map Generation
            print(f"Step 3/7: Generating depth maps...")
            print(f"  -> Will save to: {directories.get('depth_maps', 'N/A')}")
            # Step 4/7: Stereo Pair Generation
            print(f"Step 4/7: Creating left/right stereo pairs...")
            print(f"  -> Will save to: {directories.get('left_frames', 'N/A')} and {directories.get('right_frames', 'N/A')}")
            # Step 5/7: Fisheye Distortion (if enabled)
            if settings['apply_distortion']:
                print(f"Step 5/7: Applying fisheye distortion...")
                print(f"  -> Will save to: {directories.get('left_distorted', 'N/A')} and {directories.get('right_distorted', 'N/A')}")
            # Step 6/7: Cropping and Scaling
            print(f"Step 6/7: Cropping and scaling frames...")
            print(f"  -> Will save to: {directories.get('left_cropped', 'N/A')} and {directories.get('right_cropped', 'N/A')}")
            # Step 7/7: VR Frame Assembly
            print(f"Step 7/7: Assembling VR frames...")
            print(f"  -> Will save to: {directories.get('vr_frames', 'N/A')}\n")

            # Process frames with simple progress
            total = len(frame_files)
            for i, frame_file in enumerate(frame_files):
                print(f"Frame {i+1}/{total}", end='\r')
                try:
                    # Load frame
                    image = cv2.imread(str(frame_file))
                    if image is None:
                        print(f"Warning: Could not load frame {frame_file}")
                        continue
                    frame_name = frame_file.stem
                    result = self._process_single_frame(image, frame_name, directories, settings, progress_tracker, i + 1)
                    if not result:
                        print(f"Warning: Failed to process frame {frame_file}")
                        continue
                except Exception as e:
                    print(f"Error processing frame {frame_file}: {e}")
                    return False
            print(f"\nProcessed {total} frames.")
            success = True
            
            if not success:
                if settings_file:
                    update_processing_status(settings_file, "failed", {
                        "error": "Frame processing failed"
                    })
                return False
            
            # Create final video
            print("Creating final video with audio...")
            success = self._create_output_video(
                directories['vr_frames'], output_path, video_path, settings
            )
            
            progress_tracker.finish("Serial processing complete")
            
            if success:
                print(f"Processing complete. Output saved to: {output_path}")
                # Update settings file with completion status
                if settings_file:
                    update_processing_status(settings_file, "completed", {
                        "final_output": str(output_path / generate_output_filename(
                            Path(video_path).name,
                            settings['vr_format'],
                            settings['vr_resolution'],
                            settings['processing_mode']
                        )),
                        "frames_processed": len(frame_files)
                    })
            else:
                if settings_file:
                    update_processing_status(settings_file, "failed", {
                        "error": "Video creation failed"
                    })
            
            return success
            
        except Exception as e:
            print(f"Error in serial video processing: {e}")
            if settings_file:
                update_processing_status(settings_file, "failed", {
                    "error": str(e)
                })
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
    
    def _process_frames_serial(
        self,
        frame_files: List[Path],
        directories: Dict[str, Path],
        settings: Dict[str, Any],
        progress_tracker
    ) -> bool:
        """Process frames in serial mode."""
        
        for i, frame_file in enumerate(frame_files):
            try:
                # Update progress
                progress_tracker.update_progress(f"Loading frame {i+1}", i + 1, phase="extraction")
                
                # Load frame
                image = cv2.imread(str(frame_file))
                if image is None:
                    print(f"Warning: Could not load frame {frame_file}")
                    continue
                
                frame_name = frame_file.stem
                
                # Process frame
                result = self._process_single_frame(image, frame_name, directories, settings, progress_tracker, i + 1)
                if not result:
                    print(f"Warning: Failed to process frame {frame_file}")
                    continue
                
            except Exception as e:
                print(f"Error processing frame {frame_file}: {e}")
                return False
        
        return True
    
    def _process_single_frame(
        self,
        image,
        frame_name: str,
        directories: Dict[str, Path],
        settings: Dict[str, Any],
        progress_tracker,
        frame_num: int
    ) -> bool:
        """Process a single frame through the complete pipeline."""
        
        try:
            # Super sampling if needed
            progress_tracker.update_progress(f"Super sampling frame {frame_num}", frame_num, phase="super_sampling")
            if settings['super_sample'] != 'none':
                target_width = max(image.shape[1], settings['per_eye_width'] * 2)
                target_height = max(image.shape[0], settings['per_eye_height'] * 2)
                image = resize_image(image, target_width, target_height)

            # Generate depth map
            progress_tracker.update_progress(f"Generating depth map for frame {frame_num}", frame_num, phase="depth_estimation")
            depth_map = self.depth_estimator.estimate_depth(image)
            depth_map = normalize_depth_map(depth_map)

            # Save depth map if keeping intermediates
            if settings['keep_intermediates'] and 'depth_maps' in directories:
                depth_vis = (depth_map * 255).astype('uint8')
                cv2.imwrite(str(directories['depth_maps'] / f"{frame_name}.png"), depth_vis)
            
            # Create stereo pair
            progress_tracker.update_progress(f"Creating stereo pair for frame {frame_num}", frame_num, phase="stereo_generation")
            disparity_map = depth_to_disparity(
                depth_map, settings['baseline'], settings['focal_length']
            )
            
            left_img = create_shifted_image(image, disparity_map, "left")
            right_img = create_shifted_image(image, disparity_map, "right")
            
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
            
            # Apply distortion if enabled
            if settings['apply_distortion']:
                progress_tracker.update_progress(f"Applying fisheye distortion to frame {frame_num}", frame_num, phase="distortion")
                
                left_distorted = apply_fisheye_distortion(
                    left_img, settings['fisheye_fov'], settings['fisheye_projection']
                )
                right_distorted = apply_fisheye_distortion(
                    right_img, settings['fisheye_fov'], settings['fisheye_projection']
                )
                
                # Save distorted if keeping intermediates
                if settings['keep_intermediates']:
                    if 'left_distorted' in directories:
                        cv2.imwrite(str(directories['left_distorted'] / f"{frame_name}.png"), left_distorted)
                    if 'right_distorted' in directories:
                        cv2.imwrite(str(directories['right_distorted'] / f"{frame_name}.png"), right_distorted)
                
                # --- FISHEYE CROP FACTOR ---
                fisheye_crop_factor = float(settings.get('fisheye_crop_factor', 1.0))
                fisheye_crop_factor = max(0.7, min(1.5, fisheye_crop_factor))  # Clamp for safety
                left_cropped = apply_fisheye_square_crop(
                    left_distorted, settings['per_eye_width'], settings['per_eye_height'], fisheye_crop_factor
                )
                right_cropped = apply_fisheye_square_crop(
                    right_distorted, settings['per_eye_width'], settings['per_eye_height'], fisheye_crop_factor
                )
                
                # Save cropped frames if keeping intermediates
                if settings['keep_intermediates']:
                    if 'left_cropped' in directories:
                        cv2.imwrite(str(directories['left_cropped'] / f"{frame_name}.png"), left_cropped)
                    if 'right_cropped' in directories:
                        cv2.imwrite(str(directories['right_cropped'] / f"{frame_name}.png"), right_cropped)
                
                # Always resize to per-eye dimensions (safety)
                left_final = resize_image(left_cropped, settings['per_eye_width'], settings['per_eye_height'])
                right_final = resize_image(right_cropped, settings['per_eye_width'], settings['per_eye_height'])
            else:
                # --- CENTER CROP FACTOR ---
                crop_factor = float(settings.get('crop_factor', 1.0))
                crop_factor = max(0.5, min(1.0, crop_factor))  # Clamp for safety
                progress_tracker.update_progress(f"Cropping and scaling frame {frame_num}", frame_num, phase="vr_assembly")
                left_cropped = apply_center_crop(left_img, crop_factor)
                right_cropped = apply_center_crop(right_img, crop_factor)
                
                # Save cropped frames if keeping intermediates
                if settings['keep_intermediates']:
                    if 'left_cropped' in directories:
                        cv2.imwrite(str(directories['left_cropped'] / f"{frame_name}.png"), left_cropped)
                    if 'right_cropped' in directories:
                        cv2.imwrite(str(directories['right_cropped'] / f"{frame_name}.png"), right_cropped)
                
                # Always resize to per-eye dimensions (safety)
                left_final = resize_image(left_cropped, settings['per_eye_width'], settings['per_eye_height'])
                right_final = resize_image(right_cropped, settings['per_eye_width'], settings['per_eye_height'])
            
            # Save final frames if keeping intermediates
            if settings['keep_intermediates']:
                if 'left_final' in directories:
                    cv2.imwrite(str(directories['left_final'] / f"{frame_name}.png"), left_final)
                if 'right_final' in directories:
                    cv2.imwrite(str(directories['right_final'] / f"{frame_name}.png"), right_final)

            # Debug: print per-eye and output dimensions
            if self.verbose:
                print(f"[DEBUG] Frame {frame_num}: left_final {left_final.shape}, right_final {right_final.shape}, per_eye {settings['per_eye_width']}x{settings['per_eye_height']}, crop_factor={settings.get('crop_factor')}, fisheye_crop_factor={settings.get('fisheye_crop_factor')}")

            # Create final VR frame
            vr_frame = create_vr_frame(left_final, right_final, settings['vr_format'])
            # Debug: print VR frame shape
            if self.verbose:
                print(f"[DEBUG] Frame {frame_num}: VR frame shape {vr_frame.shape}")

            # Save VR frame
            if 'vr_frames' in directories:
                cv2.imwrite(str(directories['vr_frames'] / f"{frame_name}.png"), vr_frame)
            
            return True
            
        except Exception as e:
            print(f"Error in frame processing pipeline: {e}")
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
            settings['vr_resolution'],
            settings['processing_mode']
        )
        
        # Add suffix for experimental features
        if settings.get('experimental_frame_interpolation', False):
            name_parts = output_filename.split('.')
            name_parts[0] += '_interpolated'
            output_filename = '.'.join(name_parts)
        
        output_path = output_dir / output_filename
        
        # Build FFmpeg command
        base_fps = settings.get('target_fps', 30)
        # Handle None or 'None' string values
        if base_fps is None or str(base_fps) == 'None' or base_fps == 'original':
            base_fps = 30
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(base_fps),
            '-i', str(vr_frames_dir / 'frame_%06d.png'),
        ]
        
        # Prepare video filters for interpolation
        video_filters = []
        if settings.get('experimental_frame_interpolation', False):
            print("⚠️  Applying experimental frame interpolation...")
            print("   This may introduce artifacts, wobbling, or visual distortions.")
            
            # Double the frame rate using minterpolate filter
            target_fps = base_fps * 2
            video_filters.append(f'minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1')
        
        # Add video filters if any
        if video_filters:
            cmd.extend(['-vf', ','.join(video_filters)])
        
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
                if settings.get('experimental_frame_interpolation', False):
                    print("Note: Frame interpolation may have failed. Try without --experimental-frame-interpolation")
                return False
            
            if settings.get('experimental_frame_interpolation', False):
                print("⚠️  Frame interpolation completed. Review output carefully for artifacts.")
            
            return True
            
        except Exception as e:
            print(f"Error creating output video: {e}")
            return False 