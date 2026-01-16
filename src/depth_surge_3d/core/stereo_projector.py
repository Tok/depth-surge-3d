"""
Main StereoProjector class for 2D to 3D VR conversion.

This module provides the main orchestration class that coordinates all
processing steps using the modular utility functions.
"""

import cv2
import numpy as np
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from ..models.video_depth_estimator import create_video_depth_estimator
from ..models.video_depth_estimator_da3 import create_video_depth_estimator_da3
from ..utils.resolution import (
    get_resolution_dimensions,
    calculate_vr_output_dimensions,
    validate_resolution_settings,
    auto_detect_resolution,
)
from ..utils.file_operations import (
    validate_video_file,
    get_video_properties,
)
from ..processing.video_processor import VideoProcessor
from ..core.constants import DEFAULT_SETTINGS


class StereoProjector:
    """
    Main class for converting 2D videos to 3D VR format.

    Uses Video-Depth-Anything for temporal consistency across video frames.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        metric: bool = False,
        depth_model_version: str = "v2",
    ):
        """
        Initialize StereoProjector.

        Args:
            model_path: Path to video depth estimation model (DA2) or model name (DA3)
            device: Processing device ('auto', 'cuda', 'cpu')
            metric: Use metric depth model (true depth values)
            depth_model_version: Depth model version ('v2' or 'v3', default: 'v2')
        """
        self.depth_model_version = depth_model_version

        if depth_model_version == "v3":
            # Use Depth Anything V3 (model_path is used as model_name)
            model_name = model_path if model_path else None
            self.depth_estimator = create_video_depth_estimator_da3(
                model_name, device, metric
            )
        else:
            # Use Video-Depth-Anything V2 (default)
            self.depth_estimator = create_video_depth_estimator(
                model_path, device, metric
            )

        self._model_loaded = False

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        vr_format: str = None,
        baseline: float = None,
        focal_length: float = None,
        keep_intermediates: bool = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        preserve_audio: bool = None,
        target_fps: Optional[int] = None,
        min_resolution: str = None,
        super_sample: str = None,
        apply_distortion: bool = None,
        fisheye_projection: str = None,
        fisheye_fov: float = None,
        crop_factor: float = None,
        vr_resolution: str = None,
        fisheye_crop_factor: float = None,
        hole_fill_quality: str = None,
        processing_mode: str = None,
        experimental_frame_interpolation: bool = None,
    ) -> bool:
        """
        Process video to create 3D VR version.

        Args:
            video_path: Path to input video
            output_dir: Output directory path
            **kwargs: Processing parameters (defaults from DEFAULT_SETTINGS)

        Returns:
            True if processing completed successfully
        """
        # Apply defaults for None values
        settings = self._apply_default_settings(locals())

        try:
            # Validate inputs
            if not self._validate_inputs(video_path, output_dir, settings):
                return False

            # Ensure model is loaded
            if not self._ensure_model_loaded():
                return False

            # Get video properties
            video_props = get_video_properties(video_path)
            if not video_props:
                print(f"Error: Cannot read video properties from {video_path}")
                return False

            # Validate and resolve settings
            resolved_settings = self._resolve_settings(settings, video_props)

            # Create video processor (always uses temporal consistency)
            processor = VideoProcessor(
                self.depth_estimator, verbose=resolved_settings.get("verbose", False)
            )

            # Process the video
            return processor.process(
                video_path=video_path,
                output_dir=output_dir,
                video_properties=video_props,
                settings=resolved_settings,
            )

        except Exception as e:
            print(f"Error during video processing: {e}")
            return False

    def process_image(self, image_path: str, output_dir: str, **kwargs) -> bool:
        """
        Process single image to create 3D stereo pair.

        NOTE: Video-Depth-Anything is optimized for videos. For best results,
        convert your image to a short video clip first.

        Args:
            image_path: Path to input image
            output_dir: Output directory path
            **kwargs: Processing parameters

        Returns:
            True if processing completed successfully
        """
        print(
            "WARNING: Single image processing is not optimized with Video-Depth-Anything."
        )
        print("For best results, convert your image to a video first.")
        print("This feature will process the image as a single-frame video.")

        settings = self._apply_default_settings(kwargs)

        try:
            # Ensure model is loaded
            if not self._ensure_model_loaded():
                return False

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Cannot load image from {image_path}")
                return False

            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Process as single-frame video
            frames = np.array([image])  # Shape: [1, H, W, 3]

            # Get depth map using video model
            # Higher input_size = better quality but more VRAM
            # Auto-detect or use user-specified depth resolution
            depth_resolution = settings.get("depth_resolution", "auto")
            if depth_resolution == "auto":
                # Auto: match image size (never exceed source resolution)
                input_size = max(image.shape[0], image.shape[1])
            else:
                try:
                    input_size = int(depth_resolution)
                except (ValueError, TypeError):
                    input_size = 1080  # fallback to common resolution

            depth_maps = self.depth_estimator.estimate_depth_batch(
                frames, target_fps=30, input_size=input_size, fp32=False
            )

            if depth_maps is None or len(depth_maps) == 0:
                print("Error: Failed to generate depth map")
                return False

            depth_map = depth_maps[0]

            # Process using simplified pipeline
            from ..utils.image_processing import (
                resize_image,
                depth_to_disparity,
                create_shifted_image,
                apply_center_crop,
                create_vr_frame,
                hole_fill_image,
            )

            per_eye_width = settings.get("per_eye_width", 1920)
            per_eye_height = settings.get("per_eye_height", 1080)

            # Create stereo pair
            disparity_map = depth_to_disparity(
                depth_map, settings["baseline"], settings["focal_length"]
            )

            left_img = create_shifted_image(image, disparity_map, "left")
            right_img = create_shifted_image(image, disparity_map, "right")

            # Apply hole filling
            if settings["hole_fill_quality"] in ["fast", "advanced"]:
                left_img = hole_fill_image(
                    left_img, method=settings["hole_fill_quality"]
                )
                right_img = hole_fill_image(
                    right_img, method=settings["hole_fill_quality"]
                )

            # Apply center cropping
            left_cropped = apply_center_crop(left_img, settings["crop_factor"])
            right_cropped = apply_center_crop(right_img, settings["crop_factor"])

            # Resize to target dimensions
            left_final = resize_image(left_cropped, per_eye_width, per_eye_height)
            right_final = resize_image(right_cropped, per_eye_width, per_eye_height)

            # Create VR frame
            vr_frame = create_vr_frame(left_final, right_final, settings["vr_format"])

            # Save results
            base_name = Path(image_path).stem
            cv2.imwrite(str(output_path / f"{base_name}_left.png"), left_final)
            cv2.imwrite(str(output_path / f"{base_name}_right.png"), right_final)
            cv2.imwrite(str(output_path / f"{base_name}_vr.png"), vr_frame)
            cv2.imwrite(
                str(output_path / f"{base_name}_depth.png"),
                (depth_map * 255).astype("uint8"),
            )

            print(f"Image processing complete. Output saved to: {output_path}")
            return True

        except Exception as e:
            print(f"Error during image processing: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _apply_default_settings(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default settings for None parameters."""
        settings = {}
        for key, default_value in DEFAULT_SETTINGS.items():
            # Get value from params, excluding 'self' and 'video_path', 'output_dir'
            if key in params and params[key] is not None:
                settings[key] = params[key]
            else:
                settings[key] = default_value

        # Handle special cases
        special_params = [
            "video_path",
            "output_dir",
            "start_time",
            "end_time",
            "target_fps",
            "min_resolution",
        ]
        for param in special_params:
            if param in params:
                settings[param] = params[param]

        return settings

    def _validate_inputs(
        self, video_path: str, output_dir: str, settings: Dict[str, Any]
    ) -> bool:
        """Validate input parameters."""
        # Validate video file
        if not validate_video_file(video_path):
            print(f"Error: Invalid or unsupported video file: {video_path}")
            return False

        # Validate output directory
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error: Cannot create output directory {output_dir}: {e}")
            return False

        return True

    def _ensure_model_loaded(self) -> bool:
        """Ensure the depth estimation model is loaded."""
        if not self._model_loaded:
            if self.depth_estimator.load_model():
                self._model_loaded = True
            else:
                return False
        return True

    def _resolve_settings(
        self, settings: Dict[str, Any], video_props: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve and validate settings based on video properties."""
        resolved = settings.copy()

        # Resolve VR resolution
        if resolved["vr_resolution"] == "auto":
            resolved["vr_resolution"] = auto_detect_resolution(
                video_props["width"], video_props["height"], resolved["vr_format"]
            )

        # Validate resolution settings
        validation = validate_resolution_settings(
            resolved["vr_resolution"],
            resolved["vr_format"],
            video_props["width"],
            video_props["height"],
        )

        if not validation["valid"]:
            print("Warning: Invalid resolution settings")
            for warning in validation["warnings"]:
                print(f"  - {warning}")

        for recommendation in validation["recommendations"]:
            print(f"Recommendation: {recommendation}")

        # Get final resolution dimensions
        per_eye_width, per_eye_height = get_resolution_dimensions(
            resolved["vr_resolution"]
        )
        vr_output_width, vr_output_height = calculate_vr_output_dimensions(
            per_eye_width, per_eye_height, resolved["vr_format"]
        )

        resolved.update(
            {
                "per_eye_width": per_eye_width,
                "per_eye_height": per_eye_height,
                "vr_output_width": vr_output_width,
                "vr_output_height": vr_output_height,
                "source_width": video_props["width"],
                "source_height": video_props["height"],
                "source_fps": video_props["fps"],
            }
        )

        return resolved

    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        target_fps: Optional[str] = None,
        extraction_mode: str = "original",
    ) -> List[Path]:
        """
        Extract frames from video for processing.

        Args:
            video_path: Path to input video
            output_dir: Output directory path
            start_time: Start time (e.g., "00:30")
            end_time: End time (e.g., "01:30")
            target_fps: Target FPS (currently unused, extraction uses original fps)
            extraction_mode: Extraction mode (currently unused)

        Returns:
            List of extracted frame file paths
        """
        from ..utils.file_operations import get_video_properties

        # Get video properties
        video_props = get_video_properties(video_path)
        if not video_props:
            raise ValueError(f"Could not read video properties from {video_path}")

        # Create frames directory
        output_path = Path(output_dir)
        frames_dir = output_path / "00_original_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Build FFmpeg command for frame extraction with CUDA acceleration
        # Try CUDA first, fall back to CPU if unavailable
        cmd = [
            "ffmpeg",
            "-y",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-i",
            video_path,
        ]

        # Add time range if specified
        if start_time:
            cmd.extend(["-ss", start_time])
        if end_time:
            cmd.extend(["-to", end_time])

        # Extract frames as PNG
        output_pattern = str(frames_dir / "frame_%06d.png")
        cmd.extend(["-vsync", "0", output_pattern])

        # Run FFmpeg with CUDA, fall back to CPU if it fails
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                # CUDA failed, try CPU fallback
                print("  CUDA frame extraction failed, falling back to CPU")
                cmd_cpu = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                ]
                if start_time:
                    cmd_cpu.extend(["-ss", start_time])
                if end_time:
                    cmd_cpu.extend(["-to", end_time])
                cmd_cpu.extend(["-vsync", "0", output_pattern])
                subprocess.run(cmd_cpu, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg frame extraction failed: {e.stderr}")

        # Get list of extracted frames
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        return frame_files

    def determine_super_sample_resolution(
        self, original_width: int, original_height: int, super_sample: str = "auto"
    ) -> Tuple[int, int]:
        """
        Determine super sampling resolution for better quality.

        Args:
            original_width: Original video width
            original_height: Original video height
            super_sample: Super sampling mode ('auto', 'none', '1080p', '4k')

        Returns:
            Tuple of (width, height) for super sampling
        """
        if super_sample == "none":
            return original_width, original_height
        elif super_sample == "1080p":
            return 1920, 1080
        elif super_sample == "4k":
            return 3840, 2160
        elif super_sample == "auto":
            # Auto: 720p->1080p, 1080p->4K, others keep original
            if original_height <= 720:
                return 1920, 1080
            elif original_height <= 1080:
                return 3840, 2160
            else:
                return original_width, original_height
        else:
            return original_width, original_height

    def determine_vr_output_resolution(
        self,
        original_width: int,
        original_height: int,
        vr_resolution: str = "auto",
        vr_format: str = "side_by_side",
    ) -> Tuple[int, int]:
        """
        Determine VR output resolution.

        Args:
            original_width: Original video width
            original_height: Original video height
            vr_resolution: VR resolution setting
            vr_format: VR format ('side_by_side', 'over_under')

        Returns:
            Tuple of (width, height) for VR output
        """
        from ..utils.resolution import (
            get_resolution_dimensions,
            calculate_vr_output_dimensions,
        )

        if vr_resolution == "auto":
            # Use original resolution as per-eye resolution
            per_eye_width, per_eye_height = original_width, original_height
        else:
            # Get dimensions from resolution string
            per_eye_width, per_eye_height = get_resolution_dimensions(vr_resolution)

        # Calculate final VR dimensions based on format
        return calculate_vr_output_dimensions(per_eye_width, per_eye_height, vr_format)

    def _check_nvenc_available(self) -> bool:
        """Check if NVENC hardware encoding is available."""
        try:
            test_result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
            )
            return "hevc_nvenc" in test_result.stdout
        except Exception:
            return False

    def _add_video_encoder_options(self, cmd: list) -> None:
        """Add appropriate video encoder options to FFmpeg command.

        Args:
            cmd: FFmpeg command list to append encoder options to
        """
        if self._check_nvenc_available():
            print("  Using NVENC hardware encoding (H.265)")
            cmd.extend(
                [
                    "-c:v",
                    "hevc_nvenc",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    "p7",
                    "-tune",
                    "hq",
                ]
            )
        else:
            print("  Using software encoding (H.264)")
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "18",
                    "-preset",
                    "medium",
                ]
            )

    def create_output_video(
        self,
        vr_frames_dir: str,
        output_path: str,
        original_video_path: str,
        vr_format: str = "side_by_side",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        preserve_audio: bool = True,
        target_fps: Optional[str] = None,
    ) -> bool:
        """
        Create final output video from VR frames.

        Args:
            vr_frames_dir: Directory containing VR frames
            output_path: Output video file path
            original_video_path: Path to original video (for audio extraction)
            vr_format: VR format ('side_by_side', 'over_under')
            start_time: Start time for audio sync
            end_time: End time for audio sync
            preserve_audio: Whether to include audio
            target_fps: Target FPS for output video

        Returns:
            True if successful, False otherwise
        """
        vr_frames_path = Path(vr_frames_dir)

        # Get list of VR frames
        vr_frame_files = sorted(vr_frames_path.glob("*.png"))
        if not vr_frame_files:
            raise ValueError(f"No VR frames found in {vr_frames_dir}")

        # Determine frame rate
        if target_fps and target_fps != "original" and str(target_fps) != "None":
            fps_value = str(target_fps)
        else:
            fps_value = "30"  # Default fallback

        # Build base FFmpeg command with input
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            fps_value,
            "-i",
            str(vr_frames_path / "frame_%06d.png"),
        ]

        # Add audio if requested
        if preserve_audio:
            cmd.extend(["-i", original_video_path])
            if start_time:
                cmd.extend(["-ss", start_time])
            if end_time:
                cmd.extend(["-to", end_time])
            cmd.extend(["-c:a", "aac", "-shortest"])

        # Add video codec options
        self._add_video_encoder_options(cmd)

        # Add output path
        cmd.append(output_path)

        # Run FFmpeg
        try:
            print(f"Running FFmpeg command: {' '.join(cmd)}")
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg video creation failed: {e.stderr}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return self.depth_estimator.get_model_info()

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        self.depth_estimator.unload_model()
        self._model_loaded = False


def create_stereo_projector(
    model_path: Optional[str] = None,
    device: str = "auto",
    metric: bool = False,
    depth_model_version: str = "v2",
) -> StereoProjector:
    """
    Factory function to create a StereoProjector instance.

    Args:
        model_path: Path to model file (V2) or model name (V3)
        device: Processing device
        metric: Use metric depth model (true depth values)
        depth_model_version: Depth model version ('v2' or 'v3')

    Returns:
        Configured StereoProjector instance
    """
    return StereoProjector(model_path, device, metric, depth_model_version)
