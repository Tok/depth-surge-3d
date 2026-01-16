#!/usr/bin/env python3
"""
Depth Surge 3D - Convert 2D videos to immersive 3D VR format using AI depth estimation.

This is the main entry point using the new modular architecture.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from depth_surge_3d.core.stereo_projector import create_stereo_projector
from depth_surge_3d.core.constants import (
    DEFAULT_SETTINGS,
    VR_RESOLUTIONS,
    FISHEYE_PROJECTIONS,
    HOLE_FILL_METHODS,
    VALIDATION_RANGES,
)
from depth_surge_3d.utils.resolution import get_available_resolutions
from depth_surge_3d.utils.file_operations import validate_video_file, can_resume_processing, load_processing_settings
from depth_surge_3d.utils.console import warning as console_warning


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert 2D videos to immersive 3D VR format using AI depth estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4                                    # Basic conversion
  %(prog)s video.mp4 --vr-resolution 16x9-4k          # High quality 4K
  %(prog)s video.mp4 --vr-resolution custom:2560x1080 # Custom resolution
  %(prog)s --resume ./output/my_video_output/          # Resume previous job
  %(prog)s --list-resolutions                          # Show available resolutions

Note: Always uses Video-Depth-Anything for temporal consistency across frames.
        """,
    )

    # Input/output arguments
    parser.add_argument("input_video", nargs="?", help="Input video file path")

    parser.add_argument(
        "--output-dir", "-o", default=DEFAULT_SETTINGS["output_dir"], help="Output directory (default: %(default)s)"
    )

    # Resume functionality
    parser.add_argument("--resume", metavar="DIRECTORY", help="Resume processing from an existing output directory")

    # VR settings
    available_resolutions = list(VR_RESOLUTIONS.keys()) + ["auto", "custom"]
    parser.add_argument(
        "--vr-resolution",
        choices=available_resolutions,
        default=DEFAULT_SETTINGS["vr_resolution"],
        help="VR output resolution per eye (default: %(default)s)",
    )

    parser.add_argument(
        "-f",
        "--format",
        choices=["side_by_side", "over_under"],
        default=DEFAULT_SETTINGS["vr_format"],
        help="VR output format (default: %(default)s)",
    )

    # Time range
    parser.add_argument("--start", help="Start time (format: HH:MM:SS or seconds)")

    parser.add_argument("--end", help="End time (format: HH:MM:SS or seconds)")

    # Depth and stereo parameters
    parser.add_argument(
        "-b",
        "--baseline",
        type=float,
        default=DEFAULT_SETTINGS["baseline"],
        help=f'Stereo baseline distance in meters (default: {DEFAULT_SETTINGS["baseline"]} - average human IPD)',
    )
    parser.add_argument(
        "--focal-length",
        type=int,
        default=DEFAULT_SETTINGS["focal_length"],
        help=f'Camera focal length in pixels (default: {DEFAULT_SETTINGS["focal_length"]})',
    )

    # Distortion and projection
    parser.add_argument(
        "--fisheye-projection",
        choices=FISHEYE_PROJECTIONS,
        default=DEFAULT_SETTINGS["fisheye_projection"],
        help=f'Fisheye projection type (default: {DEFAULT_SETTINGS["fisheye_projection"]})',
    )
    parser.add_argument(
        "--fisheye-fov",
        type=float,
        default=DEFAULT_SETTINGS["fisheye_fov"],
        help=f'Fisheye field of view in degrees (default: {DEFAULT_SETTINGS["fisheye_fov"]})',
    )
    parser.add_argument(
        "--no-distortion", action="store_true", help="Disable fisheye distortion (keeps rectilinear projection)"
    )

    # Quality and processing options
    parser.add_argument(
        "--crop-factor",
        type=float,
        default=DEFAULT_SETTINGS["crop_factor"],
        help=f'Center crop factor (1.0 = no crop, 0.5 = crop to half) (default: {DEFAULT_SETTINGS["crop_factor"]})',
    )
    parser.add_argument(
        "--fisheye-crop-factor",
        type=float,
        default=DEFAULT_SETTINGS["fisheye_crop_factor"],
        help=f'Fisheye crop factor (default: {DEFAULT_SETTINGS["fisheye_crop_factor"]})',
    )
    parser.add_argument(
        "--hole-fill-quality",
        choices=HOLE_FILL_METHODS,
        default=DEFAULT_SETTINGS["hole_fill_quality"],
        help=f'Hole filling quality (default: {DEFAULT_SETTINGS["hole_fill_quality"]})',
    )

    # Model and device
    parser.add_argument("--model", help="Path to model file (V2) or model name (V3, e.g., 'large', 'base', 'small')")
    parser.add_argument(
        "--depth-model-version",
        choices=["v2", "v3"],
        default="v2",
        help="Depth model version: v2 (Video-Depth-Anything, default) or v3 (Depth-Anything-3, better VRAM efficiency)",
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto", help="Processing device (default: auto)"
    )
    parser.add_argument(
        "--metric", action="store_true", help="Use metric depth model (outputs real depth values in meters)"
    )

    # Output options
    parser.add_argument("--no-audio", action="store_true", help="Do not preserve audio in output")
    parser.add_argument("--no-intermediates", action="store_true", help="Do not keep intermediate processing files")
    parser.add_argument("--target-fps", type=int, help="Target output FPS (default: match source)")

    # Experimental features
    parser.add_argument(
        "--experimental-frame-interpolation",
        action="store_true",
        help="EXPERIMENTAL: Double FPS using motion interpolation. WARNING: May produce artifacts, wobbling, or poor quality. Recommended for artistic experimentation only.",
    )

    # Information and debugging
    parser.add_argument("--list-resolutions", action="store_true", help="List all available VR resolution options")
    parser.add_argument("--model-info", action="store_true", help="Show model information and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    return parser


def validate_arguments(args) -> bool:
    """Validate command line arguments."""

    # Handle resume mode
    if args.resume:
        if not Path(args.resume).exists():
            print(f"Error: Resume directory does not exist: {args.resume}")
            return False
        return True  # Skip other validations for resume mode

    # Regular mode validations
    if not args.input_video:
        print("Error: Input video is required when not resuming")
        return False

    # Validate input video
    if not validate_video_file(args.input_video):
        print(f"Error: Invalid or unsupported video file: {args.input_video}")
        return False

    # Validate ranges
    if args.baseline < VALIDATION_RANGES["baseline"][0] or args.baseline > VALIDATION_RANGES["baseline"][1]:
        print(
            f"Error: Baseline must be between {VALIDATION_RANGES['baseline'][0]} and {VALIDATION_RANGES['baseline'][1]} meters"
        )
        return False

    if (
        args.focal_length < VALIDATION_RANGES["focal_length"][0]
        or args.focal_length > VALIDATION_RANGES["focal_length"][1]
    ):
        print(
            f"Error: Focal length must be between {VALIDATION_RANGES['focal_length'][0]} and {VALIDATION_RANGES['focal_length'][1]} pixels"
        )
        return False

    if args.fisheye_fov < VALIDATION_RANGES["fisheye_fov"][0] or args.fisheye_fov > VALIDATION_RANGES["fisheye_fov"][1]:
        print(
            f"Error: FOV must be between {VALIDATION_RANGES['fisheye_fov'][0]} and {VALIDATION_RANGES['fisheye_fov'][1]} degrees"
        )
        return False

    if args.crop_factor < VALIDATION_RANGES["crop_factor"][0] or args.crop_factor > VALIDATION_RANGES["crop_factor"][1]:
        print(
            f"Error: Crop factor must be between {VALIDATION_RANGES['crop_factor'][0]} and {VALIDATION_RANGES['crop_factor'][1]}"
        )
        return False

    if args.target_fps and (
        args.target_fps < VALIDATION_RANGES["target_fps"][0] or args.target_fps > VALIDATION_RANGES["target_fps"][1]
    ):
        print(
            f"Error: Target FPS must be between {VALIDATION_RANGES['target_fps'][0]} and {VALIDATION_RANGES['target_fps'][1]}"
        )
        return False

    return True


def list_available_resolutions():
    """List all available VR resolution options."""
    print("Available VR Resolution Options:")
    print("=" * 40)

    resolutions = get_available_resolutions()

    for category, items in resolutions.items():
        if items:  # Only show categories with items
            print(f"\n{category.replace('_', ' ').title()}:")
            for item in items:
                print(f"  {item['name']:<15} - {item['description']}")

    print(f"\nCustom Resolution:")
    print(f"  custom:WxH      - Custom resolution (e.g., custom:1920x1080)")
    print(f"\nAuto Detection:")
    print(f"  auto            - Automatically detect optimal resolution")


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Handle special commands
    if args.list_resolutions:
        list_available_resolutions()
        return 0

    # Handle resume mode
    if args.resume:
        print(f"Checking resume capability for: {args.resume}")
        resume_info = can_resume_processing(Path(args.resume))

        if not resume_info["can_resume"]:
            print("Cannot resume processing:")
            for rec in resume_info["recommendations"]:
                print(f"  - {rec}")
            return 1

        print(f"Can resume processing:")
        print(f"  - Batch: {resume_info['batch_name']}")
        print(f"  - Status: {resume_info['status']}")
        if resume_info["progress_info"]:
            progress = resume_info["progress_info"]
            print(f"  - Progress: {progress['frames_processed']} frames processed")

        for rec in resume_info["recommendations"]:
            print(f"  - {rec}")

        # Load settings from the settings file
        settings_data = load_processing_settings(resume_info["settings_file"])
        if not settings_data:
            print("Could not load settings file")
            return 1

        # Extract video path and settings
        video_path = settings_data["metadata"]["source_video"]
        processing_settings = settings_data["processing_settings"]

        # Create projector with original model settings
        projector = create_stereo_projector(device=processing_settings.get("device", "auto"))

        print(f"Resuming processing...")
        print(f"Input: {video_path}")
        print(f"Output: {args.resume}")

        # Resume processing using original settings
        success = projector.process_video(
            video_path=video_path,
            output_dir=args.resume,
            **{k: v for k, v in processing_settings.items() if k not in ["output_dir", "device", "per_eye_width", "video_path", "per_eye_height", "vr_output_width", "vr_output_height",
                                                                         "source_width", "source_height", "source_fps"]},
        )

        if success:
            print("Resume processing completed successfully!")
            return 0
        else:
            print("Resume processing failed. Check error messages above.")
            return 1

    # Validate arguments for normal processing
    if not validate_arguments(args):
        return 1

    # Create stereo projector
    try:
        projector = create_stereo_projector(
            args.model,
            args.device,
            args.metric if hasattr(args, "metric") else False,
            args.depth_model_version,
        )

        if args.model_info:
            info = projector.get_model_info()
            print("Model Information:")
            print("=" * 20)
            for key, value in info.items():
                print(f"{key}: {value}")
            return 0

        # Process video
        model_name = "Depth-Anything-V3" if args.depth_model_version == "v3" else "Video-Depth-Anything (V2)"
        print(f"Starting Depth Surge 3D processing...")
        print(f"Input: {args.input_video}")
        print(f"Output: {args.output_dir} (batch subdirectory will be created)")
        print(f"Model: {model_name}")
        print(f"Format: {args.format}")
        print(f"Resolution: {args.vr_resolution}")

        # Show warning for experimental features
        if args.experimental_frame_interpolation:
            print(console_warning("WARNING: Experimental frame interpolation enabled!"))
            print("   This feature may produce artifacts, wobbling, or poor quality.")
            print("   Recommended for artistic experimentation only.")
            print()

        # Create batch-specific output directory
        if args.resume:
            # Use existing directory for resume
            batch_output_dir = args.resume
        else:
            # Create new batch directory
            from datetime import datetime
            from pathlib import Path

            video_name = Path(args.input_video).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_output_dir = Path(args.output_dir) / f"{video_name}_{timestamp}"

        success = projector.process_video(
            video_path=args.input_video,
            output_dir=str(batch_output_dir),
            vr_format=args.format,
            vr_resolution=args.vr_resolution,
            baseline=args.baseline,
            focal_length=args.focal_length,
            start_time=args.start,
            end_time=args.end,
            apply_distortion=not args.no_distortion,
            fisheye_projection=args.fisheye_projection,
            fisheye_fov=args.fisheye_fov,
            crop_factor=args.crop_factor,
            fisheye_crop_factor=args.fisheye_crop_factor,
            hole_fill_quality=args.hole_fill_quality,
            preserve_audio=not args.no_audio,
            keep_intermediates=not args.no_intermediates,
            target_fps=args.target_fps,
            experimental_frame_interpolation=args.experimental_frame_interpolation,
        )

        if success:
            print("Processing completed successfully!")
            print(f"Output saved to: {batch_output_dir}")
            return 0
        else:
            print("Processing failed. Check error messages above.")
            return 1

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1
    finally:
        # Clean up
        try:
            if "projector" in locals():
                projector.unload_model()
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())
