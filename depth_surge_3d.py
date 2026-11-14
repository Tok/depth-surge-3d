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

from depth_surge_3d.rendering import create_stereo_projector  # noqa: E402
from depth_surge_3d.core.constants import (  # noqa: E402
    DEFAULT_SETTINGS,
    VR_RESOLUTIONS,
    FISHEYE_PROJECTIONS,
    HOLE_FILL_METHODS,
    VALIDATION_RANGES,
)
from depth_surge_3d.utils import (  # noqa: E402
    get_available_resolutions,
    parse_custom_resolution,
    warning as console_warning,
)
from depth_surge_3d.io.operations import (  # noqa: E402
    validate_video_file,
    can_resume_processing,
    load_processing_settings,
)
from depth_surge_3d.core.constants import PROCESSING_MODES, BATCH_SIZE_RANGE  # noqa: E402
from depth_surge_3d.processing import assemble_partial_video  # noqa: E402
from depth_surge_3d.utils.domain.progress import TqdmProgressTracker  # noqa: E402

# Saved settings that are recomputed on every run (not passed back in when resuming)
_DERIVED_SETTINGS = {
    "output_dir",
    "device",
    "video_path",
    "per_eye_width",
    "per_eye_height",
    "vr_output_width",
    "vr_output_height",
    "source_width",
    "source_height",
    "source_fps",
    "depth_model_version",
    "depth_model_path",
    "use_metric_depth",
}


def batch_size_type(value: str) -> int:
    """Argparse type for --batch-size."""
    low, high = BATCH_SIZE_RANGE
    try:
        size = int(value)
    except ValueError:
        size = -1
    if not low <= size <= high:
        raise argparse.ArgumentTypeError(f"batch size must be a number from {low} to {high}")
    return size


def create_progress_tracker(settings: dict) -> TqdmProgressTracker:
    """Console progress bars matching the steps these settings run."""
    skipped = set()
    if not settings.get("apply_distortion", True):
        skipped.add("Fisheye Distortion")
    if settings.get("upscale_model", "none") == "none":
        skipped.add("AI Upscaling")
    return TqdmProgressTracker(
        skipped_steps=skipped,
        batched=settings.get("processing_mode", "batched") != "sequential",
    )


def print_interrupted_help(output_dir) -> None:
    """Tell the user how to continue after an interrupted run."""
    print("\nProcessing interrupted. Finished frames are kept.")
    print(f"  Continue:        depth_surge_3d.py --resume {output_dir}")
    print(f"  Video of the finished part: depth_surge_3d.py --assemble-partial {output_dir}")


def vr_resolution_type(value: str) -> str:
    """
    Argparse type for --vr-resolution.

    Accepts any preset name, "auto", or "custom:WIDTHxHEIGHT".
    """
    if value == "auto" or value in VR_RESOLUTIONS:
        return value
    if value.startswith("custom:"):
        if parse_custom_resolution(value) is not None:
            return value
        raise argparse.ArgumentTypeError(
            f"invalid custom resolution '{value}' (expected custom:WIDTHxHEIGHT, "
            "e.g. custom:1920x1080, each side 1-10000)"
        )
    raise argparse.ArgumentTypeError(
        f"invalid choice: '{value}' (use a preset from --list-resolutions, "
        "'auto', or custom:WIDTHxHEIGHT)"
    )


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
  %(prog)s --assemble-partial ./output/my_video_output/ # Video from an unfinished job
  %(prog)s video.mp4 --reuse ./output/my_video_output/ -b 0.08  # New settings, same depth maps
  %(prog)s video.mp4 --mode sequential                 # One step at a time
  %(prog)s --list-resolutions                          # Show available resolutions

Note: Always uses Video-Depth-Anything for temporal consistency across frames.
        """,
    )

    # Input/output arguments
    parser.add_argument("input_video", nargs="?", help="Input video file path")

    parser.add_argument(
        "--output-dir",
        "-o",
        default=DEFAULT_SETTINGS["output_dir"],
        help="Output directory (default: %(default)s)",
    )

    # Resume functionality
    parser.add_argument(
        "--resume", metavar="DIRECTORY", help="Resume processing from an existing output directory"
    )

    parser.add_argument(
        "--reuse",
        metavar="DIRECTORY",
        help="Run in an existing output directory with the settings of this command line. "
        "Its frames, depth maps and later steps are reused as far as the settings allow "
        "(e.g. a new baseline keeps the depth maps)",
    )
    parser.add_argument(
        "--assemble-partial",
        metavar="DIRECTORY",
        help="Create a video from the VR frames an interrupted job has finished so far",
    )

    # Processing mode
    parser.add_argument(
        "--mode",
        choices=PROCESSING_MODES,
        default=DEFAULT_SETTINGS["processing_mode"],
        help="batched: run all steps batch by batch, so finished VR frames appear early; "
        "sequential: finish each step for the whole video first (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=batch_size_type,
        default=DEFAULT_SETTINGS["batch_size"],
        metavar="FRAMES",
        help="Frames per batch (default: %(default)s)",
    )

    # VR settings
    parser.add_argument(
        "--vr-resolution",
        type=vr_resolution_type,
        default=DEFAULT_SETTINGS["vr_resolution"],
        metavar="RESOLUTION",
        help="VR output resolution per eye: a preset (see --list-resolutions), 'auto', "
        "or custom:WIDTHxHEIGHT (default: %(default)s)",
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
        "--no-distortion",
        action="store_true",
        help="Disable fisheye distortion: flat 3D that keeps the whole frame at the video's "
        "aspect ratio",
    )
    parser.add_argument(
        "--fisheye-fit",
        type=float,
        default=DEFAULT_SETTINGS["fisheye_frame_fit"],
        metavar="0-1",
        help="How much of a wide or tall video the square fisheye view shows: 0 fills the view "
        "and cuts off the sides, 1 shows the whole frame with black bars (default: %(default)s)",
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
    parser.add_argument(
        "--model", help="Path to model file (V2) or model name (V3, e.g., 'large', 'base', 'small')"
    )
    parser.add_argument(
        "--depth-model-version",
        choices=["v2", "v3"],
        default="v2",
        help="Depth model version: v2 (Video-Depth-Anything, default) or v3 (Depth-Anything-3, better VRAM efficiency)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Processing device (default: auto)",
    )
    parser.add_argument(
        "--metric",
        action="store_true",
        help="Use metric depth model (outputs real depth values in meters)",
    )

    parser.add_argument(
        "--depth-resolution",
        choices=["auto", "518", "720", "1080", "1440", "2160"],
        default="auto",
        help="Depth model input resolution in pixels; auto matches the source, never above it "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--super-sample",
        choices=["auto", "none", "2x", "1080p", "4k"],
        default=DEFAULT_SETTINGS["super_sample"],
        help="Resolution the stereo views are rendered at: auto (up to 2x, matched to the VR "
        "output), none (source resolution), 2x, 1080p or 4k (default: %(default)s)",
    )
    parser.add_argument(
        "--video-encoder",
        choices=["auto", "nvenc", "libx264", "libx265"],
        default="auto",
        help="Video encoder: auto (NVENC H.265 if available, else H.264), nvenc, libx264, "
        "libx265 (default: %(default)s)",
    )

    # Output options
    parser.add_argument("--no-audio", action="store_true", help="Do not preserve audio in output")
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep every step's images (depth maps, stereo pairs, ...) in the output directory. "
        "Needed to redo single steps with --reuse; uses a lot of disk space",
    )
    parser.add_argument(
        "--no-intermediates",
        action="store_true",
        help="Do not keep intermediate files (this is the default; kept for old command lines)",
    )
    parser.add_argument("--target-fps", type=int, help="Target output FPS (default: match source)")

    # Experimental features
    parser.add_argument(
        "--experimental-frame-interpolation",
        action="store_true",
        help="EXPERIMENTAL: Double FPS using motion interpolation. WARNING: May produce artifacts, wobbling, or poor quality. Recommended for artistic experimentation only.",
    )
    parser.add_argument(
        "--upscale-model",
        choices=["none", "x2", "x4", "x4-conservative"],
        default="none",
        help="AI upscaling model (Real-ESRGAN). Options: none (disabled, default), x2 (2x fast), x4 (4x best quality), x4-conservative (4x without GAN artifacts). Significantly increases processing time (+2-3x) and VRAM usage (+2-4GB).",
    )

    # Information and debugging
    parser.add_argument(
        "--list-resolutions", action="store_true", help="List all available VR resolution options"
    )
    parser.add_argument("--model-info", action="store_true", help="Show model information and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    return parser


def validate_arguments(args) -> bool:  # noqa: C901
    """Validate command line arguments."""

    if args.assemble_partial:
        if not Path(args.assemble_partial).is_dir():
            print(f"Error: Directory does not exist: {args.assemble_partial}")
            return False
        return True

    # Handle resume mode
    if args.resume:
        if not Path(args.resume).exists():
            print(f"Error: Resume directory does not exist: {args.resume}")
            return False
        return True  # Skip other validations for resume mode

    if args.reuse and not Path(args.reuse).is_dir():
        print(f"Error: Directory to reuse does not exist: {args.reuse}")
        return False

    # Regular mode validations
    if not args.input_video:
        print("Error: Input video is required when not resuming")
        return False

    # Validate input video
    if not validate_video_file(args.input_video):
        print(f"Error: Invalid or unsupported video file: {args.input_video}")
        return False

    # Validate ranges
    if (
        args.baseline < VALIDATION_RANGES["baseline"][0]
        or args.baseline > VALIDATION_RANGES["baseline"][1]
    ):
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

    if (
        args.fisheye_fov < VALIDATION_RANGES["fisheye_fov"][0]
        or args.fisheye_fov > VALIDATION_RANGES["fisheye_fov"][1]
    ):
        print(
            f"Error: FOV must be between {VALIDATION_RANGES['fisheye_fov'][0]} and {VALIDATION_RANGES['fisheye_fov'][1]} degrees"
        )
        return False

    if (
        args.crop_factor < VALIDATION_RANGES["crop_factor"][0]
        or args.crop_factor > VALIDATION_RANGES["crop_factor"][1]
    ):
        print(
            f"Error: Crop factor must be between {VALIDATION_RANGES['crop_factor'][0]} and {VALIDATION_RANGES['crop_factor'][1]}"
        )
        return False

    low, high = VALIDATION_RANGES["fisheye_frame_fit"]
    if not low <= args.fisheye_fit <= high:
        print(f"Error: Fisheye fit must be between {low} and {high}")
        return False

    if args.target_fps and (
        args.target_fps < VALIDATION_RANGES["target_fps"][0]
        or args.target_fps > VALIDATION_RANGES["target_fps"][1]
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

    print("\nCustom Resolution:")
    print("  custom:WxH      - Custom resolution (e.g., custom:1920x1080)")
    print("\nAuto Detection:")
    print("  auto            - Automatically detect optimal resolution")


def main():  # noqa: C901
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Handle special commands
    if args.list_resolutions:
        list_available_resolutions()
        return 0

    if args.assemble_partial:
        if not validate_arguments(args):
            return 1
        output = assemble_partial_video(Path(args.assemble_partial))
        if output is None:
            print("Could not create a partial video. Check error messages above.")
            return 1
        print(f"Partial video saved to: {output}")
        return 0

    # Handle resume mode
    if args.resume:
        if not validate_arguments(args):
            return 1
        print(f"Checking resume capability for: {args.resume}")
        resume_info = can_resume_processing(Path(args.resume))

        if not resume_info["can_resume"]:
            print("Cannot resume processing:")
            for rec in resume_info["recommendations"]:
                print(f"  - {rec}")
            return 1

        print("Can resume processing:")
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

        # Create projector with the model the job was started with
        projector = create_stereo_projector(
            processing_settings.get("depth_model_path"),
            processing_settings.get("device", "auto"),
            bool(processing_settings.get("use_metric_depth", False)),
            processing_settings.get("depth_model_version") or "v2",
        )

        print("Resuming processing...")
        print(f"Input: {video_path}")
        print(f"Output: {args.resume}")

        # Resume processing using original settings
        tracker = create_progress_tracker(processing_settings)
        try:
            success = projector.process_video(
                video_path=video_path,
                output_dir=args.resume,
                progress_tracker=tracker,
                **{k: v for k, v in processing_settings.items() if k not in _DERIVED_SETTINGS},
            )
        except (KeyboardInterrupt, InterruptedError):
            tracker.close()
            print_interrupted_help(args.resume)
            return 1
        finally:
            tracker.close()
            projector.unload_model()

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
        model_name = (
            "Depth-Anything-V3" if args.depth_model_version == "v3" else "Video-Depth-Anything (V2)"
        )
        print("Starting Depth Surge 3D processing...")
        print(f"Input: {args.input_video}")
        if args.reuse:
            print(f"Output: {args.reuse} (reusing what the settings allow)")
        else:
            print(f"Output: {args.output_dir} (batch subdirectory will be created)")
        print(f"Model: {model_name}")
        print(f"Format: {args.format}")
        print(f"Resolution: {args.vr_resolution}")
        print(f"Mode: {args.mode} (batches of {args.batch_size} frames)")

        # Show warning for experimental features
        if args.experimental_frame_interpolation:
            print(console_warning("WARNING: Experimental frame interpolation enabled!"))
            print("   This feature may produce artifacts, wobbling, or poor quality.")
            print("   Recommended for artistic experimentation only.")
            print()

        # Create batch-specific output directory
        from datetime import datetime

        video_name = Path(args.input_video).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_output_dir = Path(args.output_dir) / f"{video_name}_{timestamp}"
        if args.reuse:
            batch_output_dir = Path(args.reuse)

        tracker = create_progress_tracker(
            {
                "apply_distortion": not args.no_distortion,
                "upscale_model": args.upscale_model,
                "processing_mode": args.mode,
            }
        )
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
            fisheye_frame_fit=args.fisheye_fit,
            fisheye_projection=args.fisheye_projection,
            fisheye_fov=args.fisheye_fov,
            crop_factor=args.crop_factor,
            fisheye_crop_factor=args.fisheye_crop_factor,
            hole_fill_quality=args.hole_fill_quality,
            preserve_audio=not args.no_audio,
            keep_intermediates=args.keep_intermediates and not args.no_intermediates,
            target_fps=args.target_fps,
            experimental_frame_interpolation=args.experimental_frame_interpolation,
            upscale_model=args.upscale_model,
            depth_resolution=args.depth_resolution,
            super_sample=args.super_sample,
            video_encoder=args.video_encoder,
            processing_mode=args.mode,
            batch_size=args.batch_size,
            device=args.device,
            progress_tracker=tracker,
        )
        tracker.close()

        if success:
            print("Processing completed successfully!")
            print(f"Output saved to: {batch_output_dir}")
            return 0
        else:
            print("Processing failed. Check error messages above.")
            return 1

    except (KeyboardInterrupt, InterruptedError):
        if "tracker" in locals():
            tracker.close()
        if "batch_output_dir" in locals():
            print_interrupted_help(batch_output_dir)
        else:
            print("\nProcessing interrupted by user")
        return 1
    except Exception as e:
        if "tracker" in locals():
            tracker.close()
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
        except Exception:  # noqa: E722
            pass


if __name__ == "__main__":
    sys.exit(main())
