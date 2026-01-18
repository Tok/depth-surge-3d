#!/usr/bin/env python3
"""

from __future__ import annotations
Video Processing Utilities
Extracted from app.py to keep it under 500 LOC
"""

import numpy as np
from pathlib import Path
from collections.abc import Callable
from ...core.constants import INTERMEDIATE_DIRS


# Lazy import cv2 to avoid blocking module loading when cv2 is not available
def _get_cv2():
    """Lazy import cv2 only when needed."""
    try:
        import cv2

        return cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for image processing. Install with: pip install opencv-python"
        )


def process_video_serial(
    projector, callback: Callable, frame_files: list[Path], output_path: Path, **kwargs
) -> bool:
    """
    Process video in serial mode (frame-by-frame, complete pipeline per frame).
    This is the original processing mode where each frame goes through the entire
    pipeline before moving to the next frame.
    """

    # Get settings
    super_sample_width = kwargs.get("super_sample_width", 1920)
    super_sample_height = kwargs.get("super_sample_height", 1080)
    apply_distortion = kwargs.get("apply_distortion", True)

    # Create directories if keeping intermediates
    directories = {}
    if kwargs.get("keep_intermediates", True):
        directories = _create_intermediate_directories(output_path, **kwargs)

    # Get cv2 for this function
    cv2 = _get_cv2()

    # Process each frame completely
    for i, frame_file in enumerate(frame_files):
        try:
            frame_name = frame_file.stem
            vr_frame_path = (
                directories.get("vr_frames", output_path / INTERMEDIATE_DIRS["vr_frames"])
                / f"{frame_name}.png"
            )

            # Skip if already processed (resume support)
            if vr_frame_path.exists():
                callback.update_progress(
                    f"Skipping frame {i+1}/{len(frame_files)} - Already processed",
                    i + 1,
                    phase="extraction",
                )
                continue

            # Load frame
            callback.update_progress(
                f"Processing frame {i+1}/{len(frame_files)} - Loading...",
                i + 1,
                phase="extraction",
            )
            original_image = cv2.imread(str(frame_file))
            if original_image is None:
                continue

            # Process complete pipeline for this frame
            vr_frame = _process_single_frame_complete(
                projector,
                callback,
                original_image,
                i,
                len(frame_files),
                directories,
                frame_name,
                super_sample_width,
                super_sample_height,
                apply_distortion,
                **kwargs,
            )

            # Save VR frame
            if vr_frame is not None:
                cv2.imwrite(str(vr_frame_path), vr_frame)

        except Exception as e:
            print(f"Error processing frame {i+1}: {e}")
            continue

    return True


def process_video_batch(
    projector, callback: Callable, frame_files: list[Path], output_path: Path, **kwargs
) -> bool:
    """
    Process video in batch mode (phase-by-phase, all frames per phase).
    This processes all frames through one phase before moving to the next phase.
    """

    # Get settings
    super_sample_width = kwargs.get("super_sample_width", 1920)
    super_sample_height = kwargs.get("super_sample_height", 1080)
    apply_distortion = kwargs.get("apply_distortion", True)

    # Create directories
    directories = {}
    if kwargs.get("keep_intermediates", True):
        directories = _create_intermediate_directories(output_path, **kwargs)

    total_frames = len(frame_files)

    # Get cv2 for this function
    _get_cv2()

    # Phase 1: Super sampling (if needed)
    if super_sample_width != 1920 or super_sample_height != 1080:  # Only if different from defaults
        callback.update_progress("Phase 1: Super sampling all frames...", 0, phase="super_sampling")
        for i, frame_file in enumerate(frame_files):
            callback.update_progress(
                f"Super sampling frame {i+1}/{total_frames}",
                i + 1,
                phase="super_sampling",
            )
            _process_supersample_frame(
                projector,
                frame_file,
                directories,
                super_sample_width,
                super_sample_height,
                **kwargs,
            )

    # Phase 2: Depth estimation for all frames
    callback.update_progress("Phase 2: Generating depth maps...", 0, phase="depth_estimation")
    for i, frame_file in enumerate(frame_files):
        callback.update_progress(
            f"Generating depth map {i+1}/{total_frames}",
            i + 1,
            phase="depth_estimation",
        )
        _process_depth_frame(projector, frame_file, directories, **kwargs)

    # Phase 3: Stereo generation for all frames
    callback.update_progress("Phase 3: Creating stereo pairs...", 0, phase="stereo_generation")
    for i, frame_file in enumerate(frame_files):
        callback.update_progress(
            f"Creating stereo pair {i+1}/{total_frames}",
            i + 1,
            phase="stereo_generation",
        )
        _process_stereo_frame(projector, frame_file, directories, **kwargs)

    # Phase 4: Fisheye distortion (if enabled)
    if apply_distortion:
        callback.update_progress("Phase 4: Applying fisheye distortion...", 0, phase="distortion")
        for i, frame_file in enumerate(frame_files):
            callback.update_progress(
                f"Fisheye distortion {i+1}/{total_frames}", i + 1, phase="distortion"
            )
            _process_fisheye_frame(projector, frame_file, directories, **kwargs)

    # Phase 5: VR assembly for all frames
    callback.update_progress("Phase 5: Assembling VR frames...", 0, phase="vr_assembly")
    for i, frame_file in enumerate(frame_files):
        callback.update_progress(
            f"Assembling VR frame {i+1}/{total_frames}", i + 1, phase="vr_assembly"
        )
        _process_vr_assembly_frame(projector, frame_file, directories, apply_distortion, **kwargs)

    return True


def _create_intermediate_directories(output_path: Path, **kwargs) -> dict[str, Path]:
    """Create intermediate directories and return mapping."""
    directories = {}

    # Always create VR frames directory
    vr_dir = output_path / INTERMEDIATE_DIRS["vr_frames"]
    vr_dir.mkdir(exist_ok=True)
    directories["vr_frames"] = vr_dir

    if kwargs.get("keep_intermediates", True):
        # Create all intermediate directories
        for key, dirname in INTERMEDIATE_DIRS.items():
            if key != "vr_frames":  # Already created
                dir_path = output_path / dirname
                dir_path.mkdir(exist_ok=True)
                directories[key] = dir_path

    return directories


def _process_single_frame_complete(
    projector,
    callback,
    original_image,
    frame_idx,
    total_frames,
    directories,
    frame_name,
    super_sample_width,
    super_sample_height,
    apply_distortion,
    **kwargs,
):
    """Process a single frame through the complete pipeline (serial mode)."""

    # Get cv2 for this function
    cv2 = _get_cv2()

    # Super sampling
    if (
        super_sample_width != original_image.shape[1]
        or super_sample_height != original_image.shape[0]
    ):
        callback.update_progress(
            f"Processing frame {frame_idx+1}/{total_frames} - Super sampling...",
            frame_idx + 1,
            phase="super_sampling",
        )
        image = projector.apply_super_sampling(
            original_image, super_sample_width, super_sample_height
        )
        if "supersampled" in directories:
            cv2.imwrite(str(directories["supersampled"] / f"{frame_name}.png"), image)
    else:
        image = original_image

    # Depth estimation
    callback.update_progress(
        f"Processing frame {frame_idx+1}/{total_frames} - Generating depth map...",
        frame_idx + 1,
        phase="depth_estimation",
    )
    depth_map = projector.generate_depth_map_from_array(image)
    if "depth_maps" in directories:
        depth_vis = (depth_map * 255).astype(np.uint8)
        cv2.imwrite(str(directories["depth_maps"] / f"{frame_name}.png"), depth_vis)

    # Stereo generation
    callback.update_progress(
        f"Processing frame {frame_idx+1}/{total_frames} - Creating stereo pair...",
        frame_idx + 1,
        phase="stereo_generation",
    )
    left_img, right_img = projector.create_stereo_pair_from_depth(image, depth_map, **kwargs)
    if "left_frames" in directories:
        cv2.imwrite(str(directories["left_frames"] / f"{frame_name}.png"), left_img)
        cv2.imwrite(str(directories["right_frames"] / f"{frame_name}.png"), right_img)

    # Fisheye distortion (if enabled)
    if apply_distortion:
        callback.update_progress(
            f"Processing frame {frame_idx+1}/{total_frames} - Fisheye projection...",
            frame_idx + 1,
            phase="distortion",
        )
        left_distorted = projector.apply_fisheye_distortion(left_img, **kwargs)
        right_distorted = projector.apply_fisheye_distortion(right_img, **kwargs)
        if "left_distorted" in directories:
            cv2.imwrite(str(directories["left_distorted"] / f"{frame_name}.png"), left_distorted)
            cv2.imwrite(
                str(directories["right_distorted"] / f"{frame_name}.png"),
                right_distorted,
            )
        left_final, right_final = left_distorted, right_distorted
    else:
        left_final, right_final = left_img, right_img

    # VR assembly
    callback.update_progress(
        f"Processing frame {frame_idx+1}/{total_frames} - Creating VR frame...",
        frame_idx + 1,
        phase="vr_assembly",
    )
    vr_frame = projector.create_vr_format(
        left_final, right_final, kwargs.get("vr_format", "side_by_side")
    )

    return vr_frame


def _process_supersample_frame(projector, frame_file, directories, width, height, **kwargs):
    """Process super sampling for a single frame (batch mode)."""
    cv2 = _get_cv2()
    frame_name = frame_file.stem
    original_image = cv2.imread(str(frame_file))
    if original_image is None:
        return

    # Apply super sampling
    image = projector.apply_super_sampling(original_image, width, height)

    # Save if keeping intermediates
    if "supersampled" in directories:
        cv2.imwrite(str(directories["supersampled"] / f"{frame_name}.png"), image)


def _process_depth_frame(projector, frame_file, directories, **kwargs):
    """Process depth estimation for a single frame (batch mode)."""
    cv2 = _get_cv2()
    frame_name = frame_file.stem

    # Load super sampled frame if it exists, otherwise original
    if "supersampled" in directories:
        supersample_path = directories["supersampled"] / f"{frame_name}.png"
        if supersample_path.exists():
            image = cv2.imread(str(supersample_path))
        else:
            image = cv2.imread(str(frame_file))
    else:
        image = cv2.imread(str(frame_file))

    if image is None:
        return

    # Generate depth map
    depth_map = projector.generate_depth_map_from_array(image)

    # Save depth map
    if "depth_maps" in directories:
        depth_vis = (depth_map * 255).astype(np.uint8)
        cv2.imwrite(str(directories["depth_maps"] / f"{frame_name}.png"), depth_vis)


def _process_stereo_frame(projector, frame_file, directories, **kwargs):
    """Process stereo generation for a single frame (batch mode)."""
    cv2 = _get_cv2()
    frame_name = frame_file.stem

    # Load image and depth map
    if "supersampled" in directories:
        supersample_path = directories["supersampled"] / f"{frame_name}.png"
        if supersample_path.exists():
            image = cv2.imread(str(supersample_path))
        else:
            image = cv2.imread(str(frame_file))
    else:
        image = cv2.imread(str(frame_file))

    if image is None:
        return

    # Load depth map
    depth_path = directories["depth_maps"] / f"{frame_name}.png"
    depth_vis = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
    depth_map = depth_vis.astype(np.float32) / 255.0

    # Create stereo pair
    left_img, right_img = projector.create_stereo_pair_from_depth(image, depth_map, **kwargs)

    # Save stereo frames
    if "left_frames" in directories:
        cv2.imwrite(str(directories["left_frames"] / f"{frame_name}.png"), left_img)
        cv2.imwrite(str(directories["right_frames"] / f"{frame_name}.png"), right_img)


def _process_fisheye_frame(projector, frame_file, directories, **kwargs):
    """Process fisheye distortion for a single frame (batch mode)."""
    cv2 = _get_cv2()
    frame_name = frame_file.stem

    # Load stereo frames
    left_path = directories["left_frames"] / f"{frame_name}.png"
    right_path = directories["right_frames"] / f"{frame_name}.png"

    left_img = cv2.imread(str(left_path))
    right_img = cv2.imread(str(right_path))

    if left_img is None or right_img is None:
        return

    # Apply fisheye distortion
    left_distorted = projector.apply_fisheye_distortion(left_img, **kwargs)
    right_distorted = projector.apply_fisheye_distortion(right_img, **kwargs)

    # Save distorted frames
    if "left_distorted" in directories:
        cv2.imwrite(str(directories["left_distorted"] / f"{frame_name}.png"), left_distorted)
        cv2.imwrite(str(directories["right_distorted"] / f"{frame_name}.png"), right_distorted)


def _process_vr_assembly_frame(projector, frame_file, directories, apply_distortion, **kwargs):
    """Process VR assembly for a single frame (batch mode)."""
    cv2 = _get_cv2()
    frame_name = frame_file.stem

    # Load final stereo frames (distorted if fisheye was applied, otherwise original stereo)
    if apply_distortion and "left_distorted" in directories:
        left_path = directories["left_distorted"] / f"{frame_name}.png"
        right_path = directories["right_distorted"] / f"{frame_name}.png"
    else:
        left_path = directories["left_frames"] / f"{frame_name}.png"
        right_path = directories["right_frames"] / f"{frame_name}.png"

    left_final = cv2.imread(str(left_path))
    right_final = cv2.imread(str(right_path))

    if left_final is None or right_final is None:
        return

    # Create VR frame
    vr_frame = projector.create_vr_format(
        left_final, right_final, kwargs.get("vr_format", "side_by_side")
    )

    # Save VR frame
    if "vr_frames" in directories:
        cv2.imwrite(str(directories["vr_frames"] / f"{frame_name}.png"), vr_frame)
