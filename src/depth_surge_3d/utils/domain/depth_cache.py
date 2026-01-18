"""
Depth map caching system for faster re-processing.

Caches depth maps globally (not tied to specific output batches) so users
can experiment with different stereo/VR settings without re-computing depth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def get_cache_dir() -> Path:
    """Get the global depth cache directory."""
    # Use XDG_CACHE_HOME if available, otherwise ~/.cache
    import os

    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        cache_dir = Path(cache_home) / "depth-surge-3d" / "depth_cache"
    else:
        cache_dir = Path.home() / ".cache" / "depth-surge-3d" / "depth_cache"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def compute_cache_key(video_path: str, depth_settings: dict[str, Any]) -> str:
    """
    Compute cache key for depth maps.

    The cache key is based on:
    - Video file content hash (first 1MB + last 1MB + size + mtime)
    - Settings that affect depth: model version, model size, depth resolution,
      metric vs relative

    Args:
        video_path: Path to input video
        depth_settings: Depth-related settings

    Returns:
        32-character hex cache key
    """
    hasher = hashlib.blake2b(digest_size=16)  # 16 bytes = 32 hex chars

    # Hash video file (fast approximation: first 1MB + last 1MB + metadata)
    video_path_obj = Path(video_path)
    file_size = video_path_obj.stat().st_size
    mtime = video_path_obj.stat().st_mtime

    # Add file metadata
    hasher.update(str(file_size).encode())
    hasher.update(str(mtime).encode())

    # Sample first and last 1MB for content hash
    sample_size = min(1024 * 1024, file_size // 2)  # 1MB or half file
    with open(video_path, "rb") as f:
        # First chunk
        hasher.update(f.read(sample_size))
        # Last chunk (if file is big enough)
        if file_size > sample_size * 2:
            f.seek(-sample_size, 2)  # Seek from end
            hasher.update(f.read(sample_size))

    # Hash depth-relevant settings (sorted for consistency)
    depth_relevant_keys = [
        "depth_model_version",
        "model_size",
        "depth_resolution",
        "use_metric_depth",
        "device",  # CPU vs GPU may produce slightly different results
    ]

    settings_for_hash = {k: depth_settings.get(k) for k in depth_relevant_keys}
    settings_json = json.dumps(settings_for_hash, sort_keys=True)
    hasher.update(settings_json.encode())

    return hasher.hexdigest()


def get_cached_depth_maps(
    video_path: str, depth_settings: dict[str, Any], num_frames: int
) -> np.ndarray | None:
    """
    Try to load depth maps from cache.

    Args:
        video_path: Path to input video
        depth_settings: Depth-related settings
        num_frames: Expected number of frames

    Returns:
        Cached depth maps array, or None if not found/invalid
    """
    cache_key = compute_cache_key(video_path, depth_settings)
    cache_dir = get_cache_dir()
    cache_entry_dir = cache_dir / cache_key

    if not cache_entry_dir.exists():
        return None

    # Check metadata
    metadata_file = cache_entry_dir / "metadata.json"
    if not metadata_file.exists():
        return None

    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # Verify frame count matches
        if metadata.get("num_frames") != num_frames:
            return None

        # Load depth maps
        depth_maps = []
        for i in range(num_frames):
            depth_file = cache_entry_dir / f"depth_{i:06d}.png"
            if not depth_file.exists():
                return None

            depth_img = cv2.imread(str(depth_file), cv2.IMREAD_GRAYSCALE)
            if depth_img is None:
                return None

            # Convert back to float (assuming saved as uint16 scaled by 1000)
            depth_float = depth_img.astype(np.float32) / 1000.0
            depth_maps.append(depth_float)

        return np.array(depth_maps)

    except Exception:
        # If anything fails, just return None (cache miss)
        return None


def save_depth_maps_to_cache(
    video_path: str, depth_settings: dict[str, Any], depth_maps: np.ndarray
) -> bool:
    """
    Save depth maps to global cache.

    Args:
        video_path: Path to input video
        depth_settings: Depth-related settings
        depth_maps: Depth maps to cache [N, H, W]

    Returns:
        True if saved successfully
    """
    try:
        cache_key = compute_cache_key(video_path, depth_settings)
        cache_dir = get_cache_dir()
        cache_entry_dir = cache_dir / cache_key

        # Create cache entry directory
        cache_entry_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "num_frames": len(depth_maps),
            "video_path": str(video_path),
            "depth_settings": {
                k: depth_settings.get(k)
                for k in [
                    "depth_model_version",
                    "model_size",
                    "depth_resolution",
                    "use_metric_depth",
                    "device",
                ]
            },
            "cache_version": "1.0",
        }

        metadata_file = cache_entry_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save depth maps (scaled to uint16 for efficient storage)
        for i, depth_map in enumerate(depth_maps):
            depth_file = cache_entry_dir / f"depth_{i:06d}.png"
            # Scale to uint16 (multiply by 1000 for 3 decimal precision)
            depth_uint16 = (depth_map * 1000.0).astype(np.uint16)
            cv2.imwrite(str(depth_file), depth_uint16)

        return True

    except Exception:
        # If saving fails, just return False (non-critical)
        return False


def clear_cache() -> int:
    """
    Clear all cached depth maps.

    Returns:
        Number of cache entries removed
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return 0

    count = 0
    for entry_dir in cache_dir.iterdir():
        if entry_dir.is_dir():
            try:
                import shutil

                shutil.rmtree(entry_dir)
                count += 1
            except Exception:
                pass

    return count


def get_cache_size() -> tuple[int, int]:
    """
    Get cache statistics.

    Returns:
        (number_of_entries, total_size_bytes)
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return (0, 0)

    num_entries = 0
    total_size = 0

    for entry_dir in cache_dir.iterdir():
        if entry_dir.is_dir():
            num_entries += 1
            for file in entry_dir.rglob("*"):
                if file.is_file():
                    total_size += file.stat().st_size

    return (num_entries, total_size)
