"""
Resume state of a batch directory.

Records, per pipeline step, how many frames (counted from the start of the video) are
finished and with which settings, so an interrupted run continues where it stopped and a
re-run with changed settings only repeats the steps those settings affect.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ...core.constants import PIPELINE_STATE_FILE

# Steps in pipeline order. "depth" progress lives in the depth stream state file (it has
# to match the alignment data stored there); `done` mirrors it for display.
STEP_KEYS = ["extract", "depth", "stereo", "distortion", "crop", "upscale", "assemble", "encode"]

# Settings that change a step's output (besides the output of the step before it)
_STEP_SETTINGS: dict[str, tuple[str, ...]] = {
    "extract": ("start_time", "end_time"),
    "depth": ("depth_resolution", "temporal_window_overlap"),
    "stereo": (
        "baseline",
        "focal_length",
        "hole_fill_quality",
        "super_sample",
        "per_eye_width",
        "per_eye_height",
    ),
    "distortion": (
        "fisheye_projection",
        "fisheye_fov",
        "fisheye_crop_factor",
        "fisheye_frame_fit",
    ),
    "crop": (
        "apply_distortion",
        "crop_factor",
        "fisheye_crop_factor",
        "per_eye_width",
        "per_eye_height",
    ),
    "upscale": ("upscale_model",),
    "assemble": ("vr_format", "per_eye_width", "per_eye_height"),
    "encode": ("video_encoder", "target_fps", "source_fps", "experimental_frame_interpolation"),
}


def active_step_keys(settings: dict[str, Any]) -> list[str]:
    """PURE: Keys of the steps that run with these settings, in pipeline order."""
    keys = ["extract", "depth", "stereo"]
    if settings.get("apply_distortion", True):
        keys.append("distortion")
    keys.append("crop")
    if settings.get("upscale_model", "none") != "none":
        keys.append("upscale")
    keys.append("assemble")
    keys.append("encode")
    return keys


# Raised when depth alignment/normalization changes in a way that makes depth maps of an
# unfinished run unusable: such a run then redoes its depth maps instead of continuing.
# 2: alignment keeps contrast (0.11.0 faded to flat depth on longer videos)
DEPTH_ALIGNMENT_REVISION = 2


def step_fingerprints(settings: dict[str, Any], source_id: str, model_id: str) -> dict[str, str]:
    """
    PURE: Fingerprint of everything that determines each active step's output.

    Fingerprints are chained: a step's fingerprint includes the one of the step before it,
    so a change invalidates the step it affects and every step after it.

    Args:
        settings: Processing settings
        source_id: Identifies the source video (e.g. name and size)
        model_id: Identifies the depth model

    Returns:
        Dict of step key -> fingerprint, in pipeline order
    """
    fingerprints: dict[str, str] = {}
    previous = ""
    for key in active_step_keys(settings):
        values: dict[str, Any] = {name: settings.get(name) for name in _STEP_SETTINGS[key]}
        if key == "extract":
            values["source"] = source_id
        elif key == "depth":
            values["model"] = model_id
            values["alignment"] = DEPTH_ALIGNMENT_REVISION
        payload = json.dumps([previous, values], sort_keys=True, default=str)
        previous = hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()
        fingerprints[key] = previous
    return fingerprints


class PipelineState:
    """Progress of the pipeline in one batch directory, persisted as JSON."""

    VERSION = 1

    def __init__(self, path: Path):
        self.path = path
        self.total_frames = 0
        self.extracted = False
        self.pruned = False  # intermediates of finished frames were deleted
        self.frames_pruned = 0  # original frames before this index were deleted
        self.segments: list[list[int]] = []  # [start, end) frame ranges of encoded segments
        self.fingerprints: dict[str, str] = {}
        self.done: dict[str, int] = {}

    @classmethod
    def load(cls, output_dir: Path) -> PipelineState:
        """
        Load the state of a batch directory (a fresh state if there is none).

        Side effects:
            - Reads the state file
        """
        state = cls(Path(output_dir) / PIPELINE_STATE_FILE)
        try:
            with open(state.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") == cls.VERSION:
                state.total_frames = int(data.get("total_frames", 0))
                state.extracted = bool(data.get("extracted", False))
                state.pruned = bool(data.get("pruned", False))
                state.frames_pruned = int(data.get("frames_pruned", 0))
                state.segments = [[int(a), int(b)] for a, b in data.get("segments", [])]
                state.fingerprints = {str(k): str(v) for k, v in data["fingerprints"].items()}
                state.done = {str(k): int(v) for k, v in data["done"].items()}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return cls(state.path)
        return state

    def save(self) -> None:
        """
        Write the state atomically.

        Side effects:
            - Writes the state file
        """
        data = {
            "version": self.VERSION,
            "total_frames": self.total_frames,
            "extracted": self.extracted,
            "pruned": self.pruned,
            "frames_pruned": self.frames_pruned,
            "segments": self.segments,
            "fingerprints": self.fingerprints,
            "done": self.done,
        }
        temp_path = self.path.with_name(self.path.name + ".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, self.path)

    def frames_done(self, key: str) -> int:
        """Frames finished by a step (0 if it never ran)."""
        return self.done.get(key, 0)

    def reconcile(self, fingerprints: dict[str, str]) -> str | None:
        """
        Adopt the fingerprints of a new run and drop progress they invalidate.

        Progress of the first step whose fingerprint changed, and of all steps after it,
        is reset. If intermediates of finished frames were deleted, later steps can't be
        redone on their own, so everything from the depth step on is reset; if original
        frames were deleted too, the frames are extracted again.

        Args:
            fingerprints: Fingerprints of the new run (see ``step_fingerprints``)

        Returns:
            Key of the first step that has to start over, or None if all progress is kept
        """
        keys = list(fingerprints)
        changed = next((k for k in keys if self.fingerprints.get(k) != fingerprints[k]), None)
        self.fingerprints = dict(fingerprints)
        if changed is None:
            self.done = {k: v for k, v in self.done.items() if k in keys}
            return None

        return self.restart_from(changed)

    def restart_from(self, key: str) -> str:
        """
        Drop the progress of a step and of all steps after it.

        If intermediates of finished frames were deleted, later steps can't be redone on
        their own: the restart then moves back to the depth step, or to frame extraction
        if original frames were deleted too.

        Args:
            key: Step key to restart from (must be one of the current fingerprints)

        Returns:
            Key of the step the pipeline actually restarts from
        """
        keys = list(self.fingerprints)
        if self.pruned and key != "extract":
            key = "extract" if self.frames_pruned > 0 else "depth"
        restart = keys[keys.index(key) :]
        self.done = {k: v for k, v in self.done.items() if k in keys and k not in restart}
        if "encode" in restart:
            self.segments = []
        if "extract" in restart:
            self.extracted = False
            self.total_frames = 0
            self.frames_pruned = 0
        if "depth" in restart:
            self.pruned = False
        return key
