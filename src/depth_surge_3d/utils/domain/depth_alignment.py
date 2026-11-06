"""
Temporal alignment and normalization of depth sequences.

Relative depth models predict inverse depth only up to an unknown scale and shift, and
that scale/shift differs between separate model calls (chunks) and, for per-frame models,
between individual frames. Normalizing every frame to 0-1 on its own turns those
differences into depth flicker, which shows up directly as stereo flicker.

This module:
- aligns consecutive chunks of a temporally consistent model on overlapping frames,
- aligns per-frame model output to a running reference,
- normalizes the whole sequence with one robust range.

All inputs/outputs are inverse depth (larger = nearer).
"""

from __future__ import annotations

import numpy as np

# Pixels sampled per scale/shift fit (keeps fits fast at any resolution)
_MAX_FIT_SAMPLES = 200_000


def _sample(arrays: list[np.ndarray], max_samples: int) -> list[np.ndarray]:
    """Flatten arrays of identical shape and take the same strided sample from each."""
    size = arrays[0].size
    step = max(1, size // max_samples)
    return [a.reshape(-1)[::step].astype(np.float64) for a in arrays]


def fit_scale_shift(
    source: np.ndarray, reference: np.ndarray, max_samples: int = _MAX_FIT_SAMPLES
) -> tuple[float, float]:
    """
    PURE: Least-squares scale and shift so that ``source * scale + shift ~= reference``.

    Falls back to matching mean/std when the fit is degenerate (non-positive scale, e.g.
    for near-constant input), and to identity if that fails too.

    Args:
        source: Depth values to transform (any shape)
        reference: Target depth values (same shape)
        max_samples: Maximum number of pixels used for the fit

    Returns:
        Tuple of (scale, shift)
    """
    src, ref = _sample([source, reference], max_samples)
    valid = np.isfinite(src) & np.isfinite(ref)
    src, ref = src[valid], ref[valid]
    if src.size < 2:
        return 1.0, 0.0

    src_mean, ref_mean = src.mean(), ref.mean()
    src_var = ((src - src_mean) ** 2).mean()
    if src_var > 1e-12:
        scale = float(((src - src_mean) * (ref - ref_mean)).mean() / src_var)
        if np.isfinite(scale) and scale > 0:
            return scale, float(ref_mean - scale * src_mean)

        # Degenerate covariance: match first and second moments instead
        ref_std = ref.std()
        if ref_std > 0:
            scale = float(ref_std / np.sqrt(src_var))
            return scale, float(ref_mean - scale * src_mean)

    return 1.0, float(ref_mean - src_mean)


class DepthSequenceAligner:
    """
    Accumulates raw inverse-depth chunks into one temporally aligned sequence.

    Usage::

        aligner = DepthSequenceAligner(per_frame=False)
        for chunk, overlap in chunks:      # chunk includes `overlap` already-seen frames
            aligner.add(chunk, overlap)
        sequence = aligner.result()

    Temporally consistent models (``per_frame=False``): each chunk is fitted to the
    previous output on its overlapping frames, and the overlap is crossfaded.
    Per-frame models (``per_frame=True``): every frame is additionally fitted to an
    exponential moving average of the aligned frames before it.
    """

    def __init__(self, per_frame: bool = False, reference_decay: float = 0.3):
        """
        Args:
            per_frame: Model output has independent scale/shift per frame
            reference_decay: EMA weight of the newest frame in the per-frame reference
        """
        self.per_frame = per_frame
        self.reference_decay = reference_decay
        self._frames: list[np.ndarray] = []
        self._reference: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._frames)

    def add(self, chunk: np.ndarray, overlap: int = 0) -> None:
        """
        Add a chunk of raw inverse depth maps.

        Args:
            chunk: Array [N, H, W]; its first ``overlap`` frames repeat the last
                ``overlap`` frames already added
            overlap: Number of repeated frames at the start of the chunk
        """
        chunk = np.array(chunk, dtype=np.float32)  # own copy, transformed in place below
        overlap = min(overlap, len(self._frames), len(chunk))

        if overlap > 0:
            previous = np.stack(self._frames[-overlap:])
            scale, shift = fit_scale_shift(chunk[:overlap], previous)
            chunk *= scale
            chunk += shift

            # Crossfade the repeated frames from the old to the new prediction
            weights = np.linspace(0.0, 1.0, overlap + 2, dtype=np.float32)[1:-1]
            blended = (
                previous * (1.0 - weights[:, None, None]) + chunk[:overlap] * weights[:, None, None]
            )
            self._frames[-overlap:] = list(blended)
            chunk = chunk[overlap:]

        for frame in chunk:
            if self.per_frame:
                frame = self._align_to_reference(frame)
            self._frames.append(frame)

    def _align_to_reference(self, frame: np.ndarray) -> np.ndarray:
        """Fit a per-frame prediction to the running reference and update it."""
        if self._reference is None:
            self._reference = frame.copy()
            return frame
        scale, shift = fit_scale_shift(frame, self._reference)
        aligned = frame * scale + shift
        decay = self.reference_decay
        self._reference = (1.0 - decay) * self._reference + decay * aligned
        return aligned

    def result(self) -> np.ndarray:
        """
        Return the aligned sequence as an array [N, H, W] (float32).

        Hands the frames over: the aligner is empty afterwards, so the sequence isn't
        held in memory twice.
        """
        if not self._frames:
            return np.empty((0, 0, 0), dtype=np.float32)
        sequence = np.stack(self._frames).astype(np.float32, copy=False)
        self._frames = []
        self._reference = None
        return sequence


def normalize_depth_sequence(
    depths: np.ndarray,
    low_percentile: float = 0.5,
    high_percentile: float = 99.5,
    max_samples: int = 2_000_000,
    copy: bool = True,
) -> np.ndarray:
    """
    Normalize a whole inverse-depth sequence to 0-1 with one robust range.

    Using a single range for all frames keeps depth (and therefore stereo disparity)
    stable over time; percentiles keep a few outlier pixels from compressing the range.

    Args:
        depths: Array [N, H, W] of aligned inverse depth
        low_percentile: Percentile mapped to 0 (far)
        high_percentile: Percentile mapped to 1 (near)
        max_samples: Maximum number of pixels used to estimate the range
        copy: If False and ``depths`` is float32, normalize it in place (saves a full
            copy of the sequence, which matters for long high-resolution videos)

    Returns:
        Array [N, H, W] in [0, 1] (float32)
    """
    out = np.array(depths, dtype=np.float32, copy=copy or None)
    if out.size == 0:
        return out

    step = max(1, out.size // max_samples)
    sample = out.reshape(-1)[::step]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        out.fill(0.0)
        return out

    low, high = np.percentile(sample, [low_percentile, high_percentile])
    if high - low < 1e-8:
        out.fill(0.0)
        return out

    np.nan_to_num(out, copy=False, nan=float(low), posinf=float(high), neginf=float(low))
    out -= low
    out /= high - low
    np.clip(out, 0.0, 1.0, out=out)
    return out
