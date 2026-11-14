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


def _moments(values: np.ndarray, max_samples: int = _MAX_FIT_SAMPLES) -> tuple[float, float] | None:
    """PURE: (mean, std) of the finite values of a strided sample, or None without any."""
    (sample,) = _sample([values], max_samples)
    sample = sample[np.isfinite(sample)]
    if sample.size < 2:
        return None
    return float(sample.mean()), float(sample.std())


def _has_spread(mean: float, std: float) -> bool:
    """PURE: Whether a depth map has enough variation to take a scale from."""
    return std > 1e-8 + 1e-6 * abs(mean)


def fit_scale_shift(
    source: np.ndarray, reference: np.ndarray, max_samples: int = _MAX_FIT_SAMPLES
) -> tuple[float, float]:
    """
    PURE: Scale and shift so that ``source * scale + shift`` has the mean and spread of
    ``reference``.

    The scale is the ratio of the standard deviations, not the least-squares slope. The
    slope is smaller than that ratio whenever the two inputs aren't perfectly correlated
    (prediction noise, changed content), so a chain of least-squares fits loses a little
    contrast with every link and ends up flat. Matching the spread keeps the contrast.

    Falls back to shifting only when either input has no spread.

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

    src_mean, ref_mean = float(src.mean()), float(ref.mean())
    src_std, ref_std = float(src.std()), float(ref.std())
    if _has_spread(src_mean, src_std) and _has_spread(ref_mean, ref_std):
        scale = ref_std / src_std
        return scale, ref_mean - scale * src_mean
    return 1.0, ref_mean - src_mean


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
    Per-frame models (``per_frame=True``): every frame is additionally brought to the
    mean and spread of the first usable frame. That target is fixed, so nothing can drift
    or fade over a long video; how the values map to 0-1 is left to the normalization.
    """

    def __init__(self, per_frame: bool = False):
        """
        Args:
            per_frame: Model output has independent scale/shift per frame
        """
        self.per_frame = per_frame
        self._frames: list[np.ndarray] = []
        # per-frame target: [mean, std, scale used for the newest frame]
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
        """Bring a per-frame prediction to the mean and spread of the reference frame."""
        moments = _moments(frame)
        if moments is None:
            return frame
        mean, std = moments
        if self._reference is None:
            if _has_spread(mean, std):
                self._reference = np.array([mean, std, 1.0], dtype=np.float64)
            return frame

        ref_mean, ref_std, last_scale = (float(v) for v in self._reference)
        # A flat frame (e.g. black) has no scale of its own: keep the previous one
        scale = ref_std / std if _has_spread(mean, std) else last_scale
        self._reference[2] = scale
        return (frame * scale + (ref_mean - scale * mean)).astype(np.float32, copy=False)

    def take(self, hold_back: int = 0) -> np.ndarray:
        """
        Remove and return the frames that are final, keeping the last ``hold_back``.

        The frames kept back are the ones the next chunk repeats (its overlap): they are
        still going to be crossfaded with the next prediction, so they aren't final yet.

        Args:
            hold_back: Number of newest frames to keep in the aligner

        Returns:
            Array [N, H, W] (float32) of final frames, possibly empty
        """
        count = max(0, len(self._frames) - max(0, hold_back))
        if count == 0:
            return np.empty((0, 0, 0), dtype=np.float32)
        final = np.stack(self._frames[:count]).astype(np.float32, copy=False)
        self._frames = self._frames[count:]
        return final

    def state(self) -> dict[str, np.ndarray]:
        """Arrays needed to continue this sequence later (see ``restore``)."""
        state: dict[str, np.ndarray] = {}
        if self._frames:
            state["tail"] = np.stack(self._frames).astype(np.float32, copy=False)
        if self._reference is not None:
            state["reference"] = np.asarray(self._reference, dtype=np.float64)
        return state

    def restore(self, state: dict[str, np.ndarray]) -> None:
        """Continue a sequence from arrays returned by ``state``."""
        tail = state.get("tail")
        self._frames = [] if tail is None else [np.array(f, dtype=np.float32) for f in tail]
        self._reference = self._restored_reference(state.get("reference"))

    @staticmethod
    def _restored_reference(reference: np.ndarray | None) -> np.ndarray | None:
        """PURE: Per-frame target from a saved state (None if it can't be used)."""
        if reference is None:
            return None
        if reference.ndim == 1 and reference.size == 3:
            return np.array(reference, dtype=np.float64)
        # State of 0.11.0, which kept a reference image: continue with its mean and spread,
        # or start over if that image had already gone flat
        moments = _moments(reference)
        if moments is None or not _has_spread(*moments):
            return None
        return np.array([moments[0], moments[1], 1.0], dtype=np.float64)

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


class RunningDepthRange:
    """
    Normalizes a depth sequence to 0-1 chunk by chunk, without seeing the whole video.

    The range of the first chunk is used as is. After that the range widens at once when a
    chunk needs more room (nothing gets clipped for long) and narrows slowly when the scene
    uses less of it, so a single close-up doesn't flatten the rest of the video. Within a
    chunk the range moves linearly from the old to the new one: depth, and so stereo
    disparity, never jumps between chunks.
    """

    def __init__(
        self,
        half_life_frames: float = 600.0,
        low: float | None = None,
        high: float | None = None,
        low_percentile: float = 0.5,
        high_percentile: float = 99.5,
    ):
        """
        Args:
            half_life_frames: Frames after which the range has narrowed halfway to a
                smaller range of the content
            low: Range start to continue from (resume)
            high: Range end to continue from (resume)
            low_percentile: Percentile of a chunk mapped to 0 (far)
            high_percentile: Percentile of a chunk mapped to 1 (near)
        """
        self.half_life_frames = max(1.0, float(half_life_frames))
        self.low = low
        self.high = high
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile

    def _chunk_range(self, depths: np.ndarray, max_samples: int) -> tuple[float, float] | None:
        """Robust (low, high) of a chunk, or None if it has no usable range."""
        step = max(1, depths.size // max_samples)
        sample = depths.reshape(-1)[::step]
        sample = sample[np.isfinite(sample)]
        if sample.size == 0:
            return None
        low, high = np.percentile(sample, [self.low_percentile, self.high_percentile])
        if high - low < 1e-8:
            return None
        return float(low), float(high)

    def normalize(self, depths: np.ndarray, max_samples: int = 2_000_000) -> np.ndarray:
        """
        Normalize the next chunk of aligned inverse depth and update the range.

        Args:
            depths: Array [N, H, W] of aligned inverse depth, in video order
            max_samples: Maximum number of pixels used to estimate the chunk's range

        Returns:
            Array [N, H, W] in [0, 1] (float32)
        """
        out = np.array(depths, dtype=np.float32)
        count = len(out)
        if count == 0:
            return out

        chunk_range = self._chunk_range(out, max_samples)
        # per-frame range start/end (annotated: numpy's stubs differ between versions)
        lows: np.ndarray
        highs: np.ndarray
        if self.low is None or self.high is None:
            if chunk_range is None:
                out.fill(0.0)
                return out
            self.low, self.high = chunk_range
            lows = np.full(count, self.low, dtype=np.float32)
            highs = np.full(count, self.high, dtype=np.float32)
        else:
            old_low, old_high = self.low, self.high
            if chunk_range is not None:
                narrow = 1.0 - 0.5 ** (count / self.half_life_frames)
                low, high = chunk_range
                self.low = low if low < old_low else old_low + (low - old_low) * narrow
                self.high = high if high > old_high else old_high + (high - old_high) * narrow
            weights = np.linspace(0.0, 1.0, count + 1, dtype=np.float32)[1:]
            lows = np.asarray(old_low + (self.low - old_low) * weights, dtype=np.float32)
            highs = np.asarray(old_high + (self.high - old_high) * weights, dtype=np.float32)

        np.nan_to_num(
            out, copy=False, nan=float(lows[0]), posinf=float(highs[0]), neginf=float(lows[0])
        )
        out -= lows[:, None, None]
        out /= (highs - lows)[:, None, None]
        np.clip(out, 0.0, 1.0, out=out)
        return out
