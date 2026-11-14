"""Tests for temporal depth alignment and sequence normalization."""

import numpy as np
import pytest

from src.depth_surge_3d.utils.domain.depth_alignment import (
    DepthSequenceAligner,
    fit_scale_shift,
    normalize_depth_sequence,
)

rng = np.random.default_rng(0)


def _scene(n=30, h=24, w=32):
    """Smoothly moving synthetic inverse-depth sequence."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    frames = [1.0 + np.sin(x / 6 + t * 0.1) * 0.5 + y / h for t in range(n)]
    return np.stack(frames).astype(np.float32)


class TestFitScaleShift:
    def test_recovers_affine(self):
        ref = rng.random((40, 50))
        src = (ref - 0.3) / 2.5  # ref = src * 2.5 + 0.3
        scale, shift = fit_scale_shift(src, ref)
        assert scale == pytest.approx(2.5, rel=1e-6)
        assert shift == pytest.approx(0.3, abs=1e-6)

    def test_constant_source(self):
        scale, shift = fit_scale_shift(np.full((5, 5), 2.0), rng.random((5, 5)))
        assert np.isfinite(scale) and np.isfinite(shift)

    def test_negative_correlation_never_flips(self):
        """A negative scale would turn near into far; fall back to moment matching."""
        ref = rng.random((30, 30))
        scale, _ = fit_scale_shift(-ref, ref)
        assert scale > 0

    def test_ignores_nan(self):
        ref = rng.random((10, 10))
        src = ref * 0.5
        src[0, 0] = np.nan
        scale, shift = fit_scale_shift(src, ref)
        assert scale == pytest.approx(2.0, rel=1e-6)

    def test_empty(self):
        assert fit_scale_shift(np.array([]), np.array([])) == (1.0, 0.0)


class TestDepthSequenceAligner:
    def test_chunks_with_different_scale_are_stitched(self):
        """Regression: separate model calls used to produce depth jumps at chunk borders."""
        truth = _scene(30)
        aligner = DepthSequenceAligner(per_frame=False)
        # chunk 1: frames 0-14 at native scale; chunk 2: frames 10-29 at another scale/shift
        aligner.add(truth[:15], overlap=0)
        aligner.add(truth[10:] * 3.0 + 2.0, overlap=5)

        result = aligner.result()

        assert result.shape == truth.shape
        np.testing.assert_allclose(result, truth, rtol=1e-4, atol=1e-4)

    def test_overlap_larger_than_history_is_clamped(self):
        aligner = DepthSequenceAligner()
        aligner.add(_scene(3), overlap=10)
        assert len(aligner) == 3

    def test_per_frame_scale_flicker_removed(self):
        """Per-frame models: random per-frame scale/shift is fitted away."""
        truth = _scene(40)
        noisy = np.stack([f * rng.uniform(0.5, 2.0) + rng.uniform(-1, 1) for f in truth]).astype(
            np.float32
        )
        aligner = DepthSequenceAligner(per_frame=True)
        aligner.add(noisy[:20])
        aligner.add(noisy[20:])

        result = normalize_depth_sequence(aligner.result())
        expected = normalize_depth_sequence(truth)

        flicker_before = np.abs(np.diff(normalize_depth_sequence(noisy), axis=0)).mean()
        flicker_after = np.abs(np.diff(result, axis=0)).mean()
        assert flicker_after < flicker_before / 3
        assert np.abs(result - expected).mean() < 0.05

    def test_per_frame_no_drift_on_static_scene(self):
        frame = _scene(1)[0]
        aligner = DepthSequenceAligner(per_frame=True)
        for _ in range(500):
            aligner.add((frame * rng.uniform(0.8, 1.2))[None])
        result = aligner.result()
        assert np.abs(result[-1] - result[0]).max() < 1e-3

    def test_empty_result(self):
        assert DepthSequenceAligner().result().size == 0

    def test_per_frame_contrast_survives_changing_content(self):
        """Regression (0.11.0): depth faded to a flat gray after a few hundred frames."""
        aligner = DepthSequenceAligner(per_frame=True)
        first = last = None
        for index in range(800):
            # unrelated content in every frame, like a video full of cuts
            frame = rng.random((1, 24, 32)).astype(np.float32) * rng.uniform(0.5, 2.0)
            aligner.add(frame)
            final = aligner.take()
            first = final[0] if index == 0 else first
            last = final[-1]
        assert last.std() == pytest.approx(first.std(), rel=1e-3)
        assert last.mean() == pytest.approx(first.mean(), rel=1e-3)

    def test_chunk_contrast_survives_noisy_overlaps(self):
        """Overlap fits must not lose contrast chunk after chunk (prediction noise)."""
        aligner = DepthSequenceAligner(per_frame=False)
        scene = _scene(6)
        aligner.add(scene)
        first_std = aligner.take(2)[0].std()
        for _ in range(300):
            noisy = scene + rng.normal(0, 0.3 * scene.std(), scene.shape).astype(np.float32)
            aligner.add(noisy, overlap=2)
            final = aligner.take(2)
        assert final[-1].std() > 0.9 * first_std

    def test_flat_frame_keeps_previous_scale(self):
        aligner = DepthSequenceAligner(per_frame=True)
        frame = _scene(1)[0]
        aligner.add(frame[None])
        aligner.add((frame * 2.0)[None])
        aligner.add(np.full_like(frame, 7.0)[None])  # e.g. a black frame
        aligner.add((frame * 2.0)[None])
        result = aligner.result()
        assert np.isfinite(result).all()
        assert result[2].std() == pytest.approx(0.0, abs=1e-5)
        np.testing.assert_allclose(result[3], result[0], rtol=1e-4, atol=1e-4)

    def test_flat_first_frame_is_not_the_reference(self):
        aligner = DepthSequenceAligner(per_frame=True)
        frame = _scene(1)[0]
        aligner.add(np.zeros_like(frame)[None])
        aligner.add(frame[None])
        aligner.add((frame * 3.0 + 1.0)[None])
        result = aligner.result()
        np.testing.assert_allclose(result[2], result[1], rtol=1e-4, atol=1e-4)

    def test_restore_drops_flat_reference_image_of_0_11_0(self):
        aligner = DepthSequenceAligner(per_frame=True)
        aligner.restore({"reference": np.full((8, 8), 3.9, dtype=np.float32)})
        frame = _scene(1)[0]
        aligner.add(frame[None])
        np.testing.assert_allclose(aligner.result()[0], frame, rtol=1e-6)

    def test_restore_uses_moments_of_reference_image_of_0_11_0(self):
        reference = _scene(1)[0]
        aligner = DepthSequenceAligner(per_frame=True)
        aligner.restore({"reference": reference})
        aligner.add((reference * 4.0 - 2.0)[None])
        np.testing.assert_allclose(aligner.result()[0], reference, rtol=1e-3, atol=1e-3)


class TestNormalizeDepthSequence:
    def test_single_range_for_all_frames(self):
        seq = np.stack([np.full((4, 4), 1.0), np.full((4, 4), 2.0), np.full((4, 4), 3.0)])
        seq[:, 0, 0] += 0.01  # avoid a zero range inside frames
        result = normalize_depth_sequence(seq, 0, 100)
        assert result[0].mean() < result[1].mean() < result[2].mean()

    def test_outliers_clipped(self):
        seq = rng.random((5, 50, 50)).astype(np.float32)
        seq[0, 0, 0] = 1000.0
        result = normalize_depth_sequence(seq)
        assert result.max() == 1.0
        assert np.median(result) > 0.3  # one hot pixel doesn't squash everything

    def test_flat_and_empty(self):
        assert not normalize_depth_sequence(np.full((2, 3, 3), 5.0)).any()
        assert normalize_depth_sequence(np.empty((0, 2, 2))).size == 0

    def test_nan_safe(self):
        seq = rng.random((2, 5, 5)).astype(np.float32)
        seq[0, 0, 0] = np.nan
        assert np.isfinite(normalize_depth_sequence(seq)).all()


class TestMemoryBehaviour:
    """Long high-resolution videos: avoid holding extra full copies of the sequence."""

    def test_normalize_in_place(self):
        seq = rng.random((3, 8, 8)).astype(np.float32) * 4
        out = normalize_depth_sequence(seq, copy=False)
        assert out is seq
        assert 0.0 <= out.min() and out.max() <= 1.0

    def test_normalize_copies_by_default(self):
        seq = rng.random((3, 8, 8)).astype(np.float32) * 4
        original = seq.copy()
        normalize_depth_sequence(seq)
        np.testing.assert_array_equal(seq, original)

    def test_result_hands_over_frames(self):
        aligner = DepthSequenceAligner()
        aligner.add(_scene(5))
        assert aligner.result().shape[0] == 5
        assert len(aligner) == 0

    def test_add_does_not_modify_input(self):
        chunk = _scene(6)
        original = chunk.copy()
        aligner = DepthSequenceAligner()
        aligner.add(_scene(4))
        aligner.add(chunk * 2.0, overlap=2)
        aligner.add(chunk, overlap=2)
        np.testing.assert_array_equal(chunk, original)


class TestAlignerStreaming:
    """Taking finished frames out of the aligner and continuing a saved sequence."""

    def test_take_holds_back_newest_frames(self):
        from src.depth_surge_3d.utils.domain.depth_alignment import DepthSequenceAligner

        aligner = DepthSequenceAligner()
        aligner.add(np.arange(5, dtype=np.float32)[:, None, None] * np.ones((5, 2, 2), np.float32))

        final = aligner.take(hold_back=2)

        assert final[:, 0, 0].tolist() == [0.0, 1.0, 2.0]
        assert len(aligner) == 2
        assert aligner.take(hold_back=5).shape[0] == 0
        assert aligner.take()[:, 0, 0].tolist() == [3.0, 4.0]

    def test_streamed_result_equals_one_shot(self):
        """Emitting frames chunk by chunk gives the same sequence as collecting everything."""
        from src.depth_surge_3d.utils.domain.depth_alignment import DepthSequenceAligner

        rng = np.random.default_rng(0)
        truth = rng.random((20, 6, 8)).astype(np.float32)
        chunks = [(truth[0:10] * 2 + 1, 0), (truth[6:16] * 3 - 1, 4), (truth[12:20] * 0.5, 4)]

        one_shot = DepthSequenceAligner()
        for chunk, overlap in chunks:
            one_shot.add(chunk, overlap)
        expected = one_shot.result()

        streamed = DepthSequenceAligner()
        parts = []
        for i, (chunk, overlap) in enumerate(chunks):
            streamed.add(chunk, overlap)
            parts.append(streamed.take(0 if i == len(chunks) - 1 else 4))
        assert np.allclose(np.concatenate(parts), expected, atol=1e-5)

    @pytest.mark.parametrize("per_frame", [False, True])
    def test_state_round_trip(self, per_frame):
        from src.depth_surge_3d.utils.domain.depth_alignment import DepthSequenceAligner

        rng = np.random.default_rng(1)
        first, second = rng.random((6, 4, 4)).astype(np.float32), rng.random((6, 4, 4))

        original = DepthSequenceAligner(per_frame=per_frame)
        original.add(first)
        original.take(hold_back=2)
        restored = DepthSequenceAligner(per_frame=per_frame)
        restored.restore(original.state())

        for aligner in (original, restored):
            aligner.add(second.astype(np.float32), overlap=2)
        assert np.array_equal(original.result(), restored.result())

    def test_empty_state(self):
        from src.depth_surge_3d.utils.domain.depth_alignment import DepthSequenceAligner

        aligner = DepthSequenceAligner()
        assert aligner.state() == {}
        aligner.restore({})
        assert len(aligner) == 0


class TestRunningDepthRange:
    """Chunk-by-chunk normalization without knowing the whole video."""

    def _range(self, **kwargs):
        from src.depth_surge_3d.utils.domain.depth_alignment import RunningDepthRange

        return RunningDepthRange(low_percentile=0, high_percentile=100, **kwargs)

    @staticmethod
    def _chunk(low, high, frames=4):
        return np.tile(np.linspace(low, high, 16, dtype=np.float32).reshape(4, 4), (frames, 1, 1))

    def test_first_chunk_sets_the_range(self):
        depth_range = self._range()
        out = depth_range.normalize(self._chunk(2.0, 6.0))
        assert (depth_range.low, depth_range.high) == (2.0, 6.0)
        assert out.min() == 0.0 and out.max() == 1.0 and out.dtype == np.float32

    def test_same_content_gives_same_values(self):
        """A constant scene is normalized identically in every chunk (no flicker)."""
        depth_range = self._range()
        first = depth_range.normalize(self._chunk(2.0, 6.0))
        second = depth_range.normalize(self._chunk(2.0, 6.0))
        assert np.array_equal(first, second)

    def test_range_widens_at_once_but_gradually_within_the_chunk(self):
        depth_range = self._range()
        depth_range.normalize(self._chunk(0.0, 1.0))
        out = depth_range.normalize(self._chunk(0.0, 2.0, frames=4))

        assert depth_range.high == 2.0
        # the range moves linearly to its new value: no jump at the chunk boundary
        value_of_one = out[:, 1, 3]  # pixel with raw value 1.2
        assert np.all(np.diff(value_of_one) < 0)
        assert out[-1].max() == pytest.approx(1.0)

    def test_range_narrows_slowly(self):
        depth_range = self._range(half_life_frames=4)
        depth_range.normalize(self._chunk(0.0, 4.0))
        depth_range.normalize(self._chunk(0.0, 2.0, frames=4))
        assert depth_range.high == pytest.approx(3.0)  # halfway after one half-life

    def test_resumed_range_continues(self):
        depth_range = self._range()
        depth_range.normalize(self._chunk(0.0, 4.0))
        resumed = self._range(low=depth_range.low, high=depth_range.high)
        chunk = self._chunk(1.0, 3.0)
        assert np.array_equal(depth_range.normalize(chunk), resumed.normalize(chunk))

    def test_flat_or_invalid_chunks(self):
        depth_range = self._range()
        assert depth_range.normalize(np.full((2, 4, 4), 5.0, np.float32)).max() == 0.0
        assert depth_range.low is None  # a flat first chunk doesn't define a range
        depth_range.normalize(self._chunk(0.0, 2.0))
        flat = depth_range.normalize(np.full((2, 4, 4), 1.0, np.float32))
        assert np.allclose(flat, 0.5) and depth_range.high == 2.0
        with_nan = self._chunk(0.0, 2.0)
        with_nan[0, 0, 0] = np.nan
        assert np.isfinite(depth_range.normalize(with_nan)).all()
        assert depth_range.normalize(np.full((1, 2, 2), np.nan, np.float32)).shape == (1, 2, 2)
        assert depth_range.normalize(np.empty((0, 4, 4), np.float32)).shape == (0, 4, 4)
