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
