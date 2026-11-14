"""Tests for the disk space estimate."""

from src.depth_surge_3d.utils.domain.disk_estimate import estimate_disk_usage

SETTINGS = {
    "per_eye_width": 1920,
    "per_eye_height": 1920,
    "vr_output_width": 3840,
    "vr_output_height": 1920,
    "super_sample": "auto",
    "apply_distortion": True,
    "upscale_model": "none",
    "keep_intermediates": True,
    "batch_size": 100,
}


def _total(frames=3000, **changes):
    return estimate_disk_usage(frames, 1920, 1080, {**SETTINGS, **changes})["total"]


def test_scales_with_frame_count_when_intermediates_are_kept():
    assert _total(6000) == 2 * _total(3000)


def test_matches_a_measured_job():
    """1080p source, 10638 frames, 1536 px square eyes, intermediates kept: about 193 GB."""
    settings = {**SETTINGS, "per_eye_width": 1536, "per_eye_height": 1536}
    settings.update({"vr_output_width": 3072, "vr_output_height": 1536})
    total = estimate_disk_usage(10638, 1920, 1080, settings)["total"]
    assert 180e9 < total < 215e9


def test_without_intermediates_mostly_the_extracted_frames_remain():
    kept = _total(30000)
    pruned = estimate_disk_usage(30000, 1920, 1080, {**SETTINGS, "keep_intermediates": False})
    assert pruned["total"] < kept / 5
    assert pruned["breakdown"]["original_frames"] > 0.8 * pruned["total"]
    assert set(pruned["breakdown"]) == {"original_frames", "depth_maps", "vr_frames", "video"}
    # the frame part stops growing with the video; the video part doesn't
    longer = estimate_disk_usage(60000, 1920, 1080, {**SETTINGS, "keep_intermediates": False})
    assert longer["breakdown"]["vr_frames"] == pruned["breakdown"]["vr_frames"]
    assert longer["breakdown"]["video"] == 2 * pruned["breakdown"]["video"]


def test_short_videos_are_not_overestimated_without_intermediates():
    assert _total(50, keep_intermediates=False) < _total(50)


def test_optional_steps_add_space():
    base = _total()
    assert _total(apply_distortion=False) < base
    assert _total(upscale_model="x2") > base
    assert _total(upscale_model="x4") > _total(upscale_model="x2")
    upscaled = estimate_disk_usage(
        3000, 1920, 1080, {**SETTINGS, "upscale_model": "x2", "keep_intermediates": False}
    )
    assert {"cropped_frames", "upscaled_frames"} <= set(upscaled["breakdown"])


def test_final_video_and_breakdown():
    estimate = estimate_disk_usage(3000, 1920, 1080, SETTINGS)
    assert estimate["total"] == sum(estimate["breakdown"].values())
    assert estimate["final_video"] == estimate["breakdown"]["video"] // 2
    assert all(isinstance(size, int) for size in estimate["breakdown"].values())


def test_missing_sizes_fall_back_to_the_source():
    estimate = estimate_disk_usage(100, 1280, 720, {})
    assert estimate["total"] > 0
