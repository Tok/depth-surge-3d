"""Unit tests for image processing utilities."""

import numpy as np
import pytest
from src.depth_surge_3d.utils.imaging.image_processing import (
    normalize_depth_map,
    depth_to_disparity,
    stereo_depth_stats,
    resize_image,
    apply_center_crop,
    create_vr_frame,
    validate_image_array,
    calculate_image_statistics,
    create_shifted_image,
    calculate_fisheye_coordinates,
    apply_fisheye_distortion,
    apply_fisheye_square_crop,
    hole_fill_image,
)


class TestNormalizeDepthMap:
    """Test depth map normalization function."""

    def test_normalize_standard_range(self):
        """Test normalization of standard depth map."""
        depth = np.array([[10.0, 20.0], [30.0, 40.0]])
        normalized = normalize_depth_map(depth)

        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert normalized.shape == depth.shape

    def test_normalize_already_0_to_1(self):
        """Test normalization of already normalized depth map."""
        depth = np.array([[0.0, 0.5], [0.75, 1.0]])
        normalized = normalize_depth_map(depth)

        assert normalized.min() == 0.0
        assert normalized.max() == 1.0

    def test_normalize_constant_depth(self):
        """Test normalization of constant depth map returns zeros."""
        depth = np.full((100, 100), 5.0)
        normalized = normalize_depth_map(depth)

        # Constant depth should return all zeros
        assert np.all(normalized == 0.0)

    def test_normalize_negative_values(self):
        """Test normalization handles negative values."""
        depth = np.array([[-10.0, -5.0], [0.0, 5.0]])
        normalized = normalize_depth_map(depth)

        assert normalized.min() == 0.0
        assert normalized.max() == 1.0

    def test_normalize_large_values(self):
        """Test normalization with large depth values."""
        depth = np.array([[1000.0, 2000.0], [3000.0, 4000.0]])
        normalized = normalize_depth_map(depth)

        assert normalized.min() == 0.0
        assert normalized.max() == 1.0

    def test_normalize_single_pixel(self):
        """Test normalization of single pixel."""
        depth = np.array([[42.0]])
        normalized = normalize_depth_map(depth)

        # Single value should become 0 (since min == max)
        assert normalized[0, 0] == 0.0

    def test_normalize_preserves_shape(self):
        """Test that normalization preserves shape."""
        shapes = [(100, 100), (50, 200), (1920, 1080)]

        for shape in shapes:
            depth = np.random.rand(*shape) * 100
            normalized = normalize_depth_map(depth)
            assert normalized.shape == shape

    def test_normalize_3d_array(self):
        """Test normalization with 3D array."""
        depth = np.random.rand(10, 10, 3) * 100
        normalized = normalize_depth_map(depth)

        assert normalized.shape == depth.shape
        # Each channel should be normalized independently
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0


class TestDepthToDisparity:
    """Test depth to disparity conversion function."""

    def test_depth_to_disparity_basic(self):
        """Inverse depth maps linearly to disparity: nearer = larger disparity."""
        depth_map = np.array([[0.5, 0.75], [0.25, 1.0]])
        baseline = 0.1  # 10cm
        focal_length = 1000.0  # pixels

        disparity = depth_to_disparity(depth_map, baseline, focal_length)

        assert disparity.shape == depth_map.shape
        # nearest point (1.0) sits at STEREO_NEAR_PLANE_M = 1m -> baseline * focal px
        assert np.isclose(disparity[1, 1], 100.0)
        assert np.isclose(disparity[0, 0], 50.0)
        assert disparity[1, 1] > disparity[0, 1] > disparity[0, 0] > disparity[1, 0]

    def test_depth_to_disparity_bounded(self):
        """Regression #16: disparity never exceeds baseline * focal / near plane.

        The old formula divided by normalized depth, so the 0-valued pixel present in
        every min-max normalized frame was shifted by thousands of pixels.
        """
        depth_map = np.random.rand(64, 64)
        depth_map[0, 0] = 0.0
        depth_map[0, 1] = 1.0

        disparity = depth_to_disparity(depth_map, baseline=0.065, focal_length=1000.0)

        assert disparity.min() >= 0.0
        assert disparity.max() <= 65.0 + 1e-4
        assert disparity[0, 0] == 0.0  # farthest point -> infinity -> no parallax

    def test_depth_to_disparity_zero_depth_handling(self):
        """Test that zero depth values are handled safely."""
        depth_map = np.array([[0.0, 0.5], [1.0, 0.0]])
        baseline = 0.1
        focal_length = 1000.0

        # Should not raise division by zero error
        disparity = depth_to_disparity(depth_map, baseline, focal_length)

        assert not np.isnan(disparity).any()
        assert not np.isinf(disparity).any()

    def test_depth_to_disparity_very_small_depth(self):
        """Test handling of very small depth values."""
        depth_map = np.array([[0.0001, 0.0005], [0.001, 0.01]])
        baseline = 0.1
        focal_length = 1000.0

        disparity = depth_to_disparity(depth_map, baseline, focal_length)

        # Very small depths should produce high disparity values
        assert np.all(disparity > 0)
        assert not np.isnan(disparity).any()

    def test_depth_to_disparity_uniform_depth(self):
        """Test disparity conversion with uniform depth."""
        depth_map = np.full((50, 50), 0.5)
        baseline = 0.1
        focal_length = 1000.0

        disparity = depth_to_disparity(depth_map, baseline, focal_length)

        # All values should be equal for uniform depth
        assert np.allclose(disparity, disparity[0, 0])

    def test_depth_to_disparity_baseline_scaling(self):
        """Test that disparity scales with baseline."""
        depth_map = np.array([[0.5, 0.75]])
        focal_length = 1000.0

        disparity_1 = depth_to_disparity(depth_map, baseline=0.1, focal_length=focal_length)
        disparity_2 = depth_to_disparity(depth_map, baseline=0.2, focal_length=focal_length)

        # Doubling baseline should double disparity
        assert np.allclose(disparity_2, disparity_1 * 2)

    def test_depth_to_disparity_focal_length_scaling(self):
        """Test that disparity scales with focal length."""
        depth_map = np.array([[0.5, 0.75]])
        baseline = 0.1

        disparity_1 = depth_to_disparity(depth_map, baseline, focal_length=1000.0)
        disparity_2 = depth_to_disparity(depth_map, baseline, focal_length=2000.0)

        # Doubling focal length should double disparity
        assert np.allclose(disparity_2, disparity_1 * 2)

    def test_depth_to_disparity_preserves_shape(self):
        """Test that conversion preserves array shape."""
        shapes = [(100, 100), (50, 200), (1920, 1080)]

        for shape in shapes:
            depth_map = np.random.rand(*shape)
            disparity = depth_to_disparity(depth_map, baseline=0.1, focal_length=1000.0)
            assert disparity.shape == shape

    def test_depth_to_disparity_zero_baseline(self):
        """Test with zero baseline (edge case)."""
        depth_map = np.array([[0.5, 1.0]])
        disparity = depth_to_disparity(depth_map, baseline=0.0, focal_length=1000.0)

        # Zero baseline should produce zero disparity
        assert np.allclose(disparity, 0.0)

    def test_depth_to_disparity_zero_focal_length(self):
        """Test with zero focal length (edge case)."""
        depth_map = np.array([[0.5, 1.0]])
        disparity = depth_to_disparity(depth_map, baseline=0.1, focal_length=0.0)

        # Zero focal length should produce zero disparity
        assert np.allclose(disparity, 0.0)


class TestResizeImage:
    """Test image resizing function."""

    def test_resize_basic(self):
        """Test basic image resizing."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        resized = resize_image(image, 50, 50)

        assert resized.shape == (50, 50, 3)

    def test_resize_grayscale(self):
        """Test resizing grayscale image."""
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        resized = resize_image(image, 200, 150)

        assert resized.shape == (150, 200)

    def test_resize_upscale(self):
        """Test upscaling image."""
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        resized = resize_image(image, 100, 100)

        assert resized.shape == (100, 100, 3)

    def test_resize_different_aspect(self):
        """Test resizing to different aspect ratio."""
        image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        resized = resize_image(image, 150, 100)

        assert resized.shape == (100, 150, 3)


class TestApplyCenterCrop:
    """Test center crop function."""

    def test_center_crop_half(self):
        """Test cropping to half size."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cropped = apply_center_crop(image, crop_factor=0.5)

        assert cropped.shape == (50, 50, 3)

    def test_center_crop_no_crop(self):
        """Test with crop factor 1.0 (no cropping)."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cropped = apply_center_crop(image, crop_factor=1.0)

        assert cropped.shape == image.shape
        assert np.array_equal(cropped, image)

    def test_center_crop_greater_than_one(self):
        """Test with crop factor > 1.0 (no cropping)."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cropped = apply_center_crop(image, crop_factor=1.5)

        assert cropped.shape == image.shape

    def test_center_crop_grayscale(self):
        """Test center crop on grayscale image."""
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        cropped = apply_center_crop(image, crop_factor=0.75)

        assert cropped.shape == (75, 75)

    def test_center_crop_rectangular(self):
        """Test center crop on rectangular image."""
        image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        cropped = apply_center_crop(image, crop_factor=0.5)

        assert cropped.shape == (50, 100, 3)


class TestCreateVRFrame:
    """Test VR frame creation function."""

    def test_create_side_by_side(self):
        """Test side-by-side VR frame creation."""
        left = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        vr_frame = create_vr_frame(left, right, "side_by_side")

        assert vr_frame.shape == (100, 200, 3)

    def test_create_over_under(self):
        """Test over-under VR frame creation."""
        left = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        vr_frame = create_vr_frame(left, right, "over_under")

        assert vr_frame.shape == (200, 100, 3)

    def test_create_default_format(self):
        """Test VR frame with invalid format defaults to side_by_side."""
        left = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        vr_frame = create_vr_frame(left, right, "invalid_format")

        # Should default to side_by_side
        assert vr_frame.shape == (100, 200, 3)

    def test_create_grayscale_images(self):
        """Test VR frame with grayscale images."""
        left = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        right = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        vr_frame = create_vr_frame(left, right, "side_by_side")

        assert vr_frame.shape == (100, 200)


class TestValidateImageArray:
    """Test image array validation function."""

    def test_validate_color_image(self):
        """Test validation of color image."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        assert validate_image_array(image) is True

    def test_validate_grayscale_image(self):
        """Test validation of grayscale image."""
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        assert validate_image_array(image) is True

    def test_validate_rgba_image(self):
        """Test validation of RGBA image."""
        image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        assert validate_image_array(image) is True

    def test_validate_not_ndarray(self):
        """Test validation fails for non-ndarray."""
        assert validate_image_array([1, 2, 3]) is False

    def test_validate_wrong_dimensions(self):
        """Test validation fails for wrong dimensions."""
        image_1d = np.array([1, 2, 3])
        image_4d = np.random.rand(10, 10, 10, 10)

        assert validate_image_array(image_1d) is False
        assert validate_image_array(image_4d) is False

    def test_validate_wrong_channels(self):
        """Test validation fails for invalid channel count."""
        image = np.random.randint(0, 255, (100, 100, 5), dtype=np.uint8)
        assert validate_image_array(image) is False

    def test_validate_empty_array(self):
        """Test validation fails for empty array."""
        image = np.array([])
        assert validate_image_array(image) is False

    def test_validate_zero_size_image(self):
        """Test validation fails for zero-size image."""
        image = np.zeros((0, 0, 3), dtype=np.uint8)
        assert validate_image_array(image) is False

    def test_validate_zero_height_image(self):
        """Test validation fails for zero height image."""
        image = np.zeros((0, 100, 3), dtype=np.uint8)
        assert validate_image_array(image) is False


class TestCalculateImageStatistics:
    """Test image statistics calculation function."""

    def test_calculate_stats_grayscale(self):
        """Test statistics for grayscale image."""
        image = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        stats = calculate_image_statistics(image)

        assert stats["shape"] == (2, 2)
        assert stats["dtype"] == "uint8"
        assert stats["min"] == 10.0
        assert stats["max"] == 40.0
        assert stats["mean"] == 25.0

    def test_calculate_stats_color(self):
        """Test statistics for color image."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        stats = calculate_image_statistics(image)

        assert stats["shape"] == (100, 100, 3)
        assert stats["channels"] == 3
        assert "blue_mean" in stats
        assert "green_mean" in stats
        assert "red_mean" in stats

    def test_calculate_stats_invalid_image(self):
        """Test statistics returns empty dict for invalid image."""
        image = np.array([1, 2, 3])
        stats = calculate_image_statistics(image)

        assert stats == {}

    def test_calculate_stats_rgba(self):
        """Test statistics for RGBA image."""
        image = np.random.randint(0, 255, (50, 50, 4), dtype=np.uint8)
        stats = calculate_image_statistics(image)

        assert stats["channels"] == 4
        assert "alpha_mean" in stats


class TestCreateShiftedImage:
    """Test create_shifted_image function."""

    def test_shift_left(self):
        """Test left shift direction."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        disparity_map = np.full((100, 100), 5.0)

        shifted = create_shifted_image(image, disparity_map, direction="left")

        assert shifted.shape == image.shape
        assert shifted.dtype == image.dtype

    def test_shift_right(self):
        """Test right shift direction."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        disparity_map = np.full((100, 100), 5.0)

        shifted = create_shifted_image(image, disparity_map, direction="right")

        assert shifted.shape == image.shape
        assert shifted.dtype == image.dtype

    def test_shift_grayscale(self):
        """Test shifting grayscale image."""
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        disparity_map = np.full((100, 100), 5.0)

        shifted = create_shifted_image(image, disparity_map, direction="left")

        assert shifted.shape == image.shape
        assert len(shifted.shape) == 2

    def test_shift_varying_disparity(self):
        """Test with varying disparity values."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Create gradient disparity map
        disparity_map = np.linspace(0, 10, 10000).reshape(100, 100)

        shifted = create_shifted_image(image, disparity_map, direction="left")

        assert shifted.shape == image.shape


class TestCalculateFisheyeCoordinates:
    """Test calculate_fisheye_coordinates function."""

    def test_stereographic_projection(self):
        """Test stereographic projection mapping."""
        x_map, y_map = calculate_fisheye_coordinates(200, 200, 90.0, "stereographic")

        assert x_map.shape == (200, 200)
        assert y_map.shape == (200, 200)
        assert x_map.dtype == np.float32
        assert y_map.dtype == np.float32
        # Coordinates should be within bounds
        assert np.all(x_map >= 0) and np.all(x_map < 200)
        assert np.all(y_map >= 0) and np.all(y_map < 200)

    def test_equidistant_projection(self):
        """Test equidistant projection mapping."""
        x_map, y_map = calculate_fisheye_coordinates(200, 200, 90.0, "equidistant")

        assert x_map.shape == (200, 200)
        assert y_map.shape == (200, 200)

    def test_equisolid_projection(self):
        """Test equisolid projection mapping."""
        x_map, y_map = calculate_fisheye_coordinates(200, 200, 90.0, "equisolid")

        assert x_map.shape == (200, 200)
        assert y_map.shape == (200, 200)

    def test_orthogonal_projection(self):
        """Test orthogonal projection mapping."""
        x_map, y_map = calculate_fisheye_coordinates(200, 200, 90.0, "orthogonal")

        assert x_map.shape == (200, 200)
        assert y_map.shape == (200, 200)

    def test_different_fov(self):
        """Test with different field of view values."""
        x_map_90, y_map_90 = calculate_fisheye_coordinates(200, 200, 90.0, "stereographic")
        x_map_180, y_map_180 = calculate_fisheye_coordinates(200, 200, 180.0, "stereographic")

        # Different FOV should produce different mappings
        assert not np.array_equal(x_map_90, x_map_180)


class TestApplyFisheyeDistortion:
    """Test apply_fisheye_distortion function."""

    def test_apply_distortion_basic(self):
        """Test basic fisheye distortion application."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        distorted = apply_fisheye_distortion(image, fov_degrees=90.0)

        assert distorted.shape == image.shape
        assert distorted.dtype == image.dtype

    def test_apply_distortion_grayscale(self):
        """Test fisheye distortion on grayscale image."""
        image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)

        distorted = apply_fisheye_distortion(image, fov_degrees=90.0)

        assert distorted.shape == image.shape

    def test_different_projection_types(self):
        """Test different projection types."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        for proj_type in ["stereographic", "equidistant", "equisolid", "orthogonal"]:
            distorted = apply_fisheye_distortion(image, fov_degrees=90.0, projection_type=proj_type)
            assert distorted.shape == image.shape


class TestApplyFisheyeSquareCrop:
    """Test apply_fisheye_square_crop function."""

    def test_basic_crop(self):
        """Test basic fisheye square cropping."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        cropped = apply_fisheye_square_crop(image, 100, 100)

        assert cropped.shape == (100, 100, 3)

    def test_crop_with_factor(self):
        """Test cropping with different crop factors."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        cropped_full = apply_fisheye_square_crop(image, 100, 100, crop_factor=1.0)
        cropped_half = apply_fisheye_square_crop(image, 100, 100, crop_factor=0.5)

        assert cropped_full.shape == (100, 100, 3)
        assert cropped_half.shape == (100, 100, 3)

    def test_crop_rectangular_image(self):
        """Test cropping on rectangular image."""
        image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)

        cropped = apply_fisheye_square_crop(image, 150, 150)

        assert cropped.shape == (150, 150, 3)

    def test_crop_fallback_zero_crop_factor(self):
        """Test fallback when crop factor is zero (empty crop)."""
        # With crop_factor=0, effective_radius=0, resulting in empty crop
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # crop_factor=0 should result in empty cropped image, triggering fallback
        result = apply_fisheye_square_crop(image, 50, 50, crop_factor=0.0)

        # Should fallback to direct resize of original image
        assert result.shape == (50, 50, 3)


class TestHoleFillImage:
    """Test hole_fill_image function."""

    def test_no_holes(self):
        """Test image with no holes returns unchanged."""
        image = np.random.randint(1, 255, (100, 100, 3), dtype=np.uint8)

        filled = hole_fill_image(image)

        # Should return original image if no holes
        np.testing.assert_array_equal(filled, image)

    def test_fill_with_fast_method(self):
        """Test hole filling with fast method."""
        image = np.random.randint(1, 255, (100, 100, 3), dtype=np.uint8)
        # Create some holes (black pixels)
        image[40:60, 40:60] = 0

        filled = hole_fill_image(image, method="fast")

        assert filled.shape == image.shape
        # Holes should be filled (no longer all black)
        assert not np.all(filled[40:60, 40:60] == 0)

    def test_fill_with_advanced_method(self):
        """Test hole filling with advanced method."""
        image = np.random.randint(1, 255, (100, 100, 3), dtype=np.uint8)
        # Create some holes
        image[40:60, 40:60] = 0

        filled = hole_fill_image(image, method="advanced")

        assert filled.shape == image.shape
        assert not np.all(filled[40:60, 40:60] == 0)

    def test_fill_grayscale(self):
        """Test hole filling on grayscale image."""
        image = np.random.randint(1, 255, (100, 100), dtype=np.uint8)
        image[40:60, 40:60] = 0

        filled = hole_fill_image(image, method="fast")

        assert filled.shape == image.shape
        assert len(filled.shape) == 2

    def test_fill_with_custom_mask(self):
        """Test hole filling with custom mask."""
        image = np.random.randint(1, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 1  # Mark region as hole

        filled = hole_fill_image(image, mask=mask, method="fast")

        assert filled.shape == image.shape

    def test_fill_with_high_method(self):
        """Test hole filling with high quality method."""
        image = np.random.randint(1, 255, (100, 100, 3), dtype=np.uint8)
        # Create some holes
        image[40:60, 40:60] = 0

        filled = hole_fill_image(image, method="high")

        assert filled.shape == image.shape
        # Holes should be filled (no longer all black)
        assert not np.all(filled[40:60, 40:60] == 0)

    def test_fill_high_method_multipass(self):
        """Test high quality method with multiple holes to trigger multi-pass."""
        image = np.random.randint(1, 255, (200, 200, 3), dtype=np.uint8)
        # Create multiple scattered holes to trigger residual mask
        image[20:40, 20:40] = 0
        image[80:100, 80:100] = 0
        image[140:160, 140:160] = 0

        filled = hole_fill_image(image, method="high")

        assert filled.shape == image.shape
        # All holes should be filled
        assert not np.all(filled[20:40, 20:40] == 0)
        assert not np.all(filled[80:100, 80:100] == 0)
        assert not np.all(filled[140:160, 140:160] == 0)

    def test_empty_mask_handling(self):
        """Test that empty mask is handled correctly."""
        from src.depth_surge_3d.utils.imaging.image_processing import _create_hole_mask

        # Create image with no black pixels
        image = np.random.randint(1, 255, (50, 50, 3), dtype=np.uint8)
        mask = _create_hole_mask(image)

        # Mask should be all zeros
        assert not np.any(mask)

    def test_adaptive_radius_calculation(self):
        """Test adaptive radius calculation with various hole sizes."""
        from src.depth_surge_3d.utils.imaging.image_processing import (
            _create_hole_mask,
            _calculate_adaptive_radius,
        )

        # Small hole
        image_small = np.ones((100, 100, 3), dtype=np.uint8) * 128
        image_small[45:55, 45:55] = 0  # 10x10 hole
        mask_small = _create_hole_mask(image_small)
        radius_small = _calculate_adaptive_radius(mask_small)
        assert 3 <= radius_small <= 15

        # Large hole
        image_large = np.ones((100, 100, 3), dtype=np.uint8) * 128
        image_large[20:80, 20:80] = 0  # 60x60 hole
        mask_large = _create_hole_mask(image_large)
        radius_large = _calculate_adaptive_radius(mask_large)
        assert 3 <= radius_large <= 15
        assert radius_large > radius_small  # Larger hole should have larger radius

    def test_adaptive_radius_empty_mask(self):
        """Test adaptive radius with empty mask (no holes)."""
        from src.depth_surge_3d.utils.imaging.image_processing import _calculate_adaptive_radius

        # Empty mask (all zeros, only background)
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        radius = _calculate_adaptive_radius(empty_mask)

        # Should return minimum radius of 3
        assert radius == 3

    def test_high_quality_inpaint_with_residuals(self):
        """Test high quality inpainting that triggers second pass."""
        import cv2
        from unittest.mock import patch
        from src.depth_surge_3d.utils.imaging.image_processing import (
            _create_hole_mask,
            _apply_high_quality_inpaint,
        )

        # Create image with holes
        image = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        image[40:60, 40:60] = 0

        mask = _create_hole_mask(image)

        # Mock cv2.inpaint to simulate first pass leaving residuals
        original_inpaint = cv2.inpaint

        def mock_inpaint(img, mask, radius, method):
            # First call (INPAINT_NS): Leave some black pixels (residuals)
            if method == cv2.INPAINT_NS:
                result = original_inpaint(img, mask, radius, method)
                # Add some black pixels to simulate residuals
                result[45:47, 45:47] = 0
                return result
            # Second call (INPAINT_TELEA): Fill properly
            else:
                return original_inpaint(img, mask, radius, method)

        with patch("cv2.inpaint", side_effect=mock_inpaint):
            filled = _apply_high_quality_inpaint(image, mask, radius=10)

            assert filled.shape == image.shape
            # Should have filled the holes including residuals
            # Note: bilateral filter is applied, so values won't be exactly zero
            # but should be filled


class TestDepthToInverseDepth:
    """Test depth -> inverse depth conversion (DA3 / metric models)."""

    def test_nearer_is_larger(self):
        from src.depth_surge_3d.utils.imaging.image_processing import depth_to_inverse_depth

        depth = np.array([[1.0, 2.0], [4.0, 8.0]], dtype=np.float32)
        inverse = depth_to_inverse_depth(depth, low_percentile=0.0)

        assert np.allclose(inverse, 1.0 / depth)

    def test_invalid_pixels_become_far(self):
        from src.depth_surge_3d.utils.imaging.image_processing import depth_to_inverse_depth

        depth = np.array([[0.0, -1.0], [np.nan, 2.0]], dtype=np.float32)
        inverse = depth_to_inverse_depth(depth)

        assert inverse[0, 0] == 0.0 and inverse[0, 1] == 0.0 and inverse[1, 0] == 0.0
        assert np.isclose(inverse[1, 1], 0.5)

    def test_all_invalid(self):
        from src.depth_surge_3d.utils.imaging.image_processing import depth_to_inverse_depth

        assert not depth_to_inverse_depth(np.zeros((4, 4))).any()

    def test_outliers_clamped(self):
        from src.depth_surge_3d.utils.imaging.image_processing import depth_to_inverse_depth

        depth = np.full((100, 100), 5.0, dtype=np.float32)
        depth[0, 0] = 1e-9  # one bogus near-zero pixel
        inverse = depth_to_inverse_depth(depth)

        assert np.isclose(inverse.max(), 0.2)


class TestSuperSampleSize:
    """Test stereo warp resolution selection."""

    def test_none(self):
        from src.depth_surge_3d.utils.imaging.image_processing import calculate_super_sample_size

        assert calculate_super_sample_size(1376, 768, 2752, 1536, "none") == (1376, 768)

    def test_auto_matches_source_when_output_not_larger(self):
        """Regression #17: same-size output must not be upscaled."""
        from src.depth_surge_3d.utils.imaging.image_processing import calculate_super_sample_size

        assert calculate_super_sample_size(1376, 768, 1376, 768, "auto") == (1376, 768)

    def test_auto_covers_output_preserving_aspect(self):
        from src.depth_surge_3d.utils.imaging.image_processing import calculate_super_sample_size

        assert calculate_super_sample_size(1080, 1920, 1536, 1536, "auto") == (1536, 2731)

    def test_auto_capped_at_2x(self):
        from src.depth_surge_3d.utils.imaging.image_processing import calculate_super_sample_size

        assert calculate_super_sample_size(640, 360, 4096, 4096, "auto") == (1280, 720)

    def test_2x(self):
        from src.depth_surge_3d.utils.imaging.image_processing import calculate_super_sample_size

        assert calculate_super_sample_size(100, 50, 10, 10, "2x") == (200, 100)

    @pytest.mark.parametrize(
        "mode,src,expected",
        [
            ("1080p", (1280, 720), (1920, 1080)),
            ("4k", (1920, 1080), (3840, 2160)),
            ("4k", (1080, 1920), (2160, 3840)),  # portrait keeps aspect
            ("1080p", (3840, 2160), (3840, 2160)),  # never downscales
        ],
    )
    def test_ui_modes(self, mode, src, expected):
        """Web UI options: 'Force 1080p' / 'Force 4K'."""
        from src.depth_surge_3d.utils.imaging.image_processing import calculate_super_sample_size

        assert calculate_super_sample_size(*src, 1920, 1080, mode) == expected


class TestRenderStereoPair:
    """Test full stereo pair rendering."""

    def test_near_content_has_crossed_disparity(self):
        """A near object appears further right in the left eye than in the right eye."""
        from src.depth_surge_3d.utils.imaging.image_processing import render_stereo_pair

        frame = np.zeros((40, 200), dtype=np.uint8)
        frame[:, 95:105] = 255  # white bar in the middle
        depth = np.zeros((40, 200), dtype=np.float32)
        depth[:, 80:120] = 1.0  # bar region is nearest
        settings = {"baseline": 0.02, "focal_length": 1000, "hole_fill_quality": "none"}

        left, right = render_stereo_pair(frame, depth, settings)

        left_center = np.nonzero(left[20] > 127)[0].mean()
        right_center = np.nonzero(right[20] > 127)[0].mean()
        assert left_center - right_center == pytest.approx(20, abs=2)

    def test_disparity_scales_with_super_sampling(self):
        """Super sampling doesn't change the relative 3D strength."""
        from src.depth_surge_3d.utils.imaging.image_processing import render_stereo_pair

        frame = np.zeros((40, 200), dtype=np.uint8)
        frame[:, 95:105] = 255
        depth = np.zeros((40, 200), dtype=np.float32)
        depth[:, 80:120] = 1.0
        settings = {
            "baseline": 0.02,
            "focal_length": 1000,
            "hole_fill_quality": "none",
            "super_sample": "2x",
        }

        left, right = render_stereo_pair(frame, depth, settings)

        assert left.shape == (80, 400)
        left_center = np.nonzero(left[40] > 127)[0].mean()
        right_center = np.nonzero(right[40] > 127)[0].mean()
        assert left_center - right_center == pytest.approx(40, abs=3)


class TestOcclusionAwareWarp:
    """Forward warping with a z-buffer and background-side hole filling."""

    @staticmethod
    def _scene(width=120, height=6, bar=(50, 70), fg_disp=20.0):
        """Textured background (disparity 0) with a uniform near bar in front of it."""
        from src.depth_surge_3d.utils.imaging.image_processing import create_stereo_view  # noqa

        bg = (np.arange(width) % 7 * 30).astype(np.uint8)  # background texture
        image = np.tile(bg, (height, 1))
        image[:, bar[0] : bar[1]] = 255  # foreground bar
        disparity = np.zeros((height, width), np.float32)
        disparity[:, bar[0] : bar[1]] = fg_disp
        return image, disparity

    def test_nearer_pixel_wins(self):
        from src.depth_surge_3d.utils.imaging.image_processing import forward_warp_disparity

        disp = np.array([[0, 0, 0, 0, 4, 0, 0, 0, 0, 0]], np.float32)
        warped = forward_warp_disparity(disp, 0.5)
        # pixel 4 (disparity 4) moves 2 px right and covers the background pixels there
        assert warped[0, 6] == 4

    def test_disocclusion_is_hole(self):
        from src.depth_surge_3d.utils.imaging.image_processing import stereo_source_coordinates

        _, disp = self._scene()
        _, holes = stereo_source_coordinates(disp, "left")
        # left eye: the bar moves right by 10 px and reveals background on its left
        assert holes[0, 52:58].all()
        assert not holes[0, 75:110].any()

    def test_hole_filled_with_background_not_foreground(self):
        """Regression: backward warping smeared the foreground into revealed areas."""
        from src.depth_surge_3d.utils.imaging.image_processing import create_stereo_view

        image, disp = self._scene()
        view, holes = create_stereo_view(image, disp, "left", "fast")
        assert holes.any()
        assert (view[holes] < 255).all()  # no bar colour in the filled hole
        assert (view[:, 62:78] == 255).all()  # bar shifted by +10 px, intact

    def test_right_eye_mirrors(self):
        from src.depth_surge_3d.utils.imaging.image_processing import create_stereo_view

        image, disp = self._scene()
        view, holes = create_stereo_view(image, disp, "right", "fast")
        assert (view[:, 42:58] == 255).all()  # bar shifted by -10 px
        assert holes[0, 62:68].all() and (view[holes] < 255).all()

    def test_real_black_pixels_kept(self):
        """Regression: the old hole filler inpainted every pure-black pixel of the footage."""
        from src.depth_surge_3d.utils.imaging.image_processing import create_stereo_view

        image = np.full((20, 40, 3), 120, np.uint8)
        image[5:15, 10:30] = 0  # dark object in the footage
        view, holes = create_stereo_view(image, np.zeros((20, 40), np.float32), "left", "fast")
        assert not holes.any()
        assert (view[5:15, 10:30] == 0).all()

    def test_none_leaves_holes_black(self):
        from src.depth_surge_3d.utils.imaging.image_processing import create_stereo_view

        image, disp = self._scene()
        view, holes = create_stereo_view(image, disp, "left", "none")
        assert (view[holes] == 0).all()

    def test_advanced_only_touches_seams(self):
        from src.depth_surge_3d.utils.imaging.image_processing import create_stereo_view

        image, disp = self._scene(width=200, bar=(80, 120))
        fast, holes = create_stereo_view(image, disp, "left", "fast")
        adv, _ = create_stereo_view(image, disp, "left", "advanced")
        far_from_holes = np.ones_like(holes)
        far_from_holes[:, 70:110] = False
        assert np.array_equal(fast[far_from_holes], adv[far_from_holes])

    def test_crack_bands_merged(self):
        """Steep depth ramps leave interleaved 1 px cracks; they become one hole."""
        from src.depth_surge_3d.utils.imaging.image_processing import stereo_source_coordinates

        disp = np.zeros((1, 80), np.float32)
        disp[0, 30:40] = np.linspace(0, 30, 10)  # soft edge: 3 px of disparity per px
        disp[0, 40:] = 30
        _, holes = stereo_source_coordinates(disp, "left")
        run = holes[0].astype(int)
        assert np.count_nonzero(np.diff(run) == 1) == 1  # a single contiguous hole


class TestFittedFisheye:
    """Regression: the fisheye mapping sampled outside the source and clamped into streaks."""

    @pytest.mark.parametrize(
        "projection", ["stereographic", "equidistant", "equisolid", "orthogonal"]
    )
    @pytest.mark.parametrize("crop", [0.5, 0.7, 1.0])
    def test_kept_circle_samples_real_content(self, projection, crop):
        w, h = 320, 180
        x_map, y_map = calculate_fisheye_coordinates(w, h, 180, projection, fit_radius=crop)
        yy, xx = np.mgrid[0:h, 0:w]
        inside_circle = np.hypot(xx - w / 2, yy - h / 2) <= crop * h / 2 - 1
        assert (y_map[inside_circle] >= -0.5).all() and (y_map[inside_circle] <= h - 0.5).all()
        assert (x_map[inside_circle] >= -0.5).all() and (x_map[inside_circle] <= w - 0.5).all()

    def test_circle_edge_hits_source_edge(self):
        w, h = 320, 180
        _, y_map = calculate_fisheye_coordinates(w, h, 180, "stereographic", fit_radius=0.7)
        top_of_circle = int(round(h / 2 - 0.7 * h / 2))
        assert y_map[top_of_circle, w // 2] == pytest.approx(0, abs=1.5)

    def test_outside_source_is_black(self):
        img = np.full((90, 160, 3), 200, np.uint8)
        out = apply_fisheye_distortion(img, 180, "stereographic", fit_radius=0.7)
        assert (out[0, 0] == 0).all()  # corner lies outside the source: black, not streaked
        assert (out[45, 80] == 200).all()

    def test_legacy_mapping_unchanged(self):
        x_map, y_map = calculate_fisheye_coordinates(64, 64, 180, "stereographic")
        assert x_map.min() >= 0 and x_map.max() <= 63 and y_map.max() <= 63

    def test_maps_cached(self):
        from src.depth_surge_3d.utils.imaging.image_processing import _cached_fisheye_maps

        a = _cached_fisheye_maps(64, 48, 180.0, "equidistant", 0.7)
        b = _cached_fisheye_maps(64, 48, 180.0, "equidistant", 0.7)
        assert a[0] is b[0] and not a[0].flags.writeable


class TestFitToCanvas:
    """Regression: per-eye scaling stretched content whose aspect didn't match the target."""

    def test_same_aspect_is_plain_resize(self):
        from src.depth_surge_3d.utils.imaging.image_processing import fit_to_canvas

        out = fit_to_canvas(np.full((90, 160, 3), 200, np.uint8), 1280, 720)
        assert out.shape == (720, 1280, 3) and (out == 200).all()

    def test_square_into_wide_is_pillarboxed(self):
        from src.depth_surge_3d.utils.imaging.image_processing import fit_to_canvas

        out = fit_to_canvas(np.full((100, 100, 3), 200, np.uint8), 160, 90)
        assert out.shape == (90, 160, 3)
        assert (out[:, 35:125] == 200).all()  # 90x90 square centred
        assert (out[:, :35] == 0).all() and (out[:, 125:] == 0).all()

    def test_portrait_into_wide(self):
        from src.depth_surge_3d.utils.imaging.image_processing import fit_to_canvas

        out = fit_to_canvas(np.full((160, 90), 50, np.uint8), 160, 90)
        assert out.shape == (90, 160)
        assert out[45, 80] == 50 and out[45, 0] == 0

    def test_fisheye_crop_not_stretched(self):
        """A square fisheye crop into a 16:9 eye keeps the circle round."""
        img = np.full((100, 200, 3), 255, np.uint8)
        out = apply_fisheye_square_crop(img, 320, 180, crop_factor=1.0)
        assert out.shape == (180, 320, 3)
        assert (out[:, :60] == 0).all() and (out[:, 70:250] == 255).all()


class TestFisheyeFrameFit:
    """Zooming the fisheye view out shows more of a frame that isn't square."""

    def _source_x_at_view_edge(self, frame_fit, width=1920, height=1080):
        from src.depth_surge_3d.utils.imaging.image_processing import calculate_fisheye_coordinates

        x_map, _ = calculate_fisheye_coordinates(
            width, height, 180, "stereographic", 0.7, frame_fit
        )
        # right edge of the square view, on the horizontal centre line
        edge = int(width / 2 + 0.7 * height / 2)
        return x_map[height // 2, edge] - width / 2

    def test_fill_reaches_only_as_far_as_the_frame_is_high(self):
        assert self._source_x_at_view_edge(0.0) == pytest.approx(540, abs=3)

    def test_whole_frame_reaches_the_side_edge(self):
        assert self._source_x_at_view_edge(1.0) == pytest.approx(960, abs=5)

    def test_half_way(self):
        assert self._source_x_at_view_edge(0.5) == pytest.approx(750, abs=5)

    def test_zoom_factor(self):
        from src.depth_surge_3d.utils.imaging.image_processing import fisheye_zoom_out

        assert fisheye_zoom_out(1920, 1080, 0.0) == 1.0
        assert fisheye_zoom_out(1920, 1080, 1.0) == pytest.approx(16 / 9)
        assert fisheye_zoom_out(1080, 1920, 1.0) == pytest.approx(16 / 9)
        assert fisheye_zoom_out(1080, 1080, 1.0) == 1.0
        assert fisheye_zoom_out(1920, 1080, 7.0) == pytest.approx(16 / 9)
        assert fisheye_zoom_out(0, 0, 1.0) == 1.0

    def test_whole_frame_leaves_black_above_and_below(self):
        from src.depth_surge_3d.utils.imaging.image_processing import (
            apply_fisheye_distortion,
            apply_fisheye_square_crop,
        )

        frame = np.full((180, 320, 3), 200, np.uint8)
        filled = apply_fisheye_square_crop(
            apply_fisheye_distortion(frame, 180, "stereographic", 0.7, 0.0), 96, 96, 0.7
        )
        whole = apply_fisheye_square_crop(
            apply_fisheye_distortion(frame, 180, "stereographic", 0.7, 1.0), 96, 96, 0.7
        )

        assert filled[2, 48].min() > 150 and filled[48, 2].min() > 150  # view is filled
        assert whole[2, 48].max() == 0 and whole[93, 48].max() == 0  # black bars
        assert whole[48, 2].min() > 150 and whole[48, 48].min() > 150  # full width shown


class TestStereoDepthStats:
    def test_flat_depth_has_no_shift(self):
        stats = stereo_depth_stats(np.full((20, 30), 0.43, dtype=np.float32), 0.1, 1200)
        assert stats == {"depth_spread": 0.0, "shift_px": 0.0}

    def test_full_range_gives_maximum_shift(self):
        depth = np.tile(np.linspace(0, 1, 100, dtype=np.float32), (10, 1))
        stats = stereo_depth_stats(depth, 0.1, 1200)
        assert stats["depth_spread"] == pytest.approx(0.96, abs=0.02)
        assert stats["shift_px"] == pytest.approx(0.96 * 120, abs=3)

    def test_shift_scales_with_baseline(self):
        depth = np.tile(np.linspace(0.2, 0.6, 100, dtype=np.float32), (10, 1))
        narrow = stereo_depth_stats(depth, 0.05, 1000)["shift_px"]
        wide = stereo_depth_stats(depth, 0.1, 1000)["shift_px"]
        assert wide == pytest.approx(2 * narrow, rel=0.01)

    def test_empty_and_invalid_input(self):
        assert stereo_depth_stats(np.empty((0, 0)), 0.1, 1200)["shift_px"] == 0.0
        assert stereo_depth_stats(np.full((4, 4), np.nan), 0.1, 1200)["shift_px"] == 0.0
