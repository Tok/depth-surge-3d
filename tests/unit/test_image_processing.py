"""Unit tests for image processing utilities."""

import numpy as np
from src.depth_surge_3d.utils.image_processing import (
    normalize_depth_map,
    depth_to_disparity,
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
        """Test basic depth to disparity conversion."""
        depth_map = np.array([[0.5, 0.75], [0.25, 1.0]])
        baseline = 0.1  # 10cm
        focal_length = 1000.0  # pixels

        disparity = depth_to_disparity(depth_map, baseline, focal_length)

        # Disparity should be inversely proportional to depth
        # Closer objects (lower depth values) should have higher disparity
        assert disparity.shape == depth_map.shape
        assert np.all(disparity > 0)

        # Check inverse relationship: lower depth = higher disparity
        assert disparity[1, 0] > disparity[0, 0]  # 0.25 depth > 0.5 depth disparity

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
