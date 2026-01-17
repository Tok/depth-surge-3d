"""Unit tests for image processing utilities."""

import numpy as np
import cv2
from src.depth_surge_3d.utils.image_processing import (
    normalize_depth_map,
    depth_to_disparity,
    resize_image,
    apply_center_crop,
    create_vr_frame,
    validate_image_array,
    calculate_image_statistics,
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
