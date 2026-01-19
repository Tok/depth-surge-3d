"""Unit tests for resolution utilities."""

import pytest

from src.depth_surge_3d.utils.domain.resolution import (
    get_resolution_dimensions,
    validate_resolution_settings,
    get_available_resolutions,
    parse_custom_resolution,
    calculate_vr_output_dimensions,
    calculate_aspect_ratio,
    classify_aspect_ratio,
    auto_detect_resolution,
    get_format_recommendation,
)


class TestResolutionUtils:
    """Test resolution utility functions."""

    def test_get_resolution_dimensions_named(self):
        """Test getting dimensions for named resolutions."""
        width, height = get_resolution_dimensions("square-1k")
        assert width == 1080
        assert height == 1080

        width, height = get_resolution_dimensions("16x9-1080p")
        assert width == 1920
        assert height == 1080

    def test_get_resolution_dimensions_custom(self):
        """Test getting dimensions for custom resolution."""
        width, height = get_resolution_dimensions("custom:1920x1080")
        assert width == 1920
        assert height == 1080

        width, height = get_resolution_dimensions("custom:3840x2160")
        assert width == 3840
        assert height == 2160

    def test_get_resolution_dimensions_invalid(self):
        """Test getting dimensions for invalid resolution raises ValueError."""
        with pytest.raises(ValueError):
            get_resolution_dimensions("invalid-resolution")

    def test_get_resolution_dimensions_invalid_custom(self):
        """Test getting dimensions for invalid custom format raises ValueError."""
        with pytest.raises(ValueError):
            get_resolution_dimensions("custom:invalid")

    def test_validate_resolution_settings_valid(self):
        """Test validation of valid resolution settings."""
        result = validate_resolution_settings("square-1k", "side_by_side", 1920, 1080)
        assert isinstance(result, dict)
        assert "valid" in result or "width" in result

    def test_validate_resolution_settings_returns_dict(self):
        """Test validation returns a dictionary with info."""
        result = validate_resolution_settings("16x9-1080p", "over_under", 1920, 1080)
        assert isinstance(result, dict)

    def test_get_available_resolutions_structure(self):
        """Test available resolutions structure."""
        resolutions = get_available_resolutions()

        assert isinstance(resolutions, dict)
        # Check it has some categories (may vary by implementation)
        assert len(resolutions) > 0

        # Check each category has proper structure
        for category, items in resolutions.items():
            assert isinstance(items, list)
            if items:  # If category has items
                for item in items:
                    assert "name" in item
                    assert "description" in item

    def test_get_available_resolutions_contains_known_resolutions(self):
        """Test that available resolutions contain known formats."""
        resolutions = get_available_resolutions()

        all_names = []
        for category in resolutions.values():
            all_names.extend([item["name"] for item in category])

        # Check some known resolutions are present
        assert "square-1k" in all_names
        assert "16x9-1080p" in all_names
        assert "16x9-4k" in all_names


class TestParseCustomResolution:
    """Test parse_custom_resolution function."""

    def test_parse_valid_custom_resolution(self):
        """Test parsing valid custom resolution."""
        result = parse_custom_resolution("custom:1920x1080")
        assert result == (1920, 1080)

    def test_parse_large_custom_resolution(self):
        """Test parsing large custom resolution."""
        result = parse_custom_resolution("custom:3840x2160")
        assert result == (3840, 2160)

    def test_parse_invalid_format(self):
        """Test parsing invalid format returns None."""
        assert parse_custom_resolution("1920x1080") is None  # Missing 'custom:' prefix

    def test_parse_invalid_dimensions(self):
        """Test parsing invalid dimensions returns None."""
        assert parse_custom_resolution("custom:abc123") is None

    def test_parse_negative_dimensions(self):
        """Test parsing negative dimensions returns None."""
        assert parse_custom_resolution("custom:-1920x1080") is None
        assert parse_custom_resolution("custom:1920x-1080") is None

    def test_parse_zero_dimensions(self):
        """Test parsing zero dimensions returns None."""
        assert parse_custom_resolution("custom:0x1080") is None
        assert parse_custom_resolution("custom:1920x0") is None

    def test_parse_oversized_dimensions(self):
        """Test parsing oversized dimensions (>10000) returns None."""
        assert parse_custom_resolution("custom:15000x1080") is None
        assert parse_custom_resolution("custom:1920x15000") is None


class TestCalculateVROutputDimensions:
    """Test calculate_vr_output_dimensions function."""

    def test_side_by_side_format(self):
        """Test side-by-side format doubles width."""
        width, height = calculate_vr_output_dimensions(1920, 1080, "side_by_side")
        assert width == 3840  # 1920 * 2
        assert height == 1080

    def test_over_under_format(self):
        """Test over-under format doubles height."""
        width, height = calculate_vr_output_dimensions(1920, 1080, "over_under")
        assert width == 1920
        assert height == 2160  # 1080 * 2

    def test_invalid_format_defaults_to_side_by_side(self):
        """Test invalid format defaults to side_by_side."""
        width, height = calculate_vr_output_dimensions(1920, 1080, "invalid_format")
        assert width == 3840  # Defaults to side_by_side
        assert height == 1080


class TestCalculateAspectRatio:
    """Test calculate_aspect_ratio function."""

    def test_standard_16x9(self):
        """Test calculating 16:9 aspect ratio."""
        aspect = calculate_aspect_ratio(1920, 1080)
        assert abs(aspect - 1.7778) < 0.001  # ~16/9

    def test_square_aspect(self):
        """Test calculating square aspect ratio."""
        aspect = calculate_aspect_ratio(1080, 1080)
        assert aspect == 1.0

    def test_ultra_wide(self):
        """Test calculating ultra-wide aspect ratio."""
        aspect = calculate_aspect_ratio(3440, 1440)
        assert abs(aspect - 2.3889) < 0.001

    def test_zero_height_returns_one(self):
        """Test zero height returns 1.0 to avoid division by zero."""
        aspect = calculate_aspect_ratio(1920, 0)
        assert aspect == 1.0


class TestClassifyAspectRatio:
    """Test classify_aspect_ratio function."""

    def test_classify_ultra_wide(self):
        """Test classifying ultra-wide aspect ratios."""
        assert classify_aspect_ratio(2.5) == "ultra_wide"
        assert classify_aspect_ratio(3.0) == "ultra_wide"

    def test_classify_wide(self):
        """Test classifying wide aspect ratios."""
        assert classify_aspect_ratio(1.7778) == "wide"  # 16:9
        assert classify_aspect_ratio(1.6) == "wide"

    def test_classify_standard(self):
        """Test classifying standard aspect ratios."""
        assert classify_aspect_ratio(1.333) == "standard"  # 4:3
        assert classify_aspect_ratio(1.5) == "standard"

    def test_classify_square(self):
        """Test classifying square aspect ratios."""
        assert classify_aspect_ratio(1.0) == "standard"
        assert classify_aspect_ratio(0.9) == "standard"


class TestAutoDetectResolution:
    """Test auto_detect_resolution function."""

    def test_auto_detect_4k_ultra_wide(self):
        """Test auto-detection for 4K ultra-wide source."""
        # 3840x2160 ultra-wide cropped to 3840x1600 = ~6M pixels
        # Need > 8M pixels for cinema-4k, so use 5120x2160 = ~11M pixels
        result = auto_detect_resolution(5120, 2160, "side_by_side")
        assert result == "cinema-4k"

    def test_auto_detect_2k_ultra_wide(self):
        """Test auto-detection for 2K ultra-wide source."""
        # 2560x1080 is ~2.7M pixels, ultra-wide aspect
        result = auto_detect_resolution(2560, 1080, "side_by_side")
        assert result == "cinema-2k"

    def test_auto_detect_4k_wide(self):
        """Test auto-detection for 4K wide source."""
        # 3840x2160 is ~8.3M pixels, 16:9 aspect
        result = auto_detect_resolution(3840, 2160, "side_by_side")
        assert result == "16x9-4k"

    def test_auto_detect_1080p_wide(self):
        """Test auto-detection for 1080p wide source."""
        # 1920x1080 is ~2M pixels, 16:9 aspect
        result = auto_detect_resolution(1920, 1080, "side_by_side")
        assert result == "16x9-1080p"

    def test_auto_detect_720p_wide(self):
        """Test auto-detection for 720p wide source."""
        # 1280x720 is ~0.9M pixels, 16:9 aspect
        result = auto_detect_resolution(1280, 720, "side_by_side")
        assert result == "16x9-720p"

    def test_auto_detect_4k_square(self):
        """Test auto-detection for 4K square/standard source."""
        # 3840x3840 is huge, but standard aspect
        result = auto_detect_resolution(3840, 3840, "side_by_side")
        assert result == "square-4k"

    def test_auto_detect_2k_square(self):
        """Test auto-detection for 2K square/standard source."""
        # 2048x2048 is ~4M pixels, standard aspect
        result = auto_detect_resolution(2048, 2048, "side_by_side")
        assert result == "square-2k"

    def test_auto_detect_1k_square(self):
        """Test auto-detection for 1K square/standard source."""
        # 1280x960 is ~1.2M pixels, standard aspect
        result = auto_detect_resolution(1280, 960, "side_by_side")
        assert result == "square-1k"


class TestGetFormatRecommendation:
    """Test get_format_recommendation function."""

    def test_recommend_over_under_for_ultra_wide(self):
        """Test recommending over_under for ultra-wide content."""
        # Ultra-wide aspect ratio >= 2.0
        assert get_format_recommendation(2.5) == "over_under"
        assert get_format_recommendation(3.0) == "over_under"

    def test_recommend_side_by_side_for_standard(self):
        """Test recommending side_by_side for standard/wide content."""
        assert get_format_recommendation(1.0) == "side_by_side"  # Square
        assert get_format_recommendation(1.333) == "side_by_side"  # 4:3
        assert get_format_recommendation(1.7778) == "side_by_side"  # 16:9

    def test_recommend_side_by_side_for_wide(self):
        """Test recommending side_by_side for wide but not ultra-wide."""
        assert get_format_recommendation(1.85) == "side_by_side"


class TestValidateResolutionSettingsAdvanced:
    """Test advanced validate_resolution_settings scenarios."""

    def test_validate_with_auto_resolution(self):
        """Test validation with auto resolution detection."""
        result = validate_resolution_settings("auto", "side_by_side", 1920, 1080)

        assert result["valid"] is True
        assert result["final_resolution"] is not None
        assert len(result["recommendations"]) > 0
        # Should have auto-detection recommendation
        assert any("Auto-detected" in rec for rec in result["recommendations"])

    def test_validate_ultra_wide_with_wrong_format(self):
        """Test validation warns about ultra-wide content with wrong format."""
        # 3440x1440 is ultra-wide (aspect ~2.4)
        result = validate_resolution_settings("16x9-1080p", "side_by_side", 3440, 1440)

        assert result["valid"] is True
        assert len(result["warnings"]) > 0
        # Should warn about format
        assert any("over_under" in warning for warning in result["warnings"])

    def test_validate_with_invalid_resolution(self):
        """Test validation with invalid resolution."""
        result = validate_resolution_settings("invalid-res", "side_by_side", 1920, 1080)

        assert result["valid"] is False
        assert len(result["warnings"]) > 0

    def test_validate_normal_wide_content(self):
        """Test validation doesn't warn for normal wide content."""
        # 1920x1080 is 16:9, not ultra-wide
        result = validate_resolution_settings("16x9-1080p", "side_by_side", 1920, 1080)

        assert result["valid"] is True
        # Should not have format warnings (aspect ratio < 2.0)
        format_warnings = [w for w in result["warnings"] if "over_under" in w]
        assert len(format_warnings) == 0
