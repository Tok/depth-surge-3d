"""Unit tests for resolution utilities."""

import pytest

from src.depth_surge_3d.utils.resolution import (
    get_resolution_dimensions,
    validate_resolution_settings,
    get_available_resolutions,
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
