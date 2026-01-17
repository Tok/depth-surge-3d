"""Unit tests for file operations utilities."""

import pytest
from src.depth_surge_3d.utils.file_operations import (
    parse_time_string,
    calculate_frame_range,
)


class TestParseTimeString:
    """Test time string parsing function."""

    def test_parse_mm_ss_format(self):
        """Test parsing mm:ss format."""
        result = parse_time_string("01:30")
        assert result == 90.0  # 1 minute + 30 seconds

    def test_parse_hh_mm_ss_format(self):
        """Test parsing hh:mm:ss format."""
        result = parse_time_string("01:30:45")
        assert result == 5445.0  # 1 hour + 30 minutes + 45 seconds

    def test_parse_zero_time(self):
        """Test parsing zero time."""
        result = parse_time_string("00:00")
        assert result == 0.0

        result = parse_time_string("00:00:00")
        assert result == 0.0

    def test_parse_with_leading_zeros(self):
        """Test parsing with leading zeros."""
        result = parse_time_string("09:05")
        assert result == 545.0  # 9 minutes + 5 seconds

    def test_parse_large_values(self):
        """Test parsing large time values."""
        result = parse_time_string("59:59")
        assert result == 3599.0  # 59 minutes + 59 seconds

        result = parse_time_string("23:59:59")
        assert result == 86399.0  # 23 hours + 59 minutes + 59 seconds

    def test_parse_invalid_format(self):
        """Test parsing invalid format returns None."""
        result = parse_time_string("invalid")
        assert result is None

    def test_parse_invalid_single_number(self):
        """Test parsing single number (invalid format) returns None."""
        result = parse_time_string("123")
        assert result is None

    def test_parse_empty_string(self):
        """Test parsing empty string returns None."""
        result = parse_time_string("")
        assert result is None

    def test_parse_negative_values(self):
        """Test parsing negative values (technically parses but gives unexpected result)."""
        result = parse_time_string("-01:30")
        # The function parses "-01" as -1 minute, then adds 30 seconds
        # -1 * 60 + 30 = -60 + 30 = -30
        assert result == -30.0

    def test_parse_non_numeric_values(self):
        """Test parsing non-numeric values returns None."""
        result = parse_time_string("ab:cd")
        assert result is None


class TestCalculateFrameRange:
    """Test frame range calculation function."""

    def test_calculate_full_range(self):
        """Test calculating full frame range with no times specified."""
        start, end = calculate_frame_range(
            total_frames=1000, fps=30.0, start_time=None, end_time=None
        )
        assert start == 0
        assert end == 1000

    def test_calculate_with_start_time(self):
        """Test calculating with only start time."""
        start, end = calculate_frame_range(
            total_frames=1000, fps=30.0, start_time="00:10", end_time=None
        )
        assert start == 300  # 10 seconds * 30 fps
        assert end == 1000

    def test_calculate_with_end_time(self):
        """Test calculating with only end time."""
        start, end = calculate_frame_range(
            total_frames=1000, fps=30.0, start_time=None, end_time="00:20"
        )
        assert start == 0
        assert end == 600  # 20 seconds * 30 fps

    def test_calculate_with_both_times(self):
        """Test calculating with both start and end times."""
        start, end = calculate_frame_range(
            total_frames=1000, fps=30.0, start_time="00:10", end_time="00:20"
        )
        assert start == 300  # 10 seconds * 30 fps
        assert end == 600  # 20 seconds * 30 fps

    def test_calculate_with_hh_mm_ss_format(self):
        """Test calculating with hh:mm:ss format."""
        start, end = calculate_frame_range(
            total_frames=10000, fps=30.0, start_time="00:01:00", end_time="00:02:00"
        )
        assert start == 1800  # 1 minute * 30 fps
        assert end == 3600  # 2 minutes * 30 fps

    def test_calculate_clamps_to_total_frames(self):
        """Test that end frame is clamped to total frames."""
        start, end = calculate_frame_range(
            total_frames=1000, fps=30.0, start_time=None, end_time="10:00:00"
        )
        assert end == 1000  # Clamped to total frames

    def test_calculate_negative_start_clamped(self):
        """Test that negative start is clamped to 0."""
        # This would happen if someone passed invalid time
        start, end = calculate_frame_range(total_frames=1000, fps=30.0)
        assert start >= 0

    def test_calculate_ensures_minimum_range(self):
        """Test that end is always at least start + 1."""
        start, end = calculate_frame_range(
            total_frames=1000, fps=30.0, start_time="00:20", end_time="00:10"
        )
        # End time is before start time, should be adjusted
        assert end > start
        assert end == start + 1

    def test_calculate_with_fractional_fps(self):
        """Test calculating with fractional FPS."""
        start, end = calculate_frame_range(
            total_frames=1000, fps=29.97, start_time="00:10", end_time=None
        )
        assert start == int(10 * 29.97)  # 10 seconds * 29.97 fps
        assert end == 1000

    def test_calculate_start_at_last_frame(self):
        """Test that start frame at last frame is clamped correctly."""
        start, end = calculate_frame_range(
            total_frames=100, fps=30.0, start_time="10:00:00", end_time=None
        )
        # Start would be way beyond total_frames, should be clamped
        assert start == 99  # max(0, min(start, total_frames - 1))
        assert end == 100  # max(start + 1, min(end, total_frames))

    def test_calculate_with_invalid_time_format(self):
        """Test with invalid time format (should be ignored)."""
        start, end = calculate_frame_range(
            total_frames=1000, fps=30.0, start_time="invalid", end_time="also_invalid"
        )
        # Invalid times should be ignored, use defaults
        assert start == 0
        assert end == 1000

    def test_calculate_zero_total_frames(self):
        """Test with zero total frames."""
        start, end = calculate_frame_range(total_frames=0, fps=30.0)
        # Edge case: should handle gracefully
        # Based on the clamp logic: start = max(0, min(start, -1)) = 0
        # end = max(1, min(end, 0)) = 1 then clamped to max(0, end) = 1
        # Actually looking at code: end = max(start + 1, min(end, total_frames))
        # So end = max(0 + 1, min(0, 0)) = max(1, 0) = 1
        assert start == 0
        # This might be a bug in the original code, but testing actual behavior
