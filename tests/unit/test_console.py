"""Unit tests for console output utilities."""

from unittest.mock import patch
from io import StringIO

from src.depth_surge_3d.utils.system.console import (
    Colors,
    success,
    error,
    warning,
    info,
    dim,
    bold,
    step_complete,
    saved_to,
    title_bar,
    completion_banner,
)


class TestColorFormatting:
    """Test color formatting functions."""

    def test_success_formatting(self):
        """Test success message formatting."""
        result = success("Test message")
        assert "Test message" in result
        assert Colors.LIME_GREEN in result
        assert Colors.RESET in result

    def test_error_formatting(self):
        """Test error message formatting."""
        result = error("Error message")
        assert "Error message" in result
        assert Colors.RED in result
        assert Colors.RESET in result

    def test_warning_formatting(self):
        """Test warning message formatting."""
        result = warning("Warning message")
        assert "Warning message" in result
        assert Colors.YELLOW in result
        assert Colors.RESET in result

    def test_info_formatting(self):
        """Test info message formatting."""
        result = info("Info message")
        assert "Info message" in result
        assert Colors.BLUE in result
        assert Colors.RESET in result

    def test_dim_formatting(self):
        """Test dimmed text formatting."""
        result = dim("Dimmed text")
        assert "Dimmed text" in result
        assert Colors.GRAY in result
        assert Colors.RESET in result

    def test_bold_formatting(self):
        """Test bold text formatting."""
        result = bold("Bold text")
        assert "Bold text" in result
        assert Colors.BOLD in result
        assert Colors.RESET in result


class TestStepFormatting:
    """Test step completion and saved_to formatting."""

    def test_step_complete_formatting(self):
        """Test step completion message formatting."""
        result = step_complete("Step 1 complete")
        assert "Step 1 complete" in result
        assert "->" in result or "→" in result
        assert Colors.LIME_GREEN in result
        assert Colors.RESET in result

    def test_saved_to_formatting(self):
        """Test saved_to message formatting."""
        result = saved_to("Saved to /path/to/file")
        assert "Saved to /path/to/file" in result
        assert "->" in result or "→" in result
        assert Colors.ELECTRIC_BLUE in result
        assert Colors.RESET in result

    def test_title_bar_with_equals(self):
        """Test title bar formatting with === markers."""
        result = title_bar("=== Test Title ===")
        assert "Test Title" in result
        assert Colors.LIME_GREEN in result
        assert Colors.RESET in result

    def test_title_bar_without_equals(self):
        """Test title bar formatting without === markers."""
        result = title_bar("Plain Title")
        assert result == "Plain Title"


class TestCompletionBanner:
    """Test completion banner output."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_completion_banner_output(self, mock_stdout):
        """Test completion banner prints correct output."""
        completion_banner(
            output_file="/path/to/output.mp4",
            processing_time="1h 23m 45s",
            num_frames=300,
            vr_format="side_by_side",
        )

        output = mock_stdout.getvalue()

        # Check for key components
        assert "PROCESSING COMPLETE" in output
        assert "/path/to/output.mp4" in output
        assert "1h 23m 45s" in output
        assert "300" in output
        assert "side_by_side" in output
        assert "══" in output  # Border characters
        assert "──" in output  # Separator characters

    @patch("sys.stdout", new_callable=StringIO)
    def test_completion_banner_with_over_under(self, mock_stdout):
        """Test completion banner with over_under format."""
        completion_banner(
            output_file="/test/video.mp4",
            processing_time="5m 30s",
            num_frames=150,
            vr_format="over_under",
        )

        output = mock_stdout.getvalue()
        assert "over_under" in output
        assert "150" in output
        assert "5m 30s" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_completion_banner_with_long_path(self, mock_stdout):
        """Test completion banner handles long file paths."""
        long_path = "/very/long/path/to/some/deeply/nested/output/directory/video_file.mp4"
        completion_banner(
            output_file=long_path,
            processing_time="2h 15m",
            num_frames=1000,
            vr_format="side_by_side",
        )

        output = mock_stdout.getvalue()
        assert long_path in output
        assert "1000" in output


class TestColorsDisable:
    """Test Colors.disable() functionality."""

    def test_colors_disable(self):
        """Test that Colors.disable() removes all color codes."""
        # Save original colors
        original_lime = Colors.LIME_GREEN
        original_reset = Colors.RESET

        # Disable colors
        Colors.disable()

        # Check all colors are empty strings
        assert Colors.LIME_GREEN == ""
        assert Colors.GREEN == ""
        assert Colors.ELECTRIC_BLUE == ""
        assert Colors.RED == ""
        assert Colors.YELLOW == ""
        assert Colors.BLUE == ""
        assert Colors.GRAY == ""
        assert Colors.BOLD == ""
        assert Colors.DIM == ""
        assert Colors.RESET == ""

        # Restore colors for other tests
        Colors.LIME_GREEN = original_lime
        Colors.GREEN = "\033[38;2;0;255;65m"
        Colors.ELECTRIC_BLUE = "\033[38;2;0;217;255m"
        Colors.RED = "\033[91m"
        Colors.YELLOW = "\033[93m"
        Colors.BLUE = "\033[94m"
        Colors.GRAY = "\033[90m"
        Colors.BOLD = "\033[1m"
        Colors.DIM = "\033[2m"
        Colors.RESET = original_reset
