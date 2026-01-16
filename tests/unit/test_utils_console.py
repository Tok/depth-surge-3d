"""Unit tests for console utilities."""

from src.depth_surge_3d.utils.console import (
    success,
    error,
    warning,
    info,
    bold,
    dim,
    title_bar,
    step_complete,
    saved_to,
)


class TestConsoleUtils:
    """Test console utility functions."""

    def test_success_formatting(self):
        """Test success message formatting."""
        result = success("Test message")
        assert "Test message" in result
        assert isinstance(result, str)

    def test_error_formatting(self):
        """Test error message formatting."""
        result = error("Error occurred")
        assert "Error occurred" in result
        assert isinstance(result, str)

    def test_warning_formatting(self):
        """Test warning message formatting."""
        result = warning("Warning text")
        assert "Warning text" in result
        assert isinstance(result, str)

    def test_info_formatting(self):
        """Test info message formatting."""
        result = info("Info text")
        assert "Info text" in result
        assert isinstance(result, str)

    def test_bold_formatting(self):
        """Test bold message formatting."""
        result = bold("Bold text")
        assert "Bold text" in result
        assert isinstance(result, str)

    def test_dim_formatting(self):
        """Test dim message formatting."""
        result = dim("Dim text")
        assert "Dim text" in result
        assert isinstance(result, str)

    def test_title_bar_formatting(self):
        """Test title bar message formatting."""
        result = title_bar("=== Title ===")
        assert "Title" in result
        assert isinstance(result, str)

    def test_step_complete_formatting(self):
        """Test step complete message formatting."""
        result = step_complete("Step done")
        assert "Step done" in result
        assert isinstance(result, str)

    def test_saved_to_formatting(self):
        """Test saved to message formatting."""
        result = saved_to("/path/to/file")
        assert "/path/to/file" in result
        assert isinstance(result, str)

    def test_empty_string_handling(self):
        """Test handling of empty strings."""
        result = success("")
        assert isinstance(result, str)

        result = error("")
        assert isinstance(result, str)

    def test_special_characters(self):
        """Test handling of special characters."""
        special_text = "Test with special chars: @#$%^&*()"
        result = success(special_text)
        assert special_text in result

    def test_multiline_text(self):
        """Test handling of multiline text."""
        multiline = "Line 1\nLine 2\nLine 3"
        result = success(multiline)
        assert "Line 1" in result
        assert "Line 2" in result
