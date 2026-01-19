"""Tests for check_cuda script."""

from unittest.mock import patch


def test_check_cuda_script():
    """Test that check_cuda script runs without errors."""
    # Capture stdout to suppress output during test
    with patch("builtins.print") as mock_print:
        # Import the script which executes the print statements
        import src.depth_surge_3d.utils.system.check_cuda  # noqa: F401

    # Verify that print was called (script executed)
    assert mock_print.call_count == 3  # 3 print statements in the script
