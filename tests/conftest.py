"""Shared pytest fixtures."""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _no_xformers_guard(request):
    """
    Don't let depth estimator tests import the real xformers.

    Several tests patch sys.modules, which drops xformers again on exit; importing it a
    second time re-registers its torch ops and aborts the interpreter.
    Tests of the guard itself opt out with @pytest.mark.real_xformers_check.
    """
    if request.node.get_closest_marker("real_xformers_check"):
        yield
        return
    with (
        patch("src.depth_surge_3d.inference.depth.video_depth_estimator.ensure_attention_backend"),
        patch(
            "src.depth_surge_3d.inference.depth.video_depth_estimator_da3.ensure_attention_backend"
        ),
    ):
        yield
