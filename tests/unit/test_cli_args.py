"""Tests for CLI argument parsing (depth_surge_3d.py)."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("depth_surge_3d_cli", ROOT / "depth_surge_3d.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "src"))
    spec.loader.exec_module(module)
    return module


class TestVrResolutionArg:
    """Regression #17: --vr-resolution custom:WxH was rejected by argparse."""

    @pytest.mark.parametrize("value", ["auto", "custom:1376x768", "custom:2560x1080"])
    def test_accepts(self, cli, value):
        args = cli.create_argument_parser().parse_args(["video.mp4", "--vr-resolution", value])
        assert args.vr_resolution == value

    def test_accepts_every_preset(self, cli):
        parser = cli.create_argument_parser()
        for preset in cli.VR_RESOLUTIONS:
            assert parser.parse_args(["v.mp4", "--vr-resolution", preset]).vr_resolution == preset

    @pytest.mark.parametrize(
        "value", ["custom", "custom:abc", "custom:0x100", "custom:1920", "16x9-9000p"]
    )
    def test_rejects(self, cli, value, capsys):
        with pytest.raises(SystemExit):
            cli.create_argument_parser().parse_args(["video.mp4", "--vr-resolution", value])
        assert "--vr-resolution" in capsys.readouterr().err

    def test_default(self, cli):
        assert cli.create_argument_parser().parse_args(["video.mp4"]).vr_resolution == "auto"
