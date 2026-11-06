"""Tests for scripts/download_models.py (no network)."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.depth_surge_3d.core.constants import (
    DA3_MODEL_NAMES,
    MODEL_DOWNLOAD_URLS_METRIC,
    MODEL_PATHS_METRIC,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def dm():
    spec = importlib.util.spec_from_file_location(
        "download_models", ROOT / "scripts/download_models.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["download_models"] = module  # dataclasses resolve their module here
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("download_models", None)


def test_catalog_covers_every_supported_model(dm):
    catalog = dm._catalog()
    assert {f"da3-{name}" for name in DA3_MODEL_NAMES} <= set(catalog)
    for size in ("small", "base", "large"):
        assert f"vda-{size}" in catalog and f"vda-{size}-metric" in catalog


def test_metric_entries_use_metric_repos(dm):
    model = dm._catalog()["vda-base-metric"]
    assert model.url == MODEL_DOWNLOAD_URLS_METRIC["vitb"]
    assert model.path == MODEL_PATHS_METRIC["vitb"]
    assert "Metric-Video-Depth-Anything-Base" in model.url


def test_groups_reference_existing_models(dm):
    catalog = dm._catalog()
    for keys in dm.GROUPS.values():
        assert set(keys) <= set(catalog)


def test_resolve(dm):
    catalog = dm._catalog()
    assert [m.key for m in dm.resolve(["large", "vda-large"], catalog)] == ["vda-large"]
    assert len(dm.resolve(["all"], catalog)) == len(catalog)
    assert dm.resolve(["nope"], catalog) is None


def test_unknown_name_downloads_nothing(dm, capsys):
    with patch.object(dm, "download") as mock_download:
        assert dm.main(["vda-small", "nope"]) == 2
    mock_download.assert_not_called()


def test_vda_download_is_atomic(dm, tmp_path):
    model = dm.Model("x", "test", url="https://example.invalid/m.pth", path="m/model.pth")

    def failing(url, path, hook=None):
        Path(path).write_bytes(b"partial")
        raise OSError("connection reset")

    with (
        patch.object(dm, "ROOT", tmp_path),
        patch.object(dm.urllib.request, "urlretrieve", failing),
    ):
        assert dm.download(model) is False
    assert not (tmp_path / "m/model.pth").exists()


def test_status_lists_licences(dm, capsys):
    with patch.object(dm, "is_downloaded", return_value=False):
        dm.show_status(dm._catalog())
    out = capsys.readouterr().out
    assert "vda-large" in out and "non-commercial" in out and "recommended" in out
