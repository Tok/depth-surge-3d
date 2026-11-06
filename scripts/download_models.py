#!/usr/bin/env python3
"""
Download the depth models Depth Surge 3D can use.

Model locations come from src/depth_surge_3d/core/constants.py, so this script always
matches what the application loads:

- Video-Depth-Anything (V2 backend): .pth files under models/
- Depth Anything V3 (V3 backend): Hugging Face repos, stored in the Hugging Face cache
  (the same place the app downloads them to on first use)

Weights are never bundled with Depth Surge 3D; check each model's licence before use.

Usage:
    python scripts/download_models.py                 # show status and usage
    python scripts/download_models.py recommended     # models used by the default settings
    python scripts/download_models.py vda-large da3-large-metric
    python scripts/download_models.py all
"""

from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.depth_surge_3d.core.constants import (  # noqa: E402
    DA3_MODEL_NAMES,
    MODEL_DOWNLOAD_URLS,
    MODEL_DOWNLOAD_URLS_METRIC,
    MODEL_PATHS,
    MODEL_PATHS_METRIC,
)

# Hugging Face licences of the upstream repos (non-commercial ones are flagged)
_NON_COMMERCIAL = {"vitb", "vitl", "giant", "giant-large"}
_SIZE_NAMES = {"vits": "small", "vitb": "base", "vitl": "large"}


@dataclass(frozen=True)
class Model:
    key: str
    description: str
    url: str | None = None  # direct file download (Video-Depth-Anything)
    path: str | None = None  # local file for `url`
    repo_id: str | None = None  # Hugging Face repo (Depth Anything V3)
    non_commercial: bool = False


def _catalog() -> dict[str, Model]:
    models: dict[str, Model] = {}
    for encoder, size in _SIZE_NAMES.items():
        nc = encoder in _NON_COMMERCIAL
        models[f"vda-{size}"] = Model(
            f"vda-{size}",
            f"Video-Depth-Anything {size.capitalize()} (V2, relative depth)",
            url=MODEL_DOWNLOAD_URLS[encoder],
            path=MODEL_PATHS[encoder],
            non_commercial=nc,
        )
        models[f"vda-{size}-metric"] = Model(
            f"vda-{size}-metric",
            f"Video-Depth-Anything {size.capitalize()} (V2, metric depth)",
            url=MODEL_DOWNLOAD_URLS_METRIC[encoder],
            path=MODEL_PATHS_METRIC[encoder],
            non_commercial=nc,
        )
    for name, repo_id in DA3_MODEL_NAMES.items():
        models[f"da3-{name}"] = Model(
            f"da3-{name}",
            f"Depth Anything V3 {name} ({repo_id})",
            repo_id=repo_id,
            non_commercial=name in _NON_COMMERCIAL,
        )
    return models


GROUPS = {
    # CLI default (V2, Large, relative) + web UI default (V3 with metric depth)
    "recommended": ["vda-large", "da3-large-metric"],
    "vda": ["vda-small", "vda-base", "vda-large"],
    "vda-metric": ["vda-small-metric", "vda-base-metric", "vda-large-metric"],
    "da3": [f"da3-{name}" for name in DA3_MODEL_NAMES],
    # Backwards-compatible names from the old script
    "small": ["vda-small"],
    "base": ["vda-base"],
    "large": ["vda-large"],
}


def _hf_cached(repo_id: str) -> bool:
    try:
        from huggingface_hub import scan_cache_dir

        return any(repo.repo_id == repo_id for repo in scan_cache_dir().repos)
    except Exception:
        return False


def is_downloaded(model: Model) -> bool:
    if model.path:
        return (ROOT / model.path).is_file()
    return bool(model.repo_id) and _hf_cached(model.repo_id)


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size > 0:
        done = min(block_num * block_size, total_size)
        print(
            f"\r    {done / 2**20:8.0f} / {total_size / 2**20:.0f} MB ({100 * done / total_size:5.1f}%)",
            end="",
            flush=True,
        )


def download(model: Model) -> bool:
    if is_downloaded(model):
        print(f"  [skip] {model.key}: already downloaded")
        return True

    print(f"  [get ] {model.key}: {model.description}")
    try:
        if model.url and model.path:
            target = ROOT / model.path
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            urllib.request.urlretrieve(model.url, partial, _progress)
            partial.replace(target)
            print(f"\n    saved to {model.path}")
        elif model.repo_id:
            from huggingface_hub import snapshot_download

            location = snapshot_download(model.repo_id)
            print(f"    cached at {location}")
        return True
    except Exception as e:
        print(f"\n  [fail] {model.key}: {e}")
        if model.url:
            print(f"    Download manually from {model.url} and save it as {model.path}")
        return False


def show_status(catalog: dict[str, Model]) -> None:
    print("Depth Surge 3D models\n")
    for model in catalog.values():
        mark = "x" if is_downloaded(model) else " "
        licence = "  (non-commercial licence)" if model.non_commercial else ""
        print(f"  [{mark}] {model.key:<20} {model.description}{licence}")
    print(
        "\nGroups: " + ", ".join(f"{name}" for name in GROUPS if name not in _SIZE_NAMES.values())
    )
    print(
        "  recommended = " + " + ".join(GROUPS["recommended"]) + " (used by the default settings)"
    )
    print("\nUsage:")
    print("  ./scripts/download_models.sh recommended")
    print("  ./scripts/download_models.sh vda-base da3-large")
    print("  ./scripts/download_models.sh all")
    print("\nDA3 models also download automatically on first use.")


def resolve(args: list[str], catalog: dict[str, Model]) -> list[Model] | None:
    keys: list[str] = []
    for arg in args:
        if arg == "all":
            keys.extend(catalog)
        elif arg in GROUPS:
            keys.extend(GROUPS[arg])
        elif arg in catalog:
            keys.append(arg)
        else:
            print(f"Unknown model or group: {arg}\n")
            return None
    return [catalog[key] for key in dict.fromkeys(keys)]


def main(argv: list[str]) -> int:
    catalog = _catalog()
    if not argv or argv[0] in ("-h", "--help", "list", "status"):
        show_status(catalog)
        return 0

    selected = resolve(argv, catalog)
    if selected is None:
        show_status(catalog)
        return 2

    print(f"Downloading {len(selected)} model(s)...")
    failed = [model.key for model in selected if not download(model)]
    if any(model.non_commercial for model in selected):
        print(
            "\nNote: some of these models are licensed for non-commercial use only (CC BY-NC 4.0)."
        )
    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
