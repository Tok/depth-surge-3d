#!/bin/bash
# Download Depth Surge 3D depth models (Video-Depth-Anything and Depth Anything V3).
# Thin wrapper around scripts/download_models.py, which reads the model list from the
# application's constants. Run without arguments to see what's available and installed.
#
#   ./scripts/download_models.sh                  # status and usage
#   ./scripts/download_models.sh recommended      # models used by the default settings
#   ./scripts/download_models.sh vda-base da3-large
#   ./scripts/download_models.sh all

set -e
cd "$(dirname "$0")/.."

if [ -x ".venv/bin/python" ]; then
    exec .venv/bin/python scripts/download_models.py "$@"
elif command -v uv >/dev/null 2>&1; then
    exec uv run --no-sync python scripts/download_models.py "$@"
else
    exec python3 scripts/download_models.py "$@"
fi
