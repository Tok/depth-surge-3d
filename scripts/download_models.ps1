# Download Depth Surge 3D depth models (Video-Depth-Anything and Depth Anything V3).
# Thin wrapper around scripts/download_models.py. Run without arguments to see the status.
#
#   .\scripts\download_models.ps1                  # status and usage
#   .\scripts\download_models.ps1 recommended      # models used by the default settings
#   .\scripts\download_models.ps1 vda-base da3-large
#   .\scripts\download_models.ps1 all

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    & $venvPython scripts\download_models.py @args
} elseif (Get-Command uv -ErrorAction SilentlyContinue) {
    uv run --no-sync python scripts\download_models.py @args
} else {
    python scripts\download_models.py @args
}
exit $LASTEXITCODE
