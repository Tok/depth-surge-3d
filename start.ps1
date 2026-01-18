# Depth Surge 3D - Simple Start Script for Windows
# Usage: .\start.ps1 START_TIME END_TIME
# Example: .\start.ps1 1:18 1:33

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$StartTime,

    [Parameter(Mandatory=$true, Position=1)]
    [string]$EndTime
)

$inputVideo = "input_video.mp4"

# Check if input video exists
if (-not (Test-Path $inputVideo)) {
    Write-Host "Error: $inputVideo not found!" -ForegroundColor Red
    Write-Host "Please place your video file as 'input_video.mp4' in this directory."
    exit 1
}

Write-Host "Processing $inputVideo from $StartTime to $EndTime..." -ForegroundColor Cyan

# Check if virtual environment exists and activate it
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Using .venv virtual environment..."
    & .venv\Scripts\Activate.ps1
} elseif (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Using venv virtual environment..."
    & venv\Scripts\Activate.ps1
} else {
    Write-Host "Error: No virtual environment found. Please run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}

# Run Depth Surge 3D
python depth_surge_3d.py $inputVideo -s $StartTime -e $EndTime

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Processing complete!" -ForegroundColor Green
    Write-Host "Output video saved with audio preserved."
} else {
    Write-Host ""
    Write-Host "Processing failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
