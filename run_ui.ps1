# Depth Surge 3D - Web UI Launcher for Windows
# Starts the Flask web interface at http://localhost:5000

Write-Host "Starting Depth Surge 3D Web UI..." -ForegroundColor Cyan

# Check if virtual environment exists
$venvPath = $null
if (Test-Path ".venv\Scripts\Activate.ps1") {
    $venvPath = ".venv\Scripts\Activate.ps1"
} elseif (Test-Path "venv\Scripts\Activate.ps1") {
    $venvPath = "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Error: Virtual environment not found. Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
& $venvPath

# Install additional Flask dependencies if needed (quietly)
Write-Host "Checking dependencies..."
python -m pip install flask flask-socketio *>$null

# Install optional AI upscaling dependency if not present (quietly)
python -m pip install py-real-esrgan *>$null

# Create upload and output directories
New-Item -ItemType Directory -Force -Path "uploads" | Out-Null
New-Item -ItemType Directory -Force -Path "output" | Out-Null

# Start the Flask application
Write-Host ""
Write-Host "Starting web server at http://localhost:5000" -ForegroundColor Green
Write-Host "Navigate to http://localhost:5000"
Write-Host "Press Ctrl+C to stop the server and all background processes"
Write-Host ""

# Open browser automatically
Start-Sleep -Seconds 2
Start-Process "http://localhost:5000"

# Start the Flask app
$env:DEPTH_SURGE_UI_SCRIPT = "1"
python app.py $args

# Cleanup function for Ctrl+C
trap {
    Write-Host "`nStopping Depth Surge 3D Web UI..." -ForegroundColor Yellow
    Write-Host "All processes stopped successfully"
    exit 0
}
