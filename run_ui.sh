#!/bin/bash

# Store the PID of this script
SCRIPT_PID=$$
APP_PID=""

# Function to cleanup processes
cleanup() {
    echo ""
    echo "Stopping Depth Surge 3D Web UI..."

    # Kill the Flask app if it's running
    if [ ! -z "$APP_PID" ] && kill -0 "$APP_PID" 2>/dev/null; then
        echo "   Stopping web server (PID: $APP_PID)..."
        kill -TERM "$APP_PID" 2>/dev/null || kill -KILL "$APP_PID" 2>/dev/null
        wait "$APP_PID" 2>/dev/null
    fi

    # Kill any other Python processes related to this app
    echo "   Cleaning up any remaining processes..."
    pkill -f "python.*app\.py" 2>/dev/null || true
    pkill -f "flask.*run" 2>/dev/null || true

    # Kill any ongoing processing jobs
    echo "   Stopping any active video processing..."
    pkill -f "depth_surge_3d\.py" 2>/dev/null || true
    pkill -f "ffmpeg.*depth-surge" 2>/dev/null || true

    echo "All processes stopped successfully"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM EXIT

echo "Starting Depth Surge 3D Web UI..."

# Check if virtual environment exists and activate it
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

# Install additional Flask dependencies if needed (quietly)
python -m pip install flask flask-socketio > /dev/null 2>&1

# Create upload and output directories
mkdir -p uploads output

# Start the Flask application
echo "Starting web server at http://localhost:5000"
echo "Navigate to http://localhost:5000"
echo "Press Ctrl+C to stop the server and all background processes"

# Open browser automatically (cross-platform)
if command -v xdg-open > /dev/null; then
    echo "Opening in existing browser session."
    (sleep 2; xdg-open http://localhost:5000) &
elif command -v open > /dev/null; then
    echo "Opening in default browser."
    (sleep 2; open http://localhost:5000) &
elif command -v start > /dev/null; then
    echo "Opening in default browser."
    (sleep 2; start http://localhost:5000) &
fi

# Start the Flask app in background and store its PID
export DEPTH_SURGE_UI_SCRIPT=1
python app.py "$@" &
APP_PID=$!

# Wait for the Flask app to finish
wait "$APP_PID"
