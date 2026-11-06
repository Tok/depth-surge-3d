#!/bin/bash

APP_PID=""

# Stop the web server started by this script, and only that.
# (This used to `pkill -f` broad patterns such as "depth_surge_3d\.py" and "python.*app\.py",
# which also killed CLI conversions running in other terminals and any other process whose
# command line happened to match.)
cleanup() {
    trap - SIGINT SIGTERM EXIT
    echo ""
    echo "Stopping Depth Surge 3D Web UI..."

    if [ -n "$APP_PID" ] && kill -0 "$APP_PID" 2>/dev/null; then
        echo "   Stopping web server (PID: $APP_PID)..."
        # SIGINT, not SIGTERM: the server shuts down gracefully on SIGINT (stopping its ffmpeg
        # jobs). SIGTERM is intercepted by the glog crash handler that pycolmap (a DA3
        # dependency) installs, which prints a crash dump and aborts instead.
        kill -INT "$APP_PID" 2>/dev/null
        for _ in $(seq 1 30); do
            kill -0 "$APP_PID" 2>/dev/null || break
            sleep 0.5
        done
        if kill -0 "$APP_PID" 2>/dev/null; then
            echo "   Server didn't stop in time, forcing..."
            pkill -KILL -P "$APP_PID" 2>/dev/null || true
            kill -KILL "$APP_PID" 2>/dev/null || true
        fi
    fi

    echo "Web UI stopped"
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

# Create output directory (uploaded videos are stored in their batch folder under output/)
mkdir -p output

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
