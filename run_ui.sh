#!/bin/bash

echo "Starting Depth Surge 3D Web UI..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Install additional Flask dependencies if needed (quietly)
python -m pip install flask flask-socketio > /dev/null 2>&1

# Create upload and output directories
mkdir -p uploads output

# Start the Flask application
echo "Starting web server at http://localhost:5000"
echo "Press Ctrl+C to stop the server"

# Open browser automatically (cross-platform)
if command -v xdg-open > /dev/null; then
    (sleep 2; xdg-open http://localhost:5000) &
elif command -v open > /dev/null; then
    (sleep 2; open http://localhost:5000) &
elif command -v start > /dev/null; then
    (sleep 2; start http://localhost:5000) &
fi

# Pass through command line arguments (like -v for verbose)
python app.py "$@"