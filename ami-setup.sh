source /opt/pytorch/bin/activate
pip install --upgrade pip
pip install -r ami-requirements.txt


# Download Video-Depth-Anything repository if not present
if [ ! -d "video_depth_anything_repo" ]; then
    echo "Downloading Video-Depth-Anything repository..."
    git clone https://github.com/DepthAnything/Video-Depth-Anything.git video_depth_anything_repo
    if [ $? -eq 0 ]; then
        echo "[OK] Video-Depth-Anything repository downloaded successfully"
    else
        echo "[ERROR] Failed to download Video-Depth-Anything repository"
        exit 1
    fi
else
    echo "[OK] Video-Depth-Anything repository already exists"
fi

./download_models.sh large