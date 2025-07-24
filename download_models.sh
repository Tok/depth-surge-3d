#!/bin/bash

echo "🔄 Depth Surge 3D - Model Download Script"
echo "=========================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to download file with progress
download_file() {
    local url="$1"
    local output_path="$2"
    local description="$3"
    
    echo "🔄 Downloading $description..."
    
    # Try using curl first, then wget as fallback
    if command_exists curl; then
        curl -L "$url" -o "$output_path" --progress-bar
    elif command_exists wget; then
        wget "$url" -O "$output_path" --progress=bar
    else
        echo "❌ Error: Neither curl nor wget found. Please install one of them."
        echo "   Manual download URL: $url"
        echo "   Save to: $output_path"
        return 1
    fi
    
    if [ -f "$output_path" ]; then
        echo "✅ $description downloaded successfully"
        echo "📊 Size: $(du -h "$output_path" | cut -f1)"
        return 0
    else
        echo "❌ Failed to download $description"
        return 1
    fi
}

# Create models directory
mkdir -p models

# Model information
declare -A models
models["small"]="https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth|models/Depth-Anything-V2-Small/depth_anything_v2_vits.pth|Depth-Anything-V2-Small (24.8M params)"
models["base"]="https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth|models/Depth-Anything-V2-Base/depth_anything_v2_vitb.pth|Depth-Anything-V2-Base (97.5M params)"
models["large"]="https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth|models/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth|Depth-Anything-V2-Large (335.3M params)"

# Check command line arguments
if [ $# -eq 0 ]; then
    echo ""
    echo "📋 Available models:"
    echo "   small  - Depth-Anything-V2-Small (24.8M params)  - Fast, lower quality"
    echo "   base   - Depth-Anything-V2-Base (97.5M params)   - Balanced"
    echo "   large  - Depth-Anything-V2-Large (335.3M params) - Best quality (default)"
    echo "   all    - Download all models"
    echo ""
    echo "💡 Usage:"
    echo "   ./download_models.sh large        # Download large model (recommended)"
    echo "   ./download_models.sh small base   # Download multiple models"
    echo "   ./download_models.sh all          # Download all models"
    echo ""
    
    # Check which models already exist
    echo "📊 Current model status:"
    for model_name in small base large; do
        IFS='|' read -r url path description <<< "${models[$model_name]}"
        if [ -f "$path" ]; then
            size=$(du -h "$path" | cut -f1)
            echo "   ✅ $model_name - $description ($size)"
        else
            echo "   ❌ $model_name - $description (not downloaded)"
        fi
    done
    echo ""
    
    exit 0
fi

# Process command line arguments
requested_models=()
if [ "$1" = "all" ]; then
    requested_models=(small base large)
else
    for arg in "$@"; do
        if [[ " small base large " =~ " $arg " ]]; then
            requested_models+=("$arg")
        else
            echo "❌ Error: Unknown model '$arg'"
            echo "   Valid options: small, base, large, all"
            exit 1
        fi
    done
fi

# Download requested models
echo ""
download_count=0
skip_count=0

for model_name in "${requested_models[@]}"; do
    IFS='|' read -r url path description <<< "${models[$model_name]}"
    
    # Create model directory
    model_dir=$(dirname "$path")
    mkdir -p "$model_dir"
    
    # Check if model already exists
    if [ -f "$path" ]; then
        echo "✅ $description already exists ($(du -h "$path" | cut -f1))"
        ((skip_count++))
        continue
    fi
    
    # Download the model
    if download_file "$url" "$path" "$description"; then
        ((download_count++))
    else
        echo "❌ Failed to download $description"
        exit 1
    fi
    echo ""
done

# Download Depth-Anything-V2 repository if not present
if [ ! -d "depth_anything_v2_repo" ]; then
    echo "🔄 Downloading Depth-Anything-V2 repository..."
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git depth_anything_v2_repo
    if [ $? -eq 0 ]; then
        echo "✅ Depth-Anything-V2 repository downloaded successfully"
    else
        echo "❌ Failed to download Depth-Anything-V2 repository"
        exit 1
    fi
else
    echo "✅ Depth-Anything-V2 repository already exists"
fi

echo ""
echo "🎉 Download complete!"
echo "📊 Summary:"
echo "   Downloaded: $download_count models"
echo "   Skipped: $skip_count models (already existed)"
echo ""
echo "💡 To use a specific model, set the path in your command:"
echo "   python depth_surge_3d.py --model models/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth input.mp4"
echo ""
echo "🔧 Available models:"
for model_name in small base large; do
    IFS='|' read -r url path description <<< "${models[$model_name]}"
    if [ -f "$path" ]; then
        size=$(du -h "$path" | cut -f1)
        echo "   ✅ $path ($size)"
    fi
done 