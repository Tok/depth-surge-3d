"""
Depth estimation model management and loading.

This module handles loading and interfacing with the Depth-Anything-V2 model
with minimal side effects and clear separation of concerns.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import numpy as np

from ..core.constants import (
    DEFAULT_MODEL_PATH, DEPTH_ANYTHING_REPO_DIR, MODEL_CONFIGS,
    MODEL_DOWNLOAD_URLS, MODEL_PATHS, ERROR_MESSAGES
)


class DepthEstimator:
    """Handles depth estimation using Depth-Anything-V2 models."""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        self.model_path = model_path
        self.device = self._determine_device(device)
        self.model = None
        self.model_config = None
        
    def _determine_device(self, device: str) -> str:
        """Determine the best device to use for inference."""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device
    
    def load_model(self) -> bool:
        """
        Load the depth estimation model.
        
        Returns:
            True if model loaded successfully
        """
        try:
            # Ensure dependencies are available
            if not self._ensure_dependencies():
                return False
            
            # Determine model type from path
            model_type = self._get_model_type(self.model_path)
            if not model_type:
                print(f"Cannot determine model type from path: {self.model_path}")
                return False
            
            self.model_config = MODEL_CONFIGS[model_type]
            
            # Import and load model
            repo_path = Path(DEPTH_ANYTHING_REPO_DIR)
            sys.path.insert(0, str(repo_path))
            
            from depth_anything_v2.dpt import DepthAnythingV2
            
            self.model = DepthAnythingV2(**self.model_config)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Loaded Depth-Anything-V2 ({model_type}) on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Try running: ./download_models.sh large")
            return False
    
    def _ensure_dependencies(self) -> bool:
        """Ensure model file and repository are available."""
        # Check model file
        if not os.path.exists(self.model_path):
            print(f"Model not found at {self.model_path}")
            if not self._auto_download_model():
                return False
        
        # Check repository
        repo_path = Path(DEPTH_ANYTHING_REPO_DIR)
        if not repo_path.exists():
            print("Depth-Anything-V2 repository not found")
            if not self._auto_download_repo():
                return False
        
        return True
    
    def _auto_download_model(self) -> bool:
        """Auto-download the model if missing."""
        print("Attempting to download model automatically...")
        
        # Create model directory
        model_dir = Path(self.model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine download URL
        model_type = self._get_model_type(self.model_path)
        if not model_type or model_type not in MODEL_DOWNLOAD_URLS:
            print("Cannot determine model download URL")
            return False
        
        download_url = MODEL_DOWNLOAD_URLS[model_type]
        
        try:
            print(f"Downloading model to {self.model_path}...")
            urllib.request.urlretrieve(download_url, self.model_path)
            print("✅ Model downloaded successfully")
            return True
        except Exception as e:
            print(f"❌ Auto-download failed: {e}")
            print(f"Please run: ./download_models.sh {model_type}")
            print(f"Or download manually from: {download_url}")
            return False
    
    def _auto_download_repo(self) -> bool:
        """Auto-download the Depth-Anything-V2 repository if missing."""
        print("Attempting to download Depth-Anything-V2 repository...")
        
        try:
            result = subprocess.run([
                "git", "clone", "https://github.com/DepthAnything/Depth-Anything-V2.git", 
                DEPTH_ANYTHING_REPO_DIR
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Repository downloaded successfully")
                return True
            else:
                raise Exception(f"Git clone failed: {result.stderr}")
        except Exception as e:
            print(f"❌ Auto-download failed: {e}")
            print(f"Please run: git clone https://github.com/DepthAnything/Depth-Anything-V2.git {DEPTH_ANYTHING_REPO_DIR}")
            return False
    
    def _get_model_type(self, model_path: str) -> Optional[str]:
        """Determine model type from file path."""
        path_str = str(model_path).lower()
        
        if 'vits' in path_str:
            return 'vits'
        elif 'vitb' in path_str:
            return 'vitb'
        elif 'vitl' in path_str:
            return 'vitl'
        elif 'vitg' in path_str:
            return 'vitg'
        
        # Fallback to large model
        return 'vitl'
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth for a single image.
        
        Args:
            image: Input image array (BGR format)
            
        Returns:
            Normalized depth map (0-1 range)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            with torch.no_grad():
                depth = self.model.infer_image(image)
            
            # Normalize depth map
            if depth.max() == depth.min():
                return np.zeros_like(depth)
            
            normalized = (depth - depth.min()) / (depth.max() - depth.min())
            return np.clip(normalized, 0.0, 1.0)
            
        except Exception as e:
            raise RuntimeError(f"Depth estimation failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model_config:
            return {}
        
        return {
            "encoder": self.model_config["encoder"],
            "features": self.model_config["features"],
            "out_channels": self.model_config["out_channels"],
            "device": self.device,
            "model_path": self.model_path,
            "loaded": self.model is not None
        }
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            
            # Clear GPU cache if using CUDA
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()


def create_depth_estimator(
    model_path: Optional[str] = None,
    device: str = 'auto'
) -> DepthEstimator:
    """
    Factory function to create a depth estimator.
    
    Args:
        model_path: Path to model file (uses default if None)
        device: Device to use for inference
        
    Returns:
        Configured DepthEstimator instance
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    
    return DepthEstimator(model_path, device)


def validate_model_file(model_path: str) -> bool:
    """
    Validate that model file exists and appears to be valid.
    
    Args:
        model_path: Path to model file
        
    Returns:
        True if model file appears valid
    """
    if not os.path.exists(model_path):
        return False
    
    # Basic size check (Depth-Anything models are typically > 50MB)
    try:
        file_size = os.path.getsize(model_path)
        if file_size < 50 * 1024 * 1024:  # 50MB
            return False
    except OSError:
        return False
    
    # Check file extension
    if not model_path.lower().endswith('.pth'):
        return False
    
    return True


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """
    Get information about available models.
    
    Returns:
        Dictionary with model information
    """
    models = {}
    
    for model_name, model_path in MODEL_PATHS.items():
        models[model_name] = {
            "path": model_path,
            "exists": os.path.exists(model_path),
            "valid": validate_model_file(model_path) if os.path.exists(model_path) else False,
            "config": MODEL_CONFIGS.get(model_name.replace('small', 'vits').replace('base', 'vitb').replace('large', 'vitl'), {}),
            "download_url": MODEL_DOWNLOAD_URLS.get(model_name, "")
        }
        
        if models[model_name]["exists"]:
            try:
                size = os.path.getsize(model_path)
                models[model_name]["size_bytes"] = size
                models[model_name]["size_mb"] = round(size / (1024 * 1024), 1)
            except OSError:
                models[model_name]["size_bytes"] = 0
                models[model_name]["size_mb"] = 0
    
    return models


def estimate_model_memory_usage(model_type: str, device: str = 'cuda') -> Dict[str, float]:
    """
    Estimate memory usage for a model type.
    
    Args:
        model_type: Model type (vits, vitb, vitl, vitg)
        device: Target device
        
    Returns:
        Dictionary with memory estimates in MB
    """
    # Rough estimates based on model parameters and typical usage
    base_memory = {
        'vits': 500,   # ~24M parameters
        'vitb': 1000,  # ~97M parameters  
        'vitl': 2000,  # ~335M parameters
        'vitg': 4000   # ~1.3B parameters
    }
    
    model_memory = base_memory.get(model_type, 2000)
    
    estimates = {
        "model_memory": model_memory,
        "inference_memory": model_memory * 0.5,  # Additional memory during inference
        "total_estimated": model_memory * 1.5
    }
    
    # CPU typically uses less memory due to different allocation patterns
    if device == 'cpu':
        for key in estimates:
            estimates[key] *= 0.7
    
    return estimates 