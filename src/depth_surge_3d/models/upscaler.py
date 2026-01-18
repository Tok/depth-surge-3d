"""
AI upscaling models for output enhancement.

This module provides abstraction for various upscaling models including Real-ESRGAN,
with support for style-preserving enhancement without altering video aesthetics.

Real-ESRGAN implementation based on:
https://github.com/ai-forever/Real-ESRGAN (standalone, no basicsr dependency)
"""

from __future__ import annotations
from typing import Literal
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import cv2
from pathlib import Path

UpscaleModel = Literal["none", "x2", "x4", "x4-conservative"]

# Model SHA-256 checksums for security verification
MODEL_CHECKSUMS = {
    "RealESRGAN_x2plus.pth": "49fafd45f8fd7aa8d31ab2a22d14d91b536c577149c7e35f0f2a4c679683f0c3",
    "RealESRGAN_x4plus.pth": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
    "RealESRNet_x4plus.pth": "d0cc4d7c277a675c6c52a25d7bb823f4c53c799481e05c92b35b8a9f0bb00f47",
}


def verify_file_checksum(file_path: Path, expected_hash: str) -> bool:
    """
    Verify file SHA-256 checksum for security.

    Args:
        file_path: Path to file to verify
        expected_hash: Expected SHA-256 hash (hex string)

    Returns:
        True if checksum matches, False otherwise
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_hash


# ============================================================================
# RRDB Network Architecture (from ai-forever/Real-ESRGAN)
# ============================================================================


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights."""
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks."""
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


def pixel_unshuffle(x, scale):
    """Pixel unshuffle for downsampling."""
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block used in RRDB."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Real-ESRGAN network with Residual in Residual Dense Blocks."""

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if scale == 8:
            self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        if self.scale == 8:
            feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# ============================================================================
# Upscaler Classes
# ============================================================================


class ImageUpscaler:
    """Base class for AI upscaling models."""

    def __init__(self, device: str = "auto"):
        """
        Initialize upscaler.

        Args:
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.device = self._determine_device(device)
        self.model = None

    def _determine_device(self, device: str) -> str:
        """Determine best device for inference."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load_model(self) -> bool:
        """Load the upscaling model."""
        raise NotImplementedError

    def upscale_image(self, image: np.ndarray) -> np.ndarray:
        """Upscale a single image (BGR format)."""
        raise NotImplementedError

    def unload_model(self) -> None:
        """Free model memory."""
        if self.model is not None:
            del self.model
            self.model = None
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()


class RealESRGANUpscaler(ImageUpscaler):
    """Real-ESRGAN upscaler (standalone implementation, no external wrappers)."""

    def __init__(self, model_name: str = "x4", device: str = "auto"):
        """
        Initialize Real-ESRGAN upscaler.

        Args:
            model_name: Model variant ('x2', 'x4', 'x4-conservative')
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        super().__init__(device)
        self.model_name = model_name
        self.scale = int(model_name[1]) if model_name.startswith("x") else 4

    def _download_model_weights(self, model_url: str, model_path: Path) -> None:
        """Download model weights with timeout."""
        import urllib.request
        import socket

        print(f"Downloading Real-ESRGAN weights ({model_path.name})...")
        original_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(30)
        try:
            urllib.request.urlretrieve(model_url, model_path)
        finally:
            socket.setdefaulttimeout(original_timeout)
        print("Download complete!")

    def _verify_model_checksum(self, model_path: Path) -> None:
        """Verify model checksum for security."""
        expected_hash = MODEL_CHECKSUMS.get(model_path.name)
        if expected_hash:
            print(f"Verifying checksum for {model_path.name}...")
            if not verify_file_checksum(model_path, expected_hash):
                model_path.unlink()  # Delete corrupted file
                raise ValueError(
                    f"Checksum verification failed for {model_path.name}. "
                    "File may be corrupted or tampered with."
                )
            print("Checksum verified!")
        else:
            print(f"Warning: No checksum available for {model_path.name}")

    def load_model(self) -> bool:
        """Load Real-ESRGAN model weights."""
        try:
            # Model configurations
            if self.model_name == "x2":
                model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
                netscale = 2
            elif self.model_name == "x4-conservative":
                model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
                netscale = 4
            else:  # x4 default
                model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
                netscale = 4

            # Create models cache directory
            cache_dir = Path.home() / ".cache" / "depth-surge-3d" / "realesrgan"
            cache_dir.mkdir(parents=True, exist_ok=True)

            model_filename = model_url.split("/")[-1]
            model_path = cache_dir / model_filename

            # Download if not cached
            if not model_path.exists():
                self._download_model_weights(model_url, model_path)

            # Verify checksum for security
            self._verify_model_checksum(model_path)

            # Create network
            self.model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=netscale
            )

            # Load weights with security flag (weights_only=True for torch >= 1.13)
            try:
                loadnet = torch.load(model_path, map_location=self.device, weights_only=True)
            except TypeError:
                # Fallback for older PyTorch versions
                loadnet = torch.load(model_path, map_location=self.device)
            if "params_ema" in loadnet:
                keyname = "params_ema"
            else:
                keyname = "params"
            self.model.load_state_dict(loadnet[keyname], strict=True)

            self.model.eval()
            self.model = self.model.to(self.device)

            print(f"Loaded Real-ESRGAN ({self.model_name}) on {self.device}")
            return True

        except Exception as e:
            print(f"Error loading Real-ESRGAN: {e}")
            return False

    def upscale_image(self, image: np.ndarray) -> np.ndarray:
        """
        Upscale image using Real-ESRGAN (BGR → BGR).

        Args:
            image: Input image in BGR format

        Returns:
            Upscaled image in BGR format
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # BGR to RGB, normalize to [0, 1]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float() / 255.0

        # Convert to CHW and add batch dimension
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(img_tensor)

        # Post-process: clamp, convert to numpy
        output = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        output = (output * 255.0).astype(np.uint8)

        # RGB to BGR
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        return output_bgr


def create_upscaler(model_name: str = "none", device: str = "auto") -> ImageUpscaler | None:
    """
    Factory function to create an upscaler.

    Args:
        model_name: Upscale model ('none', 'x2', 'x4', 'x4-conservative')
        device: Device to use ('auto', 'cuda', 'cpu')

    Returns:
        ImageUpscaler instance or None if model_name is 'none'

    Raises:
        ValueError: If model_name is unknown
    """
    if model_name == "none":
        return None

    # Real-ESRGAN models
    if model_name in ["x2", "x4", "x4-conservative"]:
        return RealESRGANUpscaler(model_name, device)

    raise ValueError(f"Unknown upscale model: {model_name}")
