# Troubleshooting Guide

## Common Issues

### "uv.lock parse error"
The script automatically falls back to virtual environment mode. This is a normal fallback mechanism and doesn't affect functionality.

### "xFormers not available"
This warning is normal and doesn't affect functionality. The system will use PyTorch's built-in attention mechanisms instead.

### Slow processing
- **Verify GPU acceleration**: Run `python -c "import torch; print(torch.cuda.is_available())"` to check if CUDA is available
- **Use CPU explicitly**: If no GPU is available, use `--device cpu`
- **Reduce resolution**: Use `--resolution 720p` for faster testing
- **Reduce VR resolution**: Use `--vr-resolution 16x9-720p` for preview quality

### Out of memory errors
- **Reduce resolution**: Use `--resolution 720p` or lower
- **Use CPU mode**: Add `--device cpu` (slower but uses system RAM instead of VRAM)
- **Don't save intermediates**: Add `--no-intermediates` to reduce disk I/O
- **Process shorter clips**: Use `-s` and `-e` to process smaller time ranges

### CUDA initialization errors
- **Enable persistence mode**: Run `sudo nvidia-smi -pm 1` to enable GPU persistence mode
- **Check CUDA version**: Ensure CUDA 13.0+ is installed
- **Check driver version**: Update NVIDIA drivers to latest version
- **Reboot after driver updates**: Some CUDA changes require a system restart

### Audio extraction failed
This typically occurs when:
- Video has no audio stream (video-only file)
- Audio codec is not supported by FFmpeg
- File is corrupted

**Solution**: The system automatically detects videos without audio and skips audio extraction. For codec issues, re-encode the video with standard codecs (H.264 video, AAC audio).

### Processing restarts from beginning
Make sure you're using the **resume** feature correctly:
```bash
# Resume from existing output directory
python depth_surge_3d.py --resume ./output/timestamp_video_timestamp/
```

The system will skip already-completed processing steps.

### Model download fails
- **Check internet connection**: Model downloads are ~1.3GB for the large model
- **Manual download**: Use the manual installation instructions in [INSTALLATION.md](INSTALLATION.md)
- **Firewall/proxy**: Ensure Hugging Face URLs are not blocked

## Quality Expectations & Limitations

### When It Works Well
- Videos with clear depth variation (landscapes, interiors with furniture, people in scenes)
- Good lighting conditions with visible detail
- Source resolution 1080p or higher for best results
- Content shot with steady camera movement

### Known Limitations

#### AI depth estimation
Generated stereo effect is approximate, not true stereo capture. Results will never match content shot with actual stereo cameras.

#### Challenging scenes
May produce poor results with:
- Mirrors and reflective surfaces
- Glass and transparent objects
- Water reflections
- Very dark scenes with little detail
- Monochrome or low-contrast content

#### Fast motion
Rapid camera movements or quick object motion may cause:
- Temporal inconsistencies
- Warping artifacts
- Ghosting effects

#### Processing time
Can be significant for long/high-resolution videos:
- Typical speed: ~2-4 seconds per output frame on modern GPU (RTX 4070+)
- 1 minute of 60fps video = 3600 frames = ~2-4 hours processing time

### Resolution Recommendations

- **Low resolutions** (480p, 720p): Primarily for quick testing and preview - expect reduced quality
- **1080p+**: Recommended minimum for acceptable VR viewing quality
- **4K+**: Best results for high-end VR headsets
- **8K**: Maximum quality but very slow processing

**Note**: Results are limited by source quality. Upscaling low-quality source material will not improve the output significantly.

## Performance Tips

### GPU Acceleration
- **Check CUDA availability**: `nvidia-smi` should show your GPU
- **Enable persistence mode**: `sudo nvidia-smi -pm 1`
- **Monitor GPU usage**: Use `nvidia-smi` during processing to verify GPU utilization
- **Close other GPU applications**: Free up VRAM by closing games, browsers with hardware acceleration, etc.

### Disk I/O
- **Use SSD**: Much faster for reading/writing intermediate frames
- **Skip intermediates**: Add `--no-intermediates` to avoid saving depth maps and stereo pairs
- **Clean up old outputs**: Remove old processing directories to free disk space

### Memory Management
- **Chunked processing**: The system automatically processes frames in small batches to avoid memory issues
- **Close other applications**: Free up system RAM for better performance
- **Monitor memory usage**: Use `htop` or Task Manager to check RAM/VRAM usage

## Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Look for error messages in the terminal output
2. **Verify installation**: Run `./test.sh` to check system setup (CUDA, dependencies, models)
3. **Try a minimal example**: Test with a short, simple video clip
4. **Report issues**: Open an issue on GitHub with:
   - Full error message
   - System specs (GPU, RAM, OS)
   - Command used
   - Video properties (resolution, codec, duration)
