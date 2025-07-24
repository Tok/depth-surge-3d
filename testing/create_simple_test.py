#!/usr/bin/env python3
"""
Create simple test frames using numpy and ffmpeg
"""

import numpy as np
import os
from pathlib import Path

def create_simple_test_frame(width=1920, height=1080, frame_num=0):
    """Create a simple test frame with known depth structure using numpy"""
    # Create RGB frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background (far) - blue gradient
    for y in range(height):
        frame[y, :] = [50, 50, 100 + int(y * 155 / height)]  # Blue gradient
    
    # Create simple geometric shapes at different depths
    
    # Rectangle 1 (medium-far depth = 0.7)
    y1, y2 = 200, 400
    x1, x2 = 200, 500
    frame[y1:y2, x1:x2] = [0, 150, 0]  # Green
    
    # Rectangle 2 (medium depth = 0.5)  
    y1, y2 = 300, 500
    x1, x2 = 700, 1000
    frame[y1:y2, x1:x2] = [0, 200, 100]  # Lighter green
    
    # Rectangle 3 (medium-close depth = 0.3)
    y1, y2 = 400, 600  
    x1, x2 = 1200, 1500
    frame[y1:y2, x1:x2] = [100, 255, 200]  # Even lighter green
    
    # Circle 1 (very close depth = 0.1)
    center_y, center_x = 600, 400
    radius = 80
    y_indices, x_indices = np.ogrid[:height, :width]
    mask = (x_indices - center_x)**2 + (y_indices - center_y)**2 <= radius**2
    frame[mask] = [255, 0, 0]  # Red circle
    
    # Circle 2 (very close depth = 0.1)
    center_y, center_x = 600, 1000
    mask = (x_indices - center_x)**2 + (y_indices - center_y)**2 <= radius**2
    frame[mask] = [255, 50, 50]  # Slightly different red
    
    # Circle 3 (very close depth = 0.1)
    center_y, center_x = 600, 1600
    mask = (x_indices - center_x)**2 + (y_indices - center_y)**2 <= radius**2
    frame[mask] = [255, 100, 100]  # Even lighter red
    
    # Add text labels (simple pixel drawing)
    # Just add some distinctive patterns for identification
    
    # Add frame number indicator
    frame[50:100, 50:50+frame_num*10] = [255, 255, 255]  # White progress bar
    
    return frame

def create_simple_depth_map(width=1920, height=1080):
    """Create corresponding depth map with known values"""
    depth_map = np.ones((height, width), dtype=np.float32) * 0.8  # Background
    
    # Rectangle depths (matching the frame structure)
    depth_map[200:400, 200:500] = 0.7      # Rectangle 1
    depth_map[300:500, 700:1000] = 0.5     # Rectangle 2  
    depth_map[400:600, 1200:1500] = 0.3    # Rectangle 3
    
    # Circle depths
    y_indices, x_indices = np.ogrid[:height, :width]
    radius = 80
    
    # Circle 1
    mask = (x_indices - 400)**2 + (y_indices - 600)**2 <= radius**2
    depth_map[mask] = 0.1
    
    # Circle 2
    mask = (x_indices - 1000)**2 + (y_indices - 600)**2 <= radius**2
    depth_map[mask] = 0.1
    
    # Circle 3
    mask = (x_indices - 1600)**2 + (y_indices - 600)**2 <= radius**2
    depth_map[mask] = 0.1
    
    return depth_map

def save_frame_as_ppm(frame, filename):
    """Save frame as PPM format (simple format that ffmpeg can read)"""
    height, width, channels = frame.shape
    
    with open(filename, 'wb') as f:
        f.write(f'P6\n{width} {height}\n255\n'.encode())
        f.write(frame.tobytes())

def main():
    """Create simple test data"""
    output_dir = Path("simple_test_data")
    output_dir.mkdir(exist_ok=True)
    
    frames_dir = output_dir / "frames"
    depth_dir = output_dir / "depth_maps"
    frames_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    
    num_frames = 30  # 1 second at 30fps
    width, height = 1920, 1080
    
    print(f"Creating {num_frames} simple test frames...")
    
    for frame_num in range(num_frames):
        # Create test frame
        frame = create_simple_test_frame(width, height, frame_num)
        frame_path = frames_dir / f"frame_{frame_num:06d}.ppm"
        save_frame_as_ppm(frame, str(frame_path))
        
        # Create corresponding depth map
        depth_map = create_simple_depth_map(width, height)
        
        # Save depth map as numpy array
        depth_npy_path = depth_dir / f"depth_{frame_num:06d}.npy"
        np.save(str(depth_npy_path), depth_map)
        
        # Save depth map as grayscale image data
        depth_vis = (depth_map * 255).astype(np.uint8)
        depth_ppm_path = depth_dir / f"depth_{frame_num:06d}.ppm"
        
        # Convert grayscale to RGB for PPM
        depth_rgb = np.stack([depth_vis, depth_vis, depth_vis], axis=2)
        save_frame_as_ppm(depth_rgb, str(depth_ppm_path))
        
        if frame_num % 10 == 0:
            print(f"Created frame {frame_num}/{num_frames}")
    
    # Create video using ffmpeg
    print("Creating test video...")
    cmd = f"ffmpeg -y -framerate 30 -i {frames_dir}/frame_%06d.ppm -c:v libx264 -pix_fmt yuv420p -crf 18 {output_dir}/simple_test_video.mp4"
    
    result = os.system(cmd)
    if result == 0:
        print("Video created successfully!")
    else:
        print("FFmpeg command failed, but frames are available")
    
    print(f"\nSimple test data created in {output_dir}/")
    print("Contents:")
    print(f"  - simple_test_video.mp4: Video with known depth structure")
    print(f"  - frames/: Individual frames (.ppm format)")
    print(f"  - depth_maps/: Known depth maps (.npy and .ppm)")
    print()
    print("Depth structure:")
    print("  - Background: 0.8 (far)")
    print("  - Green rectangles: 0.7, 0.5, 0.3 (medium distances)")
    print("  - Red circles: 0.1 (very close)")
    print()
    print("Test the stereo effect:")
    print(f"  python stereo_projector.py {output_dir}/simple_test_video.mp4 -s 00:00 -e 00:01")

if __name__ == "__main__":
    main()