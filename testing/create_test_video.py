#!/usr/bin/env python3
"""
Create test video with known depth structure for stereo analysis
"""

import cv2
import numpy as np
import os
from pathlib import Path

def create_test_frame(width=1920, height=1080, frame_num=0):
    """Create a test frame with known depth structure"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background (far) - blue
    frame[:, :] = [100, 50, 0]  # Dark blue background
    
    # Middle ground objects - green rectangles at different distances
    for i, distance in enumerate([0.7, 0.5, 0.3]):  # Different depth layers
        color_intensity = int(distance * 255)
        color = [0, color_intensity, 0]  # Green with intensity based on distance
        
        # Create rectangles at different horizontal positions
        x_start = 200 + i * 500
        x_end = x_start + 300
        y_start = 200 + i * 100
        y_end = y_start + 200
        
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, -1)
        
        # Add text labels
        text = f"Depth {distance:.1f}"
        cv2.putText(frame, text, (x_start + 10, y_start + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Foreground (close) - red circles
    for i in range(3):
        center_x = 300 + i * 600
        center_y = 600 + int(50 * np.sin(frame_num * 0.1 + i))  # Slight animation
        radius = 80
        cv2.circle(frame, (center_x, center_y), radius, (0, 0, 255), -1)  # Red circles
        
        # Add text
        cv2.putText(frame, "Close", (center_x - 30, center_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add depth gradient bar on the right side for reference
    gradient_width = 100
    for y in range(height):
        depth_value = y / height  # 0 at top (close), 1 at bottom (far)
        color_val = int(depth_value * 255)
        frame[y, width-gradient_width:width] = [color_val, color_val, color_val]
    
    # Add depth scale labels
    cv2.putText(frame, "CLOSE", (width-90, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "FAR", (width-70, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add frame counter
    cv2.putText(frame, f"Frame {frame_num:03d}", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    return frame

def create_test_depth_map(width=1920, height=1080):
    """Create corresponding depth map with known values"""
    depth_map = np.ones((height, width), dtype=np.float32) * 0.8  # Background depth
    
    # Middle ground rectangles at different depths
    for i, depth_value in enumerate([0.7, 0.5, 0.3]):
        x_start = 200 + i * 500
        x_end = x_start + 300
        y_start = 200 + i * 100
        y_end = y_start + 200
        depth_map[y_start:y_end, x_start:x_end] = depth_value
    
    # Foreground circles (close)
    for i in range(3):
        center_x = 300 + i * 600
        center_y = 600
        radius = 80
        cv2.circle(depth_map, (center_x, center_y), radius, 0.1, -1)  # Very close
    
    # Add depth gradient on the right
    gradient_width = 100
    for y in range(height):
        depth_value = y / height  # Linear gradient from 0 (close) to 1 (far)
        depth_map[y, width-gradient_width:width] = depth_value
    
    return depth_map

def main():
    """Create test video and corresponding depth maps"""
    output_dir = Path("test_data")
    output_dir.mkdir(exist_ok=True)
    
    frames_dir = output_dir / "frames"
    depth_dir = output_dir / "depth_maps"
    frames_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    
    num_frames = 60  # 2 seconds at 30fps
    width, height = 1920, 1080
    
    print(f"Creating {num_frames} test frames...")
    
    for frame_num in range(num_frames):
        # Create test frame
        frame = create_test_frame(width, height, frame_num)
        frame_path = frames_dir / f"frame_{frame_num:06d}.png"
        cv2.imwrite(str(frame_path), frame)
        
        # Create corresponding depth map
        depth_map = create_test_depth_map(width, height)
        
        # Save depth map as grayscale image (0-255)
        depth_vis = (depth_map * 255).astype(np.uint8)
        depth_path = depth_dir / f"depth_{frame_num:06d}.png"
        cv2.imwrite(str(depth_path), depth_vis)
        
        # Save depth map as numpy array for analysis
        depth_npy_path = depth_dir / f"depth_{frame_num:06d}.npy"
        np.save(str(depth_npy_path), depth_map)
        
        if frame_num % 10 == 0:
            print(f"Created frame {frame_num}/{num_frames}")
    
    # Create video using ffmpeg
    print("Creating test video...")
    os.system(f"""
    ffmpeg -y -framerate 30 -i {frames_dir}/frame_%06d.png \
    -c:v libx264 -pix_fmt yuv420p -crf 18 \
    {output_dir}/test_video.mp4
    """)
    
    print(f"Test data created in {output_dir}/")
    print("Contents:")
    print(f"  - test_video.mp4: Video with known depth structure")
    print(f"  - frames/: Individual frames")
    print(f"  - depth_maps/: Known depth maps (.png and .npy)")
    print()
    print("Depth structure:")
    print("  - Background: 0.8 (far)")
    print("  - Green rectangles: 0.7, 0.5, 0.3 (medium distances)")
    print("  - Red circles: 0.1 (very close)")
    print("  - Right gradient: 0.0 (top) to 1.0 (bottom)")
    print()
    print("Test the stereo effect:")
    print(f"  python stereo_projector.py {output_dir}/test_video.mp4 -s 00:00 -e 00:02")

if __name__ == "__main__":
    main()