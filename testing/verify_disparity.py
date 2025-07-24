#!/usr/bin/env python3
"""
Simple manual verification of disparity values in stereo pairs
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

def manual_disparity_check(left_path, right_path):
    """Manually check disparity by comparing pixel positions"""
    left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
    
    if left is None or right is None:
        print(f"Error loading images: {left_path}, {right_path}")
        return
    
    height, width = left.shape
    print(f"Image size: {width}x{height}")
    
    # Expected disparity ranges based on our new formula
    max_disparity = width * 0.05   # 5% of width
    min_disparity = width * 0.005  # 0.5% of width
    
    print(f"Expected disparity range: {min_disparity:.1f} to {max_disparity:.1f} pixels")
    print()
    
    # Check specific regions that should have different depths
    regions = {
        'background': (slice(50, 150), slice(50, 150)),           # Top-left background (depth ~0.8)
        'close_circle_1': (slice(550, 650), slice(350, 450)),     # Red circle area (depth ~0.1)
        'close_circle_2': (slice(550, 650), slice(950, 1050)),    # Second red circle (depth ~0.1)
        'medium_rect_1': (slice(250, 350), slice(250, 450)),      # First green rectangle (depth ~0.7)
        'medium_rect_2': (slice(350, 450), slice(750, 950)),      # Second green rectangle (depth ~0.5)
        'medium_rect_3': (slice(450, 550), slice(1250, 1450)),    # Third green rectangle (depth ~0.3)
    }
    
    print("Manual Disparity Analysis:")
    print("="*50)
    
    for region_name, (y_slice, x_slice) in regions.items():
        try:
            # Extract region from both images
            left_region = left[y_slice, x_slice]
            right_region = right[y_slice, x_slice]
            
            # Find distinctive features in the region
            left_mean = np.mean(left_region)
            right_mean = np.mean(right_region)
            
            # Calculate cross-correlation to find horizontal shift
            correlation = cv2.matchTemplate(left_region, right_region, cv2.TM_CCOEFF_NORMED)
            
            if correlation.size > 0:
                max_corr = np.max(correlation)
                
                # For a more accurate disparity measurement, use template matching
                # across a horizontal strip around the center of the region
                center_y = left_region.shape[0] // 2
                left_strip = left_region[center_y:center_y+1, :]
                
                # Search for this strip in the right image with horizontal offsets
                best_disparity = 0
                best_match = -1
                
                for disparity in range(-int(max_disparity), int(max_disparity)):
                    if disparity < 0:
                        # Shift right image left (positive disparity)
                        right_shifted = right_region[center_y:center_y+1, -disparity:]
                        left_compare = left_strip[:, :right_shifted.shape[1]]
                    elif disparity > 0:
                        # Shift right image right (negative disparity)
                        right_shifted = right_region[center_y:center_y+1, :-disparity]
                        left_compare = left_strip[:, disparity:]
                    else:
                        right_shifted = right_region[center_y:center_y+1, :]
                        left_compare = left_strip
                    
                    if right_shifted.shape[1] > 10 and left_compare.shape[1] > 10:
                        # Calculate correlation
                        corr = cv2.matchTemplate(left_compare, right_shifted, cv2.TM_CCOEFF_NORMED)
                        if corr.size > 0:
                            match_score = np.max(corr)
                            if match_score > best_match:
                                best_match = match_score
                                best_disparity = disparity
                
                # Estimate expected disparity based on our formula
                if 'background' in region_name:
                    expected_depth = 0.8
                elif 'close_circle' in region_name:
                    expected_depth = 0.1
                elif 'rect_1' in region_name:
                    expected_depth = 0.7
                elif 'rect_2' in region_name:
                    expected_depth = 0.5
                elif 'rect_3' in region_name:
                    expected_depth = 0.3
                else:
                    expected_depth = 0.5
                
                expected_disparity = max_disparity - (expected_depth * (max_disparity - min_disparity))
                
                print(f"{region_name:15}: {best_disparity:6.1f}px disparity (expected: {expected_disparity:5.1f}px, depth: {expected_depth:.1f})")
                
                # Visual difference check
                diff = np.abs(left_mean - right_mean)
                if diff > 10:
                    print(f"                 WARNING: Large brightness difference ({diff:.1f})")
                
        except Exception as e:
            print(f"{region_name:15}: ERROR - {str(e)}")
    
    print()
    print("Expected behavior:")
    print("- close_circle regions should have HIGHEST disparity (~86px)")
    print("- background should have LOWEST disparity (~18px)")
    print("- medium rectangles should be in between, ordered by depth")
    print()
    
    # Calculate theoretical values
    print("Theoretical disparity values:")
    depths = [0.1, 0.3, 0.5, 0.7, 0.8]
    labels = ["Very Close", "Medium-Close", "Medium", "Medium-Far", "Far"]
    
    for depth, label in zip(depths, labels):
        theoretical_disp = max_disparity - (depth * (max_disparity - min_disparity))
        print(f"  Depth {depth:.1f} ({label:12}): {theoretical_disp:5.1f}px")

def main():
    parser = argparse.ArgumentParser(description='Manually verify disparity in stereo pairs')
    parser.add_argument('output_dir', help='Output directory containing stereo frames')
    parser.add_argument('--frame', type=int, default=1, help='Frame number to analyze (default: 1)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Look for final stereo frames
    left_frames_dir = output_dir / "6_left_frames_final"
    right_frames_dir = output_dir / "6_right_frames_final"
    
    if not left_frames_dir.exists() or not right_frames_dir.exists():
        print("Error: Could not find final stereo frame directories")
        print(f"Looking for: {left_frames_dir} and {right_frames_dir}")
        return
    
    # Find frame files
    left_files = sorted(left_frames_dir.glob("frame_*.png"))
    right_files = sorted(right_frames_dir.glob("frame_*.png"))
    
    if not left_files or not right_files:
        print("Error: No frame files found")
        return
    
    if args.frame > len(left_files):
        print(f"Error: Frame {args.frame} not found (available: 1-{len(left_files)})")
        return
    
    # Analyze the specified frame (1-indexed)
    frame_idx = args.frame - 1
    left_path = left_files[frame_idx]
    right_path = right_files[frame_idx]
    
    print(f"Verifying disparity in frame {args.frame}:")
    print(f"  Left: {left_path}")
    print(f"  Right: {right_path}")
    print()
    
    manual_disparity_check(left_path, right_path)

if __name__ == "__main__":
    main()