#!/usr/bin/env python3
"""
Analyze stereo pair output to verify correct depth handling
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

def load_stereo_pair(left_path, right_path):
    """Load left and right stereo images"""
    left = cv2.imread(str(left_path))
    right = cv2.imread(str(right_path))
    
    if left is None or right is None:
        raise ValueError(f"Could not load stereo pair: {left_path}, {right_path}")
    
    # Convert to grayscale for disparity analysis
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    
    return left, right, left_gray, right_gray

def calculate_disparity_map(left_gray, right_gray):
    """Calculate disparity map between left and right images"""
    # Create stereo matcher
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    
    # Calculate disparity
    disparity = stereo.compute(left_gray, right_gray)
    
    # Normalize disparity for visualization
    disparity_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return disparity, disparity_norm

def analyze_disparity_regions(disparity, left_color):
    """Analyze disparity in specific regions where we know the expected depth"""
    height, width = disparity.shape
    
    results = {}
    
    # Define regions based on our test pattern
    regions = {
        'background': (slice(0, 150), slice(0, width-100)),  # Top area, excluding gradient
        'close_circles': (slice(550, 650), slice(250, 350)),  # Red circles area
        'medium_rect_1': (slice(200, 400), slice(200, 500)),  # First green rectangle
        'medium_rect_2': (slice(300, 500), slice(700, 1000)),  # Second green rectangle
        'medium_rect_3': (slice(400, 600), slice(1200, 1500)),  # Third green rectangle
    }
    
    for region_name, (y_slice, x_slice) in regions.items():
        try:
            region_disparity = disparity[y_slice, x_slice]
            
            # Calculate statistics
            valid_mask = region_disparity > 0  # Remove invalid disparities
            if np.any(valid_mask):
                valid_disparities = region_disparity[valid_mask].astype(np.float32) / 16.0  # OpenCV uses 16-bit fixed point
                
                mean_disp = np.mean(valid_disparities)
                std_disp = np.std(valid_disparities)
                min_disp = np.min(valid_disparities)
                max_disp = np.max(valid_disparities)
                
                results[region_name] = {
                    'mean': mean_disp,
                    'std': std_disp,
                    'min': min_disp,
                    'max': max_disp,
                    'pixel_count': np.sum(valid_mask)
                }
            else:
                results[region_name] = {'error': 'No valid disparities found'}
                
        except Exception as e:
            results[region_name] = {'error': str(e)}
    
    return results

def create_analysis_visualization(left, right, disparity_norm, analysis_results, output_dir):
    """Create visualization of the stereo analysis using OpenCV"""
    # Save individual images
    cv2.imwrite(str(output_dir / "analysis_left.png"), left)
    cv2.imwrite(str(output_dir / "analysis_right.png"), right)
    
    # Create colored disparity map
    disparity_colored = cv2.applyColorMap(disparity_norm, cv2.COLORMAP_JET)
    cv2.imwrite(str(output_dir / "analysis_disparity.png"), disparity_colored)
    
    # Create analysis summary image
    summary_img = np.zeros((800, 600, 3), dtype=np.uint8)
    
    # Add title
    cv2.putText(summary_img, "Stereo Analysis Results", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Add analysis results
    y_pos = 80
    expected_order = ['close_circles', 'medium_rect_3', 'medium_rect_2', 'medium_rect_1', 'background']
    expected_depths = [0.1, 0.3, 0.5, 0.7, 0.8]
    
    for i, region in enumerate(expected_order):
        if region in analysis_results and 'mean' in analysis_results[region]:
            result = analysis_results[region]
            expected_depth = expected_depths[i]
            
            text = f"{region}: {result['mean']:.1f}px (expected depth: {expected_depth:.1f})"
            cv2.putText(summary_img, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 30
        else:
            text = f"{region}: ERROR"
            cv2.putText(summary_img, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            y_pos += 30
    
    # Check disparity order
    y_pos += 20
    cv2.putText(summary_img, "Depth Order Check:", (20, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
    y_pos += 30
    
    cv2.putText(summary_img, "(Closer objects should have LARGER disparity)", (20, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_pos += 30
    
    valid_regions = [r for r in expected_order if r in analysis_results and 'mean' in analysis_results[r]]
    if len(valid_regions) >= 2:
        disparities = [analysis_results[r]['mean'] for r in valid_regions]
        
        for i in range(len(valid_regions)-1):
            current_disp = disparities[i]
            next_disp = disparities[i+1]
            is_correct = current_disp > next_disp
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            status = "CORRECT" if is_correct else "WRONG"
            
            text = f"{valid_regions[i]} > {valid_regions[i+1]}: {status}"
            cv2.putText(summary_img, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_pos += 25
            
            detail_text = f"  {current_disp:.1f} > {next_disp:.1f}: {current_disp > next_disp}"
            cv2.putText(summary_img, detail_text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_pos += 25
    
    cv2.imwrite(str(output_dir / "analysis_summary.png"), summary_img)
    
    return summary_img

def main():
    parser = argparse.ArgumentParser(description='Analyze stereo pair output')
    parser.add_argument('output_dir', help='Output directory containing stereo frames')
    parser.add_argument('--frame', type=int, default=0, help='Frame number to analyze (default: 0)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Look for stereo frames
    left_frames_dir = output_dir / "6_left_frames_final"
    right_frames_dir = output_dir / "6_right_frames_final"
    
    if not left_frames_dir.exists() or not right_frames_dir.exists():
        print("Error: Could not find stereo frame directories")
        print(f"Looking for: {left_frames_dir} and {right_frames_dir}")
        return
    
    # Find frame files
    left_files = sorted(left_frames_dir.glob("frame_*.png"))
    right_files = sorted(right_frames_dir.glob("frame_*.png"))
    
    if not left_files or not right_files:
        print("Error: No frame files found")
        return
    
    if args.frame >= len(left_files):
        print(f"Error: Frame {args.frame} not found (max: {len(left_files)-1})")
        return
    
    # Load stereo pair
    left_path = left_files[args.frame]
    right_path = right_files[args.frame]
    
    print(f"Analyzing frame {args.frame}:")
    print(f"  Left: {left_path}")
    print(f"  Right: {right_path}")
    
    try:
        left, right, left_gray, right_gray = load_stereo_pair(left_path, right_path)
        
        # Calculate disparity
        print("Calculating disparity map...")
        disparity, disparity_norm = calculate_disparity_map(left_gray, right_gray)
        
        # Analyze regions
        print("Analyzing regions...")
        analysis_results = analyze_disparity_regions(disparity, left)
        
        # Create visualization
        print("Creating visualization...")
        summary_img = create_analysis_visualization(left, right, disparity_norm, analysis_results, output_dir)
        
        print(f"Analysis images saved to: {output_dir}/")
        print(f"  - analysis_left.png: Left stereo image")
        print(f"  - analysis_right.png: Right stereo image") 
        print(f"  - analysis_disparity.png: Calculated disparity map")
        print(f"  - analysis_summary.png: Analysis summary")
        
        # Print summary to console
        print("\n" + "="*50)
        print("STEREO ANALYSIS SUMMARY")
        print("="*50)
        
        for region, result in analysis_results.items():
            if 'mean' in result:
                print(f"{region:15}: {result['mean']:6.1f} px disparity (std: {result['std']:4.1f})")
            else:
                print(f"{region:15}: ERROR - {result.get('error', 'Unknown')}")
        
        print("\nExpected behavior:")
        print("- Closer objects should have LARGER disparity")
        print("- close_circles should have highest disparity")
        print("- background should have lowest disparity")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()