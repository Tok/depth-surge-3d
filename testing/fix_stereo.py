#!/usr/bin/env python3
"""
Analyze and fix the stereography issues in stereo_projector.py
"""

import numpy as np

def analyze_current_disparity_formula():
    """Analyze the current disparity formula to show the problem"""
    print("CURRENT DISPARITY FORMULA ANALYSIS")
    print("="*50)
    
    # Current formula: disparity = baseline * focal_length / (depth_map + 1e-6)
    baseline = 0.1
    focal_length = 1000
    
    print(f"Formula: disparity = {baseline} * {focal_length} / (depth + 1e-6)")
    print()
    
    # Test with normalized depth values [0, 1]
    depth_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # close to far
    depth_labels = ["Very Close", "Close", "Medium", "Far", "Very Far"]
    
    print("Depth Value → Disparity (pixels)")
    print("-" * 35)
    
    for depth, label in zip(depth_values, depth_labels):
        disparity = baseline * focal_length / (depth + 1e-6)
        print(f"{depth:4.1f} ({label:10}) → {disparity:8.1f} px")
    
    print()
    print("PROBLEMS:")
    print("1. Disparity values are HUGE (100-1000 pixels)")
    print("2. Tiny depth differences create massive disparity changes")
    print("3. Real stereo disparity should be 1-20 pixels typically")
    print("4. Close objects appear to have similar disparity to far objects")
    print("   due to clipping at 5% of image width")
    print()

def propose_fixed_formula():
    """Propose a corrected disparity formula"""
    print("PROPOSED FIXED FORMULA")
    print("="*50)
    
    # Corrected approach:
    # 1. Interpret depth map correctly (0=close, 1=far)
    # 2. Use reasonable disparity range (1-20 pixels)
    # 3. Linear or sqrt mapping for better control
    
    baseline = 0.1
    focal_length = 1000
    max_disparity = 20.0  # Maximum disparity in pixels
    min_disparity = 1.0   # Minimum disparity in pixels
    
    print("OPTION 1: Linear mapping")
    print(f"disparity = {max_disparity} - (depth * ({max_disparity - min_disparity}))")
    print("(Inverts depth so 0=close gets max disparity)")
    print()
    
    depth_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    depth_labels = ["Very Close", "Close", "Medium", "Far", "Very Far"]
    
    print("Depth Value → Linear Disparity")
    print("-" * 30)
    
    for depth, label in zip(depth_values, depth_labels):
        disparity = max_disparity - (depth * (max_disparity - min_disparity))
        print(f"{depth:4.1f} ({label:10}) → {disparity:6.1f} px")
    
    print()
    print("OPTION 2: Physically-based (corrected)")
    print("Convert normalized depth to real-world depth first:")
    print("real_depth = min_depth + depth * (max_depth - min_depth)")
    print("disparity = baseline * focal_length / real_depth")
    print()
    
    # Assume scene depth range: 1m to 10m
    min_real_depth = 1.0  # 1 meter
    max_real_depth = 10.0  # 10 meters
    focal_length_mm = 50  # 50mm lens equivalent
    baseline_mm = 65  # 65mm eye separation
    
    print("Assuming:")
    print(f"  Scene depth: {min_real_depth}m to {max_real_depth}m")
    print(f"  Focal length: {focal_length_mm}mm")
    print(f"  Baseline: {baseline_mm}mm")
    print()
    
    print("Depth Value → Real Depth → Disparity")
    print("-" * 40)
    
    for depth, label in zip(depth_values, depth_labels):
        real_depth = min_real_depth + depth * (max_real_depth - min_real_depth)
        # Disparity in pixels (assuming 1920px width = 24mm sensor width)
        disparity_mm = baseline_mm * focal_length_mm / (real_depth * 1000)  # Convert to mm
        disparity_px = disparity_mm * (1920 / 24)  # Convert to pixels
        
        print(f"{depth:4.1f} ({label:10}) → {real_depth:5.1f}m → {disparity_px:6.2f} px")
    
    print()
    print("RECOMMENDATION: Use Option 1 (Linear) for simplicity and control")
    print("Advantages:")
    print("- Predictable disparity range")
    print("- Easy to adjust max/min disparity")
    print("- Works regardless of scene depth assumptions")
    print()

if __name__ == "__main__":
    analyze_current_disparity_formula()
    print("\n")
    propose_fixed_formula()