#!/usr/bin/env python3
"""
Depth Surge 3D - Convert 2D videos to immersive 3D VR format using advanced AI depth estimation
"""

import argparse
import os
import sys
import subprocess
import cv2
import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import time

class ProgressTracker:
    """Enhanced progress tracking for both serial and batch processing modes"""
    def __init__(self, total_frames, processing_mode='serial'):
        self.total_frames = total_frames
        self.processing_mode = processing_mode
        self.start_time = time.time()
        
        if processing_mode == 'serial':
            self.current_frame = 0
            self.current_step = ""
        else:  # batch mode
            self.steps = [
                "Frame Extraction",
                "Super Sampling", 
                "Depth Map Generation",
                "Stereo Pair Creation",
                "Fisheye Distortion",
                "Final Processing",
                "Video Creation"
            ]
            self.current_step_index = 0
            self.step_progress = 0
            self.step_total = 0
            
    def update_serial(self, frame_num, step_description):
        """Update progress for serial processing"""
        self.current_frame = frame_num
        self.current_step = step_description
        self._display_serial()
        
    def update_batch_step(self, step_name, progress=0, total=0):
        """Update progress for batch processing step"""
        if step_name in self.steps:
            self.current_step_index = self.steps.index(step_name)
        self.step_progress = progress
        self.step_total = total
        self._display_batch()
        
    def update_batch_progress(self, progress, total=None):
        """Update progress within current batch step"""
        self.step_progress = progress
        if total is not None:
            self.step_total = total
        self._display_batch()
        
    def _display_serial(self):
        """Display serial mode progress"""
        if self.total_frames > 0:
            percentage = (self.current_frame / self.total_frames) * 100
            elapsed = time.time() - self.start_time
            if self.current_frame > 0:
                eta = (elapsed / self.current_frame) * (self.total_frames - self.current_frame)
                eta_str = f"ETA: {eta:.1f}s"
            else:
                eta_str = "ETA: --"
            print(f"\r[SERIAL] Frame {self.current_frame}/{self.total_frames} ({percentage:.1f}%) - {self.current_step} - {eta_str}", end="", flush=True)
    
    def _display_batch(self):
        """Display batch mode progress"""
        step_name = self.steps[self.current_step_index] if self.current_step_index < len(self.steps) else "Processing"
        overall_percentage = ((self.current_step_index + (self.step_progress / max(self.step_total, 1))) / len(self.steps)) * 100
        
        elapsed = time.time() - self.start_time
        eta = (elapsed / max(overall_percentage, 1)) * (100 - overall_percentage) if overall_percentage > 0 else 0
        eta_str = f"ETA: {eta:.1f}s" if eta > 0 else "ETA: --"
        
        if self.step_total > 0:
            step_percentage = (self.step_progress / self.step_total) * 100
            print(f"\r[BATCH] Step {self.current_step_index + 1}/{len(self.steps)}: {step_name} ({step_percentage:.1f}%) | Overall: {overall_percentage:.1f}% - {eta_str}", end="", flush=True)
        else:
            print(f"\r[BATCH] Step {self.current_step_index + 1}/{len(self.steps)}: {step_name} | Overall: {overall_percentage:.1f}% - {eta_str}", end="", flush=True)
    
    def finish(self, message="Processing complete"):
        """Finish progress tracking"""
        elapsed = time.time() - self.start_time
        print(f"\r{message} - Total time: {elapsed:.1f}s")
        print()  # New line

# Legacy ProgressBar for compatibility
class ProgressBar:
    """Simple console progress bar without external dependencies - Legacy wrapper"""
    def __init__(self, total, description="Processing", width=50):
        self.tracker = ProgressTracker(total, 'serial')
        self.description = description
        
    def update(self, step=1, description=None):
        self.tracker.current_frame += step
        if description:
            self.description = description
        self.tracker.update_serial(self.tracker.current_frame, self.description)
        
    def set_progress(self, current, description=None):
        if description:
            self.description = description
        self.tracker.update_serial(current, self.description)
    
    def finish(self, description=None):
        if description:
            self.description = description
        self.tracker.finish(self.description)

class StereoProjector:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Depth Anything V2 model"""
        try:
            # Import depth_anything_v2 from the model directory
            sys.path.append(str(Path(self.model_path).parent))
            from depth_anything_v2.dpt import DepthAnythingV2
            
            self.model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded Depth Anything V2 model on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def extract_frames(self, video_path, output_dir, start_time=None, end_time=None, target_fps=60, min_resolution="1080p"):
        """Extract frames from video using ffmpeg with optional upscaling (no interpolation - that happens during final video creation)"""
        frames_dir = Path(output_dir) / "1_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Get original video properties
        cap = cv2.VideoCapture(str(video_path))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        print(f"Original video: {original_width}x{original_height} @ {original_fps}fps")
        
        # Determine target resolution
        if min_resolution == "1080p":
            target_width, target_height = 1920, 1080
        elif min_resolution == "720p":
            target_width, target_height = 1280, 720
        elif min_resolution == "4k":
            target_width, target_height = 3840, 2160
        else:
            target_width, target_height = original_width, original_height
        
        # Check if we need upscaling
        need_upscale = original_width < target_width or original_height < target_height
        
        # Build video filter chain (NO interpolation - only upscaling if needed)
        filters = []
        
        # Add upscaling if needed
        if need_upscale:
            # Use Lanczos scaling for high quality upscaling
            filters.append(f"scale={target_width}:{target_height}:flags=lanczos")
            print(f"Upscaling: {original_width}x{original_height} -> {target_width}x{target_height}")
        
        # Build ffmpeg command - extract at ORIGINAL fps
        cmd = ["ffmpeg", "-y"]  # -y to overwrite existing files
        
        # Add start time if specified
        if start_time:
            cmd.extend(["-ss", start_time])
        
        cmd.extend(["-i", video_path])
        
        # Add end time if specified (as duration from start)
        if end_time and start_time:
            # Calculate duration
            start_seconds = self._time_to_seconds(start_time)
            end_seconds = self._time_to_seconds(end_time)
            duration = end_seconds - start_seconds
            if duration > 0:
                cmd.extend(["-t", str(duration)])
        elif end_time and not start_time:
            # If only end time specified, treat as duration from beginning
            cmd.extend(["-t", str(self._time_to_seconds(end_time))])
        
        # Add video filters if any
        if filters:
            cmd.extend(["-vf", ",".join(filters)])
        
        # Add remaining options - use ORIGINAL fps for extraction
        cmd.extend([
            "-r", str(original_fps),  # Extract at original fps
            "-q:v", "2",      # High quality
            "-pix_fmt", "rgb24",  # Ensure consistent pixel format
            str(frames_dir / "frame_%06d.png")
        ])
        
        time_info = ""
        if start_time or end_time:
            time_info = f" (from {start_time or '00:00'} to {end_time or 'end'})"
        
        enhancement_info = []
        if need_upscale:
            enhancement_info.append(f"upscaling to {target_width}x{target_height}")
        
        enhancement_str = " with " + " and ".join(enhancement_info) if enhancement_info else ""
        
        print(f"Extracting frames at original {original_fps}fps from {video_path}{time_info}{enhancement_str}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error extracting frames: {result.stderr}")
            sys.exit(1)
        
        # Get list of extracted frames
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        print(f"Extracted {len(frame_files)} frames at original fps")
        return frame_files
    
    def _time_to_seconds(self, time_str):
        """Convert mm:ss format to seconds"""
        try:
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes, seconds = map(int, parts)
                return minutes * 60 + seconds
            elif len(parts) == 3:
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
            else:
                raise ValueError("Invalid time format")
        except ValueError:
            print(f"Error: Invalid time format '{time_str}'. Use mm:ss or hh:mm:ss format.")
            sys.exit(1)
    
    def generate_depth_map(self, image_path):
        """Generate depth map for a single frame"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Predict depth
        with torch.no_grad():
            depth = self.model.infer_image(image)
        
        # Normalize depth to 0-1 range
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
        return depth_normalized
    
    def create_stereo_pair(self, image, depth_map, baseline=0.1, focal_length=1000, hole_fill_quality='advanced'):
        """Create left and right stereo images from depth map"""
        h, w = image.shape[:2]
        
        # Create disparity map using controlled linear mapping
        # Depth map: 0 = close, 1 = far (typical AI depth map format)
        # Disparity: larger values = closer objects (correct for stereo)
        
        max_disparity = w * 0.05   # 5% of width (96px for 1920px, 192px for 4K)
        min_disparity = w * 0.005  # 0.5% of width (10px for 1920px, 19px for 4K)
        
        # Linear mapping: close objects (depth=0) get max disparity, far objects (depth=1) get min disparity
        disparity = max_disparity - (depth_map * (max_disparity - min_disparity))
        
        # Ensure disparity is within reasonable bounds
        disparity = np.clip(disparity, min_disparity, max_disparity)
        
        # Create left and right images with symmetric shifts
        left_img = np.zeros_like(image)
        right_img = np.zeros_like(image)
        
        # Half disparity for each eye (symmetric stereo)
        half_disparity = disparity / 2
        
        # Create left image (shifted left)
        for y in range(h):
            for x in range(w):
                shift = int(half_disparity[y, x])
                src_x = x + shift
                if 0 <= src_x < w:
                    left_img[y, x] = image[y, src_x]
        
        # Create right image (shifted right)
        for y in range(h):
            for x in range(w):
                shift = int(half_disparity[y, x])
                src_x = x - shift
                if 0 <= src_x < w:
                    right_img[y, x] = image[y, src_x]
        
        # Apply hole filling based on quality setting
        if hole_fill_quality == 'advanced':
            # Advanced hole filling with multiple techniques
            left_img = self._fill_stereo_holes(left_img, depth_map, half_disparity, 'left')
            right_img = self._fill_stereo_holes(right_img, depth_map, half_disparity, 'right')
        else:
            # Fast hole filling using simple inpainting
            left_img = self._fill_holes_fast(left_img)
            right_img = self._fill_holes_fast(right_img)
        
        return left_img, right_img
    
    def _fill_holes_fast(self, image):
        """Fast hole filling using simple OpenCV inpainting"""
        # Identify holes (completely black pixels)
        if len(image.shape) == 3:
            hole_mask = np.all(image == 0, axis=2).astype(np.uint8)
        else:
            hole_mask = (image == 0).astype(np.uint8)
        
        if not np.any(hole_mask):
            return image  # No holes to fill
            
        # Use simple TELEA inpainting for speed
        return cv2.inpaint(image, hole_mask, 3, cv2.INPAINT_TELEA)
    
    def _fill_stereo_holes(self, stereo_img, depth_map, disparity_map, eye_type):
        """Advanced hole filling for stereo projection using multiple techniques
        
        Args:
            stereo_img: The stereo image with holes (zeros)
            depth_map: Original depth map for depth-guided filling
            disparity_map: Disparity map used for projection
            eye_type: 'left' or 'right' for directional filling
            
        Returns:
            Filled stereo image with natural-looking content
        """
        h, w = stereo_img.shape[:2]
        filled_img = stereo_img.copy()
        
        # 1. Identify holes (completely black pixels)
        if len(stereo_img.shape) == 3:
            hole_mask = np.all(stereo_img == 0, axis=2).astype(np.uint8)
        else:
            hole_mask = (stereo_img == 0).astype(np.uint8)
        
        if not np.any(hole_mask):
            return filled_img  # No holes to fill
            
        # 2. Small hole filling using neighboring disparity estimates
        filled_img = self._fill_small_holes(filled_img, hole_mask, depth_map, disparity_map, eye_type)
        
        # 3. Update hole mask after small hole filling
        if len(filled_img.shape) == 3:
            hole_mask = np.all(filled_img == 0, axis=2).astype(np.uint8)
        else:
            hole_mask = (filled_img == 0).astype(np.uint8)
            
        # 4. Medium hole filling using patch-based methods with depth guidance
        if np.any(hole_mask):
            filled_img = self._fill_medium_holes(filled_img, hole_mask, depth_map)
            
        # 5. Update hole mask after medium hole filling
        if len(filled_img.shape) == 3:
            hole_mask = np.all(filled_img == 0, axis=2).astype(np.uint8)
        else:
            hole_mask = (filled_img == 0).astype(np.uint8)
            
        # 6. Large hole filling using advanced inpainting for remaining areas
        if np.any(hole_mask):
            filled_img = self._fill_large_holes(filled_img, hole_mask)
            
        return filled_img
    
    def _fill_small_holes(self, image, hole_mask, depth_map, disparity_map, eye_type):
        """Fill small holes (1-5 pixels) using neighboring disparity estimates"""
        h, w = image.shape[:2]
        filled = image.copy()
        
        # Find small connected hole regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hole_mask, connectivity=8)
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area <= 5:  # Small holes only
                # Get hole coordinates
                hole_coords = np.where(labels == i)
                
                for y, x in zip(hole_coords[0], hole_coords[1]):
                    # Find nearest valid pixels with similar depth
                    target_depth = depth_map[y, x]
                    best_pixel = self._find_depth_guided_pixel(image, depth_map, x, y, target_depth, eye_type)
                    
                    if best_pixel is not None:
                        filled[y, x] = best_pixel
                        
        return filled
    
    def _find_depth_guided_pixel(self, image, depth_map, x, y, target_depth, eye_type, search_radius=10):
        """Find the best pixel to use for hole filling based on depth similarity"""
        h, w = image.shape[:2]
        best_pixel = None
        min_depth_diff = float('inf')
        
        # Search in expanding rings around the hole
        for r in range(1, search_radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:  # Only check ring perimeter
                        continue
                        
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        # Check if pixel is valid (not a hole)
                        if len(image.shape) == 3:
                            is_valid = not np.all(image[ny, nx] == 0)
                        else:
                            is_valid = image[ny, nx] != 0
                            
                        if is_valid:
                            depth_diff = abs(depth_map[ny, nx] - target_depth)
                            
                            # Prefer pixels with similar depth
                            if depth_diff < min_depth_diff:
                                min_depth_diff = depth_diff
                                best_pixel = image[ny, nx].copy()
                                
            if best_pixel is not None:
                break  # Found a good match
                
        return best_pixel
    
    def _fill_medium_holes(self, image, hole_mask, depth_map, patch_size=7):
        """Fill medium holes (6-50 pixels) using patch-based methods with depth guidance"""
        h, w = image.shape[:2]
        filled = image.copy()
        
        # Find medium connected hole regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hole_mask, connectivity=8)
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if 6 <= area <= 50:  # Medium holes only
                # Get hole bounding box
                x, y, w_hole, h_hole = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                
                # Find best matching patch based on surrounding context and depth
                best_patch = self._find_best_patch(image, depth_map, x, y, w_hole, h_hole, patch_size)
                
                if best_patch is not None:
                    # Apply patch to hole region
                    hole_coords = np.where(labels == i)
                    for hy, hx in zip(hole_coords[0], hole_coords[1]):
                        # Calculate relative position within the hole bounding box
                        rel_y, rel_x = hy - y, hx - x
                        if 0 <= rel_y < best_patch.shape[0] and 0 <= rel_x < best_patch.shape[1]:
                            filled[hy, hx] = best_patch[rel_y, rel_x]
                            
        return filled
    
    def _find_best_patch(self, image, depth_map, hole_x, hole_y, hole_w, hole_h, patch_size):
        """Find the best patch to fill a medium-sized hole based on context and depth"""
        h, w = image.shape[:2]
        
        # Calculate context around the hole
        context_size = patch_size
        best_patch = None
        min_distance = float('inf')
        
        # Get hole region depth statistics
        hole_depth_mean = np.mean(depth_map[hole_y:hole_y+hole_h, hole_x:hole_x+hole_w])
        
        # Search for similar patches in the image
        for sy in range(0, h - hole_h, 2):  # Step by 2 for efficiency
            for sx in range(0, w - hole_w, 2):
                # Skip if this region overlaps with any holes
                candidate_region = image[sy:sy+hole_h, sx:sx+hole_w]
                if len(image.shape) == 3:
                    if np.any(np.all(candidate_region == 0, axis=2)):
                        continue
                else:
                    if np.any(candidate_region == 0):
                        continue
                
                # Check depth similarity
                candidate_depth_mean = np.mean(depth_map[sy:sy+hole_h, sx:sx+hole_w])
                depth_diff = abs(candidate_depth_mean - hole_depth_mean)
                
                # Calculate context similarity (area around the patch)
                context_score = self._calculate_context_similarity(image, hole_x, hole_y, sx, sy, context_size)
                
                # Combined score (lower is better)
                total_score = depth_diff * 10 + context_score
                
                if total_score < min_distance:
                    min_distance = total_score
                    best_patch = candidate_region.copy()
                    
        return best_patch
    
    def _calculate_context_similarity(self, image, hole_x, hole_y, candidate_x, candidate_y, context_size):
        """Calculate similarity between contexts around hole and candidate patch"""
        h, w = image.shape[:2]
        
        # Get context regions (pixels around the hole and candidate)
        hole_context = self._get_context_region(image, hole_x, hole_y, context_size)
        candidate_context = self._get_context_region(image, candidate_x, candidate_y, context_size)
        
        if hole_context is None or candidate_context is None:
            return float('inf')
            
        # Calculate mean squared error between contexts
        if hole_context.shape != candidate_context.shape:
            return float('inf')
            
        diff = hole_context.astype(np.float32) - candidate_context.astype(np.float32)
        mse = np.mean(diff ** 2)
        
        return mse
    
    def _get_context_region(self, image, x, y, context_size):
        """Get context region around a position, excluding the center area"""
        h, w = image.shape[:2]
        
        # Define context bounds
        x1 = max(0, x - context_size)
        y1 = max(0, y - context_size)
        x2 = min(w, x + context_size)
        y2 = min(h, y + context_size)
        
        if x2 - x1 < context_size or y2 - y1 < context_size:
            return None
            
        context_region = image[y1:y2, x1:x2].copy()
        
        # Mask out the center (hole) area
        center_x1 = max(0, x - x1)
        center_y1 = max(0, y - y1)
        center_x2 = min(context_region.shape[1], center_x1 + (x2 - x1) // 2)
        center_y2 = min(context_region.shape[0], center_y1 + (y2 - y1) // 2)
        
        if len(image.shape) == 3:
            context_region[center_y1:center_y2, center_x1:center_x2] = 0
        else:
            context_region[center_y1:center_y2, center_x1:center_x2] = 0
            
        return context_region
    
    def _fill_large_holes(self, image, hole_mask):
        """Fill large holes using advanced inpainting algorithms"""
        # Use both TELEA and NS algorithms and blend results for better quality
        try:
            # TELEA method - good for texture preservation
            filled_telea = cv2.inpaint(image, hole_mask, 5, cv2.INPAINT_TELEA)
            
            # Navier-Stokes method - good for smooth regions
            filled_ns = cv2.inpaint(image, hole_mask, 5, cv2.INPAINT_NS)
            
            # Blend the two results based on local gradient
            # High gradient areas use TELEA (better for textures)
            # Low gradient areas use NS (better for smooth regions)
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            # Calculate gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Normalize gradient for blending weights
            gradient_norm = cv2.GaussianBlur(gradient_mag, (5, 5), 0)
            gradient_norm = (gradient_norm - gradient_norm.min()) / (gradient_norm.max() - gradient_norm.min() + 1e-8)
            
            # Create blending weights
            if len(image.shape) == 3:
                weights_telea = np.stack([gradient_norm] * 3, axis=2)
                weights_ns = 1.0 - weights_telea
            else:
                weights_telea = gradient_norm
                weights_ns = 1.0 - weights_telea
                
            # Blend the two inpainting results
            filled_blend = (filled_telea * weights_telea + filled_ns * weights_ns).astype(np.uint8)
            
            return filled_blend
            
        except Exception as e:
            # Fallback to simple TELEA if blending fails
            return cv2.inpaint(image, hole_mask, 3, cv2.INPAINT_TELEA)
    
    def apply_fisheye_distortion(self, image, projection='equidistant', fov_degrees=180):
        """Apply fisheye distortion for hemispherical VR projection
        
        Args:
            image: Input image to distort
            projection: Fisheye projection type ('equidistant', 'stereographic', 'equisolid', 'orthographic')
            fov_degrees: Field of view in degrees for the fisheye projection (120-220)
        """
        height, width = image.shape[:2]
        cx, cy = width / 2, height / 2
        
        # Create coordinate matrices for the OUTPUT image
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to centered coordinates
        x_c = x - cx
        y_c = y - cy
        
        # Calculate polar coordinates for each OUTPUT pixel
        r_out = np.sqrt(x_c**2 + y_c**2)
        theta = np.arctan2(y_c, x_c)
        
        # Use diagonal radius for full coverage - this ensures we project the entire image
        max_radius = np.sqrt(width**2 + height**2) / 2.0
        
        # Normalize radius to [0, 1] based on the diagonal
        r_norm = r_out / max_radius
        
        # All pixels within the image bounds get processed
        mask = r_out <= max_radius
        
        # Calculate the corresponding INPUT radius for each OUTPUT pixel
        # FISHEYE EFFECT: compress edges to fit wide FOV, keep center large
        distortion_strength = 1.8  # Strength of the fisheye compression
        
        # Initialize with the output radius (no distortion)
        r_input = r_out.copy().astype(np.float32)
        
        # Apply REVERSE fisheye mapping: compress edges, enlarge center
        if projection == 'equidistant':
            # Fisheye compression: r_input = r_out * (1 + k * r_out^2)
            # This pushes edge content toward the edges, compressing more FOV into the circle
            r_input[mask] = r_norm[mask] * (1 + distortion_strength * r_norm[mask]**2) * max_radius
        elif projection == 'stereographic':
            # Stereographic projection with stronger edge compression
            r_input[mask] = r_norm[mask] * (1 + 1.2 * distortion_strength * r_norm[mask]**2) * max_radius
        elif projection == 'equisolid':
            # Equisolid angle projection - moderate compression
            r_input[mask] = r_norm[mask] * (1 + 0.8 * distortion_strength * r_norm[mask]**2) * max_radius
        elif projection == 'orthographic':
            # Orthographic projection - mild compression
            r_input[mask] = r_norm[mask] * (1 + 0.6 * distortion_strength * r_norm[mask]**2) * max_radius
        else:
            # Default equidistant
            r_input[mask] = r_norm[mask] * (1 + distortion_strength * r_norm[mask]**2) * max_radius
        
        # Convert back to Cartesian coordinates for INPUT sampling
        x_input = r_input * np.cos(theta) + cx
        y_input = r_input * np.sin(theta) + cy
        
        # Ensure coordinates are within image bounds
        x_input = np.clip(x_input, 0, width - 1).astype(np.float32)
        y_input = np.clip(y_input, 0, height - 1).astype(np.float32)
        
        # Apply the mapping: for each output pixel, sample from the calculated input position
        fisheye_image = cv2.remap(image, x_input, y_input, cv2.INTER_LINEAR, 
                                 borderMode=cv2.BORDER_REFLECT_101)
        
        return fisheye_image
    
    def apply_center_crop(self, image, crop_factor=0.7):
        """Apply center crop to reduce edge artifacts in VR frames"""
        height, width = image.shape[:2]
        
        # Calculate crop dimensions
        crop_width = int(width * crop_factor)
        crop_height = int(height * crop_factor)
        
        # Calculate crop coordinates (center crop)
        x = (width - crop_width) // 2
        y = (height - crop_height) // 2
        
        # Crop the center region
        cropped = image[y:y+crop_height, x:x+crop_width]
        
        # Resize back to original dimensions
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)
    
    def apply_fisheye_square_crop(self, image, target_eye_width, target_eye_height, fisheye_crop_factor=1.25):
        """Apply smart square crop that fits optimally within fisheye circle"""
        height, width = image.shape[:2]
        
        # The fisheye circle diameter is roughly the smaller dimension
        circle_diameter = min(width, height)
        
        # Optimal inscribed square in circle: side = diameter / √2
        # Use configurable factor to control crop size relative to fisheye circle
        optimal_square_side = int(circle_diameter / 1.414 * fisheye_crop_factor)
        
        # Calculate crop position (center)
        x = (width - optimal_square_side) // 2
        y = (height - optimal_square_side) // 2
        
        # Crop to optimal square within fisheye circle
        cropped_square = image[y:y+optimal_square_side, x:x+optimal_square_side]
        
        # Scale to target eye dimensions
        return cv2.resize(cropped_square, (target_eye_width, target_eye_height), interpolation=cv2.INTER_CUBIC)
    
    def create_vr_format(self, left_img, right_img, format_type='side_by_side', apply_crop=True, crop_factor=0.7, target_resolution=None):
        """Combine left and right images for VR viewing with optional center cropping and target resolution"""
        
        # Apply center crop to reduce edge artifacts if enabled
        if apply_crop:
            left_processed = self.apply_center_crop(left_img, crop_factor)
            right_processed = self.apply_center_crop(right_img, crop_factor)
        else:
            left_processed = left_img
            right_processed = right_img
        
        # If target resolution is specified, scale each eye to the target per-eye dimensions
        if target_resolution is not None:
            final_width, final_height = target_resolution
            
            # Calculate per-eye dimensions based on VR format
            if format_type.startswith('side_by_side'):
                # Side-by-side: each eye gets half the width, full height
                eye_width = final_width // 2
                eye_height = final_height
            elif format_type.startswith('over_under'):
                # Over-under: each eye gets full width, half the height
                eye_width = final_width
                eye_height = final_height // 2
            else:
                # Default to side-by-side
                eye_width = final_width // 2
                eye_height = final_height
            
            # Scale each eye image to the target per-eye dimensions
            left_processed = self._scale_to_eye_format(left_processed, eye_width, eye_height)
            right_processed = self._scale_to_eye_format(right_processed, eye_width, eye_height)
        
        if format_type == 'side_by_side' or format_type == 'side_by_side_lr':
            # Side-by-side format (left image on left, right on right)
            vr_frame = np.hstack([left_processed, right_processed])
        elif format_type == 'side_by_side_rl':
            # Side-by-side format (right image on left, left on right)
            vr_frame = np.hstack([right_processed, left_processed])
        elif format_type == 'over_under' or format_type == 'over_under_lr':
            # Over-under format (left on top, right on bottom)
            vr_frame = np.vstack([left_processed, right_processed])
        elif format_type == 'over_under_rl':
            # Over-under format (right on top, left on bottom)
            vr_frame = np.vstack([right_processed, left_processed])
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return vr_frame
    
    def _scale_to_eye_format(self, image, target_width, target_height):
        """Scale image to target eye dimensions while maintaining aspect ratio and centering"""
        h, w = image.shape[:2]
        
        # Calculate scaling to fit within target dimensions while maintaining aspect ratio
        scale_w = target_width / w
        scale_h = target_height / h
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Create target-sized canvas and center the resized image
        canvas = np.zeros((target_height, target_width, image.shape[2] if len(image.shape) == 3 else 1), dtype=image.dtype)
        
        # Calculate centering offsets
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        
        # Place resized image in center of canvas
        if len(image.shape) == 3:
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        else:
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w, 0] = resized
        
        return canvas
    
    def determine_super_sample_resolution(self, original_width, original_height, super_sample_setting):
        """Determine target super sampling resolution based on input and setting"""
        if super_sample_setting == "none":
            return original_width, original_height
        
        # Define resolution mappings
        resolutions = {
            "720p": (1280, 720),
            "1080p": (1920, 1080), 
            "4k": (3840, 2160)
        }
        
        if super_sample_setting in resolutions:
            return resolutions[super_sample_setting]
        
        if super_sample_setting == "auto":
            # Auto super sampling logic
            if original_width <= 640:  # SD video
                return resolutions["720p"]
            elif original_width <= 1280:  # 720p video
                return resolutions["1080p"] 
            elif original_width <= 1920:  # 1080p video
                return resolutions["4k"]
            else:  # Already high resolution
                return original_width, original_height
        
        return original_width, original_height
    
    def apply_super_sampling(self, image, target_width, target_height):
        """Apply super sampling (upscaling) to image using high-quality interpolation"""
        if image.shape[1] == target_width and image.shape[0] == target_height:
            return image
            
        # Use INTER_CUBIC for upscaling, INTER_AREA for downscaling
        if target_width > image.shape[1] or target_height > image.shape[0]:
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_AREA
            
        return cv2.resize(image, (target_width, target_height), interpolation=interpolation)
    
    def determine_vr_output_resolution(self, original_width, original_height, vr_resolution_setting, vr_format):
        """Determine target VR per-eye resolution based on input and setting"""
        # Define per-eye resolutions (both square and wide formats)
        per_eye_resolutions = {
            # Square formats (optimized for VR headsets)
            "square-480": (480, 480),     # 480x480 per eye - quick test
            "square-720": (720, 720),     # 720x720 per eye - fast test
            "square-1k": (1080, 1080),    # 1080x1080 per eye
            "square-2k": (1536, 1536),    # 1536x1536 per eye 
            "square-3k": (1920, 1920),    # 1920x1920 per eye
            "square-4k": (2048, 2048),    # 2048x2048 per eye - industry standard
            "square-5k": (2560, 2560),    # 2560x2560 per eye - premium
            # Wide formats (preserve more of original content, good for over-under)
            "ultrawide": (3840, 2160),    # 3840x2160 per eye - ultra-wide 16:9
            "wide-2k": (2560, 1440),     # 2560x1440 per eye - 16:9 2K
            "wide-4k": (3840, 2160),     # 3840x2160 per eye - 16:9 4K  
            # Cinema formats (ultra-wide aspect ratios, recommend over-under)
            "cinema-2k": (2048, 858),    # 2048x858 per eye - 2.39:1 cinema
            "cinema-4k": (4096, 1716),   # 4096x1716 per eye - 2.39:1 4K cinema
        }
        
        if vr_resolution_setting in per_eye_resolutions:
            per_eye_width, per_eye_height = per_eye_resolutions[vr_resolution_setting]
        elif vr_resolution_setting == "auto":
            # Auto resolution logic based on source material quality and aspect ratio
            source_pixels = original_width * original_height
            source_aspect = original_width / original_height
            
            # Detect content type based on aspect ratio
            if source_aspect > 2.2:  # Ultra-wide content (cinema aspect ratios)
                print(f"Detected ultra-wide content ({source_aspect:.2f}:1) - recommending over-under format for best quality")
                if source_pixels <= 1920 * 1080:
                    per_eye_width, per_eye_height = per_eye_resolutions["cinema-2k"]
                else:
                    per_eye_width, per_eye_height = per_eye_resolutions["cinema-4k"]
            elif source_aspect > 1.6:  # Wide content (16:9, 16:10, etc.)
                print(f"Detected wide content ({source_aspect:.2f}:1) - consider over-under format to preserve more content")
                if source_pixels <= 1280 * 720:
                    per_eye_width, per_eye_height = per_eye_resolutions["wide-2k"]
                else:
                    per_eye_width, per_eye_height = per_eye_resolutions["wide-4k"]
            else:  # Standard or square content
                # Use square formats for better VR headset compatibility
                if source_pixels <= 640 * 480:      # SD content -> square-1k
                    per_eye_width, per_eye_height = per_eye_resolutions["square-1k"]
                elif source_pixels <= 1280 * 720:   # 720p content -> square-2k  
                    per_eye_width, per_eye_height = per_eye_resolutions["square-2k"]
                elif source_pixels <= 1920 * 1080:  # 1080p content -> square-3k
                    per_eye_width, per_eye_height = per_eye_resolutions["square-3k"]
                elif source_pixels <= 2560 * 1440:  # 1440p content -> square-4k
                    per_eye_width, per_eye_height = per_eye_resolutions["square-4k"]
                else:                                # 4K+ content -> square-5k
                    per_eye_width, per_eye_height = per_eye_resolutions["square-5k"]
        else:
            # Fallback to square-4k (industry standard)
            per_eye_width, per_eye_height = per_eye_resolutions["square-4k"]
        
        # Calculate final output dimensions based on VR format
        if vr_format.startswith('side_by_side'):
            # Side-by-side: two eyes horizontally
            final_width = per_eye_width * 2
            final_height = per_eye_height
        elif vr_format.startswith('over_under'):
            # Over-under: two eyes vertically  
            final_width = per_eye_width
            final_height = per_eye_height * 2
        else:
            # Default to side-by-side
            final_width = per_eye_width * 2
            final_height = per_eye_height
        
        # Add format recommendation for wide aspect ratios
        per_eye_aspect = per_eye_width / per_eye_height
        if per_eye_aspect > 1.8 and vr_format.startswith('side_by_side'):
            print(f"Note: Wide aspect ratio ({per_eye_aspect:.2f}:1) detected. Over-under format recommended for:")
            print(f"  - Better content preservation (no horizontal squashing)")
            print(f"  - More comfortable viewing on wide screens")
            print(f"  - Current: {final_width}x{final_height} side-by-side")
            print(f"  - Over-under would be: {per_eye_width}x{per_eye_height * 2}")
            
        return final_width, final_height
    
    def process_video(self, video_path, output_dir, vr_format='side_by_side', 
                     baseline=0.065, focal_length=1000, keep_intermediates=True,
                     start_time=None, end_time=None, preserve_audio=True, 
                     target_fps=60, min_resolution="1080p", super_sample="auto",
                     apply_distortion=True, fisheye_projection='stereographic', fisheye_fov=105,
                     crop_factor=1.0, vr_resolution='auto', fisheye_crop_factor=1.0, hole_fill_quality='fast',
                     processing_mode='serial'):
        """Process entire video to create 3D VR version"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing mode: {processing_mode.upper()}")
        
        # Extract frames (at original resolution first, then we'll apply super sampling during processing)
        frame_files = self.extract_frames(video_path, output_dir, start_time, end_time, target_fps, "original")
        
        # Get original frame dimensions to determine super sampling resolution
        if frame_files:
            sample_frame = cv2.imread(str(frame_files[0]))
            original_height, original_width = sample_frame.shape[:2]
            super_sample_width, super_sample_height = self.determine_super_sample_resolution(
                original_width, original_height, super_sample
            )
            
            print(f"Original resolution: {original_width}x{original_height}")
            if super_sample_width != original_width or super_sample_height != original_height:
                print(f"Super sampling to: {super_sample_width}x{super_sample_height}")
            
            # Determine VR output resolution
            vr_output_width, vr_output_height = self.determine_vr_output_resolution(
                original_width, original_height, vr_resolution, vr_format
            )
            print(f"VR output resolution: {vr_output_width}x{vr_output_height}")
            if vr_format.startswith('side_by_side'):
                per_eye_w, per_eye_h = vr_output_width//2, vr_output_height
                print(f"  -> {per_eye_w}x{per_eye_h} per eye (side-by-side)")
            elif vr_format.startswith('over_under'):
                per_eye_w, per_eye_h = vr_output_width, vr_output_height//2
                print(f"  -> {per_eye_w}x{per_eye_h} per eye (over-under)")
        else:
            raise Exception("No frames extracted")
        
        # Create numbered output directories that match the processing flow
        if keep_intermediates:
            # Step 1: Original frames (already in 1_frames/ from extraction)
            
            # Step 2: Super sampled frames (if applicable)
            if super_sample_width != original_width or super_sample_height != original_height:
                super_sampled_dir = output_path / "2_supersampled_frames"
                super_sampled_dir.mkdir(exist_ok=True)
            
            # Step 3: Depth maps
            depth_dir = output_path / "3_depth_maps"
            depth_dir.mkdir(exist_ok=True)
            
            # Step 4: Initial stereo pairs
            left_dir = output_path / "4_left_frames"
            right_dir = output_path / "4_right_frames"
            left_dir.mkdir(exist_ok=True)
            right_dir.mkdir(exist_ok=True)
            
            # Step 5: Fisheye frames (if distortion is applied)
            if apply_distortion:
                left_distorted_dir = output_path / "5_left_frames_fisheye"
                right_distorted_dir = output_path / "5_right_frames_fisheye"
                left_distorted_dir.mkdir(exist_ok=True)
                right_distorted_dir.mkdir(exist_ok=True)
            
            # Step 6: Final cropped frames
            left_final_dir = output_path / "6_left_frames_final"
            right_final_dir = output_path / "6_right_frames_final"
            left_final_dir.mkdir(exist_ok=True)
            right_final_dir.mkdir(exist_ok=True)
        
        # Step 7: Final VR frames
        vr_dir = output_path / "7_vr_frames"
        vr_dir.mkdir(exist_ok=True)
        
        print(f"Processing {len(frame_files)} frames...")
        
        # Choose processing method based on mode
        if processing_mode == 'batch':
            self._process_video_batch(
                frame_files, output_path, super_sample_width, super_sample_height,
                original_width, original_height, vr_output_width, vr_output_height,
                baseline, focal_length, hole_fill_quality, apply_distortion,
                fisheye_projection, fisheye_fov, fisheye_crop_factor, crop_factor,
                vr_format, keep_intermediates, depth_dir, left_dir, right_dir,
                left_distorted_dir if apply_distortion else None,
                right_distorted_dir if apply_distortion else None,
                left_final_dir, right_final_dir, vr_dir
            )
        else:
            self._process_video_serial(
                frame_files, output_path, super_sample_width, super_sample_height,
                original_width, original_height, vr_output_width, vr_output_height,
                baseline, focal_length, hole_fill_quality, apply_distortion,
                fisheye_projection, fisheye_fov, fisheye_crop_factor, crop_factor,
                vr_format, keep_intermediates, depth_dir, left_dir, right_dir,
                left_distorted_dir if apply_distortion else None,
                right_distorted_dir if apply_distortion else None,
                left_final_dir, right_final_dir, vr_dir
            )
        
        # Create final video
        print("Creating final video with audio...")
        self.create_output_video(vr_dir, output_path, video_path, vr_format, 
                                start_time, end_time, preserve_audio, target_fps)
        
        print(f"Processing complete. Output saved to: {output_path}")
    
    def _process_video_serial(self, frame_files, output_path, super_sample_width, super_sample_height,
                             original_width, original_height, vr_output_width, vr_output_height,
                             baseline, focal_length, hole_fill_quality, apply_distortion,
                             fisheye_projection, fisheye_fov, fisheye_crop_factor, crop_factor,
                             vr_format, keep_intermediates, depth_dir, left_dir, right_dir,
                             left_distorted_dir, right_distorted_dir, left_final_dir, right_final_dir, vr_dir):
        """Process frames serially (original behavior)"""
        
        # Initialize progress tracker
        progress_tracker = ProgressTracker(len(frame_files), 'serial')
        
        for i, frame_file in enumerate(frame_files):
            # Update progress - starting frame
            progress_tracker.update_serial(i + 1, f"Loading frame {i+1}")
            
            # Load original image
            original_image = cv2.imread(str(frame_file))
            frame_name = frame_file.stem
            
            # Note: Original frames are already saved in 1_frames/ from extraction
            
            # Apply super sampling if needed
            if super_sample_width != original_width or super_sample_height != original_height:
                progress_tracker.update_serial(i + 1, f"Super sampling frame {i+1}")
                image = self.apply_super_sampling(original_image, super_sample_width, super_sample_height)
                
                # Save super sampled frame if keeping intermediates
                if keep_intermediates:
                    cv2.imwrite(str(super_sampled_dir / f"{frame_name}.png"), image)
            else:
                image = original_image
            
            # Update progress - depth map generation
            progress_tracker.update_serial(i + 1, f"Generating depth map for frame {i+1}")
            
            # Generate depth map (on super sampled image if applicable)
            # Note: we need to save the super sampled image temporarily for depth map generation
            if super_sample_width != original_width or super_sample_height != original_height:
                temp_frame_path = output_path / f"temp_frame_{i}.png"
                cv2.imwrite(str(temp_frame_path), image)
                depth_map = self.generate_depth_map(temp_frame_path)
                temp_frame_path.unlink()  # Clean up temp file
            else:
                depth_map = self.generate_depth_map(frame_file)
            
            # Update progress - stereo pair creation
            progress_tracker.update_serial(i + 1, f"Creating stereo pair for frame {i+1}")
            
            # Create stereo pair
            left_img, right_img = self.create_stereo_pair(
                image, depth_map, baseline, focal_length, hole_fill_quality
            )
            
            # Save intermediates if keeping them
            if keep_intermediates:
                # Save depth map as grayscale
                depth_vis = (depth_map * 255).astype(np.uint8)
                cv2.imwrite(str(depth_dir / f"{frame_name}.png"), depth_vis)
                cv2.imwrite(str(left_dir / f"{frame_name}.png"), left_img)
                cv2.imwrite(str(right_dir / f"{frame_name}.png"), right_img)
            
            # Apply fisheye distortion if enabled
            if apply_distortion:
                progress_tracker.update_serial(i + 1, f"Applying fisheye distortion to frame {i+1}")
                left_distorted = self.apply_fisheye_distortion(left_img, fisheye_projection, fisheye_fov)
                right_distorted = self.apply_fisheye_distortion(right_img, fisheye_projection, fisheye_fov)
                
                # Save fisheye frames if keeping intermediates
                if keep_intermediates:
                    cv2.imwrite(str(left_distorted_dir / f"{frame_name}.png"), left_distorted)
                    cv2.imwrite(str(right_distorted_dir / f"{frame_name}.png"), right_distorted)
                
                # Apply fisheye-aware square cropping and scaling directly to target VR eye format
                progress_tracker.update_serial(i + 1, f"Cropping and scaling frame {i+1}")
                if vr_format.startswith('side_by_side'):
                    eye_width = vr_output_width // 2
                    eye_height = vr_output_height
                elif vr_format.startswith('over_under'):
                    eye_width = vr_output_width
                    eye_height = vr_output_height // 2
                else:
                    eye_width = vr_output_width // 2
                    eye_height = vr_output_height
                
                # Use fisheye-aware cropping that optimally fits square within circle
                left_final = self.apply_fisheye_square_crop(left_distorted, eye_width, eye_height, fisheye_crop_factor)
                right_final = self.apply_fisheye_square_crop(right_distorted, eye_width, eye_height, fisheye_crop_factor)
                
                # Save final square eye frames if keeping intermediates
                if keep_intermediates:
                    cv2.imwrite(str(left_final_dir / f"{frame_name}.png"), left_final)
                    cv2.imwrite(str(right_final_dir / f"{frame_name}.png"), right_final)
                    
            else:
                # Apply center cropping and VR eye scaling to undistorted frames
                progress_tracker.update_serial(i + 1, f"Cropping and scaling frame {i+1}")
                left_cropped = self.apply_center_crop(left_img, crop_factor)
                right_cropped = self.apply_center_crop(right_img, crop_factor)
                
                # Scale to VR eye format
                if vr_format.startswith('side_by_side'):
                    eye_width = vr_output_width // 2
                    eye_height = vr_output_height
                elif vr_format.startswith('over_under'):
                    eye_width = vr_output_width
                    eye_height = vr_output_height // 2
                else:
                    eye_width = vr_output_width // 2
                    eye_height = vr_output_height
                
                left_final = self._scale_to_eye_format(left_cropped, eye_width, eye_height)
                right_final = self._scale_to_eye_format(right_cropped, eye_width, eye_height)
                
                # Save final square eye frames if keeping intermediates
                if keep_intermediates:
                    cv2.imwrite(str(left_final_dir / f"{frame_name}.png"), left_final)
                    cv2.imwrite(str(right_final_dir / f"{frame_name}.png"), right_final)
            
            # Create final VR frame by combining already-scaled eye frames (no additional processing needed)
            progress_tracker.update_serial(i + 1, f"Creating final VR frame {i+1}")
            vr_frame = self.create_vr_format(left_final, right_final, vr_format, apply_crop=False, target_resolution=None)
            
            # Save final VR frame (already cropped)
            cv2.imwrite(str(vr_dir / f"{frame_name}.png"), vr_frame)
        
        # Finish progress tracking
        progress_tracker.finish("Serial frame processing complete")
    
    def _process_video_batch(self, frame_files, output_path, super_sample_width, super_sample_height,
                            original_width, original_height, vr_output_width, vr_output_height,
                            baseline, focal_length, hole_fill_quality, apply_distortion,
                            fisheye_projection, fisheye_fov, fisheye_crop_factor, crop_factor,
                            vr_format, keep_intermediates, depth_dir, left_dir, right_dir,
                            left_distorted_dir, right_distorted_dir, left_final_dir, right_final_dir, vr_dir):
        """Process frames in batch mode with parallelization"""
        
        # Initialize progress tracker for batch mode
        progress_tracker = ProgressTracker(len(frame_files), 'batch')
        
        # Calculate VR eye dimensions
        if vr_format.startswith('side_by_side'):
            eye_width = vr_output_width // 2
            eye_height = vr_output_height
        elif vr_format.startswith('over_under'):
            eye_width = vr_output_width
            eye_height = vr_output_height // 2
        else:
            eye_width = vr_output_width // 2
            eye_height = vr_output_height
        
        # Step 1: Super sampling (if needed)
        if super_sample_width != original_width or super_sample_height != original_height:
            progress_tracker.update_batch_step("Super Sampling", 0, len(frame_files))
            self._batch_super_sample(frame_files, output_path, super_sample_width, super_sample_height, 
                                   keep_intermediates, progress_tracker)
            sampled_files = self._get_super_sampled_files(frame_files, output_path)
        else:
            sampled_files = frame_files
        
        # Step 2: Depth map generation
        progress_tracker.update_batch_step("Depth Map Generation", 0, len(frame_files))
        depth_maps = self._batch_generate_depth_maps(sampled_files, depth_dir, keep_intermediates, progress_tracker)
        
        # Step 3: Stereo pair creation
        progress_tracker.update_batch_step("Stereo Pair Creation", 0, len(frame_files))
        stereo_pairs = self._batch_create_stereo_pairs(sampled_files, depth_maps, baseline, focal_length, 
                                                     hole_fill_quality, left_dir, right_dir, 
                                                     keep_intermediates, progress_tracker)
        
        # Step 4: Fisheye distortion (if enabled)
        if apply_distortion:
            progress_tracker.update_batch_step("Fisheye Distortion", 0, len(frame_files))
            distorted_pairs = self._batch_apply_fisheye(stereo_pairs, fisheye_projection, fisheye_fov,
                                                      left_distorted_dir, right_distorted_dir,
                                                      keep_intermediates, progress_tracker)
            final_pairs = self._batch_fisheye_crop_scale(distorted_pairs, eye_width, eye_height, 
                                                       fisheye_crop_factor, left_final_dir, right_final_dir,
                                                       keep_intermediates, progress_tracker)
        else:
            # Step 5: Center crop and scale
            progress_tracker.update_batch_step("Final Processing", 0, len(frame_files))
            final_pairs = self._batch_center_crop_scale(stereo_pairs, crop_factor, eye_width, eye_height,
                                                      left_final_dir, right_final_dir, 
                                                      keep_intermediates, progress_tracker)
        
        # Step 6: VR frame creation
        progress_tracker.update_batch_step("Video Creation", 0, len(frame_files))
        self._batch_create_vr_frames(final_pairs, vr_format, vr_dir, progress_tracker)
        
        # Finish progress tracking
        progress_tracker.finish("Batch processing complete")
    
    def _batch_super_sample(self, frame_files, output_path, target_width, target_height, keep_intermediates, progress_tracker):
        """Batch super sampling with parallelization"""
        if not keep_intermediates:
            return
        
        super_sampled_dir = output_path / "2_supersampled_frames"
        super_sampled_dir.mkdir(exist_ok=True)
        
        def process_frame(frame_file):
            image = cv2.imread(str(frame_file))
            if image is not None:
                upsampled = self.apply_super_sampling(image, target_width, target_height)
                output_path = super_sampled_dir / f"{frame_file.stem}.png"
                cv2.imwrite(str(output_path), upsampled)
                return output_path
            return None
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_frame, frame_file) for frame_file in frame_files]
            for i, future in enumerate(as_completed(futures)):
                progress_tracker.update_batch_progress(i + 1, len(frame_files))
    
    def _get_super_sampled_files(self, frame_files, output_path):
        """Get super sampled file paths"""
        super_sampled_dir = output_path / "2_supersampled_frames"
        if super_sampled_dir.exists():
            return [super_sampled_dir / f"{f.stem}.png" for f in frame_files]
        return frame_files
    
    def _batch_generate_depth_maps(self, frame_files, depth_dir, keep_intermediates, progress_tracker):
        """Batch depth map generation"""
        depth_maps = []
        
        for i, frame_file in enumerate(frame_files):
            depth_map = self.generate_depth_map(frame_file)
            depth_maps.append(depth_map)
            
            if keep_intermediates and depth_dir:
                depth_vis = (depth_map * 255).astype(np.uint8)
                cv2.imwrite(str(depth_dir / f"{frame_file.stem}.png"), depth_vis)
            
            progress_tracker.update_batch_progress(i + 1, len(frame_files))
        
        return depth_maps
    
    def _batch_create_stereo_pairs(self, frame_files, depth_maps, baseline, focal_length, hole_fill_quality,
                                 left_dir, right_dir, keep_intermediates, progress_tracker):
        """Batch stereo pair creation with parallelization"""
        stereo_pairs = []
        
        def create_pair(frame_file, depth_map):
            image = cv2.imread(str(frame_file))
            if image is not None:
                left_img, right_img = self.create_stereo_pair(image, depth_map, baseline, focal_length, hole_fill_quality)
                return (left_img, right_img, frame_file.stem)
            return None
        
        with ThreadPoolExecutor(max_workers=2) as executor:  # Limited workers due to GPU memory
            futures = [executor.submit(create_pair, frame_file, depth_map) 
                      for frame_file, depth_map in zip(frame_files, depth_maps)]
            
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result:
                    left_img, right_img, frame_name = result
                    stereo_pairs.append((left_img, right_img, frame_name))
                    
                    if keep_intermediates and left_dir and right_dir:
                        cv2.imwrite(str(left_dir / f"{frame_name}.png"), left_img)
                        cv2.imwrite(str(right_dir / f"{frame_name}.png"), right_img)
                
                progress_tracker.update_batch_progress(i + 1, len(frame_files))
        
        return stereo_pairs
    
    def _batch_apply_fisheye(self, stereo_pairs, fisheye_projection, fisheye_fov,
                           left_distorted_dir, right_distorted_dir, keep_intermediates, progress_tracker):
        """Batch fisheye distortion with parallelization"""
        distorted_pairs = []
        
        def apply_fisheye(left_img, right_img, frame_name):
            left_distorted = self.apply_fisheye_distortion(left_img, fisheye_projection, fisheye_fov)
            right_distorted = self.apply_fisheye_distortion(right_img, fisheye_projection, fisheye_fov)
            return (left_distorted, right_distorted, frame_name)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(apply_fisheye, left_img, right_img, frame_name) 
                      for left_img, right_img, frame_name in stereo_pairs]
            
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result:
                    left_distorted, right_distorted, frame_name = result
                    distorted_pairs.append((left_distorted, right_distorted, frame_name))
                    
                    if keep_intermediates and left_distorted_dir and right_distorted_dir:
                        cv2.imwrite(str(left_distorted_dir / f"{frame_name}.png"), left_distorted)
                        cv2.imwrite(str(right_distorted_dir / f"{frame_name}.png"), right_distorted)
                
                progress_tracker.update_batch_progress(i + 1, len(stereo_pairs))
        
        return distorted_pairs
    
    def _batch_fisheye_crop_scale(self, distorted_pairs, eye_width, eye_height, fisheye_crop_factor,
                                left_final_dir, right_final_dir, keep_intermediates, progress_tracker):
        """Batch fisheye crop and scale with parallelization"""
        final_pairs = []
        
        def crop_scale(left_img, right_img, frame_name):
            left_final = self.apply_fisheye_square_crop(left_img, eye_width, eye_height, fisheye_crop_factor)
            right_final = self.apply_fisheye_square_crop(right_img, eye_width, eye_height, fisheye_crop_factor)
            return (left_final, right_final, frame_name)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(crop_scale, left_img, right_img, frame_name) 
                      for left_img, right_img, frame_name in distorted_pairs]
            
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result:
                    left_final, right_final, frame_name = result
                    final_pairs.append((left_final, right_final, frame_name))
                    
                    if keep_intermediates and left_final_dir and right_final_dir:
                        cv2.imwrite(str(left_final_dir / f"{frame_name}.png"), left_final)
                        cv2.imwrite(str(right_final_dir / f"{frame_name}.png"), right_final)
                
                progress_tracker.update_batch_progress(i + 1, len(distorted_pairs))
        
        return final_pairs
    
    def _batch_center_crop_scale(self, stereo_pairs, crop_factor, eye_width, eye_height,
                               left_final_dir, right_final_dir, keep_intermediates, progress_tracker):
        """Batch center crop and scale with parallelization"""
        final_pairs = []
        
        def crop_scale(left_img, right_img, frame_name):
            left_cropped = self.apply_center_crop(left_img, crop_factor)
            right_cropped = self.apply_center_crop(right_img, crop_factor)
            left_final = self._scale_to_eye_format(left_cropped, eye_width, eye_height)
            right_final = self._scale_to_eye_format(right_cropped, eye_width, eye_height)
            return (left_final, right_final, frame_name)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(crop_scale, left_img, right_img, frame_name) 
                      for left_img, right_img, frame_name in stereo_pairs]
            
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result:
                    left_final, right_final, frame_name = result
                    final_pairs.append((left_final, right_final, frame_name))
                    
                    if keep_intermediates and left_final_dir and right_final_dir:
                        cv2.imwrite(str(left_final_dir / f"{frame_name}.png"), left_final)
                        cv2.imwrite(str(right_final_dir / f"{frame_name}.png"), right_final)
                
                progress_tracker.update_batch_progress(i + 1, len(stereo_pairs))
        
        return final_pairs
    
    def _batch_create_vr_frames(self, final_pairs, vr_format, vr_dir, progress_tracker):
        """Batch VR frame creation with parallelization"""
        
        def create_vr_frame(left_img, right_img, frame_name):
            vr_frame = self.create_vr_format(left_img, right_img, vr_format, apply_crop=False, target_resolution=None)
            output_path = vr_dir / f"{frame_name}.png"
            cv2.imwrite(str(output_path), vr_frame)
            return output_path
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(create_vr_frame, left_img, right_img, frame_name) 
                      for left_img, right_img, frame_name in final_pairs]
            
            for i, future in enumerate(as_completed(futures)):
                progress_tracker.update_batch_progress(i + 1, len(final_pairs))

    def create_output_video(self, vr_frames_dir, output_dir, original_video, vr_format,
                           start_time=None, end_time=None, preserve_audio=True, target_fps=60):
        """Create final VR video from processed frames with optional frame interpolation"""
        # Create output filename based on original video name
        original_name = Path(original_video).stem
        time_suffix = ""
        if start_time and end_time:
            time_suffix = f"_{start_time.replace(':', '')}-{end_time.replace(':', '')}"
        
        output_video = output_dir / f"{original_name}_3D_{vr_format}{time_suffix}.mp4"
        
        # Get original video fps to determine if interpolation is needed
        cap = cv2.VideoCapture(str(original_video))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Determine if we need frame interpolation
        need_interpolation = abs(original_fps - target_fps) > 0.1 and original_fps < target_fps
        
        # Use target fps for output video
        fps = target_fps
        
        if preserve_audio:
            # Step 1: Create video without audio
            temp_video = output_dir / f"temp_video_{vr_format}.mp4"
            
            cmd_video = [
                "ffmpeg", "-y",
                "-framerate", str(original_fps),  # Input at original fps
                "-i", str(vr_frames_dir / "frame_%06d.png")
            ]
            
            # Add frame interpolation filter if needed
            if need_interpolation:
                print(f"Applying frame interpolation: {original_fps}fps -> {target_fps}fps")
                cmd_video.extend([
                    "-vf", f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
                ])
            
            cmd_video.extend([
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",  # High quality
                str(temp_video)
            ])
            
            print(f"Creating video without audio: {temp_video}")
            result = subprocess.run(cmd_video, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error creating video: {result.stderr}")
                return
            
            # Step 2: Extract audio from original video with time range
            temp_audio = output_dir / "temp_audio.aac"
            
            cmd_audio = ["ffmpeg", "-y"]
            
            # Add start time if specified
            if start_time:
                cmd_audio.extend(["-ss", start_time])
            
            cmd_audio.extend(["-i", str(original_video)])
            
            # Add duration if both start and end time specified
            if end_time and start_time:
                start_seconds = self._time_to_seconds(start_time)
                end_seconds = self._time_to_seconds(end_time)
                duration = end_seconds - start_seconds
                if duration > 0:
                    cmd_audio.extend(["-t", str(duration)])
            elif end_time and not start_time:
                cmd_audio.extend(["-t", str(self._time_to_seconds(end_time))])
            
            cmd_audio.extend([
                "-vn",  # No video
                "-acodec", "aac",
                "-b:a", "128k",
                str(temp_audio)
            ])
            
            print("Extracting audio from original video...")
            result = subprocess.run(cmd_audio, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Warning: Could not extract audio: {result.stderr}")
                print("Proceeding without audio...")
                # If audio extraction fails, just rename temp video to final
                temp_video.rename(output_video)
            else:
                # Step 3: Combine video and audio
                cmd_combine = [
                    "ffmpeg", "-y",
                    "-i", str(temp_video),
                    "-i", str(temp_audio),
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-shortest",  # Match shortest stream duration
                    str(output_video)
                ]
                
                print(f"Combining video and audio: {output_video}")
                result = subprocess.run(cmd_combine, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error combining video and audio: {result.stderr}")
                    # If combining fails, just use video without audio
                    temp_video.rename(output_video)
                else:
                    print(f"VR video with audio created: {output_video}")
                
                # Clean up temporary files
                try:
                    temp_video.unlink()
                    temp_audio.unlink()
                except:
                    pass
        else:
            # Create video without audio (original behavior)
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(original_fps),  # Input at original fps
                "-i", str(vr_frames_dir / "frame_%06d.png")
            ]
            
            # Add frame interpolation filter if needed
            if need_interpolation:
                print(f"Applying frame interpolation: {original_fps}fps -> {target_fps}fps")
                cmd.extend([
                    "-vf", f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
                ])
            
            cmd.extend([
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",  # High quality
                str(output_video)
            ])
            
            print(f"Creating final video: {output_video}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error creating video: {result.stderr}")
            else:
                print(f"VR video created: {output_video}")

def main():
    parser = argparse.ArgumentParser(description='Convert 2D video to 3D VR format using depth estimation')
    parser.add_argument('input_video', help='Input video file path')
    parser.add_argument('-o', '--output', default='./output', help='Output directory (default: ./output)')
    parser.add_argument('-m', '--model', default='./models/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth', 
                       help='Path to Depth Anything V2 model file')
    parser.add_argument('-f', '--format', choices=['side_by_side', 'over_under'], 
                       default='side_by_side', help='VR output format (default: side_by_side)')
    parser.add_argument('--vr-resolution', choices=['square-480', 'square-720', 'square-1k', 'square-2k', 'square-3k', 'square-4k', 'square-5k', 'ultrawide', 'wide-2k', 'wide-4k', 'cinema-2k', 'cinema-4k', 'auto'], 
                       default='auto', help='VR output resolution format: square-480 (480x480 per eye - quick test), square-720 (720x720 per eye - fast test), square-1k (1080x1080 per eye), square-2k (1536x1536 per eye), square-3k (1920x1920 per eye), square-4k (2048x2048 per eye), square-5k (2560x2560 per eye), ultrawide (3840x2160 per eye), wide-2k (2560x1440 per eye), wide-4k (3840x2160 per eye), cinema-2k (2048x858 per eye), cinema-4k (4096x1716 per eye), auto (matches source) (default: auto)')
    parser.add_argument('-b', '--baseline', type=float, default=0.065, 
                       help='Stereo baseline distance in meters (default: 0.065 - average human IPD)')
    parser.add_argument('-fl', '--focal-length', type=float, default=1000,
                       help='Virtual focal length (default: 1000)')
    parser.add_argument('--no-intermediates', action='store_true',
                       help='Do not save intermediate depth maps and stereo frames')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='auto',
                       help='Device to use for inference (default: auto)')
    parser.add_argument('-s', '--start', dest='start_time', 
                       help='Start time in mm:ss or hh:mm:ss format (e.g., 01:30 or 00:01:30)')
    parser.add_argument('-e', '--end', dest='end_time',
                       help='End time in mm:ss or hh:mm:ss format (e.g., 03:45 or 00:03:45)')
    parser.add_argument('--no-audio', action='store_true',
                       help='Do not preserve audio from original video')
    parser.add_argument('--fps', type=int, default=60,
                       help='Target framerate for output video (default: 60)')
    parser.add_argument('--resolution', choices=['720p', '1080p', '4k', 'original'], 
                       default='1080p', help='Minimum resolution for output (default: 1080p)')
    parser.add_argument('--super-sample', choices=['1080p', '4k', 'auto', 'none'], default='auto',
                       help='Super sample for better SBS quality: 1080p->4K, 720p->1080p, etc. (default: auto)')
    parser.add_argument('--fisheye-distortion', action='store_true', default=True,
                       help='Apply fisheye distortion for hemispherical VR projection (default: enabled)')
    parser.add_argument('--no-fisheye-distortion', dest='fisheye_distortion', action='store_false',
                       help='Disable fisheye distortion')
    parser.add_argument('--fisheye-projection', choices=['equidistant', 'stereographic', 'equisolid', 'orthographic'], 
                       default='stereographic', help='Fisheye projection model (default: stereographic)')
    parser.add_argument('--fisheye-fov', type=float, default=105,
                       help='Fisheye field of view in degrees (75-180, default: 105)')
    parser.add_argument('--crop-factor', type=float, default=1.0,
                       help='Center crop factor for final VR frames to reduce edge artifacts (0.5-1.0, default: 1.0 - no crop)')
    parser.add_argument('--fisheye-crop-factor', type=float, default=1.0,
                       help='Fisheye square crop factor relative to circle diameter (0.8-1.5, default: 1.0 - no crop)')
    parser.add_argument('--hole-fill-quality', choices=['fast', 'advanced'], default='fast',
                       help='Hole filling quality for stereo pair generation (default: fast)')
    parser.add_argument('--processing-mode', choices=['serial', 'batch'], default='serial',
                       help='Processing mode: serial (frame-by-frame) or batch (task-by-task with parallelization) (default: serial)')
    
    args = parser.parse_args()
    
    # Check if input video exists
    if not os.path.exists(args.input_video):
        print(f"Error: Input video not found: {args.input_video}")
        sys.exit(1)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Determine device
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create stereo projector
    projector = StereoProjector(args.model, device)
    
    # Validate time arguments
    if args.start_time and args.end_time:
        start_seconds = projector._time_to_seconds(args.start_time)
        end_seconds = projector._time_to_seconds(args.end_time)
        if start_seconds >= end_seconds:
            print("Error: Start time must be before end time")
            sys.exit(1)
    
    # Process video
    projector.process_video(
        args.input_video,
        args.output,
        args.format,
        args.baseline,
        args.focal_length,
        not args.no_intermediates,
        args.start_time,
        args.end_time,
        not args.no_audio,  # preserve_audio is inverse of no_audio
        args.fps,
        args.resolution,
        args.super_sample,
        args.fisheye_distortion,
        args.fisheye_projection,
        args.fisheye_fov,
        args.crop_factor,
        getattr(args, 'vr_resolution'),
        getattr(args, 'fisheye_crop_factor'),
        getattr(args, 'hole_fill_quality'),
        getattr(args, 'processing_mode')
    )

if __name__ == "__main__":
    main()