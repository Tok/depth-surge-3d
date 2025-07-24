#!/usr/bin/env python3
"""
Depth Surge 3D Web UI
Flask application for converting 2D videos to immersive 3D VR format
"""

import os
import json
import time
import uuid
import subprocess
import platform
import argparse
import sys
import signal
from datetime import datetime
from pathlib import Path
import cv2
from threading import Thread
import shutil

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import numpy as np
import torch

from stereo_projector import StereoProjector

# Global verbose flag
VERBOSE = False

def vprint(*args, **kwargs):
    """Print only if verbose mode is enabled"""
    if VERBOSE:
        print(*args, **kwargs)

def signal_handler(signum, frame):
    """Handle termination signals by stopping active processing"""
    global current_processing
    if current_processing['active']:
        print("\nTermination signal received. Stopping active processing...")
        current_processing['stop_requested'] = True
        # Wait a moment for cleanup
        if current_processing['thread'] and current_processing['thread'].is_alive():
            current_processing['thread'].join(timeout=5)
    sys.exit(0)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'depth-surge-3d-secret'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
socketio = SocketIO(app, cors_allowed_origins="*", logger=False, engineio_logger=False)

# Global variables for processing state
current_processing = {
    'active': False,
    'progress': 0,
    'stage': '',
    'total_frames': 0,
    'current_frame': 0,
    'session_id': None,
    'stop_requested': False,
    'thread': None
}

def ensure_directories():
    """Ensure upload and output directories exist"""
    Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
    Path(app.config['OUTPUT_FOLDER']).mkdir(exist_ok=True)

def get_video_info(video_path):
    """Extract video information using OpenCV"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    # Calculate aspect ratio
    aspect_ratio = width / height if height > 0 else 1.0
    
    return {
        'width': width,
        'height': height,
        'fps': round(fps, 2),
        'duration': round(duration, 2),
        'frame_count': frame_count,
        'aspect_ratio': round(aspect_ratio, 3),
        'aspect_ratio_text': f"{width}:{height}"
    }


def get_system_info():
    """Get system information including GPU details"""
    info = {
        'gpu_device': 'CPU',
        'vram_usage': 'N/A',
        'device_mode': 'CPU',
        'cuda_available': False
    }
    
    try:
        if torch.cuda.is_available():
            info['cuda_available'] = True
            info['gpu_device'] = torch.cuda.get_device_name(0)
            info['device_mode'] = 'GPU'
            
            # Get VRAM usage
            try:
                allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
                reserved = torch.cuda.memory_reserved() / (1024**3)   # Convert to GB
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
                info['vram_usage'] = f"{allocated:.1f}GB / {total_memory:.1f}GB"
            except Exception as vram_error:
                vprint(f"Error getting VRAM details: {vram_error}")
                info['vram_usage'] = 'N/A'
    except Exception as e:
        vprint(f"Error getting GPU info: {e}")
    
    return info

class ProgressCallback:
    """Callback class to track processing progress"""
    def __init__(self, session_id, total_frames):
        self.session_id = session_id
        self.total_frames = total_frames
        self.current_frame = 0
        self.last_update_time = 0
        self.current_phase = "extraction"  # extraction, processing, video
        
    def update_progress(self, stage, frame_num=None, phase=None):
        global current_processing
        import time
        
        # Check if stop has been requested
        if current_processing.get('stop_requested', False):
            raise InterruptedError("Processing stopped by user request")
        
        # Throttle updates to avoid threading issues
        current_time = time.time()
        if current_time - self.last_update_time < 0.1:  # Limit to 10 updates per second  
            return
        self.last_update_time = current_time
        
        # Update phase if provided
        if phase:
            self.current_phase = phase
        
        if frame_num is not None:
            self.current_frame = frame_num
            current_processing['current_frame'] = frame_num
            
        current_processing['stage'] = stage
        
        # Calculate overall progress based on phase
        if self.current_phase == "extraction":
            # Extraction phase: 0-20%
            progress = (self.current_frame / self.total_frames * 20) if self.total_frames > 0 else 0
        elif self.current_phase == "processing":
            # Processing phase: 20-85% 
            frame_progress = (self.current_frame / self.total_frames * 65) if self.total_frames > 0 else 0
            progress = 20 + frame_progress
        elif self.current_phase == "video":
            # Video creation phase: 85-100%
            progress = 85 + 15  # Set to 100% for video phase
        else:
            progress = 0
            
        current_processing['progress'] = round(progress, 1)
        current_processing['phase'] = self.current_phase
        
        # Emit progress update
        progress_data = {
            'progress': current_processing['progress'],
            'stage': stage,
            'current_frame': self.current_frame,
            'total_frames': self.total_frames,
            'phase': self.current_phase
        }
        # Always print progress to console for web UI
        print(f"Progress: {progress_data['progress']:.1f}% - {stage} - Frame {self.current_frame}/{self.total_frames} - Phase: {self.current_phase}")
        
        try:
            socketio.emit('progress_update', progress_data, room=self.session_id)
        except Exception as e:
            vprint(f"Error emitting progress: {e}")
            # Don't let SocketIO errors stop processing - continue silently

def process_video_async(session_id, video_path, settings, output_dir):
    """Process video in background thread"""
    global current_processing
    
    try:
        current_processing['active'] = True
        current_processing['session_id'] = session_id
        current_processing['stop_requested'] = False
        
        # Initialize projector
        model_path = settings.get('model_path', './models/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth')
        device = settings.get('device', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        projector = StereoProjector(model_path, device)
        
        # Get video info for progress tracking
        video_info = get_video_info(video_path)
        if not video_info:
            raise Exception("Could not read video file")
            
        # Calculate expected frame count based on time range and ORIGINAL fps (since we extract at original fps)
        start_time = settings.get('start_time')
        end_time = settings.get('end_time')
        target_fps = settings.get('target_fps')
        
        # Always use original FPS for frame count calculation (interpolation happens at the end)
        original_fps = video_info['fps']
        
        if start_time and end_time:
            duration = projector._time_to_seconds(end_time) - projector._time_to_seconds(start_time)
        else:
            duration = video_info['duration']
            
        expected_frames = int(duration * original_fps)
        
        callback = ProgressCallback(session_id, expected_frames)
        
        # Create custom projector with progress callbacks
        projector_with_progress = StereoProjectorWithProgress(projector, callback)
        
        # Process video
        projector_with_progress.process_video(
            video_path=video_path,
            output_dir=output_dir,
            vr_format=settings.get('vr_format', 'side_by_side'),
            baseline=settings.get('baseline', 0.1),
            focal_length=settings.get('focal_length', 1000),
            keep_intermediates=settings.get('keep_intermediates', True),
            start_time=start_time,
            end_time=end_time,
            preserve_audio=settings.get('preserve_audio', True),
            target_fps=target_fps,
            min_resolution=settings.get('resolution', '1080p'),
            super_sample=settings.get('super_sample', 'auto'),
            apply_distortion=settings.get('apply_distortion', True),
            fisheye_projection=settings.get('fisheye_projection', 'equidistant'),
            fisheye_fov=settings.get('fisheye_fov', 180),
            crop_factor=settings.get('crop_factor', 0.7),
            vr_resolution=settings.get('vr_resolution', 'auto'),
            fisheye_crop_factor=settings.get('fisheye_crop_factor', 1.25),
            hole_fill_quality=settings.get('hole_fill_quality', 'advanced')
        )
        
        # Processing complete
        try:
            socketio.emit('processing_complete', {
                'success': True,
                'output_dir': str(output_dir),
                'message': 'Video processing completed successfully!'
            }, room=session_id)
        except Exception as e:
            vprint(f"Error emitting completion: {e}")
        
    except InterruptedError as e:
        # Handle user-requested stop
        try:
            socketio.emit('processing_stopped', {
                'success': True,
                'message': str(e)
            }, room=session_id)
        except Exception as emit_error:
            vprint(f"Error emitting stop: {emit_error}")
        
    except Exception as e:
        try:
            socketio.emit('processing_error', {
                'success': False,
                'error': str(e)
            }, room=session_id)
        except Exception as emit_error:
            vprint(f"Error emitting error: {emit_error}")
        print(f"Processing error: {e}")  # Always print errors
        
    finally:
        current_processing['active'] = False
        current_processing['session_id'] = None
        current_processing['stop_requested'] = False
        current_processing['thread'] = None

class StereoProjectorWithProgress:
    """Wrapper around StereoProjector to add progress callbacks"""
    
    def __init__(self, projector, callback):
        self.projector = projector
        self.callback = callback
        
    def process_video(self, **kwargs):
        """Process video with progress tracking"""
        # Update callback for frame extraction
        self.callback.update_progress("Extracting and enhancing frames...")
        
        # Get original video fps
        cap = cv2.VideoCapture(str(kwargs['video_path']))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Handle target fps for final video
        target_fps = kwargs.get('target_fps')
        if target_fps is None or target_fps == 'original':
            target_fps = original_fps
        
        # Extract frames at original fps (interpolation happens during final video creation)
        frame_files = self.projector.extract_frames(
            kwargs['video_path'], 
            kwargs['output_dir'], 
            kwargs.get('start_time'), 
            kwargs.get('end_time'),
            target_fps,  # Pass target_fps but extraction uses original_fps internally
            "original"
        )
        
        # Update total frames based on actual extraction
        self.callback.total_frames = len(frame_files)
        current_processing['total_frames'] = len(frame_files)
        
        # Get original frame dimensions to determine super sampling resolution
        if frame_files:
            sample_frame = cv2.imread(str(frame_files[0]))
            original_height, original_width = sample_frame.shape[:2]
            super_sample_width, super_sample_height = self.projector.determine_super_sample_resolution(
                original_width, original_height, kwargs.get('super_sample', 'auto')
            )
            
            # Determine VR output resolution
            vr_output_width, vr_output_height = self.projector.determine_vr_output_resolution(
                original_width, original_height, kwargs.get('vr_resolution', 'auto'), kwargs.get('vr_format', 'side_by_side')
            )
        else:
            raise Exception("No frames extracted")
        
        # Frame extraction is complete - show 100% for extraction phase
        self.callback.update_progress(f"Frame extraction complete - {len(frame_files)} frames extracted", len(frame_files), phase="extraction")
        
        # Switch to processing phase
        self.callback.update_progress(f"Frame extraction complete - {len(frame_files)} frames extracted", len(frame_files), phase="processing")
        self.callback.current_frame = 0
        current_processing['current_frame'] = 0
        
        # Create numbered output directories that match the processing flow
        output_path = Path(kwargs['output_dir'])
        if kwargs.get('keep_intermediates', True):
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
            if kwargs.get('apply_distortion', True):
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
        
        # Start processing phase
        self.callback.update_progress("Starting frame processing...", 0)
        
        # Process each frame (with resume support)
        for i, frame_file in enumerate(frame_files):
            try:
                # Check for stop request at the beginning of each frame
                if current_processing.get('stop_requested', False):
                    raise InterruptedError("Processing stopped by user request")
                
                frame_name = frame_file.stem
                
                # Check if this frame is already processed (resume support)
                vr_frame_path = vr_dir / f"{frame_name}.png"
                if vr_frame_path.exists():
                    # Frame already processed, skip it
                    self.callback.update_progress(f"Skipping frame {i+1}/{len(frame_files)} - Already processed", i+1)
                    continue
                    
                # Update progress at start of each frame
                self.callback.update_progress(f"Processing frame {i+1}/{len(frame_files)} - Loading...", i+1)
                
                original_image = cv2.imread(str(frame_file))
                if original_image is None:
                    vprint(f"Warning: Could not load frame {frame_file}, skipping...")
                    continue
                
                # Note: Original frames are already saved in 1_frames/ from extraction
                
                # Apply super sampling if needed
                if super_sample_width != original_width or super_sample_height != original_height:
                    self.callback.update_progress(f"Processing frame {i+1}/{len(frame_files)} - Super sampling...", i+1)
                    image = self.projector.apply_super_sampling(original_image, super_sample_width, super_sample_height)
                    
                    # Save super sampled frame if keeping intermediates
                    if kwargs.get('keep_intermediates', True):
                        cv2.imwrite(str(super_sampled_dir / f"{frame_name}.png"), image)
                else:
                    image = original_image
                
                # Update progress - depth map generation
                self.callback.update_progress(f"Processing frame {i+1}/{len(frame_files)} - Generating depth map...", i+1)
                
                # Generate depth map (on super sampled image if applicable)
                if super_sample_width != original_width or super_sample_height != original_height:
                    temp_frame_path = output_path / f"temp_frame_{i}.png"
                    cv2.imwrite(str(temp_frame_path), image)
                    depth_map = self.projector.generate_depth_map(temp_frame_path)
                    temp_frame_path.unlink()  # Clean up temp file
                else:
                    depth_map = self.projector.generate_depth_map(frame_file)
                
                # Update progress - stereo pair creation
                self.callback.update_progress(f"Processing frame {i+1}/{len(frame_files)} - Creating stereo pair...", i+1)
                
                # Create stereo pair
                left_img, right_img = self.projector.create_stereo_pair(
                    image, depth_map, kwargs.get('baseline', 0.1), kwargs.get('focal_length', 1000),
                    hole_fill_quality=kwargs.get('hole_fill_quality', 'advanced')
                )
                
                # Save intermediates if keeping them
                if kwargs.get('keep_intermediates', True):
                    # Save depth map as grayscale
                    depth_vis = (depth_map * 255).astype(np.uint8)
                    cv2.imwrite(str(depth_dir / f"{frame_name}.png"), depth_vis)
                    cv2.imwrite(str(left_dir / f"{frame_name}.png"), left_img)
                    cv2.imwrite(str(right_dir / f"{frame_name}.png"), right_img)
                
                # Apply fisheye distortion if enabled
                if kwargs.get('apply_distortion', True):
                    self.callback.update_progress(f"Processing frame {i+1}/{len(frame_files)} - Fisheye projection...", i+1)
                    left_distorted = self.projector.apply_fisheye_distortion(
                        left_img, kwargs.get('fisheye_projection', 'equidistant'), kwargs.get('fisheye_fov', 180)
                    )
                    right_distorted = self.projector.apply_fisheye_distortion(
                        right_img, kwargs.get('fisheye_projection', 'equidistant'), kwargs.get('fisheye_fov', 180)
                    )
                    
                    # Save fisheye frames if keeping intermediates
                    if kwargs.get('keep_intermediates', True):
                        cv2.imwrite(str(left_distorted_dir / f"{frame_name}.png"), left_distorted)
                        cv2.imwrite(str(right_distorted_dir / f"{frame_name}.png"), right_distorted)
                    
                    # Apply fisheye-aware square cropping and scaling directly to target VR eye format
                    self.callback.update_progress(f"Processing frame {i+1}/{len(frame_files)} - Fisheye square crop & scale...", i+1)
                    vr_format = kwargs.get('vr_format', 'side_by_side')
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
                    fisheye_crop_factor = kwargs.get('fisheye_crop_factor', 1.25)
                    left_final = self.projector.apply_fisheye_square_crop(left_distorted, eye_width, eye_height, fisheye_crop_factor)
                    right_final = self.projector.apply_fisheye_square_crop(right_distorted, eye_width, eye_height, fisheye_crop_factor)
                    
                    # Save final square eye frames if keeping intermediates
                    if kwargs.get('keep_intermediates', True):
                        cv2.imwrite(str(left_final_dir / f"{frame_name}.png"), left_final)
                        cv2.imwrite(str(right_final_dir / f"{frame_name}.png"), right_final)
                        
                else:
                    # Apply center cropping and VR eye scaling to undistorted frames
                    self.callback.update_progress(f"Processing frame {i+1}/{len(frame_files)} - Cropping & scaling...", i+1)
                    left_cropped = self.projector.apply_center_crop(left_img, kwargs.get('crop_factor', 0.7))
                    right_cropped = self.projector.apply_center_crop(right_img, kwargs.get('crop_factor', 0.7))
                    
                    # Scale to VR eye format
                    vr_format = kwargs.get('vr_format', 'side_by_side')
                    if vr_format.startswith('side_by_side'):
                        eye_width = vr_output_width // 2
                        eye_height = vr_output_height
                    elif vr_format.startswith('over_under'):
                        eye_width = vr_output_width
                        eye_height = vr_output_height // 2
                    else:
                        eye_width = vr_output_width // 2
                        eye_height = vr_output_height
                    
                    left_final = self.projector._scale_to_eye_format(left_cropped, eye_width, eye_height)
                    right_final = self.projector._scale_to_eye_format(right_cropped, eye_width, eye_height)
                    
                    # Save final square eye frames if keeping intermediates
                    if kwargs.get('keep_intermediates', True):
                        cv2.imwrite(str(left_final_dir / f"{frame_name}.png"), left_final)
                        cv2.imwrite(str(right_final_dir / f"{frame_name}.png"), right_final)
                
                # Create final VR frame by combining already-scaled eye frames (no additional processing needed)
                self.callback.update_progress(f"Processing frame {i+1}/{len(frame_files)} - Creating VR frame...", i+1)
                vr_frame = self.projector.create_vr_format(left_final, right_final, kwargs.get('vr_format', 'side_by_side'), apply_crop=False, target_resolution=None)
            
                # Final frame completion update
                self.callback.update_progress(f"Completed frame {i+1}/{len(frame_files)}", i+1)
                
                # Save final VR frame (already cropped)
                cv2.imwrite(str(vr_dir / f"{frame_name}.png"), vr_frame)
                
            except Exception as frame_error:
                vprint(f"Error processing frame {i+1}: {frame_error}")
                # Continue with next frame instead of stopping entire processing
                continue
        
        # Create final video
        self.callback.update_progress("Creating final video with audio...", phase="video")
        self.projector.create_output_video(
            vr_dir, output_path, kwargs['video_path'], kwargs.get('vr_format', 'side_by_side'),
            kwargs.get('start_time'), kwargs.get('end_time'), kwargs.get('preserve_audio', True),
            target_fps
        )

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file
    filename = f"{int(time.time())}_{file.filename}"
    filepath = Path(app.config['UPLOAD_FOLDER']) / filename
    file.save(filepath)
    
    # Get video information
    video_info = get_video_info(filepath)
    if not video_info:
        return jsonify({'error': 'Invalid video file'}), 400
    
    return jsonify({
        'success': True,
        'filename': filename,
        'video_info': video_info
    })

@app.route('/process', methods=['POST'])
def start_processing():
    """Start video processing"""
    global current_processing
    
    if current_processing['active']:
        return jsonify({'error': 'Processing already in progress'}), 400
    
    data = request.json
    filename = data.get('filename')
    settings = data.get('settings', {})
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    video_path = Path(app.config['UPLOAD_FOLDER']) / filename
    if not video_path.exists():
        return jsonify({'error': 'Video file not found'}), 404
    
    # Create timestamped output directory
    video_name = Path(filename).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(app.config['OUTPUT_FOLDER']) / f"{video_name}_{timestamp}"
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    # Start processing in background
    thread = Thread(target=process_video_async, args=(session_id, video_path, settings, output_dir))
    thread.daemon = True
    current_processing['thread'] = thread
    thread.start()
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'output_dir': str(output_dir)
    })

@app.route('/stop', methods=['POST'])
def stop_processing():
    """Stop current processing"""
    global current_processing
    
    data = request.json
    session_id = data.get('session_id')
    
    if not current_processing['active']:
        return jsonify({'success': False, 'error': 'No processing currently active'})
    
    if session_id != current_processing['session_id']:
        return jsonify({'success': False, 'error': 'Invalid session ID'})
    
    # Request stop
    current_processing['stop_requested'] = True
    
    return jsonify({'success': True, 'message': 'Stop request sent'})

@app.route('/resume', methods=['POST'])
def resume_processing():
    """Resume processing from a previous interrupted batch"""
    global current_processing
    
    if current_processing['active']:
        return jsonify({'error': 'Processing already in progress'}), 400
    
    data = request.json
    output_dir = data.get('output_dir')
    
    if not output_dir:
        return jsonify({'error': 'No output directory provided'}), 400
    
    output_path = Path(output_dir)
    if not output_path.exists():
        return jsonify({'error': 'Output directory does not exist'}), 404
    
    # Look for original video in the directory or parent directories
    original_video = None
    
    # Check for video files in parent directory (uploads)
    uploads_dir = Path(app.config['UPLOAD_FOLDER'])
    if uploads_dir.exists():
        for ext in ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm']:
            for video_file in uploads_dir.glob(f'*.{ext}'):
                if video_file.stem in output_path.name:
                    original_video = video_file
                    break
            if original_video:
                break
    
    if not original_video:
        return jsonify({'error': 'Could not find original video file for resuming'}), 404
    
    # Try to detect settings from existing files/directories
    settings = detect_resume_settings(output_path)
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    # Start processing in background
    thread = Thread(target=process_video_async, args=(session_id, original_video, settings, output_path))
    thread.daemon = True
    current_processing['thread'] = thread
    thread.start()
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'output_dir': str(output_path)
    })

def detect_resume_settings(output_path):
    """Detect processing settings from existing output directory"""
    settings = {
        'vr_format': 'side_by_side',
        'baseline': 0.065,
        'focal_length': 1000,
        'preserve_audio': True,
        'keep_intermediates': True,
        'device': 'auto',
        'super_sample': 'auto',
        'apply_distortion': True,
        'fisheye_projection': 'equidistant',
        'fisheye_fov': 180,
        'crop_factor': 0.7,
        'vr_resolution': 'auto',
        'fisheye_crop_factor': 1.25
    }
    
    # Try to detect VR format from directory structure
    if (output_path / 'vr_frames').exists():
        # Check if there are side-by-side or over-under frames
        vr_files = list((output_path / 'vr_frames').glob('*.png'))
        if vr_files:
            sample_frame = cv2.imread(str(vr_files[0]))
            if sample_frame is not None:
                height, width = sample_frame.shape[:2]
                if width > height * 1.5:  # Likely side-by-side
                    settings['vr_format'] = 'side_by_side'
                elif height > width * 1.5:  # Likely over-under
                    settings['vr_format'] = 'over_under'
    
    return settings

@app.route('/status')
def get_status():
    """Get current processing status"""
    return jsonify(current_processing)

@app.route('/system_info')
def get_system_info_endpoint():
    """Get system information"""
    return jsonify(get_system_info())

@app.route('/open_directory', methods=['POST'])
def open_directory():
    """Open directory in file explorer"""
    data = request.json
    directory_path = data.get('path')
    
    if not directory_path or not os.path.exists(directory_path):
        return jsonify({'success': False, 'error': 'Invalid directory path'})
    
    try:
        system = platform.system()
        if system == 'Windows':
            os.startfile(directory_path)
        elif system == 'Darwin':  # macOS
            subprocess.run(['open', directory_path])
        else:  # Linux and others
            subprocess.run(['xdg-open', directory_path])
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def analyze_batch_directory(batch_path):
    """Analyze batch directory to determine available processing stages and settings"""
    analysis = {
        'frame_count': 0,
        'vr_format': 'unknown',
        'resolution': 'unknown',
        'highest_stage': 'none',
        'has_audio': False,
        'settings_summary': 'unknown'
    }
    
    # Check for different processing stages
    stages = {
        '7_vr_frames': 'Final VR frames',
        '6_left_frames_final': 'Final stereo pairs',
        '6_right_frames_final': 'Final stereo pairs',
        '5_left_frames_fisheye': 'Fisheye frames',
        '5_right_frames_fisheye': 'Fisheye frames',
        '4_left_frames': 'Basic stereo pairs',
        '4_right_frames': 'Basic stereo pairs',
        '3_depth_maps': 'Depth maps',
        '2_supersampled_frames': 'Super sampled frames',
        '1_frames': 'Original frames'
    }
    
    highest_stage_num = 0
    for stage_dir, stage_name in stages.items():
        stage_path = batch_path / stage_dir
        if stage_path.exists():
            frame_count = len(list(stage_path.glob('*.png'))) + len(list(stage_path.glob('*.jpg')))
            if frame_count > 0:
                stage_num = int(stage_dir.split('_')[0])
                if stage_num > highest_stage_num:
                    highest_stage_num = stage_num
                    analysis['highest_stage'] = stage_name
                    analysis['frame_count'] = frame_count
    
    # Detect VR format and resolution from highest stage
    if highest_stage_num >= 6:  # Final frames available
        sample_frame_dirs = [d for d in ['6_left_frames_final', '6_right_frames_final', '7_vr_frames'] if (batch_path / d).exists()]
        for frame_dir in sample_frame_dirs:
            frame_path = batch_path / frame_dir
            sample_frames = list(frame_path.glob('*.png'))
            if sample_frames:
                try:
                    sample_img = cv2.imread(str(sample_frames[0]))
                    if sample_img is not None:
                        h, w = sample_img.shape[:2]
                        analysis['resolution'] = f"{w}x{h}"
                        
                        # Detect format based on aspect ratio
                        if frame_dir == '7_vr_frames':
                            if w > h * 1.5:
                                analysis['vr_format'] = 'side_by_side'
                            elif h > w * 1.5:
                                analysis['vr_format'] = 'over_under'
                            else:
                                analysis['vr_format'] = 'square'
                        break
                except Exception:
                    pass
    
    # Check for original video files for audio
    for video_ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        if list(batch_path.parent.parent.glob(f'uploads/{video_ext}')):  # Check uploads dir
            analysis['has_audio'] = True
            break
    
    # Generate settings summary
    if analysis['vr_format'] != 'unknown' and analysis['resolution'] != 'unknown':
        analysis['settings_summary'] = f"{analysis['vr_format']}, {analysis['resolution']}"
    
    return analysis

def create_video_from_batch(batch_path, settings):
    """Create video from batch frames using FFmpeg"""
    frame_source = settings.get('frame_source', 'auto')
    quality = settings.get('quality', 'medium')
    fps = settings.get('fps', 'original')
    include_audio = settings.get('include_audio', False)
    output_filename = settings.get('output_filename')
    
    # Determine frame directory to use
    if frame_source == 'auto':
        # Auto-detect highest available stage
        stages = ['7_vr_frames', '6_left_frames_final', '6_right_frames_final']
        frame_dir = None
        for stage in stages:
            stage_path = batch_path / stage
            if stage_path.exists() and list(stage_path.glob('*.png')):
                frame_dir = stage_path
                break
    else:
        # Use specified stage
        stage_mapping = {
            'vr_frames': '7_vr_frames',
            'left_right_final': '6_left_frames_final',
            'left_right_fisheye': '5_left_frames_fisheye',
            'left_right_basic': '4_left_frames'
        }
        stage_name = stage_mapping.get(frame_source, '7_vr_frames')
        frame_dir = batch_path / stage_name
    
    if not frame_dir or not frame_dir.exists():
        raise Exception(f"Frame directory not found: {frame_dir}")
    
    frame_files = sorted(frame_dir.glob('*.png'))
    if not frame_files:
        raise Exception(f"No frames found in {frame_dir}")
    
    # Determine output filename
    if not output_filename:
        batch_name = batch_path.name
        timestamp = datetime.now().strftime("%H%M%S")
        output_filename = f"{batch_name}_stitched_{timestamp}.mp4"
    
    output_path = batch_path / output_filename
    
    # Build FFmpeg command
    cmd = ['ffmpeg', '-y']  # -y to overwrite output
    
    # Input frames
    cmd.extend(['-framerate', str(fps) if fps != 'original' else '30'])
    cmd.extend(['-i', str(frame_dir / 'frame_%06d.png')])
    
    # Quality settings
    quality_map = {
        'high': '18',
        'medium': '23',
        'fast': '28'
    }
    crf = quality_map.get(quality, '23')
    
    cmd.extend(['-c:v', 'libx264', '-crf', crf, '-preset', 'medium'])
    cmd.extend(['-pix_fmt', 'yuv420p'])  # For compatibility
    
    # Add audio if requested and available
    if include_audio:
        # Look for original video file
        uploads_dir = Path('uploads')
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files.extend(uploads_dir.glob(ext))
        
        if video_files:
            # Use the most recent video file
            latest_video = max(video_files, key=lambda x: x.stat().st_mtime)
            cmd.extend(['-i', str(latest_video)])
            cmd.extend(['-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0'])
    
    cmd.append(str(output_path))
    
    # Execute FFmpeg
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise Exception(f"FFmpeg error: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise Exception("Video creation timed out")
    
    return output_path

@app.route('/analyze_batch', methods=['POST'])
def analyze_batch():
    """Analyze a batch directory for video stitching"""
    data = request.json
    batch_dir = data.get('batch_dir')
    
    if not batch_dir:
        return jsonify({'success': False, 'error': 'No batch directory provided'})
    
    batch_path = Path(batch_dir)
    if not batch_path.exists():
        return jsonify({'success': False, 'error': 'Batch directory does not exist'})
    
    try:
        analysis = analyze_batch_directory(batch_path)
        return jsonify({'success': True, 'analysis': analysis})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stitch_video', methods=['POST'])
def stitch_video():
    """Create video from batch frames"""
    data = request.json
    batch_dir = data.get('batch_dir')
    
    if not batch_dir:
        return jsonify({'success': False, 'error': 'No batch directory provided'})
    
    batch_path = Path(batch_dir)
    if not batch_path.exists():
        return jsonify({'success': False, 'error': 'Batch directory does not exist'})
    
    try:
        output_path = create_video_from_batch(batch_path, data)
        return jsonify({'success': True, 'output_path': str(output_path)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@socketio.on('connect')
def handle_connect():
    vprint(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    global current_processing
    vprint(f"Client disconnected: {request.sid}")
    
    # Only stop processing if there are no other connected clients
    # Don't immediately stop on disconnect since users might refresh the page
    # The processing will continue and can be rejoined

@socketio.on('join_session')
def handle_join_session(data):
    session_id = data.get('session_id')
    if session_id:
        # Join the session room for progress updates
        from flask_socketio import join_room
        join_room(session_id)
        vprint(f"Client {request.sid} joined session {session_id}")
        
        # Send initial status to joined client
        try:
            socketio.emit('progress_update', {
                'progress': current_processing.get('progress', 0),
                'stage': current_processing.get('stage', 'Initializing...'),
                'current_frame': current_processing.get('current_frame', 0),
                'total_frames': current_processing.get('total_frames', 0),
                'phase': current_processing.get('phase', 'extraction')
            }, room=session_id)
        except Exception as e:
            vprint(f"Error emitting initial progress: {e}")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Depth Surge 3D Web UI')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Enable verbose logging (shows GET/SET requests and client details)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run the server on (default: 5000)')
    parser.add_argument('--host', default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')
    args = parser.parse_args()
    
    # Set global verbose flag
    VERBOSE = args.verbose
    
    ensure_directories()
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    import torch  # Import here to avoid issues during startup
    print("Starting Depth Surge 3D Web UI...")
    print(f"Navigate to http://localhost:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=args.verbose)