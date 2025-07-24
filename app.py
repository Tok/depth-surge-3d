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

# Import our constants
from src.depth_surge_3d.core.constants import INTERMEDIATE_DIRS

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path for package imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from depth_surge_3d.core.stereo_projector import create_stereo_projector

# Import utility functions - these are part of our project and must be available
from depth_surge_3d.utils.video_processing import process_video_serial, process_video_batch
from depth_surge_3d.utils.batch_analysis import analyze_batch_directory, create_video_from_batch
from depth_surge_3d.processing.video_processor import VideoProcessor

# Global flags and state
VERBOSE = False
SHUTDOWN_FLAG = False
ACTIVE_PROCESSES = set()

def vprint(*args, **kwargs):
    """Print only if verbose mode is enabled"""
    if VERBOSE:
        print(*args, **kwargs)

def cleanup_processes():
    """Clean up any active processing threads or subprocesses"""
    global ACTIVE_PROCESSES, SHUTDOWN_FLAG
    
    SHUTDOWN_FLAG = True
    vprint("🛑 Cleaning up active processes...")
    
    # Kill any ffmpeg processes related to this app
    try:
        subprocess.run(['pkill', '-f', 'ffmpeg.*depth-surge'], check=False, capture_output=True)
    except:
        pass
    
    # Clean up any tracked processes
    for proc in list(ACTIVE_PROCESSES):
        try:
            if hasattr(proc, 'terminate'):
                proc.terminate()
            elif hasattr(proc, 'kill'):
                proc.kill()
        except:
            pass
    
    ACTIVE_PROCESSES.clear()
    vprint("✅ Process cleanup completed")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global current_processing
    print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
    
    # Stop any active processing
    if current_processing['active']:
        print("   Stopping active video processing...")
        current_processing['stop_requested'] = True
        # Wait a moment for cleanup
        if current_processing['thread'] and current_processing['thread'].is_alive():
            current_processing['thread'].join(timeout=5)
    
    # Clean up all processes
    cleanup_processes()
    sys.exit(0)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'depth-surge-3d-secret'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
socketio = SocketIO(app, cors_allowed_origins="*", logger=False, engineio_logger=False)

@app.teardown_appcontext
def cleanup_on_teardown(error):
    """Clean up processes when Flask shuts down"""
    if error:
        vprint(f"App teardown due to error: {error}")
    cleanup_processes()

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
    """Enhanced callback class to track processing progress for both serial and batch modes"""
    def __init__(self, session_id, total_frames, processing_mode='serial'):
        self.session_id = session_id
        self.total_frames = total_frames
        self.processing_mode = processing_mode
        self.current_frame = 0
        self.last_update_time = 0
        self.current_phase = "extraction"  # extraction, processing, video
        
        # Batch mode specific
        if processing_mode == 'batch':
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
        
    def update_progress(self, stage, frame_num=None, phase=None, step_name=None, step_progress=None, step_total=None):
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
        
        # Calculate progress based on processing mode
        if self.processing_mode == 'batch':
            # Batch mode: step-based progress
            if step_name and step_name in self.steps:
                self.current_step_index = self.steps.index(step_name)
            if step_progress is not None:
                self.step_progress = step_progress
            if step_total is not None:
                self.step_total = step_total
            
            # Overall progress based on steps
            step_progress_ratio = (self.step_progress / max(self.step_total, 1)) if self.step_total > 0 else 0
            overall_progress = ((self.current_step_index + step_progress_ratio) / len(self.steps)) * 100
            progress = round(overall_progress, 1)
            
            # Update current processing for batch mode
            current_processing['step_name'] = step_name or self.steps[self.current_step_index] if self.current_step_index < len(self.steps) else "Processing"
            current_processing['step_progress'] = self.step_progress
            current_processing['step_total'] = self.step_total
            current_processing['step_index'] = self.current_step_index
            
        else:
            # Serial mode: frame-based progress (original behavior)
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
        current_processing['processing_mode'] = self.processing_mode
        
        # Emit progress update
        progress_data = {
            'progress': current_processing['progress'],
            'stage': stage,
            'current_frame': self.current_frame,
            'total_frames': self.total_frames,
            'phase': self.current_phase,
            'processing_mode': self.processing_mode
        }
        
        # Add batch-specific data
        if self.processing_mode == 'batch':
            progress_data.update({
                'step_name': current_processing.get('step_name', ''),
                'step_progress': self.step_progress,
                'step_total': self.step_total,
                'step_index': self.current_step_index,
                'total_steps': len(self.steps)
            })
        
        # Console output
        if self.processing_mode == 'batch':
            step_name = current_processing.get('step_name', 'Processing')
            step_percent = (self.step_progress / max(self.step_total, 1)) * 100 if self.step_total > 0 else 0
            print(f"[BATCH] Progress: {progress_data['progress']:.1f}% - Step {self.current_step_index + 1}/{len(self.steps)}: {step_name} ({step_percent:.1f}%)")
        else:
            print(f"[SERIAL] Progress: {progress_data['progress']:.1f}% - {stage} - Frame {self.current_frame}/{self.total_frames} - Phase: {self.current_phase}")
        
        try:
            socketio.emit('progress_update', progress_data, room=self.session_id)
        except Exception as e:
            vprint(f"Error emitting progress: {e}")
            # Don't let SocketIO errors stop processing - continue silently
    
    def finish(self, message: str = "Processing complete"):
        """Finish progress tracking (compatibility with ProgressTracker interface)."""
        print(f"[{self.processing_mode.upper()}] {message}")
        try:
            socketio.emit('processing_complete', {
                'success': True,
                'message': message
            }, room=self.session_id)
        except Exception as e:
            vprint(f"Error emitting completion: {e}")

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
            
        projector = create_stereo_projector(model_path, device)
        
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
        
        processing_mode = settings.get('processing_mode', 'serial')
        callback = ProgressCallback(session_id, expected_frames, processing_mode)
        
        # Use the new VideoProcessor approach
        processor = VideoProcessor(projector.depth_estimator)
        
        # Calculate resolution settings that VideoProcessor expects
        from depth_surge_3d.utils.resolution import get_resolution_dimensions, calculate_vr_output_dimensions, auto_detect_resolution
        
        # Resolve VR resolution if auto
        vr_resolution = settings.get('vr_resolution', 'auto')
        if vr_resolution == 'auto':
            vr_resolution = auto_detect_resolution(
                video_info['width'], 
                video_info['height'], 
                settings.get('vr_format', 'side_by_side')
            )
        
        # Get resolution dimensions
        per_eye_width, per_eye_height = get_resolution_dimensions(vr_resolution)
        vr_output_width, vr_output_height = calculate_vr_output_dimensions(
            per_eye_width, per_eye_height, settings.get('vr_format', 'side_by_side')
        )
        
        # Add calculated dimensions to settings
        settings.update({
            'per_eye_width': per_eye_width,
            'per_eye_height': per_eye_height,
            'vr_output_width': vr_output_width,
            'vr_output_height': vr_output_height,
            'source_width': video_info['width'],
            'source_height': video_info['height'],
            'source_fps': video_info['fps']
        })
        
        success = processor.process(
            video_path=video_path,
            output_dir=output_dir,
            video_properties=video_info,
            settings=settings,
            progress_callback=callback
        )
        
        if not success:
            raise Exception("Video processing failed")
        
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
        'fisheye_projection': 'stereographic',
        'fisheye_fov': 105,
        'crop_factor': 1.0,
        'vr_resolution': 'auto',
        'fisheye_crop_factor': 1.0,
        'hole_fill_quality': 'fast'
    }
    
    # Try to detect VR format from directory structure
    if (output_path / INTERMEDIATE_DIRS["vr_frames"]).exists():
        # Check if there are side-by-side or over-under frames
        vr_files = list((output_path / INTERMEDIATE_DIRS["vr_frames"]).glob('*.png'))
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
        INTERMEDIATE_DIRS["vr_frames"]: 'Final VR frames',
        INTERMEDIATE_DIRS["left_final"]: 'Final left frames',
        INTERMEDIATE_DIRS["right_final"]: 'Final right frames',
        INTERMEDIATE_DIRS["left_distorted"]: 'Distorted left frames',
        INTERMEDIATE_DIRS["right_distorted"]: 'Distorted right frames',
        INTERMEDIATE_DIRS["left_cropped"]: 'Cropped left frames',
        INTERMEDIATE_DIRS["right_cropped"]: 'Cropped right frames',
        INTERMEDIATE_DIRS["left_frames"]: 'Basic left frames',
        INTERMEDIATE_DIRS["right_frames"]: 'Basic right frames',
        INTERMEDIATE_DIRS["depth_maps"]: 'Depth maps',
        INTERMEDIATE_DIRS["supersampled"]: 'Super sampled frames',
        INTERMEDIATE_DIRS["frames"]: 'Original frames'
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
        sample_frame_dirs = [d for d in [INTERMEDIATE_DIRS["left_final"], INTERMEDIATE_DIRS["right_final"], INTERMEDIATE_DIRS["vr_frames"]] if (batch_path / d).exists()]
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
                        if frame_dir == INTERMEDIATE_DIRS["vr_frames"]:
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
        stages = [INTERMEDIATE_DIRS["vr_frames"], INTERMEDIATE_DIRS["left_final"], INTERMEDIATE_DIRS["right_final"]]
        frame_dir = None
        for stage in stages:
            stage_path = batch_path / stage
            if stage_path.exists() and list(stage_path.glob('*.png')):
                frame_dir = stage_path
                break
    else:
        # Use specified stage
        stage_mapping = {
            'vr_frames': INTERMEDIATE_DIRS["vr_frames"],
            'left_right_final': INTERMEDIATE_DIRS["left_final"],
            'left_right_fisheye': INTERMEDIATE_DIRS["left_distorted"],
            'left_right_basic': INTERMEDIATE_DIRS["left_frames"]
        }
        stage_name = stage_mapping.get(frame_source, INTERMEDIATE_DIRS["vr_frames"])
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
    
    # Only print startup message if not already printed by run_ui.sh
    if not os.environ.get('DEPTH_SURGE_UI_SCRIPT'):
        print("Starting Depth Surge 3D Web UI...")
        print(f"Navigate to http://localhost:{args.port}")
    
    socketio.run(app, host=args.host, port=args.port, debug=args.verbose)