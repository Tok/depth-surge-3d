# Usage Guide

## Web UI (Recommended)

Launch the web interface for an intuitive experience:

```bash
./run_ui.sh
```

Then open http://localhost:5000 in your browser. The web UI provides:
- **Drag & drop video upload** with instant preview
- **Visual time range selection** with duration display
- **Real-time progress tracking** with frame previews
- **Interactive settings panel** with proper defaults
- **Batch management** with timestamped outputs
- **Live preview** of depth maps and stereo frames

## Quick Start Script

For command-line processing with time ranges:

```bash
# Place your video as 'input_video.mp4' in the project directory
# Then run with start and end times
./start.sh 1:11 2:22

# The script automatically:
# - Uses input_video.mp4 as source
# - Extracts the specified time range
# - Preserves audio in the output (by default)
# - Upscales to minimum 1080p resolution
# - Interpolates frames to 60fps for smooth VR experience
# - Uses optimized settings for high-quality processing
```

## Basic Command Line Usage

```bash
# With uv (recommended - runs directly without manual activation)
uv run python depth_surge_3d.py input_video.mp4

# Or activate virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Then run the script
python depth_surge_3d.py input_video.mp4
```

## Advanced Command Line Usage

```bash
# Specify output directory and format
python depth_surge_3d.py input_video.mp4 -o ./my_output -f over_under

# Process only a specific time range
python depth_surge_3d.py input_video.mp4 -s 01:30 -e 03:45

# Process with hours:minutes:seconds format
python depth_surge_3d.py input_video.mp4 -s 00:01:30 -e 00:03:45

# Process from start to specific time
python depth_surge_3d.py input_video.mp4 -e 02:00

# Process from specific time to end
python depth_surge_3d.py input_video.mp4 -s 01:00

# Adjust stereo parameters
python depth_surge_3d.py input_video.mp4 -b 0.15 -fl 1200

# Use CPU only (if no GPU available)
python depth_surge_3d.py input_video.mp4 --device cpu

# Keep every step's images (depth maps, stereo pairs, ...)
python depth_surge_3d.py input_video.mp4 --keep-intermediates

# Disable audio preservation
python depth_surge_3d.py input_video.mp4 --no-audio

# Custom framerate and resolution
python depth_surge_3d.py input_video.mp4 --fps 120 --resolution 4k

# Combined example: high-quality 4K 60fps processing
python depth_surge_3d.py input_video.mp4 -s 01:30 -e 02:00 -f over_under --resolution 4k --fps 60
```

## Output Structure

When processing a video, the tool creates a timestamped directory with all intermediate and final files:

```
output/
└── timestamp_videoname_timestamp/
    ├── original_video.mp4      # Source video (uploaded or copied)
    ├── original_audio.flac     # Pre-extracted lossless audio
    ├── frames/                 # Extracted video frames
    ├── depth_maps/            # Generated depth maps (if --keep-intermediates)
    ├── left_frames/           # Left stereo images (if --keep-intermediates)
    ├── right_frames/          # Right stereo images (if --keep-intermediates)
    ├── vr_frames/             # Combined VR frames
    └── videoname_3D_side_by_side_0118-0133.mp4  # Final VR video
```

## Processing Modes

Both modes work through the video in batches (`--batch-size`, default 120 frames: 5 s at 24 fps, 4 s at 30 fps, 2 s at 60 fps) and save
their progress after every batch.

- **Batched** (default, `--mode batched`): depth maps, stereo pairs, distortion, cropping,
  upscaling and VR assembly run for one batch of frames, then for the next. Finished VR frames
  appear early, and an interrupted job loses at most the batch it was working on.
- **Sequential** (`--mode sequential`): depth maps are finished for the whole video first, then
  the frames are processed, then the video is joined.

Stereo generation, distortion, cropping and VR assembly of a frame happen in one pass, and the
video is encoded in one segment per batch (`98_video_segments/`), joined at the end.

### Intermediate files

- **Not kept** (default): the extracted frames are deleted as the job goes; of everything
  else only about one batch exists at any time, plus the video segments. Disk use peaks near
  the size of the extracted frames (about 2 MB per 1080p frame). The job can still be resumed.
- **Kept** (`--keep-intermediates`): every step's images stay in the batch directory. Needed to
  redo single steps with changed settings, and useful for inspection. Needs many times the space
  (about 18 MB per frame for 1080p to 1536 px eyes). It is a start-up option, for the CLI and
  for the web server (`./run_ui.sh --keep-intermediates` applies it to all jobs); the web UI has
  no per-job switch.

An estimate of the needed disk space is shown before the job starts.

### Reusing a batch directory with new settings

```bash
python depth_surge_3d.py input_video.mp4 --reuse ./output/my_video_20261114_120000/ -b 0.08
```

Only the steps the changed settings affect run again (a new baseline keeps the depth maps, a
new encoder only re-encodes). This needs the intermediate files (`--keep-intermediates` on the
first run).

### Job queue (web UI)

Jobs started while another one is running are queued and run one after another. The Jobs list
shows the running job, the queue and unfinished jobs.

Depth maps are written to `02_depth_maps/` (16-bit PNG) as soon as each chunk is final, in
both modes.

## Resume Interrupted Processing

If processing is interrupted (power failure, crash, Ctrl+C, the Stop button), continue it:

```bash
python depth_surge_3d.py --resume ./output/my_video_20261114_120000/
```

In the web UI, press **Resume** next to the job in the Jobs list.

- The job continues with its original settings and model, after the last finished batch.
  The result is identical to an uninterrupted run.
- Frames are not extracted again; finished depth maps and VR frames are kept.
- Starting a job again in the same directory with changed settings only repeats the steps
  those settings affect (e.g. a new fisheye FOV keeps depth maps and stereo pairs). This
  needs the intermediate files; without them the job starts over.

### Video from an unfinished job

To get a video of the part that is finished, without resuming:

```bash
python depth_surge_3d.py --assemble-partial ./output/my_video_20261114_120000/
```

or press **Assemble** next to the job in the web UI. The video contains the finished VR
frames and the matching part of the audio (`*_partial_<N>frames.mp4`); the job can still be
resumed afterwards.

## VR Viewing

The generated video can be viewed with:
- VR headsets (Oculus, HTC Vive, etc.)
- Cardboard VR viewers
- 3D video players that support side-by-side or over-under formats
- Wide-screen displays (for cinema formats)

## Workflow Example

Here's a typical workflow:

1. **Setup** (one time):
   ```bash
   ./setup.sh
   ```

2. **Test setup** (optional but recommended):
   ```bash
   ./test.sh
   ```

3. **Prepare your video**:
   ```bash
   cp my_video.mp4 input_video.mp4
   ```

4. **Quick processing**:
   ```bash
   # Process a 15-second clip from 1:00 to 1:15 with audio
   ./start.sh 1:00 1:15
   ```

5. **Find your output**:
   ```bash
   # The processed video will be in the output/ directory
   ls -lh output/*/
   ```
