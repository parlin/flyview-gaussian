# Multi-Camera Video → 4D Gaussian Splatting
## Complete Step-by-Step Guide (January 2026)

> Turn your multi-camera event recordings into 4D Gaussian Splats

---

## Overview

```
Your MP4 Files → Sync → Extract Frames → COLMAP → 4D Gaussians → 4D Viewer
```

**Total Time**: ~2-3 hours (mostly automated)  
**Requirements**: Google Colab (free) or NVIDIA GPU

---

## Prerequisites

### On Your Mac (Preparation)
- Your multi-camera MP4 files
- FFmpeg installed: `brew install ffmpeg`
- Python 3.10+: `brew install python`

### On Cloud (Training)
- Google Colab account (free)
- Or: RunPod/Vast.ai (~$0.40/hr)

---

# PHASE 1: Local Preparation (Mac)

## Step 1: Organize Your Videos

Create a project folder:

```bash
mkdir -p ~/4dgs-project/raw_videos
cd ~/4dgs-project

# Copy your MP4 files here
cp /path/to/camera1.mp4 raw_videos/
cp /path/to/camera2.mp4 raw_videos/
cp /path/to/camera3.mp4 raw_videos/
# ... all camera angles
```

---

## Step 2: Find Synchronization Points

### Option A: Visual Inspection (Quick)

Open videos side-by-side in QuickTime or VLC:
1. Find a common event (clap, motion start, flash)
2. Note the timestamp in each video
3. Record offsets

### Option B: Audio-Based Sync (Automated)

Create `sync_audio.py`:

```python
#!/usr/bin/env python3
"""
Audio-based video synchronization.
Finds time offsets between videos using audio cross-correlation.
"""

import subprocess
import numpy as np
import os
from pathlib import Path

def extract_audio(video_path, audio_path):
    """Extract audio from video as mono WAV."""
    cmd = [
        'ffmpeg', '-y', '-i', str(video_path),
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        str(audio_path)
    ]
    subprocess.run(cmd, capture_output=True)

def load_audio(path):
    """Load WAV file as numpy array."""
    import wave
    with wave.open(str(path), 'rb') as w:
        frames = w.readframes(w.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    return audio

def find_offset(audio1, audio2, sample_rate=16000):
    """Find offset of audio2 relative to audio1 using cross-correlation."""
    # Use first 60 seconds max for speed
    max_samples = sample_rate * 60
    a1 = audio1[:max_samples]
    a2 = audio2[:max_samples]
    
    # Cross-correlation
    correlation = np.correlate(a1, a2, mode='full')
    offset_samples = np.argmax(correlation) - len(a2) + 1
    offset_seconds = offset_samples / sample_rate
    
    return offset_seconds

def main():
    raw_dir = Path("raw_videos")
    audio_dir = Path("temp_audio")
    audio_dir.mkdir(exist_ok=True)
    
    videos = sorted(raw_dir.glob("*.mp4"))
    if len(videos) < 2:
        print("Need at least 2 videos!")
        return
    
    print(f"Found {len(videos)} videos")
    
    # Extract audio from all videos
    audio_files = []
    for video in videos:
        audio_path = audio_dir / f"{video.stem}.wav"
        print(f"Extracting audio: {video.name}")
        extract_audio(video, audio_path)
        audio_files.append(audio_path)
    
    # Load reference audio (first video)
    ref_audio = load_audio(audio_files[0])
    print(f"\nReference video: {videos[0].name}")
    
    # Find offsets relative to reference
    offsets = {videos[0].name: 0.0}
    
    for i, (video, audio_path) in enumerate(zip(videos[1:], audio_files[1:]), 1):
        audio = load_audio(audio_path)
        offset = find_offset(ref_audio, audio)
        offsets[video.name] = offset
        print(f"{video.name}: {offset:+.3f}s relative to reference")
    
    # Save offsets
    with open("sync_offsets.txt", "w") as f:
        f.write("# Video sync offsets (seconds relative to first video)\n")
        f.write("# Positive = starts later, Negative = starts earlier\n\n")
        for video, offset in offsets.items():
            f.write(f"{video}: {offset:.3f}\n")
    
    print(f"\nOffsets saved to sync_offsets.txt")
    
    # Cleanup
    for f in audio_dir.glob("*.wav"):
        f.unlink()
    audio_dir.rmdir()

if __name__ == "__main__":
    main()
```

Run it:

```bash
cd ~/4dgs-project
python sync_audio.py
```

Output (`sync_offsets.txt`):
```
camera1.mp4: 0.000
camera2.mp4: +0.342
camera3.mp4: -0.156
camera4.mp4: +0.089
```

---

## Step 3: Extract Synchronized Frames

Create `extract_frames.py`:

```python
#!/usr/bin/env python3
"""
Extract synchronized frames from multiple videos.
"""

import subprocess
import os
from pathlib import Path

# === CONFIGURATION ===

# Your sync offsets (from sync_audio.py or manual inspection)
# Format: "filename": offset_in_seconds
SYNC_OFFSETS = {
    "camera1.mp4": 0.0,
    "camera2.mp4": 0.342,    # Starts 0.342s after camera1
    "camera3.mp4": -0.156,   # Starts 0.156s before camera1
    "camera4.mp4": 0.089,
    # Add all your cameras...
}

# Scene selection
START_TIME = "00:01:00"  # Start of the scene you want (in reference video)
DURATION = 5             # Seconds to extract
FPS = 30                 # Frames per second

# === END CONFIGURATION ===

def extract_frames(video_path, output_dir, start_time, duration, fps):
    """Extract frames from video."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-y',
        '-ss', start_time,
        '-i', str(video_path),
        '-t', str(duration),
        '-vf', f'fps={fps}',
        '-q:v', '2',
        str(output_dir / 'frame_%05d.jpg')
    ]
    subprocess.run(cmd, capture_output=True)

def time_to_seconds(time_str):
    """Convert HH:MM:SS.mmm to seconds."""
    parts = time_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

def seconds_to_time(seconds):
    """Convert seconds to HH:MM:SS.mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def main():
    raw_dir = Path("raw_videos")
    output_base = Path("dataset")
    
    # Clear previous output
    if output_base.exists():
        import shutil
        shutil.rmtree(output_base)
    
    base_start_seconds = time_to_seconds(START_TIME)
    
    for i, (video_name, offset) in enumerate(SYNC_OFFSETS.items()):
        video_path = raw_dir / video_name
        if not video_path.exists():
            print(f"Warning: {video_path} not found, skipping")
            continue
        
        # Calculate adjusted start time for this video
        adjusted_start = base_start_seconds + offset
        adjusted_start_str = seconds_to_time(max(0, adjusted_start))
        
        # Output directory (cam00, cam01, etc.)
        output_dir = output_base / f"cam{i:02d}"
        
        print(f"Extracting {video_name} → {output_dir.name}")
        print(f"  Start: {adjusted_start_str} (offset: {offset:+.3f}s)")
        
        extract_frames(video_path, output_dir, adjusted_start_str, DURATION, FPS)
        
        # Count extracted frames
        frame_count = len(list(output_dir.glob("*.jpg")))
        print(f"  Frames: {frame_count}")
    
    print(f"\nDataset created in: {output_base}")
    print(f"Total cameras: {len(list(output_base.iterdir()))}")

if __name__ == "__main__":
    main()
```

**Edit the configuration section**, then run:

```bash
python extract_frames.py
```

---

## Step 4: Verify Frame Alignment

Quick visual check:

```bash
# Open first frame from each camera
open dataset/cam*/frame_00001.jpg
```

All images should show the **same moment in time**.

---

## Step 5: Prepare Upload Package

```bash
cd ~/4dgs-project

# Create zip for upload to Colab
zip -r dataset.zip dataset/

# Check size (should be manageable for upload)
ls -lh dataset.zip
```

---

# PHASE 2: Training on Google Colab

## Step 1: Open Colab Notebook

Go to: **https://colab.research.google.com**

Create new notebook, set GPU:
1. **Runtime** → **Change runtime type**
2. Select **T4 GPU** (free) or **A100** (Pro)

---

## Step 2: Setup Environment

```python
# Cell 1: Clone 4D Gaussians
!git clone https://github.com/hustvl/4DGaussians.git --recursive
%cd 4DGaussians
```

```python
# Cell 2: Install dependencies
!pip install -q plyfile tqdm scipy opencv-python imageio
!pip install -e submodules/depth-diff-gaussian-rasterization
!pip install -e submodules/simple-knn

# Verify GPU
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

```python
# Cell 3: Install COLMAP
!apt-get update -qq
!apt-get install -qq colmap
!colmap -h | head -5
```

---

## Step 3: Upload Your Dataset

```python
# Cell 4: Upload dataset.zip
from google.colab import files
print("Upload your dataset.zip file:")
uploaded = files.upload()
```

```python
# Cell 5: Extract dataset
!unzip -q dataset.zip -d data/multipleview/
!mv data/multipleview/dataset data/multipleview/my_event

# Verify structure
!echo "Dataset structure:"
!find data/multipleview/my_event -type f | head -20
!echo ""
!echo "Cameras found:"
!ls data/multipleview/my_event/
```

---

## Step 4: Generate Camera Poses (COLMAP)

```python
# Cell 6: Run COLMAP pipeline
!bash multipleviewprogress.sh my_event
```

This takes 10-30 minutes depending on frame count. It:
1. Extracts features from all frames
2. Matches features across cameras
3. Reconstructs camera poses
4. Generates initial point cloud

```python
# Cell 7: Verify COLMAP output
!echo "COLMAP outputs:"
!ls -la data/multipleview/my_event/sparse_/
!echo ""
!echo "Point cloud:"
!ls -la data/multipleview/my_event/*.ply
```

---

## Step 5: Create Training Config

```python
# Cell 8: Create configuration file
config_content = '''
ModelParams = dict(
    # Model architecture
    name = "my_event",
    deform_type = 'deform',
    is_blender = False,
    hyper = False,
    
    # Point cloud
    init_point_cloud = "points3D_multipleview.ply",
)

OptimizationParams = dict(
    # Training iterations
    coarse_iterations = 3000,
    iterations = 10000,
    
    # Learning rates
    position_lr_init = 0.00016,
    position_lr_final = 0.0000016,
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.000016,
    
    # Densification
    densify_until_iter = 5000,
    densify_grad_threshold = 0.0002,
    
    # Pruning
    opacity_reset_interval = 3000,
    prune_interval = 100,
    
    # Batch size
    batch_size = 1,
)

ModelHiddenParams = dict(
    # Deformation network
    net_width = 64,
    timebase_pe = 4,
    defor_depth = 1,
    posebase_pe = 10,
    
    # Scales
    scale_rotation_activation = "sigmoid",
    
    # Multi-resolution hash encoding (for efficiency)
    multires = [1, 2, 4, 8],
)
'''

with open('arguments/multipleview/my_event.py', 'w') as f:
    f.write(config_content)

print("Config created: arguments/multipleview/my_event.py")
```

---

## Step 6: Train 4D Gaussians

```python
# Cell 9: Start training
!python train.py \
    -s data/multipleview/my_event \
    --port 6017 \
    --expname "multipleview/my_event" \
    --configs arguments/multipleview/my_event.py
```

**Training time**:
- T4 GPU: ~30-60 minutes
- A100 GPU: ~15-20 minutes

You'll see output like:
```
Iteration 100/10000: Loss=0.15, PSNR=22.3
Iteration 200/10000: Loss=0.12, PSNR=24.1
...
```

---

## Step 7: Render Results

```python
# Cell 10: Render novel views
!python render.py \
    --model_path output/multipleview/my_event \
    --configs arguments/multipleview/my_event.py \
    --skip_train \
    --skip_test
```

```python
# Cell 11: Create video from renders
import imageio
import glob
from pathlib import Path

render_dir = Path("output/multipleview/my_event/renders")
if render_dir.exists():
    frames = sorted(glob.glob(str(render_dir / "*.png")))
    if frames:
        images = [imageio.imread(f) for f in frames]
        imageio.mimwrite("novel_view.mp4", images, fps=30)
        print(f"Video saved: novel_view.mp4 ({len(frames)} frames)")
```

```python
# Cell 12: Display video
from IPython.display import Video
Video("novel_view.mp4", embed=True, width=640)
```

---

## Step 8: Download Results

```python
# Cell 13: Package results for download
import shutil

# Create results package
results_dir = Path("results_package")
results_dir.mkdir(exist_ok=True)

# Copy key outputs
shutil.copy("novel_view.mp4", results_dir)

# Copy trained model (for later use)
model_ply = Path("output/multipleview/my_event/point_cloud/iteration_10000/point_cloud.ply")
if model_ply.exists():
    shutil.copy(model_ply, results_dir / "4dgs_model.ply")

# Zip and download
shutil.make_archive("4dgs_results", "zip", results_dir)

from google.colab import files
files.download("4dgs_results.zip")
```

---

# PHASE 3: Viewing Results

## Option A: Web Viewer (Quick)

Upload your `.ply` to:
- **SuperSplat**: https://playcanvas.com/supersplat/editor
- **Luma AI**: https://lumalabs.ai/viewer

## Option B: SIBR Viewer (Interactive, Local)

```bash
# On your Mac with NVIDIA GPU (or Linux)
git clone https://gitlab.inria.fr/sibr/sibr_core.git
cd sibr_core
cmake . && make -j
./install/bin/SIBR_gaussianViewer_app -m /path/to/model
```

## Option C: Nerfstudio Viewer

```bash
pip install nerfstudio
ns-viewer --load /path/to/model
```

---

# Quick Reference

## Full Pipeline Commands

```bash
# Local (Mac)
cd ~/4dgs-project
python sync_audio.py                    # Find sync offsets
# Edit extract_frames.py with offsets
python extract_frames.py                # Extract synced frames
zip -r dataset.zip dataset/             # Package for upload

# Colab
!git clone https://github.com/hustvl/4DGaussians.git --recursive
%cd 4DGaussians
!pip install -e submodules/depth-diff-gaussian-rasterization
!pip install -e submodules/simple-knn
!apt-get install -qq colmap
# Upload dataset.zip
!unzip -q dataset.zip -d data/multipleview/
!mv data/multipleview/dataset data/multipleview/my_event
!bash multipleviewprogress.sh my_event
!python train.py -s data/multipleview/my_event --expname "multipleview/my_event" --configs arguments/multipleview/my_event.py
!python render.py --model_path output/multipleview/my_event --configs arguments/multipleview/my_event.py
```

---

## Troubleshooting

### "COLMAP failed"
- Ensure frames have enough texture/features
- Try adding more cameras or frames
- Check that frames show overlapping scene content

### "Out of memory"
- Reduce `batch_size` in config
- Use fewer frames (reduce FPS or duration)
- Upgrade to A100 GPU

### "Poor quality results"
- Add more camera angles (4+ recommended)
- Increase training iterations
- Check sync is accurate (frames should show same moment)

### "Training very slow"
- Reduce number of frames
- Lower iterations to 5000 for quick test
- Use A100 instead of T4

---

## Recommended Settings by Scene Type

| Scene Type | Cameras | Duration | FPS | Iterations |
|------------|---------|----------|-----|------------|
| Quick test | 3-4 | 2s | 15 | 5000 |
| Normal | 4-6 | 3-5s | 30 | 10000 |
| High quality | 6+ | 5-10s | 30 | 20000 |

---

## File Sizes Reference

| Cameras | Frames | Dataset Size | Model Size |
|---------|--------|--------------|------------|
| 4 | 150 | ~50MB | ~100MB |
| 6 | 300 | ~150MB | ~200MB |
| 8 | 600 | ~400MB | ~400MB |
