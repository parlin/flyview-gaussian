# 4D Gaussian Splatting from Multi-Camera Video

> For multi-camera recordings → 4D Gaussian Splatting

👉 **See [GUIDE.md](./GUIDE.md) for complete step-by-step instructions**

---

## Quick Start

```bash
# 1. On your Mac - prepare data
cd ~/4dgs-project
python scripts/sync_audio.py         # Find sync offsets
python scripts/extract_frames.py     # Extract synced frames
zip -r dataset.zip dataset/          # Package for upload

# 2. On Google Colab - train model  
# Upload 4DGS_MultiCam_Colab.ipynb and run all cells
```

---

## Files in This Folder

| File | Description |
|------|-------------|
| **GUIDE.md** | Complete step-by-step tutorial |
| **4DGS_MultiCam_Colab.ipynb** | Ready-to-use Colab notebook |
| **scripts/sync_audio.py** | Auto-sync videos using audio |
| **scripts/extract_frames.py** | Extract synchronized frames |

---

## Overview

| Method | Sync Required? | Status | Best For |
|--------|---------------|--------|----------|
| **SyncTrack4D** | ❌ No | 🔜 Coming | Your exact use case |
| **4D Gaussians** | ✅ Yes | ✅ Available | Pre-synced footage |
| **Dynamic 3DGS** | ✅ Yes | ✅ Available | Shorter clips |

---

## Option 1: SyncTrack4D (Recommended - When Released)

### Why It's Perfect for You
- Handles **unsynchronized** multi-camera video
- Automatic temporal alignment via motion tracking
- Sub-frame synchronization accuracy (<0.26 frames)
- First general 4DGS for unsync'd video sets

### Paper
- **arXiv**: https://arxiv.org/abs/2512.04315
- **Authors**: UMD + Sony (Dinesh Manocha's lab)

### Installation (When Code Releases)

```bash
# Watch for release:
# https://github.com/search?q=SyncTrack4D

git clone https://github.com/[author]/SyncTrack4D.git
cd SyncTrack4D
pip install -r requirements.txt
```

### Expected Usage

```bash
# Your unsynchronized videos
synctrack4d --videos cam1.mp4 cam2.mp4 cam3.mp4 --output scene_4d/
```

### How to Get Early Access
1. Email authors (find on arXiv page)
2. Watch GitHub for "SyncTrack4D"
3. Check project page when announced

---

## Option 2: 4D Gaussians (Available Now)

### Requirements
- Videos must be **synchronized** (same start time)
- NVIDIA GPU with CUDA
- ~10GB VRAM recommended

### Step 1: Synchronize Your Videos Manually

Find a common sync point (audio clap, visual flash, motion start):

```python
# sync_videos.py
import subprocess
import os

# Your videos with their sync timestamps (moment of clap/flash/etc)
videos = {
    "cam00": {"file": "camera1.mp4", "sync_time": "00:01:23.500"},
    "cam01": {"file": "camera2.mp4", "sync_time": "00:01:24.200"},
    "cam02": {"file": "camera3.mp4", "sync_time": "00:01:23.100"},
    "cam03": {"file": "camera4.mp4", "sync_time": "00:01:25.000"},
}

output_dir = "synced_videos"
duration = "5"  # seconds to extract

os.makedirs(output_dir, exist_ok=True)

for cam, info in videos.items():
    output = f"{output_dir}/{cam}.mp4"
    cmd = f'ffmpeg -ss {info["sync_time"]} -i "{info["file"]}" -t {duration} -c:v libx264 -an "{output}"'
    subprocess.run(cmd, shell=True)
    print(f"Created: {output}")
```

### Step 2: Extract Frames

```python
# extract_frames.py
import subprocess
import os

synced_dir = "synced_videos"
output_base = "4dgs_dataset"
fps = 30

for video in os.listdir(synced_dir):
    if video.endswith(".mp4"):
        cam_name = video.replace(".mp4", "")
        output_dir = f"{output_base}/{cam_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = f'ffmpeg -i "{synced_dir}/{video}" -vf "fps={fps}" -q:v 2 "{output_dir}/%06d.jpg"'
        subprocess.run(cmd, shell=True)
```

### Step 3: Setup 4D Gaussians (Colab)

```python
# Cell 1: Clone and install
!git clone https://github.com/hustvl/4DGaussians.git --recursive
%cd 4DGaussians
!pip install -r requirements.txt
!pip install ./submodules/diff-gaussian-rasterization
!pip install ./submodules/simple-knn

# Cell 2: Upload your prepared dataset
# Use Colab file browser to upload 4dgs_dataset/ folder
# Or use Google Drive:
from google.colab import drive
drive.mount('/content/drive')
!cp -r "/content/drive/MyDrive/4dgs_dataset" /content/

# Cell 3: Generate camera poses with COLMAP
!apt install colmap -y
!python scripts/colmap_preprocess.py --data /content/4dgs_dataset

# Cell 4: Train 4D Gaussians
!python train.py -s /content/4dgs_dataset --expname "my_event" --configs arguments/dnerf/default.py

# Cell 5: Render output
!python render.py --model_path output/my_event --configs arguments/dnerf/default.py
```

---

## Option 3: Nerfstudio Splatfacto (Easiest)

```bash
# Install
pip install nerfstudio

# Process your synced videos
ns-process-data video --data synced_videos/ --output-dir processed/

# Train
ns-train splatfacto --data processed/

# View interactively
ns-viewer --load-config outputs/processed/splatfacto/config.yml
```

---

## Data Preparation Tips

### From Your iOS Multi-Cam App

1. **Export all camera angles** as MP4 files
2. **Note any sync markers** (if your app records them)
3. **Check audio tracks** - audio can help find sync points

### Finding Sync Points

| Method | How |
|--------|-----|
| **Audio clap** | Waveform spike visible in video editor |
| **Flash/light** | Sudden brightness change |
| **Motion start** | First frame of action |
| **Timecode** | If your app embeds it |

### Using FFmpeg to Find Sync

```bash
# Extract audio for sync analysis
ffmpeg -i camera1.mp4 -vn -acodec pcm_s16le audio1.wav
ffmpeg -i camera2.mp4 -vn -acodec pcm_s16le audio2.wav

# Use Audacity or Python to find offset
```

### Python Audio Sync (Automated)

```python
# audio_sync.py
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate

def find_offset(audio1_path, audio2_path):
    rate1, audio1 = wavfile.read(audio1_path)
    rate2, audio2 = wavfile.read(audio2_path)
    
    # Cross-correlate to find offset
    correlation = correlate(audio1[:, 0], audio2[:, 0])
    offset_samples = np.argmax(correlation) - len(audio2) + 1
    offset_seconds = offset_samples / rate1
    
    return offset_seconds

# Find how much camera2 is ahead/behind camera1
offset = find_offset("audio1.wav", "audio2.wav")
print(f"Camera 2 offset: {offset:.3f} seconds")
```

---

## Recommended Workflow for Your Events

```
┌─────────────────────────────────────────────────────────┐
│  Your iOS Multi-Cam Recordings                          │
│  (Multiple MP4 files from same event)                   │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1: Audio-based Sync (or manual)                   │
│  • Extract audio, cross-correlate                       │
│  • Or find visual sync point                            │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2: Trim videos to common timeline                 │
│  • ffmpeg -ss [sync_point] -i video.mp4 ...             │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3: Extract frames                                 │
│  • 30fps recommended                                    │
│  • Same frame count per camera                          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Step 4: COLMAP camera pose estimation                  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Step 5: Train 4D Gaussians                             │
│  • Cloud GPU (Colab/RunPod)                             │
│  • ~30-60 min training                                  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Output: 4D Gaussian Splat                              │
│  • Interactive 4D viewer                                │
│  • Novel view synthesis at any time                     │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start: Minimal 4D Test

Try with just 3-4 cameras and a 2-3 second clip first:

```bash
# 1. Sync and trim (assuming you found sync points)
ffmpeg -ss 00:01:23.5 -i cam1.mp4 -t 3 -c:v libx264 synced/cam00.mp4
ffmpeg -ss 00:01:24.2 -i cam2.mp4 -t 3 -c:v libx264 synced/cam01.mp4
ffmpeg -ss 00:01:23.1 -i cam3.mp4 -t 3 -c:v libx264 synced/cam02.mp4

# 2. Extract frames
for f in synced/*.mp4; do
  name=$(basename "$f" .mp4)
  mkdir -p frames/$name
  ffmpeg -i "$f" -vf "fps=30" frames/$name/%06d.jpg
done

# 3. Upload frames/ folder to Colab and run 4D Gaussians
```

---

## Resources

- **SyncTrack4D Paper**: https://arxiv.org/abs/2512.04315
- **4D Gaussians**: https://github.com/hustvl/4DGaussians
- **Dynamic 3DGS**: https://github.com/JonathonLuiten/Dynamic3DGaussians
- **Nerfstudio**: https://docs.nerf.studio/
- **COLMAP**: https://colmap.github.io/

---

## Monitoring for SyncTrack4D Release

Set up GitHub notification:

```bash
# Search weekly
gh search repos "SyncTrack4D" --limit 5

# Or set Google Alert for "SyncTrack4D github"
```

When released, it will likely be the best solution for your unsynchronized multi-cam workflow.
