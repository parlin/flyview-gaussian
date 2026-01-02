# Gaussian Splatting Setup - Choose Your Path

Implementation paths for 3D/4D Gaussian Splatting.

## Single Image → 3DGS (Apple SHARP)
Three parallel paths for [Apple ml-sharp](https://github.com/apple/ml-sharp).

## Multi-Camera Video → 4DGS
See [4d-multicam/](./4d-multicam/SETUP.md) for dynamic 4D Gaussian Splatting from multiple camera angles.

---

## Quick Decision Guide

| Your Situation | Recommended Path | Full Features? |
|----------------|------------------|----------------|
| Just want to test quickly | [Cloud GPU](./cloud-gpu/) (Colab) | ✅ Yes |
| Have NVIDIA GPU at home | [NVIDIA PC](./nvidia-pc/) | ✅ Yes |
| Only have MacBook | [MacBook Air](./macbook-air/) | ⚠️ Partial |

---

## Paths

### 📁 [macbook-air/](./macbook-air/SETUP.md)
- **Hardware**: MacBook Air M1, M2, M3, M4
- **Features**: Prediction only (no video rendering)
- **Output**: `.ply` files viewable in SuperSplat
- **Cost**: Free (your hardware)

### 📁 [nvidia-pc/](./nvidia-pc/SETUP.md)
- **Hardware**: PC with NVIDIA RTX 20xx/30xx/40xx
- **Features**: Full (prediction + video rendering)
- **Output**: `.ply` files + `.mp4` videos
- **Cost**: Free (your hardware)

### 📁 [cloud-gpu/](./cloud-gpu/SETUP.md)
- **Hardware**: Cloud providers (Colab, RunPod, AWS, etc.)
- **Features**: Full (prediction + video rendering)
- **Output**: `.ply` files + `.mp4` videos
- **Cost**: Free (Colab) or $0.20-2.00/hr (paid)

### 📁 [4d-multicam/](./4d-multicam/SETUP.md) ⭐ NEW
- **Input**: Multiple MP4 videos from different angles
- **Features**: 4D Gaussian Splatting (dynamic scenes)
- **Key Tech**: SyncTrack4D (coming), 4D Gaussians (available)
- **Best For**: Multi-cam event recordings

---

## Feature Comparison (Single Image)

| Feature | MacBook | NVIDIA PC | Cloud GPU |
|---------|---------|-----------|-----------|
| Generate 3DGS (.ply) | ✅ | ✅ | ✅ |
| Render videos (.mp4) | ❌ | ✅ | ✅ |
| Speed | Medium | Fast | Fast |
| Setup complexity | Easy | Medium | Easy-Medium |
| Cost | Free | Free | Free-Paid |

---

## Fastest POC

**Google Colab** (5 minutes, free, full features):

1. Open https://colab.research.google.com
2. Set runtime to **GPU (T4)**
3. Run:
```python
!git clone https://github.com/apple/ml-sharp.git
%cd ml-sharp
!pip install -r requirements.txt -q

from google.colab import files
uploaded = files.upload()  # Upload your image

!mkdir -p output
!sharp predict -i /content/{list(uploaded.keys())[0]} -o output --render

files.download('output/')
```

---

## Questions?

- [Project Page](https://apple.github.io/ml-sharp/)
- [GitHub Issues](https://github.com/apple/ml-sharp/issues)
- [Paper](https://arxiv.org/abs/2512.10685)
