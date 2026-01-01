# SHARP Setup: NVIDIA PC (Windows/Linux)

> ✅ **Full Support**: All features including video rendering work with NVIDIA GPUs.

---

## Prerequisites

- NVIDIA GPU (RTX 20xx series or newer recommended)
- NVIDIA Driver 525+ installed
- CUDA Toolkit 12.x (auto-installed with PyTorch)
- ~4GB free disk space
- 8GB+ VRAM recommended

---

## Check Your GPU

```bash
# Check NVIDIA driver
nvidia-smi

# Should show your GPU model and driver version
```

---

## Option A: Linux Setup

### Step 1: Install Miniconda

```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### Step 2: Clone and Install

```bash
git clone https://github.com/apple/ml-sharp.git
cd ml-sharp

conda create -n sharp python=3.13 -y
conda activate sharp

pip install -r requirements.txt

# Verify
sharp --help
```

### Step 3: Verify CUDA

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## Option B: Windows Setup

### Step 1: Install Miniconda

1. Download: https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
2. Run installer (check "Add to PATH")
3. Open **Anaconda Prompt** (not regular CMD)

### Step 2: Clone and Install

```cmd
git clone https://github.com/apple/ml-sharp.git
cd ml-sharp

conda create -n sharp python=3.13 -y
conda activate sharp

pip install -r requirements.txt

sharp --help
```

### Step 3: Install Visual Studio Build Tools (if needed for gsplat)

If you get compilation errors:
1. Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "Desktop development with C++"
3. Restart and retry

---

## Step 4: Prepare Test Images

```bash
# Linux
mkdir -p ~/sharp-test/input ~/sharp-test/output
cp /path/to/photos/*.jpg ~/sharp-test/input/

# Windows (in Anaconda Prompt)
mkdir sharp-test\input sharp-test\output
copy C:\path\to\photos\*.jpg sharp-test\input\
```

---

## Step 5: Run Prediction (Without Rendering)

```bash
conda activate sharp
cd ml-sharp  # or full path to ml-sharp

# Single image
sharp predict -i ~/sharp-test/input/photo.jpg -o ~/sharp-test/output

# Folder of images
sharp predict -i ~/sharp-test/input/ -o ~/sharp-test/output
```

---

## Step 6: Run With Video Rendering

```bash
# Generate .ply files AND .mp4 videos
sharp predict -i ~/sharp-test/input/ -o ~/sharp-test/output --render
```

### Render Existing .ply Files

```bash
# If you already have .ply files from prediction
sharp render -i ~/sharp-test/output/ -o ~/sharp-test/renderings/
```

---

## Expected Performance

| GPU | Prediction | Rendering |
|-----|------------|-----------|
| RTX 4090 | ~0.8s | ~30s |
| RTX 4080 | ~1.0s | ~35s |
| RTX 3080 | ~1.2s | ~45s |
| RTX 3070 | ~1.5s | ~55s |
| RTX 3060 | ~2.0s | ~70s |
| RTX 2080 | ~2.5s | ~90s |

---

## Troubleshooting

### "CUDA not available"

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### "No CUDA toolkit found" (gsplat error)

```bash
# Install CUDA toolkit
# Linux (Ubuntu/Debian)
sudo apt install nvidia-cuda-toolkit

# Or set CUDA_HOME
export CUDA_HOME=/usr/local/cuda
```

### Out of Memory (OOM)

```bash
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Use smaller batch or lower resolution
# (requires code modification)
```

### Windows: "cannot import name 'csrc' from 'gsplat'"

1. Install Visual Studio Build Tools with C++ support
2. Ensure CUDA Toolkit 12.x is installed
3. Set environment variable: `set TORCH_CUDA_ARCH_LIST=8.6` (adjust for your GPU)
4. Reinstall: `pip uninstall gsplat && pip install gsplat`

---

## Quick Reference

```bash
# Activate environment
conda activate sharp

# Prediction only
sharp predict -i INPUT -o OUTPUT

# Prediction + video rendering
sharp predict -i INPUT -o OUTPUT --render

# Render from existing .ply
sharp render -i PLY_FOLDER -o VIDEO_OUTPUT

# Check GPU
nvidia-smi
```

---

## Output Files

```
output/
├── photo1.ply          # 3D Gaussian Splat
├── photo1.mp4          # Rendered video (if --render)
├── photo2.ply
├── photo2.mp4
└── ...
```

---

## Viewing Results

### Videos (.mp4)
Open directly in any video player.

### 3D Gaussians (.ply)

| Viewer | Platform | Notes |
|--------|----------|-------|
| SuperSplat | Web | https://playcanvas.com/supersplat/editor |
| Luma AI | Web | https://lumalabs.ai/viewer |
| Polycam | Desktop | Full-featured |
| Blender | Desktop | With GS addon |

---

## Full Feature Support

| Feature | Status |
|---------|--------|
| Gaussian prediction | ✅ |
| Export .ply files | ✅ |
| Video rendering | ✅ |
| Batch processing | ✅ |
| Custom trajectories | ✅ |
