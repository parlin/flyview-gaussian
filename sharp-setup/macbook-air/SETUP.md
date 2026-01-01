# SHARP Setup: MacBook Air (M1/M4)

> ⚠️ **Limitation**: Video rendering requires CUDA. On Mac, you can only generate `.ply` files and view them in external viewers.

---

## Prerequisites

- macOS 12.0+ (Monterey or later)
- Homebrew installed
- ~2GB free disk space

---

## Step 1: Install Miniforge (Conda for Apple Silicon)

```bash
# Install Miniforge (optimized for Apple Silicon)
brew install miniforge

# Initialize conda
conda init zsh  # or: conda init bash
```

**Restart your terminal after this step.**

---

## Step 2: Clone Repository

```bash
cd ~
git clone https://github.com/apple/ml-sharp.git
cd ml-sharp
```

---

## Step 3: Create Environment & Install

```bash
# Create Python 3.13 environment
conda create -n sharp python=3.13 -y
conda activate sharp

# Install dependencies
pip install -r requirements.txt

# Verify installation
sharp --help
```

---

## Step 4: Prepare Test Images

Create a test folder and add images:

```bash
mkdir -p ~/sharp-test/input
mkdir -p ~/sharp-test/output

# Copy a test image (replace with your actual image)
cp /path/to/your/photo.jpg ~/sharp-test/input/
```

### Good Test Images
- Indoor scenes with objects at different depths
- Well-lit environments
- Sharp focus (no motion blur)
- JPEG or PNG format

---

## Step 5: Run Prediction

```bash
conda activate sharp
cd ~/ml-sharp

# Single image
sharp predict -i ~/sharp-test/input/photo.jpg -o ~/sharp-test/output

# Multiple images in folder
sharp predict -i ~/sharp-test/input/ -o ~/sharp-test/output

# With verbose logging
sharp predict -i ~/sharp-test/input/ -o ~/sharp-test/output -v
```

### First Run Note
The model (~500MB) downloads automatically on first run and caches at `~/.cache/torch/hub/checkpoints/`.

---

## Step 6: View Results

Your output folder will contain `.ply` files (3D Gaussian Splats).

### Option A: SuperSplat (Web - Recommended)
1. Open https://playcanvas.com/supersplat/editor
2. Drag and drop your `.ply` file
3. Use mouse to orbit, zoom, and explore

### Option B: Polycam (Mac App)
1. Download from App Store: https://apps.apple.com/app/polycam-3d-scanner/id1532482376
2. Import `.ply` file
3. View and export

### Option C: Blender (Advanced)
1. Install Blender: `brew install --cask blender`
2. Use a Gaussian Splatting addon to import `.ply`

---

## Expected Performance

| Mac Model | Prediction Time |
|-----------|-----------------|
| M1 | ~5-8 seconds |
| M2 | ~4-6 seconds |
| M3 | ~3-5 seconds |
| M4 | ~2-4 seconds |

---

## Troubleshooting

### "MPS not available"
```bash
# Check PyTorch MPS support
python -c "import torch; print(torch.backends.mps.is_available())"
```
Should print `True`. If not, update macOS and reinstall PyTorch.

### Out of Memory
```bash
# Force CPU mode (slower but less memory)
sharp predict -i ~/sharp-test/input/ -o ~/sharp-test/output --device cpu
```

### Slow First Run
Normal - gsplat compiles kernels on first execution. Subsequent runs are faster.

---

## Quick Reference

```bash
# Activate environment
conda activate sharp

# Run prediction
sharp predict -i INPUT_PATH -o OUTPUT_PATH

# Check help
sharp predict --help
```

---

## Limitations on Mac

| Feature | Status |
|---------|--------|
| Gaussian prediction | ✅ Works |
| Export .ply files | ✅ Works |
| Video rendering (`--render`) | ❌ Requires CUDA |
| Real-time viewer | ❌ Use external apps |

**For video rendering**, use the Cloud GPU setup instead.
