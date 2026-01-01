# SHARP Setup: Cloud GPU

> ✅ **Full Support**: All features work. No local GPU required.

---

## Option 1: Google Colab (Free - Recommended for POC)

### Step 1: Open Colab
Go to https://colab.research.google.com and create a new notebook.

### Step 2: Enable GPU
1. Click **Runtime** → **Change runtime type**
2. Select **T4 GPU** (free) or **A100** (Colab Pro)
3. Click **Save**

### Step 3: Run This Notebook

Copy and paste each cell:

```python
# Cell 1: Clone repository
!git clone https://github.com/apple/ml-sharp.git
%cd ml-sharp
```

```python
# Cell 2: Install dependencies
!pip install -r requirements.txt -q
```

```python
# Cell 3: Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

```python
# Cell 4: Upload your image
from google.colab import files
uploaded = files.upload()  # Select your image file
```

```python
# Cell 5: Run prediction with rendering
import os
filename = list(uploaded.keys())[0]
!mkdir -p /content/output
!sharp predict -i /content/{filename} -o /content/output --render
```

```python
# Cell 6: Display results
from IPython.display import Video, display
import glob

# Show video
videos = glob.glob('/content/output/*.mp4')
if videos:
    display(Video(videos[0], embed=True, width=640))
```

```python
# Cell 7: Download results
from google.colab import files
import shutil

shutil.make_archive('/content/sharp_results', 'zip', '/content/output')
files.download('/content/sharp_results.zip')
```

### Colab Limitations
- Free tier: ~12 hours max session
- GPU may be unavailable during peak times
- Files deleted after session ends (download results!)

---

## Option 2: RunPod (Pay-as-you-go)

### Pricing
- RTX 4090: ~$0.40/hr
- A100 40GB: ~$1.00/hr

### Step 1: Create Account
1. Go to https://runpod.io
2. Sign up and add credits ($10 minimum)

### Step 2: Deploy Pod

1. Click **Deploy** → **GPU Pods**
2. Select template: **PyTorch 2.x**
3. Choose GPU: **RTX 4090** (best value)
4. Click **Deploy**

### Step 3: Connect via SSH or Jupyter

**Option A: Jupyter (Easy)**
1. Click **Connect** → **Jupyter Lab**
2. Open terminal in Jupyter

**Option B: SSH**
```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_rsa
```

### Step 4: Setup SHARP

```bash
git clone https://github.com/apple/ml-sharp.git
cd ml-sharp
pip install -r requirements.txt

# Test
sharp --help
```

### Step 5: Upload Images & Run

```bash
# Create directories
mkdir -p ~/input ~/output

# Upload via Jupyter file browser or:
# wget https://your-image-url.com/photo.jpg -O ~/input/photo.jpg

# Run with rendering
sharp predict -i ~/input/ -o ~/output --render
```

### Step 6: Download Results

Use Jupyter file browser to download, or:
```bash
# Zip results
zip -r results.zip ~/output/
# Download via Jupyter
```

---

## Option 3: Vast.ai (Budget Option)

### Pricing
- RTX 3090: ~$0.20/hr
- RTX 4090: ~$0.35/hr

### Step 1: Create Account
1. Go to https://vast.ai
2. Sign up and add credits

### Step 2: Find Instance

1. Click **Search** 
2. Filter: GPU = RTX 3090+, CUDA = 12.x
3. Sort by price
4. Select instance with **PyTorch** image

### Step 3: Connect & Setup

```bash
# SSH into instance
ssh -p <port> root@<ip>

# Setup
git clone https://github.com/apple/ml-sharp.git
cd ml-sharp
pip install -r requirements.txt
```

### Step 4: Run

```bash
mkdir -p ~/input ~/output
# Upload your images to ~/input
sharp predict -i ~/input/ -o ~/output --render
```

---

## Option 4: AWS EC2

### Recommended Instance
- **g4dn.xlarge**: T4 GPU, ~$0.50/hr
- **g5.xlarge**: A10G GPU, ~$1.00/hr

### Step 1: Launch Instance

1. Go to AWS Console → EC2
2. Launch instance
3. Select **Deep Learning AMI (Ubuntu)**
4. Choose **g4dn.xlarge**
5. Configure security group (allow SSH)
6. Launch

### Step 2: Connect

```bash
ssh -i your-key.pem ubuntu@<public-ip>
```

### Step 3: Setup

```bash
# Activate PyTorch environment
conda activate pytorch

# Clone and install
git clone https://github.com/apple/ml-sharp.git
cd ml-sharp
pip install -r requirements.txt
```

### Step 4: Run

```bash
mkdir -p ~/input ~/output
# Upload images via scp:
# scp -i your-key.pem photo.jpg ubuntu@<ip>:~/input/

sharp predict -i ~/input/ -o ~/output --render

# Download results
# scp -i your-key.pem -r ubuntu@<ip>:~/output ./local-output/
```

---

## Option 5: Lambda Labs

### Pricing
- A100 40GB: ~$1.10/hr
- H100: ~$2.00/hr

### Step 1: Setup

1. Go to https://lambdalabs.com/cloud
2. Create account, add SSH key
3. Launch **1x A100** instance

### Step 2: Connect & Run

```bash
ssh ubuntu@<instance-ip>

git clone https://github.com/apple/ml-sharp.git
cd ml-sharp
pip install -r requirements.txt

mkdir -p ~/input ~/output
sharp predict -i ~/input/ -o ~/output --render
```

---

## Cloud Provider Comparison

| Provider | GPU | Price/hr | Setup | Best For |
|----------|-----|----------|-------|----------|
| **Colab** | T4/A100 | Free/$10/mo | Easiest | POC/Testing |
| **RunPod** | 4090/A100 | $0.40-1.00 | Easy | Development |
| **Vast.ai** | Various | $0.20+ | Medium | Budget |
| **AWS** | T4/A10G | $0.50-1.00 | Complex | Production |
| **Lambda** | A100/H100 | $1.10-2.00 | Easy | Heavy workloads |

---

## Quick Start Script (Any Cloud)

Save as `setup_sharp.sh`:

```bash
#!/bin/bash
set -e

echo "=== Setting up SHARP ==="

# Clone repo
git clone https://github.com/apple/ml-sharp.git
cd ml-sharp

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p ~/sharp-input ~/sharp-output

# Verify installation
echo "=== Verifying GPU ==="
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

echo "=== Setup complete ==="
echo "Usage: sharp predict -i ~/sharp-input/ -o ~/sharp-output --render"
```

Run with:
```bash
bash setup_sharp.sh
```

---

## File Transfer Tips

### Upload to Cloud

```bash
# SCP (SSH)
scp -i key.pem local_image.jpg user@cloud-ip:~/sharp-input/

# rsync (faster for multiple files)
rsync -avz -e "ssh -i key.pem" ./images/ user@cloud-ip:~/sharp-input/
```

### Download from Cloud

```bash
# Single file
scp -i key.pem user@cloud-ip:~/sharp-output/result.mp4 ./

# Entire folder
scp -i key.pem -r user@cloud-ip:~/sharp-output/ ./local-results/
```

---

## Expected Performance (Cloud)

| GPU | Prediction | Rendering |
|-----|------------|-----------|
| T4 (Colab Free) | ~3s | ~90s |
| A100 (Colab Pro) | ~0.5s | ~20s |
| RTX 4090 (RunPod) | ~0.8s | ~30s |
| H100 (Lambda) | ~0.4s | ~15s |

---

## Troubleshooting

### "CUDA out of memory"
- Use a larger GPU instance
- Process fewer images at once

### "Connection timeout"
- Check security group/firewall settings
- Ensure port 22 (SSH) is open

### "Module not found"
```bash
pip install -r requirements.txt --force-reinstall
```

### Colab "GPU not available"
- Try again later (high demand)
- Upgrade to Colab Pro for priority access
