# Apple ml-sharp (SHARP) - Hardware & POC Assessment

> **SHARP**: Sharp Monocular View Synthesis in Less Than a Second  
> Converts a single photo into 3D Gaussian Splatting in <1 second

---

## 1. Hardware Requirements

### Summary Table

| Hardware | Gaussian Prediction | Video Rendering | Recommended |
|----------|---------------------|-----------------|-------------|
| MacBook Air M1 | ✅ Works (MPS) | ❌ No | Partial |
| MacBook Air M4 | ✅ Works (MPS) | ❌ No | Partial |
| PC w/ NVIDIA 30xx+ | ✅ Works (CUDA) | ✅ Works | **Best** |
| Cloud GPU (A100/T4/etc) | ✅ Works (CUDA) | ✅ Works | **Best** |
| CPU only | ✅ Works (slow) | ❌ No | Not Recommended |

### Detailed Breakdown

#### MacBook Air M1/M4 (Apple Silicon)
- **Prediction**: ✅ Supported via MPS (Metal Performance Shaders)
- **Rendering**: ❌ **Not supported** - gsplat requires CUDA
- **Workaround**: Export `.ply` files and view in external 3D viewers
- **Best for**: Quick experimentation, generating `.ply` files

#### PC with NVIDIA GPU (RTX 3060+)
- **Prediction**: ✅ Full CUDA acceleration
- **Rendering**: ✅ Full support via gsplat
- **VRAM**: ~6-8GB recommended
- **Best for**: Complete workflow including video generation

#### Cloud GPU Options
| Provider | GPU | Cost (approx) | Notes |
|----------|-----|---------------|-------|
| Google Colab | T4 (free), A100 (Pro) | Free/$10/mo | Easy setup |
| AWS | g4dn.xlarge (T4) | ~$0.50/hr | Production ready |
| Lambda Labs | A100 | ~$1.10/hr | ML-focused |
| RunPod | RTX 4090 | ~$0.40/hr | Good value |
| Vast.ai | Various | ~$0.20+/hr | Budget option |

### Critical Note from README
```
"While the gaussians prediction works for all CPU, CUDA, and MPS, 
rendering videos via the --render option currently requires a CUDA GPU."
```

---

## 2. Recommended POC (Proof of Concept)

### POC Option A: Quick Local Test (Mac/Any Machine)
**Goal**: Validate the prediction works and produces valid 3DGS

```bash
# 1. Setup
conda create -n sharp python=3.13
conda activate sharp
pip install -r requirements.txt

# 2. Run prediction on a single image (no rendering)
sharp predict -i /path/to/your/photo.jpg -o ./output

# 3. View the .ply file in:
#    - Polycam (free iOS/Mac app)
#    - SuperSplat (online): https://playcanvas.com/supersplat/editor
#    - Luma AI viewer
#    - MeshLab
```

### POC Option B: Full Pipeline (NVIDIA GPU)
**Goal**: End-to-end test with video rendering

```bash
# 1. Setup (same as above)
conda create -n sharp python=3.13
conda activate sharp
pip install -r requirements.txt

# 2. Run prediction WITH rendering
sharp predict -i /path/to/images/ -o ./output --render

# 3. Output: .ply files + .mp4 videos with camera trajectories
```

### POC Option C: Google Colab (Free, Full Features)
**Goal**: Zero setup, full CUDA support

```python
# Run in Google Colab (free GPU)
!git clone https://github.com/apple/ml-sharp.git
%cd ml-sharp
!pip install -r requirements.txt

# Upload your test image
from google.colab import files
uploaded = files.upload()

# Run prediction with rendering
!sharp predict -i /content/your_image.jpg -o /content/output --render

# Download results
files.download('/content/output/')
```

---

## 3. Best POC Scenario for You

### Recommended: "Room Scene Test"
**Why**: Indoor scenes with depth variation show off 3DGS quality best

**Steps**:
1. Take a photo of your room/office with objects at different depths
2. Ensure good lighting and minimal motion blur
3. Run SHARP prediction
4. Evaluate:
   - Novel view quality (parallax effect)
   - Edge sharpness
   - Depth estimation accuracy

### Test Image Criteria
| Good ✅ | Bad ❌ |
|---------|--------|
| Static scene | Moving subjects |
| Objects at varied depths | Flat/2D scene |
| Good lighting | Very dark/overexposed |
| Sharp focus | Motion blur |
| Rich textures | Plain walls only |

### Evaluation Checklist
- [ ] Installation completes without errors
- [ ] Model downloads successfully (~500MB)
- [ ] Prediction completes in <5 seconds (GPU) / <60s (CPU)
- [ ] `.ply` file is valid and opens in viewer
- [ ] Novel views look photorealistic
- [ ] No major artifacts at depth discontinuities

---

## 4. Quick Start Commands

### Your Mac (M1/M4)
```bash
# Install
git clone https://github.com/apple/ml-sharp.git
cd ml-sharp
conda create -n sharp python=3.13 && conda activate sharp
pip install -r requirements.txt

# Test (prediction only, no video)
sharp predict -i ~/Desktop/test_photo.jpg -o ./output

# View output .ply in SuperSplat: https://playcanvas.com/supersplat/editor
```

### Cloud/NVIDIA PC
```bash
# Same install, but with full rendering
sharp predict -i ./my_images/ -o ./output --render

# Outputs both .ply files and .mp4 videos
```

---

## 5. Expected Performance

| Device | Prediction Time | Rendering Time |
|--------|-----------------|----------------|
| NVIDIA RTX 4090 | ~0.8s | ~30s |
| NVIDIA RTX 3080 | ~1.2s | ~45s |
| Apple M4 (MPS) | ~3-5s | N/A |
| Apple M1 (MPS) | ~5-8s | N/A |
| CPU only | ~30-60s | N/A |

---

## 6. Verdict

### For MacBook Air M1/M4
👍 **Viable for prediction** - You can generate 3DGS models  
👎 **No video rendering** - Must use external viewers  
💡 **Tip**: Use SuperSplat web viewer for free real-time viewing

### Recommended Setup for Full POC
1. **Google Colab Free Tier** - Best for initial testing (has free T4 GPU)
2. **RunPod/Vast.ai** - If you need longer sessions (~$0.30-0.50/hr)
3. **Your Mac** - For quick iterations without video rendering

---

## Resources

- [Project Page](https://apple.github.io/ml-sharp/)
- [Paper (arXiv)](https://arxiv.org/abs/2512.10685)
- [GitHub Repo](https://github.com/apple/ml-sharp)
- [SuperSplat Viewer](https://playcanvas.com/supersplat/editor) (free, web-based)
- [Polycam](https://poly.cam/) (iOS/Mac app for viewing 3DGS)
