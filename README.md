# MRI → Trained Model → Features → JSON Pipeline

Complete PyTorch-based pipeline for training neural networks on IXI MRI data and extracting learned features.

## Pipeline Overview

```
IXI MRI Files (.nii)
    ↓
[Step 1] ANTsPy Atropos → Pseudo-labels (GM/WM/CSF masks)
    ↓
[Step 2] Train Autoencoder (self-supervised)  ← START HERE
    ↓
[Step 3] Train U-Net (with pseudo-labels)
    ↓
[Step 4] Train Feature Regressor (with rule-based GT)
    ↓
[Step 5] Extract Final Features → JSON
```

---

## Model Training Order & Rationale

### **Model 1: Autoencoder (Vision Encoder)** ✓ FIRST
- **Why first?**
  - Self-supervised — no labels needed
  - Trains immediately on IXI data
  - Learns general MRI representations
  - Provides learned features for downstream tasks
  
- **Training:** MRI slice → Reconstruction
- **Output:** Latent feature representations (128-256 dims)
- **File:** `train_autoencoder.py`
- **Time:** ~30-60 min on GPU (50 epochs)

### **Model 2: U-Net Segmentation** SECOND
- **Why second?**
  - Requires ANTsPy pseudo-labels (preprocessing step)
  - Uses autoencoder backbone (optional improvement)
  - Segments GM/WM/CSF regions
  
- **Training:** MRI → GM/WM/CSF segmentation masks
- **Requires:** ANTsPy preprocessing (Step 1)
- **File:** `train_unet.py`
- **Time:** ~20-40 min on GPU (50 epochs)

### **Model 3: Feature Regressor** THIRD
- **Why third?**
  - Depends on rule-based pipeline output
  - Can leverage autoencoder features
  - Directly predicts features from MRI
  
- **Training:** MRI → Predicted features
- **Requires:** Rule-based features JSON as ground truth
- **File:** `train_regressor.py`
- **Time:** ~40-80 min on GPU (100 epochs)

---

## Installation

### Prerequisites
```bash
# Python 3.8+
pip install torch torchvision torchaudio
pip install nibabel numpy scipy
pip install antspyx  # For pseudo-label generation (ANTsPy)
```

### Verify Installation
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## Quick Start

### Option A: Run Complete Pipeline
```bash
python pipeline.py
```

### Option B: Train Individual Models
```bash
# 1. Train autoencoder first
python train_autoencoder.py --epochs 50 --batch-size 8

# 2. (Optional) Train U-Net
python pipeline.py --step 3  # Requires ANTsPy masks

# 3. Train regressor (requires rule_based_features.json)
python train_regressor.py --epochs 100 --batch-size 32
```

---

## Detailed Usage

### Step 0: Prepare Data

Download the IXI T1 dataset using the Kaggle CLI:

```bash
# Install Kaggle CLI
pip install kaggle

# Download the dataset zip file
kaggle datasets download -d kbacon/ixi-t1

# Unzip the downloaded file into the ixi-t1 directory
unzip ixi-t1.zip -d ixi-t1/
```

Your workspace should have:
├── ixi-t1/
│   ├── IXI002-Guys-0828-T1.nii/
│   │   └── IXI002-Guys-0828-T1.nii
│   ├── IXI012-HH-1211-T1.nii/
│   │   └── IXI012-HH-1211-T1.nii
│   ├── ... (more IXI files)
```

### Step 1: Autoencoder Training (Self-Supervised)

**Start here** — no preprocessing needed!

```bash
python train_autoencoder.py \
  --data-dir ixi-t1 \
  --model-type 2d \           # or 3d
  --epochs 50 \
  --batch-size 8 \
  --latent-dim 128 \
  --save-dir checkpoints
```

**Output:**
- `checkpoints/autoencoder_2d_best.pth` — Trained model
- `checkpoints/autoencoder_2d_history.json` — Training history
- `autoencoder_2d_features.json` — Extracted latent features

**Parameters:**
- `--model-type 2d`: Slice-based (faster, ≈256×256)
- `--model-type 3d`: Volume-based (slower, ≈128×128×128)
- `--latent-dim`: Feature vector size (64-256)
- `--batch-size`: GPU memory dependent (8-32)

---

### Step 2: (Optional) Generate ANTsPy Pseudo-Labels

Required only for U-Net training.

```bash
# Install ANTsPy
pip install antspyx

# Generate masks
python pipeline.py --step 1

# Or manually:
python train_unet.py --generate-masks --mri-dir ixi-t1 --mask-dir ixi-t1-masks
```

**Output:**
- `ixi-t1-masks/` folder with `*_seg.nii` files
- Segmentation masks: 0=background, 1=GM, 2=WM, 3=CSF

---

### Step 3: U-Net Segmentation Training

Requires ANTsPy masks (Step 2).

```bash
python train_unet.py \
  --mri-dir ixi-t1 \
  --mask-dir ixi-t1-masks \
  --epochs 50 \
  --batch-size 16 \
  --save-dir checkpoints
```

**Output:**
- `checkpoints/unet_segmentation_best.pth`
- `checkpoints/unet_history.json`

---

### Step 4: Create Rule-Based Ground Truth

Your rule-based pipeline should generate:

**Format:** `rule_based_features.json`
```json
{
  "IXI002-Guys-0828-T1": [0.45, 0.32, 0.23, 0.15, ...],
  "IXI012-HH-1211-T1": [0.47, 0.31, 0.22, 0.18, ...],
  ...
}
```

**Generate via your pipeline:**
```python
# Pseudo-code for your rule-based pipeline
features_dict = {}
for mri_file in get_all_mri_files():
    # Extract features (GM/WM/CSF volumes, cortical thickness, etc.)
    features = compute_features(mri_file)
    features_dict[mri_file.stem] = features.tolist()

# Save
import json
with open("rule_based_features.json", "w") as f:
    json.dump(features_dict, f, indent=2)
```

---

### Step 5: Feature Regressor Training

Requires `rule_based_features.json`.

```bash
python train_regressor.py \
  --mri-dir ixi-t1 \
  --features-file rule_based_features.json \
  --num-features 64 \
  --epochs 100 \
  --batch-size 32 \
  --save-dir checkpoints
```

**Output:**
- `checkpoints/regressor_best.pth`
- `checkpoints/regressor_history.json`

**Features:**
- 80/20 train/val split
- Early stopping (patience=10)
- Feature normalization
- MSE loss with L2 regularization

---

### Step 6: Extract Final Features

```bash
python train_regressor.py \
  --mri-dir ixi-t1 \
  --checkpoint checkpoints/regressor_best.pth \
  --predict \
  --save-dir features
```

**Output:**
- `features/predicted_features.json` — Predictions for all MRI files
- Same structure as rule-based GT

---

## Full Pipeline Execution

```bash
# Run everything in order (with error checking)
python pipeline.py \
  --mri-dir ixi-t1 \
  --features-file rule_based_features.json \
  --ae-epochs 50 \
  --unet-epochs 50 \
  --regressor-epochs 100 \
  --output-dir features
```

**Options:**
- `--step 2` — Run only step 2 (autoencoder)
- `--skip-unet` — Skip U-Net, only train AE + regressor
- `--skip-synthseg` — Skip ANTsPy pseudo-label generation step
- `--autoencoder-type 3d` — Use 3D model

---

## Model Architectures

### Autoencoder (2D)
```
Encoder:  256×256 → Conv(32) → Conv(64) → Conv(128) → Conv(256) → Latent(128)
Decoder:  Latent(128) → ConvT(256) → ConvT(128) → ConvT(64) → ConvT(32) → 256×256
Loss:     MSE (reconstruction)
```

### Autoencoder (3D)
```
Encoder:  128×128×128 → Conv3d layers → Latent(256)
Decoder:  Latent(256) → ConvT3d layers → 128×128×128
Loss:     MSE
```

### U-Net (2D)
```
Input:    256×256 MRI slice
Encoder:  Conv → Pool → Conv → Pool → Conv → Pool (skip connections)
Bottleneck: Conv(512)
Decoder:  ConvT + Concatenate skip connections → Output 3-class segmentation
Loss:     CrossEntropy with class weights [0.1, 0.5, 0.4] (background, GM, WM, CSF)
```

### Feature Regressor (CNN)
```
Input:    256×256 MRI slice
Backbone: Conv(32) → Conv(64) → Conv(128) → Conv(256) → AdaptiveAvgPool
Head:     Linear(256→128) → ReLU → Dropout → Linear(128→num_features)
Loss:     MSE with L2 regularization
```

---

## Output Files

After running the complete pipeline:

```
checkpoints/
├── autoencoder_2d_best.pth          # Trained autoencoder
├── autoencoder_2d_history.json      # Training metrics
├── unet_segmentation_best.pth       # Trained U-Net
├── unet_history.json
├── regressor_best.pth               # Trained regressor
└── regressor_history.json

features/
└── predicted_features.json          # ← FINAL OUTPUT

autoencoder_2d_features.json        # Latent features from AE
rule_based_features.json            # Your ground truth
```

---

## Hyperparameters Reference

| Model | Epochs | Batch Size | Learning Rate | Device |
|-------|--------|-----------|---------------|--------|
| Autoencoder | 50-100 | 8-16 | 1e-3 | GPU |
| U-Net | 50-100 | 16-32 | 1e-3 | GPU |
| Regressor | 100-200 | 32-64 | 1e-3 | GPU |

**Tips:**
- Start with defaults, adjust if overfitting
- Increase batch size if GPU memory allows
- Reduce learning rate for finer tuning
- More epochs for better convergence (with early stopping)

---

## Troubleshooting

### No CUDA/GPU detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False: Install CUDA-compatible PyTorch
```

### Out of memory
```bash
# Reduce batch size
python train_autoencoder.py --batch-size 4

# Or use 3D model instead (processes fewer slices)
python train_autoencoder.py --model-type 3d --batch-size 4
```

### No MRI files found
```bash
# Check directory structure
ls ixi-t1/IXI002-Guys-0828-T1.nii/
# Should contain: IXI002-Guys-0828-T1.nii
```

### ANTsPy masks missing

```bash
# Generate them:
python train_unet.py --generate-masks --mri-dir ixi-t1 --mask-dir ixi-t1-masks
# Then retry U-Net training
```

### Features file not found
```bash
# Generate rule-based features first
python compute_rule_based_features.py  # Your script
# Creates: rule_based_features.json
```

---

## Performance Estimates

| Model | GPU | Time | Memory |
|-------|-----|------|--------|
| Autoencoder (2D, 50 ep) | RTX 3080 | 30-60 min | 4 GB |
| U-Net (50 ep) | RTX 3080 | 20-40 min | 6 GB |
| Regressor (100 ep) | RTX 3080 | 40-80 min | 4 GB |
| **Total** | - | **2-3 hours** | - |

---

## Next Steps

1. ✓ Train autoencoder (self-supervised)
2. ✓ Generate rule-based features
3. ✓ Train feature regressor
4. ✓ Extract final features → JSON
5. Use features for:
   - Downstream analysis
   - Classification/regression tasks
   - Clustering
   - Visualization/dimensionality reduction

---

## Files Overview

| File | Purpose |
|------|---------|
| `train_autoencoder.py` | Self-supervised feature learning |
| `train_unet.py` | Segmentation with pseudo-labels |
| `train_regressor.py` | Feature prediction from MRI |
| `pipeline.py` | Master orchestration script |
| `LLM-pipeline.py` | (Your existing pipeline) |
| `first-pipeline.py` | (Your existing pipeline) |

---

## Citation & References

- **Nibabel:** NIfTI file handling
- **PyTorch:** Deep learning framework
- **ANTsPy:** Atropos tissue segmentation & bias correction (Avants et al.)

---

## Questions?

Check the generated `.json` files and `.pth` checkpoints for:
- Training curves in `*_history.json`
- Model weights in `*_best.pth`
- Feature predictions in `predicted_features.json`

Happy training! 🚀
