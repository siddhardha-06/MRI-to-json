> **Note:** this document still describes the original v1 pipeline. The pipeline evaluated in our paper is the v2 code in `neurosphere/`: run it with `python -m neurosphere run <scan.nii> -o out.json`, or start the dashboard with `python app_server.py`.

# Quick Start Guide

**Goal:** Train MRI models and extract features in 5 steps

---

## Installation (5 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Verify GPU support
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
```

---

## Running the Pipeline

### Option 1: Full Pipeline (Automatic) ⭐

```bash
# Everything from autoencoder → feature extraction
python pipeline.py
```

Expected output:
```
✓ Autoencoder (30-60 min)
✓ U-Net Segmentation (20-40 min)
✓ Feature Regressor (40-80 min)
✓ Features saved to: features/predicted_features.json
```

---

### Option 2: Step-by-Step (Manual Control)

#### Step 1: Train Autoencoder (START HERE - no preprocessing needed)
```bash
python train_autoencoder.py \
  --data-dir ixi-t1 \
  --model-type 2d \
  --epochs 50 \
  --batch-size 8
```

✓ Creates: `checkpoints/autoencoder_2d_best.pth`

---

#### Step 2: (Optional) Generate Segmentation Masks
```bash
# Only needed if training U-Net
pip install antspyx
python train_unet.py --generate-masks --mri-dir ixi-t1 --mask-dir ixi-t1-masks
```

✓ Creates: `ixi-t1-masks/` with segmentation masks

---

#### Step 3: (Optional) Train U-Net
```bash
python train_unet.py \
  --mri-dir ixi-t1 \
  --mask-dir ixi-t1-masks \
  --epochs 50 \
  --batch-size 16
```

✓ Creates: `checkpoints/unet_segmentation_best.pth`

---

#### Step 4: Create Rule-Based Features (Required)

**Generate features from your rule-based pipeline:**

```python
import json
import nibabel as nib
from pathlib import Path

# Example: compute simple statistics as features
features_dict = {}

for nii_file in sorted(Path("ixi-t1").glob("*/*.nii")):
    # Load MRI
    data = nib.load(nii_file).get_fdata()
    
    # Compute features (example: mean, std, min, max per quadrant)
    features = [
        float(data.mean()),
        float(data.std()),
        float(data.min()),
        float(data.max()),
        # Add more features based on your pipeline
    ]
    
    features_dict[nii_file.stem] = features

# Save
with open("rule_based_features.json", "w") as f:
    json.dump(features_dict, f, indent=2)

print(f"✓ Created rule_based_features.json with {len(features_dict)} samples")
```

✓ Creates: `rule_based_features.json`

---

#### Step 5: Train Feature Regressor
```bash
python train_regressor.py \
  --mri-dir ixi-t1 \
  --features-file rule_based_features.json \
  --num-features 64 \
  --epochs 100 \
  --batch-size 32
```

✓ Creates: `checkpoints/regressor_best.pth`

---

#### Step 6: Extract Final Features
```bash
python train_regressor.py \
  --mri-dir ixi-t1 \
  --checkpoint checkpoints/regressor_best.pth \
  --predict
```

✓ Creates: `features/predicted_features.json` ← **FINAL OUTPUT**

---

## Expected Directory Structure After Running

```
c:\Desktop\CCBD\MRI\
├── train_autoencoder.py
├── train_unet.py
├── train_regressor.py
├── pipeline.py
├── README.md
├── QUICKSTART.md
├── requirements.txt
├── ixi-t1/                      (your data)
│   ├── IXI002-Guys-0828-T1.nii/
│   ├── IXI012-HH-1211-T1.nii/
│   └── ... (60+ MRI files)
├── checkpoints/                 (created)
│   ├── autoencoder_2d_best.pth
│   ├── autoencoder_2d_history.json
│   ├── unet_segmentation_best.pth
│   ├── unet_history.json
│   ├── regressor_best.pth
│   └── regressor_history.json
├── features/                    (created)
│   └── predicted_features.json  ← FINAL OUTPUT
├── ixi-t1-masks/                (created by ANTsPy)
│   ├── IXI002-Guys-0828-T1_seg.nii
│   └── ... (segmentation masks)
└── rule_based_features.json     (your ground truth)
```

---

## Output Format

**`features/predicted_features.json`** (final features):
```json
{
  "IXI002-Guys-0828-T1": [0.123, 0.456, ..., 0.789],
  "IXI012-HH-1211-T1": [0.234, 0.567, ..., 0.890],
  "IXI013-HH-1212-T1": [0.345, 0.678, ..., 0.901],
  ...
}
```

Each file has a feature vector of length `--num-features` (default: 64)

---

## Quick Reference Commands

| Task | Command |
|------|---------|
| Install deps | `pip install -r requirements.txt` |
| Check GPU | `python -c "import torch; print(torch.cuda.is_available())"` |
| Train AE | `python train_autoencoder.py` |
| Train U-Net | `python train_unet.py` |
| Train regressor | `python train_regressor.py` |
| Extract features | `python train_regressor.py --predict --checkpoint checkpoints/regressor_best.pth` |
| Full pipeline | `python pipeline.py` |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'torch'` | `pip install -r requirements.txt` |
| `CUDA out of memory` | Reduce `--batch-size` (try 4 or 8) |
| `No .nii files found` | Check `ixi-t1/` directory structure |
| `ANTsPy error` | `pip install --upgrade antspyx` |
| `rule_based_features.json not found` | Generate it first (see Step 4) |

---

## Next Steps After Getting Features

**`features/predicted_features.json` now contains learned features for all MRI files.**

Use for:
1. **Clustering** → Group similar brains
2. **Classification** → Train classifier on features
3. **Visualization** → t-SNE/UMAP projection
4. **Analysis** → Statistical tests, correlations
5. **Model fine-tuning** → Use as input to downstream tasks

---

## Training Times (RTX 3080 GPU)

| Stage | Time | Status |
|-------|------|--------|
| Autoencoder (2D, 50 ep) | 30-60 min | ✓ Fastest |
| SynthSeg preprocessing | 1-2 hours | (optional) |
| U-Net (50 ep) | 20-40 min | (optional) |
| Regressor (100 ep) | 40-80 min | ✓ Recommended |
| **Total minimal** | **60-100 min** | AE + Regressor |
| **Total full** | **2-4 hours** | All 3 models |

---

## Questions?

1. Check [README.md](README.md) for detailed docs
2. Check training history: `cat checkpoints/*_history.json`
3. Verify output: `python -c "import json; print(len(json.load(open('features/predicted_features.json'))))"`

---

**Let's train! Start with:**
```bash
python train_autoencoder.py
```
