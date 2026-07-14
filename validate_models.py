"""
Model Accuracy & Validation Test Suite
=======================================
Tests all three trained models on held-out validation data:
  1. Autoencoder  — Reconstruction quality (MSE, SSIM, PSNR)
  2. U-Net        — Segmentation accuracy (Dice, IoU, Per-class Dice)
  3. Regressor    — Feature prediction accuracy (MSE, MAE, R², Per-feature R²)

Uses the SAME deterministic seed(42) train/val splits as training
to ensure we test on data the models never trained on.
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from pathlib import Path
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

MRI_DIR       = "ixi-t1"
MASK_DIR      = "ixi-t1-masks"
FEATURES_FILE = "rule_based_features.json"
CHECKPOINT_DIR = "checkpoints"
IMG_SIZE      = 256
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def load_and_normalize(path):
    """Load NIfTI and apply 1-99 percentile normalization"""
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    img.uncache()
    p1, p99 = np.percentile(data, (1, 99))
    if p99 - p1 > 0:
        data = np.clip(data, p1, p99)
        data = (data - p1) / (p99 - p1 + 1e-5)
    else:
        data = np.zeros_like(data)
    return data


def get_middle_slice_tensor(data, img_size=256):
    """Get middle axial slice as (1,1,H,W) tensor"""
    mid = data.shape[2] // 2
    s = data[:, :, mid]
    t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=(img_size, img_size),
                                        mode='bilinear', align_corners=False)
    return t


def compute_ssim_simple(img1, img2):
    """Simplified SSIM between two numpy arrays in [0,1]"""
    C1 = 0.01**2
    C2 = 0.03**2
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    ssim = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim)


def compute_psnr(mse_val):
    """PSNR from MSE (assuming max pixel = 1.0)"""
    if mse_val < 1e-10:
        return 100.0
    return float(10 * np.log10(1.0 / mse_val))


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: AUTOENCODER
# ═══════════════════════════════════════════════════════════════════════

def test_autoencoder():
    print("\n" + "="*70)
    print("  TEST 1: AUTOENCODER — Reconstruction Quality")
    print("="*70)

    ckpt_path = os.path.join(CHECKPOINT_DIR, "autoencoder_2d_best.pth")
    if not os.path.exists(ckpt_path):
        print("  [SKIP] autoencoder_2d_best.pth not found")
        return None

    # Import model class
    from train_autoencoder import Conv2DAutoencoder

    # Load model
    model = Conv2DAutoencoder(latent_dim=128).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()

    # Get all MRI files
    mri_path = Path(MRI_DIR)
    mri_files = sorted(
        list(mri_path.rglob("*.nii")) + list(mri_path.rglob("*.nii.gz"))
    )
    mri_files = [f for f in mri_files if f.is_file()]

    # Same split as training: seed 42, take last 20%
    random.seed(42)
    shuffled = list(mri_files)
    random.shuffle(shuffled)
    val_start = int(0.8 * len(shuffled))
    val_files = shuffled[val_start:]

    print(f"  Evaluating on {len(val_files)} validation MRI files...")

    mse_list = []
    ssim_list = []

    with torch.no_grad():
        for f in tqdm(val_files[:50], desc="  AE Validation", ncols=80):  # Cap at 50 for speed
            try:
                data = load_and_normalize(f)
                if data.ndim < 3 or data.shape[2] < 3:
                    continue
                x = get_middle_slice_tensor(data, IMG_SIZE).to(DEVICE)
                recon, _z = model(x)
                
                x_np = x.cpu().numpy().squeeze()
                r_np = recon.cpu().numpy().squeeze()
                
                mse = float(((x_np - r_np) ** 2).mean())
                ssim = compute_ssim_simple(x_np, r_np)
                
                mse_list.append(mse)
                ssim_list.append(ssim)
            except Exception as e:
                pass

    avg_mse = np.mean(mse_list)
    avg_ssim = np.mean(ssim_list)
    avg_psnr = compute_psnr(avg_mse)

    results = {
        "val_samples": len(mse_list),
        "mse": round(avg_mse, 6),
        "ssim": round(avg_ssim, 4),
        "psnr_db": round(avg_psnr, 2),
    }

    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  AUTOENCODER RESULTS                │")
    print(f"  ├─────────────────────────────────────┤")
    print(f"  │  Val Samples:  {results['val_samples']:>18}  │")
    print(f"  │  MSE:          {results['mse']:>18.6f}  │")
    print(f"  │  SSIM:         {results['ssim']:>18.4f}  │")
    print(f"  │  PSNR:         {results['psnr_db']:>15.2f} dB  │")
    print(f"  └─────────────────────────────────────┘")

    return results


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: U-NET SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════

def test_unet():
    print("\n" + "="*70)
    print("  TEST 2: U-NET — Segmentation Accuracy")
    print("="*70)

    ckpt_path = os.path.join(CHECKPOINT_DIR, "unet_best.pth")
    if not os.path.exists(ckpt_path):
        print("  [SKIP] unet_best.pth not found")
        return None

    from train_unet import UNet2D

    model = UNet2D(in_channels=1, num_classes=4, base_filters=64).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()

    # Discover MRI-mask pairs (same as training)
    mri_path_dir = Path(MRI_DIR)
    mask_path_dir = Path(MASK_DIR)
    mri_files = sorted(
        list(mri_path_dir.rglob("*.nii.gz")) + list(mri_path_dir.rglob("*.nii"))
    )
    mri_files = [f for f in mri_files if f.is_file()]

    all_pairs = []
    for mri_file in mri_files:
        stem = mri_file.name.replace(".nii.gz", "").replace(".nii", "")
        mask_file = mask_path_dir / f"{stem}_seg.nii"
        if mask_file.exists():
            all_pairs.append((str(mri_file), str(mask_file)))

    # Same split as training: seed 42
    random.seed(42)
    shuffled = list(all_pairs)
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.2))
    val_pairs = shuffled[:n_val]

    print(f"  Evaluating on {len(val_pairs)} validation MRI-mask pairs...")

    class_names = ["Background", "CSF", "Gray Matter", "White Matter"]
    per_class_dice = {c: [] for c in range(4)}
    per_class_iou = {c: [] for c in range(4)}
    pixel_accuracies = []

    with torch.no_grad():
        for mri_path, mask_path in tqdm(val_pairs[:60], desc="  U-Net Validation", ncols=80):
            try:
                # Load MRI — use N4 if available (same as training)
                stem = Path(mri_path).name.replace(".nii.gz", "").replace(".nii", "")
                n4_path = os.path.join(MASK_DIR, "n4", f"{stem}_n4.nii")
                active_path = n4_path if os.path.exists(n4_path) else mri_path

                mri_data = load_and_normalize(active_path)
                mask_data = nib.load(mask_path).get_fdata().astype(np.int64)

                mid = mri_data.shape[2] // 2

                mri_slice = mri_data[:, :, mid]
                mask_slice = mask_data[:, :, mid]

                # Prepare tensors
                x = torch.tensor(mri_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                x = torch.nn.functional.interpolate(x, size=(IMG_SIZE, IMG_SIZE),
                                                     mode='bilinear', align_corners=False).to(DEVICE)

                mask_t = torch.tensor(mask_slice, dtype=torch.long).unsqueeze(0).unsqueeze(0).float()
                mask_t = torch.nn.functional.interpolate(mask_t, size=(IMG_SIZE, IMG_SIZE),
                                                          mode='nearest').squeeze().long()

                logits = model(x)
                preds = torch.argmax(logits, dim=1).squeeze().cpu()
                gt = mask_t.cpu()

                # Pixel accuracy
                correct = (preds == gt).sum().item()
                total = gt.numel()
                pixel_accuracies.append(correct / total)

                # Per-class Dice and IoU
                for c in range(4):
                    pred_c = (preds == c).float()
                    gt_c = (gt == c).float()
                    intersection = (pred_c * gt_c).sum().item()
                    union = pred_c.sum().item() + gt_c.sum().item()

                    if union > 0:
                        dice = (2.0 * intersection) / union
                        iou = intersection / (union - intersection + 1e-8)
                        per_class_dice[c].append(dice)
                        per_class_iou[c].append(iou)
            except Exception as e:
                pass

    avg_pixel_acc = np.mean(pixel_accuracies) if pixel_accuracies else 0
    mean_dice = {}
    mean_iou = {}
    for c in range(4):
        mean_dice[c] = np.mean(per_class_dice[c]) if per_class_dice[c] else 0
        mean_iou[c] = np.mean(per_class_iou[c]) if per_class_iou[c] else 0

    overall_dice = np.mean([mean_dice[c] for c in range(1, 4)])  # Foreground only
    overall_iou = np.mean([mean_iou[c] for c in range(1, 4)])

    results = {
        "val_pairs": len(val_pairs),
        "pixel_accuracy": round(avg_pixel_acc * 100, 2),
        "mean_dice_foreground": round(overall_dice * 100, 2),
        "mean_iou_foreground": round(overall_iou * 100, 2),
        "per_class_dice": {class_names[c]: round(mean_dice[c] * 100, 2) for c in range(4)},
        "per_class_iou": {class_names[c]: round(mean_iou[c] * 100, 2) for c in range(4)},
    }

    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  U-NET SEGMENTATION RESULTS                     │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │  Val Pairs:         {results['val_pairs']:>25}  │")
    print(f"  │  Pixel Accuracy:    {results['pixel_accuracy']:>23.2f}%  │")
    print(f"  │  Mean Dice (FG):    {results['mean_dice_foreground']:>23.2f}%  │")
    print(f"  │  Mean IoU  (FG):    {results['mean_iou_foreground']:>23.2f}%  │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │  Per-Class Dice Scores:                         │")
    for cls_name, dice_val in results["per_class_dice"].items():
        print(f"  │    {cls_name:<20s} {dice_val:>21.2f}%  │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │  Per-Class IoU Scores:                          │")
    for cls_name, iou_val in results["per_class_iou"].items():
        print(f"  │    {cls_name:<20s} {iou_val:>21.2f}%  │")
    print(f"  └─────────────────────────────────────────────────┘")

    return results


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: FEATURE REGRESSOR
# ═══════════════════════════════════════════════════════════════════════

def test_regressor():
    print("\n" + "="*70)
    print("  TEST 3: FEATURE REGRESSOR — Prediction Accuracy")
    print("="*70)

    ckpt_path = os.path.join(CHECKPOINT_DIR, "regressor_best.pth")
    if not os.path.exists(ckpt_path):
        print("  [SKIP] regressor_best.pth not found")
        return None

    if not os.path.exists(FEATURES_FILE):
        print(f"  [SKIP] {FEATURES_FILE} not found")
        return None

    from train_regressor import FeatureRegressorCNN

    # Load features ground truth
    with open(FEATURES_FILE, 'r') as f:
        features_gt = json.load(f)

    # Determine feature count from data
    first_key = list(features_gt.keys())[0]
    num_features = len(features_gt[first_key])
    print(f"  Feature vector size: {num_features}")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    state = ckpt.get("model_state", ckpt)
    
    # Detect feature count from checkpoint
    ckpt_num_features = state["regressor.3.bias"].shape[0]
    print(f"  Checkpoint trained for: {ckpt_num_features} features")
    
    if ckpt_num_features != num_features:
        print(f"  [WARNING] Checkpoint features ({ckpt_num_features}) != ground truth features ({num_features})")
        print(f"  [WARNING] Using checkpoint feature count ({ckpt_num_features}) and truncating ground truth")
        effective_features = ckpt_num_features
    else:
        effective_features = num_features

    model = FeatureRegressorCNN(input_channels=1, num_features=effective_features).to(DEVICE)
    model.load_state_dict(state)
    model.eval()

    # Discover valid files (same logic as training)
    mri_path_dir = Path(MRI_DIR)
    mri_files = sorted(
        list(mri_path_dir.glob("*/*.nii")) + list(mri_path_dir.glob("*/*.nii.gz"))
    )

    valid_files = []
    for f in mri_files:
        if f.stem in features_gt:
            stem_name = f.name.replace(".nii.gz", "").replace(".nii", "")
            n4_path = Path(MASK_DIR) / "n4" / f"{stem_name}_n4.nii"
            if n4_path.exists():
                valid_files.append(f)

    # Same split as training: seed 42, 80/20
    random.seed(42)
    shuffled = list(valid_files)
    random.shuffle(shuffled)
    split_idx = int(0.8 * len(shuffled))
    val_files = shuffled[split_idx:]

    # Compute normalization stats from TRAINING set only (same as training)
    train_files = shuffled[:split_idx]
    train_features = np.array([features_gt[f.stem][:effective_features] for f in train_files])
    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0) + 1e-5

    print(f"  Evaluating on {len(val_files)} validation subjects...")

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for f in tqdm(val_files, desc="  Regressor Validation", ncols=80):
            try:
                stem = f.stem
                gt_vec = np.array(features_gt[stem][:effective_features], dtype=np.float32)

                # Load N4-corrected MRI (same as training domain)
                stem_name = f.name.replace(".nii.gz", "").replace(".nii", "")
                n4_path = Path(MASK_DIR) / "n4" / f"{stem_name}_n4.nii"
                mri_data = load_and_normalize(str(n4_path))
                mid = mri_data.shape[2] // 2

                mri_slice = mri_data[:, :, mid]
                x = torch.tensor(mri_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                x = torch.nn.functional.interpolate(x, size=(IMG_SIZE, IMG_SIZE),
                                                     mode='bilinear', align_corners=False).to(DEVICE)

                pred_norm = model(x).cpu().numpy().squeeze()
                
                # Denormalize predictions
                pred = pred_norm * feature_std + feature_mean

                all_preds.append(pred)
                all_targets.append(gt_vec)
            except Exception as e:
                pass

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Overall metrics
    mse = float(((all_preds - all_targets) ** 2).mean())
    mae = float(np.abs(all_preds - all_targets).mean())

    # R² score (coefficient of determination)
    ss_res = ((all_targets - all_preds) ** 2).sum()
    ss_tot = ((all_targets - all_targets.mean(axis=0)) ** 2).sum()
    r2 = float(1 - ss_res / (ss_tot + 1e-10))

    # Per-feature R²
    per_feat_r2 = []
    for i in range(effective_features):
        ss_r = ((all_targets[:, i] - all_preds[:, i]) ** 2).sum()
        ss_t = ((all_targets[:, i] - all_targets[:, i].mean()) ** 2).sum()
        r2_i = float(1 - ss_r / (ss_t + 1e-10)) if ss_t > 0 else 0.0
        per_feat_r2.append(r2_i)

    # Feature group labels
    feature_groups = [
        ("Intensity Stats", 0, 12),
        ("Tissue Seg", 12, 19),
        ("Texture", 19, 27),
        ("Structural", 27, 33),
        ("Symmetry", 33, 35),
        ("Frequency", 35, 38),
        ("Edge", 38, 40),
    ]
    if effective_features > 40:
        feature_groups.append(("Subcortical Vols", 40, effective_features))

    results = {
        "val_subjects": len(all_preds),
        "features": effective_features,
        "mse": round(mse, 6),
        "mae": round(mae, 6),
        "r2_overall": round(r2, 4),
        "per_group_r2": {},
    }

    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  FEATURE REGRESSOR RESULTS                      │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │  Val Subjects:      {results['val_subjects']:>25}  │")
    print(f"  │  Feature Count:     {results['features']:>25}  │")
    print(f"  │  MSE:               {results['mse']:>25.6f}  │")
    print(f"  │  MAE:               {results['mae']:>25.6f}  │")
    print(f"  │  R² (Overall):      {results['r2_overall']:>25.4f}  │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │  Per-Group Mean R²:                             │")

    for group_name, start, end in feature_groups:
        end = min(end, effective_features)
        group_r2 = np.mean(per_feat_r2[start:end])
        results["per_group_r2"][group_name] = round(group_r2, 4)
        bar_len = int(max(0, group_r2) * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  │    {group_name:<18s} {group_r2:>6.4f}  {bar}  │")

    print(f"  └─────────────────────────────────────────────────┘")

    # Show best and worst individual features
    sorted_feats = sorted(range(len(per_feat_r2)), key=lambda i: per_feat_r2[i], reverse=True)
    
    feat_names_40 = [
        "mean", "std", "min", "max", "median", "p25", "p75", "skewness",
        "kurtosis", "dyn_range", "mode", "bimodality",
        "CSF%", "GM%", "WM%", "brain_total%", "otsu1", "otsu2", "CNR",
        "GLCM_contrast", "GLCM_corr", "GLCM_energy", "GLCM_homo",
        "mean_grad", "std_grad", "max_grad", "entropy",
        "brain_area%", "eccentricity", "solidity", "extent", "perimeter", "vent_proxy%",
        "symmetry", "asym_idx%",
        "low_freq", "high_freq", "freq_ratio",
        "edge_sharp", "edge_std",
    ]
    subcort_names = [
        "L_vent", "R_vent", "vent_AI",
        "L_thal", "R_thal", "thal_AI",
        "L_caud", "R_caud", "caud_AI",
        "L_puta", "R_puta", "puta_AI",
        "L_pall", "R_pall", "pall_AI",
        "L_hippo", "R_hippo", "hippo_AI",
        "L_amyg", "R_amyg", "amyg_AI",
    ]
    all_feat_names = feat_names_40 + subcort_names

    print(f"\n  Top 5 Best Predicted Features:")
    for i in sorted_feats[:5]:
        name = all_feat_names[i] if i < len(all_feat_names) else f"feat_{i}"
        print(f"    #{i:>2d} {name:<18s}  R² = {per_feat_r2[i]:.4f}")

    print(f"\n  Top 5 Worst Predicted Features:")
    for i in sorted_feats[-5:]:
        name = all_feat_names[i] if i < len(all_feat_names) else f"feat_{i}"
        print(f"    #{i:>2d} {name:<18s}  R² = {per_feat_r2[i]:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + "  NEUROSPHERE — MODEL ACCURACY VALIDATION SUITE".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    print(f"\n  Device: {DEVICE}")
    print(f"  MRI Dir: {MRI_DIR}")
    print(f"  Mask Dir: {MASK_DIR}")

    all_results = {}

    # Test 1: Autoencoder
    ae_results = test_autoencoder()
    if ae_results:
        all_results["autoencoder"] = ae_results

    # Test 2: U-Net
    unet_results = test_unet()
    if unet_results:
        all_results["unet"] = unet_results

    # Test 3: Regressor
    reg_results = test_regressor()
    if reg_results:
        all_results["regressor"] = reg_results

    # Save results
    output_path = os.path.join(CHECKPOINT_DIR, "validation_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    # Final summary table
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)

    if "autoencoder" in all_results:
        ae = all_results["autoencoder"]
        print(f"  Autoencoder:   SSIM={ae['ssim']:.4f}   PSNR={ae['psnr_db']:.1f}dB   MSE={ae['mse']:.6f}")

    if "unet" in all_results:
        un = all_results["unet"]
        print(f"  U-Net:         Dice={un['mean_dice_foreground']:.1f}%   IoU={un['mean_iou_foreground']:.1f}%   PixAcc={un['pixel_accuracy']:.1f}%")

    if "regressor" in all_results:
        rg = all_results["regressor"]
        print(f"  Regressor:     R²={rg['r2_overall']:.4f}   MAE={rg['mae']:.6f}   MSE={rg['mse']:.6f}")

    print("\n" + "="*70)
