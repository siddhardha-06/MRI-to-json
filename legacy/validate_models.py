"""
LEGACY (v2 Phase 1). Evaluates the deprecated v1 models.
- Failures are recorded in the output, never silently dropped.
- The regressor is evaluated as deployed (5-slice average) on a held-out set of the shipped
  checkpoint's own split, without subjects that leak into its training part; R2 is reported
  per feature (no pooled raw-unit R2).
- Results go to results/legacy/validate_models/<run>/ with provenance; checkpoints/ is never written.
- The autoencoder and 2D U-Net tests cannot reproduce their training splits any more (their
  file lists included derived files now in work/legacy_derived/, and ixi-t1-masks/ is gone);
  they are kept but run only with --tests ae,unet and report that caveat.

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
ROOT          = Path(__file__).resolve().parents[1]
FAILURES      = []   # {"test", "file", "error"}; written to the results file


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
                FAILURES.append({"test": "autoencoder", "file": str(f), "error": repr(e)})

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
                FAILURES.append({"test": "unet2d", "file": str(mri_path), "error": repr(e)})

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

def shipped_regressor_split(features_gt):
    """Rebuild the file list and seed-42 split of train_regressor.train_regressor (lines 257-291)
    from the feature-file keys, so it no longer depends on which derived files are on disk.
    Returns (train_stems, val_stems). Verified against the checkpoint by the caller."""
    folder_of = {}
    for d in Path(MRI_DIR).iterdir():
        if d.is_dir():
            for f in d.glob("*.nii"):
                folder_of[f.name[:-4]] = d.name
    def raw_stem(stem):
        for suf in ("_n4_n4", "_n4"):
            if stem.endswith(suf):
                return stem[: -len(suf)]
        return stem
    paths = []
    for stem in features_gt:
        if stem.endswith("_seg") or stem.endswith("_seg_seg"):
            continue
        paths.append(Path(MRI_DIR) / folder_of[raw_stem(stem)] / f"{stem}.nii")
    paths = sorted(paths)
    random.seed(42)
    shuffled = list(paths)
    random.shuffle(shuffled)
    k = int(0.8 * len(shuffled))
    return [p.stem for p in shuffled[:k]], [p.stem for p in shuffled[k:]]


def test_regressor(limit=None):
    """Evaluate the SHIPPED regressor as deployed.

    Held-out set: raw scans in the checkpoint's own validation part whose subject has no file
    (raw or _n4 copy) in its training part. Inference: predict_features_from_mri (5-slice
    average, as deployed). Metrics: per-feature R2 and MAE in train-standardised units, summarised
    by mean, median and feature group. No pooled raw-unit R2."""
    print("\n" + "=" * 70)
    print("  TEST 3: FEATURE REGRESSOR (legacy) - prediction accuracy as deployed")
    print("=" * 70)
    sys.path.insert(0, str(ROOT))
    from neurosphere.splits import subject_of
    from train_regressor import FeatureRegressorCNN, predict_features_from_mri

    ckpt_path = os.path.join(CHECKPOINT_DIR, "regressor_best.pth")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    features_gt = json.load(open(FEATURES_FILE))
    train_stems, val_stems = shipped_regressor_split(features_gt)
    mean = np.array([features_gt[s] for s in train_stems]).mean(0)
    std = np.array([features_gt[s] for s in train_stems]).std(0) + 1e-5
    if not (np.allclose(mean, ckpt["feature_mean"], rtol=1e-6) and np.allclose(std, ckpt["feature_std"], rtol=1e-6)):
        raise SystemExit("rebuilt split does not match the checkpoint's stored statistics; refusing to evaluate")
    train_subjects = {subject_of(s) for s in train_stems}
    held_out = [s for s in val_stems if not s.endswith("_n4") and not s.endswith("_n4_n4") and subject_of(s) not in train_subjects]
    excluded = sorted({subject_of(s) for s in val_stems} & train_subjects)
    if limit:
        held_out = held_out[:limit]
    print(f"  Held-out raw subjects: {len(held_out)} (excluded as leaking: {excluded})")

    model = FeatureRegressorCNN(input_channels=1, num_features=ckpt["num_features"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    folder_of = {f.name[:-4]: f for d in Path(MRI_DIR).iterdir() if d.is_dir() for f in d.glob("*.nii")}
    preds, targets, used = [], [], []
    for stem in tqdm(held_out, desc="  Regressor", ncols=80):
        try:
            p = predict_features_from_mri(model, str(folder_of[stem]), ckpt["feature_mean"], ckpt["feature_std"], device=DEVICE)
            preds.append(p); targets.append(features_gt[stem]); used.append(stem)
        except Exception as e:
            FAILURES.append({"test": "regressor", "file": stem, "error": repr(e)})
    P = (np.array(preds) - mean) / std
    T = (np.array(targets) - mean) / std
    ss_res = ((T - P) ** 2).sum(0)
    ss_tot = ((T - T.mean(0)) ** 2).sum(0)
    r2 = np.where(ss_tot > 0, 1 - ss_res / np.where(ss_tot > 0, ss_tot, 1), np.nan)
    mae = np.abs(T - P).mean(0)
    groups = [("Intensity Stats", 0, 12), ("Tissue Seg", 12, 19), ("Texture", 19, 27), ("Structural", 27, 33),
              ("Symmetry", 33, 35), ("Frequency", 35, 38), ("Edge", 38, 40), ("Subcortical Vols", 40, 61)]
    results = {
        "checkpoint": {"file": "regressor_best.pth", "epoch": int(ckpt["epoch"])},
        "held_out_definition": "raw scans in the checkpoint's validation part whose subject has no file in its training part",
        "n_held_out": len(used), "excluded_leaking_subjects": excluded,
        "inference": "predict_features_from_mri: mean of 5 central slices (as deployed)",
        "per_feature_r2": [None if not np.isfinite(v) else float(v) for v in r2],
        "per_feature_mae_std_units": [float(v) for v in mae],
        "r2_mean_over_features": float(np.nanmean(r2)), "r2_median_over_features": float(np.nanmedian(r2)),
        "n_features_r2_below_0": int(np.sum(r2 < 0)),
        "per_group_mean_r2": {g: float(np.nanmean(r2[a:b])) for g, a, b in groups},
        "subjects": used,
    }
    print(json.dumps({k: v for k, v in results.items() if k not in ("per_feature_r2", "per_feature_mae_std_units", "subjects")}, indent=2))
    return results


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser(description="Legacy v1 model validation")
    ap.add_argument("--tests", default="regressor", help="comma list of: ae, unet, regressor")
    ap.add_argument("--limit", type=int, default=None, help="regressor: first N held-out subjects (smoke test)")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from neurosphere.provenance import provenance

    tests = {t.strip() for t in a.tests.split(",")}
    all_results = {}
    if "ae" in tests:
        r = test_autoencoder()
        if r:
            r["caveat"] = "validation files are no longer the training-time split (derived files moved out of ixi-t1/)"
            all_results["autoencoder"] = r
    if "unet" in tests:
        r = test_unet()
        if r:
            all_results["unet2d"] = r
    if "regressor" in tests:
        all_results["regressor"] = test_regressor(limit=a.limit)
    all_results["failures"] = FAILURES
    all_results["provenance"] = provenance(data_subset=a.tests + (f" limit={a.limit}" if a.limit else ""))

    run = time.strftime("%Y%m%d-%H%M%S") + (f"_limit{a.limit}" if a.limit else "")
    base = ROOT / ("work/smoke" if a.limit else "results") / "legacy" / "validate_models" / run
    base.mkdir(parents=True, exist_ok=True)
    with open(base / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2, allow_nan=False)
    print(f"\n  {len(FAILURES)} failures recorded. Results: {base}")
