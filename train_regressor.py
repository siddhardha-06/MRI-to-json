"""
Feature Regression Model for MRI
Trains to predict features computed from rule-based pipeline
Can leverage autoencoder backbone for better learned representations
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
import nibabel as nib
import numpy as np
from pathlib import Path
import json
import random
from functools import lru_cache

# =============================================================================
# 0. Global Cache & Helper Functions
# =============================================================================

@lru_cache(maxsize=200)
def load_and_preprocess_mri(mri_path):
    """Load MRI NIfTI, apply percentile-clipped normalization, and uncache (shared memory cache)."""
    img = nib.load(mri_path)
    mri_img = img.get_fdata().astype(np.float32)
    img.uncache()
    
    p1, p99 = np.percentile(mri_img, (1, 99))
    if p99 - p1 > 0:
        mri_img = np.clip(mri_img, p1, p99)
        mri_img = (mri_img - p1) / (p99 - p1 + 1e-5)
    else:
        mri_img = np.zeros_like(mri_img)
        
    mri_img.flags.writeable = False  # Make read-only to prevent in-place corruption
    return mri_img

# =============================================================================
# 1. Feature Regression Model
# =============================================================================

class FeatureRegressorCNN(nn.Module):
    """CNN-based regressor to predict features from MRI"""
    
    def __init__(self, input_channels=1, num_features=64, use_pretrained_backbone=None):
        """
        Args:
            input_channels: 1 for grayscale MRI
            num_features: Number of features to predict
            use_pretrained_backbone: Optional path to trained autoencoder .pth for backbone
        """
        super(FeatureRegressorCNN, self).__init__()
        
        # Shared feature extraction backbone (6 convolutional layers matching the autoencoder)
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),      # 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),     # 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),    # 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),   # 16x16
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),   # 8x8
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),   # 4x4
            nn.ReLU(),
        )
        
        # Regression head taking 512 * 4 * 4 = 8192 inputs (preserving spatial information)
        self.regressor = nn.Sequential(
            nn.Linear(8192, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_features),
        )
        
        self.num_features = num_features
        
        # Load pretrained autoencoder backbone if provided
        if use_pretrained_backbone is not None:
            try:
                checkpoint = torch.load(use_pretrained_backbone, map_location="cpu", weights_only=True)
                state_dict = checkpoint.get("model_state", checkpoint)
                
                # Map encoder.X -> backbone.X
                backbone_state = {}
                loaded_keys = []
                for k, v in state_dict.items():
                    if k.startswith("encoder."):
                        new_key = k.replace("encoder.", "backbone.")
                        if new_key in self.state_dict():
                            backbone_state[new_key] = v
                            loaded_keys.append(new_key)
                
                if backbone_state:
                    self.load_state_dict(backbone_state, strict=False)
                    print(f"[SUCCESS] Loaded {len(loaded_keys)} pretrained autoencoder backbone weights from {use_pretrained_backbone}")
                else:
                    print(f"[WARNING] Pretrained backbone load skipped: no matching encoder weights in {use_pretrained_backbone}")
            except Exception as e:
                print(f"[WARNING] Could not load pretrained backbone: {e}")
    
    def forward(self, x):
        # Backbone
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        
        # Regression
        features = self.regressor(x)
        return features


# =============================================================================
# 2. Dataset with Rule-based Ground Truth
# =============================================================================

class MRIFeatureDataset(Dataset):
    """Load MRI + computed features from rule-based pipeline"""
    
    def __init__(self, nii_files, features_file, feature_mean, feature_std, img_size=256, is_train=True):
        """
        Args:
            nii_files: List of paths to NIfTI files
            features_file: JSON file with rule-based feature values
            feature_mean: Numpy array of feature mean values for normalization
            feature_std: Numpy array of feature standard deviation values for normalization
            img_size: Size to resize MRI slices
            is_train: True to enable multi-slice data augmentation
        """
        self.img_size = img_size
        self.is_train = is_train
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.samples = []
        
        # Load rule-based features
        with open(features_file, 'r') as f:
            self.features_gt = json.load(f)
        
        # Match MRI files with features and sample slices
        for mri_file in nii_files:
            mri_file = Path(mri_file)
            stem = mri_file.stem
            if stem in self.features_gt:
                try:
                    img = nib.load(str(mri_file))
                    D = img.shape[2]
                    img.uncache()
                    mid = D // 2
                    
                    if self.is_train:
                        # Extract 5 slices around the middle for training (data augmentation)
                        for offset in [-2, -1, 0, 1, 2]:
                            if 0 <= mid + offset < D:
                                self.samples.append((str(mri_file), stem, mid + offset))
                    else:
                        # Standard evaluation on middle slice
                        self.samples.append((str(mri_file), stem, mid))
                except Exception as e:
                    print(f"  [SKIP] Could not index {mri_file}: {e}")
        
        print(f"Loaded {len(self.samples)} samples (is_train={is_train})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        mri_path, stem, slice_idx = self.samples[idx]
        
        # Load preprocessed MRI volume from cache
        mri_img = load_and_preprocess_mri(mri_path)
        
        mri_slice = mri_img[:, :, slice_idx]
        mri_slice = torch.tensor(mri_slice, dtype=torch.float32).unsqueeze(0)
        mri_slice = torch.nn.functional.interpolate(
            mri_slice.unsqueeze(0), size=(self.img_size, self.img_size),
            mode='bilinear', align_corners=False
        ).squeeze(0)  # (1, img_size, img_size)
        
        # Get ground truth features (normalized using train subset stats)
        features_gt = np.array(self.features_gt[stem], dtype=np.float32)
        features_gt = (features_gt - self.feature_mean) / self.feature_std
        features_gt = torch.tensor(features_gt, dtype=torch.float32)
        
        return mri_slice, features_gt


class VolumeGroupedSampler(Sampler):
    """
    Samples slices grouped by volume so that consecutive items in the
    DataLoader come from the same 3D MRI. This gives ~100% LRU cache
    hit rate instead of ~2% with pure random shuffle.
    """
    def __init__(self, samples: list, shuffle: bool = True):
        from collections import defaultdict
        # Group slice list-indices by MRI file path (first element of tuple)
        vol_to_indices = defaultdict(list)
        for i, (mri_path, _stem, _slice_idx) in enumerate(samples):
            vol_to_indices[mri_path].append(i)
        self.groups = list(vol_to_indices.values())
        self.shuffle = shuffle
        self._len = sum(len(g) for g in self.groups)

    def __iter__(self):
        groups = [g.copy() for g in self.groups]
        if self.shuffle:
            random.shuffle(groups)               # shuffle volume order
            for g in groups:
                random.shuffle(g)                # shuffle slices within volume
                yield from g
        else:
            for g in groups:
                yield from g

    def __len__(self) -> int:
        return self._len


# =============================================================================
# 3. Training Loop
# =============================================================================

def train_regressor(
    mri_dir="ixi-t1",
    features_file="rule_based_features.json",  # From rule-based pipeline
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    num_features=40,
    save_dir="checkpoints",
    pretrained_backbone=None,
    limit=None,
):
    """Train feature regressor"""
    import random
    
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Check if features file exists
    if not Path(features_file).exists():
        print(f"ERROR: Features file not found: {features_file}")
        print("Generate rule-based features first using your pipeline!\n")
        return None, None
    
    # Discover all matching NIfTI files (looks for both .nii and .nii.gz)
    mri_path_dir = Path(mri_dir)
    mri_files = sorted(
        list(mri_path_dir.glob("*/*.nii")) +
        list(mri_path_dir.glob("*/*.nii.gz"))
    )
    
    with open(features_file, 'r') as f:
        features_gt = json.load(f)

    valid_files = []
    for f in mri_files:
        if f.stem in features_gt:
            # Domain alignment: only train regressor on scans that have precomputed N4-corrected
            # volumes (ensures training targets are N4-standardized features with accurate WM%).
            stem_name = f.name.replace(".nii.gz", "").replace(".nii", "")
            n4_path = Path("ixi-t1-masks") / "n4" / f"{stem_name}_n4.nii"
            if n4_path.exists():
                valid_files.append(f)
                
    if len(valid_files) == 0:
        print("ERROR: No matching MRI-feature pairs found with N4-corrected volumes!")
        return None, None
    
    # Subject-level split (deterministic to prevent data leakage)
    random.seed(42)
    shuffled_files = list(valid_files)
    random.shuffle(shuffled_files)
    
    if limit is not None and limit > 0:
        shuffled_files = shuffled_files[:limit]
        print(f"Limiting dataset to first {limit} MRI files.")
        
    split_idx = int(0.8 * len(shuffled_files))
    train_files = shuffled_files[:split_idx]
    val_files = shuffled_files[split_idx:]
    
    print(f"Split: {len(train_files)} train subjects, {len(val_files)} val subjects")
    
    # Compute feature stats from train files ONLY (preventing target stats leakage)
    train_features = np.array([features_gt[f.stem] for f in train_files])
    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0) + 1e-5  # Add epsilon for zero-variance features
    
    # Datasets (augmentation on training set only)
    train_dataset = MRIFeatureDataset(
        train_files, features_file, feature_mean, feature_std, img_size=256, is_train=True
    )
    val_dataset = MRIFeatureDataset(
        val_files, features_file, feature_mean, feature_std, img_size=256, is_train=False
    )
    
    if len(train_dataset) == 0:
        print("ERROR: No training samples found after indexing!")
        return None, None
    
    # Multiplatform workers & pin memory configuration
    num_workers = 0 if os.name == 'nt' else 2
    pin_mem = torch.cuda.is_available()
    
    train_sampler = VolumeGroupedSampler(train_dataset.samples, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem
    )
    
    # Model
    model = FeatureRegressorCNN(
        input_channels=1,
        num_features=num_features,
        use_pretrained_backbone=pretrained_backbone
    ).to(device)
    
    # Loss & optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "lr": []
    }
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            
            # Forward
            pred = model(x)
            loss = criterion(pred, y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.6f}")
        
        train_loss /= max(len(train_loader), 1)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        
        val_loss /= max(len(val_loader), 1)
        scheduler.step()
        
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(scheduler.get_last_lr()[0])
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}\n")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "num_features": num_features,
                "feature_mean": feature_mean.tolist(),
                "feature_std": feature_std.tolist(),
            }
            checkpoint_path = os.path.join(save_dir, "regressor_best.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"[SUCCESS] Saved best model: {checkpoint_path}\n")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Save history
    history_path = os.path.join(save_dir, "regressor_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"[SUCCESS] Training history saved: {history_path}")
    
    return model, history


# =============================================================================
# 4. Predict Features
# =============================================================================

def predict_features_from_mri(model, mri_path, feature_mean, feature_std, device=None):
    """Predict features from MRI using trained regressor by averaging predictions across 5 slices"""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    # Load preprocessed MRI volume from cache
    mri_img = load_and_preprocess_mri(mri_path)
    
    # Extract same 5 slices around the middle as used in training
    D = mri_img.shape[2]
    mid = D // 2
    
    slices_preds = []
    
    for offset in [-2, -1, 0, 1, 2]:
        slice_idx = mid + offset
        if 0 <= slice_idx < D:
            mri_slice = mri_img[:, :, slice_idx]
            mri_slice = torch.tensor(mri_slice, dtype=torch.float32).unsqueeze(0)
            mri_slice = torch.nn.functional.interpolate(
                mri_slice.unsqueeze(0), size=(256, 256),
                mode='bilinear', align_corners=False
            ).to(device)  # (1, 1, 256, 256)
            
            with torch.no_grad():
                pred_norm = model(mri_slice).squeeze().cpu().numpy()
                slices_preds.append(pred_norm)
                
    if not slices_preds:
        features_norm = np.zeros(model.num_features)
    else:
        features_norm = np.mean(slices_preds, axis=0)
    
    # Denormalize with zero-variance safety check (epsilon)
    feature_mean = np.array(feature_mean)
    feature_std = np.array(feature_std)
    feature_std = np.where(feature_std == 0, 1e-5, feature_std)
    features = features_norm * feature_std + feature_mean
    
    return features.tolist()


def get_metadata(mri_path):
    """Extract metadata details from NIfTI file"""
    try:
        img = nib.load(mri_path)
        hdr = img.header
        data = img.get_fdata()
        img.uncache()
        H, W, D = data.shape
        total_pixels = H * W
        bit_depth = int(hdr.get_data_dtype().itemsize * 8)
        return {
            "filename": Path(mri_path).name,
            "modality": "MRI",
            "orientation": "axial",
            "image_height": H,
            "image_width": W,
            "total_pixels": total_pixels,
            "bit_depth": bit_depth,
            "color_space": "grayscale",
            "format": "NIfTI"
        }
    except Exception as e:
        return {
            "filename": Path(mri_path).name,
            "modality": "MRI",
            "error": str(e)
        }

def compute_detailed_tissue_stats(mri_path):
    """Compute mean and std intensities for CSF/GM/WM using intensity thresholds with percentile clipping"""
    try:
        img = nib.load(mri_path)
        data = img.get_fdata().astype(np.float32)
        img.uncache()
        mid = data.shape[2] // 2
        arr = data[:, :, mid]
        
        # Percentile clipping (consistent with model and training)
        p1, p99 = np.percentile(arr, (1, 99))
        if p99 - p1 > 0:
            arr = np.clip(arr, p1, p99)
            arr = (arr - p1) / (p99 - p1 + 1e-5)
        else:
            arr = np.zeros_like(arr)
            
        total = arr.size
        bg_mask = arr < 0.05
        csf_mask = (arr >= 0.05) & (arr < 0.25)
        gm_mask = (arr >= 0.25) & (arr < 0.55)
        wm_mask = arr >= 0.55
        
        brain_total = csf_mask.sum() + gm_mask.sum() + wm_mask.sum()
        if brain_total == 0:
            brain_total = 1.0
            
        def class_stats(mask, denom):
            pixels = int(mask.sum())
            percentage = round(float((pixels / denom) * 100), 2)
            values = arr[mask]
            mean_val = round(float(np.mean(values)), 4) if pixels > 0 else 0.0
            std_val = round(float(np.std(values)), 4) if pixels > 0 else 0.0
            return {
                "pixel_count": pixels,
                "percentage": percentage,
                "mean_intensity": mean_val,
                "std_intensity": std_val
            }
        return {
            "background": class_stats(bg_mask, total),
            "csf_ventricles": class_stats(csf_mask, brain_total),
            "gray_matter": class_stats(gm_mask, brain_total),
            "white_matter": class_stats(wm_mask, brain_total),
            "brain_tissue_total_pct": round(float(((gm_mask.sum() + wm_mask.sum()) / brain_total) * 100), 2)
        }
    except Exception:
        return {}

def get_structural_details(mri_path):
    """Extract coordinates and centroids of the brain mask"""
    try:
        from skimage import measure
        img = nib.load(mri_path)
        data = img.get_fdata().astype(np.float32)
        img.uncache()
        mid = data.shape[2] // 2
        arr = data[:, :, mid]
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max - arr_min > 0:
            arr = (arr - arr_min) / (arr_max - arr_min)
        brain_mask = arr > 0.05
        labeled = measure.label(brain_mask)
        regions = measure.regionprops(labeled)
        if regions:
            brain = max(regions, key=lambda r: r.area)
            min_r, min_c, max_r, max_c = [int(x) for x in brain.bbox]
            centroid_r, centroid_c = [round(float(x), 2) for x in brain.centroid]
            return {
                "brain_area_pixels": int(brain.area),
                "brain_area_pct": round(float((brain.area / arr.size) * 100), 2),
                "bounding_box": {
                    "min_row": min_r,
                    "min_col": min_c,
                    "max_row": max_r,
                    "max_col": max_c
                },
                "centroid": {
                    "row": centroid_r,
                    "col": centroid_c
                }
            }
    except ImportError as e:
        print(f"[WARNING] scikit-image is not installed, structural details will be empty. Error: {e}")
    except Exception:
        pass
    return {
        "brain_area_pixels": 0,
        "brain_area_pct": 0.0,
        "bounding_box": {"min_row": 0, "min_col": 0, "max_row": 0, "max_col": 0},
        "centroid": {"row": 0.0, "col": 0.0}
    }


def simulate_mriqc_metrics_from_vector(vector) -> dict:
    """
    Simulate realistic MRIQC metrics deterministically from a 40-element feature vector
    in case the original 3D volume is not available.
    """
    mean_val = float(vector[0])
    std_val = float(vector[1])
    cnr_val = float(vector[18])
    entropy_val = float(vector[26])
    area_pct = float(vector[27])
    
    # 1. CJV: standard is (std_wm + std_gm) / |mean_wm - mean_gm|
    cjv = 0.32 + (std_val / (mean_val + 1e-5)) * 0.15
    cjv = np.clip(cjv, 0.25, 0.65)
    
    # 2. CNR: contrast-to-noise
    cnr = cnr_val if cnr_val > 0 else 1.85
    cnr = np.clip(cnr, 0.5, 4.0)
    
    # 3. EFC: Entropy Focus Criterion
    efc = 0.65 + (entropy_val / 5.0) * 0.2
    efc = np.clip(efc, 0.55, 0.92)
    
    # 4. FBER: Foreground-to-Background Energy Ratio
    fber = 25.0 + (area_pct * 3.5) + (mean_val / (std_val + 1e-5)) * 10.0
    fber = np.clip(fber, 12.0, 850.0)
    
    # 5. SNRs
    snr_base = mean_val / (std_val + 1e-5)
    snr_wm = 9.5 + snr_base * 1.5
    snr_wm = np.clip(snr_wm, 8.0, 14.5)
    
    snr_gm = 7.5 + snr_base * 1.0
    snr_gm = np.clip(snr_gm, 6.0, 11.5)
    
    snr_csf = 2.5 + snr_base * 0.3
    snr_csf = np.clip(snr_csf, 1.2, 4.5)
    
    # 6. QI_1: Ghosting metric
    qi_1 = 0.004 + (1.0 - (cnr / 4.0)) * 0.015
    qi_1 = np.clip(qi_1, 0.0001, 0.04)
    
    return {
        "cjv": round(float(cjv), 4),
        "cnr": round(float(cnr), 4),
        "efc": round(float(efc), 4),
        "fber": round(float(fber), 2),
        "snr_wm": round(float(snr_wm), 4),
        "snr_gm": round(float(snr_gm), 4),
        "snr_csf": round(float(snr_csf), 4),
        "qi_1": round(float(qi_1), 6)
    }


def build_structured_json(vector, mri_path=None):
    """Build structured diagnostic JSON matching sample_output.json schema (updated for 40 features)"""
    if len(vector) < 40:
        vector = list(vector) + [0.0] * (40 - len(vector))
        
    mriqc_metrics = None
        
    metadata = {
        "filename": Path(mri_path).name if mri_path else "unknown_image.nii",
        "modality": "MRI",
        "orientation": "axial",
        "image_height": 256,
        "image_width": 256,
        "total_pixels": 65536,
        "bit_depth": 16,
        "color_space": "grayscale",
        "format": "NIfTI"
    }
    if mri_path and os.path.exists(mri_path):
        metadata = get_metadata(mri_path)
        
    tissue_seg = {
        "background": {"pixel_count": 0, "percentage": 0.0, "mean_intensity": 0.0, "std_intensity": 0.0},
        "csf_ventricles": {"pixel_count": 0, "percentage": round(float(vector[12]), 2), "mean_intensity": 0.0, "std_intensity": 0.0},
        "gray_matter": {"pixel_count": 0, "percentage": round(float(vector[13]), 2), "mean_intensity": 0.0, "std_intensity": 0.0},
        "white_matter": {"pixel_count": 0, "percentage": round(float(vector[14]), 2), "mean_intensity": 0.0, "std_intensity": 0.0},
        "brain_tissue_total_pct": round(float(vector[15]), 2),
        "otsu_threshold_1": round(float(vector[16]), 4),
        "otsu_threshold_2": round(float(vector[17]), 4),
        "gm_wm_cnr": round(float(vector[18]), 4)
    }
    if mri_path and os.path.exists(mri_path):
        real_tissue = compute_detailed_tissue_stats(mri_path)
        if real_tissue:
            for key in ["background", "csf_ventricles", "gray_matter", "white_matter"]:
                if key in real_tissue:
                    tissue_seg[key]["pixel_count"] = real_tissue[key]["pixel_count"]
                    tissue_seg[key]["mean_intensity"] = real_tissue[key]["mean_intensity"]
                    tissue_seg[key]["std_intensity"] = real_tissue[key]["std_intensity"]
                    if key != "background" and tissue_seg[key]["percentage"] == 0.0:
                        tissue_seg[key]["percentage"] = real_tissue[key]["percentage"]
                        
    struct_det = {
        "brain_area_pixels": 0,
        "brain_area_pct": round(float(vector[27]), 2),
        "bounding_box": {"min_row": 0, "min_col": 0, "max_row": 0, "max_col": 0},
        "centroid": {"row": 0.0, "col": 0.0}
    }
    if mri_path and os.path.exists(mri_path):
        real_struct = get_structural_details(mri_path)
        struct_det["brain_area_pixels"] = real_struct["brain_area_pixels"]
        struct_det["bounding_box"] = real_struct["bounding_box"]
        struct_det["centroid"] = real_struct["centroid"]
        if struct_det["brain_area_pct"] == 0.0:
            struct_det["brain_area_pct"] = real_struct["brain_area_pct"]
            
    struct_det["eccentricity"] = round(float(vector[28]), 4)
    struct_det["solidity"] = round(float(vector[29]), 4)
    struct_det["extent"] = round(float(vector[30]), 4)
    struct_det["perimeter"] = round(float(vector[31]), 1)
    struct_det["ventricular_proxy_pct"] = round(float(vector[32]), 2)
    
    sol_val = struct_det["solidity"]
    ecc_val = struct_det["eccentricity"]
    struct_det["interpretation"] = {
        "solidity_close_to_1": "compact, healthy brain shape" if sol_val > 0.8 else "atrophy or tissue degradation present",
        "low_solidity": "possible irregular mass or artifact" if sol_val < 0.8 else "normal compact boundary",
        "eccentricity_near_0": "circular (axial slice normal)" if ecc_val < 0.7 else "ellipsoid / stretched shape",
        "eccentricity_near_1": "elongated, check orientation" if ecc_val > 0.7 else "standard slice orientation"
    }

    left_mean = round(float(vector[0] * (1 + vector[34]/200)), 4)
    right_mean = round(float(vector[0] * (1 - vector[34]/200)), 4)
    asym_pct = round(float(vector[34]), 2)
    
    symmetry_score = round(float(vector[33]), 4)
    sym_interpretation = "High symmetry — no asymmetry detected" if asym_pct < 5.0 else "Mild asymmetry detected — review structural alignment"
    
    total_flags = 0
    flags = []
    risk_level = "low"
    
    if asym_pct > 10.0:
        total_flags += 1
        flags.append("Significant Bilateral Asymmetry (check for focal lesions or unilateral atrophy)")
    if struct_det["solidity"] < 0.8:
        total_flags += 1
        flags.append("Abnormally Low Shape Solidity (possible mass effect or severe CSF ventricular expansion)")
    if tissue_seg["csf_ventricles"]["percentage"] > 40.0:
        total_flags += 1
        flags.append("Ventriculomegaly indicator (highly enlarged ventricles)")
        
    if total_flags >= 2:
        risk_level = "high"
    elif total_flags == 1:
        risk_level = "medium"

    return {
        "schema_version": "1.1",
        "scan_metadata": metadata,
        "intensity_statistics": {
            "mean": round(float(vector[0]), 4),
            "std": round(float(vector[1]), 4),
            "min": round(float(vector[2]), 4),
            "max": round(float(vector[3]), 4),
            "median": round(float(vector[4]), 4),
            "percentile_25": round(float(vector[5]), 4),
            "percentile_75": round(float(vector[6]), 4),
            "skewness": round(float(vector[7]), 4),
            "kurtosis": round(float(vector[8]), 4),
            "dynamic_range": round(float(vector[9]), 4),
            "mode": round(float(vector[10]), 4),
            "bimodality": round(float(vector[11]), 4)
        },
        "tissue_segmentation": tissue_seg,
        "texture_features": {
            "glcm_contrast": round(float(vector[19]), 4),
            "glcm_correlation": round(float(vector[20]), 4),
            "glcm_energy": round(float(vector[21]), 4),
            "glcm_homogeneity": round(float(vector[22]), 4),
            "mean_gradient": round(float(vector[23]), 4),
            "std_gradient": round(float(vector[24]), 4),
            "max_gradient": round(float(vector[25]), 4),
            "shannon_entropy": round(float(vector[26]), 4),
            "interpretation": {
                "high_entropy_means": "complex brain texture (generally healthy)",
                "low_entropy_means": "uniform / possibly atrophied regions",
                "high_contrast_means": "strong tissue boundaries present",
                "high_homogeneity_means": "smooth, uniform intensity regions"
            }
        },
        "structural_features": struct_det,
        "symmetry": {
            "bilateral_symmetry_score": symmetry_score,
            "left_hemisphere_mean": left_mean,
            "right_hemisphere_mean": right_mean,
            "asymmetry_index_pct": asym_pct,
            "interpretation": sym_interpretation
        },
        "frequency_domain": {
            "low_frequency_energy": round(float(vector[35]), 4),
            "high_frequency_energy": round(float(vector[36]), 4),
            "frequency_energy_ratio": round(float(vector[37]), 4)
        },
        "edge_features": {
            "cortical_edge_sharpness": round(float(vector[38]), 4),
            "cortical_edge_std": round(float(vector[39]), 4)
        },
        "pathology_indicators": {
            "total_flags": total_flags,
            "risk_level": risk_level,
            "flags": flags,
            "disclaimer": "These are AI training indicators only, NOT clinical diagnoses."
        },
        "mriqc_metrics": mriqc_metrics
    }


def batch_predict_features(model, mri_dir, output_file, feature_mean, feature_std, device=None):
    """Predict features for all MRI files and save in structured diagnostic JSON format"""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    predictions = {}
    
    # Accept either single directory or specific single file for prediction
    if os.path.isfile(mri_dir):
        mri_files = [Path(mri_dir)]
    else:
        mri_files = sorted(
            list(Path(mri_dir).glob("*/*.nii")) +
            list(Path(mri_dir).glob("*/*.nii.gz"))
        )
    
    with torch.no_grad():
        for mri_file in mri_files:
            features = predict_features_from_mri(
                model, str(mri_file), feature_mean, feature_std, device
            )
            # Create structured diagnostic representation matching sample_output.json
            predictions[mri_file.stem] = build_structured_json(features, str(mri_file))
            
    # Save to JSON
    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, 'w') as f:
        # If predicting a single file, output its dictionary directly to match sample_output.json
        if len(predictions) == 1:
            single_key = list(predictions.keys())[0]
            json.dump(predictions[single_key], f, indent=2)
        else:
            json.dump(predictions, f, indent=2)
    
    print(f"[SUCCESS] Predicted features for {len(predictions)} MRI files")
    print(f"[SUCCESS] Predictions saved to: {output_file}")
    
    return predictions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Feature Regressor")
    parser.add_argument("--mri-dir", default="ixi-t1", help="MRI dataset directory")
    parser.add_argument("--features-file", default="rule_based_features.json",
                       help="JSON file with rule-based features (ground truth)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-features", type=int, default=40, help="Number of features to predict")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--pretrained-backbone", help="Path to pretrained autoencoder")
    parser.add_argument("--predict", action="store_true", help="Predict features on new data")
    parser.add_argument("--checkpoint", help="Path to trained model checkpoint")
    
    parser.add_argument("--limit", type=int, default=None, help="Limit number of MRI files to use for training")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Feature Regression Training")
    print("=" * 70)
    print(f"MRI dir: {args.mri_dir}")
    print(f"Features file (GT): {args.features_file}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num features: {args.num_features}")
    if args.limit:
        print(f"Dataset limit: {args.limit}")
    print("=" * 70 + "\n")
    
    if not args.predict:
        # Training
        model, history = train_regressor(
            mri_dir=args.mri_dir,
            features_file=args.features_file,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_features=args.num_features,
            save_dir=args.save_dir,
            pretrained_backbone=args.pretrained_backbone,
            limit=args.limit,
        )
    else:
        # Prediction with checkpoint
        if not args.checkpoint:
            print("ERROR: --checkpoint required for prediction")
            exit(1)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model = FeatureRegressorCNN(
            input_channels=1,
            num_features=checkpoint["num_features"]
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        
        # Predict
        output_file = os.path.join(args.save_dir, "predicted_features.json")
        predictions = batch_predict_features(
            model, args.mri_dir, output_file,
            checkpoint["feature_mean"], checkpoint["feature_std"], device
        )
        
        # Automatically generate clinical reports if GEMINI_API_KEY is available (or fallback)
        api_key = os.environ.get("GEMINI_API_KEY", "REDACTED_API_KEY")
        if api_key:
            print("\nGenerating clinical reports...")
            try:
                # Ensure child imports and processes get the fallback key if not set in environment
                if "GEMINI_API_KEY" not in os.environ:
                    os.environ["GEMINI_API_KEY"] = api_key
                from generate_report import process_all_patients, process_single_patient
                reports_dir = "reports"
                if os.path.isfile(args.mri_dir):
                    patient_id = Path(args.mri_dir).stem
                    process_single_patient(output_file, patient_id, reports_dir)
                else:
                    process_all_patients(output_file, reports_dir)
            except Exception as e:
                print(f"[ERROR] Could not generate clinical reports automatically: {e}")
        else:
            print("\n[INFO] GEMINI_API_KEY not set. Skipping automatic report generation.")
