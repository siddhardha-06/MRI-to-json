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

@lru_cache(maxsize=16)
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
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),     # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),    # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),   # 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),   # 8x8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),   # 4x4
            nn.BatchNorm2d(512),
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
    num_features=61,
    save_dir="checkpoints",
    pretrained_backbone=None,
    limit=None,
    mask_dir="ixi-t1-masks",
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
        stem = f.stem
        # Exclude segmentation mask files
        if stem.endswith("_seg") or stem.endswith("_seg_seg"):
            continue
        if stem in features_gt:
            valid_files.append(f)
                
    if len(valid_files) == 0:
        print("ERROR: No matching MRI-feature pairs found!")
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


# -----------------------------------------------------------------------------
# REMOVED in v2 Phase 1: get_metadata, compute_detailed_tissue_stats,
# get_structural_details, simulate_mriqc_metrics_from_vector and build_structured_json.
# They emitted invented values (slice % x 1400 mL "volumes", formula-generated
# hemisphere means, untested interpretation / pathology rules, hard-coded
# "orientation: axial") and simulated MRIQC metrics. The original code is in git
# history (commit 5edd298). Batch prediction now writes only the predicted vectors.
# -----------------------------------------------------------------------------


def batch_predict_features(model, mri_dir, output_file, feature_mean, feature_std, device=None):
    """Predict the legacy 61-feature vector for all MRI files and save {stem: {"legacy_v1_predicted_vector": [...]}}"""
    
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
            predictions[mri_file.stem] = {"legacy_v1_predicted_vector": [float(x) for x in features]}
            
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
    parser.add_argument("--mask-dir", default="ixi-t1-masks", help="Mask directory containing N4-corrected volumes")
    parser.add_argument("--features-file", default="rule_based_features.json",
                       help="JSON file with rule-based features (ground truth)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-features", type=int, default=61, help="Number of features to predict")
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
            mask_dir=args.mask_dir,
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
        
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            print("\nGenerating clinical reports...")
            try:
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
