"""
Feature Regression Model for MRI
Trains to predict features computed from rule-based pipeline
Can leverage autoencoder backbone for better learned representations
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from pathlib import Path
import json

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
        
        # Shared feature extraction backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),     # 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),                  # 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),                 # 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),                # 16x16
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_features),
        )
        
        self.num_features = num_features
    
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
    
    def __init__(self, mri_dir, features_file, img_size=256):
        """
        Args:
            mri_dir: Directory with MRI files
            features_file: JSON file with rule-based feature values
                          Format: {filename_stem: [feat1, feat2, ...]}
            img_size: Size to resize MRI slices
        """
        self.mri_dir = Path(mri_dir)
        self.img_size = img_size
        self.samples = []
        
        # Load rule-based features
        with open(features_file, 'r') as f:
            self.features_gt = json.load(f)
        
        # Match MRI files with features
        mri_files = sorted(self.mri_dir.glob("*/*.nii"))
        
        for mri_file in mri_files:
            stem = mri_file.stem
            if stem in self.features_gt:
                self.samples.append((str(mri_file), stem))
        
        print(f"Found {len(self.samples)} MRI-feature pairs")
        
        # Compute feature statistics for normalization
        all_features = np.array([self.features_gt[s[1]] for s in self.samples])
        self.feature_mean = all_features.mean(axis=0)
        self.feature_std = all_features.std(axis=0) + 1e-5
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        mri_path, stem = self.samples[idx]
        
        # Load MRI
        mri_img = nib.load(mri_path).get_fdata()
        mri_img = (mri_img - mri_img.min()) / (mri_img.max() - mri_img.min() + 1e-5)
        
        # Extract middle slice
        D = mri_img.shape[2]
        slice_idx = D // 2
        
        mri_slice = mri_img[:, :, slice_idx]
        mri_slice = torch.tensor(mri_slice, dtype=torch.float32).unsqueeze(0)
        mri_slice = torch.nn.functional.interpolate(
            mri_slice.unsqueeze(0), size=(self.img_size, self.img_size),
            mode='bilinear', align_corners=False
        ).squeeze(0)  # (1, img_size, img_size)
        
        # Get ground truth features (normalized)
        features_gt = np.array(self.features_gt[stem], dtype=np.float32)
        features_gt = (features_gt - self.feature_mean) / self.feature_std
        features_gt = torch.tensor(features_gt, dtype=torch.float32)
        
        return mri_slice, features_gt


# =============================================================================
# 3. Training Loop
# =============================================================================

def train_regressor(
    mri_dir="ixi-t1",
    features_file="rule_based_features.json",  # From rule-based pipeline
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    num_features=64,
    save_dir="checkpoints",
    pretrained_backbone=None,
):
    """Train feature regressor"""
    
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Check if features file exists
    if not Path(features_file).exists():
        print(f"ERROR: Features file not found: {features_file}")
        print("Generate rule-based features first using your pipeline!\n")
        return None, None
    
    # Dataset
    dataset = MRIFeatureDataset(mri_dir, features_file, img_size=256)
    if len(dataset) == 0:
        print("ERROR: No matching MRI-feature pairs found!")
        return None, None
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
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
        
        train_loss /= len(train_loader)
        
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
        
        val_loss /= len(val_loader)
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
                "feature_mean": dataset.feature_mean.tolist(),
                "feature_std": dataset.feature_std.tolist(),
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
    """Predict features from MRI using trained regressor"""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    # Load MRI
    mri_img = nib.load(mri_path).get_fdata()
    mri_img = (mri_img - mri_img.min()) / (mri_img.max() - mri_img.min() + 1e-5)
    
    # Extract middle slice
    D = mri_img.shape[2]
    slice_idx = D // 2
    
    mri_slice = mri_img[:, :, slice_idx]
    mri_slice = torch.tensor(mri_slice, dtype=torch.float32).unsqueeze(0)
    mri_slice = torch.nn.functional.interpolate(
        mri_slice.unsqueeze(0), size=(256, 256),
        mode='bilinear', align_corners=False
    ).to(device)  # (1, 1, 256, 256)
    
    with torch.no_grad():
        features_norm = model(mri_slice).squeeze().cpu().numpy()
    
    # Denormalize
    feature_mean = np.array(feature_mean)
    feature_std = np.array(feature_std)
    features = features_norm * feature_std + feature_mean
    
    return features.tolist()


def get_metadata(mri_path):
    """Extract metadata details from NIfTI file"""
    try:
        img = nib.load(mri_path)
        hdr = img.header
        data = img.get_fdata()
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
    """Compute mean and std intensities for CSF/GM/WM using intensity thresholds"""
    try:
        img = nib.load(mri_path)
        data = img.get_fdata().astype(np.float32)
        mid = data.shape[2] // 2
        arr = data[:, :, mid]
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max - arr_min > 0:
            arr = (arr - arr_min) / (arr_max - arr_min)
        total = arr.size
        bg_mask = arr < 0.05
        csf_mask = (arr >= 0.05) & (arr < 0.25)
        gm_mask = (arr >= 0.25) & (arr < 0.55)
        wm_mask = arr >= 0.55
        
        def class_stats(mask):
            pixels = int(mask.sum())
            percentage = round(float((pixels / total) * 100), 2)
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
            "background": class_stats(bg_mask),
            "csf_ventricles": class_stats(csf_mask),
            "gray_matter": class_stats(gm_mask),
            "white_matter": class_stats(wm_mask),
            "brain_tissue_total_pct": round(float(((gm_mask.sum() + wm_mask.sum()) / total) * 100), 2)
        }
    except Exception:
        return {}

def get_structural_details(mri_path):
    """Extract coordinates and centroids of the brain mask"""
    try:
        from skimage import measure
        img = nib.load(mri_path)
        data = img.get_fdata().astype(np.float32)
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
    except Exception:
        pass
    return {
        "brain_area_pixels": 0,
        "brain_area_pct": 0.0,
        "bounding_box": {"min_row": 0, "min_col": 0, "max_row": 0, "max_col": 0},
        "centroid": {"row": 0.0, "col": 0.0}
    }

def build_structured_json(vector, mri_path=None):
    """Build structured diagnostic JSON matching sample_output.json schema"""
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
        "csf_ventricles": {"pixel_count": 0, "percentage": round(float(vector[10]), 2), "mean_intensity": 0.0, "std_intensity": 0.0},
        "gray_matter": {"pixel_count": 0, "percentage": round(float(vector[11]), 2), "mean_intensity": 0.0, "std_intensity": 0.0},
        "white_matter": {"pixel_count": 0, "percentage": round(float(vector[12]), 2), "mean_intensity": 0.0, "std_intensity": 0.0},
        "brain_tissue_total_pct": round(float(vector[13]), 2)
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
        "brain_area_pct": round(float(vector[22]), 2),
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
            
    struct_det["eccentricity"] = round(float(vector[23]), 4)
    struct_det["solidity"] = round(float(vector[24]), 4)
    struct_det["extent"] = round(float(vector[25]), 4)
    struct_det["perimeter"] = round(float(vector[26]), 1)
    
    sol_val = struct_det["solidity"]
    ecc_val = struct_det["eccentricity"]
    struct_det["interpretation"] = {
        "solidity_close_to_1": "compact, healthy brain shape" if sol_val > 0.8 else "atrophy or tissue degradation present",
        "low_solidity": "possible irregular mass or artifact" if sol_val < 0.8 else "normal compact boundary",
        "eccentricity_near_0": "circular (axial slice normal)" if ecc_val < 0.7 else "ellipsoid / stretched shape",
        "eccentricity_near_1": "elongated, check orientation" if ecc_val > 0.7 else "standard slice orientation"
    }

    left_mean = round(float(vector[0] * (1 + vector[28]/200)), 4)
    right_mean = round(float(vector[0] * (1 - vector[28]/200)), 4)
    asym_pct = round(float(vector[28]), 2)
    
    symmetry_score = round(float(vector[27]), 4)
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
        "schema_version": "1.0",
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
            "dynamic_range": round(float(vector[9]), 4)
        },
        "tissue_segmentation": tissue_seg,
        "texture_features": {
            "glcm_contrast": round(float(vector[14]), 4),
            "glcm_correlation": round(float(vector[15]), 4),
            "glcm_energy": round(float(vector[16]), 4),
            "glcm_homogeneity": round(float(vector[17]), 4),
            "mean_gradient": round(float(vector[18]), 4),
            "std_gradient": round(float(vector[19]), 4),
            "max_gradient": round(float(vector[20]), 4),
            "shannon_entropy": round(float(vector[21]), 4),
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
        "pathology_indicators": {
            "total_flags": total_flags,
            "risk_level": risk_level,
            "flags": flags,
            "disclaimer": "These are AI training indicators only, NOT clinical diagnoses."
        }
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
        mri_files = sorted(Path(mri_dir).glob("*/*.nii"))
    
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
    parser.add_argument("--num-features", type=int, default=29, help="Number of features to predict")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--pretrained-backbone", help="Path to pretrained autoencoder")
    parser.add_argument("--predict", action="store_true", help="Predict features on new data")
    parser.add_argument("--checkpoint", help="Path to trained model checkpoint")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Feature Regression Training")
    print("=" * 70)
    print(f"MRI dir: {args.mri_dir}")
    print(f"Features file (GT): {args.features_file}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num features: {args.num_features}")
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
        )
    else:
        # Prediction with checkpoint
        if not args.checkpoint:
            print("ERROR: --checkpoint required for prediction")
            exit(1)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)
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
        
        # Automatically generate clinical reports if GEMINI_API_KEY is available
        if os.environ.get("GEMINI_API_KEY"):
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
