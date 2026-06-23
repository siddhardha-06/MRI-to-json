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
            print(f"✓ Saved best model: {checkpoint_path}\n")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Save history
    history_path = os.path.join(save_dir, "regressor_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved: {history_path}")
    
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


def batch_predict_features(model, mri_dir, output_file, feature_mean, feature_std, device=None):
    """Predict features for all MRI files"""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    predictions = {}
    
    mri_files = sorted(Path(mri_dir).glob("*/*.nii"))
    
    with torch.no_grad():
        for mri_file in mri_files:
            features = predict_features_from_mri(
                model, str(mri_file), feature_mean, feature_std, device
            )
            predictions[mri_file.stem] = features
    
    # Save to JSON
    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"✓ Predicted features for {len(predictions)} MRI files")
    print(f"✓ Predictions saved to: {output_file}")
    
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
        batch_predict_features(
            model, args.mri_dir, output_file,
            checkpoint["feature_mean"], checkpoint["feature_std"], device
        )
