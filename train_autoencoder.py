"""
Self-supervised Autoencoder for MRI feature learning
Uses IXI T1 dataset without labels
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
from datetime import datetime

# =============================================================================
# 1. MRI Dataset Loader
# =============================================================================

class MRIDataset(Dataset):
    """Load .nii MRI files and preprocess"""
    
    def __init__(self, data_dir, slice_based=True):
        """
        Args:
            data_dir: Path to ixi-t1/ folder
            slice_based: If True, treat each 2D slice as a sample. If False, use 3D volumes.
        """
        self.data_dir = Path(data_dir)
        self.slice_based = slice_based
        self.samples = []
        
        # Find all .nii files
        nii_files = sorted(self.data_dir.glob("*/*.nii"))
        
        for nii_file in nii_files:
            if slice_based:
                # For slice-based training: store file path, will load on-the-fly
                self.samples.append((str(nii_file), "slices"))
            else:
                # For 3D volumes
                self.samples.append((str(nii_file), "volume"))
        
        print(f"Found {len(nii_files)} MRI files ({len(self.samples)} samples)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, mode = self.samples[idx]
        
        # Load NIfTI
        img = nib.load(file_path)
        data = img.get_fdata()  # Shape: (H, W, D)
        
        # Normalize to [0, 1]
        data = (data - data.min()) / (data.max() - data.min() + 1e-5)
        
        if mode == "slices":
            # Randomly select a slice (skip empty edges)
            D = data.shape[2]
            slice_idx = np.random.randint(max(1, D//4), min(D-1, 3*D//4))
            slice_2d = data[:, :, slice_idx]  # Shape: (H, W)
            
            # Resize to 256x256
            slice_2d = torch.tensor(slice_2d, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
            slice_2d = torch.nn.functional.interpolate(
                slice_2d.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False
            ).squeeze(0)  # (1, 256, 256)
            
            return slice_2d
        
        else:  # 3D volume
            # Resize to 128x128x128
            data_3d = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1, H, W, D)
            data_3d = torch.nn.functional.interpolate(
                data_3d.unsqueeze(0), size=(128, 128, 128), mode='trilinear', align_corners=False
            ).squeeze(0)  # (1, 128, 128, 128)
            
            return data_3d


# =============================================================================
# 2. Autoencoder Architecture
# =============================================================================

class Conv2DAutoencoder(nn.Module):
    """2D CNN Autoencoder for MRI slices"""
    
    def __init__(self, latent_dim=128):
        super(Conv2DAutoencoder, self).__init__()
        
        # Encoder: 256x256 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),      # 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),     # 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),    # 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),   # 16x16
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.fc_encoder = nn.Sequential(
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )
        
        # Decoder: latent_dim -> 256x256
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 16 * 16),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),     # 256x256
            nn.Sigmoid(),
        )
    
    def encode(self, x):
        """Return latent representation"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z = self.fc_encoder(x)
        return z
    
    def decode(self, z):
        """Reconstruct from latent"""
        x = self.fc_decoder(z)
        x = x.view(x.size(0), 256, 16, 16)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class Conv3DAutoencoder(nn.Module):
    """3D CNN Autoencoder for volumetric MRI"""
    
    def __init__(self, latent_dim=256):
        super(Conv3DAutoencoder, self).__init__()
        
        # Encoder: 128x128x128 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),      # 64x64x64
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),     # 32x32x32
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),    # 16x16x16
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        
        self.fc_encoder = nn.Sequential(
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )
        
        # Decoder: latent_dim -> 128x128x128
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 16 * 16 * 16),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),   # 32x32x32
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),    # 64x64x64
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),     # 128x128x128
            nn.Sigmoid(),
        )
    
    def encode(self, x):
        """Return latent representation"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z = self.fc_encoder(x)
        return z
    
    def decode(self, z):
        """Reconstruct from latent"""
        x = self.fc_decoder(z)
        x = x.view(x.size(0), 128, 16, 16, 16)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


# =============================================================================
# 3. Training Loop
# =============================================================================

def train_autoencoder(
    data_dir="ixi-t1",
    model_type="2d",
    num_epochs=50,
    batch_size=8,
    learning_rate=1e-3,
    latent_dim=128,
    save_dir="checkpoints",
):
    """Train the autoencoder"""
    
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load data
    dataset = MRIDataset(data_dir, slice_based=(model_type == "2d"))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Model
    if model_type == "2d":
        model = Conv2DAutoencoder(latent_dim=latent_dim).to(device)
    else:
        model = Conv3DAutoencoder(latent_dim=latent_dim).to(device)
    
    # Loss & optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training
    history = {"epoch": [], "loss": [], "lr": []}
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, x in enumerate(dataloader):
            x = x.to(device)
            
            # Forward
            recon, z = model(x)
            loss = criterion(recon, x)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        
        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_loss)
        history["lr"].append(scheduler.get_last_lr()[0])
        
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}\n")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": avg_loss,
                "latent_dim": latent_dim,
            }
            checkpoint_path = os.path.join(save_dir, f"autoencoder_{model_type}_best.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved best model: {checkpoint_path}\n")
    
    # Save final history
    history_path = os.path.join(save_dir, f"autoencoder_{model_type}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved: {history_path}")
    
    return model, history


# =============================================================================
# 4. Feature Extraction (after training)
# =============================================================================

def extract_features_from_mri(model, data_dir, output_file, model_type="2d", device=None):
    """Extract latent features from trained autoencoder"""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    features_dict = {}
    
    nii_files = sorted(Path(data_dir).glob("*/*.nii"))
    
    with torch.no_grad():
        for nii_file in nii_files:
            # Load MRI
            img = nib.load(nii_file)
            data = img.get_fdata()
            data = (data - data.min()) / (data.max() - data.min() + 1e-5)
            
            if model_type == "2d":
                # Extract middle slices and average features
                slices_features = []
                D = data.shape[2]
                
                for i in [D//4, D//2, 3*D//4]:
                    slice_2d = data[:, :, i]
                    slice_2d = torch.tensor(slice_2d, dtype=torch.float32).unsqueeze(0)
                    slice_2d = torch.nn.functional.interpolate(
                        slice_2d.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False
                    ).squeeze(0).unsqueeze(0).to(device)
                    
                    z = model.encode(slice_2d).cpu().numpy()
                    slices_features.append(z)
                
                # Average across slices
                features_dict[nii_file.stem] = np.mean(slices_features, axis=0)[0].tolist()
            
            else:  # 3D
                data_3d = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
                data_3d = torch.nn.functional.interpolate(
                    data_3d.unsqueeze(0), size=(128, 128, 128), mode='trilinear', align_corners=False
                ).squeeze(0).unsqueeze(0).to(device)
                
                z = model.encode(data_3d).cpu().numpy()
                features_dict[nii_file.stem] = z[0].tolist()
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(features_dict, f, indent=2)
    
    print(f"✓ Extracted features for {len(features_dict)} MRI files")
    print(f"✓ Features saved to: {output_file}")
    
    return features_dict


# =============================================================================
# 5. Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MRI Autoencoder")
    parser.add_argument("--data-dir", default="ixi-t1", help="Path to IXI dataset")
    parser.add_argument("--model-type", choices=["2d", "3d"], default="2d", help="2D or 3D autoencoder")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--extract-features", action="store_true", help="Extract features after training")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MRI Autoencoder Training")
    print("=" * 70)
    print(f"Model type: {args.model_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Learning rate: {args.lr}")
    print("=" * 70 + "\n")
    
    # Train
    model, history = train_autoencoder(
        data_dir=args.data_dir,
        model_type=args.model_type,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        latent_dim=args.latent_dim,
        save_dir=args.save_dir,
    )
    
    # Extract features
    if args.extract_features:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output_file = f"autoencoder_{args.model_type}_features.json"
        extract_features_from_mri(model, args.data_dir, output_file, args.model_type, device)
