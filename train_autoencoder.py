"""
Self-supervised Autoencoder for MRI feature learning
Uses IXI T1 dataset without labels
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
from functools import lru_cache
import random

# =============================================================================
# 1. Volume Caching & Normalization
# =============================================================================

@lru_cache(maxsize=16)  # Cache up to 16 volumes in memory
def _load_and_normalize_volume(file_path):
    """Load MRI NIfTI and apply 1st-99th percentile-clipped normalization"""
    img = nib.load(file_path)
    data = img.get_fdata().astype(np.float32)
    img.uncache()  # Release file descriptors to prevent stale file handles
    
    # 1st-99th percentile clipping to suppress artifacts/outliers
    p1, p99 = np.percentile(data, (1, 99))
    if p99 - p1 > 0:
        data = np.clip(data, p1, p99)
        data = (data - p1) / (p99 - p1 + 1e-5)
    else:
        data = np.zeros_like(data)
    
    data.flags.writeable = False  # Make read-only to prevent in-place corruption
    return data

# =============================================================================
# 2. MRI Dataset Loader
# =============================================================================

class MRIDataset(Dataset):
    """Load .nii/.nii.gz MRI files and preprocess slices/volumes"""
    
    def __init__(self, nii_files, slice_based=True, img_size=256):
        """
        Args:
            nii_files: List of paths to NIfTI files
            slice_based: If True, treat each 2D slice as a sample. If False, use 3D volumes.
            img_size: Size to resize slices (2D) or volumes (3D) to
        """
        self.slice_based = slice_based
        self.img_size = img_size
        self.samples = []
        
        print(f"Indexing slices/volumes from {len(nii_files)} files...")
        for nii_file in nii_files:
            try:
                # Load shape metadata only (extremely fast, doesn't load data array)
                img = nib.load(str(nii_file))
                shape = img.shape
                if len(shape) < 3:
                    continue
                
                if slice_based:
                    # Index all axial slices in the central 25%-75% brain region
                    D = shape[2]
                    for s in range(max(1, D//4), min(D-1, 3*D//4)):
                        self.samples.append((str(nii_file), s))
                else:
                    # For 3D volumetric mode
                    self.samples.append((str(nii_file), -1))
            except Exception as e:
                print(f"  [SKIP] Could not index {nii_file}: {e}")
        
        print(f"Total samples indexed: {len(self.samples)} (slice_based={slice_based})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, slice_idx = self.samples[idx]
        
        # Load from memory cache
        data = _load_and_normalize_volume(file_path)
        
        if self.slice_based:
            # 2D Slice mode
            slice_2d = data[:, :, slice_idx]
            
            # Convert to tensor and resize
            slice_t = torch.tensor(slice_2d, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
            slice_t = torch.nn.functional.interpolate(
                slice_t.unsqueeze(0), size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=False
            ).squeeze(0)  # (1, img_size, img_size)
            
            return slice_t
        else:
            # 3D Volume mode
            volume_t = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1, H, W, D)
            volume_t = torch.nn.functional.interpolate(
                volume_t.unsqueeze(0), size=(self.img_size, self.img_size, self.img_size),
                mode='trilinear', align_corners=False
            ).squeeze(0)  # (1, img_size, img_size, img_size)
            
            return volume_t

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
        for i, (mri_path, _slice_idx) in enumerate(samples):
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
# 3. Autoencoder Architecture
# =============================================================================

class Conv2DAutoencoder(nn.Module):
    """2D CNN Autoencoder for MRI slices (preserving spatial mapping)"""
    
    def __init__(self, latent_dim=128):
        super(Conv2DAutoencoder, self).__init__()
        
        # Encoder: 256x256 -> 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),      # 128x128
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
        
        # Flattened bottleneck size: 512 * 4 * 4 = 8192
        self.fc_encoder = nn.Sequential(
            nn.Linear(8192, latent_dim),
            # Removed ReLU here to prevent constraining the latent space to non-negative values
        )
        
        # Decoder FC mapping back to 8192 features
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 8192),
            nn.ReLU(),
        )
        
        # Decoder: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),     # 256x256
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
        x = x.view(x.size(0), 512, 4, 4)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class Conv3DAutoencoder(nn.Module):
    """3D CNN Autoencoder for volumetric MRI (preventing CPU/GPU OOM)"""
    
    def __init__(self, latent_dim=256):
        super(Conv3DAutoencoder, self).__init__()
        
        # Encoder: 128x128x128 -> 64x64x64 -> 32x32x32 -> 16x16x16 -> 8x8x8 -> 4x4x4
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1),      # 64x64x64
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),     # 32x32x32
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),     # 16x16x16
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),    # 8x8x8
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),   # 4x4x4
            nn.ReLU(),
        )
        
        # Flattened bottleneck size: 256 * 4 * 4 * 4 = 16384
        self.fc_encoder = nn.Sequential(
            nn.Linear(16384, latent_dim),
            # Removed ReLU here to prevent constraining the latent space to non-negative values
        )
        
        # Decoder FC mapping back to 16384 features
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 16384),
            nn.ReLU(),
        )
        
        # Decoder: 4x4x4 -> 8x8x8 -> 16x16x16 -> 32x32x32 -> 64x64x64 -> 128x128x128
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8x8
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),   # 16x16x16
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),    # 32x32x32
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),    # 64x64x64
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1),     # 128x128x128
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
        x = x.view(x.size(0), 256, 4, 4, 4)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

# =============================================================================
# 4. Training Loop
# =============================================================================

def train_autoencoder(
    data_dir="ixi-t1",
    model_type="2d",
    num_epochs=50,
    batch_size=8,
    learning_rate=1e-3,
    latent_dim=128,
    save_dir="checkpoints",
    val_split=0.2,
):
    """Train the autoencoder with a deterministic subject-level train/val split"""
    
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Discover all MRI files recursively (.nii or .nii.gz)
    data_path_dir = Path(data_dir)
    nii_files = sorted(
        list(data_path_dir.rglob("*.nii.gz")) +
        list(data_path_dir.rglob("*.nii"))
    )
    nii_files = [f for f in nii_files if f.is_file()]
    
    if not nii_files:
        print(f"ERROR: No MRI files found in {data_dir}")
        return None, None
        
    print(f"Discovered {len(nii_files)} MRI files. Splitting train/val...")
    
    # Split subject-level deterministically to prevent data leakage
    random.seed(42)
    shuffled_files = list(nii_files)
    random.shuffle(shuffled_files)
    
    split_idx = int(len(shuffled_files) * (1.0 - val_split))
    train_files = shuffled_files[:split_idx]
    val_files = shuffled_files[split_idx:]
    
    print(f"Subjects: {len(train_files)} train, {len(val_files)} val")
    
    # Instantiate datasets
    img_size = 256 if model_type == "2d" else 128
    train_dataset = MRIDataset(train_files, slice_based=(model_type == "2d"), img_size=img_size)
    val_dataset = MRIDataset(val_files, slice_based=(model_type == "2d"), img_size=img_size)
    
    # Force num_workers = 0 so the lru_cache is loaded in-process and shared
    num_workers = 0
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
    if model_type == "2d":
        model = Conv2DAutoencoder(latent_dim=latent_dim).to(device)
    else:
        model = Conv3DAutoencoder(latent_dim=latent_dim).to(device)
    
    # Loss & optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training Loop
    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch_idx, x in enumerate(train_loader):
            x = x.to(device)
            
            # Forward
            recon, z = model(x)
            loss = criterion(recon, x)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(train_loader)} | Train Loss: {loss.item():.6f}")
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        if len(val_loader) > 0:
            with torch.no_grad():
                for x in val_loader:
                    x = x.to(device)
                    recon, z = model(x)
                    loss = criterion(recon, x)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
        else:
            avg_val_loss = 0.0
            
        scheduler.step()
        
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(scheduler.get_last_lr()[0])
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}\n")
        
        # Save best model based on validation loss (or training loss if val set is empty)
        val_metric = avg_val_loss if len(val_loader) > 0 else avg_train_loss
        if val_metric < best_val_loss:
            best_val_loss = val_metric
            checkpoint = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "latent_dim": latent_dim,
            }
            checkpoint_path = os.path.join(save_dir, f"autoencoder_{model_type}_best.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"[SUCCESS] Saved best model: {checkpoint_path}\n")
    
    # Save final history
    history_path = os.path.join(save_dir, f"autoencoder_{model_type}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"[SUCCESS] Training history saved: {history_path}")
    
    return model, history

# =============================================================================
# 5. Feature Extraction (after training)
# =============================================================================

def extract_features_from_mri(model, data_dir, output_file, model_type="2d", device=None):
    """Extract latent features from trained autoencoder by averaging 10 slices"""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    features_dict = {}
    
    data_path_dir = Path(data_dir)
    nii_files = sorted(
        list(data_path_dir.rglob("*.nii.gz")) +
        list(data_path_dir.rglob("*.nii"))
    )
    nii_files = [f for f in nii_files if f.is_file()]
    
    with torch.no_grad():
        for nii_file in nii_files:
            try:
                # Load normalized volume from LRU cache
                data = _load_and_normalize_volume(str(nii_file))
                
                if model_type == "2d":
                    # Sample 10 evenly spaced slices across the central brain region
                    slices_features = []
                    D = data.shape[2]
                    
                    slice_indices = np.linspace(D//4, 3*D//4, 10, dtype=int)
                    for i in slice_indices:
                        slice_2d = data[:, :, i]
                        slice_2d = torch.tensor(slice_2d, dtype=torch.float32).unsqueeze(0)
                        slice_2d = torch.nn.functional.interpolate(
                            slice_2d.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False
                        ).squeeze(0).unsqueeze(0).to(device)
                        
                        z = model.encode(slice_2d).cpu().numpy()
                        slices_features.append(z)
                    
                    # Average across slices
                    features_dict[nii_file.stem] = np.mean(slices_features, axis=0)[0].tolist()
                
                else:  # 3D Mode
                    data_3d = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
                    data_3d = torch.nn.functional.interpolate(
                        data_3d.unsqueeze(0), size=(128, 128, 128), mode='trilinear', align_corners=False
                    ).squeeze(0).unsqueeze(0).to(device)
                    
                    z = model.encode(data_3d).cpu().numpy()
                    features_dict[nii_file.stem] = z[0].tolist()
            except Exception as e:
                print(f"  [ERROR] Could not extract features for {nii_file}: {e}")
                
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(features_dict, f, indent=2)
    
    print(f"[SUCCESS] Extracted features for {len(features_dict)} MRI files")
    print(f"[SUCCESS] Features saved to: {output_file}")
    
    return features_dict

# =============================================================================
# 6. Main
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
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
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
    print(f"Val split: {args.val_split}")
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
        val_split=args.val_split,
    )
    
    # Extract features
    if args.extract_features:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output_file = f"autoencoder_{args.model_type}_features.json"
        extract_features_from_mri(model, args.data_dir, output_file, args.model_type, device)
