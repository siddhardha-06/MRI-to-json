"""
U-Net Segmentation Model for MRI
Trained with pseudo-labels from ANTsPy (replaces SynthSeg)

Expected pipeline:
1. Install:  pip install antspyx nibabel torch tqdm
2. Generate pseudo-labels automatically using ANTsPy Atropos:
       python train_unet_antspy.py --generate-masks --mri-dir ixi-t1 --mask-dir ixi-t1-masks
3. Train U-Net with generated masks:
       python train_unet_antspy.py --mri-dir ixi-t1 --mask-dir ixi-t1-masks

WHY ANTsPy instead of SynthSeg:
  - pip install antspyx  → works on Windows, Mac, Linux
  - No FreeSurfer needed
  - Atropos = same quality tissue segmentation (CSF / GM / WM)
  - Produces masks in same format U-Net expects

HDF5 FIX (Windows):
  - ants.image_read() causes HDF5 errors on Windows paths (D:\\...)
  - Fixed by loading via nibabel first → ants.from_numpy()
  - This completely bypasses the HDF5 file detection step
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

# ANTsPy replaces SynthSeg for pseudo-label generation
try:
    import ants
    ANTS_AVAILABLE = True
except ImportError:
    ANTS_AVAILABLE = False
    print("ANTsPy not found. Install with: pip install antspyx")


# =============================================================================
# HELPER: Load NIfTI safely on Windows (fixes HDF5 error)
# =============================================================================

def load_nifti_as_ants(filepath: str):
    """
    Loads a .nii or .nii.gz file safely on ALL platforms including Windows.

    WHY THIS EXISTS:
      ants.image_read() triggers an internal HDF5 file check on Windows paths
      (D:\\...) which fails with:
          DF5-DIAG: Error detected in HDF5 ... unable to determine if file is accessible as HDF5
      
    FIX:
      Load with nibabel first (no HDF5 check) → convert to ANTs via from_numpy()
      This completely avoids the HDF5 detection error.

    Returns:
      img_ants  : ANTs image object (ready for n4, get_mask, atropos)
      nib_img   : nibabel image (to reuse affine when saving output)
      data      : normalized numpy array (0–1)
    """
    # Step 1: Load with nibabel — no HDF5 check, works on all Windows paths
    nib_img = nib.load(filepath)
    data    = nib_img.get_fdata().astype(np.float32)

    # Step 2: Normalize to 0–1
    dmin, dmax = data.min(), data.max()
    if dmax - dmin > 0:
        data = (data - dmin) / (dmax - dmin)

    # Step 3: Get voxel spacing from nibabel header
    try:
        spacing = [float(z) for z in nib_img.header.get_zooms()[:3]]
        spacing = [s if s > 0 else 1.0 for s in spacing]
    except Exception:
        spacing = [1.0, 1.0, 1.0]

    # Step 4: Convert numpy → ANTs image (no file I/O, no HDF5 check)
    img_ants = ants.from_numpy(data, spacing=spacing)

    return img_ants, nib_img, data


# =============================================================================
# 1. U-Net Architecture
# =============================================================================

class ConvBlock(nn.Module):
    """Double conv block with batch norm"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet2D(nn.Module):
    """2D U-Net for slice-wise segmentation (GM/WM/CSF)"""

    def __init__(self, in_channels=1, num_classes=3, base_filters=64):
        super(UNet2D, self).__init__()

        # Encoder
        self.enc1  = ConvBlock(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2  = ConvBlock(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3  = ConvBlock(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_filters * 4, base_filters * 8)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3    = ConvBlock(base_filters * 8, base_filters * 4)

        self.upconv2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2    = ConvBlock(base_filters * 4, base_filters * 2)

        self.upconv1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1    = ConvBlock(base_filters * 2, base_filters)

        # Output
        self.final = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder with skip connections
        enc1 = self.enc1(x)
        x    = self.pool1(enc1)

        enc2 = self.enc2(x)
        x    = self.pool2(enc2)

        enc3 = self.enc3(x)
        x    = self.pool3(enc3)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)

        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)

        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)

        return self.final(x)


# =============================================================================
# 2. ANTsPy Pseudo-label Generator (replaces SynthSeg)
# =============================================================================

def generate_masks_antspy(mri_dir: str, mask_dir: str):
    """
    Replaces: synthseg --i <mri.nii> --o <mask.nii>

    Uses ANTsPy Atropos to segment each MRI into 3 tissue classes:
        1 = CSF
        2 = Gray Matter
        3 = White Matter

    Saves mask as: <mask_dir>/<scan_stem>_seg.nii

    WINDOWS HDF5 FIX:
      Uses load_nifti_as_ants() instead of ants.image_read()
      to avoid HDF5 file detection errors on Windows paths.
    """
    if not ANTS_AVAILABLE:
        print("ANTsPy not installed. Run: pip install antspyx")
        return 0

    mri_path  = Path(mri_dir)
    mask_path = Path(mask_dir)
    mask_path.mkdir(parents=True, exist_ok=True)

    # Find all NIfTI files (handles nested IXI folder structure)
    mri_files = sorted(
        list(mri_path.rglob("*.nii.gz")) +
        list(mri_path.rglob("*.nii"))
    )
    # Keep only files (ignore directories ending with .nii)
    mri_files = [f for f in mri_files if f.is_file()]
    # Remove .nii files that are part of .nii.gz (avoid duplicates)
    mri_files = [f for f in mri_files if not (
        str(f).endswith(".nii") and Path(str(f) + ".gz").exists()
    )]

    if not mri_files:
        print(f"No .nii / .nii.gz files found in {mri_dir}")
        return 0

    print(f"\nFound {len(mri_files)} MRI files. Generating masks with ANTsPy...\n")
    success = 0

    for mri_file in tqdm(mri_files, desc="ANTsPy segmentation"):
        stem     = mri_file.name.replace(".nii.gz", "").replace(".nii", "")
        out_file = mask_path / f"{stem}_seg.nii"

        if out_file.exists():
            success += 1
            continue

        try:
            # ── Step 1: Load safely (fixes Windows HDF5 error) ────
            img, nib_img, _ = load_nifti_as_ants(str(mri_file))

            # ── Step 2: N4 Bias Field Correction ──────────────────
            # Corrects scanner intensity non-uniformity — improves segmentation
            img_corrected = ants.n4_bias_field_correction(img)

            # ── Step 3: Brain mask (skull stripping) ──────────────
            brain_mask = ants.get_mask(img_corrected)

            # ── Step 4: Atropos tissue segmentation ───────────────
            seg_result = ants.atropos(
                a = img_corrected,
                m = '[0.2,1x1x1]',  # smoothing
                c = '[3,0]',         # 3 iterations, convergence 0
                i = 'kmeans[3]',     # 3 tissue classes: CSF, GM, WM
                x = brain_mask
            )

            # 0=Background, 1=CSF, 2=Gray Matter, 3=White Matter
            seg_array = seg_result['segmentation'].numpy().astype(np.int16)

            # ── Step 5: Save mask with original affine ─────────────
            seg_nib = nib.Nifti1Image(seg_array, nib_img.affine)
            nib.save(seg_nib, str(out_file))
            success += 1

        except Exception as e:
            print(f"\n  [SKIP] {mri_file.name}: {e}")

    print(f"\nGenerated {success}/{len(mri_files)} masks saved to: {mask_dir}\n")
    return success


# =============================================================================
# 3. Dataset
# =============================================================================

class MRISegmentationDataset(Dataset):
    """
    Loads MRI + ANTsPy segmentation masks for U-Net training.
    Handles IXI nested folder structure and both .nii / .nii.gz.
    """

    def __init__(self, mri_dir, mask_dir, img_size=256):
        self.mri_dir  = Path(mri_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.samples  = []

        mri_files = sorted(
            list(self.mri_dir.rglob("*.nii.gz")) +
            list(self.mri_dir.rglob("*.nii"))
        )
        # Keep only files (ignore directories ending with .nii)
        mri_files = [f for f in mri_files if f.is_file()]

        for mri_file in mri_files:
            stem      = mri_file.name.replace(".nii.gz", "").replace(".nii", "")
            mask_file = self.mask_dir / f"{stem}_seg.nii"
            if mask_file.exists():
                self.samples.append((str(mri_file), str(mask_file)))

        print(f"Found {len(self.samples)} MRI-mask pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mri_path, mask_path = self.samples[idx]

        # Load MRI via nibabel (safe on Windows — no HDF5 check)
        mri_img  = nib.load(mri_path).get_fdata().astype(np.float32)
        dmin, dmax = mri_img.min(), mri_img.max()
        if dmax - dmin > 0:
            mri_img = (mri_img - dmin) / (dmax - dmin)

        # Load mask via nibabel
        mask_img = nib.load(mask_path).get_fdata()

        # Extract middle axial slice from 3D volume
        D         = mri_img.shape[2]
        slice_idx = D // 2
        mri_slice  = mri_img[:, :, slice_idx]
        mask_slice = mask_img[:, :, slice_idx]

        # Resize MRI slice to img_size
        mri_t = torch.tensor(mri_slice, dtype=torch.float32).unsqueeze(0)
        mri_t = torch.nn.functional.interpolate(
            mri_t.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode='bilinear', align_corners=False
        ).squeeze(0)  # (1, img_size, img_size)

        # Resize mask slice (nearest to preserve labels)
        mask_t = torch.tensor(mask_slice, dtype=torch.long)
        mask_t = torch.nn.functional.interpolate(
            mask_t.float().unsqueeze(0).unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode='nearest'
        ).squeeze(0).squeeze(0).long()  # (img_size, img_size)

        return mri_t, mask_t


# =============================================================================
# 4. Training Loop
# =============================================================================

def train_unet(
    mri_dir="ixi-t1",
    mask_dir="ixi-t1-masks",
    num_epochs=50,
    batch_size=16,
    learning_rate=1e-3,
    save_dir="checkpoints",
):
    """
    Train U-Net segmentation model using ANTsPy pseudo-labels.
    If no masks exist, auto-generates them first using ANTsPy.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Auto-generate masks if none exist
    mask_path = Path(mask_dir)
    num_masks = len(list(mask_path.glob("*_seg.nii")))

    if num_masks == 0:
        print("No masks found. Auto-generating with ANTsPy Atropos...\n")
        generated = generate_masks_antspy(mri_dir, mask_dir)
        if generated == 0:
            print("Mask generation failed. Check ANTsPy installation.")
            return None, None

    # Dataset & DataLoader
    dataset = MRISegmentationDataset(mri_dir, mask_dir, img_size=256)
    if len(dataset) == 0:
        print("No matching MRI-mask pairs found!")
        return None, None

    # num_workers=0 on Windows avoids multiprocessing issues
    num_workers = 0 if os.name == 'nt' else 2
    dataloader  = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )

    # Model
    model = UNet2D(in_channels=1, num_classes=4, base_filters=64).to(device)
    # num_classes=4: Background(0), CSF(1), GM(2), WM(3)

    # Loss: weighted cross-entropy (brain classes weighted higher than background)
    class_weights = torch.tensor([0.1, 0.3, 0.3, 0.3]).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    history  = {"epoch": [], "train_loss": []}
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (x, mask) in enumerate(dataloader):
            x    = x.to(device)
            mask = mask.to(device)

            logits = model(x)
            loss   = criterion(logits, mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Batch {batch_idx+1}/{len(dataloader)} | "
                      f"Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.6f}\n")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch":           epoch + 1,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss":            avg_loss,
                "num_classes":     4,
            }, os.path.join(save_dir, "unet_best.pth"))
            print(f"Saved best model (loss={avg_loss:.6f})\n")

    with open(os.path.join(save_dir, "unet_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Best loss: {best_loss:.6f}")

    return model, history


# =============================================================================
# 5. Inference — segment a new MRI
# =============================================================================

def segment_mri(model, mri_path, device=None):
    """
    Apply trained U-Net to a new MRI file.
    Returns pixel-wise label map: 0=BG, 1=CSF, 2=GM, 3=WM

    Uses nibabel loading (safe on Windows — no HDF5 error).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    # Load safely via nibabel (no HDF5 check)
    mri_img  = nib.load(mri_path).get_fdata().astype(np.float32)
    dmin, dmax = mri_img.min(), mri_img.max()
    if dmax - dmin > 0:
        mri_img = (mri_img - dmin) / (dmax - dmin)

    # Middle slice
    slice_idx = mri_img.shape[2] // 2
    mri_slice = mri_img[:, :, slice_idx]

    # Prepare tensor
    mri_t = torch.tensor(mri_slice, dtype=torch.float32).unsqueeze(0)
    mri_t = torch.nn.functional.interpolate(
        mri_t.unsqueeze(0), size=(256, 256),
        mode='bilinear', align_corners=False
    ).unsqueeze(0).to(device)  # (1, 1, 256, 256)

    with torch.no_grad():
        logits = model(mri_t)
        pred   = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    # Print tissue breakdown
    total = pred.size
    print(f"Segmentation result:")
    print(f"  Background : {(pred==0).sum()/total*100:.1f}%")
    print(f"  CSF        : {(pred==1).sum()/total*100:.1f}%")
    print(f"  Gray Matter: {(pred==2).sum()/total*100:.1f}%")
    print(f"  White Matter:{(pred==3).sum()/total*100:.1f}%")

    return pred  # numpy array (256, 256)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="U-Net MRI Segmentation — ANTsPy (Windows HDF5 fix applied)"
    )
    parser.add_argument("--mri-dir",        default="ixi-t1",       help="MRI dataset directory")
    parser.add_argument("--mask-dir",       default="ixi-t1-masks", help="Mask output directory")
    parser.add_argument("--epochs",         type=int,   default=50,  help="Training epochs")
    parser.add_argument("--batch-size",     type=int,   default=16,  help="Batch size")
    parser.add_argument("--lr",             type=float, default=1e-3,help="Learning rate")
    parser.add_argument("--save-dir",       default="checkpoints",   help="Checkpoint directory")
    parser.add_argument("--generate-masks", action="store_true",
                        help="Generate ANTsPy pseudo-label masks before training")
    parser.add_argument("--segment",        default=None,
                        help="Path to a single .nii file to segment using trained model")

    args = parser.parse_args()

    print("=" * 70)
    print("U-Net MRI Segmentation — ANTsPy (HDF5 Windows fix applied)")
    print("=" * 70)
    print(f"MRI dir        : {args.mri_dir}")
    print(f"Mask dir       : {args.mask_dir}")
    print(f"Epochs         : {args.epochs}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Generate masks : {args.generate_masks}")
    print("=" * 70 + "\n")

    if args.segment:
        # Inference only mode
        print(f"Segmenting: {args.segment}")
        ckpt  = os.path.join(args.save_dir, "unet_best.pth")
        model = UNet2D(in_channels=1, num_classes=4, base_filters=64)
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state"])
        segment_mri(model, args.segment)

    else:
        # Generate masks if requested
        if args.generate_masks:
            print("Step 1: Generating pseudo-labels with ANTsPy...\n")
            generate_masks_antspy(args.mri_dir, args.mask_dir)

        # Train
        print("Step 2: Training U-Net...\n")
        train_unet(
            mri_dir       = args.mri_dir,
            mask_dir      = args.mask_dir,
            num_epochs    = args.epochs,
            batch_size    = args.batch_size,
            learning_rate = args.lr,
            save_dir      = args.save_dir,
        )