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
from torch.utils.data import Dataset, DataLoader, Sampler
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import random
import math
import functools
from skimage import measure, morphology

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

    NOTE: Returns RAW (unnormalized) data so that N4 bias field correction
    can operate on original scanner intensities. Normalize AFTER N4.

    Returns:
      img_ants  : ANTs image object (raw intensities, ready for N4)
      nib_img   : nibabel image (to reuse affine when saving output)
      data      : raw numpy array (float32, NOT normalized)
    """
    # Step 1: Load with nibabel — no HDF5 check, works on all Windows paths
    nib_img = nib.load(filepath)
    data    = nib_img.get_fdata().astype(np.float32)

    # Step 2: Get voxel spacing from nibabel header
    try:
        spacing = [float(z) for z in nib_img.header.get_zooms()[:3]]
        spacing = [s if s > 0 else 1.0 for s in spacing]
    except Exception:
        spacing = [1.0, 1.0, 1.0]

    # Step 3: Convert numpy → ANTs image (no file I/O, no HDF5 check)
    # Uses raw intensities — normalization should happen after N4 correction
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

    def __init__(self, in_channels=1, num_classes=4, base_filters=64):
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


class DiceLoss(nn.Module):
    """
    Multiclass Dice Loss for semantic segmentation.
    Computes Dice loss for all foreground classes (1, 2, 3) and averages them.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        
        # One-hot encode targets to (B, C, H, W)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        dice_loss = 0.0
        # Exclude background (class 0) from Dice Loss optimization
        for c in range(1, num_classes):
            p_c = probs[:, c, :, :]
            t_c = targets_one_hot[:, c, :, :]
            
            intersection = (p_c * t_c).sum()
            total = p_c.sum() + t_c.sum()
            
            dice_loss += 1.0 - (2.0 * intersection + self.smooth) / (total + self.smooth)
            
        return dice_loss / (num_classes - 1)


# =============================================================================
# 2. ANTsPy Pseudo-label Generator (replaces SynthSeg)
# =============================================================================

def generate_masks_antspy(mri_dir: str, mask_dir: str, limit: int = None):
    """
    Replaces: synthseg --i <mri.nii> --o <mask.nii>

    Uses ANTsPy Atropos to segment each MRI into 3 tissue classes:
        1 = CSF
        2 = Gray Matter
        3 = White Matter

    Saves:
      <mask_dir>/<stem>_seg.nii       — integer label mask (0/1/2/3)
      <mask_dir>/n4/<stem>_n4.nii    — N4-corrected MRI (used by training
                                        to eliminate training/label domain mismatch)

    DOMAIN ALIGNMENT:
      Atropos sees N4-corrected intensities when creating labels. Saving the
      N4-corrected MRI lets the training dataset loader use the same
      intensity distribution, so the model trains on exactly what Atropos
      used to produce the pseudo-labels. Without this, training data is raw
      (non-N4) while labels come from N4 → WM gets systematically
      under-segmented at inference.

    ATROPOS LABEL VALIDATION:
      Atropos k-means assigns labels 1-3 by cluster mean in ascending order.
      For T1 MRI that is normally CSF(darkest)=1, GM=2, WM(brightest)=3.
      After segmentation we verify this ordering and remap if necessary.

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
    # Sub-directory for N4-corrected MRIs (used by _load_and_normalize_volume)
    n4_path = mask_path / "n4"
    n4_path.mkdir(parents=True, exist_ok=True)

    # Find all NIfTI files (handles nested IXI folder structure)
    mri_files = sorted(
        list(mri_path.rglob("*.nii.gz")) +
        list(mri_path.rglob("*.nii"))
    )
    # Keep only files (ignore directories ending with .nii)
    mri_files = [f for f in mri_files if f.is_file()]
    mri_files = [f for f in mri_files if not (
        str(f).endswith(".nii") and Path(str(f) + ".gz").exists()
    )]
    if limit is not None:
        mri_files = mri_files[:limit]

    if not mri_files:
        print(f"No .nii / .nii.gz files found in {mri_dir}")
        return 0

    print(f"\nFound {len(mri_files)} MRI files. Generating masks with ANTsPy...\n")
    success = 0

    for mri_file in tqdm(mri_files, desc="ANTsPy segmentation"):
        stem     = mri_file.name.replace(".nii.gz", "").replace(".nii", "")
        out_file = mask_path / f"{stem}_seg.nii"
        n4_file  = n4_path   / f"{stem}_n4.nii"

        # FIX 3: require BOTH mask AND n4 file before skipping.
        # If a prior run saved the mask but not the n4 (e.g. old script version),
        # we must continue so the n4 save block below is reached.
        if out_file.exists() and n4_file.exists():
            success += 1
            continue

        try:
            # ── Step 1: Load safely (fixes Windows HDF5 error) ────
            img, nib_img, _ = load_nifti_as_ants(str(mri_file))

            # ── Step 2: N4 Bias Field Correction ──────────────────
            # N4 must see raw scanner intensities to detect the bias field.
            img_corrected = ants.n4_bias_field_correction(img)
            arr = img_corrected.numpy().astype(np.float32)

            # ── Step 2b: Save N4-corrected MRI for training use ───
            # Training loads this file so it matches the intensity domain
            # that Atropos used when generating pseudo-labels.
            if not n4_file.exists():
                n4_nib = nib.Nifti1Image(arr, nib_img.affine)
                n4_nib.header.set_data_dtype(np.float32)
                n4_nib.header.set_qform(nib_img.affine, code=1)
                n4_nib.header.set_sform(nib_img.affine, code=1)
                n4_nib.header.set_zooms(nib_img.header.get_zooms()[:3])
                nib.save(n4_nib, str(n4_file))

            # ── Step 3: Brain mask (slice-by-slice morphology — faster) ──
            # Normalize for thresholding only (not saved; Atropos uses raw N4)
            p1  = float(np.percentile(arr, 1))
            p99 = float(np.percentile(arr, 99))
            arr_norm = np.clip(arr, p1, p99)
            arr_norm = (arr_norm - p1) / (p99 - p1 + 1e-5)

            # Coarse brain seed using a moderate threshold
            brain_seeds = arr_norm > 0.18

            # Slice-by-slice disk morphology (faster than 3D ball(2) kernel).
            # Uses current skimage 0.26+ API: closing/dilation with footprint=
            # (binary_closing/binary_dilation removed in 0.28).
            from skimage.morphology import disk, closing as sk_closing, dilation as sk_dilation
            brain_clean_mask = np.zeros_like(brain_seeds)
            d = disk(2)
            for s in range(brain_seeds.shape[2]):
                sl = brain_seeds[:, :, s]
                sl = sk_closing(sl, footprint=d)
                sl = sk_dilation(sl, footprint=d)
                brain_clean_mask[:, :, s] = sl

            # Largest connected component = brain
            labeled = measure.label(brain_clean_mask)
            regions = measure.regionprops(labeled)
            if not regions:
                raise ValueError("No brain regions found during skull stripping.")
            largest_region  = max(regions, key=lambda r: r.area)
            brain_clean_mask = (labeled == largest_region.label)

            # Convert to ANTs image
            brain_mask = ants.from_numpy(
                brain_clean_mask.astype(np.float32), spacing=img_corrected.spacing
            )

            # ── Step 4: Atropos tissue segmentation ───────────────
            seg_result = ants.atropos(
                a = img_corrected,
                m = '[0.2,1x1x1]',
                c = '[5,0.00001]',
                i = 'kmeans[3]',
                x = brain_mask
            )

            seg_array = seg_result['segmentation'].numpy().astype(np.int16)

            # ── FIX 5: Validate and remap Atropos label order ─────
            # Atropos k-means labels by ascending intensity. On T1 MRI after N4:
            #   label 1 = darkest cluster  → should be CSF
            #   label 2 = mid cluster      → should be GM
            #   label 3 = brightest cluster → should be WM
            # Compute mean N4 intensity for each label and remap if mis-ordered.
            label_means = {}
            for lbl in [1, 2, 3]:
                voxels = arr_norm[seg_array == lbl]
                label_means[lbl] = float(voxels.mean()) if len(voxels) > 0 else 0.0
            sorted_lbls = sorted(label_means, key=label_means.get)  # ascending intensity
            # sorted_lbls[0]=darkest(CSF), [1]=mid(GM), [2]=brightest(WM)
            expected = [1, 2, 3]  # CSF, GM, WM in ascending intensity order
            if sorted_lbls != expected:
                print(f"  [INFO] Remapping Atropos labels: {sorted_lbls} → {expected}")
                remap = {sorted_lbls[i]: expected[i] for i in range(3)}
                remap[0] = 0  # background stays 0
                remapped = np.vectorize(remap.get)(seg_array).astype(np.int16)
                seg_array = remapped
            else:
                # Sanity-check: WM (label 3) must have highest mean
                if label_means[3] < label_means[2]:
                    print(f"  [WARNING] {stem}: label 3 mean ({label_means[3]:.3f}) < "
                          f"label 2 mean ({label_means[2]:.3f}) — WM/GM may be swapped")

            # ── Step 5: Save mask ──────────────────────────────────
            seg_nib = nib.Nifti1Image(seg_array, nib_img.affine)
            seg_nib.header.set_data_dtype(np.int16)
            seg_nib.header.set_qform(nib_img.affine, code=1)
            seg_nib.header.set_sform(nib_img.affine, code=1)
            seg_nib.header.set_zooms(nib_img.header.get_zooms()[:3])
            nib.save(seg_nib, str(out_file))
            success += 1

        except Exception as e:
            print(f"\n  [SKIP] {mri_file.name}: {e}")

    print(f"\nGenerated {success}/{len(mri_files)} masks saved to: {mask_dir}\n")
    print(f"N4-corrected MRIs saved to: {n4_path}  (used by training loader)\n")
    return success


# =============================================================================
# 3. Dataset
# =============================================================================

# LRU cache for NIfTI volumes — avoids reloading from disk for every slice.
# maxsize=8: each N4 MRI is ~40 MB float32; 8 volumes = ~320 MB peak.
# Safe with mmap=False (already fixed WinError 8); VolumeGroupedSampler
# ensures consecutive slices come from the same volume so cache hit rate
# is ~100% instead of ~2% with pure random shuffle.
@functools.lru_cache(maxsize=8)
def _load_and_normalize_volume(filepath):
    """
    Load a NIfTI volume into RAM and apply brain-masked percentile normalization.

    WINDOWS MEMMAP FIX (WinError 8):
      nibabel uses np.memmap by default for uncompressed .nii files.
      On Windows, each memmap consumes virtual address space. After ~1 hour
      of training + mask generation, Windows runs out of address space and
      raises OSError [WinError 8]. Fix: mmap=False forces nibabel to read
      the file directly into a regular numpy array (normal malloc, no mmap).

    DOMAIN ALIGNMENT:
      Loads N4-corrected MRIs (redirected by _resolve_mri_path) so the
      training intensity distribution matches what Atropos used for labels.

    CORRUPTION GUARD:
      RuntimeError with clear message on any nibabel failure.
      lru_cache does NOT cache exceptions so the next call can retry.
    """
    try:
        # mmap=False: load directly into RAM — avoids Windows WinError 8
        img  = nib.load(filepath, mmap=False)
        data = img.get_fdata().astype(np.float32)
        img.uncache()
    except Exception as e:
        raise RuntimeError(
            f"Failed to load NIfTI file: {filepath}\n"
            f"Error: {e}\n"
            "If this is an N4-corrected file, delete it and re-run --generate-masks."
        ) from e

    # Robust brain floor: 5th percentile is safer than data>0 for scanners
    # that produce slightly negative background values.
    floor = float(np.percentile(data, 5))
    brain_voxels = data[data > floor]
    if len(brain_voxels) > 0:
        p1  = float(np.percentile(brain_voxels, 1))
        p99 = float(np.percentile(brain_voxels, 99))
    else:
        p1, p99 = float(np.percentile(data, 1)), float(np.percentile(data, 99))

    data = np.clip(data, p1, p99)
    if p99 - p1 > 0:
        data = (data - p1) / (p99 - p1)
    data.flags.writeable = False
    return data


def _n4_file_is_valid(path: Path) -> bool:
    """
    Validates an N4-corrected NIfTI file by reading its header (fast — no
    data load) and checking whether the actual file size matches what the
    header says the data should occupy.

    WHY HEADER-BASED VALIDATION:
      A fixed byte threshold (e.g. 50 KB) fails for partially-written files
      that are large enough to pass the threshold but still truncated.
      Example: a 256×256×150 float32 volume expects 39,321,600 bytes of data
      but a killed write may have flushed only 1,835,008 bytes — well above
      50 KB but only ~5% complete. nibabel raises:
        OSError: Expected 39321600 bytes, got 1835008 bytes — could the file be damaged?

      By reading the NIfTI header (348 bytes, always complete if the file
      exists), we compute the exact expected size and compare to actual.
    """
    try:
        img = nib.load(str(path))           # header-only load (fast, no mmap)
        expected_data_bytes = int(np.prod(img.shape)) * np.dtype(img.get_data_dtype()).itemsize
        actual_size = path.stat().st_size
        vox_offset  = int(img.header.get('vox_offset', 352))
        actual_data_bytes = actual_size - vox_offset
        img.uncache()
        # Allow 1% tolerance for alignment padding
        return actual_data_bytes >= expected_data_bytes * 0.99
    except Exception:
        return False


def _resolve_mri_path(mri_path: str, mask_dir: str = None) -> str:
    """
    Returns the N4-corrected MRI path if it was saved during mask generation,
    otherwise returns the original path unchanged.

    CORRUPTION GUARD:
      Uses header-based validation (_n4_file_is_valid) to detect partially-
      written N4 files of any size — not just tiny ones. Falls back silently
      to the raw MRI so training continues without crashing.
      To fix: delete the flagged file and re-run --generate-masks.
    """
    if mask_dir is None:
        return mri_path
    stem = Path(mri_path).name.replace(".nii.gz", "").replace(".nii", "")
    n4_candidate = Path(mask_dir) / "n4" / f"{stem}_n4.nii"
    if n4_candidate.exists():
        if _n4_file_is_valid(n4_candidate):
            return str(n4_candidate)
        else:
            sz_mb = n4_candidate.stat().st_size / 1_048_576
            print(f"  [WARNING] Truncated N4 file skipped: {n4_candidate.name} "
                  f"({sz_mb:.1f} MB on disk, expected much larger) — "
                  f"using raw MRI instead. Delete and re-run --generate-masks to fix.")
    return mri_path


# maxsize=8: each mask is ~10 MB int16; 8 volumes = ~80 MB peak.
@functools.lru_cache(maxsize=8)
def _load_mask_volume(filepath):
    """Load a NIfTI mask volume directly into RAM (mmap=False avoids WinError 8)."""
    img  = nib.load(filepath, mmap=False)  # mmap=False: no Windows memmap limits
    data = img.get_fdata()
    img.uncache()
    data.flags.writeable = False
    return data


class MRISegmentationDataset(Dataset):
    """
    Loads MRI + ANTsPy segmentation masks for U-Net training.
    Indexes ALL valid axial slices per volume (not just the middle one).
    Supports data augmentation for training.

    Volumes are cached in memory via LRU cache so each NIfTI file is
    loaded once and reused across all its slices (eliminates I/O bottleneck).

    Args:
        sample_pairs: list of (mri_path_str, mask_path_str) tuples
        img_size: resize target for each slice
        train: if True, apply random augmentations (flips, rotation, brightness)
        min_foreground_ratio: skip slices with less foreground than this (0 to disable)
    """

    def __init__(self, sample_pairs, img_size=256, train=True,
                 min_foreground_ratio=0.05, mask_dir=None):
        self.img_size = img_size
        self.train = train
        # Note: mask_dir is used only during __init__ for N4 path resolution.
        # The resolved paths are stored in self.slices so mask_dir is not
        # kept as an instance attribute (avoids confusion — it's not used later).
        self.slices = []  # list of (resolved_mri_path, mask_path, slice_idx)

        split_label = "train" if train else "val"
        print(f"Indexing slices from {len(sample_pairs)} {split_label} volumes...")
        n4_used = 0
        n4_missing = []
        for mri_path, mask_path in tqdm(sample_pairs, desc="Indexing slices"):
            try:
                # Release file descriptor immediately after indexing (fixes memory leak)
                mask_img  = nib.load(mask_path)
                mask_data = mask_img.get_fdata()
                mask_img.uncache()

                # Transparently redirect to N4-corrected MRI if available
                resolved = _resolve_mri_path(mri_path, mask_dir)
                if resolved != mri_path:
                    n4_used += 1
                elif mask_dir is not None:
                    # FIX 2: per-subject warning for partial mask generation.
                    # Catches the case where train subjects have N4 files but
                    # val subjects don't (e.g. partial --generate-masks run),
                    # which would cause a silent val/train loss discrepancy.
                    stem = Path(mri_path).name.replace(".nii.gz", "").replace(".nii", "")
                    n4_missing.append(stem)

                num_slices = mask_data.shape[2]
                for s in range(num_slices):
                    mask_slice = mask_data[:, :, s]
                    foreground_ratio = np.count_nonzero(mask_slice) / mask_slice.size
                    if foreground_ratio >= min_foreground_ratio:
                        self.slices.append((resolved, mask_path, s))
            except Exception as e:
                print(f"  [SKIP] Could not index {mri_path}: {e}")

        if n4_used > 0:
            print(f"  [INFO] Using N4-corrected MRIs for {n4_used}/{len(sample_pairs)} "
                  f"{split_label} volumes (eliminates training/label domain mismatch)")
        if n4_missing:
            print(f"  [WARNING] {len(n4_missing)} {split_label} subjects have NO N4 MRI "
                  f"— will use raw files (may cause intensity mismatch with labels):")
            for s in n4_missing[:5]:
                print(f"    {s}")
            if len(n4_missing) > 5:
                print(f"    ... and {len(n4_missing) - 5} more. Re-run --generate-masks.")
        elif mask_dir is not None and n4_used == 0:
            print(f"  [WARNING] No N4-corrected MRIs found in {mask_dir}/n4/. "
                  f"All {split_label} volumes will use raw MRIs — run --generate-masks first.")

        print(f"Total slices indexed: {len(self.slices)} "
              f"(from {len(sample_pairs)} {split_label} volumes)")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        mri_path, mask_path, slice_idx = self.slices[idx]

        # Load volumes via LRU cache (each file loaded once, not per-slice)
        mri_img = _load_and_normalize_volume(mri_path)
        mask_img = _load_mask_volume(mask_path)

        # Extract the specific axial slice
        mri_slice = mri_img[:, :, slice_idx]
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

        # Data augmentation (training only)
        if self.train:
            mri_t, mask_t = self._augment(mri_t, mask_t)

        return mri_t, mask_t

    def _augment(self, mri_t, mask_t):
        """
        Apply random augmentations to MRI slice and mask.
        Uses only torch operations (no extra dependencies).
        """
        # Random horizontal flip (50%)
        if random.random() > 0.5:
            mri_t = torch.flip(mri_t, dims=[2])
            mask_t = torch.flip(mask_t, dims=[1])

        # Random rotation (±15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            angle_rad = math.radians(angle)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

            # Affine rotation matrix for grid_sample
            theta = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0]
            ], dtype=torch.float32).unsqueeze(0)

            # Rotate MRI (bilinear)
            grid = torch.nn.functional.affine_grid(
                theta, mri_t.unsqueeze(0).size(), align_corners=False
            )
            mri_t = torch.nn.functional.grid_sample(
                mri_t.unsqueeze(0), grid, mode='bilinear',
                padding_mode='zeros', align_corners=False
            ).squeeze(0)

            # Rotate mask (nearest to preserve labels)
            mask_float = mask_t.float().unsqueeze(0).unsqueeze(0)
            mask_t = torch.nn.functional.grid_sample(
                mask_float, grid, mode='nearest',
                padding_mode='zeros', align_corners=False
            ).squeeze(0).squeeze(0).long()

        # Random brightness jitter (±10%)
        if random.random() > 0.5:
            factor = random.uniform(0.9, 1.1)
            mri_t = (mri_t * factor).clamp(0.0, 1.0)

        return mri_t, mask_t




class VolumeGroupedSampler(Sampler):
    """
    Samples slices grouped by volume so that consecutive items in the
    DataLoader come from the same 3D MRI. This gives ~100% LRU cache
    hit rate instead of ~2% with pure random shuffle.

    WHY THIS MATTERS:
      With random shuffle across 25,529 slices from 200 volumes, each
      batch of 8 slices likely needs 8 different volumes. Each miss
      loads a full ~40 MB N4 NIfTI from disk → ~13 sec/batch → 25 days.
      Volume-grouped sampling loads each volume once then reuses it for
      all its ~127 slices → ~0.1-0.3 sec/batch → ~2-6 hrs total.

    Shuffling behaviour:
      - Between epochs: volume order is reshuffled (epoch-level randomness)
      - Within a volume: slice order is reshuffled (slice-level randomness)
      This preserves stochastic gradient noise while eliminating I/O waste.
    """
    def __init__(self, slices: list, shuffle: bool = True):
        from collections import defaultdict
        # Group slice list-indices by MRI file path
        vol_to_indices: dict = defaultdict(list)
        for i, (mri_path, _mask_path, _s) in enumerate(slices):
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
# 4. Training Loop
# =============================================================================

def train_unet(
    mri_dir="ixi-t1",
    mask_dir="ixi-t1-masks",
    num_epochs=50,
    batch_size=16,
    learning_rate=1e-3,
    save_dir="checkpoints",
    val_split=0.2,
    resume=False,
):
    """
    Train U-Net segmentation model using ANTsPy pseudo-labels.
    If no masks exist, auto-generates them first using ANTsPy.
    Uses an 80/20 subject-level train/val split to detect overfitting.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Auto-generate masks if none exist
    mask_path_dir = Path(mask_dir)
    num_masks = len(list(mask_path_dir.glob("*_seg.nii")))

    if num_masks == 0:
        print("No masks found. Auto-generating with ANTsPy Atropos...\n")
        generated = generate_masks_antspy(mri_dir, mask_dir)
        if generated == 0:
            print("Mask generation failed. Check ANTsPy installation.")
            return None, None

    # ── Discover MRI-mask pairs ──────────────────────────────────────
    mri_path_dir = Path(mri_dir)
    mri_files = sorted(
        list(mri_path_dir.rglob("*.nii.gz")) +
        list(mri_path_dir.rglob("*.nii"))
    )
    mri_files = [f for f in mri_files if f.is_file()]

    all_pairs = []
    for mri_file in mri_files:
        stem = mri_file.name.replace(".nii.gz", "").replace(".nii", "")
        mask_file = mask_path_dir / f"{stem}_seg.nii"
        if mask_file.exists():
            all_pairs.append((str(mri_file), str(mask_file)))

    if len(all_pairs) == 0:
        print("No matching MRI-mask pairs found!")
        return None, None

    print(f"Found {len(all_pairs)} MRI-mask pairs")

    # ── Subject-level train/val split (deterministic) ────────────────
    random.seed(42)
    shuffled = list(all_pairs)
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split))
    val_pairs = shuffled[:n_val]
    train_pairs = shuffled[n_val:]

    print(f"Split: {len(train_pairs)} train subjects, {len(val_pairs)} val subjects")

    # ── Create datasets (augmentation on train only) ─────────────────
    # Pass mask_dir so _resolve_mri_path can transparently swap in N4 MRIs
    train_dataset = MRISegmentationDataset(
        train_pairs, img_size=256, train=True,  mask_dir=mask_dir
    )
    val_dataset = MRISegmentationDataset(
        val_pairs,   img_size=256, train=False, mask_dir=mask_dir
    )

    if len(train_dataset) == 0:
        print("No training slices found after indexing!")
        return None, None

    # ── DataLoaders ───────────────────────────────────────────────────
    # VolumeGroupedSampler: consecutive slices come from the same volume
    # → ~100% LRU cache hit rate → eliminates the disk I/O bottleneck
    # that caused ~13 sec/batch (25-day training time) with random shuffle.
    num_workers = 0  # lru_cache must stay in-process (no worker forks)
    pin_mem = torch.cuda.is_available()
    train_sampler = VolumeGroupedSampler(train_dataset.slices, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers,
        pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=pin_mem
    )

    start_epoch = 0
    # Model
    model = UNet2D(in_channels=1, num_classes=4, base_filters=64).to(device)
    # num_classes=4: Background(0), CSF(1), GM(2), WM(3)

    # Loss: weighted cross-entropy + Dice Loss (hybrid)
    class_weights = torch.tensor([0.1, 0.3, 0.3, 0.3]).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = DiceLoss()
    
    def hybrid_criterion(logits, targets):
        return ce_loss(logits, targets) + 0.5 * dice_loss(logits, targets)
        
    criterion = hybrid_criterion

    # Optimizer: AdamW with weight decay (L2 regularization) to prevent overfitting
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    history  = {"epoch": [], "train_loss": [], "val_loss": []}
    best_val_loss = float('inf')

    ckpt_path = os.path.join(save_dir, "unet_best.pth")
    if resume and os.path.exists(ckpt_path):
        print(f"Resuming training from checkpoint: {ckpt_path}")
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = checkpoint["epoch"]
            best_val_loss = checkpoint.get("val_loss", float('inf'))
            print(f"Loaded checkpoint at epoch {start_epoch} with val_loss: {best_val_loss:.6f}")
            
            history_path = os.path.join(save_dir, "unet_history.json")
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)
                for key in ["epoch", "train_loss", "val_loss"]:
                    history[key] = history[key][:start_epoch]
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint, starting from scratch: {e}")
            start_epoch = 0

    # Scheduler: Cosine Annealing LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5, last_epoch=start_epoch - 1
    )

    for epoch in range(start_epoch, num_epochs):
        # ── Training ─────────────────────────────────────────────────
        model.train()
        total_train_loss = 0

        for batch_idx, (x, mask) in enumerate(train_loader):
            x    = x.to(device)
            mask = mask.to(device)

            logits = model(x)
            loss   = criterion(logits, mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.6f}")

        avg_train_loss = total_train_loss / max(len(train_loader), 1)

        # ── Validation ───────────────────────────────────────────────
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for x, mask in val_loader:
                x    = x.to(device)
                mask = mask.to(device)
                logits = model(x)
                loss   = criterion(logits, mask)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / max(len(val_loader), 1)

        scheduler.step()

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f}\n")

        # Save best checkpoint (based on validation loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch":           epoch + 1,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss":      avg_train_loss,
                "val_loss":        avg_val_loss,
                "num_classes":     4,
            }, os.path.join(save_dir, "unet_best.pth"))
            print(f"Saved best model (val_loss={avg_val_loss:.6f})\n")

    with open(os.path.join(save_dir, "unet_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Best val loss: {best_val_loss:.6f}")

    return model, history


# =============================================================================
# 5. Inference — segment a new MRI
# =============================================================================

def segment_mri(model, mri_path, device=None, img_size=256, inference_batch=16):
    """
    Apply trained U-Net to a new MRI file — full 3D volume segmentation.
    Processes ALL axial slices (batched for GPU efficiency) and reconstructs
    the full 3D label volume at original resolution.

    DOMAIN ALIGNMENT FIX:
      Training pseudo-labels were generated from N4-corrected MRIs (Atropos
      saw bias-corrected intensities). To match that distribution at inference,
      we must also run N4 before normalizing. Skipping N4 here causes WM
      intensity compression → model predicts far fewer WM voxels (<10%).

    Returns:
        seg_volume: numpy array (H, W, D) with labels 0=BG, 1=CSF, 2=GM, 3=WM
        nib_img:    original nibabel image (for saving with correct affine)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # ── Step 1: Load raw (no HDF5 check on Windows) ───────────────────
    nib_img = nib.load(mri_path)
    raw_data = nib_img.get_fdata().astype(np.float32)
    H, W, D = raw_data.shape

    # ── Step 2: N4 Bias Field Correction (matches mask-generation pipeline)
    # N4 must operate on raw scanner intensities — normalize AFTER
    # FIX 6: N4 failure is now a hard error, not a silent fallback.
    #   The model was trained on N4-corrected data; raw data at inference
    #   produces a different intensity distribution → unreliable WM prediction.
    if not ANTS_AVAILABLE:
        raise RuntimeError(
            "ANTsPy is required for inference but is not installed.\n"
            "Install with:  pip install antspyx\n"
            "Without N4 bias correction, WM % will be severely underestimated "
            "because the model was trained on N4-corrected intensities."
        )

    print("  Applying N4 bias field correction (matches training distribution)...")
    try:
        spacing = [float(z) for z in nib_img.header.get_zooms()[:3]]
        spacing = [s if s > 0 else 1.0 for s in spacing]
    except Exception:
        spacing = [1.0, 1.0, 1.0]

    try:
        img_ants = ants.from_numpy(raw_data, spacing=spacing)
        img_corrected = ants.n4_bias_field_correction(img_ants)
        mri_img = img_corrected.numpy().astype(np.float32)
    except Exception as e:
        raise RuntimeError(
            f"N4 bias field correction failed: {e}\n"
            "Cannot continue inference without N4 — results would be unreliable.\n"
            "Check that antspyx is properly installed and the input MRI is valid."
        ) from e

    # ── Step 3: Brain-masked percentile normalization (FIX 3) ─────────
    # Use 5th-percentile floor (robust to scanners with negative background)
    # instead of data>0 which misses valid voxels with slightly negative values.
    floor = float(np.percentile(mri_img, 5))
    brain_voxels = mri_img[mri_img > floor]
    if len(brain_voxels) > 0:
        p1  = float(np.percentile(brain_voxels, 1))
        p99 = float(np.percentile(brain_voxels, 99))
    else:
        p1, p99 = float(np.percentile(mri_img, 1)), float(np.percentile(mri_img, 99))

    mri_img = np.clip(mri_img, p1, p99)
    if p99 - p1 > 0:
        mri_img = (mri_img - p1) / (p99 - p1)

    # Segment all axial slices in batches
    seg_volume = np.zeros((H, W, D), dtype=np.int16)

    for batch_start in range(0, D, inference_batch):
        batch_end = min(batch_start + inference_batch, D)
        batch_slices = []

        for s in range(batch_start, batch_end):
            slc = mri_img[:, :, s]
            slc_t = torch.tensor(slc, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
            slc_t = torch.nn.functional.interpolate(
                slc_t.unsqueeze(0), size=(img_size, img_size),
                mode='bilinear', align_corners=False
            ).squeeze(0)  # (1, img_size, img_size)
            batch_slices.append(slc_t)

        batch_tensor = torch.stack(batch_slices, dim=0).to(device)  # (B, 1, img_size, img_size)

        with torch.no_grad():
            logits = model(batch_tensor)
            preds = torch.argmax(logits, dim=1).cpu()  # (B, img_size, img_size)

        # Resize predictions back to original (H, W) and store
        for i, s in enumerate(range(batch_start, batch_end)):
            pred_slice = preds[i].float().unsqueeze(0).unsqueeze(0)  # (1, 1, img_size, img_size)
            pred_resized = torch.nn.functional.interpolate(
                pred_slice, size=(H, W), mode='nearest'
            ).squeeze().numpy().astype(np.int16)
            seg_volume[:, :, s] = pred_resized

    # Print tissue breakdown (full volume)
    total = seg_volume.size
    print(f"3D Segmentation result ({H}x{W}x{D} = {total:,} voxels):")
    print(f"  Background : {(seg_volume==0).sum()/total*100:.1f}%")
    print(f"  CSF        : {(seg_volume==1).sum()/total*100:.1f}%")
    print(f"  Gray Matter: {(seg_volume==2).sum()/total*100:.1f}%")
    print(f"  White Matter:{(seg_volume==3).sum()/total*100:.1f}%")

    return seg_volume, nib_img


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
    parser.add_argument("--limit",          type=int,   default=None,
                        help="Limit the number of masks to generate (useful for testing on CPU)")
    parser.add_argument("--segment",        default=None,
                        help="Path to a single .nii file to segment using trained model")
    parser.add_argument("--output",         default=None,
                        help="Output path for 3D segmentation NIfTI (used with --segment)")
    parser.add_argument("--val-split",      type=float, default=0.2,
                        help="Validation split ratio (default: 0.2 = 80/20)")
    parser.add_argument("--resume",         action="store_true",
                        help="Resume training from existing checkpoints/unet_best.pth")

    args = parser.parse_args()

    print("=" * 70)
    print("U-Net MRI Segmentation — ANTsPy (HDF5 Windows fix applied)")
    print("=" * 70)
    print(f"MRI dir        : {args.mri_dir}")
    print(f"Mask dir       : {args.mask_dir}")
    print(f"Epochs         : {args.epochs}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Generate masks : {args.generate_masks}")
    print(f"Mask limit     : {args.limit}")
    print("=" * 70 + "\n")

    if args.segment:
        # Inference only mode — full 3D volume segmentation
        print(f"Segmenting: {args.segment}")
        ckpt  = os.path.join(args.save_dir, "unet_best.pth")
        model = UNet2D(in_channels=1, num_classes=4, base_filters=64)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model_state"])
        seg_volume, nib_img = segment_mri(model, args.segment)

        # Save output if --output specified
        if args.output:
            seg_nib = nib.Nifti1Image(seg_volume.astype(np.int16), nib_img.affine)
            seg_nib.header.set_data_dtype(np.int16)
            seg_nib.header.set_qform(nib_img.affine, code=1)
            seg_nib.header.set_sform(nib_img.affine, code=1)
            seg_nib.header.set_zooms(nib_img.header.get_zooms()[:3])
            nib.save(seg_nib, args.output)
            print(f"\nSaved 3D segmentation to: {args.output}")

    else:
        # Generate masks if requested
        if args.generate_masks:
            print("Step 1: Generating pseudo-labels with ANTsPy...\n")
            generate_masks_antspy(args.mri_dir, args.mask_dir, args.limit)

        # Train
        print("Step 2: Training U-Net...\n")
        train_unet(
            mri_dir       = args.mri_dir,
            mask_dir      = args.mask_dir,
            num_epochs    = args.epochs,
            batch_size    = args.batch_size,
            learning_rate = args.lr,
            save_dir      = args.save_dir,
            val_split     = args.val_split,
            resume        = args.resume,
        )
