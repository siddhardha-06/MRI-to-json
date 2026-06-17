"""
loader.py
---------
Load brain MRI volumes from NIfTI (.nii/.nii.gz), MGZ (FreeSurfer),
or ANALYZE (.img/.hdr) format — OASIS-1 RAW files are ANALYZE format.
Extracts 2D axial slices for downstream processing.
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Union, List, Tuple, Optional


def load_volume(path: Union[str, Path]) -> Tuple[np.ndarray, object]:
    """
    Load a NIfTI, MGZ, or ANALYZE volume.

    Returns:
        data: 3D numpy array
        affine: 4x4 affine matrix
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MRI file not found: {path}")

    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine

    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]

    return data, affine


def find_mri_file(subject_dir: Union[str, Path]) -> Optional[Path]:
    """
    Given an OASIS-1 subject directory, find the primary MRI file.

    OASIS-1 RAW structure:
        OAS1_0001_MR1/
            RAW/
                OAS1_0001_MR1_mpr-1_anon.img   ← ANALYZE format (use mpr-1)
                OAS1_0001_MR1_mpr-1_anon.hdr
    """
    subject_dir = Path(subject_dir)

    # Prefer NIfTI if someone already converted
    for pattern in ["**/*.nii.gz", "**/*.nii"]:
        matches = list(subject_dir.glob(pattern))
        if matches:
            return matches[0]

    # OASIS-1 primary: mpr-1 ANALYZE .img in RAW/
    raw_dir = subject_dir / "RAW"
    if raw_dir.exists():
        # Prefer mpr-1 (first acquisition, best quality)
        mpr1 = list(raw_dir.glob("*mpr-1_anon.img"))
        if mpr1:
            return mpr1[0]
        # Fallback: any .img
        any_img = list(raw_dir.glob("*.img"))
        if any_img:
            return any_img[0]

    # MGZ fallback
    mgz = list(subject_dir.glob("**/*.mgz"))
    if mgz:
        return mgz[0]

    # Any .img anywhere
    any_img = list(subject_dir.glob("**/*.img"))
    if any_img:
        return any_img[0]

    return None


def extract_axial_slices(
    volume: np.ndarray,
    n_slices: int = 5,
    strategy: str = "central_spread",
) -> List[Tuple[int, np.ndarray]]:
    """
    Extract representative 2D axial slices from a 3D volume.
    """
    depth = volume.shape[2]

    if strategy == "central":
        idx = depth // 2
        return [(idx, volume[:, :, idx])]

    elif strategy == "central_spread":
        center = depth // 2
        half = n_slices // 2
        step = max(1, depth // (n_slices * 2))
        indices = [center + (i - half) * step for i in range(n_slices)]
        indices = [max(0, min(depth - 1, idx)) for idx in indices]
        return [(idx, volume[:, :, idx]) for idx in indices]

    elif strategy == "brain_mask":
        threshold = np.percentile(volume[volume > 0], 10) if np.any(volume > 0) else 0
        scores = []
        for z in range(depth):
            sl = volume[:, :, z]
            brain_voxels = sl[sl > threshold]
            score = brain_voxels.mean() * len(brain_voxels) if len(brain_voxels) > 0 else 0
            scores.append((score, z))
        scores.sort(reverse=True)
        top_indices = sorted([z for _, z in scores[:n_slices]])
        return [(idx, volume[:, :, idx]) for idx in top_indices]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def reorient_to_ras(volume: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Reorient volume to RAS+ standard orientation so axial = Z axis.
    """
    import nibabel.orientations as ornt

    current_ornt = ornt.io_orientation(affine)
    ras_ornt = ornt.axcodes2ornt(("R", "A", "S"))
    transform = ornt.ornt_transform(current_ornt, ras_ornt)
    return ornt.apply_orientation(volume, transform)


def load_subject(
    subject_dir: Union[str, Path],
    n_slices: int = 5,
    strategy: str = "central_spread",
) -> Tuple[List[Tuple[int, np.ndarray]], str]:
    """
    Find, load, reorient, and extract slices for one OASIS-1 subject.

    Returns:
        slices: list of (slice_index, 2D_array)
        mri_path: path to the file that was loaded
    """
    mri_path = find_mri_file(subject_dir)
    if mri_path is None:
        raise FileNotFoundError(f"No MRI file found in {subject_dir}")

    volume, affine = load_volume(mri_path)
    volume = reorient_to_ras(volume, affine)
    slices = extract_axial_slices(volume, n_slices=n_slices, strategy=strategy)

    return slices, str(mri_path)
