"""Preprocessing to the TransMorph atlas space.

raw T1 -> RAS (load_t1) -> N4 -> HD-BET brain mask -> WM intensity scaling to the FreeSurfer
target (110/255, as in the TransMorph pkl files) -> ANTs affine to the atlas T1.

Every ANTs image is built with its full orientation (origin, spacing, direction) from the NIfTI
affine. The v1 code used ants.from_numpy with spacing only, which discarded orientation;
`nib_to_ants` exists to make that impossible here.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import ants
import nibabel as nib
import numpy as np

from .atlas import WM_TARGET, atlas_affine, default_atlas_paths
from .io import load_t1

_LPS = np.diag([-1.0, -1.0, 1.0])


def nib_to_ants(data: np.ndarray, affine: np.ndarray) -> ants.ANTsImage:
    """ANTs image with the same voxel->world mapping as (data, affine). ITK works in LPS."""
    m = np.asarray(affine, float)
    spacing = np.linalg.norm(m[:3, :3], axis=0)
    direction = (_LPS @ m[:3, :3]) / spacing
    origin = _LPS @ m[:3, 3]
    return ants.from_numpy(np.ascontiguousarray(data, dtype=np.float32), origin=tuple(origin.tolist()),
                           spacing=tuple(spacing.tolist()), direction=direction)


def ants_to_affine(img: ants.ANTsImage) -> np.ndarray:
    """NIfTI (RAS) affine of an ANTs image; inverse of the mapping used in nib_to_ants."""
    d = np.asarray(img.direction) * np.asarray(img.spacing)
    a = np.eye(4)
    a[:3, :3] = _LPS @ d
    a[:3, 3] = _LPS @ np.asarray(img.origin)
    return a


_HDBET = None


def hdbet_predictor(device: str = "cuda"):
    """HD-BET (Isensee et al. 2019) nnU-Net predictor, loaded once per process, no test-time augmentation."""
    global _HDBET
    if _HDBET is None:
        import torch
        from HD_BET.hd_bet_prediction import get_hdbet_predictor
        _HDBET = get_hdbet_predictor(use_tta=False, device=torch.device(device), verbose=False)
    return _HDBET


def hdbet_mask(data_ras: np.ndarray, voxel_size: tuple[float, float, float], device: str = "cuda") -> np.ndarray:
    """Binary brain mask for a RAS array (nnU-Net expects SimpleITK (z, y, x) order and spacing)."""
    pred = hdbet_predictor(device)
    arr = np.ascontiguousarray(np.transpose(data_ras, (2, 1, 0))[None].astype(np.float32))
    seg = pred.predict_single_npy_array(arr, {"spacing": [float(s) for s in voxel_size[::-1]]}, None, None, False)
    return np.transpose(np.asarray(seg), (2, 1, 0)) > 0


def initial_wm_level(brain_values: np.ndarray) -> float:
    """First WM intensity estimate: median of the brightest of three Otsu classes inside the brain mask.
    Refined after segmentation from the U-Net's WM labels (see segment.py)."""
    from skimage.filters import threshold_multiotsu
    v = brain_values[brain_values > 0]
    t = threshold_multiotsu(v, classes=3)
    return float(np.median(v[v >= t[1]]))


@dataclass
class Preprocessed:
    native: ants.ANTsImage            # N4-corrected scan, RAS grid, full orientation
    native_affine: np.ndarray
    raw: np.ndarray                   # uncorrected scan, RAS grid (for IQMs that MRIQC computes on the original image)
    bias_field: np.ndarray            # N4 bias field, RAS grid (raw = corrected x field)
    brain_mask: np.ndarray            # RAS grid, bool
    atlas_image: np.ndarray           # (160, 192, 224) float32, brain in atlas space, WM ~ 110/255
    atlas_linear: np.ndarray          # same grid, N4 intensities before scaling (for WM refinement)
    fwdtransforms: list[str]          # ANTs transform: atlas points -> native points
    wm_level_native: float
    timings_s: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


_ATLAS = None
TEMPLATE = "train_mean_T1.nii.gz"   # mean of the 403 TransMorph training images (scripts/make_train_template.py)


def atlas_t1(root: str | Path | None = None) -> ants.ANTsImage:
    """Registration target. The voxel-wise mean of the training images, not the CycleMorph atlas.pkl:
    the latter is offset by ~4 voxels from where training images sit, and the current U-Net
    swaps left/right labels under such an offset."""
    global _ATLAS
    if _ATLAS is None:
        root = Path(root) if root else Path(__file__).resolve().parents[1]
        t1 = default_atlas_paths(root)[0].parent / TEMPLATE
        if not t1.exists():
            raise FileNotFoundError(f"{t1} missing; run scripts/make_atlas_nifti.py and scripts/make_train_template.py")
        img = nib.load(str(t1))
        if not np.allclose(img.affine, atlas_affine(img.shape)):
            raise ValueError("template NIfTI does not have the expected LIA affine")
        _ATLAS = nib_to_ants(np.asarray(img.dataobj, np.float32), img.affine)
    return _ATLAS


def scale_to_wm(img: np.ndarray, wm_level: float) -> np.ndarray:
    """Linear scaling so that WM -> 110/255, clipped to [0, 1] and quantised to 1/255 like the pkl files."""
    out = np.clip(img * (WM_TARGET / wm_level), 0.0, 1.0)
    return (np.round(out * 255.0) / 255.0).astype(np.float32)


def preprocess(path: str | Path, work_dir: str | Path, device: str = "cuda", n4_shrink: int = 4) -> Preprocessed:
    """Run all preprocessing for one scan. Transforms are written to work_dir (small .mat files)."""
    t = {}
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    t1img = load_t1(path)
    native_raw = nib_to_ants(t1img.data, t1img.affine)
    t["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    field = ants.n4_bias_field_correction(native_raw, shrink_factor=n4_shrink, return_bias_field=True).numpy().astype(np.float32)
    field = np.where(field > 1e-6, field, 1.0).astype(np.float32)
    native = native_raw.new_image_like((t1img.data / field).astype(np.float32))
    t["n4"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    n4 = native.numpy()
    mask = hdbet_mask(n4, tuple(t1img.meta["voxel_size_mm"]), device)
    t["skullstrip"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    brain = np.where(mask, n4, 0.0).astype(np.float32)
    wm0 = initial_wm_level(brain[mask])
    moving = native.new_image_like(brain)
    fixed = atlas_t1()
    prefix = str(work_dir / "native2atlas_")
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="Affine", outprefix=prefix,
                            aff_metric="mattes", aff_sampling=32, aff_random_sampling_rate=0.2)
    t["register"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    warped = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=reg["fwdtransforms"], interpolator="linear")
    atlas_linear = np.clip(warped.numpy(), 0, None).astype(np.float32)
    atlas_img = scale_to_wm(atlas_linear, wm0)
    t["warp"] = time.perf_counter() - t0
    for f in Path(work_dir).glob("native2atlas_*Warped.nii.gz"):  # ANTs may write warped images; keep only transforms
        os.remove(f)
    return Preprocessed(native=native, native_affine=t1img.affine, raw=t1img.data, bias_field=field, brain_mask=mask, atlas_image=atlas_img, atlas_linear=atlas_linear,
                        fwdtransforms=list(reg["fwdtransforms"]), wm_level_native=wm0, timings_s=t, meta=t1img.meta)
