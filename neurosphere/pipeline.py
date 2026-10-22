"""One-subject pipeline run with a resumable cache (Phase 2, extended in Phase 3).

Cache per subject in work/pipeline/<subject>/ (so preprocessing never has to be re-run):
  native2atlas_0GenericAffine.mat   ~1 KB    affine, template points -> native points
  labels_native.nii.gz              ~0.4 MB  U-Net labels on the native RAS grid (E2 Dice, E3)
  template_t1.nii.gz                ~2 MB    skull-stripped, WM-scaled template-space image, uint8 = the U-Net
                                             input x 255 (E3 perturbations, E4 image-CNN baseline, mapping check)
  brain_mask_native.nii.gz          ~0.05 MB HD-BET mask, native RAS grid
  bias_field_ds4.nii.gz             ~0.1 MB  N4 bias field, 4x downsampled, float32 (smooth; raw = corrected x field)
  result.json                       ~15 KB   volumes, IQMs, intensity, timings, provenance
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from .atlas import atlas_affine
from .features import tissue_intensity
from .preprocess import preprocess, scale_to_wm
from .qc import compute_iqms, head_air_masks
from .provenance import provenance, sha256_file
from .segment import Segmenter, segment, side_check

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work" / "pipeline"


def run_subject(path: str | Path, subject: str, seg: Segmenter, work_root: Path = WORK, force: bool = False,
                keep_atlas_image: bool = False) -> dict:
    out = Path(work_root) / subject
    res_file = out / "result.json"
    if res_file.exists() and not force and not keep_atlas_image:
        return json.loads(res_file.read_text())
    t0 = time.perf_counter()
    pre = preprocess(path, out)
    s = segment(pre, seg)
    total = time.perf_counter() - t0
    nib.save(nib.Nifti1Image(s.labels_native, s.native_affine), str(out / "labels_native.nii.gz"))
    t1 = time.perf_counter()
    unet_input = scale_to_wm(pre.atlas_linear, s.wm_level_final)
    nib.save(nib.Nifti1Image(np.round(unet_input * 255).astype(np.uint8), atlas_affine(unet_input.shape)), str(out / "template_t1.nii.gz"))
    nib.save(nib.Nifti1Image(pre.brain_mask.astype(np.uint8), s.native_affine), str(out / "brain_mask_native.nii.gz"))
    ds_aff = s.native_affine.copy(); ds_aff[:3, :3] = ds_aff[:3, :3] * 4
    nib.save(nib.Nifti1Image(pre.bias_field[::4, ::4, ::4].astype(np.float32), ds_aff), str(out / "bias_field_ds4.nii.gz"))
    head, air = head_air_masks(pre.raw)
    corrected = pre.native.numpy()
    iqms = compute_iqms(pre.raw, corrected, s.labels_native, head, air, pre.bias_field)
    intensity = tissue_intensity(corrected, s.labels_native, s.wm_level_final)
    vv = float(abs(np.linalg.det(s.native_affine[:3, :3]))) / 1000.0
    masks = {"head_ml": float(head.sum() * vv), "air_ml": float(air.sum() * vv)}
    t_qc = time.perf_counter() - t1
    res = {
        "subject": subject, "source_file": Path(path).name, "sha256": sha256_file(path),
        "orientation_original": pre.meta["orientation_original"], "voxel_size_mm": pre.meta["voxel_size_mm_original"],
        "volumes_ml": {str(k): v for k, v in s.volumes_ml.items()},
        "jacobian_det_atlas_to_native": s.jacobian_det,
        "side_check_native": side_check(s.labels_native, s.native_affine),
        "wm_level": {"initial_otsu": pre.wm_level_native, "refined_unet": s.wm_level_final},
        "brain_mask_ml": float(pre.brain_mask.sum() * abs(np.linalg.det(pre.native_affine[:3, :3])) / 1000.0),
        "iqms": iqms, "qc_masks": masks, "intensity": intensity,
        "timings_s": {**pre.timings_s, **s.timings_s, "total": total, "qc_and_save": t_qc},
        "checkpoint": seg.info,
        "provenance": provenance(),
    }
    res_file.write_text(json.dumps(res, indent=2))
    if keep_atlas_image:
        res["_atlas_image"] = unet_input
        res["_labels_atlas"] = s.labels_atlas
    return res
