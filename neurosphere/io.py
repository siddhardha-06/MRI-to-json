"""Loading T1 NIfTI files. Every v2 load path goes through load_t1.

Raw IXI T1 scans are stored PSR (third array axis = left->right). Code that indexes arrays
without reorienting silently picks the wrong anatomical plane, so load_t1 always returns the
array in RAS+ voxel order (nibabel.as_closest_canonical) and records the original axis codes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

from .provenance import sha256_file

_IXI_RE = re.compile(r"(IXI\d{3})-(Guys|HH|IOP)-")
_DERIVED_SUFFIX_RE = re.compile(r"(_n4|_seg)+$")


def parse_ixi_name(name: str) -> tuple[str | None, str | None]:
    """('IXI002', 'Guys') from any IXI file or folder name; (None, None) if it is not an IXI name."""
    m = _IXI_RE.search(Path(name).name)
    return (m.group(1), m.group(2)) if m else (None, None)


def strip_nifti_ext(name: str) -> str:
    name = Path(name).name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def is_derived(name: str) -> bool:
    """True for files written by the v1 pipeline (_n4, _n4_n4, _seg, _seg_seg), which are not raw scans."""
    return bool(_DERIVED_SUFFIX_RE.search(strip_nifti_ext(name)))


def axcodes(affine: np.ndarray) -> str:
    return "".join(nib.aff2axcodes(affine))


def voxel_volume_ml(affine: np.ndarray) -> float:
    """Volume of one voxel in mL (= cm^3) from the affine, valid for oblique and permuted grids."""
    return float(abs(np.linalg.det(np.asarray(affine, dtype=float)[:3, :3])) / 1000.0)


def to_ras(img: nib.spatialimages.SpatialImage) -> nib.spatialimages.SpatialImage:
    """Reorder/flip voxel axes so that array axes run (closest to) R, A, S. No resampling."""
    return nib.as_closest_canonical(img)


@dataclass
class T1Image:
    data: np.ndarray            # float32, RAS+ voxel order
    affine: np.ndarray          # 4x4 voxel->world affine of `data`
    meta: dict = field(default_factory=dict)


def load_t1(path: str | Path, compute_sha256: bool = True) -> T1Image:
    """Load a T1 NIfTI, reorient it to RAS+ and describe the original file.

    meta keys: source_file, sha256, shape_original, voxel_size_mm_original, orientation_original,
    dtype_on_disk, shape, voxel_size_mm, orientation, voxel_volume_ml, subject_id, site.
    """
    path = Path(path)
    img = nib.load(str(path))
    ras = to_ras(img)
    data = np.asarray(ras.get_fdata(dtype=np.float32))
    subject_id, site = parse_ixi_name(path.name)
    meta = {
        "source_file": path.name,
        "sha256": sha256_file(path) if compute_sha256 else None,
        "shape_original": [int(s) for s in img.shape[:3]],
        "voxel_size_mm_original": [round(float(z), 6) for z in img.header.get_zooms()[:3]],
        "orientation_original": axcodes(img.affine),
        "dtype_on_disk": str(img.get_data_dtype()),
        "shape": [int(s) for s in ras.shape[:3]],
        "voxel_size_mm": [round(float(z), 6) for z in ras.header.get_zooms()[:3]],
        "orientation": axcodes(ras.affine),
        "voxel_volume_ml": voxel_volume_ml(ras.affine),
        "subject_id": subject_id,
        "site": site,
    }
    return T1Image(data=data, affine=np.asarray(ras.affine, dtype=float), meta=meta)
