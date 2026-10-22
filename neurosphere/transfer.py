"""Transfer a subject's TransMorph FreeSurfer labels onto our own grids (Phase 3: E2-FS and fine-tuning pairs).

The TransMorph image (FreeSurfer brainmask in talairach space) is affinely registered to our template-space U-Net
input of the SAME subject (work/pipeline_v2/<IXI>/template_t1.nii.gz; same 160x192x224 grid). Labels follow that
affine (template grid), or that affine and then our template -> native affine inverted (native grid). Nearest label.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import ants
import nibabel as nib
import numpy as np

from .preprocess import atlas_t1, nib_to_ants

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TMRegistration:
    ours: ants.ANTsImage        # our template-space U-Net input, [0, 1]
    tm_seg: ants.ANTsImage      # TransMorph FreeSurfer labels on the template grid (before registration)
    fwdtransforms: list         # our template points -> TransMorph points
    ncc: float                  # NCC after registration inside the union of both brain masks


def register_tm_to_ours(sid: str, k: int, split: str, pipe: Path) -> TMRegistration:
    img, seg = pickle.load(open(ROOT / "IXI_data" / split / f"subject_{k}.pkl", "rb"))
    tmpl = atlas_t1()
    ours = tmpl.new_image_like(np.asarray(nib.load(str(Path(pipe) / sid / "template_t1.nii.gz")).dataobj, np.float32) / 255.0)
    tm_img = tmpl.new_image_like(np.asarray(img, np.float32))
    tm_seg = tmpl.new_image_like(np.asarray(seg, np.float32))
    reg = ants.registration(fixed=ours, moving=tm_img, type_of_transform="Affine")
    w = reg["warpedmovout"].numpy(); f = ours.numpy()
    m = (f > 0.02) | (w > 0.02)
    return TMRegistration(ours, tm_seg, list(reg["fwdtransforms"]), float(np.corrcoef(f[m], w[m])[0, 1]))


def labels_on_template(r: TMRegistration) -> np.ndarray:
    lab = ants.apply_transforms(fixed=r.ours, moving=r.tm_seg, transformlist=r.fwdtransforms, interpolator="genericLabel")
    return np.rint(lab.numpy()).astype(np.int16)


def labels_on_native(r: TMRegistration, sid: str, pipe: Path, labels_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(our native labels, transferred FreeSurfer labels on the same grid, native affine). Transforms come from
    `pipe`; our labels from `labels_dir` (default `pipe`), e.g. a re-segmentation with another checkpoint."""
    lab_nat_img = nib.load(str(Path(labels_dir or pipe) / sid / "labels_native.nii.gz"))
    ours_nat = np.asarray(lab_nat_img.dataobj).astype(np.int32)
    native_ref = nib_to_ants(ours_nat.astype(np.float32), lab_nat_img.affine)
    # points on the native grid -> template (inverse of template->native) -> TransMorph space (registration affine).
    # ANTs applies the FIRST listed transform to fixed-space points first (checked on a synthetic scale+shift).
    fs = ants.apply_transforms(fixed=native_ref, moving=r.tm_seg,
                               transformlist=[str(Path(pipe) / sid / "native2atlas_0GenericAffine.mat")] + r.fwdtransforms,
                               whichtoinvert=[True, False], interpolator="genericLabel")
    return ours_nat, np.rint(fs.numpy()).astype(np.int32), lab_nat_img.affine
