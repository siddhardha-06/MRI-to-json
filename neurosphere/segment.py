"""3D U-Net segmentation in atlas space, mapped back to native space.

Do not use L/R-flip TTA with label swapping on the current checkpoint.

Volumes are computed two independent ways and both are reported:
  * native:   labels resampled to the native grid (inverse affine, ANTs genericLabel) x native voxel volume;
  * jacobian: atlas-grid label counts x 1 mm^3 x |det(A)|, where A maps atlas points to native points.
For an affine transform the second is exact for the segmented shape; the first adds resampling error.
The primary value (pre-specified, before any comparison with FastSurfer) is `jacobian`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import ants
import numpy as np
import torch

from .preprocess import Preprocessed, atlas_t1, scale_to_wm

ROOT = Path(__file__).resolve().parents[1]
WM_LABELS = (2, 41)
DEFAULT_CHECKPOINT = "unet3d_v2_best.pth"
BASELINE_CHECKPOINT = "unet3d_best.pth"


class Segmenter:
    """Wraps training/train_unet3d.UNet3D. Loads the checkpoint once."""

    def __init__(self, ckpt: str | Path | None = None, device: str = "cuda", base: int = 16):
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from training.train_unet3d import ALL_LABELS, UNet3D
        from .provenance import sha256_file
        # default: retrained Option-A model (Phase 3 decision 1); unet3d_best.pth is the paper's baseline row
        self.ckpt = Path(ckpt) if ckpt else ROOT / "checkpoints" / DEFAULT_CHECKPOINT
        self.device = torch.device(device)
        ck = torch.load(self.ckpt, map_location=self.device, weights_only=False)
        self.model = UNet3D(base=base).to(self.device).eval()
        self.model.load_state_dict(ck["model"])
        self.labels = np.asarray(ALL_LABELS, dtype=np.int16)
        self.info = {"file": self.ckpt.name, "sha256": sha256_file(self.ckpt), "epoch_index": ck.get("epoch")}

    @torch.no_grad()
    def __call__(self, atlas_img: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(atlas_img, np.float32))[None, None].to(self.device)
        return self.labels[self.model(x).argmax(1)[0].cpu().numpy()]


def affine_det(fwdtransforms: list[str]) -> float:
    """|det| of the 3x3 part of the ANTs affine (atlas points -> native points)."""
    tx = ants.read_transform(fwdtransforms[0])
    return float(abs(np.linalg.det(np.asarray(tx.parameters[:9]).reshape(3, 3))))


@dataclass
class Segmentation:
    labels_atlas: np.ndarray            # (160,192,224) FreeSurfer label IDs
    labels_native: np.ndarray           # native RAS grid
    native_affine: np.ndarray
    volumes_ml: dict                    # {label: {"jacobian": v, "native": v}}
    wm_level_final: float
    jacobian_det: float
    timings_s: dict = field(default_factory=dict)


def segment(pre: Preprocessed, seg: Segmenter, refine_wm: bool = True) -> Segmentation:
    t = {}
    t0 = time.perf_counter()
    lab = seg(pre.atlas_image)
    wm_level = pre.wm_level_native
    if refine_wm:
        # second pass: WM level = median N4 intensity under the U-Net's own WM labels
        wm = np.isin(lab, WM_LABELS)
        if wm.sum() > 1000:
            wm_level = float(np.median(pre.atlas_linear[wm]))
            lab = seg(scale_to_wm(pre.atlas_linear, wm_level))
    t["unet"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    lab_img = atlas_t1().new_image_like(lab.astype(np.float32))
    nat = ants.apply_transforms(fixed=pre.native, moving=lab_img, transformlist=pre.fwdtransforms,
                                whichtoinvert=[True], interpolator="genericLabel")
    lab_nat = np.rint(nat.numpy()).astype(np.int16)
    t["inverse_warp"] = time.perf_counter() - t0

    det = affine_det(pre.fwdtransforms)
    vv_native = float(abs(np.linalg.det(pre.native_affine[:3, :3]))) / 1000.0
    ids_a, cnt_a = np.unique(lab, return_counts=True)
    ids_n, cnt_n = np.unique(lab_nat, return_counts=True)
    ca, cn = dict(zip(ids_a.tolist(), cnt_a.tolist())), dict(zip(ids_n.tolist(), cnt_n.tolist()))
    vols = {int(l): {"jacobian": ca.get(int(l), 0) * det / 1000.0, "native": cn.get(int(l), 0) * vv_native}
            for l in seg.labels if l != 0}
    return Segmentation(labels_atlas=lab, labels_native=lab_nat, native_affine=pre.native_affine, volumes_ml=vols,
                        wm_level_final=wm_level, jacobian_det=det, timings_s=t)


LR_PAIRS = ((2, 41), (3, 42), (4, 43), (7, 46), (8, 47), (10, 49), (11, 50), (12, 51), (13, 52), (17, 53), (18, 54), (26, 58), (28, 60))


def side_check(labels: np.ndarray, affine: np.ndarray) -> dict:
    """For each L/R pair, x_RAS(centroid of left label) - x_RAS(centroid of right label) in mm.
    Must be negative (left hemisphere at smaller x in RAS). A side swap makes every pair positive."""
    import nibabel as nib
    out = {}
    for l, r in LR_PAIRS:
        a, b = np.argwhere(labels == l), np.argwhere(labels == r)
        if len(a) == 0 or len(b) == 0:
            out[f"{l}-{r}"] = None
            out[f"{l}-{r}_reason"] = "label absent from the prediction"
            continue
        xa = nib.affines.apply_affine(affine, a.mean(0))[0]
        xb = nib.affines.apply_affine(affine, b.mean(0))[0]
        out[f"{l}-{r}"] = round(float(xa - xb), 2)
    vals = [v for k, v in out.items() if not k.endswith("_reason") and v is not None]
    out["n_correct"] = int(sum(v < 0 for v in vals))
    out["n_swapped"] = int(sum(v > 0 for v in vals))
    out["all_correct"] = bool(vals) and all(v < 0 for v in vals)
    return out
