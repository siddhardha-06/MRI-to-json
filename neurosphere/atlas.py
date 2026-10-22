"""The TransMorph IXI atlas space.

The TransMorph IXI volumes are FreeSurfer brainmask.mgz images (WM normalised to 110, /255)
affine-aligned with talairach.xfm and cropped to 160 x 192 x 224 at 1 mm. FreeSurfer's conformed
orientation is LIA, and the label centroids confirm it: axis 0 runs right->left, axis 1
superior->inferior, axis 2 posterior->anterior. The crop offset is not published, so the world
origin is placed at the volume centre; only the axis directions and the 1 mm spacing matter for
registration and volumes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SHAPE = (160, 192, 224)
WM_TARGET = 110.0 / 255.0   # FreeSurfer intensity-normalisation target, as found in the pkl files


def atlas_affine(shape=SHAPE) -> np.ndarray:
    """1 mm LIA affine with the volume centre at world (0, 0, 0)."""
    a = np.zeros((4, 4))
    a[0, 0] = -1.0   # axis 0 -> Left   (x decreases)
    a[2, 1] = -1.0   # axis 1 -> Inferior (z decreases)
    a[1, 2] = 1.0    # axis 2 -> Anterior (y increases)
    a[3, 3] = 1.0
    a[:3, 3] = -a[:3, :3] @ ((np.asarray(shape) - 1) / 2.0)
    return a


def default_atlas_paths(root: str | Path) -> tuple[Path, Path]:
    d = Path(root) / "data" / "atlas"
    return d / "atlas_T1.nii.gz", d / "atlas_labels.nii.gz"
