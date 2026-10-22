"""Segmentation agreement metrics (E1, E2). Absent-in-both gives NaN, never 1.0."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    denom = a.sum() + b.sum()
    return float("nan") if denom == 0 else float(2.0 * np.logical_and(a, b).sum() / denom)


def _border(m: np.ndarray) -> np.ndarray:
    return m & ~ndimage.binary_erosion(m, border_value=0)


def surface_distances(a: np.ndarray, b: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> tuple[float, float]:
    """(HD95, ASSD) between the borders of masks a and b, in the units of `spacing`.

    HD95 is the 95th percentile of the pooled symmetric border-to-border distances (as in
    medpy.metric.hd95); ASSD is their mean. NaN if either mask is empty.
    """
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if not a.any() or not b.any():
        return float("nan"), float("nan")
    union = a | b
    idx = np.argwhere(union)
    lo = np.maximum(idx.min(0) - 2, 0)
    hi = np.minimum(idx.max(0) + 3, union.shape)
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    a, b = a[sl], b[sl]
    ba, bb = _border(a), _border(b)
    da = ndimage.distance_transform_edt(~bb, sampling=spacing)[ba]  # a-border -> nearest b-border
    db = ndimage.distance_transform_edt(~ba, sampling=spacing)[bb]
    d = np.concatenate([da, db])
    return float(np.percentile(d, 95)), float(d.mean())


def bootstrap_ci(x, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))
