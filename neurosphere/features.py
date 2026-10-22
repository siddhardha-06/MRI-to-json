"""Feature computation (feature set v2).

Phase 1 provides only the volume primitive used by every volumetric feature, so it can be
tested on phantoms now. The full v2 feature set arrives in Phase 3.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .io import voxel_volume_ml


def volumes_from_labels(labels: np.ndarray, affine: np.ndarray, label_ids: Iterable[int]) -> dict[int, float]:
    """Volume in mL of each label: voxel count x voxel volume from the affine (native space).

    The label map must be on the grid described by `affine` (e.g. the native scan grid after
    the inverse warp of Phase 2), never on a resampled or template grid.
    """
    labels = np.asarray(labels)
    vv = voxel_volume_ml(affine)
    ids, counts = np.unique(labels, return_counts=True)
    lookup = dict(zip(ids.tolist(), counts.tolist()))
    return {int(l): float(lookup.get(int(l), 0) * vv) for l in label_ids}


def asymmetry_index_pct(left: float, right: float) -> float | None:
    """Signed AI = (L - R) / ((L + R) / 2) x 100. None when L + R == 0."""
    denom = (left + right) / 2.0
    return None if denom == 0 else (left - right) / denom * 100.0


def tissue_intensity(corrected: np.ndarray, labels: np.ndarray, wm_level: float) -> dict:
    """Within-tissue intensity summaries of the N4-corrected image, normalised so that WM = 110
    (the FreeSurfer target used for the U-Net input; `wm_level` is the refined WM level in N4 units)."""
    from .qc import CSF_LABELS, GM_LABELS, WM_LABELS_QC
    x = corrected * (110.0 / wm_level)
    out = {"normalisation": "N4-corrected, scaled so that the U-Net's WM median = 110 (FreeSurfer T1 scale)"}
    for name, ls in (("gm", GM_LABELS), ("wm", WM_LABELS_QC), ("csf_ventricular", CSF_LABELS)):
        v = x[np.isin(labels, ls)]
        out[name] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "p05": float(np.percentile(v, 5)),
                     "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "n_voxels": int(v.size)}
    out["gm_wm_ratio"] = out["gm"]["mean"] / out["wm"]["mean"]
    out["gm_wm_contrast_pct"] = 100.0 * (out["wm"]["mean"] - out["gm"]["mean"]) / ((out["wm"]["mean"] + out["gm"]["mean"]) / 2.0)
    return out


def morphometry(result: dict) -> dict:
    """Feature set v2 morphometry from a pipeline result.json.

    Primary volume = 'jacobian' (template count x |det A|); the native-grid count is kept as a check.
    Returns blocks: volumes_ml, etiv_ml, normalized_volumes_pct_etiv, asymmetry_index_pct, tissue, brain_mask_ml.
    """
    from .dictionary import (ETIV_METHOD, ETIV_SCALE_ML, GM_LABELS, PAIRS, PARENCHYMA_LABELS, STRUCTURES,
                             VENTRICLE_LABELS, WM_LABELS)
    vol = {int(k): v for k, v in result["volumes_ml"].items()}
    v = {l: vol[l]["jacobian"] for l in STRUCTURES}
    volumes = {}
    for l, (key, _) in STRUCTURES.items():
        volumes[key] = {"value": round(v[l], 4), "native_grid_check": round(vol[l]["native"], 4)}
    for name, (l, r) in PAIRS.items():
        volumes[f"total_{name}"] = {"value": round(v[l] + v[r], 4)}
    etiv = ETIV_SCALE_ML * float(result["jacobian_det_atlas_to_native"])
    asym = {}
    for name, (l, r) in PAIRS.items():
        ai = asymmetry_index_pct(v[l], v[r])
        if ai is None:
            asym[name] = None; asym[f"{name}_reason"] = "left + right volume is 0"
        else:
            asym[name] = round(ai, 3)
    gm, wm = sum(v[l] for l in GM_LABELS), sum(v[l] for l in WM_LABELS)
    vent, par = sum(v[l] for l in VENTRICLE_LABELS), sum(v[l] for l in PARENCHYMA_LABELS)
    tissue = {"gray_matter_ml": round(gm, 3), "white_matter_ml": round(wm, 3), "ventricular_csf_ml": round(vent, 3),
              "brain_parenchyma_ml": round(par, 3), "brain_parenchymal_fraction": round(par / etiv, 5),
              "gm_wm_volume_ratio": round(gm / wm, 5), "ventricle_brain_ratio": round(vent / par, 5)}
    norm = {STRUCTURES[l][0]: round(100.0 * v[l] / etiv, 5) for l in STRUCTURES}
    return {"volumes_ml": volumes, "etiv_ml": {"value": round(etiv, 2), "method": ETIV_METHOD},
            "normalized_volumes_pct_etiv": norm, "asymmetry_index_pct": asym, "tissue": tissue,
            "brain_mask_ml": round(float(result["brain_mask_ml"]), 2)}
