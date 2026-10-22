"""Quality control.

Phase 1: no IQMs are computed yet, so the gate status is "not_computed". Nothing is simulated
(the v1 fallback produced IQMs from hand-picked constants and a score that could
never fail). In-house IQMs following MRIQC's documented definitions and the data-driven gate
(robust z against the IXI training split) are implemented in Phase 3.

Official MRIQC output, when the user has run it, is stored separately and verbatim under
quality.mriqc_official. Missing IQMs stay null; they are never replaced by 0.
"""
from __future__ import annotations

import math

from .nulls import set_null

IQMS_NOT_COMPUTED = ("In-house image-quality metrics are implemented in Phase 3; "
                     "no simulated values are produced.")
GATE_NOT_COMPUTED = "No quality gate is applied until the in-house IQMs and data-driven thresholds exist (Phase 3)."
MRIQC_NOT_RUN = "Official MRIQC has not been run for this scan (optional; requires Docker)."

# IQMs reported by MRIQC for T1w images that the dashboard displays. Others are kept too.
CORE_MRIQC_IQMS = ("cjv", "cnr", "efc", "fber", "snr_csf", "snr_gm", "snr_wm", "snr_total", "snrd_total", "qi_1", "qi_2", "wm2max")


def parse_mriqc_official(mriqc_json: dict, image_id: str | None = None) -> dict:
    """Keep every finite numeric top-level IQM from an MRIQC T1w JSON. Core IQMs that are missing become null + reason."""
    out = {"source": "nipreps/mriqc", "image_id": image_id} if image_id else {"source": "nipreps/mriqc"}
    iqms = {}
    for k, v in mriqc_json.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if math.isfinite(float(v)):
            iqms[k] = float(v)
    for k in CORE_MRIQC_IQMS:
        if k not in iqms:
            set_null(iqms, k, "not present in the MRIQC output")
    out["iqms"] = dict(sorted(iqms.items()))
    return out


def quality_block(mriqc_official: dict | None = None) -> dict:
    block: dict = {}
    set_null(block, "iqms", IQMS_NOT_COMPUTED)
    gate = {"status": "not_computed", "reasons": [GATE_NOT_COMPUTED]}
    set_null(gate, "reference", "no reference distribution yet (Phase 3: IXI training split, robust z)")
    block["gate"] = gate
    if mriqc_official is None:
        set_null(block, "mriqc_official", MRIQC_NOT_RUN)
    else:
        block["mriqc_official"] = mriqc_official
    return block


# ── In-house image-quality metrics (Phase 3) ─────────────────────────────────
# Definitions follow MRIQC's documented T1w IQMs (mriqc.readthedocs.io, "IQMs for structural images";
# implementation in mriqc.qc.anatomical). The tissue classes come from our native-space U-Net labels instead
# of FSL FAST, so values are comparable in spirit, not identical, to official MRIQC (see docs/QC.md).
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

GM_LABELS = (3, 42, 8, 47, 10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60)
WM_LABELS_QC = (2, 41, 7, 46)
CSF_LABELS = (4, 43, 14, 15)          # ventricular CSF only (our label set has no sulcal CSF class)
DIETRICH_FACTOR = 1.0 / np.sqrt(2.0 / (4.0 - np.pi))


def head_air_masks(raw: np.ndarray, dilate_vox: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Head mask: Otsu threshold of the raw image, largest connected component, holes filled per axial
    slice and in 3D, closed. Air mask: everything outside the head mask dilated by `dilate_vox`."""
    from skimage.filters import threshold_otsu
    v = raw[raw > 0]
    head = raw > threshold_otsu(v[:: max(1, v.size // 2_000_000)])
    lab, n = ndimage.label(head)
    if n > 1:
        head = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    head = ndimage.binary_closing(head, iterations=3)
    head = np.stack([ndimage.binary_fill_holes(head[:, :, k]) for k in range(head.shape[2])], axis=2)
    head = ndimage.binary_fill_holes(head)
    air = ~ndimage.binary_dilation(head, iterations=dilate_vox)
    return head, air


def compute_iqms(raw: np.ndarray, corrected: np.ndarray, labels: np.ndarray, head: np.ndarray, air: np.ndarray,
                 bias_field: np.ndarray | None = None) -> dict:
    """IQMs on the native grid. corrected = N4-corrected image; raw = original image."""
    gm, wm, csf = (np.isin(labels, ls) for ls in (GM_LABELS, WM_LABELS_QC, CSF_LABELS))
    out = {}
    if min(gm.sum(), wm.sum(), air.sum()) < 100:
        for k in ("cjv", "cnr", "snr_gm", "snr_wm", "snr_csf", "snr_total", "snrd_gm", "snrd_wm", "snrd_csf", "snrd_total", "efc", "fber", "wm2max"):
            set_null(out, k, "tissue or air mask too small to compute")
        return out
    x = corrected
    mu = {t: float(x[m].mean()) for t, m in (("gm", gm), ("wm", wm), ("csf", csf)) if m.sum() > 1}
    sd = {t: float(x[m].std(ddof=1)) for t, m in (("gm", gm), ("wm", wm), ("csf", csf)) if m.sum() > 1}
    sd_air = float(x[air].std(ddof=1))
    out["cjv"] = (sd["wm"] + sd["gm"]) / abs(mu["wm"] - mu["gm"])                      # Ganzetti et al. 2016
    out["cnr"] = abs(mu["gm"] - mu["wm"]) / float(np.sqrt(sd_air ** 2 + sd["gm"] ** 2 + sd["wm"] ** 2))
    for t, m in (("gm", gm), ("wm", wm), ("csf", csf)):
        n = int(m.sum())
        out[f"snr_{t}"] = mu[t] / (sd[t] * np.sqrt(n / (n - 1))) if t in mu else None
    out["snr_total"] = float(np.mean([out["snr_gm"], out["snr_wm"], out["snr_csf"]]))
    # Dietrich et al. 2007: tissue mean over the Rayleigh-corrected air SD, on the ORIGINAL image
    sd_air_raw = float(raw[air].std(ddof=1))
    for t, m in (("gm", gm), ("wm", wm), ("csf", csf)):
        out[f"snrd_{t}"] = DIETRICH_FACTOR * float(raw[m].mean()) / sd_air_raw if sd_air_raw > 0 else None
    out["snrd_total"] = float(np.mean([out["snrd_gm"], out["snrd_wm"], out["snrd_csf"]])) if sd_air_raw > 0 else None
    # EFC (Atkinson et al. 1997), normalised by its maximum, over the whole volume
    v = np.abs(x.ravel()).astype(np.float64)
    nvox = v.size
    efc_max = nvox * (1.0 / np.sqrt(nvox)) * np.log(1.0 / np.sqrt(nvox))
    b = np.sqrt((v ** 2).sum())
    out["efc"] = float((1.0 / efc_max) * np.sum((v / b) * np.log((v + 1e-16) / b)))
    # FBER: median energy inside the head over median energy in the air mask
    bg = float(np.median(np.abs(x[air]) ** 2))
    if bg > 1e-3:
        out["fber"] = float(np.median(np.abs(x[head]) ** 2) / bg)
    else:
        set_null(out, "fber", "median air-mask energy is 0 (zero-filled background); MRIQC returns -1 in this case")
    out["wm2max"] = float(np.median(x[wm]) / np.percentile(x, 99.95))
    if bias_field is not None:
        f = bias_field[head]
        out["inu_med"] = float(np.median(f))
        out["inu_range"] = float(np.percentile(f, 95) - np.percentile(f, 5))
    for k, val in list(out.items()):
        if val is None and not k.endswith("_reason") and f"{k}_reason" not in out:
            out[f"{k}_reason"] = "zero air-mask standard deviation"
    return {k: (round(float(v), 6) if isinstance(v, float) else v) for k, v in out.items()}


# ── Data-driven quality gate ───────────────────────────────
GATE_IQMS = ("cjv", "cnr", "snr_total", "snrd_total", "efc", "wm2max", "inu_range")
GATE_WARN, GATE_FAIL = 2.5, 3.5


def fit_gate_reference(iqms_by_subject: dict[str, dict], train_ids) -> dict:
    """Robust reference per IQM from TRAIN subjects only: median and 1.4826 x MAD."""
    ref = {}
    for q in GATE_IQMS:
        v = np.array([iqms_by_subject[s].get(q) for s in train_ids if iqms_by_subject.get(s, {}).get(q) is not None], float)
        if len(v) < 20:
            continue
        med = float(np.median(v)); mad = float(1.4826 * np.median(np.abs(v - med)))
        if mad > 0:
            ref[q] = {"median": med, "robust_sd": mad, "n_train": int(len(v))}
    return {"iqms": ref, "warn_abs_z": GATE_WARN, "fail_abs_z": GATE_FAIL,
            "reference": "IXI train split (data/splits/ixi_final.csv), robust z = (x - median) / (1.4826 MAD)"}


def apply_gate(iqms: dict, reference: dict) -> dict:
    z, reasons, status = {}, [], "pass"
    for q, r in reference["iqms"].items():
        x = iqms.get(q)
        if x is None:
            continue
        zz = (x - r["median"]) / r["robust_sd"]
        z[q] = round(float(zz), 2)
        if abs(zz) > reference["fail_abs_z"]:
            status = "fail"; reasons.append(f"{q}: |z| = {abs(zz):.1f} > {reference['fail_abs_z']}")
        elif abs(zz) > reference["warn_abs_z"]:
            status = "warn" if status == "pass" else status; reasons.append(f"{q}: |z| = {abs(zz):.1f} > {reference['warn_abs_z']}")
    return {"status": status, "reasons": reasons, "robust_z": z, "reference": reference["reference"],
            "thresholds": {"warn_abs_z": reference["warn_abs_z"], "fail_abs_z": reference["fail_abs_z"]}}
