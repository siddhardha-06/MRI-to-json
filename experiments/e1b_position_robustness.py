r"""E1b: robustness of the U-Net to small translations of the input (found in Phase 2).

Registering raw scans to the CycleMorph atlas left them ~4 voxels (left-right) away from where the
TransMorph training images sit (training SD of the brain centroid along axis 0: 0.45 voxels), and
the shipped U-Net then labelled the left hemisphere as right. This experiment measures that
effect on the TransMorph TEST images (in-domain, FreeSurfer labels as truth): image and labels
are shifted together by d voxels along an axis (zero fill), the model predicts, and we record
mean foreground Dice and whether predicted left/right structures land on the wrong side.

Usage: python experiments/e1b_position_robustness.py [--ckpt unet3d_best.pth] [--limit N]
Output: results/e1b_position_robustness/<ckpt-stem>/ (per_subject.csv, metrics.json)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosphere.metrics import bootstrap_ci, dice  # noqa: E402
from neurosphere.provenance import provenance, sha256_file  # noqa: E402
from training.train_unet3d import ALL_LABELS, LR_PAIRS, UNet3D, pkload  # noqa: E402

SHIFTS = {0: [-6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6], 2: [-6, -3, 0, 3, 6]}   # axis 0 = L-R, axis 2 = A-P


def shift(a, d, axis):
    out = np.zeros_like(a)
    src = [slice(None)] * 3; dst = [slice(None)] * 3
    if d > 0:
        src[axis], dst[axis] = slice(0, -d), slice(d, None)
    elif d < 0:
        src[axis], dst[axis] = slice(-d, None), slice(0, d)
    out[tuple(dst)] = a[tuple(src)]
    return out


NL = len(ALL_LABELS)
LUT_IDX = np.zeros(256, np.int64)
for _i, _l in enumerate(ALL_LABELS):
    LUT_IDX[_l] = _i


def fast_stats(pred_idx, true_idx):
    """Per-class Dice from one confusion count, and L/R side errors from per-class axis-0 centroids (LIA: left = higher index)."""
    conf = np.bincount((pred_idx * NL + true_idx).ravel(), minlength=NL * NL).reshape(NL, NL)
    tp = np.diag(conf).astype(float); den = conf.sum(0) + conf.sum(1)
    d = np.where(den > 0, 2 * tp / np.maximum(den, 1), np.nan)
    x = np.broadcast_to(np.arange(pred_idx.shape[0])[:, None, None], pred_idx.shape).ravel()
    cp = np.bincount(pred_idx.ravel(), weights=x, minlength=NL) / np.maximum(np.bincount(pred_idx.ravel(), minlength=NL), 1)
    ct = np.bincount(true_idx.ravel(), weights=x, minlength=NL) / np.maximum(np.bincount(true_idx.ravel(), minlength=NL), 1)
    np_ = np.bincount(pred_idx.ravel(), minlength=NL); nt = np.bincount(true_idx.ravel(), minlength=NL)
    wrong = checked = 0
    for l, r in LR_PAIRS:
        il, ir = LUT_IDX[l], LUT_IDX[r]
        if min(np_[il], np_[ir], nt[il], nt[ir]) == 0:
            continue
        checked += 1
        wrong += int(cp[il] < (ct[il] + ct[ir]) / 2)
    return float(np.nanmean(d[1:])), wrong, checked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="unet3d_best.pth")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    ck_path = ROOT / "checkpoints" / a.ckpt
    ck = torch.load(ck_path, map_location="cuda", weights_only=False)
    m = UNet3D(base=16).cuda().eval(); m.load_state_dict(ck["model"])
    files = sorted((ROOT / "IXI_data" / "Test").glob("*.pkl"))[: a.limit]
    rows = []
    t0 = time.perf_counter()
    for f in files:
        img, seg = pkload(f)
        seg_idx = LUT_IDX[np.clip(seg, 0, 255)] * np.isin(seg, ALL_LABELS)   # labels outside the 30 classes -> background, as in training
        for axis, ds in SHIFTS.items():
            for d in ds:
                x, t = shift(img, d, axis), shift(seg_idx, d, axis)
                with torch.no_grad():
                    p = m(torch.from_numpy(x.astype(np.float32))[None, None].cuda()).argmax(1)[0].cpu().numpy()
                md, wrong, checked = fast_stats(p, t)
                rows.append({"file": f.name, "axis": axis, "shift_vox": d, "mean_fg_dice": md,
                             "pairs_wrong_side": wrong, "pairs_checked": checked})
        print(f.name, f"{time.perf_counter() - t0:.0f}s", flush=True)
    out = ROOT / "results" / "e1b_position_robustness" / Path(a.ckpt).stem
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_subject.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = {}
    for axis, ds in SHIFTS.items():
        for d in ds:
            rr = [r for r in rows if r["axis"] == axis and r["shift_vox"] == d]
            dv = np.array([r["mean_fg_dice"] for r in rr])
            summ[f"axis{axis}_{d:+d}"] = {"n": len(rr), "mean_fg_dice": float(dv.mean()), "ci95": list(bootstrap_ci(dv)),
                                          "subjects_with_any_side_error": int(sum(r["pairs_wrong_side"] > 0 for r in rr)),
                                          "subjects_with_majority_swapped": int(sum(r["pairs_wrong_side"] > r["pairs_checked"] / 2 for r in rr))}
    res = {"checkpoint": {"file": ck_path.name, "sha256": sha256_file(ck_path)}, "n_subjects": len(files),
           "axes": {"0": "left-right (LIA axis 0)", "2": "posterior-anterior"}, "by_shift": summ,
           "provenance": provenance(data_subset="IXI_data/Test" + (f" first {a.limit}" if a.limit else ""))}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    for k, v in summ.items():
        print(f"{k:10s} Dice {v['mean_fg_dice']:.3f}  any-side-error {v['subjects_with_any_side_error']}/{v['n']}  majority-swapped {v['subjects_with_majority_swapped']}/{v['n']}")


if __name__ == "__main__":
    main()
