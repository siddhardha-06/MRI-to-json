r"""E1c: shipped U-Net (v1) vs retrained U-Net (recipe v2, Option A) on the TransMorph Test split.

Run after training. Inputs: two E1 runs (training/train_unet3d.py --eval-only) with per_subject.csv.
Reports (pre-specified):
  * paired per-subject Wilcoxon signed-rank test on mean foreground Dice, and per class
    (Holm correction across the 29 classes); median paired difference with bootstrap 95% CI;
  * Left-Accumbens (26) Dice for both models, explicitly;
  * flip probe for each model: predict on the L-R-mirrored image, mirror back; Dice against the
    truth without and with L/R label swapping. A correctly trained Option-A model should match the
    unmirrored result only WITH swapping (labels follow anatomy position); v1 matched without swap;
  * centroid side check on unmirrored inputs (every predicted L/R structure on the correct side).
GPU: inference, one image at a time. Output: results/e1c_compare_checkpoints/<v1>_vs_<v2>/.

Usage: python experiments/e1c_compare_checkpoints.py --e1-old results/e1_unet_in_domain/unet3d_best_test \
         --e1-new results/e1_unet_in_domain/<run> --ckpt-old unet3d_best.pth --ckpt-new unet3d_v2_best.pth [--probe-limit N]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosphere.metrics import bootstrap_ci, dice  # noqa: E402
from neurosphere.provenance import provenance, sha256_file  # noqa: E402
from training.train_unet3d import ALL_LABELS, LABEL_TO_IDX, LR_PAIRS, SWAP_IDX, UNet3D, pkload  # noqa: E402

FG = [l for l in ALL_LABELS if l]


def load_e1(d):
    rows = list(csv.DictReader(open(Path(d) / "per_subject.csv")))
    out = {}
    for r in rows:
        v = r["dice"]
        out.setdefault(r["file"], {})[int(r["label"])] = float(v) if v not in ("", "nan") else np.nan
    return out


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def probe(ckpt_name, files):
    ck = torch.load(ROOT / "checkpoints" / ckpt_name, map_location="cuda", weights_only=False)
    m = UNet3D(base=16).cuda().eval(); m.load_state_dict(ck["model"])
    lut = np.asarray(ALL_LABELS)
    res = {"unflipped": [], "mirrored_no_swap": [], "mirrored_swap": [], "side_wrong_pairs": [], "side_checked_pairs": []}
    for f in files:
        img, seg = pkload(f)
        t = np.zeros_like(seg, dtype=np.int64)
        for raw, i in LABEL_TO_IDX.items():
            t[seg == raw] = i
        x = torch.from_numpy(img.astype(np.float32))[None, None].cuda()
        with torch.no_grad():
            p = m(x).argmax(1)[0].cpu().numpy()
            pf = m(torch.flip(x, dims=[2])).argmax(1)[0].cpu().numpy()[::-1]
        md = lambda a: float(np.nanmean([dice(a == i, t == i) for i in range(1, len(ALL_LABELS))]))  # noqa: E731
        res["unflipped"].append(md(p)); res["mirrored_no_swap"].append(md(pf)); res["mirrored_swap"].append(md(SWAP_IDX[pf]))
        wrong = checked = 0
        pl = lut[p]
        for l, r in LR_PAIRS:
            a, b, ta, tb = (np.argwhere(z == y) for z, y in ((pl, l), (pl, r), (seg, l), (seg, r)))
            if min(len(a), len(b), len(ta), len(tb)) == 0:
                continue
            mid = (ta[:, 0].mean() + tb[:, 0].mean()) / 2
            checked += 1
            wrong += int(a[:, 0].mean() < mid)   # LIA: left = higher axis-0 index
        res["side_wrong_pairs"].append(wrong); res["side_checked_pairs"].append(checked)
    return {"checkpoint": {"file": ckpt_name, "sha256": sha256_file(ROOT / "checkpoints" / ckpt_name)},
            "mean_fg_dice": {k: float(np.mean(res[k])) for k in ("unflipped", "mirrored_no_swap", "mirrored_swap")},
            "subjects_with_any_side_error": int(sum(w > 0 for w in res["side_wrong_pairs"])),
            "pairs_checked_total": int(sum(res["side_checked_pairs"])), "pairs_wrong_total": int(sum(res["side_wrong_pairs"])),
            "n": len(files)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-old", required=True); ap.add_argument("--e1-new", required=True)
    ap.add_argument("--ckpt-old", default="unet3d_best.pth"); ap.add_argument("--ckpt-new", default="unet3d_v2_best.pth")
    ap.add_argument("--probe-limit", type=int)
    a = ap.parse_args()
    old, new = load_e1(a.e1_old), load_e1(a.e1_new)
    subs = sorted(set(old) & set(new))
    res = {"n_subjects": len(subs), "e1_runs": {"old": a.e1_old, "new": a.e1_new}}
    mo = np.array([np.nanmean([old[s][l] for l in FG]) for s in subs])
    mn = np.array([np.nanmean([new[s][l] for l in FG]) for s in subs])
    d = mn - mo
    res["mean_fg_dice"] = {"old_mean": float(mo.mean()), "new_mean": float(mn.mean()), "median_paired_diff": float(np.median(d)),
                           "paired_diff_ci95": list(bootstrap_ci(d)), "wilcoxon_p": float(wilcoxon(mn, mo).pvalue),
                           "subjects_improved": int((d > 0).sum())}
    per = {}
    ps = []
    for l in FG:
        o = np.array([old[s][l] for s in subs]); n = np.array([new[s][l] for s in subs])
        ok = np.isfinite(o) & np.isfinite(n)
        diff = n[ok] - o[ok]
        p = float(wilcoxon(n[ok], o[ok]).pvalue) if ok.sum() > 5 and np.any(diff != 0) else 1.0
        per[str(l)] = {"old_mean": float(np.nanmean(o)), "new_mean": float(np.nanmean(n)), "median_paired_diff": float(np.median(diff)),
                       "wilcoxon_p": p}
        ps.append(p)
    for (l, v), padj in zip(per.items(), holm(np.array(ps))):
        v["wilcoxon_p_holm"] = float(padj)
    res["per_class"] = per
    res["left_accumbens_26"] = per["26"]
    files = sorted((ROOT / "IXI_data" / "Test").glob("*.pkl"))[: a.probe_limit]
    res["flip_probe_and_side_check"] = {"old": probe(a.ckpt_old, files), "new": probe(a.ckpt_new, files)}
    res["provenance"] = provenance(data_subset="IXI_data/Test" + (f" first {a.probe_limit} for the probe" if a.probe_limit else ""))
    out = ROOT / "results" / "e1c_compare_checkpoints" / f"{Path(a.ckpt_old).stem}_vs_{Path(a.ckpt_new).stem}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_class", "provenance")}, indent=2))


if __name__ == "__main__":
    main()
