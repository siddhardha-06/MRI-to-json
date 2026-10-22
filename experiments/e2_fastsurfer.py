r"""E2: end-to-end agreement of NeuroSphere volumes with FastSurfer on raw IXI scans (+ cost).

Pre-specified (before looking at results):
  * our primary volume = 'jacobian' (atlas-space count x |det A|); 'native' reported as secondary;
  * FastSurfer volume = voxel count in mri/aseg.auto_noCCseg.mgz x voxel volume (same definition as ours;
    the PV-corrected stats-file volumes are reported as a secondary column);
  * agreement metrics per structure: ICC(2,1) absolute agreement with 95% CI (pingouin), Pearson r,
    MAPE, Bland-Altman bias and 95% limits of agreement; native-space Dice;
  * subjects are grouped by U-Net split (data/transmorph_ixi_mapping.csv); the headline is the
    U-Net TEST group (never seen in training). Other groups are reported, labelled, never pooled silently.
  * our WM excludes the corpus callosum (CC labels 251-255 were background in training); FastSurfer's
    aseg.auto_noCCseg puts the CC in WM. WM agreement is therefore biased by design and labelled so.
Cost: our per-stage runtimes vs FastSurfer wall time (device stated); bytes per subject.

Inputs: work/pipeline/<IXI>/ (ours), work/fastsurfer/<IXI>/ and results/legacy/validation_results/
fastsurfer_output/<IXI...>/ (FastSurfer 2.5.4). Output: results/e2_fastsurfer/<run>/.
Usage: python experiments/e2_fastsurfer.py [--out-name NAME]
"""
import argparse
import warnings
import csv
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosphere.metrics import bootstrap_ci, dice  # noqa: E402
from neurosphere.preprocess import nib_to_ants  # noqa: E402
from neurosphere.provenance import provenance  # noqa: E402

LABELS = {2: "L-Cerebral-WM", 3: "L-Cerebral-Cortex", 4: "L-Lateral-Ventricle", 7: "L-Cerebellum-WM", 8: "L-Cerebellum-Cortex",
          10: "L-Thalamus", 11: "L-Caudate", 12: "L-Putamen", 13: "L-Pallidum", 14: "3rd-Ventricle", 15: "4th-Ventricle",
          16: "Brain-Stem", 17: "L-Hippocampus", 18: "L-Amygdala", 26: "L-Accumbens", 28: "L-VentralDC",
          41: "R-Cerebral-WM", 42: "R-Cerebral-Cortex", 43: "R-Lateral-Ventricle", 46: "R-Cerebellum-WM", 47: "R-Cerebellum-Cortex",
          49: "R-Thalamus", 50: "R-Caudate", 51: "R-Putamen", 52: "R-Pallidum", 53: "R-Hippocampus", 54: "R-Amygdala",
          58: "R-Accumbens", 60: "R-VentralDC"}
BILATERAL = {"Hippocampus": (17, 53), "Amygdala": (18, 54), "Thalamus": (10, 49), "Caudate": (11, 50), "Putamen": (12, 51),
             "Pallidum": (13, 52), "Lateral-Ventricles": (4, 43), "Accumbens": (26, 58), "Cerebral-Cortex": (3, 42), "Cerebral-WM": (2, 41)}


def fastsurfer_dir(sid):
    for d in [ROOT / "work" / "fastsurfer" / sid] + sorted((ROOT / "results/legacy/validation_results/fastsurfer_output").glob(f"{sid}-*")):
        if (d / "mri" / "aseg.auto_noCCseg.mgz").exists():
            return d
    return None


def fs_stats_volumes(d):
    f = d / "stats" / "aseg.VINN.stats"
    out = {}
    if f.exists():
        for line in f.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            out[int(p[1])] = float(p[3]) / 1000.0
    return out


def fs_runtime(d):
    """FastSurfer time to produce the aseg, from scripts/exectime.log of the LAST run in that folder:
    sum of module elapsed times (aseg+DKT network, N4, segstats, corpus callosum) excluding HypVINN
    (hypothalamus; not needed here and disabled in our new runs). Returns (aseg_s, hypvinn_s)."""
    import re
    f = d / "scripts" / "exectime.log"
    if not f.exists():
        return None, None
    mods = re.findall(r"^->\s+\S+\s+(\S+)\s+N \d+ e ([\d.]+)", f.read_text(errors="ignore"), flags=re.M)
    starts = [i for i, (m, _) in enumerate(mods) if m == "FastSurferCNN/run_prediction.py"]
    if not starts:
        return None, None
    last = mods[starts[-1]:]
    aseg = sum(float(e) for m, e in last if not m.startswith("HypVINN"))
    hyp = sum(float(e) for m, e in last if m.startswith("HypVINN"))
    return aseg, hyp


def dir_bytes(d):
    """Bytes of regular files; FastSurfer's Linux symlinks (unreadable on Windows) take no space and are skipped."""
    tot = 0
    for p in Path(d).rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                tot += p.stat().st_size
        except OSError:
            continue
    return tot


def icc21(a, b):
    import pandas as pd
    import pingouin as pg
    n = len(a)
    df = pd.DataFrame({"s": list(range(n)) * 2, "r": ["fs"] * n + ["ours"] * n, "v": list(a) + list(b)})
    # ICC(2,1) = two-way random effects, absolute agreement, single measurement; pingouin >= 0.7 labels it "ICC(A,1)"
    t = pg.intraclass_corr(data=df, targets="s", raters="r", ratings="v").set_index("Type")
    row = t.loc["ICC(A,1)"] if "ICC(A,1)" in t.index else t.loc["ICC2"]
    ci = row["CI95"] if "CI95" in row.index else row["CI95%"]
    return float(row["ICC"]), [float(x) for x in ci]


def agreement(fs, ours):
    fs, ours = np.asarray(fs, float), np.asarray(ours, float)
    d = ours - fs
    res = {"n": int(len(fs)), "fs_mean_ml": float(fs.mean()), "ours_mean_ml": float(ours.mean()),
           "pearson_r": float(np.corrcoef(fs, ours)[0, 1]) if len(fs) > 2 else None,
           "mape_pct": float(np.mean(np.abs(d) / fs) * 100), "bias_ml": float(d.mean()),
           "bias_pct": float(np.mean(d / fs) * 100), "loa_ml": [float(d.mean() - 1.96 * d.std(ddof=1)), float(d.mean() + 1.96 * d.std(ddof=1))]}
    if len(fs) >= 5:
        res["icc21"], res["icc21_ci95"] = icc21(fs, ours)
    else:
        res["icc21"] = None; res["icc21_reason"] = "fewer than 5 subjects"
    if res["pearson_r"] is None:
        res["pearson_r_reason"] = "fewer than 3 subjects"
    return res


def null_nonfinite(o):
    """Replace NaN/inf floats by None plus a sibling '<key>_reason' (null + reason rule)."""
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if isinstance(v, float) and not np.isfinite(v):
                out[k] = None
                out[f"{k}_reason"] = "not finite (zero variance or zero denominator, e.g. a structure never predicted)"
            elif isinstance(v, list) and any(isinstance(x, float) and not np.isfinite(x) for x in v):
                out[k] = None
                out[f"{k}_reason"] = "not finite (zero variance or zero denominator, e.g. a structure never predicted)"
            else:
                out[k] = null_nonfinite(v)
        return out
    if isinstance(o, list):
        return [null_nonfinite(x) for x in o]
    return o


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # zero-variance structures are reported as null + reason
    ap = argparse.ArgumentParser(); ap.add_argument("--out-name", default=None); a = ap.parse_args()
    split = {r["ixi_id"]: r["split"] for r in csv.DictReader(open(ROOT / "data" / "transmorph_ixi_mapping.csv"))}
    from neurosphere.splits import discover_raw_ixi
    raw_paths = {r["subject_id"]: ROOT / "ixi-t1" / r["raw_file"] for r in discover_raw_ixi(ROOT / "ixi-t1")}
    rows, costs = [], []
    for res_f in sorted((ROOT / "work" / "pipeline").glob("IXI*/result.json")):
        sid = res_f.parent.name
        fsd = fastsurfer_dir(sid)
        if fsd is None:
            continue
        res = json.loads(res_f.read_text())
        fs_img = nib.load(str(fsd / "mri" / "aseg.auto_noCCseg.mgz"))
        fs_lab = np.asarray(fs_img.dataobj).astype(np.int32)
        fs_vv = abs(np.linalg.det(fs_img.affine[:3, :3])) / 1000.0
        ids, cnt = np.unique(fs_lab, return_counts=True); fsc = dict(zip(ids.tolist(), cnt.tolist()))
        stats = fs_stats_volumes(fsd)
        ours_img = nib.load(str(res_f.parent / "labels_native.nii.gz"))
        ours_lab = np.asarray(ours_img.dataobj).astype(np.int32)
        import ants
        fs_on_native = ants.resample_image_to_target(nib_to_ants(fs_lab.astype(np.float32), fs_img.affine),
                                                     nib_to_ants(ours_lab.astype(np.float32), ours_img.affine), interp_type="genericLabel")
        fs_nat = np.rint(fs_on_native.numpy()).astype(np.int32)
        for l, name in LABELS.items():
            rows.append({"subject": sid, "unet_split": split.get(sid, "not_in_transmorph"), "label": l, "structure": name,
                         "fs_ml": fsc.get(l, 0) * fs_vv, "fs_stats_ml": stats.get(l, ""),
                         "ours_jacobian_ml": res["volumes_ml"][str(l)]["jacobian"], "ours_native_ml": res["volumes_ml"][str(l)]["native"],
                         "dice_native": dice(ours_lab == l, fs_nat == l),
                         "side_check_all_correct": res.get("side_check_native", {}).get("all_correct", "")})
        costs.append({"subject": sid, "ours_total_s": res["timings_s"]["total"], **{f"ours_{k}_s": v for k, v in res["timings_s"].items() if k != "total"},
                      "fs_aseg_s": fs_runtime(fsd)[0], "fs_hypvinn_s_excluded": fs_runtime(fsd)[1], "fs_bytes": dir_bytes(fsd), "ours_cache_bytes": dir_bytes(res_f.parent),
                      "raw_nifti_bytes": raw_paths[sid].stat().st_size, "json_bytes": len(json.dumps(res["volumes_ml"]).encode())})
    if not rows:
        raise SystemExit("no subject has both a NeuroSphere result and a FastSurfer output")
    run = a.out_name or time.strftime("%Y%m%d-%H%M%S")
    out = ROOT / ("work/smoke" if run.startswith("smoke") else "results") / "e2_fastsurfer" / run   # smoke runs are not results
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_subject_structure.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out / "cost_per_subject.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(costs[0])); w.writeheader(); w.writerows(costs)

    groups = {"unet_test": lambda r: r["unet_split"] == "Test", "unet_train_or_val": lambda r: r["unet_split"] in ("Train", "Val"),
              "not_in_transmorph": lambda r: r["unet_split"] == "not_in_transmorph"}
    metrics = {"groups": {}}
    for g, sel in groups.items():
        gr = [r for r in rows if sel(r)]
        subs = sorted({r["subject"] for r in gr})
        if not subs:
            continue
        gm = {"subjects": subs, "per_structure": {}, "bilateral": {}}
        for l, name in LABELS.items():
            rr = [r for r in gr if r["label"] == l]
            ag = agreement([r["fs_ml"] for r in rr], [r["ours_jacobian_ml"] for r in rr])
            ag["ours_native_vs_fs"] = agreement([r["fs_ml"] for r in rr], [r["ours_native_ml"] for r in rr])
            dd = np.array([r["dice_native"] for r in rr], float)
            ag["dice_native_mean"] = float(np.nanmean(dd)); ag["dice_native_ci95"] = list(bootstrap_ci(dd))
            if name.endswith("Cerebral-WM"):
                ag["note"] = "biased by design: our WM excludes the corpus callosum, FastSurfer's includes it"
            gm["per_structure"][name] = ag
        for name, (l, r_) in BILATERAL.items():
            per = {}
            for r in gr:
                if r["label"] in (l, r_):
                    p = per.setdefault(r["subject"], [0.0, 0.0])
                    p[0] += r["fs_ml"]; p[1] += r["ours_jacobian_ml"]
            v = np.array(list(per.values()))
            gm["bilateral"][name] = agreement(v[:, 0], v[:, 1])
        gm["mean_dice_native_over_structures"] = float(np.nanmean([gm["per_structure"][n]["dice_native_mean"] for n in LABELS.values()]))
        gm["side_swaps"] = sum(1 for s in subs if any(r["side_check_all_correct"] is False for r in gr if r["subject"] == s))
        metrics["groups"][g] = gm
    c = {k: np.array([x[k] for x in costs if x[k] is not None], float) for k in ("ours_total_s", "fs_aseg_s", "fs_bytes", "ours_cache_bytes", "raw_nifti_bytes", "json_bytes")}
    metrics["cost"] = {k: {"median": float(np.median(v)), "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))], "n": int(len(v))}
                       for k, v in c.items() if len(v)}
    metrics["cost"]["devices"] = {"ours": "NVIDIA RTX 4060 Laptop GPU (HD-BET, U-Net) + CPU (N4, ANTs)",
                                  "fastsurfer": "CPU, 4 threads (host driver too old for the image's CUDA 12.8)"}
    metrics["definitions"] = __doc__
    metrics["provenance"] = provenance(data_subset=f"{len(costs)} subjects with both outputs")
    metrics = null_nonfinite(metrics)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    for g, gm in metrics["groups"].items():
        print(f"\n== {g}: n={len(gm['subjects'])}, mean native Dice={gm['mean_dice_native_over_structures']:.3f}, side swaps={gm['side_swaps']}")
        for name, ag in gm["bilateral"].items():
            print(f"  {name:20s} FS {ag['fs_mean_ml']:7.2f}  ours {ag['ours_mean_ml']:7.2f}  r={ag['pearson_r'] if ag.get('pearson_r') is None else round(ag['pearson_r'], 3)}  "
                  f"ICC={None if ag.get('icc21') is None else round(ag['icc21'], 3)}  MAPE={ag['mape_pct']:.1f}%  bias={ag['bias_pct']:+.1f}%")
    print("\ncost:", json.dumps(metrics["cost"], indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
