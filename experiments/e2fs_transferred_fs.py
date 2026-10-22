r"""E2-FS: raw-scan segmentation agreement with each subject's own TransMorph FreeSurfer labels (Phase 3).

Used first on the 58 TransMorph Val subjects for the pre-registered fine-tuning rule,
later once on Test with the checkpoint that rule selects.

Label transfer (per subject, mapping data/transmorph_ixi_mapping.csv):
  1. affine ANTs registration of the TransMorph image (FreeSurfer brainmask, talairach space) to our template-space
     U-Net input of the SAME subject (work/pipeline_v2/<IXI>/template_t1.nii.gz; same grid);
  2. the TransMorph FreeSurfer labels follow that affine and then our template -> native affine
     (native2atlas_0GenericAffine.mat, inverted), nearest-label interpolation, onto the native grid of labels_native.
Metrics (pre-specified in the decision rule): per subject, mean over the 29 foreground classes of the native-space
Dice (NaN when a class is absent from both); per structure, volume bias 100 x (ours - FS) / FS with our primary
('jacobian') volume against the transferred FS volume (native voxel count x voxel volume). The post-registration
NCC of step 1 is reported per subject as a transfer-quality check.

Output: results/e2_fs/<run>/{per_subject_structure.csv, per_subject.csv, metrics.json}; --limit runs go to work/smoke/.
Usage: python experiments/e2fs_transferred_fs.py --split Val [--limit 3] [--out-name NAME] [--labels-dir work/pipeline_ft]
"""
import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosphere.dictionary import STRUCTURES  # noqa: E402
from neurosphere.metrics import bootstrap_ci, dice  # noqa: E402
from neurosphere.provenance import provenance  # noqa: E402
from neurosphere.transfer import labels_on_native, register_tm_to_ours  # noqa: E402

PIPE = ROOT / "work" / "pipeline_v2"
FG = list(STRUCTURES)


def transfer(sid: str, k: int, split: str, labels_dir: Path):
    """TransMorph FS labels of subject_k on the native grid of our labels, plus the registration NCC."""
    r = register_tm_to_ours(sid, k, split, PIPE)
    ours, fs, aff = labels_on_native(r, sid, PIPE, labels_dir)
    return ours, fs, aff, r.ncc


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["Val", "Test"])
    ap.add_argument("--limit", type=int); ap.add_argument("--out-name")
    ap.add_argument("--labels-dir", default=str(PIPE), help="where labels_native.nii.gz + result.json come from (default: v2 pipeline)")
    a = ap.parse_args()
    rows_map = [r for r in csv.DictReader(open(ROOT / "data" / "transmorph_ixi_mapping.csv")) if r["split"] == a.split]
    ref_f = ROOT / "work" / "mapping" / "verification_v2_refined.csv"
    unconfirmed = {r["ixi_id"] for r in csv.DictReader(open(ref_f)) if r["confirmed"] != "True"} if ref_f.exists() else set()
    rows_map = sorted(rows_map, key=lambda r: r["ixi_id"])[: a.limit]
    labels_dir = Path(a.labels_dir)
    per_struct, per_subj = [], []
    for i, r in enumerate(rows_map):
        t0 = time.perf_counter()
        sid, k = r["ixi_id"], int(r["k"])
        res = json.loads((labels_dir / sid / "result.json").read_text())
        ours, fs, aff, ncc = transfer(sid, k, a.split, labels_dir)
        vv = float(abs(np.linalg.det(aff[:3, :3]))) / 1000.0
        ids, cnt = np.unique(fs, return_counts=True); fsc = dict(zip(ids.tolist(), cnt.tolist()))
        dices = []
        for l in FG:
            d = dice(ours == l, fs == l)
            dices.append(d)
            fs_ml = fsc.get(l, 0) * vv
            ours_ml = res["volumes_ml"][str(l)]["jacobian"]
            per_struct.append({"subject": sid, "k": k, "label": l, "structure": STRUCTURES[l][0], "dice_native": d,
                               "fs_transferred_ml": fs_ml, "ours_jacobian_ml": ours_ml, "ours_native_ml": res["volumes_ml"][str(l)]["native"],
                               "bias_pct": 100.0 * (ours_ml - fs_ml) / fs_ml if fs_ml > 0 else float("nan")})
        row = {"subject": sid, "k": k, "mean_fg_dice": float(np.nanmean(dices)), "transfer_ncc": round(ncc, 4),
               "mapping_unconfirmed": sid in unconfirmed, "seconds": round(time.perf_counter() - t0, 1)}
        per_subj.append(row)
        print(f"[{i + 1}/{len(rows_map)}] {sid} k={k} Dice {row['mean_fg_dice']:.4f} NCC {ncc:.3f} {row['seconds']}s", flush=True)

    run = a.out_name or (time.strftime("%Y%m%d-%H%M%S") + f"_{a.split.lower()}" + (f"_limit{a.limit}" if a.limit else ""))
    out = ROOT / ("work/smoke" if a.limit else "results") / "e2_fs" / run
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("per_subject_structure.csv", per_struct), ("per_subject.csv", per_subj)):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    md = np.array([r["mean_fg_dice"] for r in per_subj])
    struct = {}
    for l in FG:
        rr = [x for x in per_struct if x["label"] == l]
        b = np.array([x["bias_pct"] for x in rr], float); dd = np.array([x["dice_native"] for x in rr], float)
        struct[STRUCTURES[l][0]] = {"label": l, "mean_bias_pct": float(np.nanmean(b)), "bias_pct_ci95": [float(x) for x in bootstrap_ci(b[np.isfinite(b)])],
                                    "dice_mean": float(np.nanmean(dd)), "n": int(np.isfinite(b).sum())}
    metrics = {"split": a.split, "n_subjects": len(per_subj), "checkpoint": json.loads((labels_dir / per_subj[0]["subject"] / "result.json").read_text())["checkpoint"],
               "labels_dir": str(labels_dir.relative_to(ROOT)) if labels_dir.is_relative_to(ROOT) else str(labels_dir),
               "mean_fg_dice_over_subjects": float(md.mean()), "mean_fg_dice_ci95": [float(x) for x in bootstrap_ci(md)],
               "per_subject_mean_fg_dice_sd": float(md.std(ddof=1)) if len(md) > 1 else None,
               "transfer_ncc": {"median": float(np.median([r["transfer_ncc"] for r in per_subj])), "min": float(min(r["transfer_ncc"] for r in per_subj))},
               "mapping_unconfirmed_subjects": [r["subject"] for r in per_subj if r["mapping_unconfirmed"]],
               "max_abs_structure_bias_pct": max(abs(v["mean_bias_pct"]) for v in struct.values()),
               "per_structure": struct, "definitions": __doc__, "provenance": provenance(data_subset=f"TransMorph {a.split}, n={len(per_subj)}")}
    if metrics["per_subject_mean_fg_dice_sd"] is None:
        metrics["per_subject_mean_fg_dice_sd_reason"] = "one subject"
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"mean fg Dice {md.mean():.4f}, max |structure bias| {metrics['max_abs_structure_bias_pct']:.2f} % -> {out}")


if __name__ == "__main__":
    main()
