r"""Apply the pre-registered fine-tuning rule (written 2026-10-09 before any v2 E2 result).

Rule (verbatim in substance):
  raw-scan Val Dice   = E2-FS on the 58 TransMorph Val subjects (results/e2_fs/<run>/metrics.json)
  in-domain Val Dice  = E1 --split Val with the same checkpoint (results/e1_unet_in_domain/<run>/metrics.json)
  volume bias         = per structure, mean over Val subjects of 100 x (ours_jacobian - FS) / FS (from E2-FS)
  SKIP fine-tuning iff raw-scan Val Dice >= in-domain Val Dice - 0.02 AND no structure has |volume bias| > 5 %.
Diagnostics (reported, NOT part of the rule): the in-domain Val volume bias (E1 voxel counts, prediction vs FS) and
the raw-scan bias with our native-grid volume, so a raw-scan-specific bias can be told apart from one the model
already has in-domain.

Writes results/phase3/finetune_decision.json.
Usage: python experiments/finetune_decision.py --e1 results/e1_unet_in_domain/<run> --e2fs results/e2_fs/<run>
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosphere.dictionary import STRUCTURES  # noqa: E402
from neurosphere.provenance import provenance  # noqa: E402

DICE_MARGIN, BIAS_LIMIT = 0.02, 5.0


def in_domain_bias(e1_dir):
    """Per structure, mean over subjects of 100 x (predicted - reference voxels) / reference (E1 per_subject.csv)."""
    rows = list(csv.DictReader(open(Path(e1_dir) / "per_subject.csv")))
    out = {}
    for l, (k, _) in STRUCTURES.items():
        b = [100.0 * (int(r["n_pred_vox"]) - int(r["n_ref_vox"])) / int(r["n_ref_vox"]) for r in rows if int(r["label"]) == l and int(r["n_ref_vox"]) > 0]
        out[k] = round(float(np.mean(b)), 2)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--e1", required=True); ap.add_argument("--e2fs", required=True)
    ap.add_argument("--e1-compare", help="optional E1 Val run of another checkpoint (diagnostic only)")
    a = ap.parse_args()
    e1d, e2d = ROOT / a.e1, ROOT / a.e2fs
    e1, e2 = json.loads((e1d / "metrics.json").read_text()), json.loads((e2d / "metrics.json").read_text())
    assert e1["split"] == "Val" and e2["split"] == "Val", "the rule is evaluated on Val only"
    assert e1["checkpoint"]["sha256"] == e2["checkpoint"]["sha256"], "E1 and E2-FS must use the same checkpoint"
    raw, ind = e2["mean_fg_dice_over_subjects"], e1["mean_fg_dice_over_subjects"]
    bias = {k: v["mean_bias_pct"] for k, v in e2["per_structure"].items()}
    over = {k: round(b, 2) for k, b in sorted(bias.items(), key=lambda x: -abs(x[1])) if abs(b) > BIAS_LIMIT}
    dice_ok, bias_ok = raw >= ind - DICE_MARGIN, not over

    # diagnostics
    ind_bias = in_domain_bias(e1d)
    nat_bias = {}
    ps = list(csv.DictReader(open(e2d / "per_subject_structure.csv")))
    for l, (k, _) in STRUCTURES.items():
        b = [100.0 * (float(r["ours_native_ml"]) - float(r["fs_transferred_ml"])) / float(r["fs_transferred_ml"])
             for r in ps if int(r["label"]) == l and float(r["fs_transferred_ml"]) > 0]
        nat_bias[k] = round(float(np.mean(b)), 2)

    out = {"rule": __doc__.split("Diagnostics")[0].strip(),
           "inputs": {"e1": a.e1, "e2fs": a.e2fs, "checkpoint": e1["checkpoint"]},
           "raw_scan_val_dice": raw, "in_domain_val_dice": ind, "dice_gap": raw - ind, "dice_condition_met": dice_ok,
           "structures_over_bias_limit": over, "bias_condition_met": bias_ok,
           "decision": "skip_finetuning" if (dice_ok and bias_ok) else "finetune (ask the user first, with a time estimate)",
           "raw_scan_bias_pct_jacobian": {k: round(v, 2) for k, v in bias.items()},
           "diagnostic_in_domain_val_bias_pct": ind_bias,
           "diagnostic_raw_scan_bias_pct_native_grid": nat_bias,
           "provenance": provenance()}
    if a.e1_compare:
        cmp_dir = ROOT / a.e1_compare
        cm = json.loads((cmp_dir / "metrics.json").read_text())
        cb = in_domain_bias(cmp_dir)
        out["diagnostic_compare_checkpoint"] = {
            "e1": a.e1_compare, "checkpoint": cm["checkpoint"], "in_domain_val_dice": cm["mean_fg_dice_over_subjects"],
            "in_domain_val_bias_pct": cb, "n_structures_over_limit": sum(abs(v) > BIAS_LIMIT for v in cb.values()),
            "n_structures_over_limit_this_checkpoint": sum(abs(v) > BIAS_LIMIT for v in ind_bias.values()),
            "mean_abs_bias_pct": float(np.mean([abs(v) for v in cb.values()])),
            "mean_abs_bias_pct_this_checkpoint": float(np.mean([abs(v) for v in ind_bias.values()]))}
    d = ROOT / "results" / "phase3"; d.mkdir(parents=True, exist_ok=True)
    (d / "finetune_decision.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("raw_scan_val_dice", "in_domain_val_dice", "dice_gap", "dice_condition_met",
                                          "structures_over_bias_limit", "bias_condition_met", "decision")}, indent=1))


if __name__ == "__main__":
    main()
