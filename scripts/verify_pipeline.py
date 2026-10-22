r"""Check the 581-scan v2 pipeline outputs before anything uses them (Phase 3). CPU only, ~1 min.

Checks per subject in data/splits/ixi_final.csv: result.json present, v2 checkpoint sha, all 29 volumes finite and
> 0, jacobian vs native-grid volume agreement, det(template -> native) range, side check, IQMs present, expected
derived files present. Summarises _run_log.csv (last status per subject), _error_*.txt files and disk use.

Writes work/checks/pipeline_v2.json. Usage: python scripts/verify_pipeline.py
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosphere.dictionary import STRUCTURES  # noqa: E402
from neurosphere.provenance import provenance  # noqa: E402

PIPE = ROOT / "work" / "pipeline_v2"
V2_SHA = "486e06e2f605f33514670d5264f29d5fae1dce565ad0b54c4ba82a7b4eb06ea6"   # unet3d_v2_best.pth
FILES = ("result.json", "labels_native.nii.gz", "brain_mask_native.nii.gz", "bias_field_ds4.nii.gz", "template_t1.nii.gz",
         "native2atlas_0GenericAffine.mat")


def main():
    subs = sorted(r["subject_id"] for r in csv.DictReader(open(ROOT / "data" / "splits" / "ixi_final.csv")))
    last = {}
    for r in csv.DictReader(open(PIPE / "_run_log.csv")):
        last[r["subject"]] = r
    missing, problems, per = [], {}, {}
    for s in subs:
        f = PIPE / s / "result.json"
        if not f.exists():
            missing.append(s); continue
        r = json.loads(f.read_text())
        iss = []
        if r.get("checkpoint", {}).get("sha256") != V2_SHA:
            iss.append("not the v2 checkpoint")
        absent = [x for x in FILES if not (PIPE / s / x).exists()]
        if absent:
            iss.append(f"missing files {absent}")
        v = {int(k): x for k, x in r["volumes_ml"].items()}
        bad = [l for l in STRUCTURES if l not in v or not (math.isfinite(v[l]["jacobian"]) and v[l]["jacobian"] > 0)]
        if bad:
            iss.append(f"labels with zero/non-finite volume {bad}")
        rel = [abs(v[l]["jacobian"] - v[l]["native"]) / v[l]["jacobian"] for l in STRUCTURES if l in v and v[l]["jacobian"] > 0]
        det = float(r["jacobian_det_atlas_to_native"])
        if not 0.4 < det < 1.6:
            iss.append(f"det {det:.3f} outside 0.4-1.6")
        sc = r["side_check_native"]
        if not sc.get("all_correct"):
            iss.append(f"side check: {sc.get('n_swapped')} swapped")
        iq_missing = [q for q in ("cjv", "cnr", "snr_total", "efc", "wm2max", "inu_range") if r["iqms"].get(q) is None]
        if iq_missing:
            iss.append(f"IQMs null {iq_missing}")
        total = sum(v[l]["jacobian"] for l in STRUCTURES if l in v)
        per[s] = {"det": det, "total_29_ml": total, "max_rel_jac_native": max(rel) if rel else None,
                  "median_rel_jac_native": float(np.median(rel)) if rel else None, "seconds": r["timings_s"].get("total"),
                  "bytes": sum(p.stat().st_size for p in (PIPE / s).rglob("*") if p.is_file())}
        if iss:
            problems[s] = iss
    arr = lambda k: np.array([x[k] for x in per.values() if x[k] is not None], float)  # noqa: E731
    stats = lambda a: {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}  # noqa: E731
    out = {"n_subjects": len(subs), "n_results": len(per), "missing": missing, "problems": problems,
           "run_log_last_status": {k: sum(1 for s in subs if last.get(s, {}).get("status") == k) for k in ("ok", "failed")},
           "run_log_failed_last": sorted(s for s in subs if last.get(s, {}).get("status") == "failed"),
           "error_files": sorted(p.name for p in PIPE.glob("_error_*.txt")),
           "fatal_file": (PIPE / "_fatal.txt").exists(),
           "det": stats(arr("det")), "total_29_ml": stats(arr("total_29_ml")),
           "median_rel_jac_native": stats(arr("median_rel_jac_native")), "seconds_total": stats(arr("seconds")),
           "disk_gb": round(sum(x["bytes"] for x in per.values()) / 1e9, 2), "subjects": per, "provenance": provenance()}
    d = ROOT / "work" / "checks"; d.mkdir(parents=True, exist_ok=True)
    (d / "pipeline_v2.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("subjects", "provenance")}, indent=1))


if __name__ == "__main__":
    main()
