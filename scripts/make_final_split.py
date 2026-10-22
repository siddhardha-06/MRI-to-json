r"""Final subject-level split (Phase 3, decision 5). Replaces data/splits/ixi_raw_provisional.csv.

  * the TransMorph split mapped to IXI IDs (data/transmorph_ixi_mapping.csv): 403 train / 58 val / 115 test;
  * the 5 raw subjects that are not in TransMorph (IXI423, 426, 427, 547, 553) -> test;
  * IXI430 stays in train with the flag `transmorph_image_defective` (TransMorph subject_381: failed
    FreeSurfer skull strip, so its training labels are suspect);
  * cv_fold 0..4 over the train split only (stratified by site, seed 42) for inner cross-validation;
  * age/sex from IXI.xls via neurosphere.demographics (null + reason when missing or inconsistent).
Outputs: data/splits/ixi_final.csv (committed; IDs, site, split, folds, flags - no demographics),
         results/splits/ixi_final_balance.json and .md (age/sex/site per split).
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosphere.demographics import load_demographics  # noqa: E402
from neurosphere.provenance import provenance, sha256_file  # noqa: E402
from neurosphere.splits import check_no_leakage, discover_raw_ixi  # noqa: E402

NOT_IN_TRANSMORPH = {"IXI423", "IXI426", "IXI427", "IXI547", "IXI553"}
FLAGS = {"IXI430": "transmorph_image_defective"}
SPLIT = {"Train": "train", "Val": "val", "Test": "test"}


def main():
    raw = {r["subject_id"]: r for r in discover_raw_ixi(ROOT / "ixi-t1")}
    mapping = {r["ixi_id"]: r for r in csv.DictReader(open(ROOT / "data" / "transmorph_ixi_mapping.csv"))}
    rows = []
    for sid in sorted(raw):
        if sid in mapping:
            split, src = SPLIT[mapping[sid]["split"]], f"transmorph:{mapping[sid]['transmorph_file']}"
        elif sid in NOT_IN_TRANSMORPH:
            split, src = "test", "not_in_transmorph"
        else:
            raise SystemExit(f"{sid} is neither mapped nor a known non-TransMorph subject")
        rows.append({"subject_id": sid, "site": raw[sid]["site"], "split": split, "cv_fold": -1, "source": src,
                     "flags": FLAGS.get(sid, "")})
    rng = np.random.default_rng(42)
    for site in sorted({r["site"] for r in rows}):
        tr = [r for r in rows if r["split"] == "train" and r["site"] == site]
        for k, i in enumerate(rng.permutation(len(tr))):
            tr[i]["cv_fold"] = int(k % 5)
    assert len({r["subject_id"] for r in rows}) == len(rows), "duplicate subject in split table"   # one row per subject => no leakage
    out = ROOT / "data" / "splits" / "ixi_final.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    demo = load_demographics([r["subject_id"] for r in rows])
    bal = {}
    for sp in ("train", "val", "test"):
        rr = [r for r in rows if r["split"] == sp]
        ages = np.array([demo[r["subject_id"]]["age"] for r in rr if demo[r["subject_id"]].get("age") is not None])
        sex = Counter(demo[r["subject_id"]].get("sex") for r in rr)
        bal[sp] = {"n": len(rr), "age_n": int(len(ages)), "age_mean": round(float(ages.mean()), 2), "age_sd": round(float(ages.std(ddof=1)), 2),
                   "age_min": round(float(ages.min()), 1), "age_max": round(float(ages.max()), 1),
                   "sex": {"M": sex.get("M", 0), "F": sex.get("F", 0), "missing": sex.get(None, 0)},
                   "site": dict(sorted(Counter(r["site"] for r in rr).items())),
                   "age_missing_reasons": dict(Counter(demo[r["subject_id"]].get("age_reason") for r in rr if demo[r["subject_id"]].get("age") is None))}
    meta = {"split_counts": {k: v["n"] for k, v in bal.items()}, "balance": bal, "flags": FLAGS,
            "inputs": {"transmorph_ixi_mapping.csv": sha256_file(ROOT / "data" / "transmorph_ixi_mapping.csv"),
                       "IXI.xls": sha256_file(ROOT / "data" / "raw" / "IXI.xls")},
            "provenance": provenance(seed=42)}
    d = ROOT / "results" / "splits"; d.mkdir(parents=True, exist_ok=True)
    (d / "ixi_final_balance.json").write_text(json.dumps(meta, indent=2))
    lines = ["| split | n | age mean ± SD (n with age) | range | M / F / missing | Guys / HH / IOP |", "|---|---|---|---|---|---|"]
    for sp, b in bal.items():
        s = b["site"]
        lines.append(f"| {sp} | {b['n']} | {b['age_mean']} ± {b['age_sd']} ({b['age_n']}) | {b['age_min']}–{b['age_max']} | "
                     f"{b['sex']['M']} / {b['sex']['F']} / {b['sex']['missing']} | {s.get('Guys', 0)} / {s.get('HH', 0)} / {s.get('IOP', 0)} |")
    (d / "ixi_final_balance.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({sp: b["age_missing_reasons"] for sp, b in bal.items()}, indent=1))


if __name__ == "__main__":
    main()
