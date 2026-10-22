r"""Write the PROVISIONAL subject-level split of the 581 raw IXI T1 scans (Phase 1, P10).

PROVISIONAL: the final downstream splits are made after the Phase 2 TransMorph<->IXI mapping,
so that E4 can also report results on subjects the U-Net never trained on.
Do not report paper numbers on this split.

Output: data/splits/ixi_raw_provisional.csv and data/splits/ixi_raw_provisional.json (provenance).
Usage: .\.venv\Scripts\python scripts\make_provisional_split.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosphere.provenance import provenance  # noqa: E402
from neurosphere.splits import discover_raw_ixi, make_subject_split, write_split_csv  # noqa: E402

SEED, FRACTIONS, FOLDS = 42, (0.7, 0.1, 0.2), 5

rows = make_subject_split(discover_raw_ixi(ROOT / "ixi-t1"), seed=SEED, fractions=FRACTIONS, n_folds=FOLDS)
out = ROOT / "data" / "splits" / "ixi_raw_provisional.csv"
write_split_csv(rows, out)
counts = {s: dict(Counter(r["site"] for r in rows if r["split"] == s)) for s in ("train", "val", "test")}
meta = {
    "status": "PROVISIONAL",
    "status_reason": "final splits follow the Phase 2 TransMorph<->IXI mapping",
    "n_subjects": len(rows), "seed": SEED, "fractions_train_val_test": FRACTIONS, "cv_folds_over_train_val": FOLDS,
    "stratified_by": "site", "counts_by_split_and_site": counts,
    "provenance": provenance(seed=SEED, data_subset="ixi-t1/ raw scans (derived files excluded)"),
}
out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
print(json.dumps(counts, indent=1))
