"""Splits are at subject level; derived copies belong to their subject."""
import json
import random
from collections import defaultdict
from pathlib import Path

import pytest

from neurosphere.splits import check_no_leakage, make_subject_split, read_split_csv, subject_of
from tests.conftest import ROOT

SPLIT_CSV = ROOT / "data" / "splits" / "ixi_raw_provisional.csv"


def _rows(n_per_site=(30, 20, 10)):
    rows = []
    k = 2
    for site, n in zip(("Guys", "HH", "IOP"), n_per_site):
        for _ in range(n):
            rows.append({"subject_id": f"IXI{k:03d}", "site": site, "raw_file": f"IXI{k:03d}-{site}-0000-T1.nii/x.nii"})
            k += 1
    return rows


def test_derived_files_map_to_their_subject():
    for name in ("IXI024-Guys-0705-IXI3DMPRAG_-s231_-0301-00003-000001-01.nii",
                 "IXI024-Guys-0705-IXI3DMPRAG_-s231_-0301-00003-000001-01_n4",
                 "IXI024-Guys-0705-IXI3DMPRAG_-s231_-0301-00003-000001-01_n4_n4.nii",
                 "IXI024-Guys-0705-T1.nii/IXI024-Guys-0705-x_seg.nii"):
        assert subject_of(name) == "IXI024"


def test_check_no_leakage_catches_the_k7_pattern():
    with pytest.raises(ValueError, match="1 subjects"):
        check_no_leakage({"IXI024-Guys-0705-a.nii": "train", "IXI024-Guys-0705-a_n4.nii": "val", "IXI002-Guys-0828-b.nii": "val"})


def test_subject_split_is_deterministic_stratified_and_leak_free():
    a = make_subject_split(_rows(), seed=7)
    b = make_subject_split(_rows(), seed=7)
    assert a == b
    assert a != make_subject_split(_rows(), seed=8)
    per = defaultdict(lambda: defaultdict(int))
    for r in a:
        per[r["site"]][r["split"]] += 1
    assert per["Guys"]["test"] == 6 and per["HH"]["test"] == 4 and per["IOP"]["test"] == 2
    assert all(r["cv_fold"] == -1 for r in a if r["split"] == "test")
    assert {r["cv_fold"] for r in a if r["split"] != "test"} == set(range(5))
    check_no_leakage({r["raw_file"]: r["split"] for r in a})


@pytest.mark.skipif(not SPLIT_CSV.exists(), reason="provisional split not generated")
def test_provisional_split_file():
    rows = read_split_csv(SPLIT_CSV)
    assert len(rows) == len({r["subject_id"] for r in rows}) == 581
    check_no_leakage({r["raw_file"]: r["split"] for r in rows})
    meta = json.loads(SPLIT_CSV.with_suffix(".json").read_text())
    assert meta["status"] == "PROVISIONAL"


@pytest.mark.skipif(not (ROOT / "rule_based_features.json").exists(), reason="v1 feature file not present")
def test_shipped_regressor_split_leaked():
    """Regression test for AUDIT K7: the file-level split of train_regressor.py leaks subjects."""
    feats = json.loads((ROOT / "rule_based_features.json").read_text())
    stems = [k for k in feats if not (k.endswith("_seg") or k.endswith("_seg_seg"))]
    folder = {}
    for d in (ROOT / "ixi-t1").iterdir():
        for f in d.glob("*.nii"):
            folder[f.name[:-4]] = d.name
    raw = lambda s: s[:-6] if s.endswith("_n4_n4") else s[:-3] if s.endswith("_n4") else s  # noqa: E731
    paths = sorted(Path("ixi-t1") / folder[raw(s)] / f"{s}.nii" for s in stems)
    random.seed(42); random.shuffle(paths)
    k = int(0.8 * len(paths))
    assign = {p.name: ("train" if i < k else "val") for i, p in enumerate(paths)}
    with pytest.raises(ValueError, match="7 subjects"):
        check_no_leakage(assign)
