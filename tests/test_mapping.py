"""TransMorph subject_k <-> IXI mapping invariants (Phase 2)."""
import csv

import pytest

from tests.conftest import ROOT

MAP = ROOT / "data" / "transmorph_ixi_mapping.csv"


@pytest.mark.skipif(not MAP.exists(), reason="mapping not generated")
def test_mapping_is_a_bijection_in_id_order():
    rows = list(csv.DictReader(open(MAP)))
    assert len(rows) == 576
    ks = [int(r["k"]) for r in rows]
    ids = [r["ixi_id"] for r in rows]
    assert len(set(ks)) == 576 and len(set(ids)) == 576
    order = sorted(rows, key=lambda r: int(r["k"]))
    nums = [int(r["ixi_id"][3:]) for r in order]
    assert all(a < b for a, b in zip(nums, nums[1:]))          # subject_k follows sorted IXI ID order
    raw = {d.name[:6] for d in (ROOT / "ixi-t1").iterdir()} if (ROOT / "ixi-t1").exists() else set(ids)
    assert set(ids) <= raw
    assert not {"IXI423", "IXI426", "IXI427", "IXI547", "IXI553"} & set(ids)
    assert {r["split"] for r in rows} == {"Train", "Val", "Test"}
    assert sum(r["split"] == "Test" for r in rows) == 115
    assert not any(r["confidence"].startswith("CONTRADICTED") for r in rows)
