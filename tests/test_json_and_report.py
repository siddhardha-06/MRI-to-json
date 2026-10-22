"""Interim JSON: measured values or null + reason only; report: no invented findings (AUDIT K3-K5)."""
import json
import re

import pytest

from neurosphere.jsonio import build_interim_json
from neurosphere.metrics import dice, surface_distances
from neurosphere.nulls import set_null, unexplained_nulls
from neurosphere.qc import parse_mriqc_official, quality_block
from neurosphere.report.template import render_html, render_text
from tests.conftest import IXI002, needs_ixi, save_nifti

FORBIDDEN_KEYS = {"pathology_indicators", "risk_level", "interpretation", "simulated", "score", "grade", "passed",
                  "left_hemisphere_mean", "right_hemisphere_mean", "total_brain_volume_ml"}
FINDING_WORDS = re.compile(r"(?i)\b(normal|abnormal|lesion|mass|tumou?r|infarct|haemorrhage|hemorrhage|sclerosis|"
                           r"dysplasia|atrophy|diagnos\w*|risk|recommend\w*|unremarkable)\b")


def _all_keys(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from _all_keys(v)
    elif isinstance(o, list):
        for v in o:
            yield from _all_keys(v)


@pytest.fixture
def doc(tmp_path, ras_volume):
    data, aff = ras_volume
    return build_interim_json(save_nifti(tmp_path, data, aff, "upload_scan.nii.gz"))


def test_every_null_has_a_reason(doc):
    assert unexplained_nulls(doc) == []
    assert doc["schema_version"] == "2.0.0-interim"
    assert doc["subject"]["id"] is None and doc["subject"]["id_reason"]
    assert doc["volumes_ml"] is None and "K6" in doc["volumes_ml_reason"]


def test_no_v1_invented_fields(doc):
    assert not FORBIDDEN_KEYS & set(_all_keys(doc))
    json.dumps(doc, allow_nan=False)


def test_quality_not_computed_and_never_simulated():
    q = quality_block()
    assert q["gate"]["status"] == "not_computed" and q["iqms"] is None and q["mriqc_official"] is None
    parsed = parse_mriqc_official({"cjv": 0.41, "cnr": 3.2, "efc": float("nan"), "bids_name": "x", "flag": True})
    assert parsed["iqms"]["cjv"] == 0.41
    assert parsed["iqms"]["efc"] is None and parsed["iqms"]["efc_reason"]  # missing stays null, never 0.0
    assert parsed["iqms"]["qi_1"] is None and "flag" not in parsed["iqms"]


def test_report_states_only_measurements(doc):
    text, html = render_text(doc), render_html(doc)
    # the only allowed occurrences of finding-like words are the fixed research-use disclaimers
    stripped = text.replace("not for diagnosis", "").replace("does not represent focal lesions", "")
    assert not FINDING_WORDS.search(stripped), FINDING_WORDS.findall(stripped)
    assert "not computed" in text and "<script" not in html


def test_report_refuses_unexplained_nulls(doc):
    doc["acquisition"]["shape"] = None
    with pytest.raises(ValueError):
        render_text(doc)
    set_null(doc["acquisition"], "shape", "test")
    render_text(doc)


@needs_ixi
def test_ixi_json_values_match_header():
    d = build_interim_json(IXI002)
    assert d["subject"]["id"] == "IXI002" and d["acquisition"]["site"] == "Guys"
    assert d["acquisition"]["orientation_original"] == "PSR"
    assert d["acquisition"]["voxel_volume_ml"] == pytest.approx(0.9375 * 0.9375 * 1.2 / 1000, rel=1e-5)
    assert len(d["subject"]["sha256"]) == 64


def test_dice_and_surface_distance():
    import numpy as np
    a = np.zeros((20, 20, 20), bool); a[5:10, 5:10, 5:10] = True
    b = np.zeros_like(a); b[7:12, 5:10, 5:10] = True
    assert dice(a, b) == pytest.approx(2 * 75 / 250)
    assert np.isnan(dice(np.zeros_like(a), np.zeros_like(a)))      # absent in both -> NaN, not 1.0 (K9)
    hd95, assd = surface_distances(a, a)
    assert hd95 == 0 and assd == 0
    hd95, _ = surface_distances(a, b)
    assert hd95 == pytest.approx(2.0)
    assert all(np.isnan(surface_distances(a, np.zeros_like(a))))
