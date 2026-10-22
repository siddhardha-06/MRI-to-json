"""JSON 2.0.0: eTIV/morphometry, QC gate, cross-fitted normative model, schema, dictionary drift and
serialisation round-trips. Synthetic inputs only; no pipeline outputs needed."""
import json
import math
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from neurosphere.dictionary import ETIV_SCALE_ML, PAIRS, STRUCTURES, feature_entries
from neurosphere.features import morphometry
from neurosphere.jsonio import build_json_v2
from neurosphere.normative import MIN_TRAIN, _design, _fit, fit_predict
from neurosphere.nulls import unexplained_nulls
from neurosphere.qc import GATE_IQMS, apply_gate, fit_gate_reference
from neurosphere.serialize import PROFILES, deserialize_compact, flatten, serialize, to_vector

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "neurosphere" / "schema"
DET = 0.7


def _result(det=DET, seed=0):
    rng = np.random.default_rng(seed)
    vols = {str(l): {"jacobian": float(rng.uniform(1, 300)), "native": float(rng.uniform(1, 300))} for l in STRUCTURES}
    iqms = {q: float(rng.uniform(0.5, 2)) for q in ("cjv", "cnr", "snr_gm", "snr_wm", "snr_csf", "snr_total", "snrd_gm",
                                                     "snrd_wm", "snrd_csf", "snrd_total", "efc", "wm2max", "inu_med", "inu_range")}
    iqms["fber"] = None
    iqms["fber_reason"] = "zero-filled background"
    tissue = {t: {"mean": 80.0, "sd": 5.0, "p05": 70.0, "p50": 80.0, "p95": 90.0, "n_voxels": 1000} for t in ("gm", "wm", "csf_ventricular")}
    return {"subject": "IXI999", "source_file": "IXI999-Guys-0000-T1.nii.gz", "sha256": "0" * 64, "orientation_original": "PIR",
            "volumes_ml": vols, "jacobian_det_atlas_to_native": det, "side_check_native": {"ok": True}, "brain_mask_ml": 1300.0,
            "iqms": iqms, "intensity": {"normalisation": "test", **tissue, "gm_wm_ratio": 0.7, "gm_wm_contrast_pct": 30.0},
            "checkpoint": {"file": "unet3d_v2_best.pth"}, "timings_s": {"total": 1.0}, "provenance": {"git": "test"}}


def _gate_reference(n=40):
    return fit_gate_reference({f"s{i}": {q: 1.0 + 0.01 * (i % 7) for q in GATE_IQMS} for i in range(n)}, [f"s{i}" for i in range(n)])


def _doc(normative=None, reference=None):
    r = _result()
    gate = apply_gate(r["iqms"], _gate_reference())
    acq = {"shape": [256, 256, 150], "voxel_size_mm": [0.94, 0.94, 1.2], "orientation_original": "PIR", "site": "Guys"}
    if normative is None:
        keys = [k for k, _ in STRUCTURES.values()]
        normative = {"z": {k: 0.5 for k in keys}, "percentile": {k: 69.15 for k in keys}, "flags": []}
        reference = "test reference"
    return build_json_v2(r, acq, {"age": 40.0, "sex": "F"}, gate, normative, reference)


# ── eTIV and morphometry ──────────────────────────────────────────────────────
def test_etiv_is_freesurfer_constant_times_det():
    m = morphometry(_result(det=DET))
    assert ETIV_SCALE_ML == 1948.106
    assert m["etiv_ml"]["value"] == pytest.approx(1948.106 * DET, abs=0.01)
    assert "1948.106" in m["etiv_ml"]["method"] and "[VERIFY]" not in m["etiv_ml"]["method"]


def test_etiv_scales_linearly_and_normalised_volumes_use_it():
    a, b = morphometry(_result(det=0.6)), morphometry(_result(det=1.2))
    assert b["etiv_ml"]["value"] == pytest.approx(2 * a["etiv_ml"]["value"], rel=1e-5)
    r = _result(det=0.6)
    etiv = 1948.106 * 0.6
    for l, (k, _) in STRUCTURES.items():
        assert a["normalized_volumes_pct_etiv"][k] == pytest.approx(100 * r["volumes_ml"][str(l)]["jacobian"] / etiv, abs=1e-4)
    assert a["tissue"]["brain_parenchymal_fraction"] == pytest.approx(a["tissue"]["brain_parenchyma_ml"] / etiv, abs=1e-4)


def test_totals_and_asymmetry_use_primary_jacobian_volume():
    r = _result()
    m = morphometry(r)
    for name, (l, rr) in PAIRS.items():
        vl, vr = r["volumes_ml"][str(l)]["jacobian"], r["volumes_ml"][str(rr)]["jacobian"]
        assert m["volumes_ml"][f"total_{name}"]["value"] == pytest.approx(vl + vr, abs=1e-3)
        assert m["asymmetry_index_pct"][name] == pytest.approx((vl - vr) / ((vl + vr) / 2) * 100, abs=1e-2)


def test_zero_bilateral_volume_gives_null_with_reason():
    r = _result()
    l, rr = PAIRS["accumbens"]
    r["volumes_ml"][str(l)]["jacobian"] = r["volumes_ml"][str(rr)]["jacobian"] = 0.0
    m = morphometry(r)
    assert m["asymmetry_index_pct"]["accumbens"] is None and m["asymmetry_index_pct"]["accumbens_reason"]


# ── QC gate ──────────────────────────────────────────────────────────────────
def test_gate_reference_is_robust_and_needs_20_train_values():
    vals = [1.0] * 10 + [2.0] * 10 + [3.0] * 10 + [1000.0]          # outlier must not move median/MAD much
    ref = fit_gate_reference({f"s{i}": {"cjv": v} for i, v in enumerate(vals)}, [f"s{i}" for i in range(len(vals))])
    assert ref["iqms"]["cjv"]["median"] == 2.0
    assert ref["iqms"]["cjv"]["robust_sd"] == pytest.approx(1.4826 * 1.0)
    few = fit_gate_reference({f"s{i}": {"cjv": float(i)} for i in range(19)}, [f"s{i}" for i in range(19)])
    assert "cjv" not in few["iqms"]


def test_gate_reference_ignores_non_train_subjects():
    iq = {f"s{i}": {"cjv": float(i % 5)} for i in range(40)}
    iq.update({f"t{i}": {"cjv": 1e6} for i in range(40)})
    ref = fit_gate_reference(iq, [f"s{i}" for i in range(40)])
    assert ref["iqms"]["cjv"]["median"] == 2.0


def test_gate_thresholds_and_fail_dominates_warn():
    ref = {"iqms": {"cjv": {"median": 0.0, "robust_sd": 1.0}, "cnr": {"median": 0.0, "robust_sd": 1.0}},
           "warn_abs_z": 2.5, "fail_abs_z": 3.5, "reference": "test"}
    assert apply_gate({"cjv": 2.4, "cnr": -2.4}, ref)["status"] == "pass"
    assert apply_gate({"cjv": 2.6, "cnr": 0.0}, ref)["status"] == "warn"
    assert apply_gate({"cjv": -3.6, "cnr": 0.0}, ref)["status"] == "fail"
    g = apply_gate({"cjv": 3.6, "cnr": 2.6}, ref)
    assert g["status"] == "fail" and len(g["reasons"]) == 2
    g = apply_gate({"cjv": 2.6, "cnr": 3.6}, ref)                      # warn first, then fail
    assert g["status"] == "fail"
    assert apply_gate({"cjv": None, "cnr": 0.0}, ref)["robust_z"] == {"cnr": 0.0}


# ── Normative model ──────────────────────────────────────────────────────────
def _norm_table(n_train=100, n_test=20, seed=1):
    rng = np.random.default_rng(seed)
    table, train, folds = {}, [], {}
    for i in range(n_train + n_test):
        s = f"IXI{i:03d}"
        age, sex, site, etiv = float(rng.uniform(20, 80)), ("M", "F")[i % 2], ("Guys", "HH", "IOP")[i % 3], float(rng.uniform(1200, 1800))
        y = 5000 - 10 * age - 300 * (sex == "F") + 0.5 * etiv + float(rng.normal(0, 50))
        table[s] = {"age": age, "sex": sex, "site": site, "etiv": etiv, "f": y}
        folds[s] = i % 5
        if i < n_train:
            train.append(s)
    return table, train, folds


def test_normative_cross_fitting_train_z_does_not_see_own_value():
    table, train, folds = _norm_table()
    s = train[7]
    z0 = fit_predict(table, ["f"], train, folds)["subjects"][s]["z"]["f"]
    table[s]["f"] += 1000.0
    out = fit_predict(table, ["f"], train, folds)
    # the out-of-fold model for s never saw s, so z moves by exactly delta / sd(fold model)
    fit_ids = [t for t in train if folds[t] != folds[s]]
    age_mean = float(np.mean([table[t]["age"] for t in train]))
    X = _design(*([table[t][c] for t in fit_ids] for c in ("age", "sex", "site", "etiv")), age_mean)
    _, sd_fold = _fit(X, np.array([table[t]["f"] for t in fit_ids]))
    assert out["subjects"][s]["z"]["f"] - z0 == pytest.approx(1000.0 / sd_fold, abs=2e-3)
    same_fold = [t for t in train if folds[t] == folds[s] and t != s]
    other_fold = [t for t in train if folds[t] != folds[s]]
    base = fit_predict(_norm_table()[0], ["f"], train, folds)["subjects"]
    assert all(out["subjects"][t]["z"]["f"] == base[t]["z"]["f"] for t in same_fold)   # their model excludes the fold of s too
    assert any(out["subjects"][t]["z"]["f"] != base[t]["z"]["f"] for t in other_fold)


def test_normative_test_subjects_never_influence_the_model():
    table, train, folds = _norm_table()
    base = fit_predict(table, ["f"], train, folds)
    test_ids = [s for s in table if s not in train]
    for s in test_ids:
        table[s]["f"] += 1e4
    out = fit_predict(table, ["f"], train, folds)
    assert all(out["subjects"][t]["z"] == base["subjects"][t]["z"] for t in train)
    assert out["models"] == base["models"]


def test_normative_percentile_flags_and_missing_covariates():
    table, train, folds = _norm_table()
    table["IXI105"]["age"] = None
    out = fit_predict(table, ["f"], train, folds)
    for s, o in out["subjects"].items():
        if o["z"] is None:
            assert o["z_reason"] and o["percentile"] is None and o["percentile_reason"]
            continue
        z = o["z"]["f"]
        assert o["percentile"]["f"] == pytest.approx(100 * 0.5 * (1 + math.erf(z / math.sqrt(2))), abs=0.05)  # z is stored to 3 dp
        assert ("f" in o["flags"]) == (abs(z) > 2.0)
    assert out["subjects"]["IXI105"]["z"] is None


def test_normative_excluded_and_too_few_train_subjects():
    table, train, folds = _norm_table()
    full = fit_predict(table, ["f"], train, folds)
    excl = fit_predict(table, ["f"], train, folds, exclude=train[:10])
    assert excl["models"]["f"]["n_train"] == full["models"]["f"]["n_train"] - 10
    small = fit_predict(table, ["f"], train[: MIN_TRAIN - 1], folds)
    assert all(o["z"] is None and o["z_reason"] for o in small["subjects"].values())


# ── Schema, dictionary, nulls ────────────────────────────────────────────────
def test_v2_document_validates_and_every_null_is_explained():
    schema = json.loads((SCHEMA_DIR / "neurosphere-2.0.schema.json").read_text())
    doc = _doc()
    jsonschema.Draft202012Validator(schema).validate(doc)
    assert unexplained_nulls(doc) == []


def test_v2_document_with_null_normative_validates():
    schema = json.loads((SCHEMA_DIR / "neurosphere-2.0.schema.json").read_text())
    doc = _doc(normative={"z": None, "z_reason": "no age", "percentile": None, "percentile_reason": "no age", "flags": []},
               reference="test")
    jsonschema.Draft202012Validator(schema).validate(doc)
    assert unexplained_nulls(doc) == []


def test_schema_rejects_missing_etiv_and_wrong_version():
    schema = json.loads((SCHEMA_DIR / "neurosphere-2.0.schema.json").read_text())
    v = jsonschema.Draft202012Validator(schema)
    doc = _doc()
    bad = dict(doc); bad.pop("etiv_ml")
    assert not v.is_valid(bad)
    bad = dict(doc); bad["schema_version"] = "1.0"
    assert not v.is_valid(bad)


def test_committed_dictionary_matches_code():
    """dictionary.json is generated by scripts/write_schema.py; rerun it if this fails."""
    committed = json.loads((SCHEMA_DIR / "dictionary.json").read_text())
    assert committed == {"schema_version": "2.0.0", "features": feature_entries()}


def test_dictionary_keys_unique_and_every_path_resolves():
    e = feature_entries()
    for key in ("path", "compact", "obfuscated"):
        assert len({x[key] for x in e}) == len(e)
    doc = _doc()
    missing = [x["path"] for x, v in flatten(doc) if v is None and not x["path"].startswith(("normative.", "quality.iqms.fber"))]
    assert missing == []


# ── Serialisation ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("profile", PROFILES)
def test_profiles_are_deterministic_and_strict_json(profile):
    doc = _doc()
    a, b = serialize(doc, profile), serialize(json.loads(json.dumps(doc)), profile)
    assert a == b and "NaN" not in a


@pytest.mark.parametrize("key,profile", [("compact", "compact"), ("obfuscated", "obfuscated")])
def test_compact_round_trip(key, profile):
    doc = _doc()
    back = deserialize_compact(serialize(doc, profile), key=key)
    expected = {e["path"]: round(float(v), e["precision"]) for e, v in flatten(doc) if v is not None}
    assert back == expected


def test_csv_has_one_column_per_feature():
    doc = _doc()
    head, vals = serialize(doc, "csv").split("\n")
    assert len(head.split(",")) == len(vals.split(",")) == len(flatten(doc))


@pytest.mark.parametrize("profile", ["compact", "csv", "text", "obfuscated"])
def test_covariates_and_provenance_never_serialised(profile):
    doc = _doc()
    doc["covariates"]["age"] = 12345.678                                  # sentinel
    doc["provenance"]["sentinel"] = "PROVENANCE_SENTINEL"
    t = serialize(doc, profile)
    assert "12345.678" not in t and "PROVENANCE_SENTINEL" not in t


@pytest.mark.parametrize("profile", ["compact", "obfuscated", "csv", "text"])
def test_exclude_normative_removes_z_scores(profile):
    doc = _doc()
    n_with, n_without = len(flatten(doc)), len(flatten(doc, exclude=("normative",)))
    assert n_without < n_with
    t = serialize(doc, profile, exclude=("normative",))
    if profile in ("compact", "obfuscated"):
        assert not any(p.startswith("normative.") for p in deserialize_compact(t, key=profile))
    if profile == "text":
        assert "Normative" not in t


def test_to_vector_excludes_normative_by_default_and_uses_nan():
    names, values = to_vector(_doc())
    assert not any(n.startswith("normative.") for n in names)
    assert math.isnan(values[names.index("quality.iqms.fber")])


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        serialize(_doc(), "yaml")
