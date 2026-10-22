r"""Export the legacy v1 feature vectors exactly as shipped (decision Q1).

Source: rule_based_features.json (643 keys). Kept: the 581 raw-subject keys, values untouched.
Dropped: _n4 / _n4_n4 copies and _seg / _seg_seg label-map vectors.
The values carry the defects documented in legacy/extract_features.py.

Output (work/, not committed: derived data): work/legacy_v1/ixi_legacy_v1_raw.csv and a
provenance JSON with the sha256 of the source file.
Usage: .\.venv\Scripts\python scripts\export_legacy_v1.py
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosphere.io import is_derived, parse_ixi_name  # noqa: E402
from neurosphere.provenance import provenance, sha256_file  # noqa: E402

NAMES = [  # order of legacy/extract_features.py extract_all_features
    "mean", "std", "min", "max", "median", "p25", "p75", "skewness", "kurtosis", "dynamic_range", "mode", "bimodality",
    "csf_pct", "gm_pct", "wm_pct", "brain_tissue_pct", "otsu_t1", "otsu_t2", "gm_wm_cnr",
    "glcm_contrast", "glcm_correlation", "glcm_energy", "glcm_homogeneity", "mean_gradient", "std_gradient", "max_gradient", "shannon_entropy",
    "brain_area_pct", "eccentricity", "solidity", "extent", "perimeter", "ventricular_proxy_pct",
    "bilateral_symmetry_score", "asymmetry_index_pct", "low_frequency_energy", "high_frequency_energy", "frequency_energy_ratio",
    "cortical_edge_sharpness", "cortical_edge_std",
    "l_ventricle_ml", "r_ventricle_ml", "ventricle_ai", "l_thalamus_ml", "r_thalamus_ml", "thalamus_ai",
    "l_caudate_ml", "r_caudate_ml", "caudate_ai", "l_putamen_ml", "r_putamen_ml", "putamen_ai",
    "l_pallidum_ml", "r_pallidum_ml", "pallidum_ai", "l_hippocampus_ml", "r_hippocampus_ml", "hippocampus_ai",
    "l_amygdala_ml", "r_amygdala_ml", "amygdala_ai",
]
assert len(NAMES) == 61

src = ROOT / "rule_based_features.json"
feats = json.loads(src.read_text())
rows = []
for key, vec in sorted(feats.items()):
    if is_derived(key):
        continue
    sid, site = parse_ixi_name(key)
    rows.append({"subject_id": sid, "site": site, "source_key": key, **{f"v1_{n}": v for n, v in zip(NAMES, vec)}})
assert len({r["subject_id"] for r in rows}) == len(rows), "duplicate subjects"
out = ROOT / "work" / "legacy_v1" / "ixi_legacy_v1_raw.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
meta = {"n_subjects": len(rows), "n_source_keys": len(feats), "dropped_keys": len(feats) - len(rows),
        "source": {"file": src.name, "sha256": sha256_file(src)}, "feature_names": NAMES,
        "known_defects": "mid-sagittal head slab, inverted FFT, symmetry = inferior vs superior",
        "provenance": provenance(data_subset="raw-subject keys of rule_based_features.json")}
out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
print(f"{len(rows)} subjects written to {out}")
