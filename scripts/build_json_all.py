r"""Build NeuroSphere 2.0.0 JSON for every subject from the cached pipeline outputs (Phase 3). CPU only, minutes.

Steps: QC gate reference (train IQMs) -> gate for everyone -> normative model (train subjects whose gate
is not 'fail'; 5-fold cross-fitted) -> JSON per subject, schema-validated -> summaries.
Inputs:  work/pipeline_v2/<IXI>/result.json, data/splits/ixi_final.csv, data/raw/IXI.xls
Outputs: work/json/<IXI>.json (all 581), results/phase3/{gate_summary,normative_models,serialization_sizes}.json
Usage:   python scripts/build_json_all.py [--allow-missing]
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import jsonschema
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosphere.demographics import load_demographics  # noqa: E402
from neurosphere.dictionary import PAIRS, STRUCTURES  # noqa: E402
from neurosphere.features import morphometry  # noqa: E402
from neurosphere.jsonio import build_json_v2, write_json  # noqa: E402
from neurosphere.normative import fit_predict  # noqa: E402
from neurosphere.provenance import provenance  # noqa: E402
from neurosphere.qc import apply_gate, fit_gate_reference  # noqa: E402
from neurosphere.serialize import PROFILES, count_tokens, serialize  # noqa: E402
from neurosphere.splits import discover_raw_ixi  # noqa: E402

PIPE, OUT, RES = ROOT / "work" / "pipeline_v2", ROOT / "work" / "json", ROOT / "results" / "phase3"
SMOKE_RES = ROOT / "work" / "smoke" / "phase3"   # --allow-missing (incomplete cohort) never writes to results/
NORM_FEATURES = [k for k, _ in STRUCTURES.values()] + [f"total_{p}" for p in PAIRS] + ["gray_matter_ml", "white_matter_ml", "ventricular_csf_ml", "brain_parenchyma_ml"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--allow-missing", action="store_true"); a = ap.parse_args()
    split = {r["subject_id"]: r for r in csv.DictReader(open(ROOT / "data" / "splits" / "ixi_final.csv"))}
    raw = {r["subject_id"]: ROOT / "ixi-t1" / r["raw_file"] for r in discover_raw_ixi(ROOT / "ixi-t1")}
    results = {s: json.loads((PIPE / s / "result.json").read_text()) for s in split if (PIPE / s / "result.json").exists()}
    missing = sorted(set(split) - set(results))
    if missing and not a.allow_missing:
        raise SystemExit(f"{len(missing)} subjects have no pipeline result (e.g. {missing[:5]}); rerun the pipeline or pass --allow-missing")
    demo = load_demographics(sorted(results))
    train = [s for s in results if split[s]["split"] == "train"]

    gate_ref = fit_gate_reference({s: r["iqms"] for s, r in results.items()}, train)
    gates = {s: apply_gate(r["iqms"], gate_ref) for s, r in results.items()}

    table = {}
    for s, r in results.items():
        m = morphometry(r)
        row = {"age": demo[s].get("age"), "sex": demo[s].get("sex"), "site": split[s]["site"], "etiv": m["etiv_ml"]["value"]}
        for f in NORM_FEATURES:
            row[f] = m["volumes_ml"][f]["value"] if f in m["volumes_ml"] else m["tissue"][f]
        table[s] = row
    folds = {s: int(split[s]["cv_fold"]) for s in results}
    norm = fit_predict(table, NORM_FEATURES, train, folds, exclude=[s for s in train if gates[s]["status"] == "fail"])

    schema = json.loads((ROOT / "neurosphere" / "schema" / "neurosphere-2.0.schema.json").read_text())
    val = jsonschema.Draft202012Validator(schema)
    sizes = {p: [] for p in PROFILES}
    sizes_no_norm = {p: [] for p in PROFILES if p != "full"}
    OUT.mkdir(parents=True, exist_ok=True)
    for s, r in sorted(results.items()):
        hdr = nib.load(str(raw[s]))
        acq = {"shape": [int(x) for x in hdr.shape[:3]], "voxel_size_mm": [round(float(z), 6) for z in hdr.header.get_zooms()[:3]],
               "orientation_original": r["orientation_original"], "site": split[s]["site"]}
        doc = build_json_v2(r, acq, demo[s], gates[s], norm["subjects"][s], norm["reference"])
        val.validate(doc)
        write_json(doc, OUT / f"{s}.json")
        for p in PROFILES:
            t = serialize(doc, p)
            sizes[p].append((len(t.encode()), count_tokens(t)))
            if p != "full":
                t2 = serialize(doc, p, exclude=("normative",))
                sizes_no_norm[p].append((len(t2.encode()), count_tokens(t2)))

    RES_DIR = SMOKE_RES if missing else RES
    RES_DIR.mkdir(parents=True, exist_ok=True)
    gs = {"reference": gate_ref, "by_split": {}, "fail_or_warn_subjects": {s: g for s, g in gates.items() if g["status"] != "pass"}}
    for sp in ("train", "val", "test"):
        gs["by_split"][sp] = dict(Counter(gates[s]["status"] for s in results if split[s]["split"] == sp))
    gs["provenance"] = provenance(data_subset=f"{len(results)} subjects")
    (RES_DIR / "gate_summary.json").write_text(json.dumps(gs, indent=2))
    (RES_DIR / "normative_models.json").write_text(json.dumps({"reference": norm["reference"], "age_mean_centre": norm["age_mean_centre"],
                                                           "models": norm["models"], "features": NORM_FEATURES,
                                                           "excluded_gate_fail_train": [s for s in train if gates[s]["status"] == "fail"],
                                                           "provenance": provenance()}, indent=2))
    def summ(v):
        b = np.array([x[0] for x in v]); t = np.array([x[1] for x in v])
        return {"bytes_median": float(np.median(b)), "bytes_iqr": [float(np.percentile(b, 25)), float(np.percentile(b, 75))],
                "tokens_median": float(np.median(t)), "tokens_iqr": [float(np.percentile(t, 25)), float(np.percentile(t, 75))], "n": int(len(v))}
    raw_bytes = np.array([raw[s].stat().st_size for s in results])
    sz = {"tokenizer": "Qwen2.5 (data/tokenizers/qwen2.5_tokenizer.json)", "profiles": {p: summ(v) for p, v in sizes.items()},
          "profiles_without_normative": {p: summ(v) for p, v in sizes_no_norm.items()},
          "raw_nifti_bytes_median": float(np.median(raw_bytes)), "provenance": provenance()}
    (RES_DIR / "serialization_sizes.json").write_text(json.dumps(sz, indent=2))
    print(json.dumps({"n_json": len(results), "missing": missing, "gate_by_split": gs["by_split"],
                      "bytes_tokens_median": {p: (v["bytes_median"], v["tokens_median"]) for p, v in sz["profiles"].items()}}, indent=1))


if __name__ == "__main__":
    main()
