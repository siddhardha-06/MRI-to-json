r"""Generate neurosphere/schema/neurosphere-2.0.schema.json (JSON Schema 2020-12) and dictionary.json
from neurosphere/dictionary.py, so schema, dictionary and code cannot drift apart.
Usage: python scripts/write_schema.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosphere.dictionary import PAIRS, STRUCTURES, feature_entries  # noqa: E402

NUM = {"type": "number"}
NUM_OR_NULL = {"type": ["number", "null"]}
REASON = {"type": "string", "minLength": 1}


def obj(props, required=None, additional=True):
    o = {"type": "object", "properties": props}
    if required:
        o["required"] = required
    if not additional:
        o["additionalProperties"] = False
    return o


vol_item = obj({"value": {"type": "number", "minimum": 0}, "native_grid_check": {"type": "number", "minimum": 0}}, ["value"])
volumes = obj({**{k: vol_item for k, _ in STRUCTURES.values()}, **{f"total_{p}": vol_item for p in PAIRS}},
              [k for k, _ in STRUCTURES.values()] + [f"total_{p}" for p in PAIRS], additional=False)
asym = obj({**{p: NUM_OR_NULL for p in PAIRS}, **{f"{p}_reason": REASON for p in PAIRS}}, list(PAIRS))
z_or_null = {"type": ["object", "null"], "additionalProperties": NUM}
schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://neurosphere.local/schema/neurosphere-2.0.schema.json",
    "title": "NeuroSphere T1w MRI representation 2.0.0",
    "description": "Null values must carry a sibling '<key>_reason' string (checked at runtime by neurosphere.nulls).",
    "type": "object",
    "required": ["schema_version", "subject", "acquisition", "quality", "segmentation", "volumes_ml", "etiv_ml",
                 "normalized_volumes_pct_etiv", "asymmetry_index_pct", "tissue", "intensity", "normative", "limitations", "provenance"],
    "properties": {
        "schema_version": {"const": "2.0.0"},
        "subject": obj({"id": {"type": "string"}, "source_file": {"type": "string"}, "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}},
                       ["id", "source_file", "sha256"]),
        "acquisition": obj({"shape": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
                            "voxel_size_mm": {"type": "array", "items": NUM, "minItems": 3, "maxItems": 3},
                            "orientation_original": {"type": "string", "pattern": "^[LRAPSI]{3}$"},
                            "site": {"type": ["string", "null"]}}, ["shape", "voxel_size_mm", "orientation_original"]),
        "covariates": obj({"age": NUM_OR_NULL, "sex": {"enum": ["M", "F", None]}}),
        "quality": obj({"iqms": {"type": "object"},
                        "gate": obj({"status": {"enum": ["pass", "warn", "fail", "not_computed"]}, "reasons": {"type": "array", "items": {"type": "string"}},
                                     "robust_z": {"type": "object"}, "thresholds": {"type": "object"}}, ["status", "reasons"]),
                        "mriqc_official": {"type": ["object", "null"]}}, ["iqms", "gate"]),
        "segmentation": obj({"model": {"type": "string"}, "checkpoint": {"type": "object"}, "label_set": {"type": "string"},
                             "space": {"type": "string"}, "volume_method": {"type": "string"}}, ["model", "checkpoint", "label_set", "space"]),
        "volumes_ml": volumes,
        "etiv_ml": obj({"value": {"type": "number", "minimum": 0}, "method": {"type": "string"}}, ["value", "method"]),
        "brain_mask_ml": NUM,
        "normalized_volumes_pct_etiv": obj({k: NUM for k, _ in STRUCTURES.values()}, [k for k, _ in STRUCTURES.values()]),
        "asymmetry_index_pct": asym,
        "tissue": obj({k: NUM for k in ("gray_matter_ml", "white_matter_ml", "ventricular_csf_ml", "brain_parenchyma_ml",
                                         "brain_parenchymal_fraction", "gm_wm_volume_ratio", "ventricle_brain_ratio")},
                      ["gray_matter_ml", "white_matter_ml", "ventricular_csf_ml", "brain_parenchyma_ml"]),
        "intensity": {"type": "object"},
        "normative": {"type": ["object", "null"], "properties": {"reference": {"type": "string"}, "z": z_or_null,
                                                                  "percentile": z_or_null, "flags": {"type": "array", "items": {"type": "string"}}}},
        "limitations": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "provenance": {"type": "object"},
    },
}
d = ROOT / "neurosphere" / "schema"
d.mkdir(parents=True, exist_ok=True)
(d / "neurosphere-2.0.schema.json").write_text(json.dumps(schema, indent=2))
(d / "dictionary.json").write_text(json.dumps({"schema_version": "2.0.0", "features": feature_entries()}, indent=1))
print(f"schema + dictionary ({len(feature_entries())} features) written to {d}")
