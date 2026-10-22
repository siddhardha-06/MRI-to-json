"""Serialisation profiles of a NeuroSphere 2.0.0 JSON. Deterministic and round-trip tested.

  full        the document as is (archiving, provenance)
  compact     minified {compact_key: value}, values rounded to the dictionary precision, no provenance,
              no covariates, no strings except the QC gate status. The LLM-training format.
  csv         one header line of anonymous column names (c001..) and one value line: the "bare table" baseline
  text        one natural-language sentence per feature group (TabLLM-style)
  obfuscated  the compact profile with meaningless keys (f001..), to test whether self-describing keys help

`exclude` drops feature groups (e.g. exclude=("normative",) whenever age or sex is a prediction target:
z-scores encode age and sex). Covariates (age/sex) are never serialised by the compact/csv/text/obfuscated profiles.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from .dictionary import feature_entries

PROFILES = ("full", "compact", "csv", "text", "obfuscated")
GROUP_OF = {"volumes_ml": "volumes", "asymmetry_index_pct": "asymmetry", "tissue": "tissue", "etiv_ml": "etiv",
            "brain_mask_ml": "etiv", "normalized_volumes_pct_etiv": "normalized", "intensity": "intensity",
            "quality": "quality", "normative": "normative"}
TOKENIZER_FILE = Path(__file__).resolve().parents[1] / "data" / "tokenizers" / "qwen2.5_tokenizer.json"


@lru_cache(maxsize=1)
def _entries():
    return feature_entries()


def _get(doc, path):
    cur = doc
    parts = path.split(".")
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    if isinstance(cur, dict) and "value" in cur:
        cur = cur["value"]
    return cur if isinstance(cur, (int, float)) and not isinstance(cur, bool) else None


def _fmt(v, prec):
    return round(float(v), prec)


def flatten(doc: dict, exclude=()) -> list[tuple[dict, float | None]]:
    """(dictionary entry, value or None) for every dictionary feature, in dictionary order."""
    return [(e, _get(doc, e["path"])) for e in _entries() if GROUP_OF[e["path"].split(".")[0]] not in set(exclude)]


def to_vector(doc: dict, exclude=("normative",)) -> tuple[list[str], list[float]]:
    """Names and values for tabular ML; missing values are NaN. Normative z excluded by default (leakage)."""
    pairs = flatten(doc, exclude)
    return [e["path"] for e, _ in pairs], [float("nan") if v is None else float(v) for _, v in pairs]


def _compact_obj(doc, exclude, key):
    out = {"q.gate": doc.get("quality", {}).get("gate", {}).get("status")} if "quality" not in set(exclude) else {}
    for e, v in flatten(doc, exclude):
        if v is not None:
            out[e[key]] = _fmt(v, e["precision"])
    return out


def _text(doc, exclude):
    groups: dict[str, list[str]] = {}
    for e, v in flatten(doc, exclude):
        if v is None:
            continue
        g = GROUP_OF[e["path"].split(".")[0]]
        unit = "" if e["unit"] in ("ratio", "a.u.", "z") else f" {e['unit']}"
        groups.setdefault(g, []).append(f"{e['description'].lower()} {_fmt(v, e['precision'])}{unit}")
    titles = {"volumes": "Volumes", "asymmetry": "Asymmetry", "tissue": "Tissue", "etiv": "Head size", "normalized": "Normalised volumes",
              "intensity": "Intensity", "quality": "Image quality", "normative": "Normative comparison"}
    lines = [f"{titles[g]}: " + "; ".join(items) + "." for g, items in groups.items()]
    if "quality" not in set(exclude):
        lines.insert(0, f"Quality gate: {doc.get('quality', {}).get('gate', {}).get('status')}.")
    return "\n".join(lines)


def serialize(doc: dict, profile: str = "full", exclude=()) -> str:
    if profile == "full":
        return json.dumps(doc, indent=2, allow_nan=False)
    if profile == "compact":
        return json.dumps(_compact_obj(doc, exclude, "compact"), separators=(",", ":"), allow_nan=False)
    if profile == "obfuscated":
        return json.dumps(_compact_obj(doc, exclude, "obfuscated"), separators=(",", ":"), allow_nan=False)
    if profile == "csv":
        pairs = flatten(doc, exclude)
        head = ",".join(f"c{i + 1:03d}" for i in range(len(pairs)))
        vals = ",".join("" if v is None else repr(_fmt(v, e["precision"])) for e, v in pairs)
        return head + "\n" + vals
    if profile == "text":
        return _text(doc, exclude)
    raise ValueError(f"unknown profile {profile!r}; choose from {PROFILES}")


def deserialize_compact(s: str, key: str = "compact") -> dict[str, float]:
    """Inverse of the compact/obfuscated profiles: {dictionary path: value}."""
    by = {e[key]: e["path"] for e in _entries()}
    return {by[k]: v for k, v in json.loads(s).items() if k in by}


@lru_cache(maxsize=2)
def _tokenizer(path: str):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def count_tokens(text: str, tokenizer_file: str | Path = TOKENIZER_FILE) -> int:
    """Tokens under the Qwen2.5 tokenizer (shared by Qwen2.5-0.5B/1.5B-Instruct)."""
    return len(_tokenizer(str(tokenizer_file)).encode(text).ids)
