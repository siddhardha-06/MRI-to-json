"""Deterministic report of measured values only (Phase 1 interim; a later version adds
z-score-driven template sentences and the grounded LLM summary).

Rules: no finding, impression, normal/abnormal adjective, risk level or
recommendation is ever written unless it is derived from a value in the JSON. In Phase 1 there
are no such values, so the report lists measurements and states what was not computed and why.
"""
from __future__ import annotations

import html

from ..nulls import unexplained_nulls

TITLE = "NeuroSphere: measured values"
FOOTER = "Research use only. Not a clinical report and not for diagnosis."


def _rows(d: dict, keys: list[str]) -> list[tuple[str, str]]:
    out = []
    for k in keys:
        if k not in d:
            continue
        v = d[k]
        if v is None:
            out.append((k, f"not computed: {d.get(k + '_reason', '')}"))
        elif isinstance(v, list):
            out.append((k, " x ".join(str(x) for x in v)))
        else:
            out.append((k, str(v)))
    return out


def sections(doc: dict) -> list[tuple[str, list[tuple[str, str]]]]:
    """Report content as (section title, [(label, value)]) pairs; shared by the text and HTML renderers."""
    subj, acq, q = doc.get("subject", {}), doc.get("acquisition", {}), doc.get("quality", {})
    out = [
        ("Subject", _rows(subj, ["id", "source_file", "sha256"])),
        ("Acquisition (from the NIfTI header)", _rows(acq, ["site", "shape", "voxel_size_mm", "orientation_original",
                                                             "dtype_on_disk", "voxel_volume_ml", "field_of_view_ml"])),
    ]
    gate = q.get("gate", {})
    qrows = [("gate status", gate.get("status", "unknown"))] + [("gate note", r) for r in gate.get("reasons", [])]
    if q.get("iqms") is None:
        qrows.append(("in-house IQMs", f"not computed: {q.get('iqms_reason', '')}"))
    off = q.get("mriqc_official")
    if off is None:
        qrows.append(("official MRIQC", f"not available: {q.get('mriqc_official_reason', '')}"))
    else:
        for k, v in off.get("iqms", {}).items():
            if k.endswith("_reason"):
                continue
            qrows.append((f"MRIQC {k}", "null: " + off["iqms"].get(k + "_reason", "") if v is None else f"{v:.4g}"))
    out.append(("Image quality", qrows))
    nc = [(k, doc.get(k + "_reason", "")) for k in ("segmentation", "volumes_ml", "etiv_ml", "normalized_volumes",
                                                    "asymmetry_index_pct", "intensity", "normative") if k in doc and doc[k] is None]
    if nc:
        out.append(("Not computed in this version", nc))
    out.append(("Limitations", [("", s) for s in doc.get("limitations", [])]))
    return out


def render_text(doc: dict) -> str:
    if unexplained_nulls(doc):
        raise ValueError("document has null values without a reason")
    lines = [TITLE, "=" * len(TITLE)]
    for title, rows in sections(doc):
        lines += ["", title, "-" * len(title)]
        lines += [f"  {k}: {v}" if k else f"  - {v}" for k, v in rows]
    lines += ["", FOOTER]
    return "\n".join(lines) + "\n"


def render_html(doc: dict) -> str:
    if unexplained_nulls(doc):
        raise ValueError("document has null values without a reason")
    e = html.escape
    parts = [f'<div class="ns-report"><h2>{e(TITLE)}</h2>']
    for title, rows in sections(doc):
        parts.append(f"<h3>{e(title)}</h3><table>")
        parts += [f"<tr><th>{e(k)}</th><td>{e(v)}</td></tr>" if k else f'<tr><td colspan="2">{e(v)}</td></tr>' for k, v in rows]
        parts.append("</table>")
    parts.append(f'<p class="ns-footer">{e(FOOTER)}</p></div>')
    return "".join(parts)
