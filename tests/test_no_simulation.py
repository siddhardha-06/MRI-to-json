"""Acceptance grep (Phase 1): no simulated values or hard-coded findings in runnable code.

Scans every .py/.js/.html file of the repository except: the Phase 0 evidence (audit/), the
tests themselves, documentation, the venv and work/. legacy/ IS scanned: legacy code is kept for
reproducibility, but simulated values and fixed findings were removed from it too.
"""
import re

from tests.conftest import ROOT

EXCLUDED_DIRS = {".venv", "audit", "tests", "docs", "work", ".git", "__pycache__", "node_modules"}
BANNED = [
    r"def simulate_\w*", r"simulate_mriqc\w*\(", r"[\"']simulated[\"']\s*:\s*True", r"1400\.0",
    r"(?i)within normal limits", r"(?i)no focal mass", r"(?i)evidence of cortical dysplasia", r"(?i)risk assessment is low",
    r"(?i)generally normal", r"(?i)no significant pathology", r"(?i)normal gray-white", r"(?i)follow-up (volumetric )?imaging may be considered",
    r"(?i)what gemini would generate",
]
# Files that only *name* the removed text in order to explain the removal.
ALLOWED = {("legacy/demo_report.py", r"(?i)within normal limits"), ("legacy/demo_report.py", r"(?i)no focal mass"),
           ("legacy/demo_report.py", r"(?i)what gemini would generate")}


def _files():
    for p in ROOT.rglob("*"):
        if p.suffix in {".py", ".js", ".html"} and p.is_file() and not EXCLUDED_DIRS & set(p.relative_to(ROOT).parts):
            yield p


def test_no_simulated_values_or_hardcoded_findings():
    hits = []
    for p in _files():
        rel = p.relative_to(ROOT).as_posix()
        text = p.read_text(encoding="utf-8", errors="ignore")
        for pat in BANNED:
            if (rel, pat) in ALLOWED:
                continue
            for m in re.finditer(pat, text):
                hits.append(f"{rel}:{text.count(chr(10), 0, m.start()) + 1}: {m.group(0)}")
    assert not hits, "\n".join(hits)


def test_dashboard_does_not_call_legacy_code():
    src = (ROOT / "app_server.py").read_text(encoding="utf-8") + (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    for name in ("extract_features", "train_regressor", "build_structured_json", "run_feature_extraction", "predicted_features"):
        assert name not in src, name
