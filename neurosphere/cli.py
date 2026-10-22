"""Command line: python -m neurosphere {run,batch,serialize}."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .io import is_derived, strip_nifti_ext
from .jsonio import build_interim_json, write_json
from .serialize import PROFILES, serialize


def _cmd_run(a) -> int:
    doc = build_interim_json(a.input)
    if a.output:
        write_json(doc, a.output)
    else:
        sys.stdout.write(serialize(doc, a.profile) + "\n")
    return 0


def _cmd_batch(a) -> int:
    files = sorted(p for p in Path(a.input_dir).rglob("*.nii*") if p.is_file() and not is_derived(p.name))
    if a.limit:
        files = files[: a.limit]
    out_dir = Path(a.output_dir)
    failures = []
    for f in files:
        target = out_dir / f"{strip_nifti_ext(f.name)}.json"
        if target.exists() and not a.overwrite:
            continue
        try:
            write_json(build_interim_json(f), target)
        except Exception as e:  # recorded, never silently dropped
            failures.append({"file": str(f), "error": repr(e)})
    if failures:
        write_json({"failures": failures}, out_dir / "_failures.json")
    print(f"{len(files) - len(failures)} ok, {len(failures)} failed -> {out_dir}")
    return 1 if failures else 0


def _cmd_serialize(a) -> int:
    for p in sorted(Path(a.json_dir).glob("*.json")):
        sys.stdout.write(serialize(json.loads(p.read_text(encoding="utf-8")), a.profile) + "\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="neurosphere")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one NIfTI -> JSON")
    r.add_argument("input"); r.add_argument("-o", "--output"); r.add_argument("--profile", choices=PROFILES, default="full")
    b = sub.add_parser("batch", help="every raw NIfTI under a folder -> JSON files")
    b.add_argument("input_dir"); b.add_argument("-o", "--output-dir", required=True)
    b.add_argument("--limit", type=int); b.add_argument("--overwrite", action="store_true")
    s = sub.add_parser("serialize", help="re-serialise a folder of JSON files")
    s.add_argument("json_dir"); s.add_argument("--profile", choices=PROFILES, required=True)
    a = ap.parse_args(argv)
    return {"run": _cmd_run, "batch": _cmd_batch, "serialize": _cmd_serialize}[a.cmd](a)
