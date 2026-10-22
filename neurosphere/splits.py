"""Subject-level splits.

All files of one IXI subject (raw scan and any derived _n4/_seg copies) belong to one subject
and therefore to one split. The v1 regressor shuffled files, so _n4 copies of 7
subjects landed in both train and validation.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from .io import is_derived, parse_ixi_name

SPLIT_NAMES = ("train", "val", "test")


def subject_of(name: str) -> str:
    """IXI subject ID ('IXI002') of a file, folder or feature-file key; raises if it is not an IXI name."""
    sid, _ = parse_ixi_name(name)
    if sid is None:  # fall back to the folder part of a relative path (ixi-t1/<IXIxxx-Site-nnnn-T1.nii>/<file>)
        sid = next((s for s in (parse_ixi_name(part)[0] for part in str(name).replace("\\", "/").split("/")) if s), None)
    if sid is None:
        raise ValueError(f"not an IXI name: {name!r}")
    return sid


def discover_raw_ixi(ixi_dir: str | Path) -> list[dict]:
    """One row per raw IXI T1 scan: subject_id, site, raw_file (path relative to ixi_dir). Derived files are skipped."""
    ixi_dir = Path(ixi_dir)
    rows = []
    for p in sorted(ixi_dir.rglob("*.nii*")):
        if not p.is_file() or is_derived(p.name):
            continue
        sid, site = parse_ixi_name(p.name)
        if sid is None:
            continue
        rows.append({"subject_id": sid, "site": site, "raw_file": p.relative_to(ixi_dir).as_posix()})
    seen = defaultdict(list)
    for r in rows:
        seen[r["subject_id"]].append(r["raw_file"])
    dup = {k: v for k, v in seen.items() if len(v) > 1}
    if dup:
        raise ValueError(f"more than one raw scan for subjects: {dup}")
    return rows


def make_subject_split(rows: list[dict], seed: int = 42, fractions: tuple[float, float, float] = (0.7, 0.1, 0.2),
                       n_folds: int = 5, stratify_by: str = "site") -> list[dict]:
    """Deterministic subject-level train/val/test split, stratified by `stratify_by`.

    Also assigns cv_fold (0..n_folds-1) to train+val subjects for repeated-CV experiments;
    test subjects get cv_fold = -1. Returns new rows with 'split' and 'cv_fold' added.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    strata = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["subject_id"]):
        strata[r[stratify_by]].append(r)
    out = []
    for key in sorted(strata):
        group = strata[key]
        order = rng.permutation(len(group))
        n = len(group)
        n_test = int(round(fractions[2] * n))
        n_val = int(round(fractions[1] * n))
        for rank, idx in enumerate(order):
            split = "test" if rank < n_test else "val" if rank < n_test + n_val else "train"
            out.append({**group[idx], "split": split})
    dev = [r for r in out if r["split"] != "test"]
    fold_order = rng.permutation(len(dev))
    for k, idx in enumerate(fold_order):
        dev[idx]["cv_fold"] = int(k % n_folds)
    for r in out:
        r.setdefault("cv_fold", -1)
    out.sort(key=lambda r: r["subject_id"])
    check_no_leakage({r["raw_file"]: r["split"] for r in out})
    return out


def check_no_leakage(assignment: Mapping[str, str]) -> None:
    """Raise if any subject has files in more than one split. Keys are file names/keys, values are split names."""
    by_subject = defaultdict(set)
    for name, split in assignment.items():
        by_subject[subject_of(name)].add(split)
    leaks = {s: sorted(v) for s, v in by_subject.items() if len(v) > 1}
    if leaks:
        raise ValueError(f"{len(leaks)} subjects in more than one split: {dict(list(leaks.items())[:10])}")


def write_split_csv(rows: Iterable[dict], path: str | Path) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject_id", "site", "raw_file", "split", "cv_fold"])
        w.writeheader()
        w.writerows({k: r[k] for k in w.fieldnames} for r in rows)


def read_split_csv(path: str | Path) -> list[dict]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["cv_fold"] = int(r["cv_fold"])
    return rows
