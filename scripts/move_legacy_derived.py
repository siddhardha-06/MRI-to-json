r"""Move derived files that the old dashboard code wrote into the raw dataset
(ixi-t1/<subject>/*_n4.nii, *_n4_n4.nii, *_seg.nii, *_seg_seg.nii) to
work/legacy_derived/<subject>/, keeping them (never deleted).

Writes a manifest (old path, new path, sha256, bytes) to
data/manifests/legacy_derived_moves.csv and work/legacy_derived/MANIFEST.csv.
sha256 is computed before the move and verified after it. Idempotent: refuses to
overwrite an existing destination.

Usage: python scripts/move_legacy_derived.py [--dry-run]
"""
import argparse, csv, hashlib, os, re, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "ixi-t1", ROOT / "work" / "legacy_derived"
PAT = re.compile(r"_(n4|n4_n4|seg|seg_seg)\.nii(\.gz)?$")

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
files = sorted(p for p in SRC.rglob("*") if p.is_file() and PAT.search(p.name))
rows = []
for p in files:
    rel = p.relative_to(SRC)
    dst = DST / rel
    if dst.exists():
        raise SystemExit(f"destination exists, refusing to overwrite: {dst}")
    digest, size = sha256(p), p.stat().st_size
    if not a.dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dst))
        if sha256(dst) != digest:
            raise SystemExit(f"sha256 mismatch after move: {dst}")
    rows.append({"old_path": f"ixi-t1/{rel.as_posix()}", "new_path": f"work/legacy_derived/{rel.as_posix()}",
                 "sha256": digest, "bytes": size})
print(f"{'would move' if a.dry_run else 'moved'} {len(rows)} files")
if not a.dry_run and rows:
    for out in (ROOT / "data" / "manifests" / "legacy_derived_moves.csv", DST / "MANIFEST.csv"):
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
