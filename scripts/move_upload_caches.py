r"""Move the v1 dashboard caches out of uploads/ (decision: Phase 1 review, item 1).

Kept in place: each uploaded scan uploads/<id>/<id>.nii(.gz). Moved to work/legacy_derived/uploads/
(same relative path): everything else the v1 server wrote there (features.json, _n4/_seg volumes,
_report.txt, and the MRIQC temp/output folders bids_temp/ and bids_out/). Nothing is deleted.
sha256 is computed before and verified after each move; refuses to overwrite.

Manifest: data/manifests/upload_caches_moves.csv (+ work/legacy_derived/uploads/MANIFEST.csv).
Usage: python scripts/move_upload_caches.py [--dry-run]
"""
import argparse
import csv
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "uploads", ROOT / "work" / "legacy_derived" / "uploads"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_kept_scan(p: Path) -> bool:
    folder = p.relative_to(SRC).parts[0]
    return len(p.relative_to(SRC).parts) == 2 and p.name in (f"{folder}.nii", f"{folder}.nii.gz")


ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
files = sorted(p for p in SRC.rglob("*") if p.is_file() and not is_kept_scan(p))
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
    rows.append({"old_path": f"uploads/{rel.as_posix()}", "new_path": f"work/legacy_derived/uploads/{rel.as_posix()}",
                 "sha256": digest, "bytes": size})
if not a.dry_run:
    for d in sorted((p for p in SRC.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
        if not any(d.iterdir()):
            d.rmdir()   # only empty directories left behind by the moves
print(f"{'would move' if a.dry_run else 'moved'} {len(rows)} files, {sum(r['bytes'] for r in rows) / 2**20:.0f} MB")
if not a.dry_run and rows:
    for out in (ROOT / "data" / "manifests" / "upload_caches_moves.csv", DST / "MANIFEST.csv"):
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
