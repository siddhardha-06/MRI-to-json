r"""Run FastSurfer (Docker, --seg_only) on raw IXI scans; resumable. Reference for E2/E9.

Image: deepmi/fastsurfer:latest = sha256:11d594ae6fea... (FastSurfer 2.5.4). GPU needs a host driver
with CUDA >= 12.8 (ENV.md); otherwise --device cpu. Output: work/fastsurfer/<IXIxxx>/ (only mri/aseg*,
mri/orig.mgz, stats/, scripts/ are kept by FastSurfer with these flags) and work/fastsurfer/runs.csv
with wall time per subject. Flags as in the earlier runs, plus --no_hypothal (not needed for aseg).

Usage: python scripts/run_fastsurfer.py --subjects IXI012 IXI014 [--gpu] [--threads 4]
       python scripts/run_fastsurfer.py --test-split [--limit 3]
"""
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosphere.splits import discover_raw_ixi  # noqa: E402

IMAGE = "deepmi/fastsurfer@sha256:11d594ae6fea7b1dde5b66ba5124b6295c4eedf1f401e21c7319c22eefd55c50"
OUT = ROOT / "work" / "fastsurfer"


def test_split_subjects(include_legacy=False):
    """U-Net Test subjects; by default without the 5 that already have FastSurfer 2.5.4 output in
    results/legacy/validation_results/fastsurfer_output (same image; aseg unaffected by HypVINN)."""
    rows = list(csv.DictReader(open(ROOT / "data" / "transmorph_ixi_mapping.csv")))
    legacy = {p.name[:6] for p in (ROOT / "results/legacy/validation_results/fastsurfer_output").glob("IXI*")}
    return sorted(r["ixi_id"] for r in rows if r["split"] == "Test" and (include_legacy or r["ixi_id"] not in legacy))


def run(sid: str, raw: Path, gpu: bool, threads: int) -> dict:
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    if (out / sid / "mri" / "aseg.auto_noCCseg.mgz").exists():
        return {"subject": sid, "status": "cached"}
    cmd = ["docker", "run", "--rm", "--user", "0:0"] + (["--gpus", "all"] if gpu else []) + [
        "-v", f"{raw.parent.resolve().as_posix()}:/input:ro", "-v", f"{out.resolve().as_posix()}:/output", IMAGE,
        "--t1", f"/input/{raw.name}", "--sid", sid, "--sd", "/output", "--seg_only", "--no_cereb", "--no_hypothal",
        "--allow_root", "--threads", str(threads)] + ([] if gpu else ["--device", "cpu", "--viewagg_device", "cpu"])
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True)   # bytes: Docker output is not reliably decodable as cp1252
    wall = time.perf_counter() - t0
    dec = lambda b: (b or b"").decode("utf-8", errors="replace")[-20000:]  # noqa: E731
    (out / f"{sid}.log").write_text(dec(p.stdout) + "\n--- stderr ---\n" + dec(p.stderr), encoding="utf-8")
    ok = p.returncode == 0 and (out / sid / "mri" / "aseg.auto_noCCseg.mgz").exists()
    return {"subject": sid, "status": "ok" if ok else f"failed({p.returncode})", "wall_s": round(wall, 1),
            "device": "gpu" if gpu else "cpu", "threads": threads, "image": IMAGE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--test-split", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    subs = test_split_subjects() if a.test_split else (a.subjects or [])   # cached subjects are skipped inside run()
    if a.limit:
        subs = subs[: a.limit]
    raw = {r["subject_id"]: ROOT / "ixi-t1" / r["raw_file"] for r in discover_raw_ixi(ROOT / "ixi-t1")}
    log = OUT / "runs.csv"
    new = not log.exists()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(log, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject", "status", "wall_s", "device", "threads", "image"])
        if new:
            w.writeheader()
        for sid in subs:
            r = run(sid, raw[sid], a.gpu, a.threads)
            print(r, flush=True)
            if r["status"] != "cached":
                w.writerow(r); fh.flush()


if __name__ == "__main__":
    main()
