r"""Run the raw-scan pipeline (preprocess + v2 U-Net + QC/intensity) on all raw IXI scans. Resumable.

Output: work/pipeline_v2/<IXIxxx>/ (see neurosphere/pipeline.py for the files) and
work/pipeline_v2/_run_log.csv (one row per subject: status, seconds, bytes). A subject is skipped if
its result.json exists and was made with the same checkpoint. Failures are recorded, never hidden.
Usage: python scripts/run_pipeline_all.py [--limit N] [--subjects IXI002 ...]
"""
import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosphere.pipeline import run_subject  # noqa: E402
from neurosphere.segment import Segmenter  # noqa: E402
from neurosphere.splits import discover_raw_ixi  # noqa: E402

OUT = ROOT / "work" / "pipeline_v2"


def dir_bytes(d):
    return sum(p.stat().st_size for p in Path(d).glob("*") if p.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int); ap.add_argument("--subjects", nargs="*")
    a = ap.parse_args()
    rows = discover_raw_ixi(ROOT / "ixi-t1")
    if a.subjects:
        rows = [r for r in rows if r["subject_id"] in set(a.subjects)]
    rows = rows[: a.limit] if a.limit else rows
    seg = Segmenter()
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "_run_log.csv"
    new = not log.exists()
    t_start = time.perf_counter(); n_done = 0
    with open(log, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject", "status", "seconds", "bytes", "error"])
        if new:
            w.writeheader()
        for i, r in enumerate(rows):
            sid = r["subject_id"]
            rf = OUT / sid / "result.json"
            if rf.exists() and json.loads(rf.read_text()).get("checkpoint", {}).get("sha256") == seg.info["sha256"]:
                continue
            t0 = time.perf_counter()
            try:
                run_subject(ROOT / "ixi-t1" / r["raw_file"], sid, seg, work_root=OUT, force=True)
                row = {"subject": sid, "status": "ok", "seconds": round(time.perf_counter() - t0, 1), "bytes": dir_bytes(OUT / sid), "error": ""}
            except Exception as e:
                row = {"subject": sid, "status": "failed", "seconds": round(time.perf_counter() - t0, 1), "bytes": "", "error": repr(e)[:300]}
                (OUT / f"_error_{sid}.txt").write_text(traceback.format_exc())
            w.writerow(row); fh.flush()
            n_done += 1
            el = time.perf_counter() - t_start
            print(f"[{i + 1}/{len(rows)}] {sid} {row['status']} {row['seconds']}s  elapsed {el / 3600:.2f} h  "
                  f"eta {el / n_done * (len(rows) - i - 1) / 3600:.2f} h", flush=True)
    print("done")


if __name__ == "__main__":
    import faulthandler
    OUT.mkdir(parents=True, exist_ok=True)
    _fh = open(OUT / "_faulthandler.txt", "a")          # hard crashes (segfaults) leave a stack dump here
    faulthandler.enable(_fh)
    try:
        main()
    except BaseException:                                  # incl. KeyboardInterrupt: always leave a full traceback on disk
        (OUT / "_fatal.txt").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + traceback.format_exc())
        raise
