r"""Fine-tuning pairs on raw scans (Phase 3, variant b). CPU only, resumable.

For each TransMorph Train/Val subject: our template-space U-Net input (work/pipeline_v2/<IXI>/template_t1.nii.gz)
and that subject's TransMorph FreeSurfer labels moved onto it (neurosphere/transfer.py: affine registration of the
TransMorph image to our image of the same subject, nearest label). Saved in the TransMorph pkl format so
training/train_unet3d.py --data-dir can read them: (image float16 in [0, 1], labels uint8), 160x192x224. Labels keep
TransMorph's full label set (e.g. 24 CSF, 251-255 corpus callosum); training maps non-target labels to background,
as for the original pkl files. labels_ok = no label that the TransMorph file does not have.

Excluded: IXI430 (defective FreeSurfer reference), IXI230 and IXI424 (mapping not confirmed by image). Test is never
touched. ANTs affine registration is multi-threaded and not bit-reproducible (re-runs differ by ~1e-4 in NCC).

Writes work/finetune_data/{Train,Val}/<IXI>.pkl (~20 MB each, ~9.5 GB) and work/finetune_data/transfer.csv.
Usage: python scripts/make_finetune_data.py [--split Train Val] [--limit N]
"""
import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosphere.transfer import labels_on_template, register_tm_to_ours  # noqa: E402

PIPE, OUT = ROOT / "work" / "pipeline_v2", ROOT / "work" / "finetune_data"
EXCLUDE = {"IXI430": "defective FreeSurfer reference", "IXI230": "mapping not confirmed by image",
           "IXI424": "mapping not confirmed by image"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int)
    ap.add_argument("--split", choices=["Train", "Val"], nargs="*", default=["Train", "Val"]); a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(ROOT / "data" / "transmorph_ixi_mapping.csv")) if r["split"] in a.split]
    rows = [r for r in sorted(rows, key=lambda r: (r["split"], r["ixi_id"])) if r["ixi_id"] not in EXCLUDE]
    log_f = OUT / "transfer.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    done = {r["ixi_id"] for r in csv.DictReader(open(log_f))} if log_f.exists() else set()
    todo = [r for r in rows if r["ixi_id"] not in done or not (OUT / r["split"] / f"{r['ixi_id']}.pkl").exists()][: a.limit]
    print(f"{len(rows)} pairs ({sum(r['split'] == 'Train' for r in rows)} Train, {sum(r['split'] == 'Val' for r in rows)} Val); "
          f"{len(todo)} to do; excluded {EXCLUDE}", flush=True)
    new = not log_f.exists()
    t_start = time.perf_counter()
    with open(log_f, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ixi_id", "k", "split", "transfer_ncc", "labels_ok", "seconds"])
        if new:
            w.writeheader()
        for i, r in enumerate(todo):
            t0 = time.perf_counter()
            sid, k, split = r["ixi_id"], int(r["k"]), r["split"]
            reg = register_tm_to_ours(sid, k, split, PIPE)
            lab = labels_on_template(reg)
            ok = set(np.unique(lab).tolist()) <= set(np.unique(reg.tm_seg.numpy()).astype(int).tolist())  # nothing new
            (OUT / split).mkdir(exist_ok=True)
            tmp = OUT / split / f"{sid}.pkl.tmp"
            with open(tmp, "wb") as f:
                pickle.dump((reg.ours.numpy().astype(np.float16), lab.astype(np.uint8)), f, protocol=4)
            tmp.replace(OUT / split / f"{sid}.pkl")
            row = {"ixi_id": sid, "k": k, "split": split, "transfer_ncc": round(reg.ncc, 4), "labels_ok": ok,
                   "seconds": round(time.perf_counter() - t0, 1)}
            w.writerow(row); fh.flush()
            el = time.perf_counter() - t_start
            print(f"[{i + 1}/{len(todo)}] {sid} {split} NCC {reg.ncc:.3f} labels_ok={ok} {row['seconds']}s  "
                  f"eta {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
