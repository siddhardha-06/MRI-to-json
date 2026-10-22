r"""Mapping verification, step 3: candidate re-registration (robust to small alignment differences).

The coarse scores in work/mapping/verification.csv compare our atlas-space image with every
TransMorph image directly; they are sensitive to small differences between our affine and
FreeSurfer's talairach.xfm. Here, for each verified subject, the candidate set is
{mapped k} U top-5 by residual NCC U top-5 by plain NCC; our 2x-downsampled image is affinely
registered to each candidate (ANTs, 2 mm grid) and the post-registration NCC inside the union brain
mask decides. Mapping confirmed if the mapped k is best; margin = NCC(mapped) - best other.

Reads work/mapping/ds2/<IXI>.npy (saved by verify_mapping.py). Writes work/mapping/verification_refined.csv.
Usage: python scripts/refine_mapping_verification.py [IXI002 IXI015 ...]   (default: all with ds2 files)
"""
import csv
import json
import sys
from pathlib import Path

import ants
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "work" / "mapping"
SHAPE = (80, 96, 112)


def norm(x):
    x = x - x.mean()
    return x / np.linalg.norm(x)


def post_reg_ncc(fixed_arr, moving_arr):
    f = ants.from_numpy(fixed_arr.astype(np.float32), spacing=(2.0, 2.0, 2.0))
    m = ants.from_numpy(moving_arr.astype(np.float32), spacing=(2.0, 2.0, 2.0))
    reg = ants.registration(fixed=f, moving=m, type_of_transform="Affine", aff_sampling=32, aff_random_sampling_rate=0.5)
    w = reg["warpedmovout"].numpy()
    mask = (fixed_arr > 0.02) | (w > 0.02)
    a, b = fixed_arr[mask], w[mask]
    return float(np.corrcoef(a, b)[0, 1])


def main(ids):
    ks = json.loads((MAP / "pkl_ds2_keys.json").read_text())
    stack = np.load(MAP / "pkl_ds2_stack.npy", mmap_mode="r")
    flat = stack.reshape(len(ks), -1).astype(np.float32)
    mean = flat.mean(0)
    coarse = {r["ixi_id"]: r for r in csv.DictReader(open(MAP / "verification.csv"))}
    ids = ids or sorted(p.stem for p in (MAP / "ds2").glob("IXI*.npy") if p.stem in coarse)
    out_f = MAP / "verification_refined.csv"
    done = {r["ixi_id"] for r in csv.DictReader(open(out_f))} if out_f.exists() else set()
    new = not out_f.exists()
    with open(out_f, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ixi_id", "mapped_k", "method", "n_candidates", "best_k", "ncc_mapped", "ncc_best_other", "margin", "confirmed"])
        if new:
            w.writeheader()
        for sid in ids:
            if sid in done or sid not in coarse:
                continue
            v = np.load(MAP / "ds2" / f"{sid}.npy").astype(np.float32)
            mk = int(coarse[sid]["mapped_k"])
            r_sc = np.array([norm(f - mean) @ norm(v - mean) for f in flat])
            n_sc = np.array([norm(f) @ norm(v) for f in flat])
            cand = {mk} | {ks[i] for i in np.argsort(-r_sc)[:5]} | {ks[i] for i in np.argsort(-n_sc)[:5]}
            scores = {k: post_reg_ncc(np.asarray(stack[ks.index(k)], np.float32).reshape(SHAPE), v.reshape(SHAPE)) for k in cand}
            best = max(scores, key=scores.get)
            other = max(s for k, s in scores.items() if k != mk)
            row = {"ixi_id": sid, "mapped_k": mk, "method": coarse[sid]["method"], "n_candidates": len(cand), "best_k": best,
                   "ncc_mapped": round(scores[mk], 4), "ncc_best_other": round(other, 4), "margin": round(scores[mk] - other, 4),
                   "confirmed": best == mk}
            w.writerow(row); fh.flush()
            print(row, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
