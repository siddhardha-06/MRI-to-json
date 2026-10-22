r"""TransMorph <-> IXI mapping, full image check from the cached v2 pipeline outputs (Phase 3, Next step 2).

For every mapped subject (data/transmorph_ixi_mapping.csv, 576 rows): take our template-space U-Net input
(work/pipeline_v2/<IXI>/template_t1.nii.gz, same grid as the TransMorph pkl images), downsample 2x and score it
against all 576 TransMorph images (work/mapping/pkl_ds2_stack.npy, built by scripts/verify_mapping.py):
  ncc       plain NCC;
  ncc_resid NCC after subtracting the population mean image (subject-specific anatomy).
Confirmed when the mapped subject ranks first on ncc_resid. No re-registration (the earlier
refine_mapping_verification.py step is only needed for the few unconfirmed cases). CPU only, a few minutes.

--refine: for every subject that is not rank 1, candidate re-registration as in
scripts/refine_mapping_verification.py (mapped k + top-5 by each score; affine ANTs on the 2 mm grid; post-registration
NCC decides). Resumable; ~20 s per subject on CPU.

Writes work/mapping/verification_v2_all.csv, work/checks/mapping_v2_all.json and (--refine)
work/mapping/verification_v2_refined.csv.
Usage: python scripts/verify_mapping_all.py [--refine [--limit N]]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosphere.provenance import provenance  # noqa: E402

MAPDIR, PIPE = ROOT / "work" / "mapping", ROOT / "work" / "pipeline_v2"


def ds2(a):
    a = a[: a.shape[0] // 2 * 2, : a.shape[1] // 2 * 2, : a.shape[2] // 2 * 2]
    return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2, a.shape[2] // 2, 2).mean((1, 3, 5))


def _unit(m):
    m = m - m.mean(-1, keepdims=True)
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def _template_ds2(sid):
    return ds2(np.asarray(nib.load(str(PIPE / sid / "template_t1.nii.gz")).dataobj, dtype=np.float32) / 255.0)


def refine(limit=None):
    sys.path.insert(0, str(ROOT / "scripts"))
    from refine_mapping_verification import SHAPE, post_reg_ncc
    stack = np.load(MAPDIR / "pkl_ds2_stack.npy", mmap_mode="r")
    ks = json.loads((MAPDIR / "pkl_ds2_keys.json").read_text())
    todo = [r for r in csv.DictReader(open(MAPDIR / "verification_v2_all.csv")) if r["rank_resid"] != "1"]
    out_f = MAPDIR / "verification_v2_refined.csv"
    done = {r["ixi_id"] for r in csv.DictReader(open(out_f))} if out_f.exists() else set()
    todo = [r for r in todo if r["ixi_id"] not in done][:limit]
    flat = np.asarray(stack, dtype=np.float32).reshape(len(ks), -1)
    mean_img = flat.mean(0)
    mats = {"ncc": _unit(flat), "resid": _unit(flat - mean_img)}
    new = not out_f.exists()
    print(f"{len(todo)} subjects to re-register ({len(done)} done)", flush=True)
    with open(out_f, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ixi_id", "k", "split", "n_candidates", "best_k", "ncc_mapped", "ncc_best_other",
                                           "margin", "confirmed", "seconds"])
        if new:
            w.writeheader()
        for i, r in enumerate(todo):
            t0 = time.perf_counter()
            sid, mk = r["ixi_id"], int(r["k"])
            v = _template_ds2(sid).astype(np.float32)
            vf = v.ravel()
            cand = {mk}
            for tag, vec in (("ncc", vf), ("resid", vf - mean_img)):
                cand |= {ks[j] for j in np.argsort(-(mats[tag] @ _unit(vec)))[:5]}
            scores = {k: post_reg_ncc(np.asarray(stack[ks.index(k)], np.float32).reshape(SHAPE), v.reshape(SHAPE)) for k in cand}
            best = max(scores, key=scores.get)
            other = max(s for k, s in scores.items() if k != mk)
            row = {"ixi_id": sid, "k": mk, "split": r["split"], "n_candidates": len(cand), "best_k": best,
                   "ncc_mapped": round(scores[mk], 4), "ncc_best_other": round(other, 4), "margin": round(scores[mk] - other, 4),
                   "confirmed": best == mk, "seconds": round(time.perf_counter() - t0, 1)}
            w.writerow(row); fh.flush()
            print(f"[{i + 1}/{len(todo)}]", row, flush=True)
    print("done", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--refine", action="store_true"); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.refine:
        return refine(a.limit)
    stack = np.load(MAPDIR / "pkl_ds2_stack.npy", mmap_mode="r")
    ks = json.loads((MAPDIR / "pkl_ds2_keys.json").read_text())
    flat = np.asarray(stack, dtype=np.float32).reshape(len(ks), -1)
    mean_img = flat.mean(0)
    mats = {"ncc": _unit(flat), "resid": _unit(flat - mean_img)}
    rows = list(csv.DictReader(open(ROOT / "data" / "transmorph_ixi_mapping.csv")))
    out, missing = [], []
    for r in rows:
        sid, k = r["ixi_id"], int(r["k"])
        f = PIPE / sid / "template_t1.nii.gz"
        if not f.exists():
            missing.append(sid); continue
        v = _template_ds2(sid).ravel()
        mi = ks.index(k)
        row = {"ixi_id": sid, "k": k, "split": r["split"], "previous_confidence": r["confidence"]}
        for tag, vec in (("ncc", v), ("resid", v - mean_img)):
            sc = mats[tag] @ _unit(vec)
            others = np.delete(sc, mi)
            row[f"rank_{tag}"] = int((sc > sc[mi]).sum()) + 1
            row[f"{tag}_mapped"] = round(float(sc[mi]), 4)
            row[f"{tag}_best_other"] = round(float(others.max()), 4)
            row[f"best_k_{tag}"] = ks[int(np.argmax(sc))]
        out.append(row)
    with open(MAPDIR / "verification_v2_all.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    margins = np.array([x["resid_mapped"] - x["resid_best_other"] for x in out])
    unconf = [x for x in out if x["rank_resid"] != 1]
    summ = {"n_mapped": len(rows), "n_scored": len(out), "missing_pipeline_output": missing,
            "rank1_resid": int(sum(x["rank_resid"] == 1 for x in out)), "rank1_ncc": int(sum(x["rank_ncc"] == 1 for x in out)),
            "margin_resid": {"median": float(np.median(margins)), "min": float(margins.min())},
            "not_rank1_resid": [{k: x[k] for k in ("ixi_id", "k", "split", "previous_confidence", "rank_resid", "resid_mapped",
                                                    "resid_best_other", "best_k_resid", "rank_ncc")} for x in unconf],
            "provenance": provenance()}
    d = ROOT / "work" / "checks"; d.mkdir(parents=True, exist_ok=True)
    (d / "mapping_v2_all.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "provenance"}, indent=1))


if __name__ == "__main__":
    main()
