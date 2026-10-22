r"""TransMorph <-> IXI mapping, step 2: image verification (Phase 2).

For each requested IXI subject: run the pipeline (cached in work/pipeline/), take its atlas-space
image and compute NCC against ALL 576 TransMorph images at 2x downsampling. The mapping is
confirmed when the mapped subject_k ranks first; the margin (best - second best NCC) is reported.

Writes work/mapping/verification.csv (appends; resumable) and, with --finalise, the mapping file
data/transmorph_ixi_mapping.csv (committed; no image data) with method and confidence columns.

Usage:
  python scripts/verify_mapping.py --sample          # all 22 order-inferred + IXI002..IXI042 + 20 random
  python scripts/verify_mapping.py --subjects IXI002 IXI012
  python scripts/verify_mapping.py --finalise
"""
import argparse
import csv
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosphere.pipeline import run_subject  # noqa: E402
from neurosphere.provenance import provenance  # noqa: E402
from neurosphere.segment import Segmenter  # noqa: E402
from neurosphere.splits import discover_raw_ixi  # noqa: E402

MAPDIR = ROOT / "work" / "mapping"
STACK = MAPDIR / "pkl_ds2_stack.npy"
ORDER_INFERRED = {40: "IXI054", 67: "IXI081", 73: "IXI088", 102: "IXI117", 127: "IXI145", 144: "IXI166", 200: "IXI228",
                  254: "IXI290", 295: "IXI333", 299: "IXI337", 301: "IXI340", 305: "IXI345", 306: "IXI347", 341: "IXI387",
                  407: "IXI456", 408: "IXI457", 521: "IXI589", 558: "IXI633", 562: "IXI637", 563: "IXI638", 568: "IXI643",
                  575: "IXI661"}


def ds2(a):
    a = a[: a.shape[0] // 2 * 2, : a.shape[1] // 2 * 2, : a.shape[2] // 2 * 2]
    return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2, a.shape[2] // 2, 2).mean((1, 3, 5))


def pkl_stack():
    """(576, 80, 96, 112) float16 stack of downsampled TransMorph images, built once (~1 GB in work/)."""
    files = {int(p.stem.split("_")[1]): p for p in (ROOT / "IXI_data").glob("*/subject_*.pkl")}
    ks = sorted(files)
    if not STACK.exists():
        arr = np.lib.format.open_memmap(STACK, mode="w+", dtype=np.float16, shape=(len(ks), 80, 96, 112))
        for i, k in enumerate(ks):
            arr[i] = ds2(pickle.load(open(files[k], "rb"))[0])
        arr.flush(); del arr
        (MAPDIR / "pkl_ds2_keys.json").write_text(json.dumps(ks))
    return np.load(STACK, mmap_mode="r"), json.loads((MAPDIR / "pkl_ds2_keys.json").read_text())


def mapping_table():
    rows = {}
    for r in csv.DictReader(open(MAPDIR / "demographic_match.csv")):
        k = int(r["k"])
        if r["ixi_id"]:
            rows[k] = {"k": k, "ixi_id": r["ixi_id"], "split": r["split"], "method": "demographics_exact"}
        elif k in ORDER_INFERRED:
            rows[k] = {"k": k, "ixi_id": ORDER_INFERRED[k], "split": r["split"], "method": "order_constraint"}
    return rows


def _norm(m):
    m = m - m.mean(1, keepdims=True)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def verify(subjects, seg):
    """Two similarity scores against all 576 TransMorph images (2x downsampled):
    ncc       : plain NCC of the images (dominated by shared brain shape);
    ncc_resid : NCC after subtracting the population mean image from both (subject-specific anatomy).
    The mapping is confirmed if the mapped subject ranks first on ncc_resid; z = separation in SDs."""
    stack, ks = pkl_stack()
    flat = stack.reshape(len(ks), -1).astype(np.float32)
    mean_img = flat.mean(0)
    flat_n, resid_n = _norm(flat), _norm(flat - mean_img)
    by_id = {v["ixi_id"]: v for v in mapping_table().values()}
    raw = {r["subject_id"]: ROOT / "ixi-t1" / r["raw_file"] for r in discover_raw_ixi(ROOT / "ixi-t1")}
    out_csv = MAPDIR / "verification.csv"
    (MAPDIR / "ds2").mkdir(exist_ok=True)
    done = {r["ixi_id"] for r in csv.DictReader(open(out_csv))} if out_csv.exists() else set()
    new = not out_csv.exists()
    fields = ["ixi_id", "mapped_k", "method", "best_k_resid", "rank_resid", "z_resid", "margin_resid",
              "rank_ncc", "z_ncc", "ncc_mapped", "seconds"]
    with open(out_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        for sid in subjects:
            if sid in done:
                continue
            t0 = time.perf_counter()
            r = run_subject(raw[sid], sid, seg, keep_atlas_image=True)
            v = ds2(r["_atlas_image"]).astype(np.float32).ravel()
            np.save(MAPDIR / "ds2" / f"{sid}.npy", v.astype(np.float16))   # 1.7 MB; lets scores be recomputed
            m = by_id.get(sid)
            mk = m["k"] if m else -1
            mi = ks.index(mk) if mk in ks else None
            row = {"ixi_id": sid, "mapped_k": mk, "method": m["method"] if m else "unmapped", "seconds": round(time.perf_counter() - t0, 1)}
            for tag, mat, vec in (("ncc", flat_n, v), ("resid", resid_n, v - mean_img)):
                vv = vec - vec.mean(); vv /= np.linalg.norm(vv)
                sc = mat @ vv
                order = np.argsort(-sc)
                if mi is None:
                    continue
                others = np.delete(sc, mi)
                row[f"rank_{tag}"] = int(np.where(order == mi)[0][0]) + 1
                row[f"z_{tag}"] = round(float((sc[mi] - others.mean()) / others.std()), 2)
                if tag == "resid":
                    row["best_k_resid"] = ks[order[0]]
                    row["margin_resid"] = round(float(sc[mi] - others.max()), 4)
                else:
                    row["ncc_mapped"] = round(float(sc[mi]), 4)
            w.writerow(row); fh.flush()
            print(row, flush=True)


def finalise():
    """Confidence per subject_k: 'verified_by_image' when candidate re-registration
    (scripts/refine_mapping_verification.py) ranks the mapped subject first; otherwise the
    metadata evidence ('demographics_unique' or 'order_constraint_unique')."""
    ref_f = MAPDIR / "verification_refined.csv"
    ref = {r["ixi_id"]: r for r in csv.DictReader(open(ref_f))} if ref_f.exists() else {}
    # A mapping is CONTRADICTED only if another candidate is at least as similar as the weakest confirmed
    # true match (data-driven); otherwise an unconfirmed case is 'inconclusive' (e.g. a defective TransMorph image).
    conf_ncc = [float(v["ncc_mapped"]) for v in ref.values() if v["confirmed"] == "True"]
    thr = min(conf_ncc) if conf_ncc else 1.0
    rows = []
    for k, m in sorted(mapping_table().items()):
        v = ref.get(m["ixi_id"])
        conf = ("verified_by_image" if v and v["confirmed"] == "True" else
                "CONTRADICTED_by_image" if v and float(v["ncc_best_other"]) >= thr else
                "image_check_inconclusive" if v else
                "demographics_unique" if m["method"] == "demographics_exact" else "order_constraint_unique")
        rows.append({"transmorph_file": f"subject_{k}.pkl", "k": k, "split": m["split"], "ixi_id": m["ixi_id"], "method": m["method"],
                     "confidence": conf, "ncc_after_reg_mapped": v["ncc_mapped"] if v else "",
                     "ncc_after_reg_best_other": v["ncc_best_other"] if v else "", "margin": v["margin"] if v else ""})
    out = ROOT / "data" / "transmorph_ixi_mapping.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    margins = [float(r["margin"]) for r in rows if r["margin"] != ""]
    meta = {"n": len(rows), "by_confidence": {c: sum(r["confidence"] == c for r in rows) for c in sorted({r["confidence"] for r in rows})},
            "by_method": {c: sum(r["method"] == c for r in rows) for c in sorted({r["method"] for r in rows})},
            "image_verified_margin_min": min(margins) if margins else None,
            "image_verified_margin_median": float(np.median(margins)) if margins else None,
            "contradiction_threshold_ncc": thr,
            "inconclusive_notes": {"IXI430": "TransMorph subject_381 (Train) has a failed FreeSurfer skull strip (face/neck retained); "
                                             "no candidate matches well (mapped 0.62, best other 0.76 < threshold); demographics match exactly"},
            "not_in_transmorph": ["IXI423", "IXI426", "IXI427", "IXI547", "IXI553"],
            "method_notes": "step 1 scripts/map_transmorph_ixi.py (demographics vs IXI.xls); order constraint for the rest; "
                            "step 3 scripts/refine_mapping_verification.py (candidate re-registration)",
            "provenance": provenance()}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "provenance"}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--finalise", action="store_true")
    a = ap.parse_args()
    if a.finalise:
        return finalise()
    if a.sample:
        tab = mapping_table()
        demo = sorted(v["ixi_id"] for v in tab.values() if v["method"] == "demographics_exact")
        fs = sorted(p.name[:6] for p in (ROOT / "results/legacy/validation_results/fastsurfer_output").iterdir())
        subs = sorted(set(ORDER_INFERRED.values()) | set(fs) | set(random.Random(0).sample(demo, 20)))
    elif a.all:
        subs = sorted(r["subject_id"] for r in discover_raw_ixi(ROOT / "ixi-t1"))
    else:
        subs = a.subjects
    print(f"{len(subs)} subjects", flush=True)
    verify(subs, Segmenter())


if __name__ == "__main__":
    main()
