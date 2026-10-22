r"""Smoke test of preprocess + segment on a few raw IXI scans (Phase 2). Writes to work/smoke/pipeline/.

For each subject: per-stage timings; NCC of our atlas-space image with its mapped TransMorph pkl
and with 20 random other pkl images; Dice of our atlas-space labels vs the pkl FreeSurfer labels;
jacobian vs native volume agreement. Usage: python scripts/smoke_pipeline.py IXI002 IXI012 ...
"""
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
from neurosphere.metrics import dice  # noqa: E402
from neurosphere.preprocess import preprocess  # noqa: E402
from neurosphere.segment import Segmenter, segment  # noqa: E402
from neurosphere.splits import discover_raw_ixi  # noqa: E402


def ncc(a, b, m):
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def main(ids):
    raw = {r["subject_id"]: ROOT / "ixi-t1" / r["raw_file"] for r in discover_raw_ixi(ROOT / "ixi-t1")}
    mapping = {r["ixi_id"]: r for r in csv.DictReader(open(ROOT / "work" / "mapping" / "demographic_match.csv")) if r["ixi_id"]}
    pkls = {int(p.stem.split("_")[1]): p for p in (ROOT / "IXI_data").glob("*/subject_*.pkl")}
    seg = Segmenter()
    out = []
    for sid in ids:
        t0 = time.perf_counter()
        pre = preprocess(raw[sid], ROOT / "work" / "smoke" / "pipeline" / sid)
        s = segment(pre, seg)
        total = time.perf_counter() - t0
        rec = {"subject": sid, "timings_s": {**pre.timings_s, **s.timings_s, "total": total}}
        if sid in mapping:
            k = int(mapping[sid]["k"])
            img, lab = pickle.load(open(pkls[k], "rb"))
            m = (img > 0) | (pre.atlas_image > 0)
            others = random.Random(0).sample([j for j in pkls if j != k], 20)
            nc_other = [ncc(pre.atlas_image, pickle.load(open(pkls[j], "rb"))[0], m) for j in others]
            fg = [l for l in np.unique(lab) if l in set(seg.labels.tolist()) and l != 0]
            rec.update(k=k, split=mapping[sid]["split"], ncc_mapped=ncc(pre.atlas_image, img, m),
                       ncc_others_max=max(nc_other), ncc_others_mean=float(np.mean(nc_other)),
                       dice_vs_pkl_fs_mean=float(np.nanmean([dice(s.labels_atlas == l, lab == l) for l in fg])),
                       wm_median_ours=float(np.median(pre.atlas_image[np.isin(s.labels_atlas, (2, 41))]) * 255),
                       wm_median_pkl=float(np.median(img[np.isin(lab, (2, 41))]) * 255))
        rec["vol_jacobian_vs_native_max_rel_diff"] = max(abs(v["jacobian"] - v["native"]) / max(v["jacobian"], 1e-9)
                                                         for v in s.volumes_ml.values() if v["jacobian"] > 1)
        rec["hippocampus_ml"] = {k: round(s.volumes_ml[17][k] + s.volumes_ml[53][k], 3) for k in ("jacobian", "native")}
        out.append(rec)
        print(json.dumps(rec, indent=1), flush=True)
    (ROOT / "work" / "smoke" / "pipeline" / "summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or ["IXI002"])
