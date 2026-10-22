r"""Re-segment the cached template-space inputs with another checkpoint (Phase 3: evaluate a fine-tuned model).

No preprocessing is re-run: for each subject, work/pipeline_v2/<IXI>/template_t1.nii.gz (uint8 = U-Net input x 255)
is segmented, the labels go to the native grid with the cached template -> native affine (inverted, nearest label),
and volumes are computed as in neurosphere/segment.py (primary 'jacobian' = template count x |det A|; 'native').
The output result.json is the v2 one with volumes_ml, side_check_native and checkpoint replaced, so
experiments/e2fs_transferred_fs.py --labels-dir can read it. Note: the cached input is quantised to uint8, whereas the
pipeline segmented the float image; re-segmenting with the v2 checkpoint itself shows the size of that effect.

Usage: python scripts/resegment_cached.py --ckpt unet3d_ft_best.pth --split Val --out work/pipeline_ft [--limit 3]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import ants
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosphere.preprocess import atlas_t1, nib_to_ants  # noqa: E402
from neurosphere.provenance import provenance  # noqa: E402
from neurosphere.segment import Segmenter, affine_det, side_check  # noqa: E402

PIPE = ROOT / "work" / "pipeline_v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint file name in checkpoints/ (or a path)")
    ap.add_argument("--split", choices=["Train", "Val", "Test", "all"], required=True, help="TransMorph split, or all 581 subjects")
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    ck = Path(a.ckpt) if Path(a.ckpt).exists() else ROOT / "checkpoints" / a.ckpt
    seg = Segmenter(ck)
    if a.split == "all":
        subs = sorted(p.parent.name for p in PIPE.glob("IXI*/result.json"))
    else:
        subs = sorted(r["ixi_id"] for r in csv.DictReader(open(ROOT / "data" / "transmorph_ixi_mapping.csv")) if r["split"] == a.split)
    subs = subs[: a.limit]
    out = Path(a.out); out = out if out.is_absolute() else ROOT / out
    tmpl = atlas_t1()
    for i, sid in enumerate(subs):
        t0 = time.perf_counter()
        res = json.loads((PIPE / sid / "result.json").read_text())
        x = np.asarray(nib.load(str(PIPE / sid / "template_t1.nii.gz")).dataobj, np.float32) / 255.0
        lab = seg(x)
        ref_img = nib.load(str(PIPE / sid / "labels_native.nii.gz"))
        mat = str(PIPE / sid / "native2atlas_0GenericAffine.mat")
        nat = ants.apply_transforms(fixed=nib_to_ants(np.zeros(ref_img.shape, np.float32), ref_img.affine),
                                    moving=tmpl.new_image_like(lab.astype(np.float32)), transformlist=[mat],
                                    whichtoinvert=[True], interpolator="genericLabel")
        lab_nat = np.rint(nat.numpy()).astype(np.int16)
        det = affine_det([mat])
        vv = float(abs(np.linalg.det(ref_img.affine[:3, :3]))) / 1000.0
        ia, ca = np.unique(lab, return_counts=True); inn, cn = np.unique(lab_nat, return_counts=True)
        ca, cn = dict(zip(ia.tolist(), ca.tolist())), dict(zip(inn.tolist(), cn.tolist()))
        res["volumes_ml"] = {str(int(l)): {"jacobian": ca.get(int(l), 0) * det / 1000.0, "native": cn.get(int(l), 0) * vv}
                             for l in seg.labels if l != 0}
        res["side_check_native"] = side_check(lab_nat, ref_img.affine)
        res["checkpoint"] = seg.info
        res["resegmented"] = {"from": "work/pipeline_v2 (cached uint8 template-space input)", "seconds": round(time.perf_counter() - t0, 1),
                              "provenance": provenance()}
        (out / sid).mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(lab_nat, ref_img.affine), str(out / sid / "labels_native.nii.gz"))
        (out / sid / "result.json").write_text(json.dumps(res, indent=2))
        print(f"[{i + 1}/{len(subs)}] {sid} {res['resegmented']['seconds']}s side_ok={res['side_check_native']['all_correct']}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
