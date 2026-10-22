r"""Convert IXI_data/atlas.pkl to NIfTI with a correct 1 mm LIA affine, and check the orientation.

Outputs: data/atlas/atlas_T1.nii.gz, data/atlas/atlas_labels.nii.gz (data, gitignored; regenerate
with this script), data/atlas/atlas_orientation.json and docs/figures/atlas_orientation.png.
The orientation check re-derives the axis codes from label centroids (settles K10):
  L/R hippocampus (17/53) -> left-right axis, thalamus (10/49) vs brain-stem (16) -> S-I,
  caudate (11/50) vs 4th ventricle (15) -> A-P.
"""
import json
import pickle
import sys
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosphere.atlas import atlas_affine, default_atlas_paths  # noqa: E402
from neurosphere.provenance import provenance, sha256_file  # noqa: E402


def centroid_world(lab, aff, ids):
    idx = np.argwhere(np.isin(lab, ids))
    return nib.affines.apply_affine(aff, idx.mean(0))


def main():
    src = ROOT / "IXI_data" / "atlas.pkl"
    img, lab = pickle.load(open(src, "rb"))
    lab = np.rint(lab).astype(np.int16)
    aff = atlas_affine(img.shape)
    t1_path, lab_path = default_atlas_paths(ROOT)
    t1_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(img.astype(np.float32), aff), str(t1_path))
    nib.save(nib.Nifti1Image(lab, aff), str(lab_path))

    # world-space checks (RAS+): Left has smaller x, Superior larger z, Anterior larger y
    lh, rh = centroid_world(lab, aff, [17]), centroid_world(lab, aff, [53])
    thal, bs = centroid_world(lab, aff, [10, 49]), centroid_world(lab, aff, [16])
    caud, v4 = centroid_world(lab, aff, [11, 50]), centroid_world(lab, aff, [15])
    checks = {
        "left_hippocampus_is_left_of_right (x_L < x_R)": bool(lh[0] < rh[0]),
        "thalamus_is_superior_to_brainstem (z_thal > z_bs)": bool(thal[2] > bs[2]),
        "caudate_is_anterior_to_4th_ventricle (y_caud > y_v4)": bool(caud[1] > v4[1]),
    }
    res = {"axcodes": "".join(nib.aff2axcodes(aff)), "shape": list(img.shape), "voxel_mm": 1.0, "checks": checks,
           "centroids_world_mm": {"L_hippocampus": lh.round(1).tolist(), "R_hippocampus": rh.round(1).tolist(),
                                  "thalamus": thal.round(1).tolist(), "brain_stem": bs.round(1).tolist(),
                                  "caudate": caud.round(1).tolist(), "4th_ventricle": v4.round(1).tolist()},
           "source": {"file": "IXI_data/atlas.pkl", "sha256": sha256_file(src)}, "provenance": provenance()}
    (t1_path.parent / "atlas_orientation.json").write_text(json.dumps(res, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
    c = [s // 2 for s in img.shape]
    # display in radiological-neutral form: sagittal (axis 0 fixed), coronal (axis 2 fixed), axial (axis 1 fixed)
    ax[0].imshow(img[c[0]].T if False else img[c[0], :, :], cmap="gray"); ax[0].set_title("axis 0 = L-R (sagittal slice)\nrows: axis 1 (S at top), cols: axis 2 (A right)")
    ax[1].imshow(img[:, :, c[2]].T, cmap="gray"); ax[1].set_title("axis 2 = A-P (coronal slice)\nrows: axis 1 (S top), cols: axis 0 (R left)")
    ax[2].imshow(img[:, c[1], :].T[::-1], cmap="gray"); ax[2].set_title("axis 1 = S-I (axial slice)\nrows: axis 2 (A top), cols: axis 0 (R left)")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"TransMorph atlas: axcodes {res['axcodes']}; checks: {all(checks.values())}")
    out = ROOT / "docs" / "figures" / "atlas_orientation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=90)
    print(json.dumps({k: v for k, v in res.items() if k != "provenance"}, indent=2))
    if not all(checks.values()):
        raise SystemExit("orientation check failed")


if __name__ == "__main__":
    main()
