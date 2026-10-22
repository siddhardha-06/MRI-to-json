r"""Registration template = voxel-wise mean of the 403 TransMorph *training* images (Phase 2).

Why: registering raw scans to the CycleMorph atlas.pkl left them ~3.6 voxels off (left-right) from
where the training images sit, and the current U-Net then swapped every left/right label
(it uses field-of-view position as a side cue). The mean training image
defines the frame the network was trained in. Only training subjects are used (no test data).

Outputs (gitignored data): data/atlas/train_mean_T1.nii.gz (same LIA 1 mm affine as the atlas),
data/atlas/train_mean_T1.json with centroids and provenance.
"""
import json
import pickle
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosphere.atlas import atlas_affine  # noqa: E402
from neurosphere.provenance import provenance  # noqa: E402

files = sorted((ROOT / "IXI_data" / "Train").glob("*.pkl"))
acc = None
cents = []
for f in files:
    img, lab = pickle.load(open(f, "rb"))
    acc = img.astype(np.float64) if acc is None else acc + img
    cents.append(np.argwhere(img > 0).mean(0))
mean = (acc / len(files)).astype(np.float32)
atlas_img, _ = pickle.load(open(ROOT / "IXI_data" / "atlas.pkl", "rb"))
out = ROOT / "data" / "atlas" / "train_mean_T1.nii.gz"
nib.save(nib.Nifti1Image(mean, atlas_affine(mean.shape)), str(out))
c = np.array(cents)
info = {"n_train_images": len(files),
        "brain_centroid_voxel_train_mean": c.mean(0).round(2).tolist(), "brain_centroid_voxel_train_sd": c.std(0).round(2).tolist(),
        "brain_centroid_voxel_cyclemorph_atlas": np.argwhere(atlas_img > 0).mean(0).round(2).tolist(),
        "provenance": provenance(data_subset="IXI_data/Train (403)")}
out.with_suffix("").with_suffix(".json").write_text(json.dumps(info, indent=2))
print(json.dumps({k: v for k, v in info.items() if k != "provenance"}, indent=2))
