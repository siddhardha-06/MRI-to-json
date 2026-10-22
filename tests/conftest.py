import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IXI_DIR = ROOT / "ixi-t1"
IXI002 = IXI_DIR / "IXI002-Guys-0828-T1.nii" / "IXI002-Guys-0828-MPRAGESEN_-s256_-0301-00003-000001-01.nii"
needs_ixi = pytest.mark.skipif(not IXI002.exists(), reason="IXI raw data not present")


@pytest.fixture
def ras_volume():
    """A small RAS volume with a unique value at every voxel, anisotropic voxels and an offset origin."""
    data = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
    affine = np.diag([1.2, 0.9375, 0.9375, 1.0])
    affine[:3, 3] = [-30.0, 12.5, 4.0]
    return data, affine


def save_nifti(tmp_path, data, affine, name="t1.nii.gz"):
    p = tmp_path / name
    nib.save(nib.Nifti1Image(data, affine), str(p))
    return p
