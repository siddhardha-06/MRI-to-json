"""K1: every load path returns RAS+ arrays, whatever the on-disk orientation."""
import itertools

import nibabel as nib
import numpy as np
import pytest
from nibabel.orientations import apply_orientation, inv_ornt_aff

from neurosphere.io import is_derived, load_t1, parse_ixi_name, voxel_volume_ml
from tests.conftest import IXI002, needs_ixi, save_nifti

ORIENTATIONS = [(perm, flips) for perm in itertools.permutations(range(3)) for flips in itertools.product((1, -1), repeat=3)]


@pytest.mark.parametrize("perm,flips", ORIENTATIONS, ids=[f"{p}{f}" for p, f in ORIENTATIONS])
def test_all_48_orientations_load_as_ras(tmp_path, ras_volume, perm, flips):
    ras, ras_aff = ras_volume
    ornt = np.array([[perm[i], flips[i]] for i in range(3)])
    disk = apply_orientation(ras, ornt)
    disk_aff = ras_aff @ inv_ornt_aff(ornt, ras.shape)
    img = load_t1(save_nifti(tmp_path, disk, disk_aff))
    assert img.meta["orientation"] == "RAS"
    assert img.meta["orientation_original"] == "".join(nib.aff2axcodes(disk_aff))
    np.testing.assert_array_equal(img.data, ras)
    np.testing.assert_allclose(img.affine, ras_aff, atol=1e-5)


def test_voxel_volume_is_orientation_invariant(ras_volume):
    _, aff = ras_volume
    ornt = np.array([[1, -1], [2, 1], [0, -1]])
    assert voxel_volume_ml(aff @ inv_ornt_aff(ornt, (5, 6, 7))) == pytest.approx(voxel_volume_ml(aff))


@needs_ixi
def test_ixi_header_is_psr_and_is_reoriented():
    img = load_t1(IXI002)
    assert img.meta["orientation_original"] == "PSR"       # third array axis is L->R on disk (AUDIT K1)
    assert img.meta["orientation"] == "RAS"
    assert img.meta["shape_original"] == [256, 256, 150]
    assert img.meta["shape"] == [150, 256, 256]            # L-R axis is now first
    assert img.meta["subject_id"] == "IXI002" and img.meta["site"] == "Guys"


@needs_ixi
def test_ixi_left_right_axis_is_first_after_loading():
    """Header-independent: mirroring about the true L-R axis gives the highest self-correlation."""
    v = load_t1(IXI002, compute_sha256=False).data
    v = v / np.percentile(v, 99)
    m = v > 0.1
    cc = []
    for ax in range(3):
        fv, fm = np.flip(v, ax), np.flip(m, ax)
        shift = int(round(np.argwhere(m).mean(0)[ax] - np.argwhere(fm).mean(0)[ax]))
        cc.append(np.corrcoef(v[m], np.roll(fv, shift, ax)[m])[0, 1])
    assert int(np.argmax(cc)) == 0


def test_ixi_name_parsing_and_derived_detection():
    assert parse_ixi_name("IXI035-IOP-0873-SAGFSPGR_-sIXI35_-0003-00001-000001-01.nii") == ("IXI035", "IOP")
    assert parse_ixi_name("BraTS-GLI-02072-104-t1c.nii") == (None, None)
    for suffix in ("_n4", "_n4_n4", "_seg", "_seg_seg"):
        assert is_derived(f"IXI002-Guys-0828-X{suffix}.nii")
    assert not is_derived("IXI002-Guys-0828-X.nii.gz")
