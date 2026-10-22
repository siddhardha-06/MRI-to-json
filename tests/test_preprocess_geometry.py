"""Phase 2 geometry: orientation-preserving ANTs conversion, atlas axes, round-trip warp, Jacobian volumes."""
import ants
import nibabel as nib
import numpy as np
import pytest

from neurosphere.atlas import atlas_affine
from neurosphere.metrics import dice
from neurosphere.preprocess import ants_to_affine, nib_to_ants
from neurosphere.segment import affine_det, side_check
from tests.conftest import IXI002, needs_ixi


def test_atlas_affine_is_1mm_lia_centred():
    a = atlas_affine()
    assert "".join(nib.aff2axcodes(a)) == "LIA"
    assert np.allclose(np.abs(np.linalg.det(a[:3, :3])), 1.0)
    assert np.allclose(nib.affines.apply_affine(a, (np.array((160, 192, 224)) - 1) / 2), 0)


def test_nib_to_ants_roundtrip_oblique():
    t = np.radians(7.0)
    a = np.eye(4)
    a[:3, :3] = np.array([[0, 0, -1.2], [0.9375 * np.cos(t), -0.9375 * np.sin(t), 0], [0.9375 * np.sin(t), 0.9375 * np.cos(t), 0]])
    a[:3, 3] = [80.0, -120.0, -95.0]
    img = nib_to_ants(np.zeros((10, 12, 14), np.float32), a)
    assert np.allclose(ants_to_affine(img), a, atol=1e-5)


@needs_ixi
def test_nib_to_ants_matches_itk_reader():
    can = nib.as_closest_canonical(nib.load(str(IXI002)))
    ours = nib_to_ants(np.zeros(can.shape, np.float32), can.affine)
    itk = ants.image_read(str(IXI002))
    raw = nib.load(str(IXI002))
    for vox in [(10, 20, 30), (200, 100, 140), (0, 0, 0)]:
        world_ras = nib.affines.apply_affine(raw.affine, vox)
        can_vox = np.rint(nib.affines.apply_affine(np.linalg.inv(can.affine), world_ras)).astype(int)
        assert np.allclose(ants.transform_index_to_physical_point(itk, vox),
                           ants.transform_index_to_physical_point(ours, tuple(int(v) for v in can_vox)), atol=1e-3)


def _ellipsoid(shape, center, radii, affine):
    idx = np.indices(shape).reshape(3, -1).T
    w = nib.affines.apply_affine(affine, idx)
    return ((((w - center) / radii) ** 2).sum(1) <= 1).reshape(shape)


def test_roundtrip_warp_and_jacobian_volume(tmp_path):
    """native phantom -> atlas grid (forward) -> native (inverse): Dice and volume preserved;
    atlas count x |det| equals the native volume (the 'jacobian' volume in segment.py)."""
    native_aff = np.diag([1.2, 0.9375, 0.9375, 1.0]); native_aff[:3, 3] = [-90, -120, -120]
    native_shape = (150, 256, 256)
    lab = _ellipsoid(native_shape, np.array([-20.0, 5.0, 10.0]), np.array([12.0, 18.0, 9.0]), native_aff).astype(np.float32)
    lab += 2 * _ellipsoid(native_shape, np.array([25.0, -10.0, -5.0]), np.array([8.0, 8.0, 14.0]), native_aff)
    native = nib_to_ants(lab, native_aff)
    atlas = nib_to_ants(np.zeros((160, 192, 224), np.float32), atlas_affine())
    # an affine "atlas -> native" map: rotation + anisotropic scaling + translation (ITK, LPS)
    rot = np.array([[0.99, -0.10, 0.02], [0.10, 0.99, 0.03], [-0.02, -0.03, 1.0]]) @ np.diag([1.08, 0.95, 1.02])
    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3, matrix=rot, translation=[3.0, -4.0, 2.0])
    f = str(tmp_path / "fwd.mat")
    ants.write_transform(tx, f)
    in_atlas = ants.apply_transforms(fixed=atlas, moving=native, transformlist=[f], interpolator="genericLabel")
    back = ants.apply_transforms(fixed=native, moving=in_atlas, transformlist=[f], whichtoinvert=[True], interpolator="genericLabel")
    la, lb = np.rint(in_atlas.numpy()), np.rint(back.numpy())
    det = affine_det([f])
    assert det == pytest.approx(abs(np.linalg.det(rot)))
    vv = abs(np.linalg.det(native_aff[:3, :3])) / 1000
    for l in (1, 2):
        v_true = (lab == l).sum() * vv
        assert (la == l).sum() * det / 1000 == pytest.approx(v_true, rel=0.03)    # jacobian volume
        assert (lb == l).sum() * vv == pytest.approx(v_true, rel=0.03)            # native round-trip volume
        assert dice(lb == l, lab == l) > 0.93


def test_side_check_detects_swap():
    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    lab = np.zeros((40, 10, 10), np.int16)
    lab[5:15] = 17; lab[25:35] = 53          # RAS: low x index = left -> 17 on the left: correct
    assert side_check(lab, aff)["all_correct"]
    assert side_check(lab[::-1].copy(), aff)["n_swapped"] == 1
