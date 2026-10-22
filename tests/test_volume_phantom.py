"""Volumes are voxel count x native voxel volume, independent of orientation and obliquity."""
import math

import numpy as np
import pytest
from nibabel.orientations import apply_orientation, inv_ornt_aff

from neurosphere.features import asymmetry_index_pct, volumes_from_labels
from neurosphere.io import load_t1
from tests.conftest import save_nifti

VOX = (1.2, 0.9375, 0.9375)  # IXI-like anisotropic voxels


def _box_phantom():
    lab = np.zeros((60, 80, 90), np.int16)
    lab[10:20, 10:30, 10:40] = 17       # 10*20*30 = 6000 voxels
    lab[30:35, 40:50, 50:56] = 53       # 5*10*6 = 300 voxels
    return lab, np.diag(list(VOX) + [1.0])


def test_box_volumes_native():
    lab, aff = _box_phantom()
    v = volumes_from_labels(lab, aff, [17, 53, 99])
    vv = np.prod(VOX) / 1000
    assert v[17] == pytest.approx(6000 * vv)
    assert v[53] == pytest.approx(300 * vv)
    assert v[99] == 0.0


def test_volumes_survive_reorientation_through_load(tmp_path):
    lab, aff = _box_phantom()
    ornt = np.array([[1, -1], [2, 1], [0, 1]])  # PSR-like on-disk layout
    disk = apply_orientation(lab, ornt).astype(np.float32)
    img = load_t1(save_nifti(tmp_path, disk, aff @ inv_ornt_aff(ornt, lab.shape)))
    v = volumes_from_labels(np.rint(img.data).astype(int), img.affine, [17, 53])
    assert v[17] == pytest.approx(6000 * np.prod(VOX) / 1000)
    assert v[53] == pytest.approx(300 * np.prod(VOX) / 1000)


def test_oblique_affine_keeps_voxel_volume():
    lab, aff = _box_phantom()
    t = math.radians(7.1)  # IXI headers are slightly oblique
    rot = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    obl = aff.copy(); obl[:3, :3] = rot @ aff[:3, :3]
    assert volumes_from_labels(lab, obl, [17])[17] == pytest.approx(volumes_from_labels(lab, aff, [17])[17])


def test_sphere_volume_on_anisotropic_grid():
    r_mm = 20.0
    shape = (60, 80, 80)
    c = np.array(shape) / 2
    idx = np.indices(shape).reshape(3, -1).T
    d = np.sqrt((((idx - c) * VOX) ** 2).sum(1)).reshape(shape)
    v = volumes_from_labels((d <= r_mm).astype(int), np.diag(list(VOX) + [1.0]), [1])[1]
    assert v == pytest.approx(4 / 3 * math.pi * r_mm ** 3 / 1000, rel=0.02)


def test_signed_asymmetry_index():
    assert asymmetry_index_pct(4.2, 3.8) == pytest.approx(10.0)
    assert asymmetry_index_pct(3.8, 4.2) == pytest.approx(-10.0)
    assert asymmetry_index_pct(0.0, 0.0) is None
