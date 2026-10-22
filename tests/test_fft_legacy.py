"""K2: documents the defect of the frozen legacy FFT features (they stay as shipped, decision Q1).

If this test starts failing, the legacy extractor was changed and the legacy v1 baseline is no
longer the shipped one. v2 has no slice FFT features.
"""
import importlib.util

import numpy as np

from tests.conftest import ROOT


def _legacy():
    spec = importlib.util.spec_from_file_location("legacy_extract_features", ROOT / "legacy" / "extract_features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _radial(arr):
    h, w = arr.shape
    mag = np.abs(np.fft.fftshift(np.fft.fft2(arr)))
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - h // 2, xx - w // 2) / (h / 2)
    return mag[r < 0.125].mean(), mag[r > 0.5].mean()


def test_legacy_fft_low_high_are_inverted():
    f = _legacy().extract_frequency_features
    yy, xx = np.mgrid[:256, :256]
    smooth = np.exp(-(((yy - 128) ** 2 + (xx - 128) ** 2) / (2 * 40.0 ** 2)))
    low, high = _radial(smooth)
    assert low / high > 1e3                               # truly low-frequency image
    assert f(smooth)["frequency_energy_ratio"] < 1e-3     # legacy "low/high" says the opposite
    noise = np.random.default_rng(0).random((256, 256))
    assert 0.5 < f(noise)["frequency_energy_ratio"] < 2.0  # flat spectrum: ratio near 1


def test_legacy_high_region_contains_dc():
    h = w = 256
    dc = (h // 2, w // 2)
    assert dc[0] >= h // 2 and dc[1] >= w // 2            # inside magnitude[H//2:, W//2:] ("high")
    assert not (dc[0] < h // 4 and dc[1] < w // 4)        # not inside magnitude[:H//4, :W//4] ("low")
