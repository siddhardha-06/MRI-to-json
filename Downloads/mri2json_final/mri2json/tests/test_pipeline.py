"""
tests/test_pipeline.py
----------------------
Unit tests for the mri2json pipeline.
Run with: pytest tests/
"""

import numpy as np
import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor import normalize_slice, slice_to_pil, pil_to_base64
from src.postprocessor import validate_json, enrich_with_metadata


# ─── Preprocessor tests ───────────────────────────────────────────────────────

def make_fake_slice(shape=(256, 256), with_brain=True):
    """Create a synthetic MRI-like 2D array."""
    data = np.zeros(shape, dtype=np.float32)
    if with_brain:
        # Simulate brain region with higher intensity
        cy, cx = shape[0] // 2, shape[1] // 2
        r = min(shape) // 3
        y, x = np.ogrid[:shape[0], :shape[1]]
        mask = (x - cx)**2 + (y - cy)**2 <= r**2
        data[mask] = np.random.uniform(100, 800, mask.sum())
    return data


def test_normalize_slice_range():
    sl = make_fake_slice()
    normalized = normalize_slice(sl)
    assert normalized.dtype == np.uint8
    assert normalized.min() >= 0
    assert normalized.max() <= 255


def test_normalize_empty_slice():
    """All-zero slice should not crash."""
    sl = np.zeros((128, 128), dtype=np.float32)
    normalized = normalize_slice(sl)
    assert normalized.shape == (128, 128)
    assert normalized.max() == 0


def test_slice_to_pil_size():
    sl = make_fake_slice()
    normalized = normalize_slice(sl)
    img = slice_to_pil(normalized, rotate_degrees=0, target_size=(512, 512))
    assert img.size == (512, 512)
    assert img.mode == "RGB"


def test_pil_to_base64_is_string():
    sl = make_fake_slice()
    normalized = normalize_slice(sl)
    img = slice_to_pil(normalized)
    b64 = pil_to_base64(img)
    assert isinstance(b64, str)
    assert len(b64) > 100  # Should be substantial base64


# ─── Postprocessor tests ──────────────────────────────────────────────────────

def make_valid_record():
    return {
        "modality": "T1-weighted",
        "plane": "axial",
        "image_quality": "good",
        "anatomy": {
            "visible_structures": ["cortex", "white matter"],
            "hemisphere_symmetry": "symmetric",
            "cortical_ribbon": "preserved",
            "sulcal_pattern": "normal depth",
        },
        "tissue_findings": {
            "gray_matter": "preserved",
            "white_matter": "normal",
            "ventricles": "normal size",
            "sulci_gyri": "normal pattern",
            "basal_ganglia": "not visible in this slice",
            "thalamus": "not visible in this slice",
            "cerebellum": "not visible in this slice",
            "brainstem": "not visible in this slice",
        },
        "pathology_flags": {
            "atrophy": False,
            "atrophy_severity": "none",
            "white_matter_lesions": False,
            "lesion_count_estimate": "none",
            "mass_effect": False,
            "midline_shift": False,
            "hydrocephalus": False,
            "infarct": False,
            "hemorrhage": False,
            "notes": "no significant findings",
        },
        "global_impression": "Normal brain MRI for age.",
        "confidence": 0.9,
    }


def test_validate_valid_record():
    record = make_valid_record()
    is_valid, issues = validate_json(record)
    assert is_valid, f"Should be valid but got issues: {issues}"
    assert len(issues) == 0


def test_validate_missing_key():
    record = make_valid_record()
    del record["anatomy"]
    is_valid, issues = validate_json(record)
    assert not is_valid
    assert any("anatomy" in issue for issue in issues)


def test_validate_error_record():
    record = {"error": "Parse failed", "confidence": 0.0}
    is_valid, issues = validate_json(record)
    assert not is_valid


def test_validate_confidence_range():
    record = make_valid_record()
    record["confidence"] = 1.5  # Out of range
    is_valid, issues = validate_json(record)
    assert not is_valid
    assert any("confidence" in issue for issue in issues)


def test_enrich_adds_provenance():
    record = make_valid_record()
    enriched = enrich_with_metadata(record, "OAS1_0001_MR1", 90)
    assert enriched["subject_id"] == "OAS1_0001_MR1"
    assert enriched["slice_index"] == 90
    assert "generated_at" in enriched
    assert enriched["oasis_metadata"] == {}


def test_enrich_with_metadata_df():
    import pandas as pd
    record = make_valid_record()
    df = pd.DataFrame([{
        "ID": "OAS1_0001_MR1",
        "Age": 74,
        "M/F": "F",
        "Hand": "R",
        "Educ": 14.0,
        "MMSE": 29.0,
        "CDR": 0.0,
        "nWBV": 0.737,
        "eTIV": 1344353.0,
        "ASF": 1.168,
    }]).set_index("ID")

    enriched = enrich_with_metadata(record, "OAS1_0001_MR1", 90, metadata_df=df)
    assert enriched["oasis_metadata"]["age"] == 74
    assert enriched["oasis_metadata"]["cdr"] == 0.0
    assert enriched["oasis_metadata"]["sex"] == "F"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
