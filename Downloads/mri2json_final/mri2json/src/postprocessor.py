"""
postprocessor.py
----------------
Validate the JSON from describer.py against our schema,
and enrich it with OASIS-1 ground-truth metadata (age, CDR, nWBV, etc.)

The merged output is the final training-ready record.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


REQUIRED_KEYS = {
    "modality", "plane", "anatomy", "tissue_findings",
    "pathology_flags", "confidence"
}

ANATOMY_KEYS = {"visible_structures", "hemisphere_symmetry", "cortical_ribbon", "sulcal_pattern"}
TISSUE_KEYS = {"gray_matter", "white_matter", "ventricles"}
PATHOLOGY_KEYS = {"atrophy", "white_matter_lesions", "mass_effect", "midline_shift"}


def validate_json(record: dict) -> tuple[bool, list[str]]:
    """
    Validate that a description record has the required structure.

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    if "error" in record:
        return False, [f"Parse error: {record.get('error', 'unknown')}"]

    for key in REQUIRED_KEYS:
        if key not in record:
            issues.append(f"Missing top-level key: {key}")

    anatomy = record.get("anatomy", {})
    for key in ANATOMY_KEYS:
        if key not in anatomy:
            issues.append(f"Missing anatomy.{key}")

    tissue = record.get("tissue_findings", {})
    for key in TISSUE_KEYS:
        if key not in tissue:
            issues.append(f"Missing tissue_findings.{key}")

    pathology = record.get("pathology_flags", {})
    for key in PATHOLOGY_KEYS:
        if key not in pathology:
            issues.append(f"Missing pathology_flags.{key}")

    confidence = record.get("confidence", None)
    if confidence is not None and not (0.0 <= float(confidence) <= 1.0):
        issues.append(f"confidence out of range: {confidence}")

    return len(issues) == 0, issues


def load_oasis_metadata(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load the OASIS-1 cross-sectional metadata CSV or XLSX.

    Expected columns (key ones):
        ID, M/F, Hand, Age, Educ, SES, MMSE, CDR, eTIV, nWBV, ASF, Delay
    """
    csv_path = Path(csv_path)
    if csv_path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(str(csv_path))
    else:
        df = pd.read_csv(str(csv_path))
    # Normalize the ID column name (sometimes it's 'ID', sometimes 'Subject ID')
    df.columns = [c.strip() for c in df.columns]
    if "Subject ID" in df.columns:
        df = df.rename(columns={"Subject ID": "ID"})
    df = df.set_index("ID")
    return df


def enrich_with_metadata(
    record: dict,
    subject_id: str,
    slice_index: int,
    metadata_df: Optional[pd.DataFrame] = None,
    mri_path: Optional[str] = None,
) -> dict:
    """
    Add provenance and OASIS metadata to the description record.

    Args:
        record: JSON from describer.py
        subject_id: e.g. "OAS1_0001_MR1"
        slice_index: which axial slice
        metadata_df: loaded OASIS CSV as DataFrame
        mri_path: path to source MRI file

    Returns:
        Enriched dict ready for final output
    """
    enriched = {
        "subject_id": subject_id,
        "slice_index": slice_index,
        "source_file": mri_path or "unknown",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "pipeline_version": "0.1.0",
        **record,
    }

    # Merge OASIS ground truth
    oasis_meta = {}
    if metadata_df is not None and subject_id in metadata_df.index:
        row = metadata_df.loc[subject_id]
        oasis_meta = {
            "age": int(row.get("Age", -1)) if pd.notna(row.get("Age")) else None,
            "sex": str(row.get("M/F", "")).strip() or None,
            "handedness": str(row.get("Hand", "")).strip() or None,
            "education_years": row.get("Educ") if pd.notna(row.get("Educ", None)) else None,
            "mmse": row.get("MMSE") if pd.notna(row.get("MMSE", None)) else None,
            # CDR: 0=normal, 0.5=very mild AD, 1=mild AD, 2=moderate AD
            "cdr": row.get("CDR") if pd.notna(row.get("CDR", None)) else None,
            # nWBV: normalized whole brain volume (atrophy proxy)
            "nwbv": row.get("nWBV") if pd.notna(row.get("nWBV", None)) else None,
            "etiv": row.get("eTIV") if pd.notna(row.get("eTIV", None)) else None,
            "asf": row.get("ASF") if pd.notna(row.get("ASF", None)) else None,
        }
    
    enriched["oasis_metadata"] = oasis_meta

    return enriched


def postprocess(
    raw_record: dict,
    subject_id: str,
    slice_index: int,
    metadata_df: Optional[pd.DataFrame] = None,
    mri_path: Optional[str] = None,
    strict: bool = False,
) -> tuple[dict, bool, list[str]]:
    """
    Full postprocessing: validate + enrich.

    Args:
        strict: if True, raise ValueError on validation failure

    Returns:
        (enriched_record, is_valid, issues)
    """
    is_valid, issues = validate_json(raw_record)

    if not is_valid and strict:
        raise ValueError(f"Invalid JSON for {subject_id} slice {slice_index}: {issues}")

    enriched = enrich_with_metadata(
        raw_record, subject_id, slice_index, metadata_df, mri_path
    )
    enriched["validation"] = {
        "passed": is_valid,
        "issues": issues,
    }

    return enriched, is_valid, issues


def save_json(record: dict, output_dir: Union[str, Path], subject_id: str, slice_index: int) -> Path:
    """Save a single enriched record to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{subject_id}_slice{slice_index:03d}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return out_path
