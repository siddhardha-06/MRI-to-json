"""
pipeline.py
-----------
Orchestrates the full mri2json pipeline:
    Load → Preprocess → Describe → Postprocess → Save

Can be run as a module:
    python -m src.pipeline --subject OAS1_0001_MR1 --data-dir data/raw
    python -m src.pipeline --all --data-dir data/raw --metadata oasis_cross-sectional.csv
"""

import argparse
import logging
import json
import os
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from .loader import load_subject, find_mri_file
from .preprocessor import preprocess_slice
from .describer import describe_slice
from .postprocessor import postprocess, save_json, load_oasis_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("mri2json")


def run_subject(
    subject_id: str,
    subject_dir: Path,
    output_dir: Path,
    processed_dir: Optional[Path] = None,
    metadata_df=None,
    n_slices: int = 3,
    save_pngs: bool = False,
) -> list[dict]:
    """
    Run the full pipeline for a single OASIS-1 subject.

    Returns list of enriched JSON records (one per slice).
    """
    log.info(f"Processing subject: {subject_id}")
    results = []

    # Step 1: Load
    try:
        slices, mri_path = load_subject(subject_dir, n_slices=n_slices)
    except FileNotFoundError as e:
        log.warning(f"  SKIP {subject_id}: {e}")
        return []

    log.info(f"  Loaded {len(slices)} slices from {mri_path}")

    # Build context string (age/sex only, NOT CDR — that's our label)
    subject_context = None
    if metadata_df is not None and subject_id in metadata_df.index:
        row = metadata_df.loc[subject_id]
        age = int(row.get("Age", 0)) if str(row.get("Age", "")).replace(".", "").isdigit() else "?"
        sex_code = str(row.get("M/F", "")).strip()
        sex = "male" if sex_code == "M" else ("female" if sex_code == "F" else "unknown sex")
        subject_context = f"{age}-year-old {sex}"

    # Step 2–5: For each slice
    for slice_index, slice_2d in slices:
        log.info(f"  Slice {slice_index} → preprocess")

        # Preprocess
        b64, pil_img = preprocess_slice(
            slice_2d,
            slice_index=slice_index,
            subject_id=subject_id,
            output_dir=processed_dir,
            save_png=save_pngs,
        )

        # Describe
        log.info(f"  Slice {slice_index} → Claude Vision API")
        try:
            raw_record = describe_slice(
                base64_image=b64,
                subject_id=subject_id,
                slice_index=slice_index,
                subject_context=subject_context,
            )
        except Exception as e:
            log.error(f"  ERROR on slice {slice_index}: {e}")
            raw_record = {"error": str(e), "confidence": 0.0}

        # Postprocess
        enriched, is_valid, issues = postprocess(
            raw_record,
            subject_id=subject_id,
            slice_index=slice_index,
            metadata_df=metadata_df,
            mri_path=mri_path,
        )

        if not is_valid:
            log.warning(f"  Slice {slice_index} validation issues: {issues}")

        # Save
        out_path = save_json(enriched, output_dir, subject_id, slice_index)
        log.info(f"  Saved → {out_path}")

        results.append(enriched)

    return results


def run_all(
    data_dir: Path,
    output_dir: Path,
    processed_dir: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    n_slices: int = 3,
    save_pngs: bool = False,
    limit: Optional[int] = None,
) -> None:
    """
    Run the pipeline over all OASIS-1 subject directories in data_dir.
    """
    metadata_df = None
    if metadata_path and metadata_path.exists():
        metadata_df = load_oasis_metadata(metadata_path)
        log.info(f"Loaded metadata for {len(metadata_df)} subjects")

    subject_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("OAS")
    ])

    if limit:
        subject_dirs = subject_dirs[:limit]

    log.info(f"Found {len(subject_dirs)} subjects to process")

    summary = {"processed": 0, "skipped": 0, "errors": 0}

    for subject_dir in tqdm(subject_dirs, desc="Subjects"):
        subject_id = subject_dir.name
        try:
            results = run_subject(
                subject_id=subject_id,
                subject_dir=subject_dir,
                output_dir=output_dir,
                processed_dir=processed_dir,
                metadata_df=metadata_df,
                n_slices=n_slices,
                save_pngs=save_pngs,
            )
            if results:
                summary["processed"] += 1
            else:
                summary["skipped"] += 1
        except Exception as e:
            log.error(f"Unexpected error for {subject_id}: {e}")
            summary["errors"] += 1

    log.info(f"\nDone. {summary}")

    # Write summary
    summary_path = output_dir / "_run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MRI2JSON: Convert OASIS-1 brain MRI images to structured JSON descriptions"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"),
                        help="Directory containing OASIS-1 subject folders")
    parser.add_argument("--output-dir", type=Path, default=Path("data/output"),
                        help="Where to save output JSON files")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"),
                        help="Where to save intermediate PNG slices")
    parser.add_argument("--metadata", type=Path, default=Path("oasis_cross-sectional.xlsx"),
                        help="Path to OASIS-1 metadata CSV")
    parser.add_argument("--subject", type=str, default=None,
                        help="Run on a single subject ID (e.g. OAS1_0001_MR1)")
    parser.add_argument("--all", action="store_true",
                        help="Run on all subjects in --data-dir")
    parser.add_argument("--n-slices", type=int, default=3,
                        help="Number of slices to extract per subject")
    parser.add_argument("--save-pngs", action="store_true",
                        help="Save intermediate PNG slices to --processed-dir")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N subjects (for testing)")

    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: Set GEMINI_API_KEY environment variable first.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = None
    if args.metadata.exists():
        metadata_df = load_oasis_metadata(args.metadata)

    if args.subject:
        subject_dir = args.data_dir / args.subject
        if not subject_dir.exists():
            print(f"ERROR: Subject directory not found: {subject_dir}")
            return
        run_subject(
            subject_id=args.subject,
            subject_dir=subject_dir,
            output_dir=args.output_dir,
            processed_dir=args.processed_dir if args.save_pngs else None,
            metadata_df=metadata_df,
            n_slices=args.n_slices,
            save_pngs=args.save_pngs,
        )

    elif args.all:
        run_all(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            processed_dir=args.processed_dir if args.save_pngs else None,
            metadata_path=args.metadata,
            n_slices=args.n_slices,
            save_pngs=args.save_pngs,
            limit=args.limit,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
