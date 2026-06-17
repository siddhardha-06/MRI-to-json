# MRI2JSON — Project Context for LLMs

> Read this file first. It tells you exactly what this project is, what problem it solves,
> what has been built, and how to continue working on it.

---

## What This Project Does

**One-line summary:** Plain MRI image → structured descriptive JSON with anatomy labels, tissue findings, and pathology flags — with no DICOM tools, no segmentation masks, and no radiology reports required.

**Input:** A single 2D brain MRI slice (NIfTI `.nii`/`.nii.gz` or PNG/JPEG export)
**Output:** A structured JSON object like:

```json
{
  "modality": "T1-weighted MRI",
  "plane": "axial",
  "anatomy": {
    "visible_structures": ["cerebral cortex", "white matter", "lateral ventricles", "caudate nucleus", "thalamus"],
    "hemisphere_symmetry": "mild asymmetry noted left > right"
  },
  "tissue_findings": {
    "gray_matter": "preserved cortical thickness",
    "white_matter": "no focal hyperintensities",
    "ventricles": "mildly enlarged, consistent with age-related atrophy"
  },
  "pathology_flags": {
    "atrophy": true,
    "lesions": false,
    "mass_effect": false,
    "midline_shift": false,
    "notes": "mild generalized cortical atrophy, age-appropriate"
  },
  "confidence": 0.82,
  "slice_index": 42,
  "source_subject": "OAS1_0001_MR1"
}
```

---

## Why This Is Novel

Existing tools that are closest:
- **MedSAM / SAM-Med2D** — require segmentation masks as input
- **RadBERT / BioViL** — work on radiology *reports*, not raw images
- **CheXpert labeler** — chest X-ray only, requires existing report text
- **July 2025 MRI-JSON paper** — still needs segmentation masks + clinical metadata input

**This pipeline goes further:** image only in → descriptive JSON out. No DICOM viewer. No radiologist report. No masks. Anyone can use the output JSONs to train AI models.

---

## Dataset: OASIS-1

- **Full name:** Open Access Series of Imaging Studies, Cross-Sectional (OASIS-1)
- **URL:** https://www.oasis-brains.org/
- **Subjects:** 416 subjects aged 18–96, mix of cognitively normal and early Alzheimer's
- **Format:** NIfTI files (`.nii` / `.nii.gz`), organized by subject
- **Key metadata available (from `oasis_cross-sectional.csv`):**
  - `ID` — subject ID (e.g. `OAS1_0001_MR1`)
  - `Age`, `Gender (M/F)`, `Hand` (handedness)
  - `CDR` — Clinical Dementia Rating (0 = normal, 0.5 = very mild, 1 = mild, 2 = moderate)
  - `MMSE` — Mini-Mental State Examination score
  - `eTIV` — Estimated Total Intracranial Volume
  - `nWBV` — Normalized Whole Brain Volume (key atrophy marker)
  - `ASF` — Atlas Scaling Factor
- **File structure expected:**
  ```
  data/raw/
    OAS1_0001_MR1/
      mri/
        orig/
          001.mgz  (or converted .nii.gz)
    OAS1_0002_MR1/
    ...
  oasis_cross-sectional.csv
  ```

---

## Project Architecture

```
mri2json/
├── CLAUDE.md                  ← YOU ARE HERE
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── loader.py              ← Load NIfTI/MGZ, extract 2D slices
│   ├── preprocessor.py        ← Normalize, window, export PNG
│   ├── describer.py           ← Core: image → JSON via Claude Vision API
│   ├── postprocessor.py       ← Validate, clean, enrich JSON output
│   ├── pipeline.py            ← Orchestrates full run
│   └── utils.py               ← Helpers (logging, paths, metadata CSV)
├── data/
│   ├── raw/                   ← OASIS-1 NIfTI files go here
│   ├── processed/             ← Exported PNG slices
│   └── output/                ← Final JSON files per subject
├── notebooks/
│   └── explore_oasis.ipynb    ← EDA notebook
├── tests/
│   └── test_pipeline.py
└── docs/
    └── schema.md              ← JSON output schema specification
```

---

## Core Pipeline Steps

1. **Load** — Read `.nii.gz` or `.mgz` file with `nibabel`. Extract axial slices.
2. **Select slice** — Pick the most informative slice (center-of-mass heuristic or all slices).
3. **Preprocess** — Normalize intensity (min-max or percentile), convert to 8-bit PNG.
4. **Describe** — Send PNG to Claude Vision (`claude-sonnet-4-6`) with a structured prompt.
5. **Parse** — Extract JSON from response, validate against schema.
6. **Enrich** — Merge with OASIS metadata (CDR, nWBV, age) into final JSON.
7. **Save** — Write per-subject JSON to `data/output/`.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Use Claude Vision API | Best zero-shot medical image understanding without fine-tuning |
| 2D axial slices (not 3D volume) | API accepts images; axial is most clinically standard |
| Structured prompt with schema | Forces consistent JSON output, easier to validate |
| Merge OASIS metadata | Adds ground-truth CDR/nWBV for downstream training signal |
| No segmentation required | This is the core novelty — pure image-to-text pipeline |

---

## Prompt Strategy (in `describer.py`)

The system prompt tells Claude to act as a neuroradiologist describing what it sees. The user message includes:
- The MRI slice as base64 PNG
- A JSON schema to fill
- Instructions to flag pathology conservatively
- Instruction to return ONLY valid JSON, no prose

See `src/describer.py` for the exact prompts. Tune them there.

---

## Output JSON Schema

Full schema defined in `docs/schema.md`. Top-level keys:
- `subject_id`, `slice_index`, `modality`, `plane`
- `anatomy` → `visible_structures`, `hemisphere_symmetry`
- `tissue_findings` → `gray_matter`, `white_matter`, `ventricles`, `sulci`
- `pathology_flags` → boolean flags + `notes`
- `metadata` → merged OASIS fields (age, CDR, nWBV, etc.)
- `confidence` → model self-reported confidence 0–1

---

## How to Run

```bash
# Install deps
pip install -r requirements.txt

# Run on a single subject
python -m src.pipeline --subject OAS1_0001_MR1 --data-dir data/raw

# Run on all subjects
python -m src.pipeline --all --data-dir data/raw --metadata oasis_cross-sectional.csv

# Output goes to data/output/<subject_id>.json
```

Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Current Status

- [x] Project scaffold created
- [x] CLAUDE.md written
- [x] Schema defined
- [x] loader.py — NIfTI/MGZ loading
- [x] preprocessor.py — normalization + PNG export
- [x] describer.py — Claude Vision API call
- [x] postprocessor.py — JSON validation + metadata merge
- [x] pipeline.py — orchestration
- [ ] Test on real OASIS-1 data (awaiting user to place files in data/raw/)
- [ ] Evaluate JSON quality vs CDR ground truth
- [ ] Build eval notebook
- [ ] Consider batch processing with async API calls

---

## Known Issues / Watch Out For

- OASIS-1 raw files may be in `.mgz` format (FreeSurfer). Use `mri_convert` or nibabel's MGH reader.
- Claude Vision sometimes returns prose before JSON — `postprocessor.py` strips this.
- Some OASIS subjects have very thin slices (1mm isotropic) — pick every 5th slice to avoid redundancy.
- CDR=0 subjects still have varying nWBV — don't assume CDR=0 means "normal-looking" MRI.

---

## Useful References

- OASIS-1 paper: Marcus et al., 2007 — *Journal of Cognitive Neuroscience*
- Claude Vision API docs: https://docs.anthropic.com/en/docs/vision
- nibabel docs: https://nipy.org/nibabel/
- NIfTI format reference: https://nifti.nimh.nih.gov/
- Closest prior work: "Automated MRI Report Generation via Vision-Language Models" (July 2025)

---

*Last updated by: Claude Sonnet 4.6 | Project started: June 2026*
