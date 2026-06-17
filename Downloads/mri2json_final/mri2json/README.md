# MRI2JSON — Brain MRI to Structured JSON

A **novel pipeline** that converts raw brain MRI images into **structured, descriptive JSON** using Google's Gemini Vision API — **without requiring segmentation masks, DICOM tools, or radiology reports**.

This repository explores whether modern vision-language models can generate meaningful, structured descriptions directly from medical MRI slices and enrich them with clinical metadata (CDR, MMSE, nWBV, age, education) from the OASIS-1 dataset.

## What This Does

**Input:** Raw 2D brain MRI slice (NIfTI `.nii`/`.nii.gz` or PNG)  
**Output:** Structured JSON with anatomy, tissue findings, pathology flags, confidence scores, and OASIS metadata

```json
{
  "subject_id": "OAS1_0001_MR1",
  "modality": "T1-weighted",
  "plane": "sagittal",
  "anatomy": {
    "visible_structures": ["cerebral cortex", "white matter", "ventricles"],
    "hemisphere_symmetry": "symmetric",
    "cortical_ribbon": "preserved",
    "sulcal_pattern": "mildly widened"
  },
  "pathology_flags": {
    "atrophy": true,
    "atrophy_severity": "mild",
    "white_matter_lesions": false
  },
  "confidence": 0.95,
  "oasis_metadata": {
    "age": 74,
    "cdr": 0.0,
    "mmse": 29.0,
    "nwbv": 0.743
  }
}
```

## Why This Is Novel

Existing medical AI tools require:
- **MedSAM / SAM-Med2D** — segmentation masks as input
- **RadBERT / BioViL** — radiology reports, not images
- **CheXpert labeler** — chest X-rays only
- **July 2025 MRI-JSON paper** — still needs masks + metadata

**This pipeline is different:** raw image only → JSON. No masks. No reports. Direct zero-shot inference.

## Dataset

**OASIS-1 (Open Access Series of Imaging Studies)**
- 416 subjects aged 18–96
- Cognitively normal to early Alzheimer's
- NIfTI format with rich metadata
- Download: https://www.oasis-brains.org/

Current processing: **39 subjects** completed
## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/rohithkarning-debug/mri2json.git
cd mri2json
pip install -r requirements.txt
```

### 2. Set API Key
```bash
# Windows
$env:GEMINI_API_KEY = "your-api-key"

# Linux/macOS
export GEMINI_API_KEY="your-api-key"
```

Get free key: https://ai.google.dev/

### 3. Prepare Data
```
data/raw/
  OAS1_0001_MR1/RAW/OAS1_0001_MR1_mpr-1_anon.img
  OAS1_0002_MR1/RAW/...
oasis_cross-sectional.xlsx
```

### 4. Run Pipeline
```bash
# Single subject
python -m src.pipeline --subject OAS1_0001_MR1 --data-dir data/raw --metadata oasis_cross-sectional.xlsx

# All subjects
python -m src.pipeline --all --data-dir data/raw --metadata oasis_cross-sectional.xlsx --limit 5

# Save PNGs
python -m src.pipeline --all --data-dir data/raw --metadata oasis_cross-sectional.xlsx --save-pngs
```

Output: `data/output/{SUBJECT}_slice{INDEX}.json`

## Architecture

```
OASIS MRI (.img/.hdr/.nii.gz)
         ↓
    Slice Extraction (3 slices per subject)
         ↓
    Image Preprocessing (normalize, PNG export)
         ↓
    Gemini 2.5 Flash Lite Vision API
         ↓
    JSON Parsing & Validation
         ↓
    Metadata Enrichment (CDR, MMSE, nWBV, age)
         ↓
    Save → data/output/{SUBJECT}.json
```



### Components

| Module | Purpose |
|--------|---------|
| `loader.py` | Load NIfTI/MGZ/ANALYZE files with nibabel |
| `preprocessor.py` | Normalize intensity, clip outliers, export PNG |
| `describer.py` | Send PNG to Gemini 2.5 Flash Lite + parse JSON |
| `postprocessor.py` | Validate JSON, merge OASIS metadata |
| `pipeline.py` | Orchestrate full workflow + CLI |
| `utils.py` | Logging, path helpers, metadata CSV loading |

## Key Features

✅ **Zero-shot learning** — No fine-tuning needed  
✅ **No segmentation masks** — Works directly on raw images  
✅ **Scalable** — Batch process 39+ subjects  
✅ **Metadata fusion** — Enriches with CDR, MMSE, nWBV, age, education  
✅ **Conservative** — Only flags pathology if clearly visible  
✅ **Validated** — JSON schema validation + error handling  
✅ **Confidence scores** — Model self-reports confidence per slice  

## Performance

| Metric | Value |
|--------|-------|
| Model | Gemini 2.5 Flash Lite |
| Subjects processed | 31–39 (quota-limited) |
| Slices per subject | 3 (configurable) |
| Time per slice | ~5 seconds |
| JSON pass rate | ~95% |
| Avg confidence | 0.85–0.95 |

## Limitations & Known Issues

⚠️ **Free tier quota** — Gemini limits ~20 requests/day (free)
- Workaround: Use `--limit N` or upgrade to paid tier

⚠️ **2D slices only** — No 3D volume reconstruction

⚠️ **T1-weight focus** — Best on T1; T2/FLAIR may vary

⚠️ **429 errors on batch** — Quota exhausted after ~20 slices

⚠️ **Plane confusion** — Some sagittal/axial mislabels (edge case)

## Project Structure

```
mri2json/
├── README.md              ← You are here
├── CLAUDE.md              ← Full LLM context
├── requirements.txt
├── src/
│   ├── loader.py          ← NIfTI/ANALYZE loading
│   ├── preprocessor.py    ← Normalization + PNG export
│   ├── describer.py       ← Gemini API call
│   ├── postprocessor.py   ← JSON validation + metadata merge
│   ├── pipeline.py        ← CLI orchestration
│   └── utils.py           ← Helpers
├── data/
│   ├── raw/               ← OASIS-1 files
│   ├── processed/         ← PNG slices
│   └── output/            ← Final JSONs
├── docs/
│   └── schema.md          ← JSON schema spec
├── notebooks/
│   └── explore_oasis.ipynb
└── tests/
    └── test_pipeline.py
```

## Usage Examples

### Single Subject
```bash
python -m src.pipeline --subject OAS1_0001_MR1 --data-dir data/raw --metadata oasis_cross-sectional.xlsx
```

### Batch (Limited to 5)
```bash
python -m src.pipeline --all --limit 5 --data-dir data/raw --metadata oasis_cross-sectional.xlsx
```

### Save Intermediate PNGs
```bash
python -m src.pipeline --subject OAS1_0001_MR1 --data-dir data/raw --save-pngs
```

### Extract 5 Slices Per Subject
```bash
python -m src.pipeline --all --n-slices 5 --data-dir data/raw --metadata oasis_cross-sectional.xlsx
```

## Output Format

JSON saved to `data/output/{SUBJECT_ID}_slice{INDEX}.json`

**Top-level keys:**
- `subject_id`, `slice_index`, `modality`, `plane`
- `anatomy` → structures, symmetry, cortical ribbon, sulcal pattern
- `tissue_findings` → gray matter, white matter, ventricles, sulci
- `pathology_flags` → atrophy, lesions, mass effect, midline shift, etc.
- `oasis_metadata` → age, CDR, MMSE, nWBV, eTIV, education
- `confidence` → 0–1 score
- `validation` → pass/fail + issues list

See [docs/schema.md](docs/schema.md) for full schema.

## Research Applications

- MRI dataset enrichment for downstream ML
- Medical AI pretraining corpora
- Retrieval-Augmented Generation (RAG)
- Vision-language model evaluation
- Structured medical dataset creation
- Brain imaging metadata exploration

## Future Work

- [ ] Batch processing with intelligent retry + rate limiting
- [ ] 3D volume reconstruction from stacked 2D JSONs
- [ ] T2 / FLAIR sequence support
- [ ] Comparison with radiologist annotations
- [ ] Fine-tuned model for better domain performance
- [ ] Web UI for interactive slice visualization
- [ ] Full OASIS-1 (416 subjects) + ADNI, BraTS

## Disclaimer

⚠️ **Research use only.** Generated descriptions:
- Are produced by an AI model
- Are NOT radiologist reports
- Should NOT be used for diagnosis
- May contain inaccuracies or hallucinations

**No clinical decisions should be made based on this output.**

## References

- OASIS-1 paper: Marcus et al., 2007 — *Journal of Cognitive Neuroscience*
- Gemini API: https://ai.google.dev/gemini-api/docs/
- nibabel: https://nipy.org/nibabel/
- NIfTI format: https://nifti.nimh.nih.gov/

## Contributing

Pull requests welcome! Interested in:
- Better slice selection heuristics
- Domain-specific prompt engineering
- New medical imaging datasets (ADNI, BraTS, HCP)
- Validation against radiologist ground truth

## License

MIT License — See LICENSE file

## Author

**Rohith Karning**  
Machine Learning & Medical Imaging Enthusiast  
GitHub: https://github.com/rohithkarning-debug

---

**Get started:** Place OASIS data in `data/raw/`, set `GEMINI_API_KEY`, and run:
```bash
python -m src.pipeline --subject OAS1_0001_MR1 --data-dir data/raw --metadata oasis_cross-sectional.xlsx
```
