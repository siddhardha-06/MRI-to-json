# mri2json 🧠

**Plain MRI image → Structured descriptive JSON**

Convert raw brain MRI scans into machine-readable JSON descriptions using a vision-language model. No segmentation masks, radiology reports, or specialized clinical software required.

---

## Overview

mri2json is an experimental pipeline that transforms raw OASIS-1 brain MRI scans into structured JSON records containing:

- Anatomical observations
- Tissue descriptions
- Imaging features
- Model-generated impressions
- OASIS metadata (CDR, MMSE, nWBV, eTIV, etc.)

The goal is to create machine-readable MRI datasets that can be used for AI research, dataset enrichment, retrieval systems, and downstream machine learning applications.

---

## Motivation

Medical imaging datasets often require:

- Manual annotations
- Radiology reports
- Segmentation masks
- Clinical software pipelines

Many open MRI datasets provide scans but very little structured text.

mri2json explores whether modern vision-language models can generate useful structured descriptions directly from MRI slices and combine them with existing dataset metadata.

---

## Dataset

This project currently uses a subset of the OASIS-1 dataset.

### Current Evaluation

- OASIS-1 Disc 1
- 39 subjects processed
- Includes cognitively normal and early Alzheimer's subjects
- Ground truth metadata available:
  - Age
  - Sex
  - CDR
  - MMSE
  - nWBV
  - eTIV

### Future Work

The pipeline is designed to scale to the complete OASIS-1 dataset (416 subjects) without modification.

OASIS Dataset:

https://www.oasis-brains.org/

---

## Pipeline

```text
OASIS MRI (.img/.hdr/.nii/.mgz)
                │
                ▼
       MRI Slice Extraction
                │
                ▼
       Image Preprocessing
                │
                ▼
      Vision-Language Model
                │
                ▼
       Structured JSON Output
                │
                ▼
      OASIS Metadata Merge
```

---

## Example Output

```json
{
  "subject_id": "OAS1_0001_MR1",
  "modality": "T1-weighted",
  "plane": "sagittal",

  "anatomy": {
    "visible_structures": [
      "cerebral cortex",
      "corpus callosum",
      "thalamus"
    ],
    "hemisphere_symmetry": "symmetric",
    "cortical_ribbon": "preserved",
    "sulcal_pattern": "mildly widened"
  },

  "tissue_findings": {
    "gray_matter": "preserved",
    "white_matter": "unremarkable",
    "ventricles": "mild prominence"
  },

  "observed_features": {
    "ventricular_enlargement": "mild",
    "sulcal_widening": "mild"
  },

  "global_impression": "Mild generalized age-related cerebral volume loss.",

  "confidence": 0.84,

  "oasis_metadata": {
    "age": 74,
    "sex": "F",
    "cdr": 0.0,
    "mmse": 29,
    "nwbv": 0.743,
    "etiv": 1344
  }
}
```

---

## Project Structure

```text
mri2json/
│
├── README.md
├── CLAUDE.md
├── requirements.txt
│
├── src/
│   ├── loader.py
│   ├── preprocessor.py
│   ├── describer.py
│   ├── postprocessor.py
│   └── pipeline.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── output/
│
├── docs/
│   └── schema.md
│
└── tests/
    └── test_pipeline.py
```

---

## Features

### MRI Loading

Supports common neuroimaging formats:

- ANALYZE (.img/.hdr)
- NIfTI (.nii/.nii.gz)
- MGZ

### Slice Extraction

- Automatic slice selection
- Configurable number of slices
- PNG export support

### Structured Description Generation

Produces:

- Anatomy descriptions
- Tissue observations
- Symmetry assessment
- Ventricular appearance
- Sulcal appearance
- Global imaging impression

### Metadata Integration

Automatically merges:

- Age
- Sex
- CDR
- MMSE
- nWBV
- eTIV

from OASIS metadata tables.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/mri2json.git

cd mri2json
```

### Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

```text
python>=3.10

google-genai>=0.8.0
nibabel>=5.1.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
scipy>=1.11.0
openpyxl>=3.1.0
tqdm>=4.66.0
```

---

## Dataset Setup

Download OASIS-1 and place files as follows:

```text
data/
└── raw/
    ├── OAS1_0007_MR1/
    ├── OAS1_0009_MR1/
    ├── OAS1_0010_MR1/
    ├── ...
```

Place metadata file at project root:

```text
oasis_cross-sectional.xlsx
```

---

## Configuration

Set your Gemini API key.

### Windows

```bash
setx GEMINI_API_KEY "YOUR_API_KEY"
```

### Linux/macOS

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

---

## Usage

### Process Single Subject

```bash
python -m src.pipeline \
    --subject OAS1_0007_MR1 \
    --data-dir data/raw \
    --metadata oasis_cross-sectional.xlsx
```

### Process All Subjects

```bash
python -m src.pipeline \
    --all \
    --data-dir data/raw \
    --metadata oasis_cross-sectional.xlsx
```

### Process First 5 Subjects

```bash
python -m src.pipeline \
    --all \
    --limit 5 \
    --data-dir data/raw \
    --metadata oasis_cross-sectional.xlsx
```

### Save Intermediate PNGs

```bash
python -m src.pipeline \
    --all \
    --save-pngs \
    --data-dir data/raw \
    --metadata oasis_cross-sectional.xlsx
```

### Change Slice Count

```bash
python -m src.pipeline \
    --all \
    --n-slices 5 \
    --data-dir data/raw \
    --metadata oasis_cross-sectional.xlsx
```

---

## Output

Generated JSON files are stored in:

```text
data/output/
```

Example:

```text
data/output/
├── OAS1_0007_MR1.json
├── OAS1_0009_MR1.json
├── OAS1_0010_MR1.json
└── ...
```

---

## Research Use Cases

Possible applications include:

- MRI dataset enrichment
- Medical AI pretraining
- Retrieval-Augmented Generation (RAG)
- Structured medical dataset creation
- Vision-language model evaluation
- MRI metadata exploration

---

## Limitations

This project is intended for research purposes only.

The generated descriptions:

- Are produced by a vision-language model
- Are not radiologist reports
- Should not be used for diagnosis
- May contain inaccuracies or hallucinations

No clinical decisions should be made based on the generated output.

---

## Future Work

- Full OASIS-1 evaluation (416 subjects)
- Multi-slice reasoning
- Better anatomical localization
- Validation against expert reports
- MRI-to-JSON benchmark dataset
- Support for additional neuroimaging datasets

---

## Citation

```bibtex
@software{mri2json2026,
  author = {Rohith Karning},
  title = {mri2json: MRI to Structured JSON Generation Using Vision-Language Models},
  year = {2026},
  url = {https://github.com/yourusername/mri2json}
}
```

---

## Acknowledgements

- OASIS-1 Dataset
- nibabel
- Google Gemini
- Open-source neuroimaging community

---
