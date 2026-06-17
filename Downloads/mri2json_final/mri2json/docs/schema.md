# MRI2JSON Output Schema

Each output file is `{subject_id}_slice{NNN}.json`.

## Top-Level Fields

| Field | Type | Description |
|---|---|---|
| `subject_id` | string | OASIS-1 subject ID (e.g. `OAS1_0001_MR1`) |
| `slice_index` | int | Axial slice index in the volume |
| `source_file` | string | Absolute path to the source NIfTI/MGZ file |
| `generated_at` | string | ISO 8601 UTC timestamp |
| `pipeline_version` | string | Version of mri2json that created this |
| `modality` | string | `T1-weighted`, `T2-weighted`, `FLAIR`, or `unknown` |
| `plane` | string | `axial`, `coronal`, or `sagittal` |
| `image_quality` | string | `good`, `fair`, or `poor` |
| `global_impression` | string | 1-2 sentence radiologist-style summary |
| `confidence` | float | Model confidence in assessment (0.0–1.0) |

## `anatomy` Object

| Field | Type | Description |
|---|---|---|
| `visible_structures` | string[] | List of anatomical structures identifiable in this slice |
| `hemisphere_symmetry` | string | Symmetry assessment with brief note |
| `cortical_ribbon` | string | `preserved`, `thinned`, or `not visible` |
| `sulcal_pattern` | string | Sulcal widening assessment |

## `tissue_findings` Object

| Field | Type | Description |
|---|---|---|
| `gray_matter` | string | Description of cortical gray matter |
| `white_matter` | string | White matter signal — normal or abnormalities |
| `ventricles` | string | Ventricular size and appearance |
| `sulci_gyri` | string | Sulcal/gyral pattern description |
| `basal_ganglia` | string | Basal ganglia appearance or `not visible in this slice` |
| `thalamus` | string | Thalamic appearance or `not visible in this slice` |
| `cerebellum` | string | Cerebellar appearance or `not visible in this slice` |
| `brainstem` | string | Brainstem appearance or `not visible in this slice` |

## `pathology_flags` Object

| Field | Type | Description |
|---|---|---|
| `atrophy` | bool | True if cortical or global atrophy is visible |
| `atrophy_severity` | string | `none`, `mild`, `moderate`, or `severe` |
| `white_matter_lesions` | bool | True if WM hyperintensities/lesions visible |
| `lesion_count_estimate` | string | `none`, `1-3`, `4-10`, or `>10` |
| `mass_effect` | bool | True if mass effect is present |
| `midline_shift` | bool | True if midline shift is visible |
| `hydrocephalus` | bool | True if hydrocephalus is present |
| `infarct` | bool | True if infarct/ischemic change visible |
| `hemorrhage` | bool | True if hemorrhage is visible |
| `notes` | string | Free text for any other findings |

## `oasis_metadata` Object (ground truth from CSV)

| Field | Type | Description |
|---|---|---|
| `age` | int | Subject age in years |
| `sex` | string | `M` or `F` |
| `handedness` | string | `R`, `L`, or `A` (ambidextrous) |
| `education_years` | float | Years of education |
| `mmse` | float | Mini-Mental State Examination score (0–30) |
| `cdr` | float | **Clinical Dementia Rating** — key label for training (0, 0.5, 1, 2) |
| `nwbv` | float | **Normalized Whole Brain Volume** — atrophy proxy |
| `etiv` | float | Estimated Total Intracranial Volume (mm³) |
| `asf` | float | Atlas Scaling Factor |

## `validation` Object

| Field | Type | Description |
|---|---|---|
| `passed` | bool | Whether the record passed schema validation |
| `issues` | string[] | List of any validation issues found |

---

## Example Record

```json
{
  "subject_id": "OAS1_0001_MR1",
  "slice_index": 90,
  "source_file": "/home/user/mri2json/data/raw/OAS1_0001_MR1/mri/orig/001.mgz",
  "generated_at": "2026-06-17T10:23:41Z",
  "pipeline_version": "0.1.0",
  "modality": "T1-weighted",
  "plane": "axial",
  "image_quality": "good",
  "anatomy": {
    "visible_structures": ["cerebral cortex", "white matter", "lateral ventricles", "caudate nucleus", "putamen", "thalamus", "corpus callosum"],
    "hemisphere_symmetry": "symmetric",
    "cortical_ribbon": "thinned",
    "sulcal_pattern": "moderately widened"
  },
  "tissue_findings": {
    "gray_matter": "Thinned cortical ribbon with reduced gray-white matter differentiation",
    "white_matter": "Mild periventricular signal changes bilaterally, no focal lesions",
    "ventricles": "Mildly enlarged lateral ventricles bilaterally, symmetric",
    "sulci_gyri": "Moderately widened sulci consistent with volume loss",
    "basal_ganglia": "Caudate and putamen appear normal in signal and size",
    "thalamus": "Thalami symmetric, normal signal",
    "cerebellum": "not visible in this slice",
    "brainstem": "not visible in this slice"
  },
  "pathology_flags": {
    "atrophy": true,
    "atrophy_severity": "mild",
    "white_matter_lesions": false,
    "lesion_count_estimate": "none",
    "mass_effect": false,
    "midline_shift": false,
    "hydrocephalus": false,
    "infarct": false,
    "hemorrhage": false,
    "notes": "Mild generalized cortical atrophy with sulcal widening, age-related pattern"
  },
  "global_impression": "Mild cortical atrophy with mildly enlarged ventricles, consistent with age-related volume loss in a 74-year-old. No focal pathology identified.",
  "confidence": 0.85,
  "oasis_metadata": {
    "age": 74,
    "sex": "F",
    "handedness": "R",
    "education_years": 14.0,
    "mmse": 29.0,
    "cdr": 0.0,
    "nwbv": 0.737,
    "etiv": 1344353.0,
    "asf": 1.168
  },
  "validation": {
    "passed": true,
    "issues": []
  }
}
```
