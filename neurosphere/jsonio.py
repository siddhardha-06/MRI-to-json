"""Build and write NeuroSphere JSON documents.

Phase 1 produces an *interim* document (schema_version "2.0.0-interim"): only values that are
actually measured from the file (header geometry, checksum) plus explicit null + reason for
every section that later phases will fill. It deliberately contains none of the v1 content:
no slice-percentage "volumes", no formula-generated hemisphere means, no interpretation strings,
no pathology flags, no simulated QC and no U-Net volumes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import __version__
from .io import load_t1
from .nulls import assert_explained, set_null
from .provenance import provenance
from .qc import quality_block

INTERIM_SCHEMA_VERSION = "2.0.0-interim"

SEGMENTATION_NOT_COMPUTED = (
    "Not computed. The v1 3D U-Net inference discarded the image orientation and counted voxels in "
    "template space, so its volumes are not reported. Native-space segmentation with an "
    "inverse warp is implemented in Phase 2.")
NORMATIVE_NOT_COMPUTED = "Normative z-scores require the IXI demographics and the cross-fitted model of Phase 3."
LIMITATIONS = [
    "Interim document: only header-derived acquisition values are measured at this stage.",
    "T1-only morphometry; does not represent focal lesions or signal abnormality.",
    "Research use only; not for diagnosis.",
]


def build_interim_json(path: str | Path, mriqc_official: dict | None = None, subject_id: str | None = None) -> dict:
    t0 = time.perf_counter()
    img = load_t1(path)
    m = img.meta
    doc: dict = {"schema_version": INTERIM_SCHEMA_VERSION}

    subject = {"source_file": m["source_file"], "sha256": m["sha256"]}
    sid = subject_id or m["subject_id"]
    if sid:
        subject["id"] = sid
    else:
        set_null(subject, "id", "file name does not follow the IXI naming scheme and no ID was given")
    doc["subject"] = subject

    acq = {
        "shape": m["shape_original"],
        "voxel_size_mm": m["voxel_size_mm_original"],
        "orientation_original": m["orientation_original"],
        "dtype_on_disk": m["dtype_on_disk"],
        "shape_ras": m["shape"],
        "voxel_size_mm_ras": m["voxel_size_mm"],
        "voxel_volume_ml": round(m["voxel_volume_ml"], 9),
        "field_of_view_ml": round(m["voxel_volume_ml"] * float(img.data.size), 3),
    }
    if m["site"]:
        acq["site"] = m["site"]
    else:
        set_null(acq, "site", "site is only known for IXI file names (Guys, HH, IOP)")
    doc["acquisition"] = acq

    doc["quality"] = quality_block(mriqc_official)
    for key in ("segmentation", "volumes_ml", "etiv_ml", "normalized_volumes", "asymmetry_index_pct"):
        set_null(doc, key, SEGMENTATION_NOT_COMPUTED)
    set_null(doc, "intensity", "Within-tissue intensity summaries need the tissue labels of Phase 2-3.")
    set_null(doc, "normative", NORMATIVE_NOT_COMPUTED)
    doc["limitations"] = list(LIMITATIONS)
    doc["provenance"] = provenance(runtime_s={"total": round(time.perf_counter() - t0, 3)})
    doc["provenance"]["pipeline_version"] = __version__
    assert_explained(doc)
    return doc


def write_json(doc: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


# ── NeuroSphere JSON 2.0.0 (Phase 3) ──────────────────────────────────────────
SCHEMA_VERSION = "2.0.0"
LIMITATIONS_V2 = [
    "T1-only global and regional morphometry, tissue intensity and image quality; does not represent focal lesions or signal abnormality.",
    "Segmentation by a 3D U-Net trained on FreeSurfer aseg labels (TransMorph IXI); it inherits FreeSurfer's definitions and errors.",
    "Cortex is the FreeSurfer volumetric (pre-surface) aseg label; white matter excludes the corpus callosum.",
    "Normative reference: healthy IXI adults (3 London sites); z-scores outside this population are extrapolations.",
    "Research use only; not for diagnosis.",
]


def build_json_v2(result: dict, acquisition: dict, demographics: dict, gate: dict, normative: dict | None,
                  normative_reference: str | None) -> dict:
    """Assemble the 2.0.0 document from a pipeline result.json and the cohort-level gate/normative outputs.
    `demographics` (age/sex from IXI.xls) are covariates, not image-derived features: they are excluded from
    to_vector() and from the LLM serialisation profiles by default."""
    from .features import morphometry
    m = morphometry(result)
    doc = {"schema_version": SCHEMA_VERSION,
           "subject": {"id": result["subject"], "source_file": result["source_file"], "sha256": result["sha256"]},
           "acquisition": acquisition,
           "covariates": {**demographics, "source": "IXI.xls (not image-derived; excluded from feature vectors)"},
           "quality": {"iqms": result["iqms"], "gate": gate},
           "segmentation": {"model": "3D U-Net (Option A recipe)", "checkpoint": result["checkpoint"],
                            "label_set": "FreeSurfer aseg, 29 structures", "space": "native (affine inverse of template registration)",
                            "volume_method": "template-space voxel count x |det(affine)| (primary); native-grid count as check",
                            "side_check": result["side_check_native"]},
           **m,
           "intensity": result["intensity"]}
    set_null(doc["quality"], "mriqc_official", "official MRIQC not run for this subject (optional)")
    doc["segmentation"]["uncertainty"] = None
    doc["segmentation"]["uncertainty_reason"] = "flip-TTA uncertainty not computed in this release"
    if normative is None:
        set_null(doc, "normative", "no normative model output for this subject")
    else:
        doc["normative"] = {"reference": normative_reference, **normative}
    doc["limitations"] = list(LIMITATIONS_V2)
    doc["provenance"] = {"pipeline_version": __version__, "pipeline_run": result.get("provenance"),
                         "runtime_s": result.get("timings_s"), "json_built": provenance()}
    assert_explained(doc)
    return doc
