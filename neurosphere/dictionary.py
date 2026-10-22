"""Single source of truth for feature keys, units, compact keys and precision.

`neurosphere/schema/dictionary.json` is generated from this module (scripts/write_schema.py) so the
JSON, the dictionary, the serialisation profiles and docs/FEATURES.md cannot drift apart.
"""
from __future__ import annotations

# FreeSurfer aseg label -> (json key, description)
STRUCTURES = {
    2: ("left_cerebral_white_matter", "Left cerebral white matter (excludes corpus callosum)"),
    3: ("left_cerebral_cortex", "Left cerebral cortex (FreeSurfer aseg volumetric definition)"),
    4: ("left_lateral_ventricle", "Left lateral ventricle"),
    7: ("left_cerebellum_white_matter", "Left cerebellar white matter"),
    8: ("left_cerebellum_cortex", "Left cerebellar cortex"),
    10: ("left_thalamus", "Left thalamus"),
    11: ("left_caudate", "Left caudate"),
    12: ("left_putamen", "Left putamen"),
    13: ("left_pallidum", "Left pallidum"),
    14: ("third_ventricle", "Third ventricle"),
    15: ("fourth_ventricle", "Fourth ventricle"),
    16: ("brain_stem", "Brain stem"),
    17: ("left_hippocampus", "Left hippocampus"),
    18: ("left_amygdala", "Left amygdala"),
    26: ("left_accumbens", "Left nucleus accumbens"),
    28: ("left_ventral_dc", "Left ventral diencephalon"),
    41: ("right_cerebral_white_matter", "Right cerebral white matter (excludes corpus callosum)"),
    42: ("right_cerebral_cortex", "Right cerebral cortex (FreeSurfer aseg volumetric definition)"),
    43: ("right_lateral_ventricle", "Right lateral ventricle"),
    46: ("right_cerebellum_white_matter", "Right cerebellar white matter"),
    47: ("right_cerebellum_cortex", "Right cerebellar cortex"),
    49: ("right_thalamus", "Right thalamus"),
    50: ("right_caudate", "Right caudate"),
    51: ("right_putamen", "Right putamen"),
    52: ("right_pallidum", "Right pallidum"),
    53: ("right_hippocampus", "Right hippocampus"),
    54: ("right_amygdala", "Right amygdala"),
    58: ("right_accumbens", "Right nucleus accumbens"),
    60: ("right_ventral_dc", "Right ventral diencephalon"),
}
PAIRS = {  # bilateral name -> (left label, right label)
    "cerebral_white_matter": (2, 41), "cerebral_cortex": (3, 42), "lateral_ventricle": (4, 43),
    "cerebellum_white_matter": (7, 46), "cerebellum_cortex": (8, 47), "thalamus": (10, 49), "caudate": (11, 50),
    "putamen": (12, 51), "pallidum": (13, 52), "hippocampus": (17, 53), "amygdala": (18, 54),
    "accumbens": (26, 58), "ventral_dc": (28, 60),
}
GM_LABELS = (3, 42, 8, 47, 10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60)
WM_LABELS = (2, 41, 7, 46)
VENTRICLE_LABELS = (4, 43, 14, 15)
PARENCHYMA_LABELS = GM_LABELS + WM_LABELS + (16,)

# eTIV: atlas scaling factor method of Buckner et al. 2004 (doi:10.1016/j.neuroimage.2004.06.018) with
# FreeSurfer's constant for talairach.xfm. FreeSurfer computes eTIV = 1948.106 * 1000 / det(talairach.xfm) mm^3
# (mri_segstats.cpp L264-266 and utils/mri.cpp MRIestimateTIV L15887-15905, freesurfer/freesurfer @ 766ac05,
# https://github.com/freesurfer/freesurfer/blob/766ac058320a4b0417b9328ad85221db7cfbf700/mri_segstats/mri_segstats.cpp#L264-L266).
# Buckner's own constant (1755) belongs to a different atlas and is not used. The template is the mean TransMorph
# image, which is in FreeSurfer talairach (MNI305) space, so det(template -> native) plays the role of
# 1/det(talairach.xfm). The constant was calibrated for FreeSurfer's talairach.xfm, not for our affine; that transfer
# is not yet checked against an external eTIV.
ETIV_SCALE_ML = 1948.106
ETIV_METHOD = ("atlas scaling factor: eTIV = 1948.106 mL x |det(A)|, A = affine from the TransMorph/FreeSurfer "
               "talairach-space template to native space (Buckner et al. 2004; constant from FreeSurfer mri_segstats)")


def _abbr(key: str) -> str:
    """Deterministic compact key: initials of the words, side prefix kept (l_/r_)."""
    words = key.split("_")
    side = ""
    if words[0] in ("left", "right"):
        side, words = words[0][0], words[1:]
    ab = "".join(w[:2] if len(words) <= 2 else w[0] for w in words)
    return (side + ab) if side else ab


def feature_entries() -> list[dict]:
    """Every numeric leaf that the full JSON can contain: path, description, unit, compact key, precision."""
    e = []
    for label, (k, d) in STRUCTURES.items():
        e.append({"path": f"volumes_ml.{k}", "description": f"{d} volume", "unit": "mL", "label": label, "precision": 3})
    for k in PAIRS:
        e.append({"path": f"volumes_ml.total_{k}", "description": f"Left + right {k.replace('_', ' ')} volume", "unit": "mL", "precision": 3})
        e.append({"path": f"asymmetry_index_pct.{k}", "description": f"Signed asymmetry (L-R)/((L+R)/2) x 100 of the {k.replace('_', ' ')}",
                  "unit": "%", "precision": 2})
    tissue = {"gray_matter_ml": ("Total gray matter (cortical, cerebellar cortex, deep gray)", "mL", 1),
              "white_matter_ml": ("Total white matter (cerebral and cerebellar, excludes corpus callosum)", "mL", 1),
              "ventricular_csf_ml": ("Lateral, third and fourth ventricles", "mL", 2),
              "brain_parenchyma_ml": ("Gray + white matter + brain stem", "mL", 1),
              "brain_parenchymal_fraction": ("Brain parenchyma / eTIV", "ratio", 4),
              "gm_wm_volume_ratio": ("Gray matter / white matter volume", "ratio", 4),
              "ventricle_brain_ratio": ("Ventricular CSF / brain parenchyma", "ratio", 4)}
    for k, (d, u, p) in tissue.items():
        e.append({"path": f"tissue.{k}", "description": d, "unit": u, "precision": p})
    e.append({"path": "etiv_ml.value", "description": "Estimated total intracranial volume", "unit": "mL", "precision": 1})
    e.append({"path": "brain_mask_ml", "description": "HD-BET brain-mask volume", "unit": "mL", "precision": 1})
    for label, (k, d) in STRUCTURES.items():
        e.append({"path": f"normalized_volumes_pct_etiv.{k}", "description": f"{d} volume as % of eTIV", "unit": "% eTIV", "precision": 4})
    for t in ("gm", "wm", "csf_ventricular"):
        for s in ("mean", "sd", "p05", "p50", "p95"):
            e.append({"path": f"intensity.{t}.{s}", "description": f"{t} intensity {s} (N4, WM = 110)", "unit": "a.u.", "precision": 2})
    e.append({"path": "intensity.gm_wm_ratio", "description": "Mean GM / mean WM intensity", "unit": "ratio", "precision": 4})
    e.append({"path": "intensity.gm_wm_contrast_pct", "description": "(WM - GM) / mean(WM, GM) x 100", "unit": "%", "precision": 2})
    for q in ("cjv", "cnr", "snr_gm", "snr_wm", "snr_csf", "snr_total", "snrd_gm", "snrd_wm", "snrd_csf", "snrd_total",
              "efc", "fber", "wm2max", "inu_med", "inu_range"):
        e.append({"path": f"quality.iqms.{q}", "description": f"Image-quality metric {q} (MRIQC definition, docs/QC.md)", "unit": "a.u.", "precision": 4})
    for k in [s[0] for s in STRUCTURES.values()] + [f"total_{p}" for p in PAIRS] + ["gray_matter_ml", "white_matter_ml", "ventricular_csf_ml", "brain_parenchyma_ml"]:
        e.append({"path": f"normative.z.{k}", "description": f"Normative z-score of {k} (age, age^2, sex, site, eTIV; IXI train)",
                  "unit": "z", "precision": 2})
        e.append({"path": f"normative.percentile.{k}", "description": f"Normative percentile of {k}", "unit": "percentile", "precision": 1})
    seen = set()
    for i, x in enumerate(e):
        leaf = x["path"].split(".")
        group = {"volumes_ml": "v", "asymmetry_index_pct": "ai", "tissue": "t", "etiv_ml": "", "normalized_volumes_pct_etiv": "nv",
                 "intensity": "i", "quality": "q", "normative": "", "brain_mask_ml": ""}[leaf[0]]
        if leaf[0] == "normative":
            group = "z" if leaf[1] == "z" else "pc"
        name = {"etiv_ml": "etiv", "brain_mask_ml": "bm"}.get(leaf[0]) or (_abbr(leaf[-1]) if leaf[0] != "intensity" else f"{leaf[1][:3]}{leaf[-1]}" if len(leaf) == 3 else _abbr(leaf[-1]))
        ck = f"{group}.{name}" if group else name
        n = 2
        while ck in seen:
            ck = f"{group}.{name}{n}" if group else f"{name}{n}"; n += 1
        seen.add(ck)
        x["compact"] = ck
        x["obfuscated"] = f"f{i + 1:03d}"
    return e
