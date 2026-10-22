r"""Check FastSurfer outputs for the Test split before E2 uses them (Phase 3). CPU only, ~1 min.

For every Test subject in data/splits/ixi_final.csv: find aseg.auto_noCCseg.mgz (work/fastsurfer, or the 5
FastSurfer 2.5.4 outputs in results/legacy/validation_results/fastsurfer_output), load it, check that all 29
aseg structures are present with plausible volumes, and scan the per-subject Docker log for errors.
Wall times come from the ok rows of work/fastsurfer/runs.csv (failed(1) rows of the Docker-off attempt ignored).

Writes work/checks/fastsurfer_test.json. Usage: python scripts/verify_fastsurfer.py
"""
import csv
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosphere.dictionary import STRUCTURES  # noqa: E402
from neurosphere.provenance import provenance  # noqa: E402

WORK, LEGACY = ROOT / "work" / "fastsurfer", ROOT / "results" / "legacy" / "validation_results" / "fastsurfer_output"
ASEG = Path("mri") / "aseg.auto_noCCseg.mgz"
ERR = re.compile(r"Traceback|ERROR|Error:|Killed|out of memory", re.I)


def main():
    test = sorted(r["subject_id"] for r in csv.DictReader(open(ROOT / "data" / "splits" / "ixi_final.csv")) if r["split"] == "test")
    legacy = {p.name[:6]: p for p in LEGACY.glob("IXI*")}
    runs = {r["subject"]: r for r in csv.DictReader(open(WORK / "runs.csv")) if r["status"] == "ok"}
    per, missing, problems, cc_warnings = {}, [], {}, []
    for s in test:
        src = "work" if (WORK / s / ASEG).exists() else "legacy" if s in legacy and (legacy[s] / ASEG).exists() else None
        if src is None:
            missing.append(s); continue
        f = (WORK / s / ASEG) if src == "work" else (legacy[s] / ASEG)
        img = nib.load(str(f))
        lab = np.asarray(img.dataobj).astype(np.int32)
        vv = float(abs(np.linalg.det(img.affine[:3, :3]))) / 1000.0
        ids, n = np.unique(lab, return_counts=True)
        cnt = dict(zip(ids.tolist(), n.tolist()))
        vol = {int(l): cnt.get(l, 0) * vv for l in STRUCTURES}
        issues = [f"label {l} absent" for l, v in vol.items() if v == 0]
        brain = sum(vol.values())
        if not 700 < brain < 2000:
            issues.append(f"29-structure total {brain:.0f} mL outside 700-2000")
        log = WORK / f"{s}.log"
        if src == "work" and log.exists():
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            # the corpus-callosum module runs after aseg.auto_noCCseg.mgz is written; its messages do not affect that file
            cc_start = next((i for i, ln in enumerate(lines) if "CorpusCallosum/fastsurfer_cc.py" in ln), len(lines))
            hits = [ln.strip()[:200] for ln in lines if ERR.search(ln)]
            cc = [ln.strip()[:200] for ln in lines[cc_start:] if ERR.search(ln)]
            if cc:
                cc_warnings.append(s)
            if len(hits) > len(cc):
                issues.append("log: " + " | ".join([h for h in hits if h not in cc][:3]))
        per[s] = {"source": src, "shape": list(lab.shape), "voxel_ml": round(vv, 6), "total_29_ml": round(brain, 1),
                  "wall_s": float(runs[s]["wall_s"]) if s in runs else None}
        if issues:
            problems[s] = issues
    walls = np.array([v["wall_s"] for v in per.values() if v["wall_s"] is not None])
    tot = np.array([v["total_29_ml"] for v in per.values()])
    out = {"n_test": len(test), "n_with_aseg": len(per), "by_source": {k: sum(v["source"] == k for v in per.values()) for k in ("work", "legacy")},
           "missing": missing, "problems": problems, "cc_module_warnings": cc_warnings,
           "wall_s_cpu4": {"n": int(walls.size), "median": float(np.median(walls)), "min": float(walls.min()), "max": float(walls.max())},
           "total_29_ml": {"median": float(np.median(tot)), "min": float(tot.min()), "max": float(tot.max())},
           "shapes": sorted({str(v["shape"]) for v in per.values()}), "subjects": per, "provenance": provenance()}
    d = ROOT / "work" / "checks"; d.mkdir(parents=True, exist_ok=True)
    (d / "fastsurfer_test.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("subjects", "provenance")}, indent=1))


if __name__ == "__main__":
    main()
