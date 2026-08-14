"""
validate_vs_fastsurfer.py
=========================
Benchmarks our pipeline's volumetric outputs against FastSurfer on the
first 100 IXI T1 subjects.

Pipeline
--------
1. Discover the first 100 NIfTI files from ixi-t1/.
2. Run FastSurfer (Docker) on each -> parse aseg.stats for reference volumes.
3. Run our pipeline's run_feature_extraction() on each -> collect volumetric_statistics.
4. Compute MAE, MAPE, Pearson-r, ICC(2,1) for every matched structure.
5. Save results/ CSV + PNG Bland-Altman plots.

Requirements
------------
  pip install scipy pingouin matplotlib pandas nibabel ants
  docker pull deepmi/fastsurfer:latest   (needs Docker + GPU recommended)

Usage
-----
  python validate_vs_fastsurfer.py
  python validate_vs_fastsurfer.py --n 50          # run on 50 subjects
  python validate_vs_fastsurfer.py --skip-fastsurfer  # if already run
  python validate_vs_fastsurfer.py --cpu             # force CPU mode for FastSurfer
"""

import os
import sys
import json
import subprocess
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).parent.resolve()
IXI_DIR        = SCRIPT_DIR / "ixi-t1"
RESULTS_DIR    = SCRIPT_DIR / "validation_results"
FASTSURFER_OUT = RESULTS_DIR / "fastsurfer_output"
PIPELINE_OUT   = RESULTS_DIR / "pipeline_output"

RESULTS_DIR.mkdir(exist_ok=True)
FASTSURFER_OUT.mkdir(exist_ok=True)
PIPELINE_OUT.mkdir(exist_ok=True)


# ── 1. Discover subjects ────────────────────────────────────────────────────────
def discover_subjects(n: int = 100):
    """Return list of (subject_id, nii_path) for the first n IXI subjects."""
    subjects = []
    for subj_dir in sorted(IXI_DIR.iterdir()):
        if not subj_dir.is_dir():
            continue
        nii_files = list(subj_dir.glob("*.nii")) + list(subj_dir.glob("*.nii.gz"))
        if not nii_files:
            continue
        nii_path = nii_files[0]
        subjects.append((subj_dir.name, str(nii_path)))
        if len(subjects) >= n:
            break
    print(f"[discover] Found {len(subjects)} subjects (requested {n})")
    return subjects


# ── 2. FastSurfer ───────────────────────────────────────────────────────────────
def to_docker_path(p) -> str:
    """Convert a Windows absolute path to Docker-compatible (WSL-style) path."""
    p = str(p).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        p = f"/{drive}/{p[3:]}"
    return p


def check_docker() -> bool:
    """
    Verify Docker daemon is reachable before attempting any FastSurfer runs.
    Returns True if Docker is up, False otherwise.
    """
    try:
        result = subprocess.run(
            "docker info",
            shell=True, capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def ensure_fastsurfer_image() -> bool:
    """
    Check if the FastSurfer Docker image is present locally.
    If not, pull it. Returns True on success.
    """
    print("  [docker] Checking for FastSurfer image...")
    check = subprocess.run(
        "docker image inspect deepmi/fastsurfer:latest",
        shell=True, capture_output=True, text=True
    )
    if check.returncode == 0:
        print("  [docker] Image already present.")
        return True

    print("  [docker] Image not found locally. Pulling deepmi/fastsurfer:latest ...")
    print("           This may take several minutes on first run.")
    pull = subprocess.run(
        "docker pull deepmi/fastsurfer:latest",
        shell=True, text=True, timeout=600
    )
    if pull.returncode != 0:
        print("  [docker] ERROR: Failed to pull FastSurfer image.")
        print("           Make sure Docker Desktop has internet access and try again.")
        return False
    print("  [docker] Image pulled successfully.")
    return True


def _build_fastsurfer_cmd(nii_dir, nii_fname, subject_id, use_gpu: bool) -> str:
    gpu_flag = "--gpus all" if use_gpu else ""
    return (
        f"docker run --rm "
        f"--user 0:0 "          # Windows fix: map as root to avoid nonroot user error
        f"{gpu_flag} "
        f"-v \"{nii_dir}\":\"/input\" "
        f"-v \"{FASTSURFER_OUT.resolve()}\":\"/output\" "
        f"deepmi/fastsurfer:latest "
        f"--t1 /input/{nii_fname} "
        f"--sid {subject_id} "
        f"--sd /output "
        f"--seg_only "
        f"--no_cereb "
        f"--allow_root "         # required when running container as root (Windows fix)
        f"--parallel "
        f"--threads 4"
    )

# Keywords in stderr that indicate a GPU/CUDA incompatibility
_CUDA_ERROR_HINTS = [
    "cuda>=",
    "nvidia-container-cli",
    "unsatisfied condition",
    "error running prestart hook",
    "failed to create shim task",
]


def run_fastsurfer(subject_id: str, nii_path: str, use_gpu: bool = False) -> bool:
    """
    Run FastSurfer via Docker for a single subject.
    If GPU mode fails with a CUDA incompatibility error, automatically
    retries in CPU mode.
    Output lands in FASTSURFER_OUT/<subject_id>/stats/aseg.stats
    Returns True on success, False on failure.
    """
    stats_path = FASTSURFER_OUT / subject_id / "stats" / "aseg.stats"
    if stats_path.exists():
        print(f"  [fastsurfer] {subject_id}: already done, skipping.")
        return True

    nii_abs   = Path(nii_path).resolve()
    nii_dir   = nii_abs.parent
    nii_fname = nii_abs.name

    modes = ([True, False] if use_gpu else [False])  # try GPU first, then CPU
    for gpu in modes:
        mode_label = "GPU" if gpu else "CPU"
        cmd = _build_fastsurfer_cmd(nii_dir, nii_fname, subject_id, use_gpu=gpu)
        print(f"  [fastsurfer] Running ({mode_label}): {subject_id}")
        t0 = time.time()
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True,
                encoding='utf-8', errors='replace', timeout=1800
            )
            elapsed = time.time() - t0
            if result.returncode == 0:
                print(f"  [fastsurfer] Done ({subject_id}) in {elapsed:.0f}s [{mode_label}]")
                return True

            stderr_lower = result.stderr.lower()
            is_cuda_error = gpu and any(h in stderr_lower for h in _CUDA_ERROR_HINTS)
            if is_cuda_error:
                print(
                    f"  [fastsurfer] GPU failed (CUDA incompatibility). "
                    f"Auto-retrying in CPU mode..."
                )
                continue  # retry with CPU

            print(f"  [fastsurfer] FAILED ({subject_id}) in {elapsed:.0f}s [{mode_label}]")
            # Show both stdout and stderr — the real error may be in either
            combined = (result.stdout + result.stderr).strip()
            if combined:
                print(f"    output: {combined[-800:]}")
            else:
                print(f"    (no output captured — check Docker Desktop logs)")
            return False

        except subprocess.TimeoutExpired:
            print(f"  [fastsurfer] TIMEOUT ({subject_id}) [{mode_label}]")
            return False
        except Exception as e:
            print(f"  [fastsurfer] ERROR ({subject_id}): {e}")
            return False

    return False



# ── 3. Parse aseg.stats ─────────────────────────────────────────────────────────
def parse_aseg_stats(subject_id: str):
    """
    Parse FastSurfer aseg.stats and return a dict of volumes in mL.
    Returns None if the file doesn't exist or can't be parsed.
    """
    # FastSurfer --seg_only produces aseg.VINN.stats (not aseg.stats).
    # Fall back to aseg+DKT.VINN.stats if that also exists.
    stats_dir = FASTSURFER_OUT / subject_id / "stats"
    stats_file = None
    for candidate in ["aseg.VINN.stats", "aseg+DKT.VINN.stats", "aseg.stats"]:
        f = stats_dir / candidate
        if f.exists() and f.stat().st_size > 0:
            stats_file = f
            break
    if stats_file is None:
        return None

    label_to_vol_mm3 = {}
    tbv_mm3 = None

    with open(stats_file, "r") as f:
        for line in f:
            line = line.strip()
            if "BrainSegVolNotVent" in line and line.startswith("#"):
                try:
                    # FastSurfer format: # Measure X, BrainSegVolNotVent, Desc, VALUE, mm^3
                    parts_csv = line.split(",")
                    tbv_mm3 = float(parts_csv[3].strip())
                except Exception:
                    pass
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                seg_id  = int(parts[1])
                vol_mm3 = float(parts[3])
                label_to_vol_mm3[seg_id] = vol_mm3
            except (ValueError, IndexError):
                continue

    if not label_to_vol_mm3:
        return None

    mm3_to_ml = 1.0 / 1000.0
    vols = {}

    def pair(left_id, right_id, key):
        l = label_to_vol_mm3.get(left_id, 0)
        r = label_to_vol_mm3.get(right_id, 0)
        vols[f"left_{key}"]   = l * mm3_to_ml
        vols[f"right_{key}"]  = r * mm3_to_ml
        vols[f"total_{key}"]  = (l + r) * mm3_to_ml

    pair(17, 53, "hippocampus_volume_ml")
    # Lateral ventricles (main body + inferior horns)
    l_v = label_to_vol_mm3.get(4, 0) + label_to_vol_mm3.get(5, 0)
    r_v = label_to_vol_mm3.get(43, 0) + label_to_vol_mm3.get(44, 0)
    vols["left_ventricle_volume_ml"]  = l_v * mm3_to_ml
    vols["right_ventricle_volume_ml"] = r_v * mm3_to_ml
    vols["total_ventricle_volume_ml"] = (l_v + r_v) * mm3_to_ml

    pair(10, 49, "thalamus_volume_ml")
    pair(11, 50, "caudate_volume_ml")
    pair(12, 51, "putamen_volume_ml")
    pair(13, 52, "pallidum_volume_ml")
    pair(18, 54, "amygdala_volume_ml")

    if tbv_mm3 is not None:
        vols["total_brain_volume_ml"] = tbv_mm3 * mm3_to_ml
    else:
        l_wm = label_to_vol_mm3.get(2, 0) + label_to_vol_mm3.get(3, 0)
        r_wm = label_to_vol_mm3.get(41, 0) + label_to_vol_mm3.get(42, 0)
        vols["total_brain_volume_ml"] = (l_wm + r_wm) * mm3_to_ml

    return vols


# ── 4. Run our pipeline ─────────────────────────────────────────────────────────
def run_our_pipeline(subject_id: str, nii_path: str):
    """
    Call run_feature_extraction() from app_server and return volumetric_statistics.
    Caches result to PIPELINE_OUT/<subject_id>.json.
    """
    cache_file = PIPELINE_OUT / f"{subject_id}.json"
    if cache_file.exists():
        print(f"  [pipeline] {subject_id}: cached, loading.")
        with open(cache_file) as f:
            return json.load(f)

    print(f"  [pipeline] Running: {subject_id}")
    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        from app_server import run_feature_extraction

        t0 = time.time()
        result = run_feature_extraction(nii_path)
        elapsed = time.time() - t0
        print(f"  [pipeline] Done ({subject_id}) in {elapsed:.0f}s")

        vol_stats = result.get("volumetric_statistics")
        if vol_stats is None:
            print(f"  [pipeline] WARNING: no volumetric_statistics returned for {subject_id}")
            return None

        with open(cache_file, "w") as f:
            json.dump(vol_stats, f, indent=2)
        return vol_stats

    except Exception as e:
        print(f"  [pipeline] ERROR ({subject_id}): {e}")
        import traceback; traceback.print_exc()
        return None


# ── 5. Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(fs_vals: np.ndarray, our_vals: np.ndarray,
                    mae_mape_threshold: float = 10.0) -> dict:
    """Compute MAE, MAPE, Pearson-r, and ICC(2,1).

    MAE and MAPE are computed only on pairs where the FastSurfer reference
    value exceeds ``mae_mape_threshold`` (default 10 mL).  This avoids
    percentage-error blow-up on near-zero / very small structures.
    Pearson-r and ICC are computed on the full paired arrays.
    """
    from scipy.stats import pearsonr

    n = len(fs_vals)
    assert n == len(our_vals) and n > 1

    # ── Filter to reference values > threshold for MAE / MAPE ──────────────
    mask     = fs_vals > mae_mape_threshold
    fs_filt  = fs_vals[mask]
    our_filt = our_vals[mask]

    if len(fs_filt) > 0:
        mae  = float(np.mean(np.abs(our_filt - fs_filt)))
        mape = float(np.mean(np.abs(our_filt - fs_filt) / np.abs(fs_filt)) * 100)
    else:
        mae  = float("nan")
        mape = float("nan")
    r, p_r = pearsonr(fs_vals, our_vals)

    try:
        import pingouin as pg
        df = pd.DataFrame({
            "subject": list(range(n)) * 2,
            "rater":   ["fastsurfer"] * n + ["pipeline"] * n,
            "volume":  list(fs_vals) + list(our_vals),
        })
        icc_df  = pg.intraclass_corr(data=df, targets="subject", raters="rater", ratings="volume")
        icc_row = icc_df[icc_df["Type"] == "ICC2"].iloc[0]
        icc     = float(icc_row["ICC"])
        icc_ci  = (float(icc_row["CI95%"][0]), float(icc_row["CI95%"][1]))
    except Exception:
        icc, icc_ci = float("nan"), (float("nan"), float("nan"))

    bias = float(np.mean(our_vals - fs_vals))
    loa  = 1.96 * float(np.std(our_vals - fs_vals))

    return {
        "n":        n,
        "mae":      mae,
        "mape":     mape,
        "r":        float(r),
        "p_r":      float(p_r),
        "icc":      icc,
        "icc_ci_lo": icc_ci[0],
        "icc_ci_hi": icc_ci[1],
        "bias":     bias,
        "loa":      loa,
    }


# ── 6. Bland-Altman + scatter plot ──────────────────────────────────────────────
def bland_altman_plot(fs_vals: np.ndarray, our_vals: np.ndarray,
                      struct_name: str, metrics: dict, out_path: str):
    mean_vals = (fs_vals + our_vals) / 2.0
    diff_vals  = our_vals - fs_vals
    bias = metrics["bias"]
    loa  = metrics["loa"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{struct_name}  —  Pipeline vs FastSurfer", fontsize=13, fontweight="bold")

    # Left: scatter
    ax = axes[0]
    ax.scatter(fs_vals, our_vals, color="#3b82f6", alpha=0.65, s=28, edgecolors="none")
    mn = min(fs_vals.min(), our_vals.min()) * 0.92
    mx = max(fs_vals.max(), our_vals.max()) * 1.08
    ax.plot([mn, mx], [mn, mx], "k--", lw=1, label="Identity")
    ax.set_xlabel("FastSurfer (mL)", fontsize=11)
    ax.set_ylabel("Our Pipeline (mL)", fontsize=11)
    ax.set_title(f"r = {metrics['r']:.3f}   ICC = {metrics['icc']:.3f}", fontsize=10, color="#374151")
    ax.legend(fontsize=9)
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    ax.grid(True, alpha=0.3)

    # Right: Bland-Altman
    ax2 = axes[1]
    ax2.scatter(mean_vals, diff_vals, color="#10b981", alpha=0.65, s=28, edgecolors="none")
    ax2.axhline(bias,       color="#ef4444", lw=1.5, linestyle="--", label=f"Bias = {bias:+.2f} mL")
    ax2.axhline(bias + loa, color="#f59e0b", lw=1,   linestyle=":",  label=f"+LoA = {bias+loa:.2f}")
    ax2.axhline(bias - loa, color="#f59e0b", lw=1,   linestyle=":",  label=f"-LoA = {bias-loa:.2f}")
    ax2.axhline(0,           color="black",  lw=0.5)
    ax2.set_xlabel("Mean of FastSurfer & Pipeline (mL)", fontsize=11)
    ax2.set_ylabel("Pipeline − FastSurfer (mL)", fontsize=11)
    ax2.set_title(f"MAE = {metrics['mae']:.2f} mL   MAPE = {metrics['mape']:.1f}%", fontsize=10, color="#374151")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved -> {out_path}")


# ── 7. Summary bar chart ─────────────────────────────────────────────────────────
def save_summary_plot(all_metrics: dict, out_path: str):
    structs = list(all_metrics.keys())
    maes    = [all_metrics[s]["mae"]  for s in structs]
    mapes   = [all_metrics[s]["mape"] for s in structs]
    iccs    = [all_metrics[s]["icc"]  for s in structs]
    rs      = [all_metrics[s]["r"]    for s in structs]
    colors  = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#ec4899","#14b8a6","#f97316"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Pipeline vs FastSurfer — Summary", fontsize=14, fontweight="bold")

    def bar(ax, vals, title, ylabel, threshold=None):
        bars = ax.bar(range(len(structs)), vals, color=colors[:len(structs)], alpha=0.85)
        if threshold is not None:
            ax.axhline(threshold, color="red", linestyle="--", lw=1.2, label=f"Threshold {threshold}")
            ax.legend(fontsize=8)
        ax.set_xticks(range(len(structs)))
        ax.set_xticklabels([s.replace(" ", "\n") for s in structs], fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        mx_v = max(vals) if vals else 1
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01 * mx_v,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    bar(axes[0, 0], maes,  "Mean Absolute Error (MAE)",          "MAE (mL)")
    bar(axes[0, 1], mapes, "Mean Absolute Percentage Error (MAPE)", "MAPE (%)",  threshold=10)
    bar(axes[1, 0], iccs,  "Intraclass Correlation ICC(2,1)",    "ICC",         threshold=0.85)
    bar(axes[1, 1], rs,    "Pearson Correlation (r)",            "r",           threshold=0.90)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [plot] Summary chart -> {out_path}")


# ── 8. Console summary table ─────────────────────────────────────────────────────
def print_summary_table(all_metrics: dict):
    print("\n" + "=" * 84)
    print(f"{'Structure':<26}  {'N':>4}  {'MAE':>8}  {'MAPE%':>7}  {'r':>7}  {'ICC':>7}  {'ICC 95% CI':<16}  {'Bias':>8}")
    print("-" * 84)
    for struct, m in all_metrics.items():
        ci = f"[{m['icc_ci_lo']:.3f}, {m['icc_ci_hi']:.3f}]"
        print(
            f"{struct:<26}  {m['n']:>4}  {m['mae']:>8.3f}  "
            f"{m['mape']:>7.2f}  {m['r']:>7.3f}  {m['icc']:>7.3f}  {ci:<16}  {m['bias']:>+8.3f}"
        )
    print("=" * 84)


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Validate pipeline vs FastSurfer")
    parser.add_argument("--n",               type=int,  default=30,   help="Number of subjects (default 30)")
    parser.add_argument("--skip-fastsurfer", action="store_true",      help="Skip FastSurfer runs (use cached aseg.stats)")
    parser.add_argument("--skip-pipeline",   action="store_true",      help="Skip pipeline runs (use cached JSON)")
    parser.add_argument("--gpu",             action="store_true",      help="Use GPU for FastSurfer (requires CUDA >=12.8 driver)")
    args = parser.parse_args()

    use_gpu = args.gpu  # CPU is the safe default; opt-in with --gpu

    print(f"\n{'='*60}")
    print(f"  Pipeline vs FastSurfer Validation  |  N = {args.n}")
    print(f"  FastSurfer mode: {'GPU (--gpu)' if use_gpu else 'CPU (default)'}")
    print(f"{'='*60}\n")

    subjects = discover_subjects(args.n)

    # Phase 1: FastSurfer
    fs_failed = []
    if not args.skip_fastsurfer:
        # ── Docker health-check ──────────────────────────────────────────────
        if not check_docker():
            print(
                "\n" + "!" * 60 + "\n"
                "  ERROR: Docker is not running or not reachable.\n"
                "\n"
                "  FastSurfer requires Docker Desktop to be running.\n"
                "  Please start Docker Desktop and try again.\n"
                "\n"
                "  On Windows:\n"
                "    - Open Docker Desktop from the Start menu or taskbar.\n"
                "    - Wait until the whale icon shows 'Docker Desktop is running'.\n"
                "    - Then re-run this script.\n"
                "\n"
                "  If you have pre-existing FastSurfer outputs (aseg.stats files),\n"
                "  you can skip this phase entirely:\n"
                "    python validate_vs_fastsurfer.py --skip-fastsurfer\n"
                + "!" * 60
            )
            sys.exit(1)
        # ────────────────────────────────────────────────────────────────────
        # Pull the FastSurfer image if not already present
        if not ensure_fastsurfer_image():
            sys.exit(1)
        print(f"\n[Phase 1] Running FastSurfer on {len(subjects)} subjects...\n")
        for sid, nii in subjects:
            ok = run_fastsurfer(sid, nii, use_gpu=use_gpu)
            if not ok:
                fs_failed.append(sid)
        print(f"\n  FastSurfer: {len(subjects)-len(fs_failed)} succeeded, {len(fs_failed)} failed")
    else:
        print("[Phase 1] Skipping FastSurfer (--skip-fastsurfer set)")

    # Phase 2: Our pipeline
    pipeline_failed = []
    if not args.skip_pipeline:
        print(f"\n[Phase 2] Running our pipeline on {len(subjects)} subjects...\n")
        for sid, nii in subjects:
            result = run_our_pipeline(sid, nii)
            if result is None:
                pipeline_failed.append(sid)
        print(f"\n  Pipeline: {len(subjects)-len(pipeline_failed)} succeeded, {len(pipeline_failed)} failed")
    else:
        print("[Phase 2] Skipping pipeline runs (--skip-pipeline set)")

    # Phase 3: Collect paired data
    print(f"\n[Phase 3] Collecting paired results...\n")

    struct_keys = {
        "Hippocampus":        "total_hippocampus_volume_ml",
        "Lat. Ventricles":    "total_ventricle_volume_ml",
        "Thalamus":           "total_thalamus_volume_ml",
        "Caudate":            "total_caudate_volume_ml",
        "Putamen":            "total_putamen_volume_ml",
        "Pallidum":           "total_pallidum_volume_ml",
        "Amygdala":           "total_amygdala_volume_ml",
        "Total Brain Vol.":   "total_brain_volume_ml",
    }

    paired = {s: ([], []) for s in struct_keys}
    rows_csv = []

    for sid, _ in subjects:
        fs_vols  = parse_aseg_stats(sid)
        our_file = PIPELINE_OUT / f"{sid}.json"
        our_vols = json.loads(our_file.read_text()) if our_file.exists() else None

        if fs_vols is None or our_vols is None:
            continue

        row = {"subject_id": sid}
        for display, key in struct_keys.items():
            fs_v  = fs_vols.get(key)
            our_v = our_vols.get(key)
            if fs_v is not None and our_v is not None and fs_v > 0:
                paired[display][0].append(fs_v)
                paired[display][1].append(our_v)
                row[f"fs_{key}"]       = round(fs_v, 4)
                row[f"pipeline_{key}"] = round(our_v, 4)
        rows_csv.append(row)

    if not rows_csv:
        print("[ERROR] No paired data found. Make sure both FastSurfer and pipeline have been run.")
        sys.exit(1)

    csv_path = RESULTS_DIR / "per_subject_volumes.csv"
    pd.DataFrame(rows_csv).to_csv(csv_path, index=False, encoding='utf-8')
    print(f"  Per-subject CSV -> {csv_path}  ({len(rows_csv)} subjects paired)")

    # Phase 4: Metrics + plots
    print(f"\n[Phase 4] Computing metrics and generating plots...\n")
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    all_metrics = {}
    for display, (fs_list, our_list) in paired.items():
        if len(fs_list) < 3:
            print(f"  [skip] {display}: only {len(fs_list)} samples — not enough")
            continue
        fs_arr  = np.array(fs_list,  dtype=float)
        our_arr = np.array(our_list, dtype=float)
        m = compute_metrics(fs_arr, our_arr)
        all_metrics[display] = m
        safe = display.replace(" ", "_").replace(".", "").lower()
        bland_altman_plot(fs_arr, our_arr, display, m, str(plots_dir / f"{safe}.png"))

    print_summary_table(all_metrics)

    metrics_path = RESULTS_DIR / "metrics_summary.json"
    with open(metrics_path, "w", encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Metrics JSON -> {metrics_path}")

    save_summary_plot(all_metrics, str(RESULTS_DIR / "summary_metrics.png"))

    print(f"\n  All outputs saved to: {RESULTS_DIR}\n")
    print("Done.\n")


if __name__ == "__main__":
    main()
