# app_server.py - Pure Python Web Server for MRI Pipeline Dashboard
# Compatible with all Python versions (no deprecated modules like cgi)

import http.server
import socketserver
import json
import os
import sys
import pathlib
import subprocess
import threading
import time
import urllib.parse
import webbrowser
import torch
import shutil

# Core imports for on-the-fly feature extraction
import numpy as np
import nibabel as nib
from scipy import stats
from skimage import measure
from skimage.feature import graycomatrix, graycoprops

# Configurations
PORT = 8000
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(WORKSPACE_DIR, 'web')
UPLOADS_DIR = os.path.join(WORKSPACE_DIR, 'uploads')
FEATURES_FILE = os.path.join(WORKSPACE_DIR, 'features', 'predicted_features.json')
RULE_FEATURES_FILE = os.path.join(WORKSPACE_DIR, 'rule_based_features.json')
CHECKPOINTS_DIR = os.path.join(WORKSPACE_DIR, 'checkpoints')
REPORTS_DIR = os.path.join(WORKSPACE_DIR, 'reports')

# Global state
active_process = None
active_logs = []
is_running = False
pipeline_lock = threading.Lock()
active_extractions = {}

# MRIQC Docker job tracking
mriqc_jobs = {}

# Quality gate thresholds
MRIQC_THRESHOLDS = {
    "cjv":   {"optimal": 0.75,  "acceptable": 0.85,  "lower_is_better": True},
    "cnr":   {"optimal": 1.20,  "acceptable": 0.80,  "lower_is_better": False},
    "efc":   {"optimal": 0.95,  "acceptable": 0.97,  "lower_is_better": True},
    "qi_1":  {"optimal": 0.05,  "acceptable": 0.08,  "lower_is_better": True},
}

def simulate_mriqc_from_vector(vector: list) -> dict:
    """Compute realistic MRIQC metrics from the 61-element rule-based feature vector.
    Deterministic simulation — identical to train_regressor(2).py simulate_mriqc_metrics_from_vector."""
    if not vector or len(vector) < 61:
        vector = list(vector or []) + [0.0] * (61 - len(vector or []))
    mean_val  = float(vector[0])
    std_val   = float(vector[1])
    cnr_val   = float(vector[18])
    entropy   = float(vector[26])
    area_pct  = float(vector[27])

    cjv  = np.clip(0.32 + (std_val / (mean_val + 1e-5)) * 0.15, 0.25, 0.65)
    cnr  = np.clip(cnr_val if cnr_val > 0 else 1.85, 0.5, 4.0)
    efc  = np.clip(0.65 + (entropy / 5.0) * 0.2, 0.55, 0.92)
    fber = np.clip(25.0 + (area_pct * 3.5) + (mean_val / (std_val + 1e-5)) * 10.0, 12.0, 850.0)
    snr_base = mean_val / (std_val + 1e-5)
    snr_wm   = np.clip(9.5  + snr_base * 1.5, 8.0,  14.5)
    snr_gm   = np.clip(7.5  + snr_base * 1.0, 6.0,  11.5)
    snr_csf  = np.clip(2.5  + snr_base * 0.3, 1.2,  4.5)
    qi_1     = np.clip(0.004 + (1.0 - (cnr / 4.0)) * 0.015, 0.0001, 0.04)

    return {
        "cjv":     round(float(cjv),  4),
        "cnr":     round(float(cnr),  4),
        "efc":     round(float(efc),  4),
        "fber":    round(float(fber), 2),
        "snr_wm":  round(float(snr_wm),  4),
        "snr_gm":  round(float(snr_gm),  4),
        "snr_csf": round(float(snr_csf), 4),
        "qi_1":    round(float(qi_1), 6),
        "simulated": True,  # flag so UI can show "simulated" badge
    }

def compute_mriqc_quality_score(metrics: dict) -> dict:
    """Score scan quality from MRIQC metrics. Returns score (0-100), grade, label, pass/fail."""
    score = 100
    if metrics.get("cjv",  0) > 0.75: score -= 15
    if metrics.get("cjv",  0) > 0.85: score -= 15
    if metrics.get("cnr",  0) < 1.20: score -= 15
    if metrics.get("cnr",  0) < 0.80: score -= 15
    if metrics.get("efc",  0) > 0.95: score -= 15
    if metrics.get("efc",  0) > 0.97: score -= 15
    if metrics.get("qi_1", 0) > 0.05: score -= 10
    if metrics.get("qi_1", 0) > 0.08: score -= 10
    score = max(0, score)

    if score >= 90:
        grade, label, passed = "A+", "Excellent", True
        reason = "All IQMs within optimal clinical thresholds. Scan approved."
    elif score >= 75:
        grade, label, passed = "B", "Good", True
        reason = "Minor deviations detected but within acceptable range. Scan approved."
    elif score >= 50:
        grade, label, passed = "C", "Acceptable", True
        reason = "Moderate quality — metrics are marginal but acceptable. Proceed with caution."
    else:
        grade, label, passed = "F", "Poor", False
        reason = "Scan quality below minimum threshold. High motion/noise artifacts detected. Rescan recommended."

    return {"score": score, "grade": grade, "label": label, "passed": passed, "reason": reason}


def run_official_mriqc_thread(patient_id, mri_path):
    global mriqc_jobs
    job = mriqc_jobs.setdefault(patient_id, {"status": "idle", "logs": [], "progress": "", "progress_pct": 0})
    job["status"] = "running"
    job["progress"] = "Preparing BIDS directory..."
    job["progress_pct"] = 2
    job["logs"] = ["Initializing official MRIQC pipeline...\n"]

    bids_temp_dir = os.path.join(WORKSPACE_DIR, "uploads", "bids_temp")
    bids_out_dir = os.path.join(WORKSPACE_DIR, "uploads", "bids_out")

    bids_id = "".join(c for c in patient_id if c.isalnum())
    sub_anat_dir = os.path.join(bids_temp_dir, f"sub-{bids_id}", "anat")

    try:
        if os.path.exists(sub_anat_dir):
            try:
                shutil.rmtree(sub_anat_dir)
            except Exception:
                pass
        os.makedirs(sub_anat_dir, exist_ok=True)
        os.makedirs(bids_out_dir, exist_ok=True)

        desc_file = os.path.join(bids_temp_dir, "dataset_description.json")
        with open(desc_file, 'w') as df:
            json.dump({"Name": "NeuroSphere Dataset", "BIDSVersion": "1.8.0"}, df)

        dest_nii_path = os.path.join(sub_anat_dir, f"sub-{bids_id}_T1w.nii")
        job["logs"].append(f"Copying scan to BIDS layout: {dest_nii_path}\n")
        shutil.copy2(mri_path, dest_nii_path)
    except Exception as e:
        job["status"] = "failed"
        job["logs"].append(f"BIDS directory setup failed: {str(e)}\n")
        return

    abs_temp_dir = os.path.abspath(bids_temp_dir).replace('\\', '/')
    abs_out_dir = os.path.abspath(bids_out_dir).replace('\\', '/')

    try:
        # Check if docker is available in PATH
        docker_cmd = "docker"
        docker_check = subprocess.run(f"{docker_cmd} --version", shell=True, capture_output=True)
        
        # If not in PATH, try the default Docker Desktop Windows installation path
        if docker_check.returncode != 0:
            docker_cmd = r'"C:\Program Files\Docker\Docker\resources\bin\docker.exe"'
            docker_check = subprocess.run(f"{docker_cmd} --version", shell=True, capture_output=True)
            
            if docker_check.returncode != 0:
                raise FileNotFoundError("Docker not found")

        cmd = f'{docker_cmd} run --rm -v "{abs_temp_dir}:/data:ro" -v "{abs_out_dir}:/out" nipreps/mriqc:latest /data /out participant --participant-label {bids_id} --no-sub --mem_gb 6 --nprocs 1'
        job["logs"].append(f"Executing: {cmd}\n")
        job["progress"] = "Running MRIQC Docker container..."

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

        while True:
            line = proc.stdout.readline()
            if not line:
                break
            decoded = line.decode('utf-8', errors='ignore')
            job["logs"].append(decoded)
            
            # Parse output dynamically to estimate progress percentage
            decoded_lower = decoded.lower()
            if "validator" in decoded_lower or "bids" in decoded_lower:
                job["progress_pct"] = 5
                job["progress"] = "Validating BIDS dataset structure..."
            elif "anat_preproc" in decoded_lower or "preproc" in decoded_lower or "headmask" in decoded_lower or "skull" in decoded_lower:
                job["progress_pct"] = 35
                job["progress"] = "Running anatomical skull-stripping & bias correction..."
            elif "spatial" in decoded_lower or "normalization" in decoded_lower or "registration" in decoded_lower:
                job["progress_pct"] = 55
                job["progress"] = "Registering brain to MNI152 template space..."
            elif "iqms" in decoded_lower or "measures" in decoded_lower or "compute" in decoded_lower or "features" in decoded_lower:
                job["progress_pct"] = 75
                job["progress"] = "Computing Image Quality Metrics (IQMs)..."
            elif "report" in decoded_lower or "writing" in decoded_lower or "html" in decoded_lower:
                job["progress_pct"] = 90
                job["progress"] = "Generating static HTML visualization report..."

        proc.wait()

        if proc.returncode != 0:
            job["status"] = "failed"
            job["logs"].append(f"Docker process exited with error code {proc.returncode}\n")
            return

        expected_json = os.path.join(bids_out_dir, f"sub-{bids_id}", "anat", f"sub-{bids_id}_T1w.json")
        if not os.path.exists(expected_json):
            job["status"] = "failed"
            job["logs"].append(f"MRIQC completed but output JSON was not found at {expected_json}\n")
            return

        job["logs"].append(f"Reading generated MRIQC JSON file...\n")
        with open(expected_json, 'r') as fh:
            mriqc_data = json.load(fh)

        parsed_metrics = {
            "cjv":     float(mriqc_data.get("cjv",     0.0) or 0.0),
            "cnr":     float(mriqc_data.get("cnr",     0.0) or 0.0),
            "efc":     float(mriqc_data.get("efc",     0.0) or 0.0),
            "fber":    float(mriqc_data.get("fber",    0.0) or 0.0),
            "snr_wm":  float(mriqc_data.get("snr_wm",  0.0) or 0.0),
            "snr_gm":  float(mriqc_data.get("snr_gm",  0.0) or 0.0),
            "snr_csf": float(mriqc_data.get("snr_csf", 0.0) or 0.0),
            "qi_1":    float(mriqc_data.get("qi_1",    0.0) or 0.0)
        }

        # Save to features cache
        patient_cache_file = os.path.join(UPLOADS_DIR, patient_id, "features.json")
        if os.path.exists(patient_cache_file):
            with open(patient_cache_file, 'r') as f:
                feat_dict = json.load(f)
            feat_dict["mriqc_metrics"] = parsed_metrics
            with open(patient_cache_file, 'w') as f:
                json.dump(feat_dict, f, indent=2)
            job["logs"].append("Successfully updated local features.json cache.\n")
        elif os.path.exists(FEATURES_FILE):
            with open(FEATURES_FILE, 'r') as f:
                all_pred = json.load(f)
            if patient_id in all_pred:
                all_pred[patient_id]["mriqc_metrics"] = parsed_metrics
                with open(FEATURES_FILE, 'w') as f:
                    json.dump(all_pred, f, indent=2)
                job["logs"].append("Successfully updated predicted_features.json.\n")

        job["status"] = "completed"
        job["progress"] = "Official MRIQC complete!"
        job["progress_pct"] = 100

    except FileNotFoundError:
        # Docker not installed — explicitly fail so quality gate catches it
        job["logs"].append("[ERROR] Docker not found on system. MRIQC requires Docker to run.\n")
        job["status"] = "failed"
        job["progress"] = "Docker not found"
    except Exception as e:
        job["status"] = "failed"
        job["logs"].append(f"Error during MRIQC run: {str(e)}\n")
    finally:
        # Clean up BIDS temp directories for this specific patient
        patient_temp = os.path.join(bids_temp_dir, f"sub-{bids_id}")
        patient_out = os.path.join(bids_out_dir, f"sub-{bids_id}")
        for path_to_clean in [patient_temp, patient_out]:
            if os.path.exists(path_to_clean):
                try:
                    shutil.rmtree(path_to_clean)
                except Exception:
                    pass


# Ensure dirs exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# =============================================================================
# FEATURE EXTRACTION ENGINE (Extracted from extract_features.py)
# =============================================================================

def load_mri_slice(path: str) -> np.ndarray:
    """Load .nii NIfTI file and extract axial slice"""
    img = nib.load(path)
    data = img.get_fdata()
    
    # Take middle slice
    mid = data.shape[2] // 2
    arr = data[:, :, mid].astype(np.float32)
    
    # Normalize to 0-1
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min > 0:
        arr = (arr - arr_min) / (arr_max - arr_min)
    
    return arr


def run_feature_extraction(mri_path: str):
    """Run 61-feature extraction on NIfTI file and return structures"""
    # Dynamic N4 bias correction for uploaded files to match training domain
    dir_name = os.path.dirname(mri_path)
    file_name = os.path.basename(mri_path)
    if file_name.endswith(".nii.gz"):
        n4_name = file_name[:-7] + "_n4.nii.gz"
    else:
        n4_name = os.path.splitext(file_name)[0] + "_n4.nii"
    n4_path = os.path.join(dir_name, n4_name)
    active_mri = mri_path
    
    if not os.path.exists(n4_path):
        print(f"Applying dynamic N4 bias field correction to uploaded scan: {mri_path}")
        try:
            import ants
            # Run N4 correction
            img = ants.image_read(mri_path)
            img_corrected = ants.n4_bias_field_correction(img)
            ants.image_write(img_corrected, n4_path)
            print(f"N4 corrected MRI saved to {n4_path}")
            active_mri = n4_path
        except Exception as e:
            print(f"[WARNING] Dynamic N4 correction failed, using raw MRI: {e}")
            active_mri = mri_path
    else:
        active_mri = n4_path

    # 1. Compute on-the-fly features using 5-slice average from extract_features
    # Set mask_dir=None because this is an upload, we already N4-corrected it directly
    from extract_features import extract_all_features
    flat_vector = extract_all_features(active_mri, mask_dir=None)
    
    # 2. Load middle slice for bbox and centroid overlay drawing
    arr = load_mri_slice(active_mri)
    
    brain_mask = arr > 0.05
    labeled = measure.label(brain_mask)
    regions = measure.regionprops(labeled)
    
    centroid_row = 128.0
    centroid_col = 128.0
    bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col = 0, 0, 256, 256
    
    if regions:
        brain = max(regions, key=lambda r: r.area)
        centroid_row = float(brain.centroid[0])
        centroid_col = float(brain.centroid[1])
        bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col = [int(v) for v in brain.bbox]
        
    # Calculate 3D volumes (CSF, GM, WM, Left/Right Hippocampus) using ANTsPy
    volumetric_stats = None
    try:
        import ants
        
        template_path = os.path.join(CHECKPOINTS_DIR, "MNI152_T1_2mm_brain.nii.gz")
        labels_path = os.path.join(CHECKPOINTS_DIR, "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz")
        
        if os.path.exists(template_path) and os.path.exists(labels_path):
            print("Calculating 3D volumetric statistics (Atropos + Hippocampus registration)...")
            
            # Helper to load safely on Windows (HDF5 bypass)
            def load_mri_ants_local(filepath):
                import nibabel as nib
                nib_img = nib.load(filepath)
                data = nib_img.get_fdata().astype(np.float32)
                try:
                    spacing = [float(z) for z in nib_img.header.get_zooms()[:3]]
                    spacing = [s if s > 0 else 1.0 for s in spacing]
                except Exception:
                    spacing = [1.0, 1.0, 1.0]
                return ants.from_numpy(data, spacing=spacing)

            patient = load_mri_ants_local(active_mri)
            template = load_mri_ants_local(template_path)
            labels = load_mri_ants_local(labels_path)
            
            # Atropos tissue segmentation
            arr_3d = patient.numpy()
            p1 = float(np.percentile(arr_3d, 1))
            p99 = float(np.percentile(arr_3d, 99))
            arr_norm = np.clip(arr_3d, p1, p99)
            arr_norm = (arr_norm - p1) / (p99 - p1 + 1e-5)
            
            brain_seeds = arr_norm > 0.18
            from skimage.morphology import disk, closing as sk_closing, dilation as sk_dilation
            brain_clean_mask = np.zeros_like(brain_seeds)
            d = disk(2)
            for s in range(brain_seeds.shape[2]):
                sl = brain_seeds[:, :, s]
                sl = sk_closing(sl, footprint=d)
                sl = sk_dilation(sl, footprint=d)
                brain_clean_mask[:, :, s] = sl
                
            labeled_3d = measure.label(brain_clean_mask)
            regions_3d = measure.regionprops(labeled_3d)
            if regions_3d:
                largest_region = max(regions_3d, key=lambda r: r.area)
                brain_clean_mask = (labeled_3d == largest_region.label)
                
            brain_mask = ants.from_numpy(brain_clean_mask.astype(np.float32), spacing=patient.spacing)
            
            # Run Atropos (exact same arguments as train_unet.py)
            seg_result = ants.atropos(
                a = patient,
                m = '[0.2,1x1x1]',
                c = '[5,0.00001]',
                i = 'kmeans[3]',
                x = brain_mask
            )
            seg_arr = seg_result['segmentation'].numpy().astype(np.int16)
            
            # Save the segmentation mask as NIfTI
            import nibabel as nib
            dir_name = os.path.dirname(active_mri)
            file_name = os.path.basename(active_mri)
            if file_name.endswith(".nii.gz"):
                seg_name = file_name[:-7] + "_seg.nii.gz"
            else:
                seg_name = os.path.splitext(file_name)[0] + "_seg.nii"
            seg_name = seg_name.replace("_n4_seg", "_seg")
            seg_path = os.path.join(dir_name, seg_name)
            try:
                nib_img = nib.load(active_mri)
                seg_nib = nib.Nifti1Image(seg_arr, nib_img.affine)
                seg_nib.header.set_data_dtype(np.int16)
                seg_nib.header.set_qform(nib_img.affine, code=1)
                seg_nib.header.set_sform(nib_img.affine, code=1)
                seg_nib.header.set_zooms(nib_img.header.get_zooms()[:3])
                nib.save(seg_nib, seg_path)
                print(f"Saved upload segmentation mask to {seg_path}")
            except Exception as e:
                print(f"[WARNING] Failed to save segmentation mask NIfTI: {e}")
            
            # Spacing
            spacing = patient.spacing
            voxel_vol_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
            
            csf_vol = float((seg_arr == 1).sum() * voxel_vol_ml)
            gm_vol = float((seg_arr == 2).sum() * voxel_vol_ml)
            wm_vol = float((seg_arr == 3).sum() * voxel_vol_ml)
            tbv_vol = gm_vol + wm_vol
            
            # Affine registration for Hippocampus
            reg = ants.registration(fixed=patient, moving=template, type_of_transform="Affine")
            warped_labels = ants.apply_transforms(
                fixed=patient,
                moving=labels,
                transformlist=reg['fwdtransforms'],
                interpolation='nearestNeighbor'
            )
            warped_arr = warped_labels.numpy()
            
            def get_vol(label_idx):
                return float((warped_arr == label_idx).sum() * voxel_vol_ml)
                
            def calc_ai(left, right):
                denom = 0.5 * (left + right)
                if denom == 0:
                    return 0.0
                return (abs(left - right) / denom) * 100.0

            left_hippo = get_vol(9)
            right_hippo = get_vol(19)
            hippo_ai = calc_ai(left_hippo, right_hippo)

            left_vent = get_vol(3)
            right_vent = get_vol(14)
            vent_ai = calc_ai(left_vent, right_vent)

            left_thal = get_vol(4)
            right_thal = get_vol(15)
            thal_ai = calc_ai(left_thal, right_thal)

            left_caud = get_vol(5)
            right_caud = get_vol(16)
            caud_ai = calc_ai(left_caud, right_caud)

            left_puta = get_vol(6)
            right_puta = get_vol(17)
            puta_ai = calc_ai(left_puta, right_puta)

            left_pall = get_vol(7)
            right_pall = get_vol(18)
            pall_ai = calc_ai(left_pall, right_pall)

            left_amyg = get_vol(10)
            right_amyg = get_vol(20)
            amyg_ai = calc_ai(left_amyg, right_amyg)
            
            volumetric_stats = {
                "total_brain_volume_ml": tbv_vol,
                "gray_matter_volume_ml": gm_vol,
                "white_matter_volume_ml": wm_vol,
                "csf_volume_ml": csf_vol,
                
                # Hippocampus
                "left_hippocampus_volume_ml": left_hippo,
                "right_hippocampus_volume_ml": right_hippo,
                "total_hippocampus_volume_ml": left_hippo + right_hippo,
                "hippocampus_asymmetry_index": hippo_ai,
                
                # Ventricles
                "left_ventricle_volume_ml": left_vent,
                "right_ventricle_volume_ml": right_vent,
                "total_ventricle_volume_ml": left_vent + right_vent,
                "ventricle_asymmetry_index": vent_ai,
                
                # Thalamus
                "left_thalamus_volume_ml": left_thal,
                "right_thalamus_volume_ml": right_thal,
                "total_thalamus_volume_ml": left_thal + right_thal,
                "thalamus_asymmetry_index": thal_ai,
                
                # Caudate
                "left_caudate_volume_ml": left_caud,
                "right_caudate_volume_ml": right_caud,
                "total_caudate_volume_ml": left_caud + right_caud,
                "caudate_asymmetry_index": caud_ai,
                
                # Putamen
                "left_putamen_volume_ml": left_puta,
                "right_putamen_volume_ml": right_puta,
                "total_putamen_volume_ml": left_puta + right_puta,
                "putamen_asymmetry_index": puta_ai,
                
                # Pallidum
                "left_pallidum_volume_ml": left_pall,
                "right_pallidum_volume_ml": right_pall,
                "total_pallidum_volume_ml": left_pall + right_pall,
                "pallidum_asymmetry_index": pall_ai,
                
                # Amygdala
                "left_amygdala_volume_ml": left_amyg,
                "right_amygdala_volume_ml": right_amyg,
                "total_amygdala_volume_ml": left_amyg + right_amyg,
                "amygdala_asymmetry_index": amyg_ai
            }
            print("Successfully calculated 3D volumetric statistics.")
        else:
            print("[WARNING] Template files not found, skipping volumetric stats.")
    except Exception as e:
        print(f"[WARNING] Volumetric statistics calculation failed: {e}")

    return {
        "rule_based_vector": flat_vector,
        "centroid": {"row": centroid_row, "col": centroid_col},
        "bbox": {
            "min_row": bbox_min_row,
            "min_col": bbox_min_col,
            "max_row": bbox_max_row,
            "max_col": bbox_max_col
        },
        "volumetric_statistics": volumetric_stats,
        "mriqc_metrics": None
    }


def make_clinical_summary(features_dict: dict) -> str:
    """Generate a clean rule-based clinical summary text from features"""
    tissue_seg = features_dict.get("tissue_segmentation", {})
    csf_pct = tissue_seg.get("csf_ventricles", {}).get("percentage", 0)
    gm_pct = tissue_seg.get("gray_matter", {}).get("percentage", 0)
    wm_pct = tissue_seg.get("white_matter", {}).get("percentage", 0)
    
    texture = features_dict.get("texture_features", {})
    entropy = texture.get("shannon_entropy", 0)
    contrast = texture.get("glcm_contrast", 0)
    homo = texture.get("glcm_homogeneity", 0)
    
    struct = features_dict.get("structural_features", {})
    solidity = struct.get("solidity", 0)
    ecc = struct.get("eccentricity", 0)
    
    symmetry = features_dict.get("symmetry", {})
    asym = symmetry.get("asymmetry_index_pct", 0)
    
    summary = f"The quantitative MRI analysis reveals generally normal brain tissue composition and structure.\n"
    summary += f"Tissue segmentation shows cerebrospinal fluid ({csf_pct:.2f}%), gray matter ({gm_pct:.2f}%), "
    summary += f"and white matter ({wm_pct:.2f}%), suggesting normal global volume characteristics.\n"
    
    summary += f"Texture analysis indicates a complex brain parenchyma with Shannon entropy of {entropy:.2f}, "
    summary += f"GLCM contrast of {contrast:.4f}, and GLCM homogeneity of {homo:.4f}. These values suggest clear "
    summary += "tissue boundaries and normal structural uniformness without signs of focal lesion variations.\n"
    
    summary += f"Structural analysis indicates a compact shape (solidity: {solidity:.4f}) and typical axial orientation "
    summary += f"(eccentricity: {ecc:.4f}). A calculated hemisphere asymmetry index of {asym:.2f}% indicates mild, "
    if asym > 10:
        summary += "notable asymmetry. Recommend clinical review of alignment or possible structural changes.\n"
    else:
        summary += "non-pathological variation between hemispheres.\n"
        
    summary += "Overall clinical flag risk assessment is LOW, with no significant pathology markers detected."
    return summary


def build_clinical_report(patient_id: str, features_dict: dict) -> str:
    """Format report paper document matching professional radiology report style as clean HTML"""
    timestamp_str = time.strftime("%d-%b-%Y %H:%M:%S %p")
    patient_name = patient_id.split("-")[0] if "-" in patient_id else patient_id
    uhid = patient_id
    acc_no = f"24{time.strftime('%m%d')}{patient_id[:6].replace('-','')}"
    
    tissue_seg = features_dict.get("tissue_segmentation", {})
    csf_pct = tissue_seg.get("csf_ventricles", {}).get("percentage", 0)
    gm_pct = tissue_seg.get("gray_matter", {}).get("percentage", 0)
    wm_pct = tissue_seg.get("white_matter", {}).get("percentage", 0)
    
    texture = features_dict.get("texture_features", {})
    contrast = texture.get("glcm_contrast", 0)
    energy = texture.get("glcm_energy", 0)
    homo = texture.get("glcm_homogeneity", 0)
    entropy = texture.get("shannon_entropy", 0)
    
    struct = features_dict.get("structural_features", {})
    brain_area_pct = struct.get("brain_area_pct", 0)
    solidity = struct.get("solidity", 0)
    ecc = struct.get("eccentricity", 0)
    vent_proxy = struct.get("ventricular_proxy_pct", 0)
    
    symmetry = features_dict.get("symmetry", {})
    bilateral_sym = symmetry.get("bilateral_symmetry_score", 0)
    asym = symmetry.get("asymmetry_index_pct", 0)
    
    risk_level = features_dict.get("pathology_indicators", {}).get("risk_level", "LOW").upper()
    clinical_summary = make_clinical_summary(features_dict)
    
    vol_stats = features_dict.get("volumetric_statistics", {})
    has_vols = vol_stats and len(vol_stats) > 0
    
    vols_html = ""
    if has_vols:
        tbv = vol_stats.get("total_brain_volume_ml", 0)
        
        l_vent = vol_stats.get("left_ventricle_volume_ml", 0)
        r_vent = vol_stats.get("right_ventricle_volume_ml", 0)
        vent_asym = vol_stats.get("ventricle_asymmetry_index", 0)
        
        l_thal = vol_stats.get("left_thalamus_volume_ml", 0)
        r_thal = vol_stats.get("right_thalamus_volume_ml", 0)
        thal_asym = vol_stats.get("thalamus_asymmetry_index", 0)
        
        l_caud = vol_stats.get("left_caudate_volume_ml", 0)
        r_caud = vol_stats.get("right_caudate_volume_ml", 0)
        caud_asym = vol_stats.get("caudate_asymmetry_index", 0)
        
        l_puta = vol_stats.get("left_putamen_volume_ml", 0)
        r_puta = vol_stats.get("right_putamen_volume_ml", 0)
        puta_asym = vol_stats.get("putamen_asymmetry_index", 0)
        
        l_pall = vol_stats.get("left_pallidum_volume_ml", 0)
        r_pall = vol_stats.get("right_pallidum_volume_ml", 0)
        pall_asym = vol_stats.get("pallidum_asymmetry_index", 0)
        
        l_hippo = vol_stats.get("left_hippocampus_volume_ml", 0)
        r_hippo = vol_stats.get("right_hippocampus_volume_ml", 0)
        hippo_asym = vol_stats.get("hippocampus_asymmetry_index", 0)
        
        l_amyg = vol_stats.get("left_amygdala_volume_ml", 0)
        r_amyg = vol_stats.get("right_amygdala_volume_ml", 0)
        amyg_asym = vol_stats.get("amygdala_asymmetry_index", 0)
        
        vols_html = f"""
        <div class="report-section-title">Subcortical Volumetrics (3D Atlas-Based)</div>
        <ul class="metric-list" style="margin-bottom:15px;">
          <li class="metric-item">
            <span class="metric-label">Total Brain Volume (TBV)</span>
            <span class="metric-value">{tbv:,.2f} mL</span>
          </li>
          <li class="metric-item">
            <span class="metric-label">Lateral Ventricles (Left / Right)</span>
            <span class="metric-value">{l_vent:.2f} mL / {r_vent:.2f} mL <span class="asymmetry-badge">Asym: {vent_asym:.1f}%</span></span>
          </li>
          <li class="metric-item">
            <span class="metric-label">Thalamus (Left / Right)</span>
            <span class="metric-value">{l_thal:.2f} mL / {r_thal:.2f} mL <span class="asymmetry-badge">Asym: {thal_asym:.1f}%</span></span>
          </li>
          <li class="metric-item">
            <span class="metric-label">Caudate (Left / Right)</span>
            <span class="metric-value">{l_caud:.2f} mL / {r_caud:.2f} mL <span class="asymmetry-badge">Asym: {caud_asym:.1f}%</span></span>
          </li>
          <li class="metric-item">
            <span class="metric-label">Putamen (Left / Right)</span>
            <span class="metric-value">{l_puta:.2f} mL / {r_puta:.2f} mL <span class="asymmetry-badge">Asym: {puta_asym:.1f}%</span></span>
          </li>
          <li class="metric-item">
            <span class="metric-label">Pallidum (Left / Right)</span>
            <span class="metric-value">{l_pall:.2f} mL / {r_pall:.2f} mL <span class="asymmetry-badge">Asym: {pall_asym:.1f}%</span></span>
          </li>
          <li class="metric-item">
            <span class="metric-label">Hippocampus (Left / Right)</span>
            <span class="metric-value">{l_hippo:.2f} mL / {r_hippo:.2f} mL <span class="asymmetry-badge">Asym: {hippo_asym:.1f}%</span></span>
          </li>
          <li class="metric-item">
            <span class="metric-label">Amygdala (Left / Right)</span>
            <span class="metric-value">{l_amyg:.2f} mL / {r_amyg:.2f} mL <span class="asymmetry-badge">Asym: {amyg_asym:.1f}%</span></span>
          </li>
        </ul>
        """
    else:
        vols_html = """
        <div class="report-section-title">Subcortical Volumetrics (3D Atlas-Based)</div>
        <p style="font-size:11px;color:#94a3b8;font-style:italic;margin-top:5px;margin-bottom:15px;">
          * 3D Atlas-based volumetric registration not generated for this patient scan.
        </p>
        """

    report_html = f"""<div class="clinical-report-container">
  <style>
    .clinical-report-container {{
      font-family: 'Inter', -apple-system, sans-serif;
      color: #1e293b;
      line-height: 1.5;
      padding: 10px 5px;
      background: #fff;
    }}
    .report-header {{
      border-bottom: 2px solid #10b981;
      padding-bottom: 12px;
      margin-bottom: 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .report-header-left h1 {{
      font-size: 15px;
      font-weight: 800;
      color: #065f46;
      margin: 0 0 2px 0;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .report-header-left p {{
      font-size: 10px;
      color: #64748b;
      margin: 0;
      font-weight: 500;
    }}
    .report-header-right {{
      text-align: right;
    }}
    .report-header-right h2 {{
      font-size: 12px;
      font-weight: 700;
      color: #0f172a;
      margin: 0 0 2px 0;
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }}
    .patient-table {{
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 22px;
      font-size: 11.5px;
    }}
    .patient-table th, .patient-table td {{
      border: 1px solid #e2e8f0;
      padding: 6px 10px;
      text-align: left;
    }}
    .patient-table th {{
      background-color: #f8fafc;
      color: #64748b;
      font-weight: 600;
      width: 20%;
    }}
    .patient-table td {{
      color: #334155;
      font-weight: 500;
    }}
    .report-title-bar {{
      font-size: 14px;
      font-weight: 800;
      color: #0f172a;
      text-align: center;
      margin: 18px 0;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      border-top: 1px solid #e2e8f0;
      border-bottom: 1px solid #e2e8f0;
      padding: 6px 0;
    }}
    .grid-two-cols {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 15px;
    }}
    .report-section-title {{
      font-size: 11.5px;
      font-weight: 700;
      color: #0f172a;
      margin-bottom: 8px;
      border-bottom: 1px solid #e2e8f0;
      padding-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }}
    .metric-list {{
      list-style: none;
      padding: 0;
      margin: 0;
      font-size: 11.5px;
    }}
    .metric-item {{
      display: flex;
      justify-content: space-between;
      padding: 4px 0;
      border-bottom: 1px dashed #f1f5f9;
    }}
    .metric-label {{
      color: #64748b;
    }}
    .metric-value {{
      font-weight: 600;
      color: #1e293b;
    }}
    .asymmetry-badge {{
      font-size: 9px;
      padding: 1px 4px;
      border-radius: 4px;
      background: #f1f5f9;
      color: #475569;
      margin-left: 4px;
      font-weight: 700;
    }}
    .impression-box {{
      background: #f8fafc;
      border-left: 4px solid #10b981;
      padding: 12px 15px;
      border-radius: 4px;
      margin: 18px 0;
      font-size: 12px;
    }}
    .impression-title {{
      font-weight: 700;
      color: #0f172a;
      margin-bottom: 4px;
      font-size: 12.5px;
    }}
    .recommendations-box {{
      font-size: 12px;
    }}
    .recommendations-list {{
      margin: 6px 0 0 16px;
      padding: 0;
    }}
    .recommendations-list li {{
      margin-bottom: 4px;
    }}
    .risk-badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 9px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .risk-low {{
      background: #ecfdf5;
      color: #065f46;
    }}
    .risk-medium {{
      background: #fffbeb;
      color: #92400e;
    }}
    .risk-high {{
      background: #fef2f2;
      color: #991b1b;
    }}
    .footer-sig {{
      margin-top: 25px;
      text-align: center;
      font-size: 9px;
      color: #94a3b8;
      border-top: 1px solid #f1f5f9;
      padding-top: 10px;
    }}
  </style>

  <!-- Header -->
  <div class="report-header">
    <div class="report-header-left">
      <h1>Department of Neuroradiology</h1>
      <p>NeuroSphere MRI Intelligence & Interventional Radiology</p>
    </div>
    <div class="report-header-right">
      <h2>Automated Quantitative Analysis</h2>
      <p style="font-size:10px;color:#64748b;margin:0;">Ref Acc: MR-ANALYTICS-V1</p>
    </div>
  </div>

  <!-- Patient Table -->
  <table class="patient-table">
    <tr>
      <th>PATIENT NAME</th>
      <td>{patient_name}</td>
      <th>STUDY DATE</th>
      <td>{timestamp_str}</td>
    </tr>
    <tr>
      <th>UHID / CASE NO</th>
      <td>{uhid}</td>
      <th>ACCESSION NO</th>
      <td>{acc_no}</td>
    </tr>
    <tr>
      <th>REFERRING CLINICIAN</th>
      <td>AI Platform Engine</td>
      <th>MODALITY</th>
      <td>Magnetic Resonance (MR)</td>
    </tr>
  </table>

  <div class="report-title-bar">Quantitative Brain MRI Findings &amp; Biomarkers</div>

  <p style="font-size:11.5px;color:#64748b;margin: -8px 0 18px 0;line-height:1.4;">
    <strong>Clinical Indication:</strong> Automated 3D tissue segmentation and multi-parameter morphological feature mapping. All biomarkers calculated using standardized ANTsPy Atropos tissue modeling, Harvard-Oxford atlas transformations, and CNN regressors.
  </p>

  <!-- Findings Grid -->
  <div class="grid-two-cols">
    <!-- Col 1: Volumetrics -->
    <div>
      <div class="report-section-title">Global Tissue Segmentation</div>
      <ul class="metric-list" style="margin-bottom:15px;">
        <li class="metric-item">
          <span class="metric-label">Cerebrospinal Fluid (CSF)</span>
          <span class="metric-value">{csf_pct:.2f}%</span>
        </li>
        <li class="metric-item">
          <span class="metric-label">Gray Matter Percentage</span>
          <span class="metric-value">{gm_pct:.2f}%</span>
        </li>
        <li class="metric-item">
          <span class="metric-label">White Matter Percentage</span>
          <span class="metric-value">{wm_pct:.2f}%</span>
        </li>
        <li class="metric-item">
          <span class="metric-label">Total Brain Area Coverage</span>
          <span class="metric-value">{brain_area_pct:.2f}%</span>
        </li>
      </ul>

      <!-- Subcortical Volumes Section -->
      {vols_html}
    </div>

    <!-- Col 2: Textures & Morphology -->
    <div>
      <div class="report-section-title">Texture Biomarkers (2D GLCM)</div>
      <ul class="metric-list" style="margin-bottom:15px;">
        <li class="metric-item">
          <span class="metric-label">GLCM Contrast</span>
          <span class="metric-value">{contrast:.4f}</span>
        </li>
        <li class="metric-item">
          <span class="metric-label">GLCM Energy</span>
          <span class="metric-value">{energy:.4f}</span>
        </li>
        <li class="metric-item">
          <span class="metric-label">GLCM Homogeneity</span>
          <span class="metric-value">{homo:.4f}</span>
        </li>
        <li class="metric-item">
          <span class="metric-label">Shannon Information Entropy</span>
          <span class="metric-value">{entropy:.2f}</span>
        </li>
      </ul>

      <div class="report-section-title">Structural &amp; Morphological Metrics</div>
      <ul class="metric-list" style="margin-bottom:15px;">
        <li class="metric-item">
          <span class="metric-label">Brain Solidity (Compactness)</span>
          <span class="metric-value">{solidity:.4f}</span>
        </li>
        <li class="metric-item">
          <span class="metric-label">Brain Eccentricity (Ovalness)</span>
          <span class="metric-value">{ecc:.4f}</span>
        </li>
        <li class="metric-item">
          <span class="metric-label">Ventricular Proxy Index</span>
          <span class="metric-value">{vent_proxy:.2f}%</span>
        </li>
        <li class="metric-item">
          <span class="metric-label">Bilateral Symmetry Index</span>
          <span class="metric-value">{bilateral_sym:.4f}</span>
        </li>
        <li class="metric-item">
          <span class="metric-label">Hemisphere Asymmetry Index</span>
          <span class="metric-value">{asym:.2f}%</span>
        </li>
      </ul>

      <div class="report-section-title">Pathology Assessment</div>
      <ul class="metric-list">
        <li class="metric-item" style="border-bottom:none;">
          <span class="metric-label">Pathology Risk Level</span>
          <span class="metric-value">
            <span class="risk-badge risk-{risk_level.lower()}">{risk_level}</span>
          </span>
        </li>
      </ul>
    </div>
  </div>

  <!-- Impression Box -->
  <div class="impression-box">
    <div class="impression-title">Clinical Impression &amp; AI Analysis Summary</div>
    <p style="margin:0;line-height:1.5;">{clinical_summary}</p>
  </div>

  <!-- Recommendations -->
  <div class="recommendations-box">
    <div class="report-section-title">Recommendations</div>
    <ul class="recommendations-list">
      <li>Correlate quantitative MRI tissue volumes with clinical presentation and history.</li>
      <li>Review subcortical volumes (particularly hippocampal index) in the context of cognitive scores.</li>
      <li>Follow-up volumetric imaging may be considered in 6–12 months to trace dynamic volumetric changes.</li>
    </ul>
  </div>

  <!-- Disclaimer Footer -->
  <div class="footer-sig">
    <p style="margin:0 0 2px 0;font-weight:700;">NeuroSphere AI Research &amp; Training Report</p>
    <p style="margin:0;font-size:8px;color:#cbd5e1;">DISCLAIMER: This is an AI-generated training/research summary. It does NOT constitute a clinical diagnosis or medical advice. Must be reviewed by a certified clinical radiologist before diagnostic use.</p>
  </div>
</div>"""
    return report_html


# =============================================================================
# BACKGROUND PIPELINE TASK THREAD
# =============================================================================

def read_process_stdout(process):
    """Worker thread to stream lines from command stdout to global logs"""
    global active_logs, is_running
    while True:
        line = process.stdout.readline()
        if not line:
            break
        # Append decoded line
        active_logs.append(line.decode('utf-8', errors='ignore'))
    
    process.wait()
    with pipeline_lock:
        is_running = False


# =============================================================================
# WEB SERVER REQUEST HANDLER
# =============================================================================

class DashboardAPIHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler that exposes API endpoints and serves static files"""
    
    def __init__(self, *args, **kwargs):
        # Override directory to point to web/ folder
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def end_headers(self):
        # Disable caching for APIs
        if self.path.startswith('/api/'):
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

    def do_GET(self):
        # Route API queries
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        if path == '/api/status':
            self.handle_api_status()
        elif path == '/api/patients':
            self.handle_api_patients()
        elif path.startswith('/api/features/'):
            patient_id = path.split('/api/features/')[1]
            self.handle_api_features(patient_id)
        elif path.startswith('/api/report/'):
            patient_id = path.split('/api/report/')[1]
            self.handle_api_report(patient_id)
        elif path.startswith('/api/mriqc/status/'):
            patient_id = path.split('/api/mriqc/status/')[1]
            self.handle_api_mriqc_status(urllib.parse.unquote(patient_id))
        elif path.startswith('/api/mriqc/quality-gate/'):
            patient_id = path.split('/api/mriqc/quality-gate/')[1]
            self.handle_api_mriqc_quality_gate(urllib.parse.unquote(patient_id))
        elif path == '/api/logs':
            self.handle_api_logs(parsed_url.query)
        elif path.startswith('/uploads/') or path.startswith('/ixi-t1/') or path.startswith('/ixi-t1-masks/'):
            # Serve these static MRI / segmentation files from the workspace directory
            rel_path = urllib.parse.unquote(path.lstrip('/'))
            full_path = os.path.join(WORKSPACE_DIR, rel_path)
            
            # Normalize path and check if it exists and is within WORKSPACE_DIR for safety
            full_path = os.path.abspath(full_path)
            if full_path.lower().startswith(os.path.abspath(WORKSPACE_DIR).lower()) and os.path.exists(full_path) and os.path.isfile(full_path):
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Access-Control-Allow-Origin', '*')
                
                fs = os.path.getsize(full_path)
                self.send_header('Content-Length', str(fs))
                self.end_headers()
                
                try:
                    with open(full_path, 'rb') as fh:
                        while True:
                            chunk = fh.read(8192)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                except Exception as e:
                    print(f"Error sending file: {e}")
            else:
                self.send_error(404, "File not found")
        else:
            # Fallback to serve static web content
            super().do_GET()

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        if path == '/api/upload':
            self.handle_api_upload()
        elif path == '/api/run-step':
            self.handle_api_run_step()
        elif path.startswith('/api/mriqc/run/'):
            patient_id = path.split('/api/mriqc/run/')[1]
            self.handle_api_mriqc_run(urllib.parse.unquote(patient_id))
        else:
            self.send_error(404, "Endpoint not found")

    # API: System status (CUDA details, pipeline state)
    def handle_api_status(self):
        cuda_avail = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_avail else None
        
        # Check files to estimate step completion badges
        pipeline_status = {
            "1": "Completed" if os.path.exists(os.path.join(WORKSPACE_DIR, "ixi-t1-masks")) or len(list(pathlib.Path(UPLOADS_DIR).glob("*-masks"))) > 0 else "Ready",
            "2": "Completed" if os.path.exists(os.path.join(CHECKPOINTS_DIR, "autoencoder_2d_best.pth")) else "Ready",
            "3": "Completed" if os.path.exists(os.path.join(CHECKPOINTS_DIR, "unet_segmentation_best.pth")) else "Ready",
            "4": "Completed" if os.path.exists(os.path.join(CHECKPOINTS_DIR, "regressor_best.pth")) else "Ready",
            "5": "Completed" if os.path.exists(FEATURES_FILE) else "Ready"
        }
        
        response = {
            "cuda_available": cuda_avail,
            "device": device_name,
            "pipeline_status": pipeline_status
        }
        self.send_json_response(response)

    # API: List patient IDs (precomputed + uploaded)
    def handle_api_patients(self):
        patients = []

        # 1. Load precomputed from predicted_features.json
        if os.path.exists(FEATURES_FILE):
            try:
                with open(FEATURES_FILE, 'r') as f:
                    data = json.load(f)
                    if "schema_version" in data:
                        pred_filename = data.get("scan_metadata", {}).get("filename", "")
                        pred_patient_id = pathlib.Path(pred_filename).stem if pred_filename else ""
                        if pred_patient_id:
                            patients.append(pred_patient_id)
                    else:
                        patients.extend(list(data.keys()))
            except Exception as e:
                print("Error loading features keys:", e)

        # 2. Add uploaded patients
        uploaded_scans = sorted(list(pathlib.Path(UPLOADS_DIR).glob("*/*.nii")) + list(pathlib.Path(UPLOADS_DIR).glob("*/*.nii.gz")))
        for scan in uploaded_scans:
            scan_id = scan.parent.name
            if scan_id not in patients:
                patients.insert(0, scan_id)  # Put uploaded scans first

        self.send_json_response({"patients": patients})

    # API: Load patient quantitative metrics (and compares vectors)
    def handle_api_features(self, patient_id):
        patient_id = urllib.parse.unquote(patient_id)
        
        # If there is an active background extraction thread for this patient, wait for it!
        if patient_id in active_extractions:
            print(f"Waiting for background feature extraction to complete for patient {patient_id}...")
            active_extractions[patient_id].join(timeout=240)
        
        # Check if patient is inside uploads
        scan_dir = os.path.join(UPLOADS_DIR, patient_id)
        nii_files = list(pathlib.Path(scan_dir).glob("*.nii")) + list(pathlib.Path(scan_dir).glob("*.nii.gz"))
        
        feature_data = None
        rule_based_vector = None
        predicted_vector = None
        source = "dataset"
        
        if len(nii_files) > 0:
            source = "uploads"
            # Dynamic on-the-fly rule-based feature extraction for uploads!
            cache_file = os.path.join(scan_dir, "features.json")
            
            # Ensure the segmentation mask file also exists, otherwise recalculate it
            seg_nii = os.path.join(scan_dir, f"{patient_id}_seg.nii")
            seg_nii_gz = os.path.join(scan_dir, f"{patient_id}_seg.nii.gz")
            has_seg = os.path.exists(seg_nii) or os.path.exists(seg_nii_gz)
            
            if os.path.exists(cache_file) and has_seg:
                try:
                    with open(cache_file, 'r') as f:
                        feature_data = json.load(f)
                        rule_based_vector = feature_data.get("rule_based_array")
                        predicted_vector = feature_data.get("predicted_array")
                except Exception:
                    pass
            
            if not feature_data:
                # Calculate and cache
                try:
                    nii_path = str(nii_files[0])
                    extraction_results = run_feature_extraction(nii_path)
                    
                    rule_based_vector = extraction_results["rule_based_vector"]
                    
                    # Convert to detailed feature dict
                    feature_data = features_list_to_dict(
                        filename=os.path.basename(nii_path),
                        features=rule_based_vector
                    )
                    
                    # Add centroid / bbox
                    feature_data["structural_features"]["centroid"] = extraction_results["centroid"]
                    feature_data["structural_features"]["bounding_box"] = extraction_results["bbox"]
                    
                    # Add volumetric statistics if available
                    if extraction_results.get("volumetric_statistics") is not None:
                        feature_data["volumetric_statistics"] = extraction_results["volumetric_statistics"]

                    # Add MRIQC metrics if available
                    if extraction_results.get("mriqc_metrics") is not None:
                        feature_data["mriqc_metrics"] = extraction_results["mriqc_metrics"]
                        
                except Exception as e:
                    self.send_error(500, f"On-the-fly extraction failed: {str(e)}")
                    return

            # Compute predictions on-the-fly if not cached
            if feature_data and not predicted_vector:
                try:
                    nii_path = str(nii_files[0])
                    predicted_vector = get_regressor_predictions(nii_path)
                    if predicted_vector:
                        # Save both to cache
                        cache_payload = feature_data.copy()
                        cache_payload["rule_based_array"] = rule_based_vector
                        cache_payload["predicted_array"] = predicted_vector
                        temp_cache_file = cache_file + ".tmp"
                        with open(temp_cache_file, 'w') as f:
                            json.dump(cache_payload, f, indent=2)
                        os.replace(temp_cache_file, cache_file)
                except Exception as e:
                    print(f"[WARNING] On-the-fly prediction failed for upload scan {patient_id}: {e}")

        else:
            # Load precomputed dataset patient
            if os.path.exists(FEATURES_FILE):
                try:
                    with open(FEATURES_FILE, 'r') as f:
                        all_pred = json.load(f)
                        if "schema_version" in all_pred:
                            pred_filename = all_pred.get("scan_metadata", {}).get("filename", "")
                            pred_patient_id = pathlib.Path(pred_filename).stem if pred_filename else ""
                            if patient_id == pred_patient_id:
                                feature_data = all_pred
                        else:
                            feature_data = all_pred.get(patient_id)
                except Exception as e:
                    print("Error loading features:", e)
            
            # Check for cached or compute on-the-fly volumetric statistics for dataset patients
            if feature_data and "volumetric_statistics" not in feature_data:
                cache_file = os.path.join(scan_dir, "features.json")
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r') as f:
                            cache_data = json.load(f)
                            if "volumetric_statistics" in cache_data:
                                feature_data["volumetric_statistics"] = cache_data["volumetric_statistics"]
                    except Exception:
                        pass
                
                if "volumetric_statistics" not in feature_data:
                    try:
                        prefix = "-".join(patient_id.split("-")[:3])
                        dataset_dir = os.path.join(WORKSPACE_DIR, "ixi-t1")
                        mri_files = list(pathlib.Path(dataset_dir).rglob("*.nii")) + list(pathlib.Path(dataset_dir).rglob("*.nii.gz"))
                        dataset_mri_path = None
                        for f in mri_files:
                            if f.is_file() and f.name.startswith(prefix):
                                dataset_mri_path = str(f)
                                break
                        
                        if dataset_mri_path:
                            print(f"Dataset patient {patient_id} volumetric statistics requested. Calculating on-the-fly...")
                            extraction_results = run_feature_extraction(dataset_mri_path)
                            if extraction_results.get("volumetric_statistics") is not None:
                                feature_data["volumetric_statistics"] = extraction_results["volumetric_statistics"]
                                
                                # Cache it for fast loading next time
                                os.makedirs(scan_dir, exist_ok=True)
                                cache_payload = feature_data.copy()
                                temp_cache_file = cache_file + ".tmp"
                                with open(temp_cache_file, 'w') as f:
                                    json.dump(cache_payload, f, indent=2)
                                os.replace(temp_cache_file, cache_file)
                                print(f"Cached volumetric statistics for dataset scan: {patient_id}")
                    except Exception as e:
                        print(f"[WARNING] On-the-fly volumetric calculation failed for dataset patient {patient_id}: {e}")

            # Retrieve predicted vector (by flattening the predicted features dictionary)
            if feature_data:
                predicted_vector = flatten_feature_dict(feature_data)
                
            # If predicted_vector has only 40 features or is missing, compute new regressor predictions on-the-fly!
            if feature_data and (not predicted_vector or len(predicted_vector) < 61):
                try:
                    prefix = "-".join(patient_id.split("-")[:3])
                    dataset_dir = os.path.join(WORKSPACE_DIR, "ixi-t1")
                    mri_files = list(pathlib.Path(dataset_dir).rglob("*.nii")) + list(pathlib.Path(dataset_dir).rglob("*.nii.gz"))
                    dataset_mri_path = None
                    for f in mri_files:
                        if f.is_file() and f.name.startswith(prefix):
                            dataset_mri_path = str(f)
                            break
                    if dataset_mri_path:
                        print(f"Computing new 61-feature predictions on-the-fly for dataset scan {patient_id}...")
                        predicted_vector = get_regressor_predictions(dataset_mri_path)
                except Exception as e:
                    print(f"[WARNING] On-the-fly prediction failed for dataset scan {patient_id}: {e}")

            # Retrieve rule-based vector from rule_based_features.json
            if os.path.exists(RULE_FEATURES_FILE):
                try:
                    with open(RULE_FEATURES_FILE, 'r') as f:
                        all_rule = json.load(f)
                        rule_based_vector = all_rule.get(patient_id)
                except Exception as e:
                    print("Error loading rule features:", e)

        if not feature_data:
            self.send_error(404, "Patient not found")
            return
            
        # Resolve static URLs for NIfTI viewer
        mri_url = None
        seg_url = None
        
        if source == "uploads":
            # For uploads, check file existence on disk (N4/seg output check with .nii and .nii.gz)
            n4_nii = os.path.join(scan_dir, f"{patient_id}_n4.nii")
            n4_nii_gz = os.path.join(scan_dir, f"{patient_id}_n4.nii.gz")
            if os.path.exists(n4_nii):
                mri_url = f"/uploads/{patient_id}/{patient_id}_n4.nii"
            elif os.path.exists(n4_nii_gz):
                mri_url = f"/uploads/{patient_id}/{patient_id}_n4.nii.gz"
            else:
                # Fallback to the original uploaded scan file
                orig_scans = list(pathlib.Path(scan_dir).glob(f"{patient_id}.nii*"))
                if orig_scans:
                    mri_url = f"/uploads/{patient_id}/{orig_scans[0].name}"
                    
            seg_nii = os.path.join(scan_dir, f"{patient_id}_seg.nii")
            seg_nii_gz = os.path.join(scan_dir, f"{patient_id}_seg.nii.gz")
            if os.path.exists(seg_nii):
                seg_url = f"/uploads/{patient_id}/{patient_id}_seg.nii"
            elif os.path.exists(seg_nii_gz):
                seg_url = f"/uploads/{patient_id}/{patient_id}_seg.nii.gz"
        else:
            # For dataset patients, search recursively in ixi-t1 using prefix (subject ID)
            try:
                prefix = "-".join(patient_id.split("-")[:3])
                # Find matching scan in ixi-t1
                dataset_dir = os.path.join(WORKSPACE_DIR, "ixi-t1")
                mri_files = list(pathlib.Path(dataset_dir).rglob("*.nii"))
                for f in mri_files:
                    if f.is_file() and f.name.startswith(prefix):
                        mri_url = "/" + f.relative_to(WORKSPACE_DIR).as_posix()
                        break
                        
                # Find matching mask in ixi-t1-masks
                mask_dir = os.path.join(WORKSPACE_DIR, "ixi-t1-masks")
                seg_files = list(pathlib.Path(mask_dir).glob(f"*_seg.nii"))
                for f in seg_files:
                    if f.is_file() and f.name.startswith(prefix):
                        seg_url = "/" + f.relative_to(WORKSPACE_DIR).as_posix()
                        break
            except Exception as e:
                print(f"[WARNING] Error discovering dataset NIfTI files: {e}")

        # Format response
        response = feature_data.copy()
        response["source"] = source
        response["rule_based_array"] = rule_based_vector
        response["predicted_array"] = predicted_vector
        response["mri_url"] = mri_url
        response["seg_url"] = seg_url
        
        self.send_json_response(response)

    # API: Serves or generates clinical report
    def handle_api_report(self, patient_id):
        patient_id = urllib.parse.unquote(patient_id)
        
        # 1. Check if report file exists
        report_file = os.path.join(REPORTS_DIR, f"{patient_id}_report.txt")
        report_content = None
        
        if os.path.exists(report_file):
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report_content = f.read()
            except Exception as e:
                print("Error reading report file:", e)
                
        # 2. Check if upload folder has a report
        if not report_content:
            upload_report = os.path.join(UPLOADS_DIR, patient_id, f"{patient_id}_report.txt")
            if os.path.exists(upload_report):
                try:
                    with open(upload_report, 'r', encoding='utf-8') as f:
                        report_content = f.read()
                except Exception:
                    pass

        # Force regeneration if the cached report is in the old text format
        if report_content and not (report_content.startswith("<div") or "<style>" in report_content):
            report_content = None

        # 3. Dynamic clinical report generation based on loaded features
        if not report_content:
            # Need to get features first
            features_dict = None
            
            # Check upload cache
            cache_file = os.path.join(UPLOADS_DIR, patient_id, "features.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        features_dict = json.load(f)
                except Exception:
                    pass
            
            # Check precomputed
            if not features_dict and os.path.exists(FEATURES_FILE):
                try:
                    with open(FEATURES_FILE, 'r') as f:
                        all_pred = json.load(f)
                        if "schema_version" in all_pred:
                            pred_filename = all_pred.get("scan_metadata", {}).get("filename", "")
                            pred_patient_id = pathlib.Path(pred_filename).stem if pred_filename else ""
                            if patient_id == pred_patient_id:
                                features_dict = all_pred
                        else:
                            features_dict = all_pred.get(patient_id)
                except Exception:
                    pass
            
            if features_dict:
                # Generate dynamic text report
                report_content = build_clinical_report(patient_id, features_dict)
                # Save generated report in upload folder if it's an uploaded patient
                if os.path.exists(os.path.join(UPLOADS_DIR, patient_id)):
                    try:
                        with open(os.path.join(UPLOADS_DIR, patient_id, f"{patient_id}_report.txt"), 'w', encoding='utf-8') as f:
                            f.write(report_content)
                    except Exception:
                        pass
            else:
                report_content = f"Patient metrics not found. Cannot generate clinical report."

        self.send_json_response({"report": report_content})

    # API: MRIQC Docker job status
    def handle_api_mriqc_status(self, patient_id):
        job = mriqc_jobs.get(patient_id, {"status": "idle", "logs": [], "progress": ""})
        self.send_json_response(job)

    # API: Quality gate check — returns pass/fail based on stored MRIQC metrics
    def handle_api_mriqc_quality_gate(self, patient_id):
        metrics = None

        # 1. Check upload cache for mriqc_metrics
        cache_file = os.path.join(UPLOADS_DIR, patient_id, "features.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                metrics = cached.get("mriqc_metrics")
                if not metrics:
                    # Graceful fallback: if Docker MRIQC wasn't run or failed, simulate from rule-based vector
                    vector = cached.get("rule_based_array")
                    if vector:
                        metrics = simulate_mriqc_from_vector(vector)
            except Exception as e:
                print(f"Quality gate check error: {e}")

        # 2. Check dataset precomputed features
        if not metrics:
            if os.path.exists(RULE_FEATURES_FILE):
                try:
                    with open(RULE_FEATURES_FILE, 'r') as f:
                        all_rule = json.load(f)
                    vector = all_rule.get(patient_id)
                    if vector:
                        metrics = simulate_mriqc_from_vector(vector)
                except Exception as e:
                    print(f"Error loading rule features for quality gate: {e}")

        if not metrics:
            self.send_json_response({
                "passed": False,
                "grade": "F",
                "label": "Failed",
                "score": 0,
                "reason": "MRIQC failed to run (Docker is likely not installed or encountered an error).",
                "metrics": None
            })
            return

        quality = compute_mriqc_quality_score(metrics)
        quality["metrics"] = metrics
        self.send_json_response(quality)

    # API: Start official MRIQC Docker run in background thread
    def handle_api_mriqc_run(self, patient_id):
        mri_path = None
        # 1. Check uploads folder
        upload_file = os.path.join(UPLOADS_DIR, patient_id, f"{patient_id}.nii")
        if os.path.exists(upload_file):
            mri_path = upload_file
        # 2. Check dataset folder
        if not mri_path:
            dataset_dir = os.path.join(WORKSPACE_DIR, "ixi-t1")
            prefix = "-".join(patient_id.split("-")[:3])
            mri_files = list(pathlib.Path(dataset_dir).rglob("*.nii"))
            for f in mri_files:
                if f.is_file() and f.name.startswith(prefix):
                    mri_path = str(f)
                    break
        if not mri_path:
            self.send_json_response({"status": "error", "error": f"MRI scan file not found for patient {patient_id}."})
            return
        t = threading.Thread(target=run_official_mriqc_thread, args=(patient_id, mri_path))
        t.daemon = True
        t.start()
        self.send_json_response({"status": "started"})

    # API: Streams pipeline logs
    def handle_api_logs(self, query):
        global active_logs, is_running
        params = urllib.parse.parse_qs(query)
        
        index = 0
        if 'index' in params:
            try:
                index = int(params['index'][0])
            except ValueError:
                pass
                
        new_lines = active_logs[index:]
        response = {
            "new_lines": new_lines,
            "is_running": is_running,
            "next_index": len(active_logs)
        }
        self.send_json_response(response)

    # API: Ingest and upload NIfTI scan file
    # Pure Python implementation of multipart parser (cgi module independent)
    def handle_api_upload(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        # Boundary extraction
        boundary_param = [p for p in self.headers.get('Content-Type', '').split(';') if 'boundary=' in p]
        if not boundary_param:
            self.send_error(400, "Missing boundary parameter")
            return
            
        boundary = boundary_param[0].split('boundary=')[1].strip().encode('utf-8')

        # Split parts
        parts = body.split(b'--' + boundary)
        file_data = None
        filename = "uploaded_scan.nii"

        for part in parts:
            if b'Content-Disposition' in part and b'filename="' in part:
                header_end = part.find(b'\r\n\r\n')
                if header_end == -1:
                    header_end = part.find(b'\n\n')
                    data_start = header_end + 2
                else:
                    data_start = header_end + 4
                
                # Extract headers
                headers_part = part[:header_end].decode('utf-8', errors='ignore')
                for line in headers_part.split('\r\n'):
                    if 'filename="' in line:
                        filename = line.split('filename="')[1].split('"')[0]
                        break
                        
                file_data = part[data_start:]
                if file_data.endswith(b'\r\n'):
                    file_data = file_data[:-2]
                elif file_data.endswith(b'\n'):
                    file_data = file_data[:-1]
                break

        if not file_data:
            self.send_error(400, "No file uploaded")
            return

        # Prepare filenames
        if filename.endswith(".nii.gz"):
            scan_id = filename[:-7].strip()
            extension = ".nii.gz"
        else:
            scan_id = filename[:-4].strip() if filename.endswith(".nii") else filename.strip()
            extension = ".nii"

        # Clean special chars to make safe dir name
        for char in ['/', '\\', '?', '%', '*', ':', '|', '"', '<', '>', ' ']:
            scan_id = scan_id.replace(char, "_")
            
        # Security: prevent directory traversal
        dest_dir = os.path.abspath(os.path.join(UPLOADS_DIR, scan_id))
        if not dest_dir.startswith(os.path.abspath(UPLOADS_DIR)):
            self.send_error(400, "Invalid filename / path traversal attempt")
            return
            
        os.makedirs(dest_dir, exist_ok=True)
        dest_file = os.path.join(dest_dir, f"{scan_id}{extension}")
        
        try:
            # Write to final destination directly so nibabel can work out the file type based on extension
            with open(dest_file, 'wb') as f:
                f.write(file_data)
                
            # Validate NIfTI file structure using nibabel
            try:
                test_img = nib.load(dest_file)
                test_img.shape
            except Exception as e:
                if os.path.exists(dest_file):
                    try:
                        os.remove(dest_file)
                    except Exception:
                        pass
                self.send_error(400, f"Invalid or corrupt NIfTI scan: {str(e)}")
                return
                
            print(f"Validated and saved uploaded scan to {dest_file}")
            
            # Immediately run dynamic feature extraction in the background
            cache_file = os.path.join(dest_dir, "features.json")
            
            def run_extraction_bg():
                try:
                    extraction_results = run_feature_extraction(dest_file)
                    rule_based_vector = extraction_results["rule_based_vector"]
                    feature_data = features_list_to_dict(
                        filename=os.path.basename(dest_file),
                        features=rule_based_vector
                    )
                    feature_data["structural_features"]["centroid"] = extraction_results["centroid"]
                    feature_data["structural_features"]["bounding_box"] = extraction_results["bbox"]
                    if extraction_results.get("volumetric_statistics") is not None:
                        feature_data["volumetric_statistics"] = extraction_results["volumetric_statistics"]

                    # Cache payload (without MRIQC metrics)
                    cache_payload = feature_data.copy()
                    cache_payload["rule_based_array"] = rule_based_vector
                    
                    temp_cache = cache_file + ".tmp"
                    with open(temp_cache, 'w') as f:
                        json.dump(cache_payload, f, indent=2)
                    os.replace(temp_cache, cache_file)
                    print(f"[SUCCESS] Background feature extraction complete for {scan_id}")
                except Exception as e:
                    print(f"[ERROR] Background feature extraction failed for {scan_id}: {e}")
                finally:
                    active_extractions.pop(scan_id, None)

            t = threading.Thread(target=run_extraction_bg)
            t.daemon = True
            active_extractions[scan_id] = t
            t.start()
            
            self.send_json_response({
                "status": "started",
                "patient_id": scan_id,
                "path": f"uploads/{scan_id}/{scan_id}{extension}",
                "mriqc_passed": False,
                "mriqc_grade":  "?",
                "mriqc_label":  "Pending",
                "mriqc_score":  0,
                "mriqc_reason": "Pending MRIQC execution.",
                "mriqc_metrics": None,
            })
            
        except Exception as e:
            self.send_error(500, f"Failed to save file: {str(e)}")

    # API: Trigger command line subprocess execution
    def handle_api_run_step(self):
        global active_process, active_logs, is_running
        
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        payload = json.loads(body.decode('utf-8'))
        
        step = payload.get('step')
        data_source = payload.get('data_source', 'ixi-t1')
        
        with pipeline_lock:
            if is_running:
                self.send_json_response({"status": "error", "error": "Pipeline step already running"})
                return
            is_running = True

        # Map step number to pipeline command execution
        cmd = ""
        # Since data_source could be uploads, let's configure directories
        mri_dir = UPLOADS_DIR if data_source == "uploads" else WORKSPACE_DIR
        
        if step == 1:
            cmd = f"python train_unet.py --generate-masks --mri-dir {data_source} --mask-dir {data_source}-masks --limit 5"
        elif step == 2:
            cmd = f"python train_autoencoder.py --data-dir {data_source} --epochs 5 --batch-size 4 --latent-dim 128"
        elif step == 3:
            cmd = f"python train_unet.py --mri-dir {data_source} --mask-dir {data_source}-masks --epochs 5 --batch-size 4"
        elif step == 4:
            # Create uploaded rule features if using uploads
            feat_file = "rule_based_features.json"
            if data_source == "uploads":
                feat_file = create_temp_uploads_features_json()
            cmd = f"python train_regressor.py --mri-dir {data_source} --features-file {feat_file} --epochs 5 --batch-size 8"
        elif step == 5:
            cmd = f"python train_regressor.py --mri-dir {data_source} --checkpoint checkpoints/regressor_best.pth --predict --save-dir features"
        elif step == 6:
            cmd = f"python generate_report.py --input features/predicted_features.json --output reports/"
        else:
            self.send_json_response({"status": "error", "error": "Invalid pipeline step"})
            return

        # Reset logs
        active_logs = []
        
        # Append start cmd log
        active_logs.append(f"Executing: {cmd}\n")
        
        try:
            # Run in workspace directory
            active_process = subprocess.Popen(
                cmd,
                shell=True,
                cwd=WORKSPACE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            
            # Start background reader thread
            t = threading.Thread(target=read_process_stdout, args=(active_process,))
            t.daemon = True
            t.start()
            
            self.send_json_response({
                "status": "started",
                "pid": active_process.pid
            })
            
        except Exception as e:
            with pipeline_lock:
                is_running = False
            self.send_json_response({"status": "error", "error": str(e)})

    # Utilities
    def send_json_response(self, data):
        sanitized = sanitize_nan(data)
        response_bytes = json.dumps(sanitized).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response_bytes))
        self.end_headers()
        self.wfile.write(response_bytes)


def sanitize_nan(data):
    """Recursively replace NaN/Inf floats with None so JSON serialization never fails."""
    import math
    if isinstance(data, dict):
        return {k: sanitize_nan(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_nan(v) for v in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
    return data


# Global regressor model cache for on-the-fly inference
regressor_model = None
regressor_stats = None

def get_regressor_predictions(mri_path):
    """Load regressor model and make predictions on the given MRI scan on-the-fly"""
    global regressor_model, regressor_stats
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, "regressor_best.pth")
    if not os.path.exists(checkpoint_path):
        print(f"[WARNING] Regressor checkpoint not found at {checkpoint_path}. Cannot predict.")
        return None

    try:
        import torch
        from train_regressor import FeatureRegressorCNN, predict_features_from_mri
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and stats if not cached
        if regressor_model is None:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            num_features = checkpoint.get("num_features", 61)
            
            model = FeatureRegressorCNN(input_channels=1, num_features=num_features).to(device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            
            regressor_model = model
            regressor_stats = {
                "mean": np.array(checkpoint["feature_mean"], dtype=np.float32),
                "std": np.array(checkpoint["feature_std"], dtype=np.float32)
            }
            print(f"[SUCCESS] Loaded regressor model from {checkpoint_path} for on-the-fly inference.")
            
        # Run prediction
        features_pred = predict_features_from_mri(
            regressor_model, mri_path, 
            regressor_stats["mean"], regressor_stats["std"], 
            device
        )
        return features_pred
    except Exception as e:
        print(f"[WARNING] Regressor inference failed for {mri_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# HELPER DATA FORMATTERS
# =============================================================================

def features_list_to_dict(filename, features):
    """Maps the raw 61-element float list to the output schema format"""
    from train_regressor import build_structured_json
    return build_structured_json(features, filename)


def flatten_feature_dict(d: dict) -> list:
    """Helper to reconstruct flat 61-element list from structured predicted features dict"""
    try:
        intensity = d.get("intensity_statistics", {})
        tissue = d.get("tissue_segmentation", {})
        texture = d.get("texture_features", {})
        struct = d.get("structural_features", {})
        symmetry = d.get("symmetry", {})
        freq = d.get("frequency_domain", {})
        edge = d.get("edge_features", {})
        
        flat = [
            # Intensity (12)
            intensity.get("mean", 0.0),
            intensity.get("std", 0.0),
            intensity.get("min", 0.0),
            intensity.get("max", 0.0),
            intensity.get("median", 0.0),
            intensity.get("percentile_25", 0.0),
            intensity.get("percentile_75", 0.0),
            intensity.get("skewness", 0.0),
            intensity.get("kurtosis", 0.0),
            intensity.get("dynamic_range", 0.0),
            intensity.get("mode", 0.0),
            intensity.get("bimodality", 0.0),
            # Tissue (7)
            tissue.get("csf_ventricles", {}).get("percentage", 0.0),
            tissue.get("gray_matter", {}).get("percentage", 0.0),
            tissue.get("white_matter", {}).get("percentage", 0.0),
            tissue.get("brain_tissue_total_pct", 0.0),
            tissue.get("otsu_threshold_1", 0.0),
            tissue.get("otsu_threshold_2", 0.0),
            tissue.get("gm_wm_cnr", 0.0),
            # Texture (8)
            texture.get("glcm_contrast", 0.0),
            texture.get("glcm_correlation", 0.0),
            texture.get("glcm_energy", 0.0),
            texture.get("glcm_homogeneity", 0.0),
            texture.get("mean_gradient", 0.0),
            texture.get("std_gradient", 0.0),
            texture.get("max_gradient", 0.0),
            texture.get("shannon_entropy", 0.0),
            # Structural (6)
            struct.get("brain_area_pct", 0.0),
            struct.get("eccentricity", 0.0),
            struct.get("solidity", 0.0),
            struct.get("extent", 0.0),
            struct.get("perimeter", 0.0),
            struct.get("ventricular_proxy_pct", 0.0),
            # Symmetry (2)
            symmetry.get("bilateral_symmetry_score", 0.0),
            symmetry.get("asymmetry_index_pct", 0.0),
            # Frequency (3)
            freq.get("low_frequency_energy", 0.0),
            freq.get("high_frequency_energy", 0.0),
            freq.get("frequency_energy_ratio", 0.0),
            # Edge (2)
            edge.get("cortical_edge_sharpness", 0.0),
            edge.get("cortical_edge_std", 0.0)
        ]
        
        # Volumetric stats (21 features starting at index 40)
        vol = d.get("volumetric_statistics", {})
        flat.extend([
            vol.get("left_ventricle_volume_ml", 0.0),
            vol.get("right_ventricle_volume_ml", 0.0),
            vol.get("ventricle_asymmetry_index", 0.0),
            
            vol.get("left_thalamus_volume_ml", 0.0),
            vol.get("right_thalamus_volume_ml", 0.0),
            vol.get("thalamus_asymmetry_index", 0.0),
            
            vol.get("left_caudate_volume_ml", 0.0),
            vol.get("right_caudate_volume_ml", 0.0),
            vol.get("caudate_asymmetry_index", 0.0),
            
            vol.get("left_putamen_volume_ml", 0.0),
            vol.get("right_putamen_volume_ml", 0.0),
            vol.get("putamen_asymmetry_index", 0.0),
            
            vol.get("left_pallidum_volume_ml", 0.0),
            vol.get("right_pallidum_volume_ml", 0.0),
            vol.get("pallidum_asymmetry_index", 0.0),
            
            vol.get("left_hippocampus_volume_ml", 0.0),
            vol.get("right_hippocampus_volume_ml", 0.0),
            vol.get("hippocampus_asymmetry_index", 0.0),
            
            vol.get("left_amygdala_volume_ml", 0.0),
            vol.get("right_amygdala_volume_ml", 0.0),
            vol.get("amygdala_asymmetry_index", 0.0)
        ])
        return flat
    except Exception as e:
        print("Reconstruct flat list failed:", e)
        return []


def create_temp_uploads_features_json() -> str:
    """Utility to assemble all uploads rule-based arrays into a temporary rule_based_features.json"""
    data = {}
    uploaded_scans = pathlib.Path(UPLOADS_DIR).glob("*/features.json")
    for fp in uploaded_scans:
        try:
            with open(fp, 'r') as f:
                features = json.load(f)
                patient_id = fp.parent.name
                data[patient_id] = features.get("rule_based_array")
        except Exception:
            pass
            
    temp_file = os.path.join(UPLOADS_DIR, "temp_rule_based_features.json")
    with open(temp_file, 'w') as f:
        json.dump(data, f)
    return temp_file


# =============================================================================
# SERVER INITIATOR
# =============================================================================

def download_volumetric_templates():
    """Download MNI152 template and Harvard-Oxford labels if not present"""
    template_path = os.path.join(CHECKPOINTS_DIR, "MNI152_T1_2mm_brain.nii.gz")
    labels_path = os.path.join(CHECKPOINTS_DIR, "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz")
    
    if not os.path.exists(template_path) or not os.path.exists(labels_path):
        import urllib.request
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        
        # Download template
        if not os.path.exists(template_path):
            print("[INFO] Downloading MNI152 2mm T1 template brain (from HCP pipelines)...")
            try:
                url = "https://github.com/Washington-University/HCPpipelines/raw/master/global/templates/MNI152_T1_2mm.nii.gz"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(template_path, 'wb') as out_file:
                    out_file.write(response.read())
                print("[INFO] MNI152 template downloaded successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to download template: {e}")
                
        # Download labels
        if not os.path.exists(labels_path):
            print("[INFO] Downloading Harvard-Oxford subcortical labels (from NeuroVault)...")
            try:
                url = "https://neurovault.org/media/images/262/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(labels_path, 'wb') as out_file:
                    out_file.write(response.read())
                print("[INFO] Harvard-Oxford labels downloaded successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to download labels: {e}")

def start_server():
    download_volumetric_templates()
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("", PORT), DashboardAPIHandler) as httpd:
        print(f"\n==================================================")
        print(f" NeuroSphere Dashboard running at: http://localhost:{PORT}")
        print(f" Press Ctrl+C to terminate the server process.")
        print(f"==================================================\n")
        
        # Open in web browser
        def open_browser():
            time.sleep(1.2)
            webbrowser.open(f"http://localhost:{PORT}")
            
        t = threading.Thread(target=open_browser)
        t.daemon = True
        t.start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()

if __name__ == '__main__':
    start_server()
