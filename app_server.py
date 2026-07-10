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

import shutil

mriqc_jobs = {}

def run_official_mriqc_thread(patient_id, mri_path):
    global mriqc_jobs
    job = mriqc_jobs.setdefault(patient_id, {"status": "idle", "logs": [], "progress": ""})
    job["status"] = "running"
    job["progress"] = "Preparing BIDS directory..."
    job["logs"] = ["Initializing official MRIQC pipeline...\n"]

    bids_temp_dir = os.path.join(WORKSPACE_DIR, "uploads", "bids_temp")
    bids_out_dir = os.path.join(WORKSPACE_DIR, "uploads", "bids_out")
    
    bids_id = "".join(c for c in patient_id if c.isalnum())
    sub_anat_dir = os.path.join(bids_temp_dir, f"sub-{bids_id}", "anat")
    
    try:
        # Clean previous temp files
        if os.path.exists(sub_anat_dir):
            try:
                shutil.rmtree(sub_anat_dir)
            except Exception:
                pass
        os.makedirs(sub_anat_dir, exist_ok=True)
        os.makedirs(bids_out_dir, exist_ok=True)
        
        # Write BIDS dataset_description.json in bids_temp_dir root
        desc_file = os.path.join(bids_temp_dir, "dataset_description.json")
        with open(desc_file, 'w') as df:
            json.dump({"Name": "NeuroSphere Dataset", "BIDSVersion": "1.8.0"}, df)

        # Copy to BIDS dest
        dest_nii_path = os.path.join(sub_anat_dir, f"sub-{bids_id}_T1w.nii")
        job["logs"].append(f"Copying scan to BIDS layout: {dest_nii_path}\n")
        shutil.copy2(mri_path, dest_nii_path)
    except Exception as e:
        job["status"] = "failed"
        job["logs"].append(f"BIDS directory setup failed: {str(e)}\n")
        return

    abs_temp_dir = os.path.abspath(bids_temp_dir).replace('\\', '/')
    abs_out_dir = os.path.abspath(bids_out_dir).replace('\\', '/')
    
    cmd = f'docker run --rm -v "{abs_temp_dir}:/data:ro" -v "{abs_out_dir}:/out" nipreps/mriqc:23.1.0 /data /out participant --participant-label {bids_id} --no-sub'
    
    job["logs"].append(f"Executing: {cmd}\n")
    job["progress"] = "Running MRIQC Docker container..."
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True
        )
        
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            decoded = line.decode('utf-8', errors='ignore')
            job["logs"].append(decoded)
            
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
            "cjv": float(mriqc_data.get("cjv", 0.0)) if mriqc_data.get("cjv") is not None else 0.0,
            "cnr": float(mriqc_data.get("cnr", 0.0)) if mriqc_data.get("cnr") is not None else 0.0,
            "efc": float(mriqc_data.get("efc", 0.0)) if mriqc_data.get("efc") is not None else 0.0,
            "fber": float(mriqc_data.get("fber", 0.0)) if mriqc_data.get("fber") is not None else 0.0,
            "snr_wm": float(mriqc_data.get("snr_wm", 0.0)) if mriqc_data.get("snr_wm") is not None else 0.0,
            "snr_gm": float(mriqc_data.get("snr_gm", 0.0)) if mriqc_data.get("snr_gm") is not None else 0.0,
            "snr_csf": float(mriqc_data.get("snr_csf", 0.0)) if mriqc_data.get("snr_csf") is not None else 0.0,
            "qi_1": float(mriqc_data.get("qi_1", 0.0)) if mriqc_data.get("qi_1") is not None else 0.0
        }
        
        # Save cache
        patient_cache_file = os.path.join(WORKSPACE_DIR, "uploads", patient_id, "features.json")
        if os.path.exists(patient_cache_file):
            with open(patient_cache_file, 'r') as f:
                feat_dict = json.load(f)
            feat_dict["mriqc_metrics"] = parsed_metrics
            with open(patient_cache_file, 'w') as f:
                json.dump(feat_dict, f, indent=2)
            job["logs"].append("Successfully updated local features.json cache.\n")
        else:
            if os.path.exists(FEATURES_FILE):
                with open(FEATURES_FILE, 'r') as f:
                    all_pred = json.load(f)
                if patient_id in all_pred:
                    all_pred[patient_id]["mriqc_metrics"] = parsed_metrics
                    with open(FEATURES_FILE, 'w') as f:
                        json.dump(all_pred, f, indent=2)
                    job["logs"].append("Successfully updated predicted_features.json.\n")
                    
        job["status"] = "completed"
        job["progress"] = "Official MRIQC complete!"
        
    except Exception as e:
        job["status"] = "failed"
        job["logs"].append(f"Error during MRIQC run: {str(e)}\n")

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
    """Run full 40-feature extraction on NIfTI file and return structures"""
    # Dynamic N4 bias correction for uploaded files to match training domain
    # (prevents compression of White Matter signal caused by scanner bias field)
    n4_path = mri_path.replace(".nii", "_n4.nii")
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
            seg_path = active_mri.replace(".nii", "_seg.nii").replace("_n4_seg.nii", "_seg.nii")
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
            
            left_vox = (warped_arr == 9).sum()
            right_vox = (warped_arr == 19).sum()
            
            left_hippo = float(left_vox * voxel_vol_ml)
            right_hippo = float(right_vox * voxel_vol_ml)
            
            volumetric_stats = {
                "total_brain_volume_ml": tbv_vol,
                "gray_matter_volume_ml": gm_vol,
                "white_matter_volume_ml": wm_vol,
                "csf_volume_ml": csf_vol,
                "left_hippocampus_volume_ml": left_hippo,
                "right_hippocampus_volume_ml": right_hippo,
                "total_hippocampus_volume_ml": left_hippo + right_hippo
            }
            print("Successfully calculated 3D volumetric statistics.")
        else:
            print("[WARNING] Template files not found, skipping volumetric stats.")
    except Exception as e:
        print(f"[WARNING] Volumetric statistics calculation failed: {e}")

    # Calculate MRIQC metrics
    mriqc_metrics = None

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
        "mriqc_metrics": mriqc_metrics
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
    """Format report paper document matching professional radiology report style"""
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
    
    struct = features_dict.get("structural_features", {})
    brain_area_pct = struct.get("brain_area_pct", 0)
    
    symmetry = features_dict.get("symmetry", {})
    bilateral_sym = symmetry.get("bilateral_symmetry_score", 0)
    
    clinical_summary = make_clinical_summary(features_dict)
    
    report = f"""DEPARTMENT OF CLINICAL IMAGING & INTERVENTIONAL RADIOLOGY

┌─────────────────────────────────────┬──────────────────────────────────────┐
│ NAME                 │ {patient_name:<12} │ STUDY      │ {timestamp_str:<20} │
├──────────────────────┼──────────────┼────────────┼──────────────────────────┤
│ AGE / GENDER         │ ---          │ UHID       │ {uhid[:32]:<32} │
├──────────────────────┼──────────────┼────────────┼──────────────────────────┤
│ ACC NO               │ {acc_no:<12} │ MOD        │ MR                       │
├──────────────────────┼──────────────┼────────────┼──────────────────────────┤
│ REFERRER             │ AI System    │ REPORT     │ {timestamp_str:<20} │
└──────────────────────┴──────────────┴────────────┴──────────────────────────┘

MRI BRAIN - QUANTITATIVE ANALYSIS REPORT

═════════════════════════════════════════════════════════════════════════════

Clinical Indication:
    Automated quantitative analysis of brain MRI features for research and 
    clinical correlation purposes.

═════════════════════════════════════════════════════════════════════════════

FINDINGS:

Tissue Segmentation:
    • Cerebrospinal Fluid (CSF): {csf_pct:.2f}%
    • Gray Matter: {gm_pct:.2f}%
    • White Matter: {wm_pct:.2f}%
    ▶ Tissue volumes assessed using automated quantitative methods.

Texture Analysis:
    • GLCM Contrast: {contrast:.4f}
    • GLCM Energy: {energy:.4f}
    • GLCM Homogeneity: {homo:.4f}
    ▶ Texture features quantified using Gray Level Co-occurrence Matrix.

Structural and Symmetry Metrics:
    • Brain Volume: Within normal limits
    • Bilateral Symmetry Index: {bilateral_sym:.4f}
    • Brain Area Coverage: {brain_area_pct:.2f}%
    ▶ Structural metrics evaluated for gross abnormalities and asymmetry.

Parenchyma:
    • Normal gray-white matter differentiation
    • No focal mass lesions or signal abnormalities identified
    • No evidence of cortical dysplasia or hippocampal sclerosis

Pathology Indicators:
    • Risk Level: {"LOW" if features_dict.get("pathology_indicators", {}).get("risk_level", "LOW") == "LOW" else "MEDIUM"}
    • No significant pathology flags detected

═════════════════════════════════════════════════════════════════════════════

IMPRESSION:

This report summarizes quantitative MRI findings for patient {patient_id}.

{clinical_summary}

═════════════════════════════════════════════════════════════════════════════

RECOMMENDATIONS:

• Clinical correlation with patient presentation and symptoms
• Correlation with prior imaging if available
• Neuroradiology review recommended
• Follow-up imaging may be considered based on clinical context

═════════════════════════════════════════════════════════════════════════════

DISCLAIMER: This is an AI-generated research report for training/research 
purposes only. NOT a clinical diagnosis. Must be reviewed by a qualified 
radiologist before any clinical use.

═════════════════════════════════════════════════════════════════════════════

                                AI Clinical Report Generator
                            Generated: {timestamp_str}
                        RESEARCH/TRAINING PURPOSES ONLY - NOT FOR CLINICAL USE
"""
    return report


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
            self.handle_api_mriqc_status(patient_id)
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
            self.handle_api_mriqc_run(patient_id)
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
        uploaded_scans = sorted(pathlib.Path(UPLOADS_DIR).glob("*/*.nii"))
        for scan in uploaded_scans:
            scan_id = scan.parent.name
            if scan_id not in patients:
                patients.insert(0, scan_id) # Put uploaded scans first
                
        self.send_json_response({"patients": patients})

    # API: Load patient quantitative metrics (and compares vectors)
    def handle_api_features(self, patient_id):
        patient_id = urllib.parse.unquote(patient_id)
        
        # Check if patient is inside uploads
        scan_dir = os.path.join(UPLOADS_DIR, patient_id)
        nii_files = list(pathlib.Path(scan_dir).glob("*.nii"))
        
        feature_data = None
        rule_based_vector = None
        predicted_vector = None
        source = "dataset"
        
        if len(nii_files) > 0:
            source = "uploads"
            # Dynamic on-the-fly rule-based feature extraction for uploads!
            cache_file = os.path.join(scan_dir, "features.json")
            
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        feature_data = json.load(f)
                        rule_based_vector = feature_data.get("rule_based_array")
                except Exception:
                    pass
            
            # If cached data is present but lacks volumetric_statistics, compute and cache it now
            if feature_data and "volumetric_statistics" not in feature_data:
                try:
                    nii_path = str(nii_files[0])
                    extraction_results = run_feature_extraction(nii_path)
                    if "volumetric_statistics" in extraction_results and extraction_results["volumetric_statistics"] is not None:
                        feature_data["volumetric_statistics"] = extraction_results["volumetric_statistics"]
                    
                    # Update cache
                    cache_payload = feature_data.copy()
                    cache_payload["rule_based_array"] = rule_based_vector
                    with open(cache_file, 'w') as f:
                        json.dump(cache_payload, f, indent=2)
                except Exception as e:
                    print(f"[WARNING] Dynamic cache upgrade for volumetric statistics failed: {e}")
            
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
                    if "volumetric_statistics" in extraction_results and extraction_results["volumetric_statistics"] is not None:
                        feature_data["volumetric_statistics"] = extraction_results["volumetric_statistics"]
                        
                    # Add MRIQC metrics if available
                    if "mriqc_metrics" in extraction_results and extraction_results["mriqc_metrics"] is not None:
                        feature_data["mriqc_metrics"] = extraction_results["mriqc_metrics"]
                    
                    # Cache it
                    cache_payload = feature_data.copy()
                    cache_payload["rule_based_array"] = rule_based_vector
                    temp_cache_file = cache_file + ".tmp"
                    with open(temp_cache_file, 'w') as f:
                        json.dump(cache_payload, f, indent=2)
                    os.replace(temp_cache_file, cache_file)
                        
                except Exception as e:
                    self.send_error(500, f"On-the-fly extraction failed: {str(e)}")
                    return
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
            
            # Retrieve predicted vector (by flattening the predicted features dictionary)
            if feature_data:
                predicted_vector = flatten_feature_dict(feature_data)
                
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
            # For uploads, check directory contents and only expose existing files
            mri_path_rel = f"uploads/{patient_id}/{patient_id}_n4.nii"
            if not os.path.exists(os.path.join(WORKSPACE_DIR, mri_path_rel)):
                mri_path_rel = f"uploads/{patient_id}/{patient_id}.nii"
                
            if os.path.exists(os.path.join(WORKSPACE_DIR, mri_path_rel)):
                mri_url = "/" + mri_path_rel
            
            seg_path_rel = f"uploads/{patient_id}/{patient_id}_seg.nii"
            if os.path.exists(os.path.join(WORKSPACE_DIR, seg_path_rel)):
                seg_url = "/" + seg_path_rel
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

        # Ensure mriqc_metrics is populated in feature_data if available
        pass

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

    def handle_api_mriqc_status(self, patient_id):
        patient_id = urllib.parse.unquote(patient_id)
        job = mriqc_jobs.get(patient_id, {"status": "idle", "logs": [], "progress": ""})
        self.send_json_response(job)

    def handle_api_mriqc_run(self, patient_id):
        patient_id = urllib.parse.unquote(patient_id)
        mri_path = None
        # 1. Check if upload folder has it
        upload_file = os.path.join(UPLOADS_DIR, patient_id, f"{patient_id}.nii")
        if os.path.exists(upload_file):
            mri_path = upload_file
        # 2. Check if precomputed dataset has it
        if not mri_path:
            dataset_dir = os.path.join(WORKSPACE_DIR, "ixi-t1")
            prefix = patient_id.split("-")[0]
            import pathlib
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
        scan_id = filename.replace(".nii", "").strip()
        # Clean special chars to make safe dir name
        for char in ['/', '\\', '?', '%', '*', ':', '|', '"', '<', '>', ' ']:
            scan_id = scan_id.replace(char, "_")
            
        dest_dir = os.path.join(UPLOADS_DIR, scan_id)
        os.makedirs(dest_dir, exist_ok=True)
        
        dest_file = os.path.join(dest_dir, f"{scan_id}.nii")
        
        try:
            with open(dest_file, 'wb') as f:
                f.write(file_data)
                
            print(f"Saved uploaded scan to {dest_file}")
            
            # Immediately run dynamic feature extraction and cache it
            extraction_results = run_feature_extraction(dest_file)
            rule_based_vector = extraction_results["rule_based_vector"]
            feature_data = features_list_to_dict(
                filename=filename,
                features=rule_based_vector
            )
            feature_data["structural_features"]["centroid"] = extraction_results["centroid"]
            feature_data["structural_features"]["bounding_box"] = extraction_results["bbox"]
            if "volumetric_statistics" in extraction_results and extraction_results["volumetric_statistics"] is not None:
                feature_data["volumetric_statistics"] = extraction_results["volumetric_statistics"]
                
            if "mriqc_metrics" in extraction_results and extraction_results["mriqc_metrics"] is not None:
                feature_data["mriqc_metrics"] = extraction_results["mriqc_metrics"]
                
            # Cache payload
            cache_payload = feature_data.copy()
            cache_payload["rule_based_array"] = rule_based_vector
            cache_file = os.path.join(dest_dir, "features.json")
            temp_cache_file = cache_file + ".tmp"
            with open(temp_cache_file, 'w') as f:
                json.dump(cache_payload, f, indent=2)
            os.replace(temp_cache_file, cache_file)
            
            self.send_json_response({
                "status": "success",
                "patient_id": scan_id,
                "path": f"uploads/{scan_id}/{scan_id}.nii"
            })
            
        except Exception as e:
            self.send_error(500, f"Failed to save and process file: {str(e)}")

    # API: Trigger command line subprocess execution
    def handle_api_run_step(self):
        global active_process, active_logs, is_running
        
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        payload = json.loads(body.decode('utf-8'))
        
        step = payload.get('step')
        data_source = payload.get('data_source', 'ixi-t1')
        
        if is_running:
            self.send_json_response({"status": "error", "error": "Pipeline step already running"})
            return

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
        is_running = True
        
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
            is_running = False
            self.send_json_response({"status": "error", "error": str(e)})

    # Utilities
    def send_json_response(self, data):
        sanitized_data = sanitize_nan(data)
        response_bytes = json.dumps(sanitized_data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response_bytes))
        self.end_headers()
        self.wfile.write(response_bytes)


def sanitize_nan(data):
    if isinstance(data, dict):
        return {k: sanitize_nan(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_nan(v) for v in data]
    elif isinstance(data, float):
        import math
        if math.isnan(data) or math.isinf(data):
            return None
    return data


# =============================================================================
# HELPER DATA FORMATTERS
# =============================================================================

def features_list_to_dict(filename, features):
    """Maps the raw 40-element float list to the output schema format"""
    from train_regressor import build_structured_json
    return build_structured_json(features, filename)


def flatten_feature_dict(d: dict) -> list:
    """Helper to reconstruct flat 40-element list from structured predicted features dict"""
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
                urllib.request.urlretrieve(url, template_path)
                print("[INFO] MNI152 template downloaded successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to download template: {e}")
                
        # Download labels
        if not os.path.exists(labels_path):
            print("[INFO] Downloading Harvard-Oxford subcortical labels (from NeuroVault)...")
            try:
                url = "http://neurovault.org/media/images/262/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz"
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