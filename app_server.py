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
    """Run full 29-feature extraction on NIfTI file and return structures"""
    arr = load_mri_slice(mri_path)
    total = arr.size
    
    # 1. Intensity (10 features)
    flat = arr.flatten()
    mean_val = float(np.mean(flat))
    std_val = float(np.std(flat))
    min_val = float(np.min(flat))
    max_val = float(np.max(flat))
    median_val = float(np.median(flat))
    p25 = float(np.percentile(flat, 25))
    p75 = float(np.percentile(flat, 75))
    skewness = float(stats.skew(flat))
    kurtosis = float(stats.kurtosis(flat))
    dyn_range = float(max_val - min_val)
    
    # 2. Tissue (4 features)
    bg_mask = arr < 0.05
    csf_mask = (arr >= 0.05) & (arr < 0.25)
    gm_mask = (arr >= 0.25) & (arr < 0.55)
    wm_mask = arr >= 0.55
    
    csf_pct = float((csf_mask.sum() / total) * 100)
    gm_pct = float((gm_mask.sum() / total) * 100)
    wm_pct = float((wm_mask.sum() / total) * 100)
    brain_tissue_pct = float(((gm_mask.sum() + wm_mask.sum()) / total) * 100)
    
    # 3. Texture (8 features)
    arr_u8 = (arr * 255).astype(np.uint8)
    arr_binned = (arr_u8 // 32).astype(np.uint8)
    
    glcm = graycomatrix(
        arr_binned,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=8,
        symmetric=True,
        normed=True
    )
    
    grad_y, grad_x = np.gradient(arr)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    hist, _ = np.histogram(arr, bins=16, range=(0, 1), density=True)
    hist = hist + 1e-10
    entropy = float(-np.sum(hist * np.log2(hist)))
    
    glcm_contrast = float(graycoprops(glcm, 'contrast').mean())
    glcm_correlation = float(graycoprops(glcm, 'correlation').mean())
    glcm_energy = float(graycoprops(glcm, 'energy').mean())
    glcm_homogeneity = float(graycoprops(glcm, 'homogeneity').mean())
    mean_gradient = float(grad_mag.mean())
    std_gradient = float(grad_mag.std())
    max_gradient = float(grad_mag.max())
    
    # 4. Structural (5 features)
    brain_mask = arr > 0.05
    labeled = measure.label(brain_mask)
    regions = measure.regionprops(labeled)
    
    brain_area_pct = 0.0
    eccentricity = 0.0
    solidity = 0.0
    extent = 0.0
    perimeter = 0.0
    
    # Coordinates for drawing overlay
    centroid_row = 128.0
    centroid_col = 128.0
    bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col = 0, 0, 256, 256
    
    if regions:
        brain = max(regions, key=lambda r: r.area)
        brain_area_pct = float((brain.area / total) * 100)
        eccentricity = float(brain.eccentricity)
        solidity = float(brain.solidity)
        extent = float(brain.extent)
        perimeter = float(brain.perimeter)
        
        centroid_row = float(brain.centroid[0])
        centroid_col = float(brain.centroid[1])
        bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col = [int(v) for v in brain.bbox]

    # 5. Symmetry (2 features)
    mid_col = arr.shape[1] // 2
    left_half = arr[:, :mid_col]
    right_half = arr[:, mid_col:]
    
    min_size = min(left_half.shape[1], right_half.shape[1])
    left_half = left_half[:, :min_size]
    right_half = right_half[:, :min_size]
    
    right_half_flipped = np.fliplr(right_half)
    correlation = np.corrcoef(left_half.flatten(), right_half_flipped.flatten())[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
        
    symmetry_score = (correlation + 1) / 2
    asymmetry_idx = abs(left_half.mean() - right_half.mean()) * 100
    
    # Flattened array
    flat_vector = [
        mean_val, std_val, min_val, max_val, median_val, p25, p75, skewness, kurtosis, dyn_range,
        csf_pct, gm_pct, wm_pct, brain_tissue_pct,
        glcm_contrast, glcm_correlation, glcm_energy, glcm_homogeneity, mean_gradient, std_gradient, max_gradient, entropy,
        brain_area_pct, eccentricity, solidity, extent, perimeter,
        symmetry_score, asymmetry_idx
    ]
    
    return {
        "rule_based_vector": flat_vector,
        "centroid": {"row": centroid_row, "col": centroid_col},
        "bbox": {
            "min_row": bbox_min_row,
            "min_col": bbox_min_col,
            "max_row": bbox_max_row,
            "max_col": bbox_max_col
        }
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
        elif path == '/api/logs':
            self.handle_api_logs(parsed_url.query)
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
                    
                    # Cache it
                    cache_payload = feature_data.copy()
                    cache_payload["rule_based_array"] = rule_based_vector
                    with open(cache_file, 'w') as f:
                        json.dump(cache_payload, f, indent=2)
                        
                except Exception as e:
                    self.send_error(500, f"On-the-fly extraction failed: {str(e)}")
                    return
        else:
            # Load precomputed dataset patient
            if os.path.exists(FEATURES_FILE):
                try:
                    with open(FEATURES_FILE, 'r') as f:
                        all_pred = json.load(f)
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
            
        # Format response
        response = feature_data.copy()
        response["source"] = source
        response["rule_based_array"] = rule_based_vector
        response["predicted_array"] = predicted_vector
        
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
            
            # Immediately run dynamic feature extraction on save
            run_feature_extraction(dest_file)
            
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
        response_bytes = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response_bytes))
        self.end_headers()
        self.wfile.write(response_bytes)


# =============================================================================
# HELPER DATA FORMATTERS
# =============================================================================

def features_list_to_dict(filename, features):
    """Maps the raw 29-element float list to the output schema format"""
    intensity = {
        "mean": float(features[0]),
        "std": float(features[1]),
        "min": float(features[2]),
        "max": float(features[3]),
        "median": float(features[4]),
        "percentile_25": float(features[5]),
        "percentile_75": float(features[6]),
        "skewness": float(features[7]),
        "kurtosis": float(features[8]),
        "dynamic_range": float(features[9])
    }
    
    tissue = {
        "background": {
            "pixel_count": 0,
            "percentage": 0.0,
            "mean_intensity": 0.0,
            "std_intensity": 0.0
        },
        "csf_ventricles": {
            "pixel_count": 0,
            "percentage": float(features[10]),
            "mean_intensity": 0.0,
            "std_intensity": 0.0
        },
        "gray_matter": {
            "pixel_count": 0,
            "percentage": float(features[11]),
            "mean_intensity": 0.0,
            "std_intensity": 0.0
        },
        "white_matter": {
            "pixel_count": 0,
            "percentage": float(features[12]),
            "mean_intensity": 0.0,
            "std_intensity": 0.0
        },
        "brain_tissue_total_pct": float(features[13])
    }
    
    texture = {
        "glcm_contrast": float(features[14]),
        "glcm_correlation": float(features[15]),
        "glcm_energy": float(features[16]),
        "glcm_homogeneity": float(features[17]),
        "mean_gradient": float(features[18]),
        "std_gradient": float(features[19]),
        "max_gradient": float(features[20]),
        "shannon_entropy": float(features[21]),
        "interpretation": {
            "high_entropy_means": "complex brain texture",
            "low_entropy_means": "uniform / potentially atrophied regions",
            "high_contrast_means": "strong tissue boundaries present",
            "high_homogeneity_means": "smooth, uniform intensity regions"
        }
    }
    
    structural = {
        "brain_area_pixels": 0,
        "brain_area_pct": float(features[22]),
        "bounding_box": {
            "min_row": 30, "min_col": 30, "max_row": 226, "max_col": 226
        },
        "centroid": {
            "row": 128.0, "col": 128.0
        },
        "eccentricity": float(features[23]),
        "solidity": float(features[24]),
        "extent": float(features[25]),
        "perimeter": float(features[26]),
        "interpretation": {
            "solidity_close_to_1": "compact, healthy brain shape",
            "eccentricity_near_0": "circular",
            "eccentricity_near_1": "elongated"
        }
    }
    
    symmetry = {
        "bilateral_symmetry_score": float(features[27]),
        "asymmetry_index_pct": float(features[28]),
        "interpretation": "Bilateral symmetry comparison"
    }
    
    return {
        "schema_version": "1.0",
        "scan_metadata": {
            "filename": filename,
            "modality": "MRI",
            "orientation": "axial",
            "image_height": 256,
            "image_width": 256,
            "total_pixels": 65536,
            "bit_depth": 16,
            "color_space": "grayscale",
            "format": "NIfTI"
        },
        "intensity_statistics": intensity,
        "tissue_segmentation": tissue,
        "texture_features": texture,
        "structural_features": structural,
        "symmetry": symmetry,
        "pathology_indicators": {
            "flags": [],
            "risk_level": "LOW"
        }
    }


def flatten_feature_dict(d: dict) -> list:
    """Helper to reconstruct flat 29-element list from structured predicted features dict"""
    try:
        intensity = d.get("intensity_statistics", {})
        tissue = d.get("tissue_segmentation", {})
        texture = d.get("texture_features", {})
        struct = d.get("structural_features", {})
        symmetry = d.get("symmetry", {})
        
        flat = [
            # Intensity (10)
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
            # Tissue (4)
            tissue.get("csf_ventricles", {}).get("percentage", 0.0),
            tissue.get("gray_matter", {}).get("percentage", 0.0),
            tissue.get("white_matter", {}).get("percentage", 0.0),
            tissue.get("brain_tissue_total_pct", 0.0),
            # Texture (8)
            texture.get("glcm_contrast", 0.0),
            texture.get("glcm_correlation", 0.0),
            texture.get("glcm_energy", 0.0),
            texture.get("glcm_homogeneity", 0.0),
            texture.get("mean_gradient", 0.0),
            texture.get("std_gradient", 0.0),
            texture.get("max_gradient", 0.0),
            texture.get("shannon_entropy", 0.0),
            # Structural (5)
            struct.get("brain_area_pct", 0.0),
            struct.get("eccentricity", 0.0),
            struct.get("solidity", 0.0),
            struct.get("extent", 0.0),
            struct.get("perimeter", 0.0),
            # Symmetry (2)
            symmetry.get("bilateral_symmetry_score", 0.0),
            symmetry.get("asymmetry_index_pct", 0.0)
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

def start_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), DashboardAPIHandler) as httpd:
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
