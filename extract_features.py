"""
Extract Rule-Based Features from MRI Pipeline Output
Converts nested JSON into flat feature vectors for ML training

Input: MRI files (.nii)
Output: rule_based_features.json with 61 features per MRI

Feature order (61 total):
  0-11:  Intensity Statistics (12) — computed on brain-masked voxels (arr>0.05)
  12-18: Tissue Segmentation   (7) — adaptive multi-Otsu on brain voxels
  19-26: Texture Features      (8) — GLCM on brain bounding box crop
  27-32: Structural Features   (6) — shape morphology of largest brain region
  33-34: Symmetry              (2) — bilateral correlation & asymmetry index
  35-37: Frequency Domain      (3) — 2-D FFT low/high energy ratio
  38-39: Edge Features         (2) — Sobel sharpness on brain-masked edges
  40-60: Subcortical Volumes   (21) — 3D Harvard-Oxford atlas segmentations
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
from scipy import stats
from skimage import measure, filters, feature
from skimage.feature import graycomatrix, graycoprops


# =============================================================================
# FEATURE EXTRACTION (from first-pipeline.py, optimized for .nii files)
# =============================================================================

def load_mri(path: str) -> np.ndarray:
    """Load .nii/.nii.gz MRI file and extract middle axial slice using percentile clipping"""
    img = nib.load(path)
    data = img.get_fdata()
    img.uncache()  # Release file descriptors to prevent stale file handles
    
    # Take middle slice
    mid = data.shape[2] // 2
    arr = data[:, :, mid].astype(np.float32)
    
    # Percentile-clipped normalization (consistent with autoencoder and regressor)
    p1, p99 = np.percentile(arr, (1, 99))
    if p99 - p1 > 0:
        arr = np.clip(arr, p1, p99)
        arr = (arr - p1) / (p99 - p1 + 1e-5)
    else:
        arr = np.zeros_like(arr)
        
    return arr


def extract_intensity_stats(arr: np.ndarray) -> dict:
    """12 intensity features computed on brain-masked voxels only (arr > 0.05)"""
    brain_mask = arr > 0.05
    flat = arr[brain_mask].flatten() if brain_mask.any() else np.array([0.0])
    
    # Mode estimation using histogram bin peak
    if len(flat) > 0:
        counts, bin_edges = np.histogram(flat, bins=100)
        mode_val = float(bin_edges[np.argmax(counts)])
    else:
        mode_val = 0.0
        
    skew = stats.skew(flat) if len(flat) > 2 else 0.0
    kurt = stats.kurtosis(flat) if len(flat) > 2 else 0.0
    bimodality = (skew**2 + 1) / (kurt + 3) if (kurt + 3) > 0 else 0.0
    
    return {
        "mean":          float(np.mean(flat)),
        "std":           float(np.std(flat)),
        "min":           float(np.min(flat)),
        "max":           float(np.max(flat)),
        "median":        float(np.median(flat)),
        "percentile_25": float(np.percentile(flat, 25)),
        "percentile_75": float(np.percentile(flat, 75)),
        "skewness":      skew,
        "kurtosis":      kurt,
        "dynamic_range": float(np.max(flat) - np.min(flat)),
        "mode":          mode_val,
        "bimodality":    bimodality
    }


def extract_tissue_segmentation(arr: np.ndarray) -> dict:
    """7 tissue features using adaptive multi-Otsu thresholding and GM/WM CNR"""
    total = arr.size
    brain_mask = arr > 0.05
    brain_voxels = arr[brain_mask]
    
    # Adaptive thresholding using multi-Otsu
    try:
        from skimage.filters import threshold_multiotsu
        thresholds = threshold_multiotsu(brain_voxels, classes=3)
        t1, t2 = thresholds[0], thresholds[1]
    except Exception:
        t1, t2 = 0.25, 0.55
        
    bg_mask = arr <= 0.05
    csf_mask = (arr > 0.05) & (arr < t1)
    gm_mask = (arr >= t1) & (arr < t2)
    wm_mask = (arr >= t2)
    
    gm_voxels = arr[gm_mask]
    wm_voxels = arr[wm_mask]
    if len(gm_voxels) > 0 and len(wm_voxels) > 0:
        cnr = abs(gm_voxels.mean() - wm_voxels.mean()) / (wm_voxels.std() + 1e-5)
    else:
        cnr = 0.0
        
    brain_total = float((arr > 0.05).sum())
    if brain_total == 0:
        brain_total = 1.0
        
    return {
        "csf_pct":           float((csf_mask.sum() / brain_total) * 100),
        "gm_pct":            float((gm_mask.sum() / brain_total) * 100),
        "wm_pct":            float((wm_mask.sum() / brain_total) * 100),
        "brain_tissue_pct":  float(((gm_mask.sum() + wm_mask.sum()) / brain_total) * 100),
        "otsu_threshold_1":  float(t1),
        "otsu_threshold_2":  float(t2),
        "gm_wm_cnr":         float(cnr)
    }


def extract_texture_features(arr: np.ndarray) -> dict:
    """8 texture features computed on brain bounding box cropped slice"""
    # Crop slice to brain bounding box to exclude background dominance in GLCM
    brain_mask = arr > 0.05
    if brain_mask.any():
        rows = np.any(brain_mask, axis=1)
        cols = np.any(brain_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        arr_cropped = arr[rmin:rmax+1, cmin:cmax+1]
    else:
        arr_cropped = arr
        
    # GLCM
    arr_u8 = (arr_cropped * 255).astype(np.uint8)
    arr_binned = (arr_u8 // 32).astype(np.uint8)
    
    glcm = graycomatrix(
        arr_binned,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=8,
        symmetric=True,
        normed=True
    )
    
    # Gradient
    grad_y, grad_x = np.gradient(arr_cropped)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Entropy (calculated as probability distribution over brain-masked voxels)
    brain_mask_crop = arr_cropped > 0.05
    brain_voxels = arr_cropped[brain_mask_crop] if brain_mask_crop.any() else arr_cropped.flatten()
    hist, _ = np.histogram(brain_voxels, bins=16, range=(0, 1))
    hist = hist / (hist.sum() + 1e-10)
    hist = hist + 1e-10
    entropy = float(-np.sum(hist * np.log2(hist)))
    
    return {
        "glcm_contrast":    float(graycoprops(glcm, 'contrast').mean()),
        "glcm_correlation": float(graycoprops(glcm, 'correlation').mean()),
        "glcm_energy":      float(graycoprops(glcm, 'energy').mean()),
        "glcm_homogeneity": float(graycoprops(glcm, 'homogeneity').mean()),
        "mean_gradient":    float(grad_mag.mean()),
        "std_gradient":     float(grad_mag.std()),
        "max_gradient":     float(grad_mag.max()),
        "shannon_entropy":  entropy
    }


def extract_structural_features(arr: np.ndarray) -> dict:
    """6 structural features with try/except guards for regional properties and connected CSF ventricle proxy"""
    brain_mask = arr > 0.05
    labeled = measure.label(brain_mask)
    regions = measure.regionprops(labeled)
    
    if not regions:
        return {
            "brain_area_pct": 0.0,
            "eccentricity": 0.0,
            "solidity": 0.0,
            "extent": 0.0,
            "perimeter": 0.0,
            "ventricular_proxy_pct": 0.0
        }
    
    brain = max(regions, key=lambda r: r.area)
    
    # Compute t1 using Otsu to identify CSF ventricular components
    try:
        from skimage.filters import threshold_multiotsu
        thresholds = threshold_multiotsu(arr[arr > 0.05], classes=3)
        t1 = thresholds[0]
    except Exception:
        t1 = 0.25
        
    brain_total = float((arr > 0.05).sum())
    if brain_total == 0:
        brain_total = 1.0
        
    csf_mask = (arr > 0.05) & (arr < t1)
    labeled_csf = measure.label(csf_mask)
    regions_csf = measure.regionprops(labeled_csf)
    if regions_csf:
        largest_ventricle = max(regions_csf, key=lambda r: r.area)
        ventricular_proxy_pct = float((largest_ventricle.area / brain_total) * 100)
    else:
        ventricular_proxy_pct = 0.0
        
    # Safely retrieve properties (skimage version compatibility)
    def get_prop(obj, prop_name, default=0.0):
        try:
            return float(getattr(obj, prop_name))
        except Exception:
            return default
            
    return {
        "brain_area_pct":  float((brain.area / arr.size) * 100),
        "eccentricity":    get_prop(brain, "eccentricity"),
        "solidity":        get_prop(brain, "solidity"),
        "extent":          get_prop(brain, "extent"),
        "perimeter":       get_prop(brain, "perimeter"),
        "ventricular_proxy_pct": ventricular_proxy_pct
    }


def extract_symmetry_features(arr: np.ndarray) -> dict:
    """2 symmetry features using brain-masked left/right halves"""
    mid_col = arr.shape[1] // 2
    
    left_half = arr[:, :mid_col]
    right_half = arr[:, mid_col:]
    
    # Ensure same size
    min_size = min(left_half.shape[1], right_half.shape[1])
    left_half = left_half[:, :min_size]
    right_half = right_half[:, :min_size]
    
    left_brain = left_half[left_half > 0.05]
    right_brain = right_half[right_half > 0.05]
    overall_brain = arr[arr > 0.05]
    
    if len(left_brain) > 0 and len(right_brain) > 0 and len(overall_brain) > 0:
        asymmetry_index = abs(left_brain.mean() - right_brain.mean()) / (overall_brain.mean() + 1e-5) * 100
    else:
        asymmetry_index = 0.0
        
    # Flip right half for comparison
    right_half_flipped = np.fliplr(right_half)
    
    # Correlation as symmetry score
    correlation = np.corrcoef(left_half.flatten(), right_half_flipped.flatten())[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    
    symmetry_score = (correlation + 1) / 2  # Normalize to 0-1
    
    return {
        "bilateral_symmetry_score": float(max(0, min(1, symmetry_score))),
        "asymmetry_index_pct":      float(asymmetry_index)
    }


def extract_frequency_features(arr: np.ndarray) -> dict:
    """3 frequency domain features using 2D Fourier Transform"""
    H, W = arr.shape
    fft = np.fft.fft2(arr)
    magnitude = np.abs(np.fft.fftshift(fft))
    
    low_freq_energy = float(magnitude[:H//4, :W//4].mean())
    high_freq_energy = float(magnitude[H//2:, W//2:].mean())
    ratio = low_freq_energy / (high_freq_energy + 1e-5)
    
    return {
        "low_frequency_energy": low_freq_energy,
        "high_frequency_energy": high_freq_energy,
        "frequency_energy_ratio": ratio
    }


def extract_edge_features(arr: np.ndarray) -> dict:
    """2 edge sharpness features using Sobel filter over brain-masked regions"""
    from skimage.filters import sobel
    edges = sobel(arr)
    brain_mask = arr > 0.05
    brain_edges = edges[brain_mask] if brain_mask.any() else np.array([0.0])
    
    return {
        "cortical_edge_sharpness": float(brain_edges.mean()),
        "cortical_edge_std":       float(brain_edges.std())
    }


def _n4_file_is_valid(path: Path) -> bool:
    """Read NIfTI header (fast) and check if file size matches expected shape+dtype"""
    try:
        img = nib.load(str(path))
        expected_data_bytes = int(np.prod(img.shape)) * np.dtype(img.get_data_dtype()).itemsize
        actual_size = path.stat().st_size
        offset = int(img.header.get('vox_offset', 352))
        actual_data_bytes = actual_size - offset
        img.uncache()
        return actual_data_bytes >= expected_data_bytes * 0.99
    except Exception:
        return False


def _resolve_mri_path(mri_path: str, mask_dir: str = "ixi-t1-masks") -> str:
    """Returns the N4-corrected MRI path if it exists and is valid, otherwise raw path"""
    if not mask_dir:
        return mri_path
    path_obj = Path(mri_path)
    stem = path_obj.name.replace(".nii.gz", "").replace(".nii", "")
    n4_candidate = Path(mask_dir) / "n4" / f"{stem}_n4.nii"
    if n4_candidate.exists() and _n4_file_is_valid(n4_candidate):
        return str(n4_candidate)
    return mri_path


def extract_subcortical_features(mri_path: str) -> list:
    """Extract 21 subcortical structure features (left/right volume and asymmetry) using ANTsPy registration"""
    try:
        import ants
        workspace_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(workspace_dir, "checkpoints", "MNI152_T1_2mm_brain.nii.gz")
        labels_path = os.path.join(workspace_dir, "checkpoints", "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz")
        
        if not os.path.exists(template_path) or not os.path.exists(labels_path):
            return [0.0] * 21
            
        fixed_img = ants.image_read(mri_path)
        moving_img = ants.image_read(template_path)
        labels = ants.image_read(labels_path)
        
        reg = ants.registration(fixed=fixed_img, moving=moving_img, type_of_transform='Affine')
        warped_labels = ants.apply_transforms(
            fixed=fixed_img,
            moving=labels,
            transformlist=reg['fwdtransforms'],
            interpolation='nearestNeighbor'
        )
        warped_arr = warped_labels.numpy()
        
        voxel_vol_ml = (fixed_img.spacing[0] * fixed_img.spacing[1] * fixed_img.spacing[2]) / 1000.0
        
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
        
        return [
            left_vent, right_vent, vent_ai,
            left_thal, right_thal, thal_ai,
            left_caud, right_caud, caud_ai,
            left_puta, right_puta, puta_ai,
            left_pall, right_pall, pall_ai,
            left_hippo, right_hippo, hippo_ai,
            left_amyg, right_amyg, amyg_ai
        ]
    except Exception as e:
        print(f"[WARNING] Failed to extract subcortical features for {mri_path}: {e}")
        return [0.0] * 21


def extract_all_features(mri_path: str, mask_dir: str = "ixi-t1-masks") -> list:
    """
    Extract and average all 61 features across 5 central slices of the MRI file
    Returns: [feat1, feat2, ..., feat61]
    """
    resolved_path = _resolve_mri_path(mri_path, mask_dir)
    if resolved_path != mri_path:
        print(f"  [INFO] Extracting features from resolved N4-corrected path: {Path(resolved_path).name}")
        
    img = nib.load(resolved_path)
    data = img.get_fdata().astype(np.float32)
    img.uncache()
    
    D = data.shape[2]
    mid = D // 2
    
    slices_features = []
    
    # 5 central slices (consistent with regressor multi-slice training and inference)
    for offset in [-2, -1, 0, 1, 2]:
        slice_idx = mid + offset
        if 0 <= slice_idx < D:
            arr = data[:, :, slice_idx]
            
            # Preprocess using percentile clipping
            p1, p99 = np.percentile(arr, (1, 99))
            if p99 - p1 > 0:
                arr = np.clip(arr, p1, p99)
                arr = (arr - p1) / (p99 - p1 + 1e-5)
            else:
                arr = np.zeros_like(arr)
                
            intensity = extract_intensity_stats(arr)
            tissue = extract_tissue_segmentation(arr)
            texture = extract_texture_features(arr)
            structural = extract_structural_features(arr)
            symmetry = extract_symmetry_features(arr)
            frequency = extract_frequency_features(arr)
            edge = extract_edge_features(arr)
            
            # Flatten into single list of 40 features in order
            slice_vector = [
                # Intensity Statistics (12)
                intensity["mean"],
                intensity["std"],
                intensity["min"],
                intensity["max"],
                intensity["median"],
                intensity["percentile_25"],
                intensity["percentile_75"],
                intensity["skewness"],
                intensity["kurtosis"],
                intensity["dynamic_range"],
                intensity["mode"],
                intensity["bimodality"],
                # Tissue Segmentation (7)
                tissue["csf_pct"],
                tissue["gm_pct"],
                tissue["wm_pct"],
                tissue["brain_tissue_pct"],
                tissue["otsu_threshold_1"],
                tissue["otsu_threshold_2"],
                tissue["gm_wm_cnr"],
                # Texture Features (8)
                texture["glcm_contrast"],
                texture["glcm_correlation"],
                texture["glcm_energy"],
                texture["glcm_homogeneity"],
                texture["mean_gradient"],
                texture["std_gradient"],
                texture["max_gradient"],
                texture["shannon_entropy"],
                # Structural Features (6)
                structural["brain_area_pct"],
                structural["eccentricity"],
                structural["solidity"],
                structural["extent"],
                structural["perimeter"],
                structural["ventricular_proxy_pct"],
                # Symmetry Features (2)
                symmetry["bilateral_symmetry_score"],
                symmetry["asymmetry_index_pct"],
                # Frequency Domain (3)
                frequency["low_frequency_energy"],
                frequency["high_frequency_energy"],
                frequency["frequency_energy_ratio"],
                # Edge Features (2)
                edge["cortical_edge_sharpness"],
                edge["cortical_edge_std"]
            ]
            slices_features.append(slice_vector)
            
    if not slices_features:
        return [0.0] * 61
        
    slice_avg = np.mean(slices_features, axis=0).tolist()
    subcortical_feats = extract_subcortical_features(resolved_path)
    return slice_avg + subcortical_feats


# =============================================================================
# BATCH EXTRACTION
# =============================================================================

def extract_features_for_all_mris(
    mri_dir="ixi-t1",
    output_file="rule_based_features.json"
):
    """Extract features for all MRI files and save to JSON with incremental resume capability"""
    
    mri_dir = Path(mri_dir)
    mri_files = sorted(
        list(mri_dir.rglob("*.nii")) +
        list(mri_dir.rglob("*.nii.gz"))
    )
    mri_files = [f for f in mri_files if f.is_file()]
    
    print(f"Found {len(mri_files)} MRI files\n")
    
    features_dict = {}
    if Path(output_file).exists():
        try:
            with open(output_file, 'r') as f:
                loaded = json.load(f)
                # Only keep loaded samples that match the expected 61 features length
                features_dict = {k: v for k, v in loaded.items() if isinstance(v, list) and len(v) == 61}
                if features_dict:
                    print(f"Resuming feature extraction: found {len(features_dict)} existing 61-feature entries in {output_file}")
        except Exception as e:
            print(f"[WARNING] Could not load existing features for resume: {e}")
            features_dict = {}
            
    errors = []
    new_extracted_count = 0
    
    for mri_file in tqdm(mri_files, desc="Extracting features"):
        if mri_file.stem in features_dict:
            continue
            
        try:
            features = extract_all_features(str(mri_file))
            features_dict[mri_file.stem] = features
            new_extracted_count += 1
            
            # Periodically save every 10 samples to prevent data loss on interruption
            if new_extracted_count % 10 == 0:
                temp_output = output_file + ".tmp"
                with open(temp_output, 'w') as f:
                    json.dump(features_dict, f, indent=2)
                os.replace(temp_output, output_file)
        except Exception as e:
            errors.append((mri_file.stem, str(e)))
            print(f"\n[WARNING] Error processing {mri_file.stem}: {e}")
    
    # Final save
    temp_output = output_file + ".tmp"
    with open(temp_output, 'w') as f:
        json.dump(features_dict, f, indent=2)
    os.replace(temp_output, output_file)
    
    print(f"\n[SUCCESS] Total features dataset size: {len(features_dict)} MRI files")
    if new_extracted_count > 0:
        print(f"[SUCCESS] Extracted {new_extracted_count} new files in this run")
    print(f"[SUCCESS] Features saved to: {output_file}")
    
    if errors:
        print(f"\n[WARNING] {len(errors)} files had errors:")
        for stem, error in errors[:5]:
            print(f"   {stem}: {error}")
        if len(errors) > 5:
            print(f"   ... and {len(errors)-5} more")
    
    # Print sample
    if features_dict:
        sample_key = list(features_dict.keys())[0]
        print(f"\nSample feature vector (first 5 features of {len(features_dict[sample_key])}): {features_dict[sample_key][:5]}")
    
    return features_dict


# =============================================================================
# FEATURE SUMMARY
# =============================================================================

def print_feature_schema():
    """Print feature order and descriptions"""
    print("\n" + "="*70)
    print("Feature Schema (61 total features)")
    print("="*70)
    print("\nIntensity Statistics (12):")
    print("  0. mean, 1. std, 2. min, 3. max, 4. median")
    print("  5. percentile_25, 6. percentile_75, 7. skewness, 8. kurtosis, 9. dynamic_range")
    print("  10. mode, 11. bimodality")
    
    print("\nTissue Segmentation (7):")
    print("  12. CSF%, 13. GM%, 14. WM%, 15. brain_tissue_total%")
    print("  16. otsu_threshold_1, 17. otsu_threshold_2, 18. gm_wm_cnr")
    
    print("\nTexture Features (8):")
    print("  19. GLCM contrast, 20. GLCM correlation, 21. GLCM energy, 22. GLCM homogeneity")
    print("  23. mean_gradient, 24. std_gradient, 25. max_gradient, 26. shannon_entropy")
    
    print("\nStructural Features (6):")
    print("  27. brain_area%, 28. eccentricity, 29. solidity, 30. extent, 31. perimeter")
    print("  32. ventricular_proxy_pct")
    
    print("\nSymmetry Features (2):")
    print("  33. bilateral_symmetry_score, 34. asymmetry_index%")
    
    print("\nFrequency Domain (3):")
    print("  35. low_frequency_energy, 36. high_frequency_energy, 37. frequency_energy_ratio")
    
    print("\nEdge Features (2):")
    print("  38. cortical_edge_sharpness, 39. cortical_edge_std")
    
    print("\nSubcortical Volumes (21):")
    print("  40. L_ventricle, 41. R_ventricle, 42. ventricle_AI")
    print("  43. L_thalamus, 44. R_thalamus, 45. thalamus_AI")
    print("  46. L_caudate, 47. R_caudate, 48. caudate_AI")
    print("  49. L_putamen, 50. R_putamen, 51. putamen_AI")
    print("  52. L_pallidum, 53. R_pallidum, 54. pallidum_AI")
    print("  55. L_hippocampus, 56. R_hippocampus, 57. hippocampus_AI")
    print("  58. L_amygdala, 59. R_amygdala, 60. amygdala_AI")
    print("\n" + "="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract rule-based features from MRI files")
    parser.add_argument("--mri-dir", default="ixi-t1", help="MRI directory")
    parser.add_argument("--output", default="rule_based_features.json", help="Output JSON file")
    parser.add_argument("--schema", action="store_true", help="Print feature schema and exit")
    
    args = parser.parse_args()
    
    if args.schema:
        print_feature_schema()
    else:
        extract_features_for_all_mris(args.mri_dir, args.output)
