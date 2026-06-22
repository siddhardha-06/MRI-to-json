"""
Extract Rule-Based Features from MRI Pipeline Output
Converts nested JSON into flat feature vectors for ML training

Input: MRI files (.nii)
Output: rule_based_features.json with 29 features per MRI
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
    """Load .nii MRI file and extract middle axial slice"""
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


def extract_intensity_stats(arr: np.ndarray) -> dict:
    """10 intensity features"""
    flat = arr.flatten()
    return {
        "mean":          float(np.mean(flat)),
        "std":           float(np.std(flat)),
        "min":           float(np.min(flat)),
        "max":           float(np.max(flat)),
        "median":        float(np.median(flat)),
        "percentile_25": float(np.percentile(flat, 25)),
        "percentile_75": float(np.percentile(flat, 75)),
        "skewness":      float(stats.skew(flat)),
        "kurtosis":      float(stats.kurtosis(flat)),
        "dynamic_range": float(np.max(flat) - np.min(flat))
    }


def extract_tissue_segmentation(arr: np.ndarray) -> dict:
    """4 tissue features"""
    total = arr.size
    
    bg_mask  = arr < 0.05
    csf_mask = (arr >= 0.05) & (arr < 0.25)
    gm_mask  = (arr >= 0.25) & (arr < 0.55)
    wm_mask  = arr >= 0.55
    
    return {
        "csf_pct":           float((csf_mask.sum() / total) * 100),
        "gm_pct":            float((gm_mask.sum() / total) * 100),
        "wm_pct":            float((wm_mask.sum() / total) * 100),
        "brain_tissue_pct":  float(((gm_mask.sum() + wm_mask.sum()) / total) * 100)
    }


def extract_texture_features(arr: np.ndarray) -> dict:
    """8 texture features"""
    # GLCM
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
    
    # Gradient
    grad_y, grad_x = np.gradient(arr)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Entropy
    hist, _ = np.histogram(arr, bins=16, range=(0, 1), density=True)
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
    """5 structural features"""
    brain_mask = arr > 0.05
    labeled = measure.label(brain_mask)
    regions = measure.regionprops(labeled)
    
    if not regions:
        return {
            "brain_area_pct": 0.0,
            "eccentricity": 0.0,
            "solidity": 0.0,
            "extent": 0.0,
            "perimeter": 0.0
        }
    
    brain = max(regions, key=lambda r: r.area)
    
    return {
        "brain_area_pct":  float((brain.area / arr.size) * 100),
        "eccentricity":    float(brain.eccentricity),
        "solidity":        float(brain.solidity),
        "extent":          float(brain.extent),
        "perimeter":       float(brain.perimeter)
    }


def extract_symmetry_features(arr: np.ndarray) -> dict:
    """2 symmetry features"""
    mid_col = arr.shape[1] // 2
    
    left_half = arr[:, :mid_col]
    right_half = arr[:, mid_col:]
    
    # Ensure same size
    min_size = min(left_half.shape[1], right_half.shape[1])
    left_half = left_half[:, :min_size]
    right_half = right_half[:, :min_size]
    
    # Flip right half for comparison
    right_half_flipped = np.fliplr(right_half)
    
    # Correlation as symmetry score
    correlation = np.corrcoef(left_half.flatten(), right_half_flipped.flatten())[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    
    symmetry_score = (correlation + 1) / 2  # Normalize to 0-1
    asymmetry_index = abs(left_half.mean() - right_half.mean()) * 100
    
    return {
        "bilateral_symmetry_score": float(max(0, min(1, symmetry_score))),
        "asymmetry_index_pct":      float(asymmetry_index)
    }


def extract_all_features(mri_path: str) -> list:
    """
    Extract all 29 features from MRI file
    Returns: [feat1, feat2, ..., feat29]
    """
    arr = load_mri(mri_path)
    
    intensity = extract_intensity_stats(arr)
    tissue = extract_tissue_segmentation(arr)
    texture = extract_texture_features(arr)
    structural = extract_structural_features(arr)
    symmetry = extract_symmetry_features(arr)
    
    # Flatten into single list in order
    features = [
        # Intensity (10)
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
        # Tissue (4)
        tissue["csf_pct"],
        tissue["gm_pct"],
        tissue["wm_pct"],
        tissue["brain_tissue_pct"],
        # Texture (8)
        texture["glcm_contrast"],
        texture["glcm_correlation"],
        texture["glcm_energy"],
        texture["glcm_homogeneity"],
        texture["mean_gradient"],
        texture["std_gradient"],
        texture["max_gradient"],
        texture["shannon_entropy"],
        # Structural (5)
        structural["brain_area_pct"],
        structural["eccentricity"],
        structural["solidity"],
        structural["extent"],
        structural["perimeter"],
        # Symmetry (2)
        symmetry["bilateral_symmetry_score"],
        symmetry["asymmetry_index_pct"],
    ]
    
    return features


# =============================================================================
# BATCH EXTRACTION
# =============================================================================

def extract_features_for_all_mris(
    mri_dir="ixi-t1",
    output_file="rule_based_features.json"
):
    """Extract features for all MRI files and save to JSON"""
    
    mri_dir = Path(mri_dir)
    mri_files = sorted(mri_dir.glob("*/*.nii"))
    
    print(f"Found {len(mri_files)} MRI files\n")
    
    features_dict = {}
    errors = []
    
    for mri_file in tqdm(mri_files, desc="Extracting features"):
        try:
            features = extract_all_features(str(mri_file))
            features_dict[mri_file.stem] = features
        except Exception as e:
            errors.append((mri_file.stem, str(e)))
            print(f"\n⚠️  Error processing {mri_file.stem}: {e}")
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(features_dict, f, indent=2)
    
    print(f"\n✓ Extracted features for {len(features_dict)} MRI files")
    print(f"✓ Features saved to: {output_file}")
    
    if errors:
        print(f"\n⚠️  {len(errors)} files had errors:")
        for stem, error in errors[:5]:
            print(f"   {stem}: {error}")
        if len(errors) > 5:
            print(f"   ... and {len(errors)-5} more")
    
    # Print sample
    sample_key = list(features_dict.keys())[0]
    print(f"\nSample feature vector (first 5 features of {len(features_dict[sample_key])}): {features_dict[sample_key][:5]}")
    
    return features_dict


# =============================================================================
# FEATURE SUMMARY
# =============================================================================

def print_feature_schema():
    """Print feature order and descriptions"""
    print("\n" + "="*70)
    print("Feature Schema (29 total features)")
    print("="*70)
    print("\nIntensity Statistics (10):")
    print("  0. mean, 1. std, 2. min, 3. max, 4. median")
    print("  5. percentile_25, 6. percentile_75, 7. skewness, 8. kurtosis, 9. dynamic_range")
    
    print("\nTissue Segmentation (4):")
    print("  10. CSF%, 11. GM%, 12. WM%, 13. brain_tissue_total%")
    
    print("\nTexture Features (8):")
    print("  14. GLCM contrast, 15. GLCM correlation, 16. GLCM energy, 17. GLCM homogeneity")
    print("  18. mean_gradient, 19. std_gradient, 20. max_gradient, 21. shannon_entropy")
    
    print("\nStructural Features (5):")
    print("  22. brain_area%, 23. eccentricity, 24. solidity, 25. extent, 26. perimeter")
    
    print("\nSymmetry Features (2):")
    print("  27. bilateral_symmetry_score, 28. asymmetry_index%")
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
        print_feature_schema()
        print()
        extract_features_for_all_mris(args.mri_dir, args.output)
