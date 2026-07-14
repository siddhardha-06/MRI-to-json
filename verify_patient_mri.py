import json
import os
import numpy as np
import nibabel as nib
from scipy import stats

def verify_patient(patient_id):
    print(f"=== NeuroSphere MRI Statistics Verification ===")
    print(f"Target Patient ID: {patient_id}")
    
    # 1. Paths
    upload_dir = os.path.join("uploads", patient_id)
    nii_path = os.path.join(upload_dir, f"{patient_id}.nii")
    features_json = os.path.join(upload_dir, "features.json")
    nii_n4 = os.path.join(upload_dir, f"{patient_id}_n4.nii")
    if os.path.exists(nii_n4):
        print("Using N4 corrected MRI scan for statistics verification.")
        nii_path = nii_n4
    elif not os.path.exists(nii_path):
        print(f"Error: Raw NIfTI file not found at {nii_path}")
        return
    if not os.path.exists(features_json):
        print(f"Error: Extracted features JSON not found at {features_json}")
        return
        
    # 2. Load JSON features
    with open(features_json, "r") as f:
        cached = json.load(f)
    
    cached_intensity = cached.get("intensity_statistics", {})
    cached_volumes = cached.get("volumetric_statistics", {})
    
    # 3. Compute statistics from the NIfTI file
    print("\nLoading and normalizing NIfTI file...")
    img = nib.load(nii_path)
    data = img.get_fdata().astype(np.float32)
    
    D = data.shape[2]
    mid_idx = D // 2
    
    means = []
    stds = []
    mins = []
    maxs = []
    medians = []
    skews = []
    kurts = []
    
    # Extract and average statistics across the 5 central slices
    for offset in [-2, -1, 0, 1, 2]:
        slice_idx = mid_idx + offset
        if 0 <= slice_idx < D:
            slice_2d = data[:, :, slice_idx]
            
            # Percentile-clipped normalization to [0, 1] range
            p1, p99 = np.percentile(slice_2d, (1, 99))
            if p99 - p1 > 0:
                norm_slice = np.clip(slice_2d, p1, p99)
                norm_slice = (norm_slice - p1) / (p99 - p1 + 1e-5)
            else:
                norm_slice = np.zeros_like(slice_2d)
                
            # Filter out background noise (mask > 0.05)
            brain_mask = norm_slice > 0.05
            flat_voxels = norm_slice[brain_mask].flatten() if brain_mask.any() else np.array([0.0])
            
            # Compute metrics for this slice
            means.append(float(np.mean(flat_voxels)))
            stds.append(float(np.std(flat_voxels)))
            mins.append(float(np.min(flat_voxels)))
            maxs.append(float(np.max(flat_voxels)))
            medians.append(float(np.median(flat_voxels)))
            skews.append(float(stats.skew(flat_voxels)) if len(flat_voxels) > 2 else 0.0)
            kurts.append(float(stats.kurtosis(flat_voxels)) if len(flat_voxels) > 2 else 0.0)
            
    calc_mean = float(np.mean(means))
    calc_std = float(np.mean(stds))
    calc_min = float(np.mean(mins))
    calc_max = float(np.mean(maxs))
    calc_median = float(np.mean(medians))
    calc_skew = float(np.mean(skews))
    calc_kurt = float(np.mean(kurts))
    
    # 4. Compare side-by-side
    print("\n" + "="*70)
    print(f"{'METRIC':<20} | {'DASHBOARD VALUE':<18} | {'VERIFIED VALUE':<18} | {'STATUS'}")
    print("="*70)
    
    def print_row(name, cached_val, calc_val):
        diff = abs(cached_val - calc_val)
        status = "MATCH" if diff < 1e-4 else "MISMATCH"
        print(f"{name:<20} | {cached_val:<18.4f} | {calc_val:<18.4f} | {status}")
        
    print_row("Mean Intensity", cached_intensity.get("mean", 0.0), calc_mean)
    print_row("Std Deviation", cached_intensity.get("std", 0.0), calc_std)
    print_row("Min Intensity", cached_intensity.get("min", 0.0), calc_min)
    print_row("Max Intensity", cached_intensity.get("max", 0.0), calc_max)
    print_row("Median Intensity", cached_intensity.get("median", 0.0), calc_median)
    print_row("Skewness", cached_intensity.get("skewness", 0.0), calc_skew)
    print_row("Kurtosis", cached_intensity.get("kurtosis", 0.0), calc_kurt)
    
    # 5. Volumetric validation info
    print("\n" + "="*70)
    print("3D VOLUMETRIC ANALYSIS VALIDATION DETAILS:")
    print("="*70)
    voxel_dims = img.header.get_zooms()[:3]
    voxel_vol_ml = (voxel_dims[0] * voxel_dims[1] * voxel_dims[2]) / 1000.0
    print(f"Voxel Dimensions: {voxel_dims[0]}mm x {voxel_dims[1]}mm x {voxel_dims[2]}mm")
    print(f"Unit Voxel Volume: {voxel_vol_ml:.6f} mL")
    
    if cached_volumes:
        print(f"Total Brain Volume: {cached_volumes.get('total_brain_volume_ml', 0.0):.2f} mL")
        print(f"Gray Matter Volume: {cached_volumes.get('gray_matter_volume_ml', 0.0):.2f} mL")
        print(f"White Matter Volume: {cached_volumes.get('white_matter_volume_ml', 0.0):.2f} mL")
        print(f"CSF Volume:         {cached_volumes.get('csf_volume_ml', 0.0):.2f} mL")
        print(f"Left Hippocampus:   {cached_volumes.get('left_hippocampus_volume_ml', 0.0):.3f} mL")
        print(f"Right Hippocampus:  {cached_volumes.get('right_hippocampus_volume_ml', 0.0):.3f} mL")
    else:
        print("No volumetric statistics found.")
    print("="*70)
    print("Verification complete.")

if __name__ == "__main__":
    import sys
    patient = "IXI002-Guys-0828-MPRAGESEN_-s256_-0301-00003-000001-01"
    if len(sys.argv) > 1:
        patient = sys.argv[1]
    verify_patient(patient)
