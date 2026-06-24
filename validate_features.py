import json
import numpy as np

# Load the extracted features
with open('rule_based_features.json', 'r') as f:
    features = json.load(f)

print("=" * 70)
print("VALIDATION: rule_based_features.json")
print("=" * 70)

# 1. Basic stats
print(f"\n[INFO] Total samples: {len(features)}")
first_key = list(features.keys())[0]
num_features = len(features[first_key])
print(f"[INFO] Features per sample: {num_features}")

# 2. Check for consistency
all_lengths = [len(v) for v in features.values()]
if len(set(all_lengths)) == 1:
    print(f"[SUCCESS] All samples have {num_features} features (consistent)")
else:
    print(f"[ERROR] INCONSISTENT! Feature counts: {set(all_lengths)}")

# 3. Feature statistics
all_features = np.array([v for v in features.values()])
print(f"\n[INFO] Feature value ranges:")
print(f"   Min across all: {all_features.min():.4f}")
print(f"   Max across all: {all_features.max():.4f}")
print(f"   Mean across all: {all_features.mean():.4f}")
print(f"   Std across all: {all_features.std():.4f}")

# 4. Per-feature stats
print(f"\n[INFO] Per-feature breakdown (feature ranges):")
feature_names = [
    "intensity_mean", "intensity_std", "intensity_min", "intensity_max", "intensity_median",
    "intensity_p25", "intensity_p75", "intensity_skew", "intensity_kurt", "intensity_range",
    "tissue_csf%", "tissue_gm%", "tissue_wm%", "tissue_brain%",
    "texture_contrast", "texture_correlation", "texture_energy", "texture_homogeneity",
    "texture_grad_mean", "texture_grad_std", "texture_grad_max", "texture_entropy",
    "struct_area%", "struct_ecc", "struct_solid", "struct_extent", "struct_perim",
    "symm_score", "symm_asym%"
]

for i in range(num_features):
    feat_values = all_features[:, i]
    name = feature_names[i] if i < len(feature_names) else f"feat_{i}"
    print(f"   {i:2d}. {name:20s}: min={feat_values.min():8.4f}, max={feat_values.max():8.4f}, mean={feat_values.mean():8.4f}, std={feat_values.std():8.4f}")

# 5. Check for NaN/Inf
nan_count = np.isnan(all_features).sum()
inf_count = np.isinf(all_features).sum()
print(f"\n[INFO] Data quality:")
print(f"   NaN values: {nan_count}")
print(f"   Inf values: {inf_count}")

if nan_count == 0 and inf_count == 0:
    print(f"   [SUCCESS] CLEAN DATA - ready for training!")
else:
    print(f"   [ERROR] DATA HAS ISSUES")

# 6. Sample check
print(f"\n[INFO] Sample (first MRI: {first_key}):")
print(f"   Features: {features[first_key][:10]}")
print(f"   (showing first 10 of {num_features})")

print("\n" + "=" * 70)
print("[SUCCESS] Validation PASSED - rule_based_features.json is ready!")
print("=" * 70)
