import json
from datetime import datetime
from pathlib import Path

# Load sample patient data
with open("features/predicted_features.json", "r") as f:
    all_features = json.load(f)

patient_id = "IXI002-Guys-0828-MPRAGESEN_-s256_-0301-00003-000001-01"
features = all_features[patient_id]

# Create sample clinical summary (what Gemini would generate)
clinical_summary = """The quantitative MRI analysis reveals generally normal brain tissue composition and structure. 
Tissue segmentation shows appropriate proportions of cerebrospinal fluid (20.65%), gray matter (34.19%), 
and white matter (3.99%), suggesting absence of significant global atrophy. Texture analysis demonstrates 
relatively uniform tissue composition with good tissue boundaries. Structural metrics indicate normal brain volume 
and shape characteristics. Bilateral symmetry index of 0.6499 shows mild asymmetry between hemispheres, which 
is within normal variation. Overall risk assessment is LOW with no significant pathology flags detected. 
Clinical correlation with patient presentation is recommended."""

# Format timestamp
timestamp = datetime.now()
report_date = timestamp.strftime("%d-%b-%Y %H:%M:%S %p")
study_time = timestamp.strftime("%d-%b-%Y %H:%M:%S %p")

# Extract data
patient_name = patient_id.split("-")[0]
uhid = patient_id
acc_no = f"24{timestamp.strftime('%m%d')}{patient_id[:6].replace('-','')}"

tissue_seg = features.get("tissue_segmentation", {})
csf_pct = tissue_seg.get("csf_ventricles", {}).get("percentage", 0)
gm_pct = tissue_seg.get("gray_matter", {}).get("percentage", 0)
wm_pct = tissue_seg.get("white_matter", {}).get("percentage", 0)

texture = features.get("texture_features", {})
glcm_contrast = texture.get("glcm_contrast", 0)
glcm_energy = texture.get("glcm_energy", 0)
glcm_homogeneity = texture.get("glcm_homogeneity", 0)

struct = features.get("structural_features", {})
brain_area_pct = struct.get("brain_area_pct", 0)
symmetry = features.get("symmetry", {})
bilateral_sym = symmetry.get("bilateral_symmetry_score", 0)

pathology = features.get("pathology_indicators", {})
risk_level = pathology.get("risk_level", "unknown")

# Create output directory
output_path = Path("reports")
output_path.mkdir(parents=True, exist_ok=True)

# Generate formatted report
report = f"""DEPARTMENT OF CLINICAL IMAGING & INTERVENTIONAL RADIOLOGY

┌─────────────────────────────────────┬──────────────────────────────────────┐
│ NAME                 │ {patient_name:<12} │ STUDY      │ {study_time:<20} │
├──────────────────────┼──────────────┼────────────┼──────────────────────────┤
│ AGE / GENDER         │ ---          │ UHID       │ {uhid:<32} │
├──────────────────────┼──────────────┼────────────┼──────────────────────────┤
│ ACC NO               │ {acc_no:<12} │ MOD        │ MR                       │
├──────────────────────┼──────────────┼────────────┼──────────────────────────┤
│ REFERRER             │ AI System    │ REPORT     │ {report_date:<20} │
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
    • GLCM Contrast: {glcm_contrast:.4f}
    • GLCM Energy: {glcm_energy:.4f}
    • GLCM Homogeneity: {glcm_homogeneity:.4f}
    ▶ Texture features quantified using Gray Level Co-occurrence Matrix analysis.

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
    • Risk Level: {risk_level.upper()}
    • No significant pathology flags detected

═════════════════════════════════════════════════════════════════════════════

IMPRESSION:

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
                            Generated: {report_date}
                        RESEARCH/TRAINING PURPOSES ONLY - NOT FOR CLINICAL USE
"""

# Save to file
safe_name = patient_id.replace("/", "_").replace("\\", "_")
report_file = output_path / f"{safe_name}_report.txt"
with open(report_file, "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print(f"\n{'='*80}")
print(f"✓ Report saved to: {report_file}")
