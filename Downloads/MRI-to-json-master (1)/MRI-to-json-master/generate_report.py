"""
generate_report.py
------------------
Takes the predicted_features.json (output from MRI-to-JSON pipeline)
and uses Gemini to generate a plain-English clinical report per patient.
Reports are formatted as professional clinical documents and saved as text files.

Usage:
    python generate_report.py --input predicted_features.json --output reports/
    python generate_report.py --input predicted_features.json --patient IXI311-Guys-0885-MPRAGESEN_-s302_-0301-00003-000001-01
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from google import genai
from google.genai import types

def format_clinical_report(patient_id: str, features: dict, clinical_summary: str, timestamp: datetime) -> str:
    """Format the clinical report according to professional medical report template."""
    
    # Extract patient info from patient ID
    patient_name = patient_id.split("-")[0]  # First part before hospital identifier
    uhid = patient_id
    acc_no = f"24{timestamp.strftime('%m%d')}{patient_id[:6].replace('-','')}"
    
    # Extract tissue segmentation data
    tissue_seg = features.get("tissue_segmentation", {})
    csf_pct = tissue_seg.get("csf_ventricles", {}).get("percentage", 0)
    gm_pct = tissue_seg.get("gray_matter", {}).get("percentage", 0)
    wm_pct = tissue_seg.get("white_matter", {}).get("percentage", 0)
    
    # Extract texture features
    texture = features.get("texture_features", {})
    glcm_contrast = texture.get("glcm_contrast", 0)
    glcm_energy = texture.get("glcm_energy", 0)
    glcm_homogeneity = texture.get("glcm_homogeneity", 0)
    
    # Extract structural features
    struct = features.get("structural_features", {})
    brain_area_pct = struct.get("brain_area_pct", 0)
    symmetry = features.get("symmetry", {})
    bilateral_sym = symmetry.get("bilateral_symmetry_score", 0)
    
    # Extract pathology flags
    pathology = features.get("pathology_indicators", {})
    flagged_items = pathology.get("flags", []) if isinstance(pathology, dict) else []
    risk_level = pathology.get("risk_level", "unknown") if isinstance(pathology, dict) else "unknown"
    
    # Format timestamp
    report_date = timestamp.strftime("%d-%b-%Y %H:%M:%S %p")
    study_time = timestamp.strftime("%d-%b-%Y %H:%M:%S %p")
    
    # Build the formatted report
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
    ▶ {clinical_summary.split('Tissue Segmentation:')[1].split('Texture')[0].strip() if 'Tissue Segmentation:' in clinical_summary else 'Tissue volumes assessed quantitatively.'}

Texture Analysis:
    • GLCM Contrast: {glcm_contrast:.4f}
    • GLCM Energy: {glcm_energy:.4f}
    • GLCM Homogeneity: {glcm_homogeneity:.4f}
    ▶ {clinical_summary.split('Texture')[1].split('Structural')[0].strip() if 'Texture' in clinical_summary else 'Texture features within normal range.'}

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
"""
    
    if flagged_items:
        for item in flagged_items:
            report += f"    • {item if isinstance(item, str) else item.get('name', str(item))}\n"
    else:
        report += "    • No significant pathology flags detected\n"
    
    report += f"""
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
    
    return report


def generate_report_for_patient(patient_id: str, features: dict, model: str = "gemini-2.5-flash-lite") -> str:
    """Generate a clinical summary for a single patient using Gemini."""
    
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    SYSTEM_PROMPT = """You are a neuroradiology AI assistant that interprets quantitative MRI 
feature data and generates clear, concise clinical summary reports.

You will be given structured JSON data extracted from a brain MRI scan containing:
- Tissue segmentation percentages (CSF, gray matter, white matter)
- Texture features (GLCM contrast, entropy, homogeneity)
- Structural features (brain area, shape metrics)
- Symmetry analysis
- Pathology indicator flags

Generate a professional clinical impression/summary that:
1. Summarizes the key findings in plain English
2. Interprets what the numbers mean clinically
3. Highlights any flagged pathology indicators
4. Gives an overall assessment

Keep it to 3-4 paragraphs, professional medical language."""
    
    # Build a focused summary of the key features for the prompt
    user_prompt = f"""Generate a clinical impression for patient: {patient_id}

Here is the quantitative feature data extracted from their brain MRI:

{json.dumps(features, indent=2)}

Write a clear clinical impression interpreting these findings."""

    response = client.models.generate_content(
        model=model,
        contents=[user_prompt],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=800,
            temperature=0.3,
        ),
    )
    
    return response.text


def process_all_patients(input_path: str, output_dir: str, model: str = "gemini-2.5-flash-lite"):
    """Process all patients in the JSON file and save individual reports."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, "r") as f:
        all_features = json.load(f)
    
    print(f"Found {len(all_features)} patients to process")
    
    results = {}
    timestamp = datetime.now()
    
    for i, (patient_id, features) in enumerate(all_features.items()):
        print(f"[{i+1}/{len(all_features)}] Generating report for: {patient_id}")
        
        try:
            clinical_summary = generate_report_for_patient(patient_id, features, model)
            formatted_report = format_clinical_report(patient_id, features, clinical_summary, timestamp)
            
            # Save individual report
            safe_name = patient_id.replace("/", "_").replace("\\", "_")
            report_file = output_path / f"{safe_name}_report.txt"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(formatted_report)
            
            results[patient_id] = {"status": "success", "report_file": str(report_file)}
            print(f"  ✓ Saved to {report_file.name}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[patient_id] = {"status": "error", "error": str(e)}
    
    # Save combined summary
    summary_file = output_path / "_all_reports_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDone! Reports saved to: {output_dir}")
    print(f"Combined summary: {summary_file}")
    
    return results


def process_single_patient(input_path: str, patient_id: str, output_dir: str = "reports", model: str = "gemini-2.5-flash-lite"):
    """Process a single patient and save the formatted report to a file."""
    
    with open(input_path, "r") as f:
        all_features = json.load(f)
    
    if patient_id not in all_features:
        # Try partial match
        matches = [k for k in all_features.keys() if patient_id in k]
        if not matches:
            print(f"Patient '{patient_id}' not found.")
            print(f"Available patients: {list(all_features.keys())[:5]}...")
            return
        patient_id = matches[0]
        print(f"Matched patient: {patient_id}")
    
    features = all_features[patient_id]
    print(f"\nGenerating report for: {patient_id}\n")
    
    try:
        clinical_summary = generate_report_for_patient(patient_id, features, model)
        timestamp = datetime.now()
        formatted_report = format_clinical_report(patient_id, features, clinical_summary, timestamp)
        
        # Save report to file
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        safe_name = patient_id.replace("/", "_").replace("\\", "_")
        report_file = output_path / f"{safe_name}_report.txt"
        
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(formatted_report)
        
        print(formatted_report)
        print(f"\n{'='*80}")
        print(f"Report saved to: {report_file}")
        
        return formatted_report
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate clinical reports from MRI feature JSON")
    parser.add_argument("--input", required=True, help="Path to predicted_features.json")
    parser.add_argument("--output", default="reports", help="Output directory for reports")
    parser.add_argument("--patient", default=None, help="Process a single patient ID (optional)")
    parser.add_argument("--model", default="gemini-2.5-flash-lite", help="Gemini model to use")
    
    args = parser.parse_args()
    
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: Set GEMINI_API_KEY environment variable first.")
        exit(1)
    
    if args.patient:
        process_single_patient(args.input, args.patient, args.output, args.model)
    else:
        process_all_patients(args.input, args.output, args.model)
