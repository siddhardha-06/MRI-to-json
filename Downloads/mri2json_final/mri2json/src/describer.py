"""
describer.py
------------
Sends a preprocessed MRI slice (as base64 PNG) to the Google Gemini API
and returns a structured JSON description.
"""

import os
import json
import re
from PIL import Image
import io
import base64
from google import genai
from google.genai import types


# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a board-certified neuroradiologist with 20 years of experience 
reading brain MRI scans. You are precise, systematic, and conservative — you never 
speculate beyond what the image clearly shows.

Your task is to examine the provided brain MRI slice and return a structured JSON 
description. You must return ONLY valid JSON — no prose, no markdown, no code fences.
Do not include any text before or after the JSON object."""

USER_PROMPT_TEMPLATE = """Examine this brain MRI slice carefully and fill in the following 
JSON structure with your observations. Be specific about what you can see. 
For pathology flags, be conservative — only flag true if clearly visible.

Return ONLY this JSON, completed with your observations:

{{
  "modality": "<T1-weighted | T2-weighted | FLAIR | unknown>",
  "plane": "<axial | coronal | sagittal>",
  "image_quality": "<good | fair | poor>",
  "anatomy": {{
    "visible_structures": ["<list structures you can clearly identify>"],
    "hemisphere_symmetry": "<symmetric | mild asymmetry | moderate asymmetry | marked asymmetry> — brief note",
    "cortical_ribbon": "<preserved | thinned | not visible>",
    "sulcal_pattern": "<normal depth | mildly widened | moderately widened | significantly widened>"
  }},
  "tissue_findings": {{
    "gray_matter": "<brief description of cortical gray matter>",
    "white_matter": "<brief description — any signal changes, lesions, or normal>",
    "ventricles": "<size and appearance of visible ventricles>",
    "sulci_gyri": "<description of sulcal/gyral pattern>",
    "basal_ganglia": "<appearance if visible, else 'not visible in this slice'>",
    "thalamus": "<appearance if visible, else 'not visible in this slice'>",
    "cerebellum": "<appearance if visible, else 'not visible in this slice'>",
    "brainstem": "<appearance if visible, else 'not visible in this slice'>"
  }},
  "pathology_flags": {{
    "atrophy": <true | false>,
    "atrophy_severity": "<none | mild | moderate | severe>",
    "white_matter_lesions": <true | false>,
    "lesion_count_estimate": "<none | 1-3 | 4-10 | >10>",
    "mass_effect": <true | false>,
    "midline_shift": <true | false>,
    "hydrocephalus": <true | false>,
    "infarct": <true | false>,
    "hemorrhage": <true | false>,
    "notes": "<any other notable findings, or 'no significant findings'>"
  }},
  "global_impression": "<1-2 sentence overall impression as a radiologist would write>",
  "confidence": <0.0 to 1.0 — your confidence in this assessment given image quality>
}}

Subject context: {context}"""


# ─── Main function ─────────────────────────────────────────────────────────────

def describe_slice(
    base64_image: str,
    subject_id: str,
    slice_index: int,
    subject_context=None,
    model: str = "gemini-2.5-flash-lite",
    max_tokens: int = 1500,
) -> dict:

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    img_bytes = base64.b64decode(base64_image)
    pil_image = Image.open(io.BytesIO(img_bytes))

    context_str = subject_context or "Age and demographics not provided."
    user_prompt = USER_PROMPT_TEMPLATE.format(context=context_str)

    response = client.models.generate_content(
        model=model,
        contents=[user_prompt, pil_image],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=max_tokens,
            temperature=0.1,
        ),
    )

    return _parse_json_response(response.text, subject_id, slice_index)


def _parse_json_response(raw_text: str, subject_id: str, slice_index: int) -> dict:
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text)
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {
        "error": "Failed to parse JSON response",
        "raw_response": raw_text[:500],
        "subject_id": subject_id,
        "slice_index": slice_index,
        "modality": "unknown",
        "plane": "unknown",
        "confidence": 0.0,
    }