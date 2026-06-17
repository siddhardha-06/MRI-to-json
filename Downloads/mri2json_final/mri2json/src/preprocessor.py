"""
preprocessor.py
---------------
Normalize MRI slice intensities and convert to 8-bit PNG for API submission.

MRI images have arbitrary intensity ranges (scanner-dependent).
We normalize to [0, 255] using robust percentile clipping so that
bright artifacts don't wash out tissue contrast.
"""

import io
import base64
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
from PIL import Image


def normalize_slice(
    slice_2d: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """
    Robust min-max normalization using percentile clipping.

    Clips to [low_pct, high_pct] range before scaling to [0, 255].
    This handles outlier voxels (e.g. skull, background noise)
    that would otherwise collapse tissue contrast.

    Returns: uint8 array in [0, 255]
    """
    # Work only on non-zero region for percentile calculation
    nonzero = slice_2d[slice_2d > 0]
    if len(nonzero) == 0:
        return np.zeros_like(slice_2d, dtype=np.uint8)

    low = np.percentile(nonzero, low_percentile)
    high = np.percentile(nonzero, high_percentile)

    if high == low:
        return np.zeros_like(slice_2d, dtype=np.uint8)

    clipped = np.clip(slice_2d, low, high)
    normalized = (clipped - low) / (high - low) * 255.0
    return normalized.astype(np.uint8)


def slice_to_pil(
    slice_2d: np.ndarray,
    rotate_degrees: int = 90,
    target_size: Tuple[int, int] = (512, 512),
) -> Image.Image:
    """
    Convert a normalized 2D numpy array to a PIL Image.

    Args:
        slice_2d: normalized uint8 array
        rotate_degrees: MRI slices often need rotation to display correctly.
                        90° CCW is standard for axial OASIS-1 slices.
        target_size: resize to this (width, height) for API consistency.
    
    Returns:
        PIL Image in 'RGB' mode (API expects color image)
    """
    img = Image.fromarray(slice_2d, mode="L")  # grayscale
    
    if rotate_degrees:
        img = img.rotate(rotate_degrees, expand=True)
    
    img = img.resize(target_size, Image.LANCZOS)
    img = img.convert("RGB")  # Claude Vision expects RGB
    
    return img


def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """
    Encode a PIL Image to a base64 string for Anthropic API submission.
    """
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return base64.standard_b64encode(buffer.read()).decode("utf-8")


def save_slice_png(
    img: Image.Image,
    output_path: Union[str, Path],
) -> None:
    """Save a PIL Image as PNG to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), format="PNG")


def preprocess_slice(
    slice_2d: np.ndarray,
    slice_index: int,
    subject_id: str,
    output_dir: Optional[Union[str, Path]] = None,
    save_png: bool = False,
) -> Tuple[str, Image.Image]:
    """
    Full preprocessing pipeline for a single MRI slice:
    normalize → PIL → base64.

    Args:
        slice_2d: raw float32 2D array from loader
        slice_index: which slice this is (for filename)
        subject_id: OASIS subject ID (for filename)
        output_dir: where to save PNG if save_png=True
        save_png: whether to persist the PNG to disk

    Returns:
        (base64_string, pil_image)
    """
    normalized = normalize_slice(slice_2d)
    img = slice_to_pil(normalized)

    if save_png and output_dir is not None:
        out_path = Path(output_dir) / f"{subject_id}_slice{slice_index:03d}.png"
        save_slice_png(img, out_path)

    b64 = pil_to_base64(img)
    return b64, img
