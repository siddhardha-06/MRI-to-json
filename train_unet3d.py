"""Thin compatibility wrapper: the 3D U-Net code moved to training/train_unet3d.py (v2 Phase 1).

`from train_unet3d import UNet3D, ALL_LABELS, ...` keeps working, and
`python train_unet3d.py --eval-only` runs the new evaluation.
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from training.train_unet3d import *  # noqa: F401,F403,E402
from training.train_unet3d import pkload, dice_per_class, evaluate  # noqa: F401,E402  (underscore-free re-exports)

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "training" / "train_unet3d.py"), run_name="__main__")
