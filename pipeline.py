"""
Complete MRI → Model → Features → JSON Pipeline

Three-stage training:
1. Autoencoder (self-supervised)
2. U-Net (with SynthSeg pseudo-labels)
3. Feature Regressor (with rule-based ground truth)
"""

import os
import argparse
import subprocess
import json
from pathlib import Path
import torch


def run_command(cmd, description):
    """Run a shell command with description"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"❌ FAILED: {description}")
        return False
    
    print(f"✓ COMPLETED: {description}")
    return True


def step_1_generate_synthseg_masks(mri_dir="ixi-t1", output_dir="ixi-t1-masks"):
    """
    Step 1: Generate pseudo-labels using SynthSeg
    
    Prerequisites:
    pip install synthseg
    """
    print("\n" + "="*70)
    print("STEP 1: Generate SynthSeg Pseudo-Labels")
    print("="*70)
    
    print("\n⚠️  IMPORTANT: SynthSeg must be installed:")
    print("  pip install synthseg")
    print("\nThis step runs SynthSeg on all IXI MRI files to generate GM/WM/CSF masks.")
    print("This is required for training the U-Net segmentation model.\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .nii files
    mri_files = sorted(Path(mri_dir).glob("*/*.nii"))
    
    if not mri_files:
        print(f"❌ ERROR: No .nii files found in {mri_dir}")
        return False
    
    print(f"Found {len(mri_files)} MRI files. Running SynthSeg...\n")
    
    # Run SynthSeg (this could take a while)
    # Uses GPU if available, falls back to CPU
    cmd = f"synthseg --i {mri_dir} --o {output_dir}"
    
    success = run_command(cmd, "Running SynthSeg on IXI dataset")
    
    if success:
        # Verify masks were created
        masks = list(Path(output_dir).glob("*_seg.nii"))
        print(f"\n✓ Generated {len(masks)} segmentation masks")
    
    return success


def step_2_train_autoencoder(
    mri_dir="ixi-t1",
    model_type="2d",
    epochs=50,
    batch_size=8,
    latent_dim=128,
    save_dir="checkpoints",
):
    """
    Step 2: Train autoencoder
    
    Training: MRI → Reconstruction
    No labels needed (self-supervised)
    Learns general MRI features
    """
    print("\n" + "="*70)
    print("STEP 2: Train Autoencoder")
    print("="*70)
    
    os.makedirs(save_dir, exist_ok=True)
    
    cmd = (
        f"python train_autoencoder.py "
        f"--data-dir {mri_dir} "
        f"--model-type {model_type} "
        f"--epochs {epochs} "
        f"--batch-size {batch_size} "
        f"--latent-dim {latent_dim} "
        f"--save-dir {save_dir}"
    )
    
    return run_command(cmd, f"Training {model_type.upper()} Autoencoder")


def step_3_train_unet(
    mri_dir="ixi-t1",
    mask_dir="ixi-t1-masks",
    epochs=50,
    batch_size=16,
    save_dir="checkpoints",
):
    """
    Step 3: Train U-Net segmentation
    
    Training: MRI → GM/WM/CSF segmentation
    Uses SynthSeg pseudo-labels as ground truth
    Requires SynthSeg masks (from Step 1)
    """
    print("\n" + "="*70)
    print("STEP 3: Train U-Net Segmentation")
    print("="*70)
    
    # Check if masks exist
    masks = list(Path(mask_dir).glob("*_seg.nii"))
    if not masks:
        print(f"❌ ERROR: No segmentation masks found in {mask_dir}")
        print("You must run Step 1 (SynthSeg) first!")
        return False
    
    os.makedirs(save_dir, exist_ok=True)
    
    cmd = (
        f"python train_unet.py "
        f"--mri-dir {mri_dir} "
        f"--mask-dir {mask_dir} "
        f"--epochs {epochs} "
        f"--batch-size {batch_size} "
        f"--save-dir {save_dir}"
    )
    
    return run_command(cmd, "Training U-Net Segmentation")


def step_4_train_regressor(
    mri_dir="ixi-t1",
    features_file="rule_based_features.json",
    num_features=64,
    epochs=100,
    batch_size=32,
    save_dir="checkpoints",
):
    """
    Step 4: Train feature regressor
    
    Training: MRI → Predicted features
    Uses rule-based pipeline output as ground truth
    Requires rule_based_features.json
    """
    print("\n" + "="*70)
    print("STEP 4: Train Feature Regressor")
    print("="*70)
    
    # Check if features file exists
    if not Path(features_file).exists():
        print(f"❌ ERROR: Features file not found: {features_file}")
        print("You must generate rule-based features first!")
        print("Use your rule-based pipeline to compute features and save as JSON.\n")
        print("Expected format:")
        print("  {")
        print('    "IXI002-Guys-0828-T1": [feat1, feat2, ...],')
        print('    "IXI012-HH-1211-T1": [feat1, feat2, ...],')
        print("    ...")
        print("  }\n")
        return False
    
    os.makedirs(save_dir, exist_ok=True)
    
    cmd = (
        f"python train_regressor.py "
        f"--mri-dir {mri_dir} "
        f"--features-file {features_file} "
        f"--num-features {num_features} "
        f"--epochs {epochs} "
        f"--batch-size {batch_size} "
        f"--save-dir {save_dir}"
    )
    
    return run_command(cmd, "Training Feature Regressor")


def step_5_extract_final_features(
    mri_dir="ixi-t1",
    checkpoint_dir="checkpoints",
    output_dir="features",
):
    """
    Step 5: Extract final features using trained regressor
    
    Output: JSON with predicted features for all MRI files
    """
    print("\n" + "="*70)
    print("STEP 5: Extract Final Features")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find best checkpoint
    checkpoint_path = Path(checkpoint_dir) / "regressor_best.pth"
    
    if not checkpoint_path.exists():
        print(f"❌ ERROR: Checkpoint not found: {checkpoint_path}")
        print("You must train the regressor first (Step 4)!")
        return False
    
    cmd = (
        f"python train_regressor.py "
        f"--mri-dir {mri_dir} "
        f"--checkpoint {checkpoint_path} "
        f"--save-dir {output_dir} "
        f"--predict"
    )
    
    return run_command(cmd, "Extracting Features with Trained Regressor")


def full_pipeline(args):
    """Run complete pipeline"""
    
    print("\n" + "="*70)
    print("MRI → Model → Features → JSON Pipeline")
    print("="*70)
    print(f"MRI Directory: {args.mri_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Save Directory: {args.save_dir}\n")
    
    all_steps = []
    
    # Step 1: SynthSeg (if needed)
    if args.step >= 1 or args.step == 0:
        if args.skip_synthseg:
            print("\n⊘ Skipping SynthSeg (already have masks)")
        else:
            success = step_1_generate_synthseg_masks(args.mri_dir, args.mask_dir)
            all_steps.append(("SynthSeg Pseudo-labels", success))
            if not success and args.step == 1:
                return all_steps
    
    # Step 2: Train Autoencoder
    if args.step >= 2 or args.step == 0:
        success = step_2_train_autoencoder(
            args.mri_dir,
            model_type=args.autoencoder_type,
            epochs=args.ae_epochs,
            batch_size=args.ae_batch_size,
            latent_dim=args.ae_latent_dim,
            save_dir=args.save_dir,
        )
        all_steps.append(("Autoencoder", success))
        if not success and args.step == 2:
            return all_steps
    
    # Step 3: Train U-Net (optional)
    if (args.step >= 3 or args.step == 0) and not args.skip_unet:
        success = step_3_train_unet(
            args.mri_dir,
            args.mask_dir,
            epochs=args.unet_epochs,
            batch_size=args.unet_batch_size,
            save_dir=args.save_dir,
        )
        all_steps.append(("U-Net Segmentation", success))
        if not success and args.step == 3:
            return all_steps
    
    # Step 4: Train Regressor
    if args.step >= 4 or args.step == 0:
        success = step_4_train_regressor(
            args.mri_dir,
            features_file=args.features_file,
            num_features=args.num_features,
            epochs=args.regressor_epochs,
            batch_size=args.regressor_batch_size,
            save_dir=args.save_dir,
        )
        all_steps.append(("Feature Regressor", success))
        if not success and args.step == 4:
            return all_steps
    
    # Step 5: Extract features
    if args.step >= 5 or args.step == 0:
        success = step_5_extract_final_features(args.mri_dir, args.save_dir, args.output_dir)
        all_steps.append(("Feature Extraction", success))
    
    return all_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complete MRI → Model → Features → JSON Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run entire pipeline
  python pipeline.py
  
  # Run specific step (2 = autoencoder)
  python pipeline.py --step 2
  
  # Skip U-Net, only train autoencoder + regressor
  python pipeline.py --skip-unet
  
  # Skip SynthSeg if masks already exist
  python pipeline.py --skip-synthseg
  
  # Custom configuration
  python pipeline.py --ae-epochs 100 --regressor-epochs 200 --num-features 128
        """
    )
    
    # Data
    parser.add_argument("--mri-dir", default="ixi-t1", help="MRI dataset directory")
    parser.add_argument("--mask-dir", default="ixi-t1-masks", help="Segmentation masks (from SynthSeg)")
    parser.add_argument("--features-file", default="rule_based_features.json", help="Rule-based features JSON")
    parser.add_argument("--save-dir", default="checkpoints", help="Save trained models")
    parser.add_argument("--output-dir", default="features", help="Output features directory")
    
    # Steps
    parser.add_argument("--step", type=int, default=0, help="Run specific step (0=all, 1-5)")
    parser.add_argument("--skip-synthseg", action="store_true", help="Skip SynthSeg step")
    parser.add_argument("--skip-unet", action="store_true", help="Skip U-Net training")
    
    # Autoencoder
    parser.add_argument("--autoencoder-type", choices=["2d", "3d"], default="2d", help="2D or 3D autoencoder")
    parser.add_argument("--ae-epochs", type=int, default=50, help="Autoencoder epochs")
    parser.add_argument("--ae-batch-size", type=int, default=8, help="Autoencoder batch size")
    parser.add_argument("--ae-latent-dim", type=int, default=128, help="Latent dimension")
    
    # U-Net
    parser.add_argument("--unet-epochs", type=int, default=50, help="U-Net epochs")
    parser.add_argument("--unet-batch-size", type=int, default=16, help="U-Net batch size")
    
    # Regressor
    parser.add_argument("--regressor-epochs", type=int, default=100, help="Regressor epochs")
    parser.add_argument("--regressor-batch-size", type=int, default=32, help="Regressor batch size")
    parser.add_argument("--num-features", type=int, default=64, help="Number of features to predict")
    
    args = parser.parse_args()
    
    # Run pipeline
    results = full_pipeline(args)
    
    # Summary
    print("\n" + "="*70)
    print("Pipeline Summary")
    print("="*70)
    
    for step_name, success in results:
        status = "✓" if success else "❌"
        print(f"{status} {step_name}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n✓ Pipeline completed successfully!")
        print(f"✓ Features saved to: {args.output_dir}/predicted_features.json")
    else:
        print("\n❌ Pipeline had failures. Check logs above.")
