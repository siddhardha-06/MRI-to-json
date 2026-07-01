# MRI Quantitative Analysis & Clinical Report Pipeline

Complete PyTorch-based pipeline for training neural networks on T1-weighted brain MRI scans, extracting anatomical/texture features, and generating LLM-powered clinical reports.

---

## 🛠️ Virtual Environment & GPU Setup

Since medical image processing and deep learning are compute-intensive, using your GPU (CUDA) is highly recommended. 

### 1. Verify GPU Availability
To confirm PyTorch can detect your NVIDIA graphics card, run:
```powershell
.\.venv\Scripts\python check_cuda.py
```
*(If it returns `CUDA available: False`, follow the step below to install the correct CUDA-enabled PyTorch binaries).*

### 2. Install GPU-Enabled PyTorch (CUDA 12.1)
If you do not have the GPU version of PyTorch, install it inside your virtual environment using:
```powershell
.\.venv\Scripts\pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Activate Virtual Environment (Alternative)
If you prefer not to prefix every command with `.\.venv\Scripts\python`, activate the environment in your PowerShell terminal first:
```powershell
.\.venv\Scripts\Activate.ps1
```
Once activated, the command prompt will be prefixed with `(.venv)`, and you can use standard `python` commands.

---

## 🧬 Pipeline Architecture & Workflow

The pipeline consists of a self-supervised representation model, a supervised segmentation model, a quantitative feature extractor, and a regressor to link raw MRI images directly to clinical summaries.

```
                   MRI Files (.nii)
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
[Step 1] ANTsPy Atropos                 [Step 2] Train Autoencoder
  (Bias field correction &                (Self-supervised learning of
   tissue segmentation)                   general brain features)
       │                                       │
       ▼                                       ▼
[Step 3] Train U-Net                     [Step 4] Extract Rule-Based Features
  (Learns to segment GM/WM/CSF             (Extracts intensity, tissue, texture,
   from raw MRI slices)                     structure, and symmetry features)
                                               │
                                               ▼
                                        [Step 5] Train Feature Regressor
                                          (Learns raw slice -> features)
                                               │
                                               ▼
                                        [Step 6] Extract Final Features
                                          (Produces predicted_features.json)
                                               │
                                               ▼
                                        [Step 7] Clinical Reports (Gemini)
                                          (Generates reports in reports/)
```

---

## 🚀 Running the Pipeline

### Option A: Complete Automated Pipeline (Recommended)
You can run the entire workflow end-to-end with a single orchestrator command:
```powershell
.\.venv\Scripts\python pipeline.py --ae-epochs 50 --unet-epochs 50 --regressor-epochs 100
```
This script handles the execution flow, performs validations, and creates all outputs automatically.

### Option B: Step-by-Step Execution (Manual Control)

#### Step 1: Generate ANTsPy Tissue Segmentation Masks
Generate high-quality tissue masks (Background, CSF, Gray Matter, White Matter) using ANTsPy Atropos (with N4 bias field correction):
```powershell
.\.venv\Scripts\python train_unet.py --generate-masks --mri-dir ixi-t1 --mask-dir ixi-t1-masks
```
*   *For a fast test run (e.g., first 3 files):* append `--limit 3` to the command.
*   **Output**: `ixi-t1-masks/` containing `*_seg.nii` files.

#### Step 2: Train Autoencoder (Self-Supervised)
Train the representation learning network directly on raw MRI volumes:
```powershell
.\.venv\Scripts\python train_autoencoder.py --data-dir ixi-t1 --model-type 2d --epochs 50 --batch-size 8
```
*   **Output**: `checkpoints/autoencoder_2d_best.pth`

#### Step 3: Train U-Net Segmentation Network
Train the U-Net model using the generated ANTsPy pseudo-labels as ground truth:
```powershell
.\.venv\Scripts\python train_unet.py --mri-dir ixi-t1 --mask-dir ixi-t1-masks --epochs 50 --batch-size 16
```
*   **Output**: `checkpoints/unet_best.pth`

#### Step 4: Extract Rule-Based Features
Extract 29 anatomical and texture metrics from the middle slices of all raw MRIs to create training targets for the regressor:
```powershell
.\.venv\Scripts\python extract_features.py --mri-dir ixi-t1 --output rule_based_features.json
```
*   **Output**: `rule_based_features.json`

#### Step 5: Train Feature Regressor
Train the regression network to predict the 29 clinical features directly from raw MRI slices:
```powershell
.\.venv\Scripts\python train_regressor.py --mri-dir ixi-t1 --features-file rule_based_features.json --num-features 29 --epochs 100 --batch-size 32
```
*   **Output**: `checkpoints/regressor_best.pth`

#### Step 6: Extract Final Predictions & Generate Reports
Use the trained regressor to extract features for all MRI scans and automatically compile clinical reports:
```powershell
# Set your API Key (Powershell)
$env:GEMINI_API_KEY="REDACTED_API_KEY"

.\.venv\Scripts\python train_regressor.py --mri-dir ixi-t1 --checkpoint checkpoints/regressor_best.pth --predict
```
*   **Outputs**:
    *   `features/predicted_features.json` (Numerical predictions)
    *   `reports/` (Individual plain-English diagnostic reports)

---

## 🖥️ NeuroSphere Web Dashboard

The application includes an interactive, multi-threaded dashboard for clinical visualization and pipeline monitoring.

### Launch the Server
```powershell
.\.venv\Scripts\python app_server.py
```
This will automatically launch the dashboard in your default browser at **`http://localhost:8000`**.

### Dashboard Capabilities:
*   **Interactive 3D Slice Viewer**: Click through axial, coronal, and sagittal planes of any patient's T1 MRI.
*   **Overlay segmentations**: Toggle U-Net tissue segmentations (CSF, GM, WM) directly on top of the raw scan.
*   **Dynamic Feature Extraction**: Drag and drop any new `.nii` file to segment it, calculate its 29 structural features, and view them immediately.
*   **Clinical Summary**: Read the Gemini-generated medical report side-by-side with the brain model.

---

## 📊 Feature Schema (29 Features)

The quantitative extraction script (`extract_features.py`) calculates 29 features grouped into five categories:

1.  **Intensity Statistics (10)**: Mean, standard deviation, minimum, maximum, median, 25th percentile, 75th percentile, skewness, kurtosis, and dynamic range.
2.  **Tissue Volumes (4)**: CSF %, Gray Matter %, White Matter %, and Total Brain Tissue %.
3.  **Texture Metrics (8)**: GLCM contrast, GLCM correlation, GLCM energy, GLCM homogeneity, mean gradient, standard deviation of gradient, maximum gradient, and Shannon entropy.
4.  **Structural Features (5)**: Brain area %, eccentricity, solidity, extent, and perimeter.
5.  **Symmetry Metrics (2)**: Bilateral symmetry score and asymmetry index %.

---

## 📁 Workspace File Outputs

Once the entire pipeline runs successfully, the folder structure will look like this:

```
c:\Desktop\CCBD\MRI\
├── checkpoints/                 # Model checkpoints (val_loss optimized)
│   ├── autoencoder_2d_best.pth
│   ├── autoencoder_2d_history.json
│   ├── unet_best.pth            # Trained U-Net segmentation weights
│   ├── unet_history.json
│   ├── regressor_best.pth       # Trained feature regressor weights
│   └── regressor_history.json
├── features/
│   └── predicted_features.json  # Final output feature JSON
├── ixi-t1/                      # Raw dataset folders (.nii)
├── ixi-t1-masks/                # ANTsPy pseudo-labels (.nii with correct header)
├── reports/                     # Clinical reports generated by Gemini (.txt)
├── rule_based_features.json     # Ground truth extracted target features
├── train_autoencoder.py         # Step 2 script
├── train_unet.py                # Steps 1 & 3 script
├── extract_features.py          # Step 4 script
├── train_regressor.py           # Steps 5 & 6 script
├── pipeline.py                  # Step orchestrator
└── app_server.py                # Web dashboard server
```

---

## ⚠️ Troubleshooting & Tips

*   **CUDA Out of Memory**: If training crashes with a GPU memory error, decrease the `--batch-size` parameter (e.g., use `--batch-size 8` for U-Net or `--batch-size 4` for Autoencoder).
*   **Incorrect Voxel Spacing/Orientation in BrainViewer**: Make sure you generate masks using the updated `train_unet.py` script. The updated saving code copies the correct affines and sets `qform` / `sform` to `1` so they align properly in 3D viewers.
*   **GEMINI_API_KEY environment variable missing**: Set your key in the environment before running predictions to automatically get report writeups.
