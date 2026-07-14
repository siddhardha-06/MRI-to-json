# MRI-JSON Pipeline Technical Breakdown

## A. Dataset
- **Expected Format:** The pipeline strictly expects NIfTI files (`.nii` or `.nii.gz`).
- **Data Loading:** Data loading is handled via caching functions to circumvent Windows file descriptor limits (WinError 8):
  - In `train_regressor.py` and `extract_features.py`: Uses `load_and_preprocess_mri()` (or `load_mri()`) calling `nibabel.load()`.
  - In `train_autoencoder.py`: Uses `_load_and_normalize_volume()` calling `nibabel.load()`.
  - In `train_unet.py` (ANTsPy masking): Uses a custom safe wrapper `load_nifti_as_ants()` which loads via `nibabel` first and then converts to ANTs via `ants.from_numpy()` to bypass internal HDF5 detection errors on Windows paths.
- **Dataset Filtering:** Both `train_unet.py` and `train_regressor.py` implement an optional dataset subset limit. This is triggered via the command-line argument `--limit <int>`, which truncates the list of discovered files: `if limit is not None and limit > 0: shuffled_files = shuffled_files[:limit]`.
- **Preprocessing:**
  - **Normalization:** In almost all models, intensity is clipped to the 1st and 99th percentiles using `np.percentile(arr, (1, 99))`. The clipped values are then min-max scaled to a `[0, 1]` range: `(arr - p1) / (p99 - p1 + 1e-5)`.
  - **Resizing:** For the 2D networks, slices are resized to 256×256 using bilinear interpolation (`torch.nn.functional.interpolate` with `mode='bilinear'`).
  - **Slicing Strategy:** The pipeline relies on a 5-slice central extraction strategy for features and regression. Instead of processing the entire 3D volume, it isolates the central slice (`mid = D // 2`) and takes the 5 slices surrounding it (`[-2, -1, 0, 1, 2]`), averaging their outputs. `train_autoencoder.py` is the exception, extracting 10 evenly spaced slices across the central 50% of the volume (`np.linspace(D//4, 3*D//4, 10)`).

## B. Scan Quality Validation
- **Quality Validation Logic:** Quality validation is split into two methods handled primarily in `app_server.py`.
- **Computed Metrics:** Eight distinct metrics are evaluated: CJV (Coefficient of Joint Variation), CNR (Contrast-to-Noise Ratio), EFC (Entropy Focus Criterion), FBER (Foreground-to-Background Energy Ratio), SNR_WM, SNR_GM, SNR_CSF, and QI_1 (Artifact/Ghosting Index).
- **Paths and Triggers:**
  1. **Docker MRIQC Path:** Triggered manually via the dashboard (`run_official_mriqc_thread`). It packages the NIfTI into a temporary BIDS directory format and runs the `nipreps/mriqc:latest` Docker image. It parses the resulting JSON file.
  2. **Fast Emulator (Fallback):** Automatically triggered to populate the UI if the Docker container is not run. It executes `simulate_mriqc_from_vector(vector)` which uses deterministically "fake" clinical physics formulas derived entirely from the feature vector. For example: `cjv = np.clip(0.32 + (std_val / (mean_val + 1e-5)) * 0.15, 0.25, 0.65)`.
- **Pass/Fail Thresholds:** A strict scoring logic exists in `compute_mriqc_quality_score()`. It deducts points from a starting score of 100 based on threshold violations (e.g., `-15` if CJV > 0.85, `-15` if EFC > 0.97). The thresholds are:
  - **A+ (Pass):** Score >= 90
  - **B (Pass):** Score >= 75
  - **C (Pass):** Score >= 50
  - **F (Fail):** Score < 50 ("Scan quality below minimum threshold. High motion/noise artifacts detected. Rescan recommended.")

## C. Preprocessing
- **Bias Correction and Tissue Segmentation:** Handled by `train_unet.py` inside the `generate_masks_antspy()` function.
- **Function Calls:**
  - Bias field correction: `ants.n4_bias_field_correction(img)`
  - Tissue segmentation: `ants.atropos(a=img_corrected, m='[0.2,1x1x1]', c='[5,0.00001]', i='kmeans[3]', x=brain_mask)`
- **Pseudo-labels:** The pseudo-labels are generated as integer arrays mapped to background (0) and 3 K-Means clusters representing CSF (1), GM (2), and WM (3). The script includes a fail-safe check to verify the intensity order of the K-Means clusters and remaps them if the initialization swapped GM and WM. They are saved as NIfTI format: `<mask_dir>/<stem>_seg.nii`. The N4-corrected volume is concurrently saved to `<mask_dir>/n4/<stem>_n4.nii` to prevent domain shift during downstream model training.

## D. Self-Supervised Representation Learning (Autoencoder)
- **Exact Architecture (2D):** 
  - **Encoder:** 6 Convolutional Layers: `Conv2d(1->32)`, `Conv2d(32->64)`, `Conv2d(64->128)`, `Conv2d(128->256)`, `Conv2d(256->512)`, `Conv2d(512->512)`. All convolutions use `kernel_size=4`, `stride=2`, `padding=1`, followed by `ReLU`. This is flattened into a `Linear(8192 -> latent_dim)`. No activation on the bottleneck.
  - **Decoder:** `Linear(latent_dim -> 8192)` -> `ReLU`, reshaped to `(512, 4, 4)`. Followed by 6 Transposed Convolutional Layers: `ConvTranspose2d(512->256)`, `ConvTranspose2d(256->128)`, `ConvTranspose2d(128->64)`, `ConvTranspose2d(64->32)`, `ConvTranspose2d(32->16)` (all `k=4, s=2, p=1` + `ReLU`), and a final `ConvTranspose2d(16->1)` + `Sigmoid`.
- **Input / Latent Shape:** The input is resized to `(1, 256, 256)`. The default `latent_dim` is 128.
- **Hyperparameters:** Loss: `nn.MSELoss()`. Optimizer: `Adam(lr=1e-3)`. Scheduler: `CosineAnnealingLR(T_max=num_epochs)`.
- **Defaults:** Batch size is 8. Epoch count is 50.
- **Split & Checkpoints:** Uses an 80/20 train/validation split (`val_split=0.2`). The split is applied at the subject level deterministically via `random.seed(42)`. The checkpoint is saved exclusively if the validation loss improves: `if val_metric < best_val_loss`.

## E. Supervised Tissue Segmentation (U-Net)
- **Exact Architecture:**
  - Features 3 Downsampling/Encoder blocks, 1 Bottleneck, and 3 Upsampling/Decoder blocks.
  - Channels progress: `1 -> 64 -> 128 -> 256 -> 512 (Bottleneck) -> 256 -> 128 -> 64 -> 4`.
  - Block definition (`ConvBlock`): `Conv2d(k=3, p=1)` -> `BatchNorm2d` -> `ReLU` -> `Conv2d(k=3, p=1)` -> `BatchNorm2d` -> `ReLU`.
  - Downsampling utilizes `MaxPool2d(2, 2)`. Upsampling utilizes `ConvTranspose2d(k=2, s=2)`.
  - Skip connections use `torch.cat([x, enc_n], dim=1)`. Final layer is a `Conv2d(k=1)` outputting raw logits.
- **Loss Function:** A hybrid criterion combining Weighted Cross-Entropy and Dice Loss: `ce_loss(logits, targets) + 0.5 * dice_loss(logits, targets)`. The Cross-Entropy weights are set to `[0.1, 0.3, 0.3, 0.3]` to penalize background pixels less.
- **Hyperparameters:** Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`. Scheduler: `CosineAnnealingLR`. Batch size defaults to 16, epochs to 50.
- **Ground Truth:** Derived directly from the ANTsPy pseudo-labels. The model outputs 4 classes: Background (0), CSF (1), Gray Matter (2), and White Matter (3).

## F. Quantitative Feature Extraction
- **Extracted Features (Total 61):**
  - **Intensity Statistics (12):** `mean`, `std`, `min`, `max`, `median`, `percentile_25`, `percentile_75`, `skewness`, `kurtosis`, `dynamic_range`, `mode`, `bimodality`. Computed exclusively on masked brain voxels (`arr > 0.05`).
  - **Tissue Segmentation (7):** `csf_pct`, `gm_pct`, `wm_pct`, `brain_tissue_pct`, `otsu_threshold_1`, `otsu_threshold_2`, `gm_wm_cnr`.
  - **Texture Metrics (8):** `glcm_contrast`, `glcm_correlation`, `glcm_energy`, `glcm_homogeneity`, `mean_gradient`, `std_gradient`, `max_gradient`, `shannon_entropy`. GLCM computed via `skimage.feature.graycomatrix` with distance=1, angles=[0, π/4, π/2, 3π/4], levels=8 on a bounding box cropped to the brain.
  - **Structural Features (6):** `brain_area_pct`, `eccentricity`, `solidity`, `extent`, `perimeter`, `ventricular_proxy_pct`. Area derived via `skimage.measure.regionprops`.
  - **Symmetry Metrics (2):** `bilateral_symmetry_score`, `asymmetry_index_pct`. Score derived by correlating the left half with the flipped right half: `np.corrcoef(left, flipped_right)`.
  - **Frequency Domain (3):** `low_frequency_energy`, `high_frequency_energy`, `frequency_energy_ratio`. Computed via `np.fft.fft2()`.
  - **Edge Features (2):** `cortical_edge_sharpness`, `cortical_edge_std`. Extracted using `skimage.filters.sobel(arr)`.
  - **Subcortical Volumes (21):** Extracted via a secondary method using `extract_subcortical_features(mri_path)`. Uses `ants.registration` to align the subject to the MNI152 template and overlays the Harvard-Oxford max-probability atlas. Extracts `Left` volume, `Right` volume, and `Asymmetry Index` for 7 structures: Ventricles, Thalamus, Caudate, Putamen, Pallidum, Hippocampus, and Amygdala.
- **Output Count:** The pipeline outputs exactly 61 completely flattened scalar values per volume into a single vector array.

## G. Feature Regression
- **Exact Architecture:**
  - **Backbone (Shared with AE Encoder):** 6 Convolutional Layers: `Conv2d(1->32)`, `Conv2d(32->64)`, `Conv2d(64->128)`, `Conv2d(128->256)`, `Conv2d(256->512)`, `Conv2d(512->512)`. All convolutions use `kernel_size=4`, `stride=2`, `padding=1`, followed by `ReLU`.
  - **Regressor Head:** Flattens the `(512, 4, 4)` tensor into 8192 inputs. `Linear(8192 -> 128)` -> `ReLU` -> `Dropout(0.3)` -> `Linear(128 -> num_features)`.
- **Feature Matching:** The argument parser default for `num_features` is exactly `61`. The codebase invokes the model utilizing all 61 values dynamically normalized against a `feature_mean` and `feature_std` calculated from the training subset. There are no subset exclusions; it is a 1:1 match.
- **Input Shape:** The regressor uses multi-slice augmentation during training. Slices are resized via bilinear interpolation to `(1, 256, 256)`. During training, the regressor loops over offsets `[-2, -1, 0, 1, 2]` from the center.
- **Hyperparameters:** Loss: `nn.MSELoss()`. Optimizer: `Adam(lr=1e-3, weight_decay=1e-5)`. Scheduler: `CosineAnnealingLR`. Batch size defaults to 32, epochs to 100. Incorporates Early Stopping with `patience=10`.
- **Inference Bypass:** At inference time, executing `train_regressor.py` with the `--predict` flag averts rule-based processing. The system feeds the target image's 5 central slices through the trained regression model (`model(mri_slice)`), averages the 5 predictions (`np.mean(slices_preds, axis=0)`), and denormalizes the resultant 61 features into `predicted_features.json`.

## H. Clinical Report Generation
- **LLM Integration:** `generate_report.py` executes calls to the Gemini API (`google-genai` library, specifying `gemini-2.5-flash-lite` by default).
- **Prompt:** The system utilizes the prompt: *"You are a neuroradiology AI assistant that interprets quantitative MRI feature data and generates clear, concise clinical summary reports. You will be given structured JSON data extracted from a brain MRI scan... Generate a professional clinical impression/summary that: 1. Summarizes the key findings in plain English 2. Interprets what the numbers mean clinically 3. Highlights any flagged pathology indicators 4. Gives an overall assessment. Keep it to 3-4 paragraphs, professional medical language."*
- **Formatting and Output:** The text returned by Gemini is purely the interpretive summary block. The pipeline injects this response into a rigid, professional ASCII-art document template (`format_clinical_report`), mapping key extracted features to structured fields (e.g., CSF%, GLCM Contrast, Brain Area Coverage). Reports are saved as plaintext `.txt` files in the `reports/` directory.
- **Error Handling:** If the `GEMINI_API_KEY` is not detected in the environment variables, the script immediately prints an error and halts execution `exit(1)`.

## I. Dashboard and Deployment
- **API Endpoints (`app_server.py`):**
  - `GET /api/status`: System health and pipeline step readiness.
  - `GET /api/patients`: Lists analyzed patients.
  - `GET /api/features/<id>`: Returns the fully constructed JSON schema.
  - `GET /api/report/<id>`: Fetches the Gemini clinical report.
  - `GET /api/mriqc/status/<id>`: Polling state for Docker MRIQC.
  - `GET /api/mriqc/quality-gate/<id>`: MRIQC score interpretation logic.
  - `GET /api/logs`: Terminal log streaming.
  - `POST /api/upload`: Receives and begins extraction on dragged/dropped NIfTIs.
  - `POST /api/run-step`: Triggering modular pipeline steps.
  - `POST /api/mriqc/run/<id>`: Kicks off the MRIQC subprocess.
- **Model Usage in Web:** The web server strictly leverages `extract_features.py`'s rule-based logic (via `run_feature_extraction()`) dynamically upon file upload. It calculates 3D volumetric metrics and dynamic N4 correction live. It does *not* load or utilize the trained U-Net or Regressor natively within the app interface.
- **New File Drag-And-Drop:** An uploaded NIfTI runs through dynamic N4 bias field correction, computes the fast 61-feature extraction over the central 5 slices, creates a bounding box representation of the middle slice, dynamically invokes `ants.atropos` and affine registration to the Harvard-Oxford template for 3D volumetric counts, and synthesizes the final JSON response.
- **Docker MRIQC Integration:** A backend thread writes the file to a temporary BIDS-compliant directory (`sub-<id>/anat/sub-<id>_T1w.nii`) and writes a `dataset_description.json`. It triggers a local Docker socket execution: `docker run --rm -v ... nipreps/mriqc:latest ...` and tails the stdout. The resulting JSON file is parsed, merged into the local cache, and visually replaces the emulator numbers.
- **3D Slice Viewer:** Implemented using NIfTI parsing logic on the web client-side (relying on static Javascript like `NIFTI-Reader-JS` or WebGL volumetric rendering natively in the browser).

## Final Cross-Check (Discrepancies)
1. **Fallback Quality Metrics are Fake:** The fast fallback for MRIQC quality metrics generates deterministic values derived via synthetic multipliers off standard intensities (e.g., `cjv = np.clip(0.32 + (std_val / mean_val) * 0.15)`). The README implies this is a "fast mathematical emulator", but it is clinically fabricated approximation logic without true physical artifact rendering.
2. **Feature Extraction Breadth:** The README suggests that `extract_features.py` operates universally on "all raw MRIs" representing global metrics. In reality, the codebase extracts its primary intensity, texture, and structural features solely from the 5 central axial slices (middle ± 2) and averages them (`extract_all_features()` lines 423-494).
3. **Hard Dependencies Missing:** The README lacks reference to two hard dependencies required for the Subcortical Volumes extraction: `MNI152_T1_2mm_brain.nii.gz` and `HarvardOxford-sub-maxprob-thr25-2mm.nii.gz` located in `checkpoints/`. If absent, the volumetric logic silently outputs an empty sequence of zeros.
4. **Segmentation Classes Count:** The README refers to segmenting `(CSF, GM, WM)`, implying 3 classes. `train_unet.py` fundamentally structures and penalizes the output across 4 semantic classes: Background (0), CSF (1), GM (2), and WM (3).
