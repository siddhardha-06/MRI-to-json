// app.js - Frontend Logic and Brain Slice Visualizer for NeuroSphere

document.addEventListener("DOMContentLoaded", () => {
    // State management
    let activePatientId = null;
    let patientData = null;
    let availablePatients = [];
    let isRunningStep = false;
    let logPollInterval = null;

    // Elements
    const activeScanName = document.getElementById("active-scan-name");
    const emptyViewport = document.getElementById("empty-viewport-message");
    const analysisViewport = document.getElementById("analysis-viewport");
    const metadataSummary = document.getElementById("scan-metadata-summary");
    const metadataGrid = document.getElementById("metadata-grid");
    
    // Canvas
    const canvas = document.getElementById("brain-canvas");
    const ctx = canvas.getContext("2d");
    const toggleBB = document.getElementById("toggle-bounding-box");
    const toggleCentroid = document.getElementById("toggle-centroid");

    // Progress bars
    const barCsf = document.getElementById("bar-csf-pct");
    const barGm = document.getElementById("bar-gm-pct");
    const barWm = document.getElementById("bar-wm-pct");
    const labelCsf = document.getElementById("label-csf-pct");
    const labelGm = document.getElementById("label-gm-pct");
    const labelWm = document.getElementById("label-wm-pct");

    // System Status
    const cudaText = document.getElementById("cuda-status-text");
    const cudaDot = document.getElementById("cuda-status-dot");
    const totalScansText = document.getElementById("total-scans-text");

    // Tables
    const tableIntensity = document.getElementById("table-intensity");
    const tableMorphology = document.getElementById("table-morphology");
    const tableTexture = document.getElementById("table-texture");
    const comparisonTbody = document.getElementById("comparison-tbody");
    const comparisonStatusText = document.getElementById("comparison-status-text");

    // Report
    const reportPaper = document.getElementById("report-text-content");
    const printReportBtn = document.getElementById("btn-print-report");

    // Console
    const consoleOutput = document.getElementById("console-output");
    const clearConsoleBtn = document.getElementById("btn-clear-console");
    const consolePulse = document.getElementById("console-pulse");

    // Upload
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const uploadProgress = document.getElementById("upload-progress");
    const uploadProgressText = document.getElementById("upload-progress-text");
    const uploadBar = document.getElementById("upload-bar");

    // Step status badges
    const stepBadges = {
        1: document.getElementById("badge-step-1"),
        2: document.getElementById("badge-step-2"),
        3: document.getElementById("badge-step-3"),
        4: document.getElementById("badge-step-4"),
        5: document.getElementById("badge-step-5"),
        6: document.getElementById("badge-step-5") // combined badge
    };

    // Mapping 29-dimension features array indexes to readable names
    const FEATURE_NAMES = [
        /* 0 - 9: Intensity */
        "Intensity Mean", "Intensity Std", "Intensity Min", "Intensity Max", "Intensity Median", 
        "Percentile 25", "Percentile 75", "Skewness", "Kurtosis", "Dynamic Range",
        /* 10 - 13: Tissue */
        "CSF Percentage", "Gray Matter Percentage", "White Matter Percentage", "Total Brain Tissue Pct",
        /* 14 - 21: Texture */
        "GLCM Contrast", "GLCM Correlation", "GLCM Energy", "GLCM Homogeneity", 
        "Mean Gradient", "Std Gradient", "Max Gradient", "Shannon Entropy",
        /* 22 - 26: Structural */
        "Brain Area Percentage", "Eccentricity (Ovalness)", "Solidity (Compactness)", 
        "Extent", "Perimeter Length",
        /* 27 - 28: Symmetry */
        "Bilateral Symmetry Score", "Asymmetry Index %"
    ];

    // Initialize UI
    init();

    function init() {
        fetchSystemStatus();
        setupEventListeners();
        drawPlaceholderBrain();
        loadLatestUpload();
    }

    // Setup event handlers
    function setupEventListeners() {
        // Tabs
        const tabButtons = document.querySelectorAll(".tab-btn");
        tabButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                const targetTab = btn.getAttribute("data-tab");
                
                // Toggle active buttons
                tabButtons.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");

                // Toggle panels
                const panels = document.querySelectorAll(".tab-pane");
                panels.forEach(panel => {
                    if (panel.id === targetTab) {
                        panel.classList.add("active");
                    } else {
                        panel.classList.remove("active");
                    }
                });
            });
        });

        // Canvas redraw on options toggle
        toggleBB.addEventListener("change", () => drawBrainSlice());
        toggleCentroid.addEventListener("change", () => drawBrainSlice());

        // File upload drag & drop
        dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropZone.classList.add("dragover");
        });

        ["dragleave", "drop"].forEach(event => {
            dropZone.addEventListener(event, () => {
                dropZone.classList.remove("dragover");
            });
        });

        dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].name.endsWith(".nii")) {
                handleFileUpload(files[0]);
            } else {
                appendConsoleLog("Error: Only .nii (NIfTI) files are supported.", "error-msg");
            }
        });

        fileInput.addEventListener("change", (e) => {
            const files = e.target.files;
            if (files.length > 0) {
                handleFileUpload(files[0]);
            }
        });

        // Run Pipeline steps
        const runButtons = document.querySelectorAll(".btn-run-step");
        runButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                const step = btn.getAttribute("data-step");
                runPipelineStep(step);
            });
        });

        // Clear console
        clearConsoleBtn.addEventListener("click", () => {
            consoleOutput.innerHTML = "";
            appendConsoleLog("Console cleared.", "system-msg");
        });

        // Print Report
        printReportBtn.addEventListener("click", () => {
            const printWindow = window.open('', '_blank');
            const reportContent = document.getElementById("report-text-content").innerHTML;
            printWindow.document.write(`
                <html>
                <head>
                    <title>Clinical Report - ${activePatientId}</title>
                    <style>
                        body { font-family: sans-serif; padding: 40px; color: #222; }
                        pre { white-space: pre-wrap; font-family: monospace; font-size: 13px; line-height: 1.5; }
                    </style>
                </head>
                <body>
                    ${reportContent}
                    <script>
                        window.onload = function() { window.print(); window.close(); }
                    </script>
                </body>
                </html>
            `);
            printWindow.document.close();
        });
    }

    // Fetch details of CUDA support & Pipeline state
    function fetchSystemStatus() {
        fetch("/api/status")
            .then(res => res.json())
            .then(data => {
                if (data.cuda_available) {
                    cudaText.textContent = `${data.device || "Available"}`;
                    cudaDot.className = "status-dot status-green";
                } else {
                    cudaText.textContent = "CPU Only";
                    cudaDot.className = "status-dot status-amber";
                }
                
                // Update pipeline step statuses
                updatePipelineStepBadges(data.pipeline_status);
            })
            .catch(err => {
                cudaText.textContent = "Disconnected";
                cudaDot.className = "status-dot status-amber";
            });
    }

    function updatePipelineStepBadges(status) {
        if (!status) return;
        for (let step in status) {
            const badge = stepBadges[step];
            if (!badge) continue;
            badge.className = `flow-badge badge-${status[step].toLowerCase()}`;
            badge.textContent = status[step];
        }
    }

    // Fetch latest uploaded scan and status
    function loadLatestUpload() {
        fetch("/api/patients")
            .then(res => res.json())
            .then(data => {
                availablePatients = data.patients || [];
                
                // Filter out default dataset scans, only keep uploads
                const uploadedPatients = availablePatients.filter(id => !id.startsWith("IXI002") && !id.startsWith("IXI012") && !id.startsWith("IXI013") && !id.startsWith("IXI014") && !id.startsWith("IXI015") && !id.startsWith("IXI"));
                
                totalScansText.textContent = `${uploadedPatients.length} Uploaded`;
                
                if (uploadedPatients.length > 0) {
                    selectPatient(uploadedPatients[0]);
                } else {
                    emptyViewport.style.display = "flex";
                    analysisViewport.style.display = "none";
                    activeScanName.textContent = "No Scan Uploaded Yet";
                    metadataSummary.innerHTML = `<span class="meta-tag">Awaiting Scan Upload</span>`;
                }
            })
            .catch(err => {
                appendConsoleLog("Error checking scan upload status.", "error-msg");
            });
    }

    // Select and load patient data
    function selectPatient(patientId) {
        if (!patientId) return;
        activePatientId = patientId;
        activeScanName.textContent = patientId;
        
        // Show loaders
        emptyViewport.style.display = "none";
        analysisViewport.style.display = "flex";
        metadataSummary.innerHTML = `<span class="meta-tag">Loading details...</span>`;
        
        fetch(`/api/features/${patientId}`)
            .then(res => res.json())
            .then(data => {
                patientData = data;
                renderPatientDetails();
                drawBrainSlice();
            })
            .catch(err => {
                appendConsoleLog(`Error fetching features for patient ${patientId}`, "error-msg");
            });

        fetch(`/api/report/${patientId}`)
            .then(res => res.json())
            .then(data => {
                reportPaper.innerHTML = `<pre>${data.report || "No report generated yet."}</pre>`;
            })
            .catch(err => {
                reportPaper.innerHTML = `<p class="error-msg">Error loading report.</p>`;
            });
    }

    function renderPatientDetails() {
        if (!patientData) return;
        
        const isUploaded = patientData.source === "uploads";
        metadataSummary.innerHTML = `
            <span class="meta-tag">${isUploaded ? 'Uploaded Scan' : 'Dataset File'}</span>
            <span class="meta-tag">${patientData.scan_metadata.orientation.toUpperCase()} view</span>
            <span class="meta-tag">${patientData.scan_metadata.format} format</span>
        `;

        // Render Metadata Table
        metadataGrid.innerHTML = `
            <div class="meta-detail-item">
                <span class="meta-detail-label">Scan ID</span>
                <span class="meta-detail-value text-glow" style="font-size: 11px; word-break: break-all;">${activePatientId.substring(0, 32)}...</span>
            </div>
            <div class="meta-detail-item">
                <span class="meta-detail-label">Resolution</span>
                <span class="meta-detail-value">${patientData.scan_metadata.image_width} × ${patientData.scan_metadata.image_height}</span>
            </div>
            <div class="meta-detail-item">
                <span class="meta-detail-label">Total Voxels</span>
                <span class="meta-detail-value">${patientData.scan_metadata.total_pixels.toLocaleString()}</span>
            </div>
            <div class="meta-detail-item">
                <span class="meta-detail-label">Modality</span>
                <span class="meta-detail-value">${patientData.scan_metadata.modality}</span>
            </div>
        `;

        // Render Tissue Distribution Progress Bars
        const csf = patientData.tissue_segmentation.csf_ventricles.percentage;
        const gm = patientData.tissue_segmentation.gray_matter.percentage;
        const wm = patientData.tissue_segmentation.white_matter.percentage;

        barCsf.style.width = `${csf}%`;
        barGm.style.width = `${gm}%`;
        barWm.style.width = `${wm}%`;

        labelCsf.textContent = `${csf.toFixed(2)}%`;
        labelGm.textContent = `${gm.toFixed(2)}%`;
        labelWm.textContent = `${wm.toFixed(2)}%`;

        // Populate Intensity Table
        const intensity = patientData.intensity_statistics;
        tableIntensity.innerHTML = `
            <tr><td>Mean Intensity</td><td>${intensity.mean.toFixed(4)}</td></tr>
            <tr><td>Std Deviation</td><td>${intensity.std.toFixed(4)}</td></tr>
            <tr><td>Minimum</td><td>${intensity.min.toFixed(4)}</td></tr>
            <tr><td>Maximum</td><td>${intensity.max.toFixed(4)}</td></tr>
            <tr><td>Median</td><td>${intensity.median.toFixed(4)}</td></tr>
            <tr><td>Kurtosis</td><td>${intensity.kurtosis.toFixed(4)}</td></tr>
            <tr><td>Skewness</td><td>${intensity.skewness.toFixed(4)}</td></tr>
            <tr><td>Dynamic Range</td><td>${intensity.dynamic_range.toFixed(4)}</td></tr>
        `;

        // Populate Morphology Table
        const struct = patientData.structural_features;
        const symmetry = patientData.symmetry;
        tableMorphology.innerHTML = `
            <tr><td>Brain Area Pct</td><td>${struct.brain_area_pct.toFixed(2)}%</td></tr>
            <tr><td>Compactness (Solidity)</td><td>${struct.solidity.toFixed(4)}</td></tr>
            <tr><td>Ovalness (Eccentricity)</td><td>${struct.eccentricity.toFixed(4)}</td></tr>
            <tr><td>Perimeter</td><td>${struct.perimeter.toFixed(1)} px</td></tr>
            <tr><td>Bilateral Symmetry</td><td>${symmetry.bilateral_symmetry_score.toFixed(4)}</td></tr>
            <tr><td>Hemisphere Asymmetry</td><td>${symmetry.asymmetry_index_pct.toFixed(2)}%</td></tr>
        `;

        // Populate Texture Table
        const texture = patientData.texture_features;
        tableTexture.innerHTML = `
            <tr><td>GLCM Contrast</td><td>${texture.glcm_contrast.toFixed(4)}</td></tr>
            <tr><td>GLCM Correlation</td><td>${texture.glcm_correlation.toFixed(4)}</td></tr>
            <tr><td>GLCM Energy</td><td>${texture.glcm_energy.toFixed(4)}</td></tr>
            <tr><td>GLCM Homogeneity</td><td>${texture.glcm_homogeneity.toFixed(4)}</td></tr>
            <tr><td>Mean Gradient</td><td>${texture.mean_gradient.toFixed(4)}</td></tr>
            <tr><td>Shannon Entropy</td><td>${texture.shannon_entropy.toFixed(4)}</td></tr>
        `;

        // Populate Vector Comparison (29 features)
        const ruleBasedArray = patientData.rule_based_array || [];
        const predictedArray = patientData.predicted_array || [];
        
        comparisonTbody.innerHTML = "";
        
        let totalDeviation = 0;
        let validComparisons = 0;

        for (let i = 0; i < FEATURE_NAMES.length; i++) {
            const ruleValue = ruleBasedArray[i] !== undefined ? ruleBasedArray[i] : null;
            const predValue = predictedArray[i] !== undefined ? predictedArray[i] : null;
            
            let diff = 0;
            let pctDeviation = 0;
            if (ruleValue !== null && predValue !== null) {
                diff = Math.abs(ruleValue - predValue);
                const scale = Math.max(Math.abs(ruleValue), Math.abs(predValue));
                pctDeviation = scale > 0 ? (diff / scale) * 100 : 0;
                totalDeviation += pctDeviation;
                validComparisons++;
            }

            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>#${String(i).padStart(2, "0")}</td>
                <td class="cell-name">${FEATURE_NAMES[i]}</td>
                <td>${ruleValue !== null ? ruleValue.toFixed(4) : "N/A"}</td>
                <td>${predValue !== null ? predValue.toFixed(4) : "Pending Training"}</td>
                <td>${predValue !== null && ruleValue !== null ? diff.toFixed(4) : "N/A"}</td>
                <td>
                    ${predValue !== null && ruleValue !== null ? `
                    <div class="deviation-track" title="Deviation: ${pctDeviation.toFixed(2)}%">
                        <div class="deviation-bar" style="width: ${Math.min(100, pctDeviation * 5)}%; background-color: ${pctDeviation > 15 ? 'var(--color-pink)' : pctDeviation > 5 ? 'var(--color-amber)' : 'var(--color-green)'}"></div>
                    </div>
                    ` : '<span style="color:var(--text-muted)">N/A</span>'}
                </td>
            `;
            comparisonTbody.appendChild(tr);
        }

        // Comparison status text
        if (validComparisons > 0) {
            const avgDev = totalDeviation / validComparisons;
            if (avgDev < 2) {
                comparisonStatusText.textContent = `Excellent alignment! Average vector deviation is ${(avgDev).toFixed(2)}% (Target: < 5%).`;
                comparisonStatusText.style.color = "var(--color-green)";
            } else if (avgDev < 5) {
                comparisonStatusText.textContent = `Good alignment. Average vector deviation is ${(avgDev).toFixed(2)}%.`;
                comparisonStatusText.style.color = "var(--color-cyan)";
            } else {
                comparisonStatusText.textContent = `High vector deviation detected: ${(avgDev).toFixed(2)}%. The regressor may need more epochs.`;
                comparisonStatusText.style.color = "var(--color-amber)";
            }
        } else {
            comparisonStatusText.textContent = "Train the Feature Regressor (Step 4) to verify predictions.";
            comparisonStatusText.style.color = "var(--text-muted)";
        }
    }

    // Brain Slice Procedural Visualizer
    function drawBrainSlice() {
        if (!patientData) return;

        // Clear
        ctx.fillStyle = "#040509";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const width = canvas.width;
        const height = canvas.height;
        const cx = width / 2;
        const cy = height / 2;

        // Extract metrics for procedural scaling
        const solidity = patientData.structural_features.solidity;     // compactness
        const eccentricity = patientData.structural_features.eccentricity; // elongation
        const csf = patientData.tissue_segmentation.csf_ventricles.percentage;
        const gm = patientData.tissue_segmentation.gray_matter.percentage;
        const wm = patientData.tissue_segmentation.white_matter.percentage;
        const asymmetry = patientData.symmetry.asymmetry_index_pct;
        const bbox = patientData.structural_features.bounding_box;
        const centroid = patientData.structural_features.centroid;

        // Radial bounds
        const baseRadius = 80;
        // Eccentricity makes it oval (normal is axial slice slightly oval)
        // eccentricity goes from 0 (circle) to 1 (line).
        // Standard brain eccentricity in axial is ~0.5 - 0.7.
        const rx = baseRadius * (1 - eccentricity * 0.15);
        const ry = baseRadius * (1 + eccentricity * 0.15);

        // Compute noise seed based on solidity to simulate wrinkled brain cortex
        // Solidity is close to 1. The lower the solidity, the more lobulated/wrinkled
        const wrinkles = Math.max(0, (1 - solidity) * 20);

        // 1. Draw outer Brain / Gray Matter band
        ctx.save();
        ctx.translate(cx, cy);
        
        // Draw Lobes/Cortex outer shape
        ctx.beginPath();
        for (let angle = 0; angle < Math.PI * 2; angle += 0.05) {
            // Add custom sine wrinkles to simulate sulci
            const wrinkleFreq = 16;
            const wrinkleAmp = wrinkles * 8;
            const rOffset = Math.sin(angle * wrinkleFreq) * wrinkleAmp * Math.random(); 
            // Also divide left and right slightly using asymmetry
            let shapeAsym = 1.0;
            if (angle > Math.PI / 2 && angle < 3 * Math.PI / 2) {
                // Left hemisphere
                shapeAsym += (asymmetry / 200);
            } else {
                // Right hemisphere
                shapeAsym -= (asymmetry / 200);
            }

            const x = Math.cos(angle) * (rx * shapeAsym + rOffset);
            const y = Math.sin(angle) * (ry + rOffset);
            if (angle === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.closePath();
        
        // Gray matter gradient fill
        const gmGrad = ctx.createRadialGradient(0, 0, 10, 0, 0, ry);
        gmGrad.addColorStop(0, "#252736");
        gmGrad.addColorStop(0.7, "#585b73");
        gmGrad.addColorStop(1, "#808080");
        
        ctx.fillStyle = gmGrad;
        ctx.fill();
        ctx.strokeStyle = "#9aa0c2";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.restore();

        // 2. Draw Inner White Matter Core
        ctx.save();
        ctx.translate(cx, cy);
        
        // White matter tracks shape (smaller, less lobulated)
        const wmRx = rx * 0.75;
        const wmRy = ry * 0.78;
        ctx.beginPath();
        for (let angle = 0; angle < Math.PI * 2; angle += 0.05) {
            // Core has bigger lobes
            const wrinkleFreq = 6;
            const wrinkleAmp = 4 + wrinkles * 2;
            const rOffset = Math.sin(angle * wrinkleFreq) * wrinkleAmp;
            const x = Math.cos(angle) * (wmRx + rOffset);
            const y = Math.sin(angle) * (wmRy + rOffset);
            if (angle === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.closePath();
        
        // Fill white matter
        const wmGrad = ctx.createRadialGradient(0, 0, 5, 0, 0, wmRy);
        wmGrad.addColorStop(0, "#bfbfcf");
        wmGrad.addColorStop(1, "#e6e6e6");
        ctx.fillStyle = wmGrad;
        ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.4)";
        ctx.lineWidth = 0.8;
        ctx.stroke();
        ctx.restore();

        // 3. Draw Lateral Ventricles (CSF - Center)
        // Scaled by CSF percentage
        ctx.save();
        ctx.translate(cx, cy);
        ctx.fillStyle = "#3b3be6"; // CSF deep blue
        ctx.strokeStyle = "#5e5ef0"; // Cyan border
        ctx.lineWidth = 1.5;

        const ventScale = Math.max(3, csf * 0.8);
        
        // Draw Left Ventricle (butterfly shape)
        ctx.beginPath();
        ctx.ellipse(-12, -5, ventScale * 0.8, ventScale * 1.5, Math.PI / 12, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // Draw Right Ventricle
        ctx.beginPath();
        ctx.ellipse(12, -5, ventScale * 0.8, ventScale * 1.5, -Math.PI / 12, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // Third ventricle (thin slit)
        ctx.beginPath();
        ctx.rect(-2, -20, 4, 30);
        ctx.fill();
        ctx.stroke();

        ctx.restore();

        // 4. Overlay: Bounding Box
        if (toggleBB.checked && bbox) {
            ctx.save();
            // Scale bbox back to canvas sizing
            const scaleX = width / 256;
            const scaleY = height / 256;
            
            const x = bbox.min_col * scaleX;
            const y = bbox.min_row * scaleY;
            const w = (bbox.max_col - bbox.min_col) * scaleX;
            const h = (bbox.max_row - bbox.min_row) * scaleY;

            ctx.strokeStyle = "rgba(236, 72, 153, 0.65)"; // Pink BB
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.strokeRect(x, y, w, h);
            
            // Draw label
            ctx.fillStyle = "rgba(236, 72, 153, 0.9)";
            ctx.font = "9px JetBrains Mono";
            ctx.fillText("BBOX", x + 4, y + 12);
            ctx.restore();
        }

        // 5. Overlay: Centroid
        if (toggleCentroid.checked && centroid) {
            ctx.save();
            const scaleX = width / 256;
            const scaleY = height / 256;
            const cenX = centroid.col * scaleX;
            const cenY = centroid.row * scaleY;

            // Draw crosshair
            ctx.strokeStyle = "rgba(6, 182, 212, 0.8)"; // Cyan
            ctx.lineWidth = 1;
            
            ctx.beginPath();
            ctx.moveTo(cenX - 8, cenY);
            ctx.lineTo(cenX + 8, cenY);
            ctx.moveTo(cenX, cenY - 8);
            ctx.lineTo(cenX, cenY + 8);
            ctx.stroke();

            // Draw center dot
            ctx.beginPath();
            ctx.arc(cenX, cenY, 2, 0, Math.PI * 2);
            ctx.fillStyle = "var(--color-cyan)";
            ctx.fill();

            // Text
            ctx.fillStyle = "rgba(6, 182, 212, 0.9)";
            ctx.font = "9px JetBrains Mono";
            ctx.fillText(`C (${centroid.col.toFixed(0)},${centroid.row.toFixed(0)})`, cenX + 8, cenY - 4);
            ctx.restore();
        }
    }

    function drawPlaceholderBrain() {
        ctx.fillStyle = "#040509";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = varColor("--text-muted");
        ctx.font = "12px var(--font-sans)";
        ctx.textAlign = "center";
        ctx.fillText("Brain Canvas", canvas.width / 2, canvas.height / 2);
    }

    function varColor(varName) {
        return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
    }

    // Append standard logs to console widget
    function appendConsoleLog(text, className = "") {
        const line = document.createElement("div");
        line.className = `console-line ${className}`;
        
        // Add timestamp
        const time = new Date().toLocaleTimeString();
        line.textContent = `[${time}] ${text}`;
        
        consoleOutput.appendChild(line);
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
    }

    // Handles Drag-Drop & Input Uploading
    function handleFileUpload(file) {
        uploadProgress.style.display = "flex";
        dropZone.querySelector(".drop-zone-content").style.display = "none";
        uploadBar.style.width = "0%";
        uploadProgressText.textContent = "Uploading scan file to server...";

        const xhr = new XMLHttpRequest();
        const formData = new FormData();
        formData.append("file", file);

        xhr.upload.addEventListener("progress", (e) => {
            if (e.lengthComputable) {
                const percentComplete = (e.loaded / e.total) * 100;
                uploadBar.style.width = `${percentComplete}%`;
                uploadProgressText.textContent = `Uploading: ${percentComplete.toFixed(0)}%`;
            }
        });

        xhr.addEventListener("load", () => {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                
                uploadProgressText.textContent = "Success! Parsing features...";
                appendConsoleLog(`Upload successful: ${file.name}`, "system-msg");
                appendConsoleLog(`Server generated patient: ${response.patient_id}`, "cmd-msg");
                
                // Reload latest upload and select it
                setTimeout(() => {
                    loadLatestUpload();
                    setTimeout(() => {
                        selectPatient(response.patient_id);
                        
                        // Reset upload widget
                        uploadProgress.style.display = "none";
                        dropZone.querySelector(".drop-zone-content").style.display = "flex";
                    }, 500);
                }, 1000);
            } else {
                handleUploadFailure("Server returned upload error.");
            }
        });

        xhr.addEventListener("error", () => {
            handleUploadFailure("Network upload failed.");
        });

        xhr.open("POST", "/api/upload");
        xhr.send(formData);
    }

    function handleUploadFailure(msg) {
        uploadProgress.style.display = "none";
        dropZone.querySelector(".drop-zone-content").style.display = "flex";
        appendConsoleLog(`Upload Failed: ${msg}`, "error-msg");
        alert(`Upload Failed: ${msg}`);
    }

    // Execution of Pipeline Steps
    function runPipelineStep(step) {
        if (isRunningStep) {
            alert("Another pipeline step is currently running. Please wait.");
            return;
        }

        const dataSource = document.getElementById("pipeline-data-source").value;
        isRunningStep = true;
        consolePulse.style.display = "inline-block";
        appendConsoleLog(`Triggering Pipeline Step ${step} on dataSource '${dataSource}'...`, "cmd-msg");

        // Clear active logs and request execution
        fetch("/api/run-step", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ step: parseInt(step), data_source: dataSource })
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === "started") {
                appendConsoleLog(`Step ${step} started in background. Pid: ${data.pid}`, "system-msg");
                
                // Start polling logs
                startPollingLogs();
            } else {
                appendConsoleLog(`Failed to start Step ${step}: ${data.error}`, "error-msg");
                isRunningStep = false;
                consolePulse.style.display = "none";
            }
        })
        .catch(err => {
            appendConsoleLog(`Request error running Step ${step}`, "error-msg");
            isRunningStep = false;
            consolePulse.style.display = "none";
        });
    }

    function startPollingLogs() {
        if (logPollInterval) clearInterval(logPollInterval);
        
        logPollInterval = setInterval(() => {
            fetch("/api/logs")
                .then(res => res.json())
                .then(data => {
                    // Append new lines
                    if (data.new_lines && data.new_lines.length > 0) {
                        data.new_lines.forEach(line => {
                            let cleanLine = line.trim();
                            if (cleanLine.includes("[FAILED]") || cleanLine.includes("Error") || cleanLine.includes("[ERROR]")) {
                                appendConsoleLog(cleanLine, "error-msg");
                            } else if (cleanLine.includes("[SUCCESS]") || cleanLine.includes("✓")) {
                                appendConsoleLog(cleanLine, "cmd-msg");
                            } else {
                                appendConsoleLog(cleanLine);
                            }
                        });
                    }

                    // Check if complete
                    if (data.is_running === false) {
                        clearInterval(logPollInterval);
                        logPollInterval = null;
                        isRunningStep = false;
                        consolePulse.style.display = "none";
                        appendConsoleLog("Pipeline step execution finished.", "system-msg");
                        
                        // Reload status and details to capture updates
                        fetchSystemStatus();
                        if (activePatientId) selectPatient(activePatientId);
                    }
                })
                .catch(err => {
                    clearInterval(logPollInterval);
                    logPollInterval = null;
                    isRunningStep = false;
                    consolePulse.style.display = "none";
                });
        }, 800);
    }
});
