// app.js - Frontend Logic and Brain Slice Visualizer for NeuroSphere

document.addEventListener("DOMContentLoaded", () => {
    // State management
    let activePatientId = null;
    let patientData = null;
    let availablePatients = [];
    let isRunningStep = false;
    let logPollInterval = null;
    let mriqcPollingInterval = null;
    let lastMriqcLogCount = 0;

    // Elements
    const activeScanName = document.getElementById("active-scan-name");
    const emptyViewport = document.getElementById("empty-viewport-message");
    const analysisViewport = document.getElementById("analysis-viewport");
    const metadataSummary = document.getElementById("scan-metadata-summary");
    const metadataGrid = document.getElementById("metadata-grid");
    
    // Standalone 3D Visualizer Launch Button
    const btnLaunchViewer = document.getElementById("btn-launch-viewer");

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
    const reportToolbar = document.getElementById("report-toolbar");
    const reportDocument = document.getElementById("clinical-document-paper");
    const reportCta = document.getElementById("report-generate-cta");
    const reportGeneratingState = document.getElementById("report-generating-state");
    const generateReportBtn = document.getElementById("btn-generate-report");
    const printReportBtn = document.getElementById("btn-print-report");
    const downloadReportBtn = document.getElementById("btn-download-report");

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

    // Mapping 40-dimension features array indexes to readable names
    const FEATURE_NAMES = [
        /* 0 - 11: Intensity */
        "Intensity Mean", "Intensity Std", "Intensity Min", "Intensity Max", "Intensity Median", 
        "Percentile 25", "Percentile 75", "Skewness", "Kurtosis", "Dynamic Range",
        "Intensity Mode", "Bimodality Coefficient",
        /* 12 - 18: Tissue */
        "CSF Percentage", "Gray Matter Percentage", "White Matter Percentage", "Total Brain Tissue Pct",
        "Otsu Threshold 1", "Otsu Threshold 2", "GM/WM CNR",
        /* 19 - 26: Texture */
        "GLCM Contrast", "GLCM Correlation", "GLCM Energy", "GLCM Homogeneity", 
        "Mean Gradient", "Std Gradient", "Max Gradient", "Shannon Entropy",
        /* 27 - 32: Structural */
        "Brain Area Percentage", "Eccentricity (Ovalness)", "Solidity (Compactness)", 
        "Extent", "Perimeter Length", "Ventricular Proxy Index %",
        /* 33 - 34: Symmetry */
        "Bilateral Symmetry Score", "Hemisphere Asymmetry Index %",
        /* 35 - 37: Frequency */
        "Low Frequency Energy", "High Frequency Energy", "Frequency Energy Ratio",
        /* 38 - 39: Edge */
        "Cortical Edge Sharpness", "Cortical Edge Std"
    ];

    // Initialize UI
    init();

    function init() {
        fetchSystemStatus();
        setupEventListeners();
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

        // Launch fullscreen 3D NiiVue viewer in new tab
        if (btnLaunchViewer) {
            btnLaunchViewer.addEventListener("click", () => {
                if (activePatientId) {
                    window.open(`viewer.html?patient_id=${encodeURIComponent(activePatientId)}`, "_blank");
                } else {
                    appendConsoleLog("Please select a patient first.", "warning-msg");
                }
            });
        }

        // Run Official MRIQC Docker container
        const btnRunMriqcDocker = document.getElementById("btn-run-mriqc-docker");
        const mriqcDockerConsole = document.getElementById("mriqc-docker-console");
        const mriqcDockerStatus = document.getElementById("mriqc-docker-status");
        const mriqcDockerOutput = document.getElementById("mriqc-docker-output");
        const mriqcDockerSpinner = document.getElementById("mriqc-docker-spinner");

        if (btnRunMriqcDocker) {
            btnRunMriqcDocker.addEventListener("click", () => {
                if (!activePatientId) {
                    appendConsoleLog("Please select a patient scan first.", "warning-msg");
                    return;
                }
                
                mriqcDockerConsole.style.display = "flex";
                mriqcDockerSpinner.style.display = "block";
                mriqcDockerStatus.textContent = "Status: Starting Docker run...";
                mriqcDockerOutput.innerHTML = '<div class="console-line system-msg">Initializing Docker run request...</div>';
                btnRunMriqcDocker.disabled = true;
                btnRunMriqcDocker.textContent = "🐳 Running...";
                lastMriqcLogCount = 0;

                fetch(`/api/mriqc/run/${encodeURIComponent(activePatientId)}`, { method: 'POST' })
                    .then(res => res.json())
                    .then(data => {
                        if (data.status === "started") {
                            appendConsoleLog(`Official MRIQC Docker container started for patient ${activePatientId}.`);
                            mriqcPollingInterval = setInterval(() => {
                                pollMriqcStatus(activePatientId);
                            }, 2000);
                        } else {
                            mriqcDockerStatus.textContent = "Status: Failed to start";
                            mriqcDockerSpinner.style.display = "none";
                            btnRunMriqcDocker.disabled = false;
                            btnRunMriqcDocker.textContent = "🐳 Run Official MRIQC (Docker)";
                            mriqcDockerOutput.innerHTML += `<div class="console-line error-msg">Error: ${data.error || "Failed to start Docker process"}</div>`;
                        }
                    })
                    .catch(err => {
                        mriqcDockerStatus.textContent = "Status: Connection Error";
                        mriqcDockerSpinner.style.display = "none";
                        btnRunMriqcDocker.disabled = false;
                        btnRunMriqcDocker.textContent = "🐳 Run Official MRIQC (Docker)";
                        mriqcDockerOutput.innerHTML += `<div class="console-line error-msg">Error connecting to server: ${err}</div>`;
                    });
            });
        }

        function pollMriqcStatus(patientId) {
            fetch(`/api/mriqc/status/${encodeURIComponent(patientId)}`)
                .then(res => res.json())
                .then(data => {
                    mriqcDockerStatus.textContent = `Status: ${data.progress || data.status}`;
                    
                    if (data.logs && data.logs.length > lastMriqcLogCount) {
                        const newLogs = data.logs.slice(lastMriqcLogCount);
                        newLogs.forEach(line => {
                            const lineEl = document.createElement("div");
                            lineEl.className = "console-line";
                            if (line.includes("ERROR") || line.includes("failed") || line.includes("exited with error")) {
                                lineEl.classList.add("error-msg");
                            } else if (line.includes("Successfully") || line.includes("complete") || line.includes("Reading")) {
                                lineEl.classList.add("cmd-msg");
                            }
                            lineEl.textContent = line.trim();
                            mriqcDockerOutput.appendChild(lineEl);
                        });
                        lastMriqcLogCount = data.logs.length;
                        mriqcDockerOutput.scrollTop = mriqcDockerOutput.scrollHeight;
                    }

                    if (data.status === "completed") {
                        clearInterval(mriqcPollingInterval);
                        mriqcPollingInterval = null;
                        mriqcDockerSpinner.style.display = "none";
                        btnRunMriqcDocker.disabled = false;
                        btnRunMriqcDocker.textContent = "🐳 Run Official MRIQC (Docker)";
                        appendConsoleLog(`Official MRIQC Docker run completed successfully for patient ${patientId}.`);
                        selectPatient(patientId);
                    } else if (data.status === "failed") {
                        clearInterval(mriqcPollingInterval);
                        mriqcPollingInterval = null;
                        mriqcDockerSpinner.style.display = "none";
                        btnRunMriqcDocker.disabled = false;
                        btnRunMriqcDocker.textContent = "🐳 Run Official MRIQC (Docker)";
                        appendConsoleLog(`Official MRIQC Docker run failed for patient ${patientId}.`, "error-msg");
                    }
                })
                .catch(err => {
                    console.error("Error polling MRIQC status:", err);
                });
        }

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

        // Generate Report button
        generateReportBtn.addEventListener("click", () => {
            if (!activePatientId) {
                appendConsoleLog("Select a scan first before generating a report.", "error-msg");
                return;
            }
            // Show spinner, hide the button
            generateReportBtn.style.display = "none";
            reportGeneratingState.style.display = "flex";
            appendConsoleLog(`Generating clinical report for ${activePatientId}...`, "cmd-msg");

            fetch(`/api/report/${activePatientId}`)
                .then(res => res.json())
                .then(data => {
                    // Hide CTA, show document + toolbar
                    reportCta.style.display = "none";
                    reportDocument.style.display = "block";
                    reportToolbar.style.display = "flex";
                    reportPaper.innerHTML = `<pre>${data.report || "No report content returned."}</pre>`;
                    appendConsoleLog("Clinical report generated.", "cmd-msg");
                })
                .catch(err => {
                    // Restore button on error
                    generateReportBtn.style.display = "inline-flex";
                    reportGeneratingState.style.display = "none";
                    appendConsoleLog("Error generating report. Is the server running?", "error-msg");
                });
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

        // Download Report button
        downloadReportBtn.addEventListener("click", () => {
            if (!activePatientId) return;
            const reportText = document.getElementById("report-text-content").innerText;
            const blob = new Blob([reportText], { type: "text/plain" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `${activePatientId}_report.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });

        // Download JSON button
        const downloadJsonBtn = document.getElementById("btn-download-json");
        downloadJsonBtn.addEventListener("click", () => {
            if (!patientData) return;
            const blob = new Blob([JSON.stringify(patientData, null, 2)], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `${activePatientId}_features.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
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
                    document.getElementById("btn-download-json").style.display = "none";
                }
            })
            .catch(err => {
                appendConsoleLog("Error checking scan upload status.", "error-msg");
            });
    }

    // ── Report Tab: reset to CTA state when switching patient ────────
    function resetReportTab() {
        reportCta.style.display = "flex";
        generateReportBtn.style.display = "inline-flex";
        reportGeneratingState.style.display = "none";
        reportDocument.style.display = "none";
        reportToolbar.style.display = "none";
        reportPaper.innerHTML = "";
    }

    // Select and load patient data
    function selectPatient(patientId) {
        if (!patientId) return;
        activePatientId = patientId;
        activeScanName.textContent = patientId;
        
        // Reset MRIQC tab controls and polling state
        if (mriqcPollingInterval) {
            clearInterval(mriqcPollingInterval);
            mriqcPollingInterval = null;
        }
        lastMriqcLogCount = 0;
        
        const btnRunMriqcDocker = document.getElementById("btn-run-mriqc-docker");
        const mriqcDockerConsole = document.getElementById("mriqc-docker-console");
        const mriqcDockerStatus = document.getElementById("mriqc-docker-status");
        
        if (mriqcDockerConsole) mriqcDockerConsole.style.display = "none";
        if (btnRunMriqcDocker) {
            btnRunMriqcDocker.disabled = false;
            btnRunMriqcDocker.textContent = "🐳 Run Official MRIQC (Docker)";
        }
        if (mriqcDockerStatus) mriqcDockerStatus.textContent = "Status: Idle (Docker MRIQC not run)";
        
        // Show loaders
        emptyViewport.style.display = "none";
        analysisViewport.style.display = "flex";
        metadataSummary.innerHTML = `<span class="meta-tag">Loading details...</span>`;
        document.getElementById("btn-download-json").style.display = "none";

        // Reset report tab — user must click Generate Report
        resetReportTab();
        
        fetch(`/api/features/${patientId}`)
            .then(res => res.json())
            .then(data => {
                patientData = data;
                renderPatientDetails();
                appendConsoleLog("Interactive 3D NIfTI scan loaded. Click 'Launch Fullscreen 3D Visualizer' to explore in WebGL.");
                document.getElementById("btn-download-json").style.display = "inline-flex";
            })
            .catch(err => {
                appendConsoleLog(`Error fetching features for patient ${patientId}`, "error-msg");
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

        // Populate Volumetric Statistics Card
        const tblVolumes = document.getElementById("table-volumes");
        if (tblVolumes) {
            if (patientData.volumetric_statistics) {
                const vols = patientData.volumetric_statistics;
                tblVolumes.innerHTML = `
                    <tr><td>Total Brain Vol (GM+WM)</td><td>${vols.total_brain_volume_ml.toFixed(1)} mL</td></tr>
                    <tr><td>Gray Matter Volume</td><td>${vols.gray_matter_volume_ml.toFixed(1)} mL</td></tr>
                    <tr><td>White Matter Volume</td><td>${vols.white_matter_volume_ml.toFixed(1)} mL</td></tr>
                    <tr><td>CSF Volume</td><td>${vols.csf_volume_ml.toFixed(1)} mL</td></tr>
                    <tr><td>Left Hippocampus</td><td>${vols.left_hippocampus_volume_ml.toFixed(3)} mL</td></tr>
                    <tr><td>Right Hippocampus</td><td>${vols.right_hippocampus_volume_ml.toFixed(3)} mL</td></tr>
                    <tr><td>Total Hippocampus</td><td>${vols.total_hippocampus_volume_ml.toFixed(3)} mL</td></tr>
                `;
            } else {
                tblVolumes.innerHTML = `
                    <tr><td colspan="2" style="color: #64748b; text-align: center; font-style: italic; padding: 20px 0; font-size: 11px;">3D volumes only computed on-the-fly for uploaded T1 scans.</td></tr>
                `;
            }
        }

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

        // Populate MRIQC Quality Control Tab
        const tblMriqc = document.getElementById("table-mriqc-details");
        if (tblMriqc && patientData.mriqc_metrics) {
            const mriqcDockerStatus = document.getElementById("mriqc-docker-status");
            if (mriqcDockerStatus) {
                mriqcDockerStatus.textContent = "Status: Completed (Official MRIQC run)";
            }
            const mriqc = patientData.mriqc_metrics;
            tblMriqc.innerHTML = `
                <tr><td>Coefficient of Joint Variation (CJV)</td><td>${mriqc.cjv.toFixed(4)}</td></tr>
                <tr><td>Contrast-to-Noise Ratio (CNR)</td><td>${mriqc.cnr.toFixed(4)}</td></tr>
                <tr><td>Entropy Focus Criterion (EFC)</td><td>${mriqc.efc.toFixed(4)}</td></tr>
                <tr><td>Foreground-to-Background Energy Ratio (FBER)</td><td>${mriqc.fber.toFixed(2)}</td></tr>
                <tr><td>White Matter SNR (SNR_WM)</td><td>${mriqc.snr_wm.toFixed(4)}</td></tr>
                <tr><td>Gray Matter SNR (SNR_GM)</td><td>${mriqc.snr_gm.toFixed(4)}</td></tr>
                <tr><td>CSF SNR (SNR_CSF)</td><td>${mriqc.snr_csf.toFixed(4)}</td></tr>
                <tr><td>Ghosting Artifact Index (QI_1)</td><td>${mriqc.qi_1.toFixed(6)}</td></tr>
            `;

            // Calculate overall quality score and grade
            // CJV optimal < 0.75; CNR optimal > 1.2; EFC optimal < 0.95; QI_1 optimal < 0.05
            let score = 100;
            if (mriqc.cjv > 0.75) score -= 15;
            if (mriqc.cjv > 0.85) score -= 15;
            if (mriqc.cnr < 1.20) score -= 15;
            if (mriqc.cnr < 0.80) score -= 15;
            if (mriqc.efc > 0.95) score -= 15;
            if (mriqc.efc > 0.97) score -= 15;
            if (mriqc.qi_1 > 0.05) score -= 10;
            if (mriqc.qi_1 > 0.08) score -= 10;

            let grade = "A+";
            let label = "Excellent";
            let desc = "This scan meets or exceeds all clinical quality thresholds. Signal-to-noise ratio is high, and motion artifacts are minimal.";
            let gradeClass = "grade-excellent";

            if (score >= 90) {
                grade = "A+"; label = "Excellent"; gradeClass = "grade-excellent";
            } else if (score >= 75) {
                grade = "B"; label = "Good"; desc = "Good quality scan. Mild blur or intensity variation detected, but fully sufficient for reliable clinical feature extraction."; gradeClass = "grade-good";
            } else if (score >= 60) {
                grade = "C"; label = "Acceptable"; desc = "Acceptable scan quality. Moderate noise, distortion, or artifact levels present. Interpret metrics with slight caution."; gradeClass = "grade-acceptable";
            } else {
                grade = "F"; label = "Poor"; desc = "Poor scan quality. Significant motion artifacts, low signal-to-noise ratio, or severe distortion detected. Rescan recommended."; gradeClass = "grade-poor";
            }

            const badge = document.getElementById("mriqc-rating-badge");
            const qLabel = document.getElementById("mriqc-quality-label");
            const qDesc = document.getElementById("mriqc-quality-desc");

            if (badge) {
                badge.textContent = grade;
                badge.className = "rating-badge " + gradeClass;
            }
            if (qLabel) qLabel.textContent = label;
            if (qDesc) qDesc.textContent = desc;

            // Update threshold status dots (green, yellow, red)
            const updateDot = (id, isGreen, isYellow) => {
                const el = document.getElementById(id);
                if (el) {
                    el.className = "threshold-dot " + (isGreen ? "status-green" : isYellow ? "status-yellow" : "status-red");
                }
            };

            updateDot("dot-cjv", mriqc.cjv < 0.75, mriqc.cjv < 0.85);
            updateDot("dot-cnr", mriqc.cnr > 1.20, mriqc.cnr > 0.80);
            updateDot("dot-efc", mriqc.efc < 0.95, mriqc.efc < 0.97);
            updateDot("dot-qi1", mriqc.qi_1 < 0.05, mriqc.qi_1 < 0.08);
        } else if (tblMriqc) {
            tblMriqc.innerHTML = `
                <tr><td colspan="2" style="color: #64748b; text-align: center; font-style: italic; padding: 20px 0; font-size: 11px;">Image Quality Metrics unavailable for this scan.</td></tr>
            `;
        }
    }

    // Standalone viewer launched in separate tab; local definitions removed.

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
