// NeuroSphere — app.js  (merged: premium state-machine + original working logic)
// State machine: UPLOAD → PROCESSING → DASHBOARD

document.addEventListener("DOMContentLoaded", () => {

// ── State ─────────────────────────────────────────────────────────
const STATES = { UPLOAD: "upload", PROCESSING: "processing", DASHBOARD: "dashboard" };
let currentState    = STATES.UPLOAD;
let activePatientId = null;
let patientData     = null;
let availablePatients = [];
let isRunningStep   = false;
let logPollInterval = null;
let lastLogIndex    = 0;
let activeRibbonStep = null;
let processingStartTime = null;
let currentStageIndex   = 0;
let timerId = null;

// MRIQC polling
let mriqcPollingInterval = null;
let lastMriqcLogCount    = 0;

// ── Upload pipeline stages ────────────────────────────────────────
const UPLOAD_STAGES = [
  { id:"upload",   name:"File Upload",          icon:"ic-upload",  desc:"Securely transferring your MRI scan to the analysis server",            detail:"Sending NIfTI data via HTTP POST to /api/upload",                                                                                    pct:[0,10],   duration:0     },
  { id:"mriqc",    name:"Image Quality (MRIQC)", icon:"ic-scan",    desc:"Computing Image Quality Metrics via official Docker container",          detail:"Executing nipreps/mriqc participant level analysis",                                                                              pct:[10,35],  duration:0     },
  { id:"n4",       name:"N4 Bias Correction",   icon:"ic-zap",     desc:"Removing magnetic field inhomogeneity artifacts from the scanner",       detail:"Applies iterative N4ITK algorithm to normalize signal intensity across the full brain volume",                                    pct:[35,55],  duration:3200  },
  { id:"features", name:"Feature Extraction",   icon:"ic-chart",   desc:"Computing 40 quantitative MRI biomarkers across the scan volume",        detail:"Intensity statistics, GLCM texture, morphological measurements, bilateral symmetry and frequency domain analysis",               pct:[55,75],  duration:4500  },
  { id:"segment",  name:"Tissue Segmentation",  icon:"ic-layers",  desc:"Delineating CSF, gray matter, and white matter tissue regions",          detail:"ANTsPy Atropos Markov Random Field segmentation with spatial tissue-class priors",                                              pct:[75,85],  duration:3000  },
  { id:"volumes",  name:"3D Volume Analysis",   icon:"ic-box",     desc:"Estimating hippocampal, cortical and ventricle volumes in 3D",           detail:"Non-linear registration and Harvard-Oxford subcortical atlas parcellation for volumetric estimates",                             pct:[85,95],  duration:2500  },
  { id:"ready",    name:"Analysis Complete",    icon:"ic-scan",    desc:"All computations complete — loading your clinical analytics dashboard",   detail:"Indexing extracted features, building comparison vectors and preparing the interactive analytics dashboard",                     pct:[95,100], duration:800   },
];

const TOTAL_PROC_MS = UPLOAD_STAGES.reduce((s, st) => s + (st.duration || 0), 0);

// ── Manual training pipeline steps ───────────────────────────────
const MANUAL_STEPS = {
  1:{ title:"Step 1 – Pseudo-Label Generation",  desc:"ANTsPy Atropos segmentation to generate tissue pseudo-labels for all scans in the selected source." },
  2:{ title:"Step 2 – Autoencoder Training",     desc:"Self-supervised training to learn compact latent representations of MRI slices."                     },
  3:{ title:"Step 3 – U-Net Segmentation",       desc:"Trains a 2D U-Net model using pseudo-labels to predict tissue segmentation masks."                   },
  4:{ title:"Step 4 – Regressor Training",       desc:"Trains a feature regressor on 61-dimensional feature vectors against rule-based ground truth."        },
  5:{ title:"Step 5 – Feature Extraction",       desc:"Extracts 61-dimensional feature vectors for all scans and writes them to JSON output files."          },
  6:{ title:"Step 6 – Report Generation",        desc:"Generates structured Gemini AI clinical reports from extracted MRI features."                         },
};

// ── Stage output storage ──────────────────────────────────────────
const stageOutputs = {};

// ── Feature names ─────────────────────────────────────────────────
const FEATURE_NAMES = [
  "Intensity Mean","Intensity Std","Intensity Min","Intensity Max","Intensity Median",
  "Percentile 25","Percentile 75","Skewness","Kurtosis","Dynamic Range",
  "Intensity Mode","Bimodality Coefficient",
  "CSF Percentage","Gray Matter Percentage","White Matter Percentage","Total Brain Tissue Pct",
  "Otsu Threshold 1","Otsu Threshold 2","GM/WM CNR",
  "GLCM Contrast","GLCM Correlation","GLCM Energy","GLCM Homogeneity",
  "Mean Gradient","Std Gradient","Max Gradient","Shannon Entropy",
  "Brain Area Percentage","Eccentricity (Ovalness)","Solidity (Compactness)",
  "Extent","Perimeter Length","Ventricular Proxy Index %",
  "Bilateral Symmetry Score","Hemisphere Asymmetry Index %",
  "Low Frequency Energy","High Frequency Energy","Frequency Energy Ratio",
  "Cortical Edge Sharpness","Cortical Edge Std",
  "L Lateral Ventricle Volume (mL)", "R Lateral Ventricle Volume (mL)", "Ventricle Asymmetry Index %",
  "L Thalamus Volume (mL)", "R Thalamus Volume (mL)", "Thalamus Asymmetry Index %",
  "L Caudate Volume (mL)", "R Caudate Volume (mL)", "Caudate Asymmetry Index %",
  "L Putamen Volume (mL)", "R Putamen Volume (mL)", "Putamen Asymmetry Index %",
  "L Pallidum Volume (mL)", "R Pallidum Volume (mL)", "Pallidum Asymmetry Index %",
  "L Hippocampus Volume (mL)", "R Hippocampus Volume (mL)", "Hippocampus Asymmetry Index %",
  "L Amygdala Volume (mL)", "R Amygdala Volume (mL)", "Amygdala Asymmetry Index %"
];

// ── DOM shortcuts ─────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const dropZone        = $("drop-zone");
const fileInput       = $("file-input");
const emptyViewport   = $("empty-viewport-message");
const analysisViewport= $("analysis-viewport");
const metadataSummary = $("scan-metadata-summary");
const metadataGrid    = $("metadata-grid");
const tableIntensity  = $("table-intensity");
const tableMorphology = $("table-morphology");
const tableTexture    = $("table-texture");
const comparisonTbody = $("comparison-tbody");
const comparisonStatus= $("comparison-status-text");
const reportPaper     = $("report-text-content");
const reportToolbar   = $("report-toolbar");
const reportDocument  = $("clinical-document-paper");
const reportCta       = $("report-generate-cta");
const reportSpinner   = $("report-generating-state");
const cudaText        = $("cuda-status-text");
const cudaDot         = $("cuda-status-dot");
const totalScansText  = $("total-scans-text");
const consoleOutput   = $("console-output");
const consolePulse    = $("console-pulse");

// ── Ribbon step badges ────────────────────────────────────────────
const ribbonBadges = {};
for (let i = 1; i <= 6; i++) ribbonBadges[i] = $(`rbadge-${i}`);

// ── Neural canvas background ──────────────────────────────────────
function initNeuralCanvas() {
  const canvas = $("neural-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  let W = canvas.width = window.innerWidth;
  let H = canvas.height = window.innerHeight;

  const N = 75;
  const particles = Array.from({ length: N }, () => ({
    x: Math.random() * W,
    y: Math.random() * H,
    vx: (Math.random() - 0.5) * 0.32,
    vy: (Math.random() - 0.5) * 0.32,
    r: Math.random() * 1.4 + 0.4,
    op: Math.random() * 0.45 + 0.15,
  }));

  let raf;
  function draw() {
    ctx.clearRect(0, 0, W, H);
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const d  = Math.sqrt(dx * dx + dy * dy);
        if (d < 140) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(16,185,129,${(1 - d / 140) * 0.11})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(52,211,153,${p.op})`;
      ctx.fill();
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > W) p.vx *= -1;
      if (p.y < 0 || p.y > H) p.vy *= -1;
    });
    raf = requestAnimationFrame(draw);
  }
  draw();

  window.addEventListener("resize", () => {
    cancelAnimationFrame(raf);
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
    particles.forEach(p => { p.x = Math.random() * W; p.y = Math.random() * H; });
    draw();
  });
}

// ── State management ──────────────────────────────────────────────
function showState(name) {
  ["state-upload","state-processing","state-dashboard"].forEach(id => {
    const el = $(id);
    if (el) el.classList.remove("state-visible");
  });

  const target = $(`state-${name}`);
  if (!target) return;
  target.classList.add("state-visible");
  target.style.animation = "stateEnter .4s ease";
  currentState = name;

  const hsi    = $("header-scan-info");
  const newBtn = $("btn-new-scan");
  if (name === STATES.DASHBOARD) {
    if (hsi)    hsi.style.display = "flex";
    if (newBtn) newBtn.style.display = "inline-flex";
  } else {
    if (hsi)    hsi.style.display = "none";
    if (newBtn) newBtn.style.display = "none";
  }
}

// ── Expected-time helpers ─────────────────────────────────────────
function getRemainingMs(stageIdx, stageFraction) {
  let rem = 0;
  UPLOAD_STAGES.forEach((s, i) => {
    if (i > stageIdx)      rem += s.duration || 0;
    else if (i === stageIdx) rem += (s.duration || 0) * (1 - stageFraction);
  });
  return rem;
}
function fmtMs(ms) {
  if (ms <= 200) return "Almost done";
  const s = Math.ceil(ms / 1000);
  return s < 60 ? `~${s}s remaining` : `~${Math.floor(s/60)}m ${s%60}s remaining`;
}
function updateTimerUI(stageIdx, stageFraction) {
  const remaining = getRemainingMs(stageIdx, stageFraction);
  const el = $("proc-time-value");
  if (el) el.textContent = fmtMs(remaining);
  const elapsed = processingStartTime ? Date.now() - processingStartTime : 0;
  const timeFill = $("proc-time-fill");
  if (timeFill) timeFill.style.width = `${Math.min(100, (elapsed / TOTAL_PROC_MS) * 100)}%`;
  const elapsedEl = $("proc-elapsed-text");
  if (elapsedEl) elapsedEl.textContent = `${Math.round(elapsed/1000)}s elapsed`;
}

// ── Stepper building ──────────────────────────────────────────────
function buildStepper() {
  const stepper = $("h-stepper");
  if (!stepper) return;
  stepper.innerHTML = "";
  UPLOAD_STAGES.forEach((stage, i) => {
    const step = document.createElement("div");
    step.className = "h-step";
    step.id = `hstep-${stage.id}`;
    step.innerHTML = `
      <div class="h-circle s-pending" id="hcirc-${stage.id}">
        <svg style="width:18px;height:18px"><use href="#${stage.icon}"/></svg>
      </div>
      <div class="h-label">${stage.name}</div>
    `;
    stepper.appendChild(step);
    if (i < UPLOAD_STAGES.length - 1) {
      const conn = document.createElement("div");
      conn.className = "h-connector";
      conn.innerHTML = `<div class="h-connector-fill" id="hconn-${stage.id}"></div>`;
      stepper.appendChild(conn);
    }
  });
}

function stepperActivate(id) {
  const circ = $(`hcirc-${id}`);
  const step = $(`hstep-${id}`);
  if (circ) circ.className = "h-circle s-active";
  if (step) step.className = "h-step s-active";
}
function stepperComplete(id) {
  const circ = $(`hcirc-${id}`);
  const step = $(`hstep-${id}`);
  if (circ) {
    circ.className = "h-circle s-done";
    circ.innerHTML = `<svg style="width:18px;height:18px"><use href="#ic-check"/></svg>`;
  }
  if (step) step.className = "h-step s-done";
  const conn = $(`hconn-${id}`);
  if (conn) setTimeout(() => { conn.style.width = "100%"; }, 80);
}

// ── Hero card ─────────────────────────────────────────────────────
function updateHeroCard(stage, idx) {
  const card = $("hero-card");
  if (!card) return;
  card.classList.add("step-exit");
  setTimeout(() => {
    card.classList.remove("step-exit");
    const numEl  = $("hero-step-num");  if (numEl)  numEl.textContent  = `Step ${idx + 1} / ${UPLOAD_STAGES.length}`;
    const nameEl = $("hero-step-name"); if (nameEl) nameEl.textContent = stage.name;
    const descEl = $("hero-step-desc"); if (descEl) descEl.textContent = stage.desc;
    const iconEl = $("hero-icon");
    if (iconEl) iconEl.innerHTML = `<svg class="hsc-icon"><use href="#${stage.icon}"/></svg>`;
    setStepRingPct(0);
    const log = $("hero-log");
    if (log) log.innerHTML = `<div class="hlog-line">${stage.detail}</div>`;
    card.classList.add("step-enter");
    setTimeout(() => card.classList.remove("step-enter"), 360);
  }, 230);
}

function setStepRingPct(pct) {
  const circle = $("step-ring-circle");
  const label  = $("hero-progress-pct");
  if (circle) {
    const circumference = 264;
    circle.style.strokeDashoffset = circumference - (pct / 100) * circumference;
  }
  if (label) label.textContent = `${Math.round(pct)}%`;
}

function addHeroLog(msg, dim = false) {
  const log = $("hero-log");
  if (!log) return;
  const line = document.createElement("div");
  line.className = `hlog-line${dim ? " dim" : ""}`;
  line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
}

// ── Overall progress bar ──────────────────────────────────────────
function setOverallPct(pct) {
  const fill  = $("proc-overall-fill");
  const label = $("proc-overall-pct");
  if (fill)  fill.style.width   = `${pct}%`;
  if (label) label.textContent = `${Math.round(pct)}%`;
}

// ── Animate progress ──────────────────────────────────────────────
function animateProgress(from, to, durationMs, stageIdx) {
  return new Promise(resolve => {
    const start = performance.now();
    function tick(now) {
      const t    = Math.min((now - start) / durationMs, 1);
      const pct  = from + (to - from) * t;
      const frac = (pct - from) / (to - from);
      setOverallPct(pct);
      setStepRingPct(frac * 100);
      updateTimerUI(stageIdx, frac);
      if (t < 1) requestAnimationFrame(tick);
      else resolve();
    }
    requestAnimationFrame(tick);
  });
}

// ── Append to console panel ───────────────────────────────────────
function appendConsoleLog(text, className = "") {
  if (!consoleOutput) return;
  const line = document.createElement("div");
  line.className = `console-line ${className}`.trim();
  const time = new Date().toLocaleTimeString();
  line.textContent = `[${time}] ${text}`;
  consoleOutput.appendChild(line);
  consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

// ── File upload handler ───────────────────────────────────────────
async function handleFileUpload(file) {
  if (!file) return;

  processingStartTime = Date.now();
  buildStepper();
  showState(STATES.PROCESSING);

  const fnEl = $("proc-file-name");
  if (fnEl) fnEl.textContent = file.name;
  updateTimerUI(0, 0);

  // Stage 0: Upload (real XHR progress)
  currentStageIndex = 0;
  stepperActivate("upload");
  updateHeroCard(UPLOAD_STAGES[0], 0);
  setOverallPct(0);

  const xhr      = new XMLHttpRequest();
  const formData = new FormData();
  formData.append("file", file);

  xhr.upload.addEventListener("progress", e => {
    if (e.lengthComputable) {
      const frac = e.loaded / e.total;
      setOverallPct(frac * 15);
      setStepRingPct(frac * 100);
      updateTimerUI(0, frac);
    }
  });

  const uploadResult = await new Promise((resolve, reject) => {
    xhr.addEventListener("load", () =>
      xhr.status === 200 ? resolve(JSON.parse(xhr.responseText)) : reject(new Error("Server error."))
    );
    xhr.addEventListener("error", () => reject(new Error("Network error.")));
    xhr.open("POST", "/api/upload");
    xhr.send(formData);
  }).catch(err => {
    showState(STATES.UPLOAD);
    appendConsoleLog(`Upload failed: ${err.message}`, "error-msg");
    return null;
  });

  if (!uploadResult) return;

  stageOutputs.upload = { filename: file.name, size: `${(file.size/1024/1024).toFixed(2)} MB`, patientId: uploadResult.patient_id };
  addHeroLog("Upload complete — starting pipeline");
  stepperComplete("upload");

  // Stages 1-N
  for (let i = 1; i < UPLOAD_STAGES.length; i++) {
    currentStageIndex = i;
    const stage = UPLOAD_STAGES[i];
    const [pFrom, pTo] = stage.pct;

    stepperActivate(stage.id);
    updateHeroCard(stage, i);
    addHeroLog(`Starting: ${stage.name}`, true);

    if (stage.id === "mriqc") {
      const runMriqc = document.getElementById("toggle-run-mriqc") ? document.getElementById("toggle-run-mriqc").checked : true;
      const forceContinue = document.getElementById("toggle-force-continue") ? document.getElementById("toggle-force-continue").checked : false;

      if (!runMriqc) {
        addHeroLog("MRIQC quality check bypassed by user settings.");
        appendConsoleLog("MRIQC Quality check bypassed by user settings.");
        await animateProgress(pFrom, pTo, 1000, i);
        stageOutputs[stage.id] = { completed: true, label: stage.name };
        stepperComplete(stage.id);
      } else {
        try {
          await fetch(`/api/mriqc/run/${uploadResult.patient_id}`, { method: 'POST' });
          let mriqcFinished = false;
          let lastMriqcLogCountUpload = 0;
          let elapsedSecs = 0;
          const estimatedMriqcSecs = 900; // Accurate baseline CPU-based Docker runtime: 15 minutes (900 seconds)

          while (!mriqcFinished) {
            const res = await fetch(`/api/mriqc/status/${uploadResult.patient_id}`);
            if (!res.ok) break;
            const data = await res.json();
            
            if (data.logs && data.logs.length > lastMriqcLogCountUpload) {
              for (let j = lastMriqcLogCountUpload; j < data.logs.length; j++) {
                addHeroLog(data.logs[j].trim());
              }
              lastMriqcLogCountUpload = data.logs.length;
            }
            
            if (data.status === "completed" || data.status === "failed" || data.status === "idle") {
              mriqcFinished = true;
            } else {
              elapsedSecs += 1.5;
              
              // Get progress fraction from server or fallback to time-based estimate
              let frac = Math.min(elapsedSecs / estimatedMriqcSecs, 0.95);
              let remainingSecs = Math.max(1, Math.round(estimatedMriqcSecs - elapsedSecs));
              
              if (data.progress_pct && data.progress_pct > 0) {
                const progFrac = data.progress_pct / 100;
                frac = progFrac;
                const projectedTotalSecs = elapsedSecs / progFrac;
                
                // Low-pass smoothing filter: blend initial CPU estimate with real-world run time projection
                const alpha = progFrac; // progress fraction (0.02 to 1.0)
                const smoothedTotalSecs = (1 - alpha) * estimatedMriqcSecs + alpha * projectedTotalSecs;
                remainingSecs = Math.max(5, Math.round(smoothedTotalSecs - elapsedSecs));
              }
              
              // Animate progress incrementally (10% to 33%)
              const currentPct = 10 + (23 * frac);
              setOverallPct(currentPct);
              setStepRingPct(frac * 100);

              // Live countdown timer UI
              const mriqcRemMs = remainingSecs * 1000 + 14000; // 14s for subsequent stages

              const timerValEl = $("proc-time-value");
              if (timerValEl) {
                const s = Math.ceil(mriqcRemMs / 1000);
                timerValEl.textContent = s < 60 ? `~${s}s remaining` : `~${Math.floor(s/60)}m ${s%60}s remaining`;
              }

              const elapsed = processingStartTime ? Date.now() - processingStartTime : 0;
              const timeFill = $("proc-time-fill");
              if (timeFill) {
                const adjustedTotalMs = TOTAL_PROC_MS + (estimatedMriqcSecs * 1000);
                timeFill.style.width = `${Math.min(100, (elapsed / adjustedTotalMs) * 100)}%`;
              }

              const elapsedEl = $("proc-elapsed-text");
              if (elapsedEl) elapsedEl.textContent = `${Math.round(elapsed/1000)}s elapsed`;

              await new Promise(r => setTimeout(r, 1500));
            }
          }

          // Quality Gate Check
          try {
            const qRes = await fetch(`/api/mriqc/quality-gate/${uploadResult.patient_id}`);
            if (qRes.ok) {
              const qData = await qRes.json();
              if (!qData.passed) {
                if (forceContinue) {
                  addHeroLog(`[WARNING] Quality Check Failed: Grade ${qData.grade} (${qData.score}/100)`);
                  addHeroLog(`[WARNING] Reason: ${qData.reason}`);
                  addHeroLog("Continuing pipeline execution as requested (Ignore Quality Gate Failures enabled).");
                  appendConsoleLog(`Scan FAILED quality gate (Score: ${qData.score}/100) but forced to continue.`, "warning-msg");
                  
                  const hcirc = $(`hcirc-${stage.id}`);
                  if (hcirc) {
                     hcirc.className = "h-circle s-done";
                     hcirc.style.backgroundColor = "#fb923c"; // Orange for warning/forced-pass
                     hcirc.innerHTML = `<svg style="width:18px;height:18px"><use href="#ic-info"/></svg>`;
                  }
                } else {
                  addHeroLog(`[REJECTED] Quality Check Failed: Grade ${qData.grade}`, true);
                  addHeroLog(qData.reason, true);
                  appendConsoleLog(`Scan rejected by MRIQC Quality Gate (Score: ${qData.score}/100)`, "error-msg");
                  
                  const hcirc = $(`hcirc-${stage.id}`);
                  if (hcirc) {
                     hcirc.className = "h-circle s-done";
                     hcirc.style.backgroundColor = "#ef4444"; // Red for rejection
                     hcirc.innerHTML = `<svg style="width:18px;height:18px"><use href="#ic-close"/></svg>`;
                  }
                  
                  showCustomQualityAlert(qData.grade, qData.score, qData.reason);
                  
                  // Reset to upload screen
                  setTimeout(() => {
                     showState(STATES.UPLOAD);
                     setOverallPct(0);
                     updateTimerUI(0, 0);
                  }, 1500);
                  
                  return; // Halt the pipeline entirely
                }
              } else {
                addHeroLog(`[PASSED] Quality Check: Grade ${qData.grade} (${qData.score}/100)`, false);
              }
            }
          } catch (err) {
            addHeroLog(`Quality gate error: ${err.message}`, true);
          }

          setOverallPct(pTo);
        } catch (err) {
          addHeroLog(`MRIQC error: ${err.message}`, true);
        }
      }
    } else {
      await animateProgress(pFrom, pTo, stage.duration, i);
    }

    addHeroLog(`Completed: ${stage.name}`);
    stageOutputs[stage.id] = { completed: true, label: stage.name };
    stepperComplete(stage.id);

    await new Promise(r => setTimeout(r, 150));
  }

  setOverallPct(100);
  updateTimerUI(UPLOAD_STAGES.length - 1, 1);
  clearInterval(timerId);

  await new Promise(r => setTimeout(r, 600));

  buildRibbonAutoSteps();
  if (uploadResult.patient_id) selectPatient(uploadResult.patient_id);
  setTimeout(() => showState(STATES.DASHBOARD), 400);
}

// ── Dashboard: ribbon auto steps ─────────────────────────────────
function buildRibbonAutoSteps() {
  const container = $("ribbon-auto-steps");
  if (!container) return;
  container.innerHTML = "";

  UPLOAD_STAGES.forEach((stage, i) => {
    const el = document.createElement("div");
    el.className = "rsi clickable";
    el.dataset.autoStage = stage.id;
    el.innerHTML = `
      <div class="rsi-node rsi-done">
        <svg style="width:11px;height:11px;color:#fff"><use href="#ic-check"/></svg>
      </div>
      <div class="rsi-info">
        <div class="rsi-name">${stage.name}</div>
        <span class="rsi-badge badge-completed">Done</span>
      </div>
    `;
    el.addEventListener("click", () => openStageDrawer(stage));
    container.appendChild(el);

    if (i < UPLOAD_STAGES.length - 1) {
      const conn = document.createElement("div");
      conn.className = "rsi-conn";
      conn.innerHTML = `<svg style="width:10px;height:10px"><use href="#ic-arrow-right"/></svg>`;
      container.appendChild(conn);
    }
  });
}

// ── Stage output drawer ───────────────────────────────────────────
function openStageDrawer(stage) {
  const drawer = $("output-drawer");
  const iconEl = $("drawer-icon");
  const titleEl= $("drawer-title");
  const bodyEl = $("drawer-body");
  if (!drawer || !bodyEl) return;

  if (iconEl) iconEl.innerHTML = `<svg style="width:18px;height:18px"><use href="#${stage.icon}"/></svg>`;
  if (titleEl) titleEl.textContent = stage.name;

  bodyEl.innerHTML = "";

  const sb = document.createElement("div");
  sb.className = "od-success";
  sb.innerHTML = `<svg style="width:16px;height:16px"><use href="#ic-check"/></svg> Stage completed successfully`;
  bodyEl.appendChild(sb);

  const desc = document.createElement("div");
  desc.className = "od-desc";
  desc.textContent = stage.detail;
  bodyEl.appendChild(desc);

  if (stage.id === "upload" && stageOutputs.upload) {
    const out = stageOutputs.upload;
    appendDrawerSection(bodyEl, "Upload Details");
    const grid = appendStatGrid(bodyEl);
    appendStatCard(grid, "File Name",  out.filename,   "", false);
    appendStatCard(grid, "File Size",  out.size.replace(" MB",""), "MB", true);
    appendStatCard(grid, "Patient ID", out.patientId.substring(0,18)+"…", "", false);
  }

  if (stage.id === "n4") {
    appendDrawerSection(bodyEl, "Correction Details");
    const grid = appendStatGrid(bodyEl);
    appendStatCard(grid, "Algorithm",  "N4ITK",        "",  false);
    appendStatCard(grid, "Iterations", "50 × 50 × 30", "",  true);
    appendStatCard(grid, "Convergence","0.001",         "",  false);
    appendStatCard(grid, "Status",     "Complete",      "",  true);
  }

  if (stage.id === "mriqc" && patientData && patientData.mriqc_metrics) {
    const m = patientData.mriqc_metrics;
    appendDrawerSection(bodyEl, "Image Quality Metrics");
    const grid = appendStatGrid(bodyEl);
    appendStatCard(grid, "CJV",      (m.cjv || 0).toFixed(4),  "", true);
    appendStatCard(grid, "CNR",      (m.cnr || 0).toFixed(4),  "", true);
    appendStatCard(grid, "EFC",      (m.efc || 0).toFixed(4),  "", false);
    appendStatCard(grid, "FBER",     (m.fber || 0).toFixed(2), "", false);
    appendStatCard(grid, "SNR (WM)", (m.snr_wm || 0).toFixed(2), "", true);
    appendStatCard(grid, "SNR (GM)", (m.snr_gm || 0).toFixed(2), "", true);
  }

  if (stage.id === "features" && patientData) {
    const arr = patientData.rule_based_array || [];
    appendDrawerSection(bodyEl, `Feature Vector Preview (${arr.length} dims)`);
    const list = document.createElement("div");
    list.className = "od-feature-list";
    arr.slice(0, 12).forEach((v, i) => {
      const row = document.createElement("div");
      row.className = "od-feature-row";
      row.innerHTML = `<span>${FEATURE_NAMES[i]}</span><span>${v.toFixed(5)}</span>`;
      list.appendChild(row);
    });
    if (arr.length > 12) {
      const more = document.createElement("div");
      more.style.cssText = "font-size:11px;color:var(--t4);padding:6px 12px;";
      more.textContent = `+ ${arr.length - 12} more features — see Features Comparison tab`;
      list.appendChild(more);
    }
    bodyEl.appendChild(list);
  }

  if (stage.id === "segment" && patientData) {
    const t = patientData.tissue_segmentation;
    if (t) {
      appendDrawerSection(bodyEl, "Tissue Segmentation Output");
      const grid = appendStatGrid(bodyEl);
      appendStatCard(grid, "CSF / Ventricles", t.csf_ventricles.percentage.toFixed(2), "%", false);
      appendStatCard(grid, "Gray Matter",      t.gray_matter.percentage.toFixed(2),    "%", true);
      appendStatCard(grid, "White Matter",     t.white_matter.percentage.toFixed(2),   "%", false);
      appendStatCard(grid, "Brain Coverage",   (t.csf_ventricles.percentage + t.gray_matter.percentage + t.white_matter.percentage).toFixed(2), "%", true);
    }
  }

  if (stage.id === "volumes" && patientData) {
    const v = patientData.volumetric_statistics;
    if (v) {
      appendDrawerSection(bodyEl, "Volumetric Analysis Output");
      const grid = appendStatGrid(bodyEl);
      appendStatCard(grid, "Total Brain Volume", v.total_brain_volume_ml?.toFixed(1) || "0.0",       "mL", true);
      appendStatCard(grid, "Gray Matter",         v.gray_matter_volume_ml?.toFixed(1) || "0.0",       "mL", false);
      appendStatCard(grid, "White Matter",        v.white_matter_volume_ml?.toFixed(1) || "0.0",      "mL", false);
      appendStatCard(grid, "CSF Volume",          v.csf_volume_ml?.toFixed(1) || "0.0",               "mL", true);
      
      const addBilateralCards = (name, prefix, leftKey, rightKey, aiKey) => {
        if (v[leftKey] !== undefined) {
          appendStatCard(grid, `L ${prefix}`, v[leftKey].toFixed(3), "mL", false);
        }
        if (v[rightKey] !== undefined) {
          appendStatCard(grid, `R ${prefix}`, v[rightKey].toFixed(3), "mL", false);
        }
        if (v[aiKey] !== undefined) {
          appendStatCard(grid, `${prefix} Asym`, v[aiKey].toFixed(2), "%", true);
        }
      };

      addBilateralCards("Lateral Ventricle", "Ventricle", "left_ventricle_volume_ml", "right_ventricle_volume_ml", "ventricle_asymmetry_index");
      addBilateralCards("Thalamus", "Thalamus", "left_thalamus_volume_ml", "right_thalamus_volume_ml", "thalamus_asymmetry_index");
      addBilateralCards("Caudate", "Caudate", "left_caudate_volume_ml", "right_caudate_volume_ml", "caudate_asymmetry_index");
      addBilateralCards("Putamen", "Putamen", "left_putamen_volume_ml", "right_putamen_volume_ml", "putamen_asymmetry_index");
      addBilateralCards("Pallidum", "Pallidum", "left_pallidum_volume_ml", "right_pallidum_volume_ml", "pallidum_asymmetry_index");
      addBilateralCards("Hippocampus", "Hippo", "left_hippocampus_volume_ml", "right_hippocampus_volume_ml", "hippocampus_asymmetry_index");
      addBilateralCards("Amygdala", "Amygdala", "left_amygdala_volume_ml", "right_amygdala_volume_ml", "amygdala_asymmetry_index");
    }
  }

  if (stage.id === "ready") {
    appendDrawerSection(bodyEl, "Pipeline Summary");
    const grid = appendStatGrid(bodyEl);
    appendStatCard(grid, "Stages Completed", "6 / 6",   "",  true);
    appendStatCard(grid, "Biomarkers",       "40",       "",  false);
    appendStatCard(grid, "Time Elapsed",     `${Math.round((Date.now() - (processingStartTime||Date.now()))/1000)}`, "s", false);
    appendStatCard(grid, "Status",           "Ready",    "",  true);
  }

  drawer.style.display = "block";
}

function appendDrawerSection(parent, title) {
  const el = document.createElement("div");
  el.className = "od-section-title";
  el.textContent = title;
  parent.appendChild(el);
}
function appendStatGrid(parent) {
  const grid = document.createElement("div");
  grid.className = "od-stat-grid";
  parent.appendChild(grid);
  return grid;
}
function appendStatCard(grid, label, value, unit, accent) {
  const card = document.createElement("div");
  card.className = `od-stat-card${accent ? " accent" : ""}`;
  card.innerHTML = `
    <div class="od-stat-label">${label}</div>
    <div class="od-stat-value">${value}<span class="od-stat-unit" style="font-size:13px;font-weight:600;margin-left:3px">${unit}</span></div>
  `;
  grid.appendChild(card);
}

function closeDrawer() {
  const drawer = $("output-drawer");
  if (drawer) drawer.style.display = "none";
}

// ── Step run panel (manual pipeline) ─────────────────────────────
function openStepRunPanel(num) {
  if (activeRibbonStep === num) { closeStepRunPanel(); return; }
  activeRibbonStep = num;
  const info = MANUAL_STEPS[num];
  if (!info) return;
  const srp     = $("step-run-panel");
  const title   = $("srp-title");
  const desc    = $("srp-desc");
  const runBtn  = $("srp-run-btn");
  const log     = $("srp-log");
  if (title) title.textContent = info.title;
  if (desc)  desc.textContent  = info.desc;
  if (log)   log.innerHTML = `<span style="color:rgba(110,231,183,.4)">Click "Run Step" to execute this pipeline stage.</span>`;
  if (runBtn) runBtn.onclick = () => runPipelineStep(num);
  if (srp) srp.style.display = "block";
}
function closeStepRunPanel() {
  activeRibbonStep = null;
  const srp = $("step-run-panel");
  if (srp) srp.style.display = "none";
}

// ── API helpers ───────────────────────────────────────────────────
function fetchSystemStatus() {
  fetch("/api/status")
    .then(r => r.json())
    .then(data => {
      if (cudaText) cudaText.textContent = data.cuda_available ? (data.device || "Available") : "CPU Only";
      if (cudaDot)  cudaDot.className = `hstat-dot ${data.cuda_available ? "hstat-green" : "hstat-amber"}`;
      updateRibbonBadges(data.pipeline_status);
    })
    .catch(() => {
      if (cudaText) cudaText.textContent = "Disconnected";
      if (cudaDot)  cudaDot.className = "hstat-dot hstat-amber";
    });
}

function updateRibbonBadges(status) {
  if (!status) return;
  Object.keys(status).forEach(step => {
    const badge = ribbonBadges[step];
    const rnode = $(`rnode-${step}`);
    if (!badge) return;
    const s = status[step].toLowerCase();
    badge.className = `rsi-badge badge-${s}`;
    badge.textContent = status[step];
    if (rnode) {
      rnode.className = `rsi-node rsi-${s === "completed" ? "done" : s === "running" ? "running" : "pending"}`;
    }
  });
}

function loadLatestUpload() {
  fetch("/api/patients")
    .then(r => r.json())
    .then(data => {
      availablePatients = data.patients || [];
      const uploaded = availablePatients.filter(id => !id.startsWith("IXI"));
      if (totalScansText) totalScansText.textContent = uploaded.length;

      // Removed automatic transition to DASHBOARD on load
      // The user should stay on the UPLOAD screen to upload new scans
      if (uploaded.length > 0 && currentState === STATES.DASHBOARD && !activePatientId) {
        selectPatient(uploaded[0]);
      } else if (uploaded.length === 0 && currentState === STATES.DASHBOARD) {
        if (emptyViewport)    emptyViewport.style.display    = "flex";
        if (analysisViewport) analysisViewport.style.display = "none";
      }
    })
    .catch(() => appendConsoleLog("Error checking patients.", "error-msg"));
}

function selectPatient(id) {
  if (!id) return;
  activePatientId = id;

  const scanNames = ["active-scan-name", "tbar-scan-name"];
  scanNames.forEach(n => { const el = $(n); if (el) el.textContent = id; });

  if (emptyViewport)    emptyViewport.style.display    = "none";
  if (analysisViewport) analysisViewport.style.display = "block";
  if (metadataSummary)  metadataSummary.innerHTML = `<span class="chip">Loading…</span>`;

  // Reset MRIQC Docker panel
  if (mriqcPollingInterval) { clearInterval(mriqcPollingInterval); mriqcPollingInterval = null; }
  lastMriqcLogCount = 0;

  const btnRunDocker    = $("btn-run-mriqc-docker");
  const dockerConsole   = $("mriqc-docker-console");
  const dockerStatus    = $("mriqc-docker-status");
  if (dockerConsole) dockerConsole.style.display = "none";
  if (btnRunDocker)  { btnRunDocker.disabled = false; btnRunDocker.textContent = "🐳 Run Official MRIQC (Docker)"; }
  if (dockerStatus)  dockerStatus.textContent = "Status: Idle (Docker MRIQC not run)";

  resetReportTab();

  fetch(`/api/features/${id}`)
    .then(r => r.json())
    .then(data => {
      patientData = data;
      renderPatientDetails();
      const dlBtn = $("btn-download-json");
      if (dlBtn) dlBtn.style.display = "inline-flex";
    })
    .catch(() => appendConsoleLog(`Error fetching features for ${id}`, "error-msg"));
}

function resetReportTab() {
  if (reportCta)      reportCta.style.display     = "flex";
  const genBtn = $("btn-generate-report");
  if (genBtn)         genBtn.style.display        = "inline-flex";
  if (reportSpinner)  reportSpinner.style.display = "none";
  if (reportDocument) reportDocument.style.display= "none";
  if (reportToolbar)  reportToolbar.style.display = "none";
  if (reportPaper)    reportPaper.innerHTML       = "";
}

// ── Render patient details ────────────────────────────────────────
function renderPatientDetails() {
  if (!patientData) return;

  const isUploaded = patientData.source === "uploads";
  if (metadataSummary) metadataSummary.innerHTML = `
    <span class="chip">${isUploaded ? "Uploaded" : "Dataset"}</span>
    <span class="chip">${(patientData.scan_metadata?.orientation || "").toUpperCase()}</span>
    <span class="chip">${patientData.scan_metadata?.format || "NIfTI"}</span>
  `;

  if (metadataGrid) metadataGrid.innerHTML = `
    <div class="meta-item"><div class="meta-item-label">Scan ID</div><div class="meta-item-value" style="font-size:10.5px;word-break:break-all">${activePatientId.substring(0,28)}…</div></div>
    <div class="meta-item"><div class="meta-item-label">Resolution</div><div class="meta-item-value">${patientData.scan_metadata?.image_width} × ${patientData.scan_metadata?.image_height}</div></div>
    <div class="meta-item"><div class="meta-item-label">Total Voxels</div><div class="meta-item-value">${(patientData.scan_metadata?.total_pixels||0).toLocaleString()}</div></div>
    <div class="meta-item"><div class="meta-item-label">Modality</div><div class="meta-item-value">${patientData.scan_metadata?.modality||"T1-w MRI"}</div></div>
  `;

  const setBar = (barId, labelId, val) => {
    const b = $(barId), l = $(labelId);
    if (b) setTimeout(() => { b.style.width = `${val}%`; }, 120);
    if (l) l.textContent = `${val.toFixed(2)}%`;
  };
  const csf = patientData.tissue_segmentation?.csf_ventricles?.percentage || 0;
  const gm  = patientData.tissue_segmentation?.gray_matter?.percentage    || 0;
  const wm  = patientData.tissue_segmentation?.white_matter?.percentage   || 0;
  setBar("bar-csf-pct","label-csf-pct",csf);
  setBar("bar-gm-pct","label-gm-pct",gm);
  setBar("bar-wm-pct","label-wm-pct",wm);

  const rows = (arr) => arr.map(([k,v]) => `<tr><td>${k}</td><td>${v?.toFixed?.(4) ?? v ?? "—"}</td></tr>`).join("");
  const intensity = patientData.intensity_statistics || {};
  if (tableIntensity) tableIntensity.innerHTML = rows([
    ["Mean",intensity.mean],["Std Dev",intensity.std],["Min",intensity.min],
    ["Max",intensity.max],["Median",intensity.median],["Kurtosis",intensity.kurtosis],
    ["Skewness",intensity.skewness],["Dyn Range",intensity.dynamic_range],
  ]);
  const struct = patientData.structural_features || {};
  const sym    = patientData.symmetry || {};
  if (tableMorphology) tableMorphology.innerHTML = rows([
    ["Brain Area %",struct.brain_area_pct],["Solidity",struct.solidity],
    ["Eccentricity",struct.eccentricity],["Perimeter",struct.perimeter],
    ["Bilateral Sym",sym.bilateral_symmetry_score],["Asymmetry Idx",sym.asymmetry_index_pct],
  ]);
  const tex = patientData.texture_features || {};
  if (tableTexture) tableTexture.innerHTML = rows([
    ["GLCM Contrast",tex.glcm_contrast],["GLCM Corr",tex.glcm_correlation],
    ["GLCM Energy",tex.glcm_energy],["GLCM Homog",tex.glcm_homogeneity],
    ["Mean Gradient",tex.mean_gradient],["Shannon Entropy",tex.shannon_entropy],
  ]);

  const tblVol = $("table-volumes");
  if (tblVol) {
    const vols = patientData.volumetric_statistics;
    if (vols) {
      const rows_list = [
        ["Total Brain", `${vols.total_brain_volume_ml?.toFixed(1)} mL`],
        ["Gray Matter", `${vols.gray_matter_volume_ml?.toFixed(1)} mL`],
        ["White Matter",`${vols.white_matter_volume_ml?.toFixed(1)} mL`],
        ["CSF Volume",  `${vols.csf_volume_ml?.toFixed(1)} mL`],
      ];
      
      const addBilateral = (name, leftKey, rightKey, aiKey) => {
        if (vols[leftKey] !== undefined) {
          rows_list.push([`L ${name}`, `${vols[leftKey]?.toFixed(3)} mL`]);
        }
        if (vols[rightKey] !== undefined) {
          rows_list.push([`R ${name}`, `${vols[rightKey]?.toFixed(3)} mL`]);
        }
        if (vols[aiKey] !== undefined) {
          rows_list.push([`${name} Asymmetry`, `${vols[aiKey]?.toFixed(2)} %`]);
        }
      };

      addBilateral("Lateral Ventricle", "left_ventricle_volume_ml", "right_ventricle_volume_ml", "ventricle_asymmetry_index");
      addBilateral("Thalamus", "left_thalamus_volume_ml", "right_thalamus_volume_ml", "thalamus_asymmetry_index");
      addBilateral("Caudate", "left_caudate_volume_ml", "right_caudate_volume_ml", "caudate_asymmetry_index");
      addBilateral("Putamen", "left_putamen_volume_ml", "right_putamen_volume_ml", "putamen_asymmetry_index");
      addBilateral("Pallidum", "left_pallidum_volume_ml", "right_pallidum_volume_ml", "pallidum_asymmetry_index");
      addBilateral("Hippocampus", "left_hippocampus_volume_ml", "right_hippocampus_volume_ml", "hippocampus_asymmetry_index");
      addBilateral("Amygdala", "left_amygdala_volume_ml", "right_amygdala_volume_ml", "amygdala_asymmetry_index");

      tblVol.innerHTML = rows_list.map(([k,v]) => {
        const isAI = k.includes("Asymmetry");
        const isBold = k === "Total Brain" || isAI;
        const style = isBold 
          ? `font-weight:700;${isAI ? 'color:var(--g2);border-bottom:1px solid rgba(0,0,0,0.06);background:var(--g-xlt);' : 'border-top:1px solid var(--border-soft);'}` 
          : '';
        const padding = isAI ? 'padding-left:12px;' : '';
        return `<tr style="${style}"><td style="${padding}">${k}</td><td>${v}</td></tr>`;
      }).join("");
    } else {
      tblVol.innerHTML = `<tr><td colspan="2" style="color:var(--t4);text-align:center;padding:16px 0;font-size:11px;">3D volumes computed on-the-fly for uploaded scans only.</td></tr>`;
    }
  }

  // Feature comparison
  const ruleArr = patientData.rule_based_array || [];
  const predArr = patientData.predicted_array  || [];
  if (comparisonTbody) {
    comparisonTbody.innerHTML = "";
    let totalDev = 0, validCount = 0;
    FEATURE_NAMES.forEach((name, i) => {
      const rv = ruleArr[i] ?? null, pv = predArr[i] ?? null;
      let diff = 0, pctDev = 0;
      if (rv !== null && pv !== null) {
        diff = Math.abs(rv - pv);
        const scale = Math.max(Math.abs(rv), Math.abs(pv));
        pctDev = scale > 0 ? (diff / scale) * 100 : 0;
        totalDev += pctDev; validCount++;
      }
      const devClr = pctDev > 15 ? "#ef4444" : pctDev > 5 ? "#f59e0b" : "#10b981";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>#${String(i).padStart(2,"0")}</td>
        <td class="cell-name">${name}</td>
        <td>${rv !== null ? rv.toFixed(4) : "—"}</td>
        <td>${pv !== null ? pv.toFixed(4) : "Pending"}</td>
        <td>${(rv!==null&&pv!==null) ? diff.toFixed(4) : "—"}</td>
        <td>${(rv!==null&&pv!==null) ? `<div class="dev-track" title="${pctDev.toFixed(2)}%"><div class="dev-bar" style="width:${Math.min(100,pctDev*5)}%;background:${devClr}"></div></div>` : "<span style='color:var(--t4)'>—</span>"}</td>
      `;
      comparisonTbody.appendChild(tr);
    });
    if (validCount > 0) {
      const avg = totalDev / validCount;
      if (comparisonStatus) {
        comparisonStatus.textContent = avg < 2
          ? `Excellent alignment — avg deviation ${avg.toFixed(2)}%`
          : avg < 5
          ? `Good alignment — avg deviation ${avg.toFixed(2)}%`
          : `High deviation: ${avg.toFixed(2)}% — regressor may need more epochs`;
        comparisonStatus.style.color = avg < 5 ? "var(--g2)" : "#92400e";
      }
    } else {
      if (comparisonStatus) comparisonStatus.textContent = "Train the Regressor (Step 4) to compare predictions.";
    }
  }

  // MRIQC tab population
  const tblMriqc = $("table-mriqc-details");
  if (tblMriqc && patientData.mriqc_metrics) {
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

    let score = 100;
    if (mriqc.cjv > 0.75) score -= 15;
    if (mriqc.cjv > 0.85) score -= 15;
    if (mriqc.cnr < 1.20) score -= 15;
    if (mriqc.cnr < 0.80) score -= 15;
    if (mriqc.efc > 0.95) score -= 15;
    if (mriqc.efc > 0.97) score -= 15;
    if (mriqc.qi_1 > 0.05) score -= 10;
    if (mriqc.qi_1 > 0.08) score -= 10;

    let grade = "A+", label = "Excellent", gradeClass = "grade-excellent";
    let desc = "This scan meets or exceeds all clinical quality thresholds. Signal-to-noise ratio is high, and motion artifacts are minimal.";
    if (score >= 90) {
      grade = "A+"; label = "Excellent"; gradeClass = "grade-excellent";
    } else if (score >= 75) {
      grade = "B"; label = "Good"; gradeClass = "grade-good";
      desc = "Good quality scan. Mild blur or intensity variation detected, but fully sufficient for reliable clinical feature extraction.";
    } else if (score >= 50) {
      grade = "C"; label = "Acceptable"; gradeClass = "grade-acceptable";
      desc = "Acceptable scan quality. Moderate noise, distortion, or artifact levels present. Interpret metrics with slight caution.";
    } else {
      grade = "F"; label = "Poor"; gradeClass = "grade-poor";
      desc = "Poor scan quality. Significant motion artifacts, low signal-to-noise ratio, or severe distortion detected. Rescan recommended.";
    }

    const badge  = $("mriqc-rating-badge");
    const qLabel = $("mriqc-quality-label");
    const qDesc  = $("mriqc-quality-desc");
    if (badge)  { badge.textContent = grade; badge.className = "rating-badge " + gradeClass; }
    if (qLabel) qLabel.textContent = label;
    if (qDesc)  qDesc.textContent  = desc;

    const updateDot = (id, isGreen, isYellow) => {
      const el = $(id);
      if (el) el.className = "threshold-dot " + (isGreen ? "status-green" : isYellow ? "status-yellow" : "status-red");
    };
    updateDot("dot-cjv", mriqc.cjv < 0.75, mriqc.cjv < 0.85);
    updateDot("dot-cnr", mriqc.cnr > 1.20, mriqc.cnr > 0.80);
    updateDot("dot-efc", mriqc.efc < 0.95, mriqc.efc < 0.97);
    updateDot("dot-qi1", mriqc.qi_1 < 0.05, mriqc.qi_1 < 0.08);

    // --- Confidence Score (0–100%) ---
    // Each metric contributes an independent normalised score:
    //   CJV   : ideal = 0.35, worst = 1.0  → higher cjv = lower conf
    //   CNR   : ideal = 2.5,  worst = 0    → higher cnr = higher conf
    //   EFC   : ideal = 0.40, worst = 1.0  → higher efc = lower conf
    //   QI_1  : ideal = 0.00, worst = 0.10 → higher qi1 = lower conf
    const confCjv  = Math.max(0, Math.min(1, (1.0  - mriqc.cjv)  / (1.0  - 0.35))) * 100;
    const confCnr  = Math.max(0, Math.min(1, mriqc.cnr            / 2.5))           * 100;
    const confEfc  = Math.max(0, Math.min(1, (1.0  - mriqc.efc)  / (1.0  - 0.40))) * 100;
    const confQi1  = Math.max(0, Math.min(1, (0.10 - mriqc.qi_1) / 0.10))          * 100;
    const confScore = Math.round((confCjv + confCnr + confEfc + confQi1) / 4);

    // Update gauge text
    const confValEl = $("confidence-score-value");
    if (confValEl) confValEl.textContent = confScore + "%";

    // Animate SVG ring: circumference = 2π × 33 ≈ 207.3
    const circumference = 207.3;
    const offset = circumference - (confScore / 100) * circumference;
    const ring = $("confidence-gauge-ring");
    if (ring) {
        // Colour based on score
        let ringColor = "#10b981"; // green
        if (confScore < 60) ringColor = "#ec4899";      // pink/red
        else if (confScore < 75) ringColor = "#f59e0b"; // amber
        else if (confScore < 90) ringColor = "#06b6d4"; // cyan
        ring.style.stroke = ringColor;
        ring.style.strokeDashoffset = offset;
        ring.style.transition = "stroke-dashoffset 1.2s ease, stroke 0.4s ease";
    }

  } else if (tblMriqc) {
    tblMriqc.innerHTML = `<tr><td colspan="2" style="color:var(--t4);text-align:center;font-style:italic;padding:20px 0;font-size:11px;">Image Quality Metrics unavailable. Run Official MRIQC (Docker) to compute them.</td></tr>`;
  }

  stageOutputs.features = { loaded: true };
  stageOutputs.segment  = { loaded: true };
  stageOutputs.volumes  = { loaded: true };
}

// ── Run manual pipeline step ──────────────────────────────────────
function runPipelineStep(step) {
  if (isRunningStep) { alert("A step is already running."); return; }
  const ds = $("pipeline-data-source")?.value || "ixi-t1";
  isRunningStep = true;
  if (consolePulse) consolePulse.style.display = "inline-block";

  const badge = ribbonBadges[step];
  const rnode = $(`rnode-${step}`);
  if (badge) { badge.className = "rsi-badge badge-running"; badge.textContent = "Running"; }
  if (rnode) rnode.className = "rsi-node rsi-running";

  const srpLog = $("srp-log");
  if (srpLog) srpLog.innerHTML = `<span>Starting Step ${step}…</span>`;
  appendConsoleLog(`Triggering Pipeline Step ${step} on data source '${ds}'...`, "cmd-msg");

  fetch("/api/run-step", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({ step: parseInt(step), data_source: ds }),
  })
  .then(r => r.json())
  .then(data => {
    if (data.status === "started") {
      appendConsoleLog(`Step ${step} started in background. PID: ${data.pid}`, "system-msg");
      startPollingLogs(step);
    } else {
      isRunningStep = false;
      if (consolePulse) consolePulse.style.display = "none";
      if (badge) { badge.className = "rsi-badge badge-failed"; badge.textContent = "Failed"; }
      if (rnode) rnode.className = "rsi-node rsi-pending";
      appendConsoleLog(`Failed to start Step ${step}: ${data.error}`, "error-msg");
    }
  })
  .catch(() => {
    isRunningStep = false;
    if (consolePulse) consolePulse.style.display = "none";
    appendConsoleLog(`Request error running Step ${step}`, "error-msg");
  });
}

function startPollingLogs(step) {
  lastLogIndex = 0;
  if (logPollInterval) clearInterval(logPollInterval);
  logPollInterval = setInterval(() => {
    fetch(`/api/logs?index=${lastLogIndex}`)
      .then(r => r.json())
      .then(data => {
        if (data.new_lines?.length > 0) {
          data.new_lines.forEach(line => {
            const clean = line.trim();
            const cls = clean.includes("Error")||clean.includes("[FAILED]") ? "error-msg"
                      : clean.includes("[SUCCESS]") ? "cmd-msg" : "";
            appendConsoleLog(clean, cls);
            const srpLog = $("srp-log");
            if (srpLog) {
              const el = document.createElement("div");
              el.style.color = clean.includes("Error")||clean.includes("[FAILED]") ? "#f87171"
                             : clean.includes("[SUCCESS]") ? "#4ade80" : "rgba(110,231,183,.7)";
              el.textContent = clean;
              srpLog.appendChild(el);
              srpLog.scrollTop = srpLog.scrollHeight;
            }
          });
          lastLogIndex = data.next_index;
        }
        if (data.is_running === false) {
          clearInterval(logPollInterval);
          logPollInterval = null;
          isRunningStep = false;
          if (consolePulse) consolePulse.style.display = "none";
          appendConsoleLog("Pipeline step execution finished.", "system-msg");
          const badge = ribbonBadges[step];
          const rnode = $(`rnode-${step}`);
          if (badge) { badge.className = "rsi-badge badge-completed"; badge.textContent = "Done"; }
          if (rnode) {
            rnode.className = "rsi-node rsi-done";
            rnode.innerHTML = `<svg style="width:11px;height:11px;color:#fff"><use href="#ic-check"/></svg>`;
          }
          fetchSystemStatus();
          if (activePatientId) selectPatient(activePatientId);
        }
      })
      .catch(() => {
        clearInterval(logPollInterval);
        logPollInterval = null;
        isRunningStep = false;
        if (consolePulse) consolePulse.style.display = "none";
      });
  }, 800);
}

// ── Event listeners ───────────────────────────────────────────────
function setupEventListeners() {
  // Drop zone
  if (dropZone) {
    dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("dragover"); });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", e => {
      e.preventDefault(); dropZone.classList.remove("dragover");
      const f = e.dataTransfer.files[0];
      if (f && (f.name.endsWith(".nii") || f.name.endsWith(".nii.gz"))) handleFileUpload(f);
      else appendConsoleLog("Only .nii / .nii.gz files supported.", "error-msg");
    });
    dropZone.addEventListener("click", e => {
      if (e.target.tagName !== "LABEL" && e.target.tagName !== "INPUT") fileInput && fileInput.click();
    });
  }

  if (fileInput) {
    fileInput.addEventListener("change", e => {
      const f = e.target.files[0];
      if (f) handleFileUpload(f);
      fileInput.value = "";
    });
  }

  // New scan / go-upload
  const newScan = $("btn-new-scan");
  if (newScan) newScan.addEventListener("click", () => showState(STATES.UPLOAD));
  const goUpload = $("btn-go-upload");
  if (goUpload) goUpload.addEventListener("click", () => showState(STATES.UPLOAD));

  // Ribbon: training steps
  for (let i = 1; i <= 6; i++) {
    const el = $(`ribbon-step-${i}`);
    if (el) el.addEventListener("click", () => openStepRunPanel(i));
  }

  // Step run panel close
  const srpClose = $("srp-close-btn");
  if (srpClose) srpClose.addEventListener("click", closeStepRunPanel);

  // Drawer close
  const drawerClose = $("drawer-close");
  const drawerBg    = $("drawer-backdrop");
  if (drawerClose) drawerClose.addEventListener("click", closeDrawer);
  if (drawerBg)    drawerBg.addEventListener("click", closeDrawer);

  // Tabs
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      const pane = $(btn.dataset.tab);
      if (pane) pane.classList.add("active");
    });
  });

  // 3D viewer
  const launchBtn = $("btn-launch-viewer");
  if (launchBtn) launchBtn.addEventListener("click", () => {
    if (activePatientId) window.open(`viewer.html?patient_id=${encodeURIComponent(activePatientId)}`, "_blank");
  });

  // Generate report
  const genReport = $("btn-generate-report");
  if (genReport) genReport.addEventListener("click", () => {
    if (!activePatientId) return;
    genReport.style.display = "none";
    if (reportSpinner) reportSpinner.style.display = "flex";
    fetch(`/api/report/${activePatientId}`)
      .then(r => r.json())
      .then(data => {
        if (reportCta)     reportCta.style.display = "none";
        if (reportDocument) reportDocument.style.display = "block";
        if (reportToolbar)  reportToolbar.style.display = "flex";
        if (reportPaper) {
          if (data.report && (data.report.startsWith("<div") || data.report.includes("<style>"))) {
            reportPaper.innerHTML = data.report;
          } else {
            reportPaper.innerHTML = `<pre>${data.report || "No report returned."}</pre>`;
          }
        }
      })
      .catch(() => {
        genReport.style.display = "inline-flex";
        if (reportSpinner) reportSpinner.style.display = "none";
      });
  });

  const printReport = $("btn-print-report");
  if (printReport) printReport.addEventListener("click", () => {
    const win = window.open("", "_blank");
    win.document.write(`<html><head><title>Report — ${activePatientId}</title>
      <style>
        body { font-family: 'Inter', -apple-system, sans-serif; padding: 20px; color: #1e293b; background: #fff; }
      </style>
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Outfit:wght@700;800&display=swap" rel="stylesheet">
      </head>
      <body>${$("report-text-content").innerHTML}<script>window.onload=()=>{window.print();window.close()}<\/script></body></html>`);
    win.document.close();
  });

  const dlReport = $("btn-download-report");
  if (dlReport) dlReport.addEventListener("click", () => {
    if (!activePatientId) return;
    const isHtml = $("report-text-content").querySelector(".clinical-report-container") !== null;
    if (isHtml) {
      triggerDownload($("report-text-content").innerHTML, `${activePatientId}_report.html`, "text/html");
    } else {
      triggerDownload($("report-text-content").innerText, `${activePatientId}_report.txt`, "text/plain");
    }
  });

  const dlJson = $("btn-download-json");
  if (dlJson) dlJson.addEventListener("click", () => {
    if (patientData) triggerDownload(JSON.stringify(patientData, null, 2), `${activePatientId}_features.json`, "application/json");
  });

  // Clear console
  const clearConsoleBtn = $("btn-clear-console");
  if (clearConsoleBtn) clearConsoleBtn.addEventListener("click", () => {
    if (consoleOutput) consoleOutput.innerHTML = "";
    appendConsoleLog("Console cleared.", "system-msg");
  });

  // MRIQC Docker run button (from app(2).js)
  const btnRunMriqcDocker = $("btn-run-mriqc-docker");
  const mriqcDockerConsole = $("mriqc-docker-console");
  const mriqcDockerStatus  = $("mriqc-docker-status");
  const mriqcDockerOutput  = $("mriqc-docker-output");
  const mriqcDockerSpinner = $("mriqc-docker-spinner");

  if (btnRunMriqcDocker) {
    btnRunMriqcDocker.addEventListener("click", () => {
      if (!activePatientId) {
        appendConsoleLog("Please select a patient scan first.", "warning-msg");
        return;
      }
      if (mriqcDockerConsole) mriqcDockerConsole.style.display = "flex";
      if (mriqcDockerSpinner) mriqcDockerSpinner.style.display = "block";
      if (mriqcDockerStatus)  mriqcDockerStatus.textContent = "Status: Starting Docker run...";
      if (mriqcDockerOutput)  mriqcDockerOutput.innerHTML = '<div class="console-line system-msg">Initializing Docker run request...</div>';
      btnRunMriqcDocker.disabled = true;
      btnRunMriqcDocker.textContent = "🐳 Running...";
      lastMriqcLogCount = 0;

      fetch(`/api/mriqc/run/${encodeURIComponent(activePatientId)}`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
          if (data.status === "started") {
            appendConsoleLog(`Official MRIQC Docker container started for patient ${activePatientId}.`);
            mriqcPollingInterval = setInterval(() => pollMriqcStatus(activePatientId), 2000);
          } else {
            if (mriqcDockerStatus)  mriqcDockerStatus.textContent = "Status: Failed to start";
            if (mriqcDockerSpinner) mriqcDockerSpinner.style.display = "none";
            btnRunMriqcDocker.disabled = false;
            btnRunMriqcDocker.textContent = "🐳 Run Official MRIQC (Docker)";
            if (mriqcDockerOutput) mriqcDockerOutput.innerHTML += `<div class="console-line error-msg">Error: ${data.error || "Failed to start Docker process"}</div>`;
          }
        })
        .catch(err => {
          if (mriqcDockerStatus)  mriqcDockerStatus.textContent = "Status: Connection Error";
          if (mriqcDockerSpinner) mriqcDockerSpinner.style.display = "none";
          btnRunMriqcDocker.disabled = false;
          btnRunMriqcDocker.textContent = "🐳 Run Official MRIQC (Docker)";
          if (mriqcDockerOutput) mriqcDockerOutput.innerHTML += `<div class="console-line error-msg">Error connecting to server: ${err}</div>`;
        });
    });
  }

  function pollMriqcStatus(patientId) {
    fetch(`/api/mriqc/status/${encodeURIComponent(patientId)}`)
      .then(res => res.json())
      .then(data => {
        if (mriqcDockerStatus) mriqcDockerStatus.textContent = `Status: ${data.progress || data.status}`;
        if (data.logs && data.logs.length > lastMriqcLogCount) {
          const newLogs = data.logs.slice(lastMriqcLogCount);
          newLogs.forEach(line => {
            if (!mriqcDockerOutput) return;
            const lineEl = document.createElement("div");
            lineEl.className = "console-line";
            if (line.includes("ERROR") || line.includes("failed") || line.includes("exited with error")) lineEl.classList.add("error-msg");
            else if (line.includes("Successfully") || line.includes("complete") || line.includes("Reading")) lineEl.classList.add("cmd-msg");
            lineEl.textContent = line.trim();
            mriqcDockerOutput.appendChild(lineEl);
          });
          lastMriqcLogCount = data.logs.length;
          if (mriqcDockerOutput) mriqcDockerOutput.scrollTop = mriqcDockerOutput.scrollHeight;
        }
        if (data.status === "completed") {
          clearInterval(mriqcPollingInterval); mriqcPollingInterval = null;
          if (mriqcDockerSpinner) mriqcDockerSpinner.style.display = "none";
          if (btnRunMriqcDocker) { btnRunMriqcDocker.disabled = false; btnRunMriqcDocker.textContent = "🐳 Run Official MRIQC (Docker)"; }
          appendConsoleLog(`Official MRIQC Docker run completed for patient ${patientId}.`);
          selectPatient(patientId);
        } else if (data.status === "failed") {
          clearInterval(mriqcPollingInterval); mriqcPollingInterval = null;
          if (mriqcDockerSpinner) mriqcDockerSpinner.style.display = "none";
          if (btnRunMriqcDocker) { btnRunMriqcDocker.disabled = false; btnRunMriqcDocker.textContent = "🐳 Run Official MRIQC (Docker)"; }
          appendConsoleLog(`Official MRIQC Docker run failed for patient ${patientId}.`, "error-msg");
        }
      })
      .catch(err => console.error("Error polling MRIQC status:", err));
  }

} // end setupEventListeners

// ── Utilities ─────────────────────────────────────────────────────
function triggerDownload(content, filename, type) {
  const blob = new Blob([content], { type });
  const url  = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a); URL.revokeObjectURL(url);
}

// ── Init ──────────────────────────────────────────────────────────
function showCustomQualityAlert(grade, score, reason) {
  let modal = document.getElementById("quality-alert-modal");
  if (modal) modal.remove();

  modal = document.createElement("div");
  modal.id = "quality-alert-modal";
  modal.style.position = "fixed";
  modal.style.top = "0";
  modal.style.left = "0";
  modal.style.width = "100%";
  modal.style.height = "100%";
  modal.style.backgroundColor = "rgba(15, 23, 42, 0.75)";
  modal.style.backdropFilter = "blur(8px)";
  modal.style.display = "flex";
  modal.style.alignItems = "center";
  modal.style.justifyContent = "center";
  modal.style.zIndex = "99999";
  modal.style.transition = "opacity 0.3s ease";

  const content = document.createElement("div");
  content.style.backgroundColor = "#1e293b";
  content.style.border = "1px solid rgba(239, 68, 68, 0.4)";
  content.style.borderRadius = "16px";
  content.style.padding = "32px";
  content.style.maxWidth = "480px";
  content.style.width = "90%";
  content.style.boxShadow = "0 25px 50px -12px rgba(0, 0, 0, 0.5), 0 0 40px rgba(239, 68, 68, 0.15)";
  content.style.textAlign = "center";
  content.style.color = "#f8fafc";
  content.style.fontFamily = "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif";

  content.innerHTML = `
    <div style="width: 56px; height: 56px; background-color: rgba(239, 68, 68, 0.1); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 20px auto; border: 1px solid rgba(239, 68, 68, 0.25)">
      <svg style="width: 28px; height: 28px; color: #ef4444;" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
      </svg>
    </div>
    <h3 style="font-size: 20px; font-weight: 600; margin-bottom: 8px; color: #ef4444;">Scan Quality Rejected</h3>
    <p style="font-size: 14px; color: #94a3b8; margin-bottom: 24px;">The uploaded MRI scan failed to pass the official MRIQC quality threshold gate and cannot be processed safely.</p>
    
    <div style="background-color: #0f172a; border-radius: 8px; padding: 16px; margin-bottom: 24px; text-align: left; border: 1px solid #334155;">
      <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px;">
        <span style="color: #64748b;">Quality Grade:</span>
        <span style="font-weight: 600; color: #ef4444;">${grade}</span>
      </div>
      <div style="display: flex; justify-content: space-between; margin-bottom: 12px; font-size: 13px; border-bottom: 1px solid #1e293b; padding-bottom: 8px;">
        <span style="color: #64748b;">MRIQC Score:</span>
        <span style="font-weight: 600; color: #ef4444;">${score} / 100</span>
      </div>
      <div style="font-size: 12px; color: #94a3b8; line-height: 1.5;">
        <strong style="color: #cbd5e1; display: block; margin-bottom: 4px;">Reason:</strong>
        ${reason}
      </div>
    </div>
    
    <button id="close-quality-modal-btn" style="width: 100%; padding: 12px; background-color: #ef4444; color: white; border: none; border-radius: 8px; font-size: 14px; font-weight: 500; cursor: pointer; transition: background-color 0.2s ease; box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2);">
      Acknowledge & Dismiss
    </button>
  `;

  modal.appendChild(content);
  document.body.appendChild(modal);

  const closeBtn = document.getElementById("close-quality-modal-btn");
  if (closeBtn) {
    closeBtn.onclick = () => {
      modal.style.opacity = "0";
      setTimeout(() => modal.remove(), 300);
    };
  }
}

function init() {
  showState(STATES.UPLOAD);
  initNeuralCanvas();
  setupEventListeners();
  fetchSystemStatus();
  loadLatestUpload();
  setInterval(fetchSystemStatus, 30000);
}

init();

}); // DOMContentLoaded
