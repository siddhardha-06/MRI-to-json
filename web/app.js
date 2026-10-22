// NeuroSphere dashboard (v2 Phase 1). Shows exactly what the server's JSON contains:
// measured values, and null values together with their reason. No client-side values,
// progress animations or interpretations are generated here.
"use strict";

const $ = (id) => document.getElementById(id);
let subjects = [];
let current = null;
let mriqcTimer = null;

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function getJSON(url, opts) {
  const res = await fetch(url, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `${res.status} ${res.statusText}`);
  return body;
}

function fmt(v) {
  if (Array.isArray(v)) return v.join(" × ");
  if (typeof v === "object" && v !== null) return JSON.stringify(v);
  return String(v);
}

// One table row per key; a null value is shown as "not computed" with its <key>_reason.
function rowsFor(obj) {
  return Object.keys(obj)
    .filter((k) => !k.endsWith("_reason"))
    .map((k) => {
      const v = obj[k];
      const cell = v === null
        ? `<span class="null">not computed</span> <span class="muted">${esc(obj[k + "_reason"] || "")}</span>`
        : esc(fmt(v));
      return `<tr><th>${esc(k)}</th><td>${cell}</td></tr>`;
    }).join("");
}

function renderSummary(doc) {
  const sections = [["Subject", doc.subject], ["Acquisition (NIfTI header)", doc.acquisition]];
  let html = sections.map(([t, o]) => `<h2>${esc(t)}</h2><table>${rowsFor(o)}</table>`).join("");
  const top = {};
  for (const k of ["segmentation", "volumes_ml", "etiv_ml", "normalized_volumes", "asymmetry_index_pct", "intensity", "normative"]) {
    if (k in doc) { top[k] = doc[k]; top[k + "_reason"] = doc[k + "_reason"]; }
  }
  html += `<h2>Morphometry, intensity and normative values</h2><table>${rowsFor(top)}</table>`;
  html += `<h2>Limitations</h2><ul>${(doc.limitations || []).map((l) => `<li>${esc(l)}</li>`).join("")}</ul>`;
  $("panel-summary").innerHTML = html;
}

function renderQuality(q) {
  const gate = q.gate || {};
  let html = `<h2>Quality gate</h2><table><tr><th>status</th><td>${esc(gate.status)}</td></tr>`;
  html += (gate.reasons || []).map((r) => `<tr><th>note</th><td>${esc(r)}</td></tr>`).join("") + "</table>";
  html += `<h2>In-house IQMs</h2>` + (q.iqms === null
    ? `<p><span class="null">not computed</span> <span class="muted">${esc(q.iqms_reason || "")}</span></p>`
    : `<table>${rowsFor(q.iqms)}</table>`);
  html += `<h2>Official MRIQC</h2>` + (q.mriqc_official === null
    ? `<p><span class="null">not available</span> <span class="muted">${esc(q.mriqc_official_reason || "")}</span></p>`
    : `<p class="muted">source: ${esc(q.mriqc_official.image_id || q.mriqc_official.source)}</p><table>${rowsFor(q.mriqc_official.iqms)}</table>`);
  $("quality-body").innerHTML = html;
}

async function selectSubject(id) {
  current = id;
  document.querySelectorAll("#subjects li").forEach((li) => li.classList.toggle("active", li.dataset.id === id));
  $("empty").hidden = true;
  $("subject-view").hidden = false;
  $("subject-title").textContent = id;
  $("viewer-link").href = `viewer.html?patient_id=${encodeURIComponent(id)}`;
  $("panel-summary").textContent = "Loading…";
  try {
    const doc = await getJSON(`/api/features/${encodeURIComponent(id)}`);
    if (current !== id) return;
    renderSummary(doc);
    $("json-pre").textContent = JSON.stringify(doc, null, 2);
    renderQuality(doc.quality);
    const rep = await getJSON(`/api/report/${encodeURIComponent(id)}`);
    if (current === id) $("panel-report").innerHTML = rep.report_html;
  } catch (e) {
    $("panel-summary").innerHTML = `<p class="error">${esc(e.message)}</p>`;
  }
}

function renderSubjects() {
  const f = $("filter").value.trim().toLowerCase();
  $("subjects").innerHTML = subjects
    .filter((s) => !f || s.id.toLowerCase().includes(f))
    .map((s) => `<li data-id="${esc(s.id)}" class="${s.id === current ? "active" : ""}">${esc(s.id)}${s.source === "upload" ? ' <span class="tag">upload</span>' : ""}</li>`)
    .join("");
}

async function loadSubjects() {
  subjects = (await getJSON("/api/patients")).patients;
  renderSubjects();
}

async function pollMriqc(id) {
  clearInterval(mriqcTimer);
  mriqcTimer = setInterval(async () => {
    const job = await getJSON(`/api/mriqc/status/${encodeURIComponent(id)}`).catch(() => null);
    if (!job) return;
    $("mriqc-log").textContent = `status: ${job.status}${job.error ? " — " + job.error : ""}\n` + (job.logs || []).slice(-40).join("");
    if (job.status !== "running") {
      clearInterval(mriqcTimer);
      if (job.status === "completed" && current === id) selectSubject(id);
    }
  }, 3000);
}

document.addEventListener("DOMContentLoaded", async () => {
  const st = await getJSON("/api/status").catch(() => null);
  if (st) {
    $("version").textContent = st.version;
    $("status").textContent = `${st.n_dataset_subjects} dataset subjects · ${st.n_uploads} uploads · ${st.cuda_available ? st.device : "CPU"}`;
    $("stage-note").textContent = st.stage;
  }
  await loadSubjects();

  $("filter").addEventListener("input", renderSubjects);
  $("subjects").addEventListener("click", (e) => {
    const li = e.target.closest("li");
    if (li) selectSubject(li.dataset.id);
  });
  document.querySelectorAll("button.tab").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("button.tab").forEach((x) => x.classList.toggle("active", x === b));
    document.querySelectorAll(".panel").forEach((p) => { p.hidden = p.id !== `panel-${b.dataset.tab}`; });
  }));
  $("upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = $("file-input").files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file, file.name);
    $("upload-msg").textContent = "Uploading…";
    try {
      const r = await getJSON("/api/upload", { method: "POST", body: fd });
      $("upload-msg").textContent = `Uploaded as ${r.patient_id}.`;
      await loadSubjects();
      selectSubject(r.patient_id);
    } catch (err) {
      $("upload-msg").textContent = `Upload failed: ${err.message}`;
    }
  });
  $("mriqc-run").addEventListener("click", async () => {
    if (!current) return;
    try {
      await getJSON(`/api/mriqc/run/${encodeURIComponent(current)}`, { method: "POST" });
      pollMriqc(current);
    } catch (err) {
      $("mriqc-log").textContent = `Could not start MRIQC: ${err.message}`;
    }
  });
});
