"""NeuroSphere dashboard server (v2 Phase 1).

A thin HTTP layer over the neurosphere package. It serves only what the package computes:
the interim JSON (header-derived values plus null + reason for everything not yet
implemented), a measured-values-only report, and official MRIQC output when the user runs it.
The v1 server (simulated QC, hard-coded findings, invalid U-Net geometry)
is kept for reference in legacy/app_server_v1.py.

Run: .\\.venv\\Scripts\\python app_server.py   (http://127.0.0.1:8000; HOST/PORT env vars override)
Nothing is written into ixi-t1/ or next to uploaded scans except the upload itself; all derived
output goes to work/.
"""
from __future__ import annotations

import http.server
import json
import os
import re
import socketserver
import subprocess
import threading
import urllib.parse
from pathlib import Path

from neurosphere import __version__
from neurosphere.io import is_derived, strip_nifti_ext
from neurosphere.jsonio import build_interim_json, write_json
from neurosphere.provenance import sha256_file
from neurosphere.qc import parse_mriqc_official
from neurosphere.report.template import render_html, render_text

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
IXI_DIR = ROOT / "ixi-t1"
UPLOADS_DIR = ROOT / "uploads"
WORK_DIR = ROOT / "work"
MRIQC_DIR = WORK_DIR / "mriqc"
MRIQC_IMAGE = os.environ.get("NEUROSPHERE_MRIQC_IMAGE", "nipreps/mriqc:latest")  # pin by digest when recorded (ENV.md)
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
MAX_UPLOAD_BYTES = 2 * 1024 ** 3
_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

mriqc_jobs: dict[str, dict] = {}


# ── subjects ──────────────────────────────────────────────────────────────────
_DATASET_CACHE: dict[str, Path] | None = None


def dataset_subjects() -> dict[str, Path]:
    """{'IXI002-Guys-0828': path to raw scan} for every raw scan in ixi-t1/ (indexed once; the dataset is read-only)."""
    global _DATASET_CACHE
    if _DATASET_CACHE is not None:
        return _DATASET_CACHE
    out = {}
    if IXI_DIR.exists():
        for d in sorted(p for p in IXI_DIR.iterdir() if p.is_dir()):
            scans = [f for f in d.iterdir() if f.is_file() and f.name.endswith((".nii", ".nii.gz")) and not is_derived(f.name)]
            if len(scans) == 1:
                out[re.sub(r"-T1\.nii$", "", d.name)] = scans[0]
    _DATASET_CACHE = out
    return out


def upload_subjects() -> dict[str, Path]:
    out = {}
    if UPLOADS_DIR.exists():
        for d in sorted(p for p in UPLOADS_DIR.iterdir() if p.is_dir()):
            for ext in (".nii.gz", ".nii"):
                f = d / f"{d.name}{ext}"
                if f.is_file():
                    out[d.name] = f
                    break
    return out


def resolve(subject: str) -> tuple[Path, str] | None:
    if not _ID_RE.match(subject):
        return None
    up = upload_subjects()
    if subject in up:
        return up[subject], "upload"
    ds = dataset_subjects()
    if subject in ds:
        return ds[subject], "dataset"
    return None


def official_mriqc(subject: str) -> dict | None:
    f = WORK_DIR / "mriqc_official" / f"{subject}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def subject_json(subject: str) -> dict | None:
    hit = resolve(subject)
    if hit is None:
        return None
    path, source = hit
    doc = build_interim_json(path, mriqc_official=official_mriqc(subject),
                             subject_id=None if source == "dataset" else subject)
    return doc


# ── official MRIQC (optional, Docker) ─────────────────────────────────────────
def run_official_mriqc(subject: str, scan: Path) -> None:
    job = mriqc_jobs[subject]
    bids_id = re.sub(r"[^A-Za-z0-9]", "", subject)
    bids, out = MRIQC_DIR / subject / "bids", MRIQC_DIR / subject / "out"
    anat = bids / f"sub-{bids_id}" / "anat"
    try:
        anat.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
        (bids / "dataset_description.json").write_text(json.dumps({"Name": "NeuroSphere", "BIDSVersion": "1.8.0"}))
        dest = anat / f"sub-{bids_id}_T1w{''.join(scan.suffixes[-2:]) if scan.name.endswith('.nii.gz') else '.nii'}"
        if not dest.exists():
            dest.write_bytes(scan.read_bytes())
        cmd = ["docker", "run", "--rm", "-v", f"{bids.resolve().as_posix()}:/data:ro", "-v", f"{out.resolve().as_posix()}:/out",
               MRIQC_IMAGE, "/data", "/out", "participant", "--participant-label", bids_id, "--no-sub", "--mem_gb", "6", "--nprocs", "1"]
        job["command"] = " ".join(cmd)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            job["logs"].append(line.decode("utf-8", errors="ignore"))
            job["logs"] = job["logs"][-400:]
        proc.wait()
        if proc.returncode != 0:
            job.update(status="failed", error=f"docker exited with code {proc.returncode}"); return
        res = out / f"sub-{bids_id}" / "anat" / f"sub-{bids_id}_T1w.json"
        if not res.exists():
            job.update(status="failed", error=f"MRIQC output not found at {res}"); return
        parsed = parse_mriqc_official(json.loads(res.read_text()), image_id=MRIQC_IMAGE)
        parsed["raw_output"] = res.relative_to(ROOT).as_posix()
        parsed["raw_output_sha256"] = sha256_file(res)
        write_json(parsed, WORK_DIR / "mriqc_official" / f"{subject}.json")
        job["status"] = "completed"
    except FileNotFoundError:
        job.update(status="failed", error="docker not found on PATH")
    except Exception as e:  # reported to the UI, never hidden
        job.update(status="failed", error=repr(e))


# ── HTTP ──────────────────────────────────────────────────────────────────────
class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB_DIR), **kw)

    def _json(self, obj, code=200):
        body = json.dumps(obj, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg):
        self._json({"error": msg}, code)

    def do_GET(self):
        path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
        try:
            if path == "/api/status":
                return self._status()
            if path == "/api/patients":
                ds, up = dataset_subjects(), upload_subjects()
                return self._json({"patients": [{"id": k, "source": "upload"} for k in up] +
                                               [{"id": k, "source": "dataset"} for k in ds if k not in up]})
            m = re.match(r"^/api/(features|report|quality|mriqc/status)/([^/]+)$", path)
            if m:
                return self._subject_get(m.group(1), m.group(2))
            m = re.match(r"^/scans/(dataset|upload)/([^/]+)(?:/[^/]+)?$", path)  # trailing file name lets viewers infer the format
            if m:
                return self._scan(m.group(1), m.group(2))
            return super().do_GET()
        except Exception as e:
            return self._err(500, repr(e))

    def do_POST(self):
        path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
        try:
            if path == "/api/upload":
                return self._upload()
            m = re.match(r"^/api/mriqc/run/([^/]+)$", path)
            if m:
                return self._mriqc_run(m.group(1))
            if path == "/api/run-step":
                return self._err(410, "Training steps were removed from the dashboard (v1 pipeline retired; see legacy/).")
            return self._err(404, "unknown endpoint")
        except Exception as e:
            return self._err(500, repr(e))

    def _status(self):
        try:
            import torch
            cuda = torch.cuda.is_available()
            dev = torch.cuda.get_device_name(0) if cuda else None
        except ImportError:
            cuda, dev = False, None
        self._json({"version": __version__, "cuda_available": cuda, "device": dev,
                    "n_dataset_subjects": len(dataset_subjects()), "n_uploads": len(upload_subjects()),
                    "stage": "Phase 1 interim: header-derived values only; segmentation, volumes, QC and normative values arrive in Phases 2-3."})

    def _subject_get(self, kind, subject):
        if kind == "mriqc/status":
            return self._json(mriqc_jobs.get(subject, {"status": "idle", "logs": []}))
        doc = subject_json(subject)
        if doc is None:
            return self._err(404, "unknown subject")
        if kind == "features":
            path, source = resolve(subject)
            doc = dict(doc)
            doc["mri_url"] = f"/scans/{source}/{urllib.parse.quote(subject)}/{urllib.parse.quote(path.name)}"
            doc["seg_url"] = None
            doc["seg_url_reason"] = "no valid segmentation yet; Phase 2"
            return self._json(doc)
        if kind == "quality":
            return self._json(doc["quality"])
        return self._json({"report_html": render_html(doc), "report_text": render_text(doc)})

    def _scan(self, source, subject):
        hit = resolve(subject)
        if hit is None or hit[1] != source:
            return self._err(404, "unknown scan")
        f = hit[0]
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(f.stat().st_size))
        self.send_header("Content-Disposition", f'inline; filename="{f.name}"')
        self.end_headers()
        with open(f, "rb") as fh:
            while chunk := fh.read(1 << 16):
                self.wfile.write(chunk)

    def _upload(self):
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > MAX_UPLOAD_BYTES:
            return self._err(413, "empty or too large upload")
        ctype = self.headers.get("Content-Type", "")
        if "boundary=" not in ctype:
            return self._err(400, "multipart/form-data with a boundary is required")
        boundary = ("--" + ctype.split("boundary=")[1].split(";")[0].strip().strip('"')).encode()
        body = self.rfile.read(n)
        filename, data = None, None
        for part in body.split(boundary):
            head, sep, rest = part.partition(b"\r\n\r\n")
            m = re.search(rb'filename="([^"]+)"', head)
            if sep and m:
                filename, data = m.group(1).decode("utf-8", "ignore"), rest[:-2] if rest.endswith(b"\r\n") else rest
                break
        if not filename or data is None:
            return self._err(400, "no file in the upload")
        name = Path(filename).name
        if not name.endswith((".nii", ".nii.gz")):
            return self._err(400, "only .nii or .nii.gz files are accepted")
        subject = re.sub(r"[^A-Za-z0-9._-]", "_", strip_nifti_ext(name))[:128]
        dest_dir = (UPLOADS_DIR / subject).resolve()
        if UPLOADS_DIR.resolve() not in dest_dir.parents:
            return self._err(400, "invalid file name")
        dest = dest_dir / (subject + (".nii.gz" if name.endswith(".nii.gz") else ".nii"))
        if dest.exists():
            return self._err(409, f"an upload named {subject} already exists; rename the file")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        try:
            doc = subject_json(subject)
        except Exception as e:
            return self._err(422, f"saved, but the file could not be read as NIfTI: {e!r}")
        self._json({"patient_id": subject, "schema_version": doc["schema_version"]})

    def _mriqc_run(self, subject):
        hit = resolve(subject)
        if hit is None:
            return self._err(404, "unknown subject")
        if mriqc_jobs.get(subject, {}).get("status") == "running":
            return self._json(mriqc_jobs[subject])
        mriqc_jobs[subject] = {"status": "running", "logs": [], "image": MRIQC_IMAGE}
        threading.Thread(target=run_official_mriqc, args=(subject, hit[0]), daemon=True).start()
        self._json(mriqc_jobs[subject])

    def log_message(self, fmt, *args):
        if "/api/mriqc/status" not in (str(args[0]) if args else ""):
            super().log_message(fmt, *args)


def main():
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((HOST, PORT), Handler) as httpd:
        print(f"NeuroSphere {__version__} dashboard: http://{'localhost' if HOST in ('127.0.0.1', '0.0.0.0') else HOST}:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            httpd.shutdown()


if __name__ == "__main__":
    main()
