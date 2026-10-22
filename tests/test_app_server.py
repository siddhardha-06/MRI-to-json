"""The dashboard serves only package output and never writes into the dataset."""
import json
import threading
import urllib.error
import urllib.request

import pytest

import app_server
from tests.conftest import save_nifti


@pytest.fixture
def server(tmp_path, ras_volume, monkeypatch):
    data, aff = ras_volume
    ixi = tmp_path / "ixi-t1" / "IXI002-Guys-0828-T1.nii"
    ixi.mkdir(parents=True)
    save_nifti(ixi, data, aff, "IXI002-Guys-0828-MPRAGESEN_-s256_-0301-00003-000001-01.nii")
    save_nifti(ixi, data, aff, "IXI002-Guys-0828-MPRAGESEN_-s256_-0301-00003-000001-01_n4.nii")  # derived: must be ignored
    monkeypatch.setattr(app_server, "IXI_DIR", tmp_path / "ixi-t1")
    monkeypatch.setattr(app_server, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(app_server, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(app_server, "_DATASET_CACHE", None)
    httpd = app_server.socketserver.ThreadingTCPServer(("127.0.0.1", 0), app_server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", tmp_path
    httpd.shutdown(); httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def test_features_report_and_listing(server):
    base, tmp = server
    before = sorted(p.name for p in (tmp / "ixi-t1").rglob("*"))
    assert _get(base + "/api/patients")["patients"] == [{"id": "IXI002-Guys-0828", "source": "dataset"}]
    doc = _get(base + "/api/features/IXI002-Guys-0828")
    assert doc["schema_version"] == "2.0.0-interim" and doc["volumes_ml"] is None and doc["seg_url"] is None
    assert doc["mri_url"].endswith(".nii")
    rep = _get(base + "/api/report/IXI002-Guys-0828")
    assert "not computed" in rep["report_text"]
    assert _get(base + "/api/quality/IXI002-Guys-0828")["gate"]["status"] == "not_computed"
    assert sorted(p.name for p in (tmp / "ixi-t1").rglob("*")) == before   # nothing written into the dataset


def test_unknown_subject_and_path_traversal(server):
    base, _ = server
    for path in ("/api/features/IXI999", "/api/features/..%2F..%2Fsecret", "/scans/dataset/..%2F.env"):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + path)
        assert e.value.code in (400, 404)


def test_training_steps_removed(server):
    base, _ = server
    req = urllib.request.Request(base + "/api/run-step", data=b"{}", method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 410


def test_upload_roundtrip(server, ras_volume, tmp_path):
    base, tmp = server
    data, aff = ras_volume
    f = save_nifti(tmp_path, data, aff, "my scan.nii.gz")
    boundary = "XyZ123"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"my scan.nii.gz\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + f.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(base + "/api/upload", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req) as r:
        out = json.loads(r.read())
    assert out["patient_id"] == "my_scan"
    assert (tmp / "uploads" / "my_scan" / "my_scan.nii.gz").read_bytes() == f.read_bytes()
    doc = _get(base + "/api/features/my_scan")
    assert doc["subject"]["id"] == "my_scan" and doc["acquisition"]["site"] is None
