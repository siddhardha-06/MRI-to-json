# ─────────────────────────────────────────────────────────────────────────────
# NeuroSphere v2 dashboard - CPU-only image (v2 Phase 1)
# The dashboard currently needs no model weights (segmentation arrives in Phase 2),
# so checkpoints/ is no longer baked into the image. Not yet built/tested: the Docker
# daemon was unavailable during Phase 1.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

LABEL description="NeuroSphere v2 dashboard (CPU-only)"
ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU build of the same torch version as the development venv (constraints.txt pins the CUDA build there).
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN grep -v -E "^(torch|torchvision|torchaudio)==" requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt

COPY neurosphere/ ./neurosphere/
COPY training/ ./training/
COPY train_unet3d.py app_server.py ./
COPY web/ ./web/

RUN mkdir -p uploads work ixi-t1
EXPOSE 8000
CMD ["python", "app_server.py"]
