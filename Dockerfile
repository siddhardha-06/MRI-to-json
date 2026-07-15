# ─────────────────────────────────────────────────────────────────────────────
# MRI Pipeline – CPU-only Dockerfile
# Base: Python 3.11 slim (Debian Trixie)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="MRI Pipeline Team"
LABEL description="NeuroSphere MRI Pipeline Dashboard (CPU-only)"

# ── Environment ───────────────────────────────────────────────────────────────
# Set early so subsequent RUN steps inherit these values
# DOCKER_ENV=1     : tells app_server.py NOT to open a browser window
# MPLBACKEND=Agg   : non-interactive backend — no display/OpenGL needed
# PYTHONUNBUFFERED : real-time stdout/stderr in docker logs
ENV DOCKER_ENV=1 \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

# ── System dependencies ───────────────────────────────────────────────────────
# libgomp1        : OpenMP runtime required by PyTorch CPU
# libglib2.0-0t64 : required by some antspyx / scikit-image internals
# git             : antspyx may pull assets at install time
# curl            : useful for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0t64 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker layer-caches the install step.
COPY requirements.txt .

# Install CPU-only torch/torchvision/torchaudio (avoids downloading CUDA wheels)
RUN pip install --no-cache-dir \
        "torch>=2.0.0" \
        "torchvision>=0.15.0" \
        "torchaudio>=2.0.0" \
        --index-url https://download.pytorch.org/whl/cpu

# Install remaining project dependencies
RUN pip install --no-cache-dir \
        "nibabel>=4.0.0" \
        "numpy>=1.21.0" \
        "scipy>=1.7.0" \
        "scikit-image>=0.19.0" \
        "pillow>=9.0.0" \
        "google-genai>=0.1.0" \
        "tqdm>=4.62.0" \
        "matplotlib>=3.5.0" \
        "scikit-learn>=1.0.0" \
        "antspyx>=0.3.8"

# ── Application source ────────────────────────────────────────────────────────
COPY app_server.py .
COPY pipeline.py .
COPY generate_report.py .
COPY extract_features.py .
COPY train_regressor.py .
COPY train_autoencoder.py .
COPY train_unet.py .
COPY web/ ./web/

# ── Model checkpoints (baked in) ─────────────────────────────────────────────
# ~330 MB of .pth weights + atlas files included directly in the image.
COPY checkpoints/ ./checkpoints/

# ── Runtime directories ───────────────────────────────────────────────────────
# Docker Compose will mount volumes at these paths to persist user data.
RUN mkdir -p uploads uploads-masks features reports

# ── Networking ────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["python", "app_server.py"]
