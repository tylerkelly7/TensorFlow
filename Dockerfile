# ============================================================
# TensorFlow - Dockerfile
# Author: Tyler Kelly
# Purpose: Containerize TensorFlow repo for reproducible training
# ============================================================

# ---- Base image ----
FROM tensorflow/tensorflow:2.14.0

# ---- Set working directory ----
WORKDIR /app

# ---- Copy project files ----
COPY . /app

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- Upgrade pip & core build tools ----
RUN pip install --upgrade pip setuptools wheel

# ---- Install project dependencies ----
RUN pip install --no-cache-dir --ignore-installed blinker -r requirements.txt

# ---- Environment variables ----
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    USE_MLFLOW=false \
    USE_WANDB=false

# ---- Entrypoint ----
CMD ["python", "-m", "src.train", "--task", "mnist"]