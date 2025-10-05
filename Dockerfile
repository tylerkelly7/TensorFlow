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

# ---- Install dependencies ----
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# ---- Environment variables ----
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# ---- Default command ----
CMD ["python", "-m", "src.train", "--task", "mnist"]