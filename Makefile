# ============================================================
# TensorFlow - Makefile
# Author: Tyler Kelly
# Purpose: Build, run, train, and clean TensorFlow Docker project
# ============================================================

IMAGE_NAME = tensorflow
CONTAINER_NAME = tf_project_container
RESULTS_DIR = $(PWD)/results

# ------------------------------------------------------------
# Core Commands
# ------------------------------------------------------------

# Build Docker image
build:
	@echo "[INFO] Building Docker image: $(IMAGE_NAME)"
	docker build -t $(IMAGE_NAME) .

# Run container interactively (for shell debugging)
run:
	@echo "[INFO] Launching interactive container shell..."
	docker run -it --rm --name $(CONTAINER_NAME) $(IMAGE_NAME) /bin/bash

# Train model for specified task (default: mnist)
train:
	@echo "[INFO] Training model inside container..."
	docker run -it --rm \
		-v $(RESULTS_DIR):/app/results \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME) \
		python -m src.train --task mnist

# Evaluate trained model
evaluate:
	@echo "[INFO] Evaluating model..."
	docker run -it --rm \
		-v $(RESULTS_DIR):/app/results \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME) \
		python -m src.evaluate --model results/models/mnist_best.h5 --task mnist

# Clean up Docker images, containers, and cache
clean:
	@echo "[INFO] Cleaning up Docker artifacts..."
	docker system prune -af

# ------------------------------------------------------------
# Utility / Helpers
# ------------------------------------------------------------

# Build + Train in one command
train-all: build train

# Inspect built image
inspect:
	docker image inspect $(IMAGE_NAME) | jq '.[0] | {RepoTags, Size, Created}'

# Smoke test (used in CI)
smoke:
	@echo "[INFO] Verifying TensorFlow installation..."
	docker run --rm $(IMAGE_NAME) python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"