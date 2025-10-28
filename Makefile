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

# ------------------------------------------------------------
# Publishing
# ------------------------------------------------------------

# 🏷️ Tag local image for registry
tag:
	@if [ -z "$(REGISTRY)" ] || [ -z "$(DOCKER_USER)" ] || [ -z "$(TAG)" ]; then \
		echo "[ERROR] Missing REGISTRY, DOCKER_USER, or TAG variables."; \
		echo "Usage: make publish REGISTRY=ghcr.io DOCKER_USER=tylerkelly7 TAG=latest"; \
		exit 1; \
	fi
	@echo "[INFO] Tagging image as $(REGISTRY)/$(DOCKER_USER)/$(IMAGE_NAME):$(TAG)"
	docker tag $(IMAGE_NAME) $(REGISTRY)/$(DOCKER_USER)/$(IMAGE_NAME):$(TAG)

# 🔐 Authenticate to registry (Docker Hub or GHCR)
login:
	@if [ -z "$(REGISTRY)" ] || [ -z "$(DOCKER_USER)" ]; then \
		echo "[ERROR] Missing REGISTRY or DOCKER_USER variables."; \
		exit 1; \
	fi
	@echo "[INFO] Logging in to registry: $(REGISTRY)"
ifeq ($(REGISTRY),docker.io)
	docker login -u $(DOCKER_USER)
else
	echo "$${GH_TOKEN}" | docker login $(REGISTRY) -u $(DOCKER_USER) --password-stdin
endif

# 🚀 Publish image to registry (with retry logic for GHCR)
publish: build tag
	@if [ -z "$(REGISTRY)" ] || [ -z "$(DOCKER_USER)" ]; then \
		echo "[ERROR] Missing REGISTRY or DOCKER_USER variables."; \
		exit 1; \
	fi
	@echo "[INFO] Publishing image to $(REGISTRY)/$(DOCKER_USER)/$(IMAGE_NAME):$(TAG)"
	$(MAKE) login
	@echo "[INFO] Starting push attempts (max 3)..."
	@for i in 1 2 3; do \
		docker push --disable-content-trust=true $(REGISTRY)/$(DOCKER_USER)/$(IMAGE_NAME):$(TAG) && break || \
		( echo "[WARN] Push attempt $$i failed, retrying in 30s..." && sleep 30 ); \
	done
	@echo "[SUCCESS] Image published: $(REGISTRY)/$(DOCKER_USER)/$(IMAGE_NAME):$(TAG)"

	@echo "[INFO] Tagging 'latest' for convenience..."
	docker tag $(REGISTRY)/$(DOCKER_USER)/$(IMAGE_NAME):$(TAG) $(REGISTRY)/$(DOCKER_USER)/$(IMAGE_NAME):latest
	@for i in 1 2 3; do \
		docker push --disable-content-trust=true $(REGISTRY)/$(DOCKER_USER)/$(IMAGE_NAME):latest && break || \
		( echo "[WARN] Push attempt $$i (latest) failed, retrying in 30s..." && sleep 30 ); \
	done
	@echo "[SUCCESS] 'latest' tag updated."