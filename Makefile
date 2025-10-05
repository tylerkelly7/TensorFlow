# ============================================================
# TensorFlow - Makefile
# Tyler Kelly
# ============================================================

IMAGE_NAME=tensorflow
CONTAINER_NAME=tf_project_container

# Build Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run container interactively
run:
	docker run -it --rm --name $(CONTAINER_NAME) $(IMAGE_NAME)

# Train with specific task
train:
	docker run -it --rm -v $(PWD)/results:/app/results $(IMAGE_NAME) \
	python -m src.train --task mnist

# Evaluate model
evaluate:
	docker run -it --rm -v $(PWD)/results:/app/results $(IMAGE_NAME) \
	python -m src.evaluate --model results/models/mnist_best.h5 --task mnist

# Clean Docker artifacts
clean:
	docker system prune -f