# TensorFlow

![Build](https://github.com/tylerkelly7/TensorFlow/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![Coverage](https://img.shields.io/badge/coverage-85%25-green)
![Docs](https://img.shields.io/badge/docs-online-blue)

End-to-end TensorFlow repository demonstrating model development, reproducibility, and MLOps practices.  
Includes experiment tracking, CI/CD testing, and modular design similar to professional ML engineering workflows.

---

## 🚀 Features
- **Data Layer:** Config-driven loading for MNIST, CIFAR-10, IMDB.
- **Model Architectures:** DNN, CNN, RNN, and MobileNetV2 transfer learning.
- **Training Pipeline:** Early stopping, checkpoints, TensorBoard visualization.
- **Experiment Tracking:** MLflow (default) or Weights & Biases integration.
- **Reproducibility:** Centralized config + seed control.
- **Testing & CI/CD:** Automated tests with GitHub Actions + pytest.
- **Visualization:** Accuracy curves, confusion matrices, ROC curves.

---

## Installation

```bash
# Clone and install dependencies
git clone https://github.com/tylerkelly7/TensorFlow.git
cd TensorFlow
pip install -r requirements.txt
```

To create a conda environment:

```bash
conda env create -f environment.yml
conda activate tensorflow
```

---

## Training

Train a model directly from the command line:

```bash
python -m src.train --task mnist
python -m src.train --task cifar10
python -m src.train --task imdb
```

View TensorBoard:

```bash
tensorboard --logdir results/logs
```

Launch MLflow UI (if enabled):

```bash
mlflow ui --backend-store-uri file:./results/mlruns
```

---

## Testing

Run all tests with:

```bash
pytest -q --disable-warnings --maxfail=1
```

Continuous Integration automatically runs via GitHub Actions on every push.

---

<!--
## Results (Example)

| Model | Dataset | Test Accuracy | Framework |
|--------|----------|----------------|------------|
| Dense NN | MNIST | 98.1% | TensorFlow |
| CNN | CIFAR-10 | 84.5% | TensorFlow |
| LSTM | IMDB | 88.0% | TensorFlow |
| MobileNetV2 | Flowers | 93.7% | TensorFlow (Transfer Learning) |
-->

---

## Tech Stack

| Category | Tools |
|-----------|--------|
| Framework | TensorFlow 2.x, Keras |
| Tracking | MLflow / W&B |
| CI/CD | GitHub Actions |
| Testing | Pytest |
| Visualization | Matplotlib, TensorBoard |
| Config | YAML |
| Environment | Conda, Pip |
| Containerization | Docker |

---

## Next Steps

This repository is complete through Docker & Deployment Integration**.  
Planned next steps for continued development include:

| Task | Name | Goal |
|------|------|------|
| **Hyperparameter Tuning + MLflow Integration** | Add `src/tune.py` for config-driven tuning (grid/random search) with automatic MLflow logging. |
| **Model Registry + Versioned Artifacts** | Register best MLflow runs as versioned models; enable lineage tracking. |
| **Inference API + FastAPI Deployment** | Build REST endpoint (`src/api.py`) exposing `/predict` with trained model. |
| **Docker + FastAPI Integration** | Extend Dockerfile/Makefile to deploy API via `uvicorn`. |
| **Monitoring & Logging Extensions** | Add model drift detection, TensorBoard scalars, and custom callbacks. |
| **Experiment Dashboard (Streamlit)** | Visualize MLflow results interactively. |
| **Docs & Packaging** | Auto-generate docs (`pdoc`/Sphinx) and add final badges. |
| **Final Portfolio Polish** | Add architecture diagram, DockerHub link, and README showcase images. |

---

## 👨‍💻 Author
**Tyler Kelly**  
M.S. Biostatistics — University of Pittsburgh  
Portfolio: [github.com/tylerkelly7](https://github.com/tylerkelly7)