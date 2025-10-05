# TensorFlow

![Build](https://github.com/tylerkelly7/TensorFlow_Projects/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![Coverage](https://img.shields.io/badge/coverage-85%25-

End-to-end TensorFlow repository demonstrating model development, reproducibility, and MLOps practices.  
Includes experiment tracking, CI/CD testing, and modular designs.

---

## Features
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
git clone https://github.com/tylerkelly7/TensorFlow_Projects.git
cd TensorFlow_Projects
pip install -r requirements.txt
```

To create a conda environment:

```bash
conda env create -f environment.yml
conda activate tensorflow-projects
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
tensorboard --logdir Results/logs
```

Launch MLflow UI (if enabled):

```bash
mlflow ui --backend-store-uri file:./Results/mlruns
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

---
-->

## 🧰 Tech Stack

| Category | Tools |
|-----------|--------|
| Framework | TensorFlow 2.x, Keras |
| Tracking | MLflow / W&B |
| CI/CD | GitHub Actions |
| Testing | Pytest |
| Visualization | Matplotlib, TensorBoard |
| Config | YAML |
| Environment | Conda, Pip |

---

## 🔮 Next Enhancements
- Add **Dockerfile** for containerized training and inference.  
- Add **Hyperparameter Tuning** with `keras-tuner` or `Optuna`.  
- Extend MLflow tracking to include **artifact versioning and model registry**.  
- Generate **HTML reports** of metrics for quick sharing.

---

## 👨‍💻 Author
**Tyler Kelly**  
M.S. Biostatistics — University of Pittsburgh  
Portfolio: [github.com/tylerkelly7](https://github.com/tylerkelly7)