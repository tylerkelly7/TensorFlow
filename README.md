# TensorFlow

This repository demonstrates proficiency with TensorFlow 2.x across multiple deep learning applications.
## Contents
| Notebook | Description |
|-----------|--------------|
| `01_tensor_basics_tf.ipynb` | Tensor creation, GPU usage, and gradients |
| `02_nn_classification_tf.ipynb` | Dense neural network for MNIST |
| `03_cnn_image_tf.ipynb` | CNN for CIFAR-10 with data augmentation |
| `04_rnn_text_tf.ipynb` | RNN/LSTM for IMDB sentiment analysis |
| `05_transfer_learning_tf.ipynb` | Transfer learning with MobileNetV2 |

## ⚙️ Installation
```bash
git clone https://github.com/YOUR_USERNAME/TensorFlow_Projects.git
cd TensorFlow_Projects
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Structure

TensorFlow/
│── config/         # Configs for hyperparameters and seeding
│── data/           # Raw and processed data
│── notebooks/      # All TensorFlow notebooks
│── src/            # Modular Python scripts (data, models, training)
│── tests/          # Unit tests for reproducibility
│── Results/        # Model outputs, metrics, plots
│── .github/        # CI/CD workflows

## Skills

- TensorFlow 2.x (Keras API)
- CNNs, RNNs, Transfer Learning
- Config-driven design and modular code
- CI/CD Integration using GitHub Actions