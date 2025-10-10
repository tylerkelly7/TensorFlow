# src/evaluate.py

"""
src/evaluate.py
---------------
Loads trained models and evaluates them on test sets.
Supports reloading via CLI or notebook import.
"""

import argparse
import tensorflow as tf
from src import data_prep as data_utils
from src.utils import get_config, set_global_seed

config = get_config()
set_global_seed(config["global"]["seed"])

def evaluate_model(model_path, task):
    """Loads a trained model and evaluates on the test dataset."""
    print(f"[INFO] Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    if task == "mnist":
        _, _, x_test, y_test = data_utils.load_mnist()
    elif task == "cifar10":
        _, _, x_test, y_test = data_utils.load_cifar10()
    elif task == "imdb":
        _, _, x_test, y_test = data_utils.load_imdb()
    else:
        raise ValueError("Unsupported task")

    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"[RESULT] {task.upper()} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")
    return loss, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained TensorFlow model.")
    parser.add_argument("--model", type=str, required=True, help="Path to model file (.h5)")
    parser.add_argument("--task", type=str, required=True, choices=["mnist", "cifar10", "imdb"])
    args = parser.parse_args()
    evaluate_model(args.model, args.task)