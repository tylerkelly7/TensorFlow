# src/train.py

"""
src/train.py
-------------
Handles training workflows for TensorFlow_Projects.
Loads config, dataset, and model dynamically.
Saves trained models, metrics, and logs for reproducibility.
"""

import argparse
import os
import tensorflow as tf
from datetime import datetime
from src.utils import get_config, set_global_seed
from src import data as data_utils
from src import models as model_utils
from src.utils import init_experiment_tracking, log_experiment_metrics

# ------------------------------------------------------------
# 1. Load configuration and seed
# ------------------------------------------------------------
config = get_config()
set_global_seed(config["global"]["seed"])

# ------------------------------------------------------------
# 2. Setup directory structure
# ------------------------------------------------------------
def prepare_output_dirs(base_dir="results"):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
    return base_dir

# ------------------------------------------------------------
# 3. Get model and dataset
# ------------------------------------------------------------
def get_task_components(task):
    """Returns (x_train, y_train, x_test, y_test, model) for a given task."""
    if task == "mnist":
        x_train, y_train, x_test, y_test = data_utils.load_mnist()
        model = model_utils.dense_mnist_model()
    elif task == "cifar10":
        x_train, y_train, x_test, y_test = data_utils.load_cifar10()
        model = model_utils.cnn_cifar10_model()
    elif task == "imdb":
        x_train, y_train, x_test, y_test = data_utils.load_imdb()
        model = model_utils.rnn_imdb_model()
    elif task == "transfer":
        model = model_utils.mobilenetv2_transfer_model()
        x_train = y_train = x_test = y_test = None
    else:
        raise ValueError(f"Unknown task: {task}")
    return x_train, y_train, x_test, y_test, model

# ------------------------------------------------------------
# 4. Define callbacks
# ------------------------------------------------------------
def get_callbacks(log_dir, model_dir, task):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = os.path.join(model_dir, f"{task}_best_{timestamp}.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config["train"]["early_stopping_patience"], restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_dir, f"{task}_{timestamp}"))
    ]
    return callbacks, checkpoint_path

# ------------------------------------------------------------
# 5. Training function
# ------------------------------------------------------------
def train_task(task="mnist"):
    base_dir = prepare_output_dirs(config["global"]["output_dir"])
    log_dir = os.path.join(base_dir, "logs")
    model_dir = os.path.join(base_dir, "models")

    # Load dataset and model
    x_train, y_train, x_test, y_test, model = get_task_components(task)
    callbacks, checkpoint_path = get_callbacks(log_dir, model_dir, task)
    
    # Start experiment tracking (MLflow / W&B)
    run = init_experiment_tracking(task)
    
    print(f"[INFO] Training {task.upper()} model...")
    history = model.fit(
        x_train, y_train,
        validation_split=config["train"]["validation_split"],
        epochs=config["train"]["epochs"],
        batch_size=config["train"]["batch_size"],
        callbacks=callbacks,
        verbose=2,
    )

    print(f"[INFO] Training complete. Best model saved to {checkpoint_path}")

    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"[RESULT] {task.upper()} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    # Log results to MLflow / W&B
    test_metrics = (test_loss, test_acc)
    log_experiment_metrics(history, test_metrics, run, task)

    # End tracking session
    end_experiment_tracking()

    # Save final model
    final_model_path = os.path.join(model_dir, f"{task}_final_{datetime.now().strftime('%H%M%S')}.h5")
    model.save(final_model_path)
    print(f"[INFO] Final model saved to {final_model_path}")

    return history, test_metrics


# ------------------------------------------------------------
# 6. CLI entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TensorFlow models from config.")
    parser.add_argument("--task", type=str, required=True,
                        choices=["mnist", "cifar10", "imdb", "transfer"],
                        help="Choose which model to train.")
    args = parser.parse_args()
    train_task(args.task)