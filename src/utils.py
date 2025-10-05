# src/utils.py
"""
Reusable utilities for visualization, callbacks, metrics, and seed/config control.
"""

import os
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# ============================================================
# 1. Configuration & Global Seeding
# ============================================================

# Locate config.yaml (assumed in ../config/)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")

def get_config():
    """Load YAML configuration file."""
    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print("[WARNING] config.yaml not found. Using default parameters.")
        return {"global": {"seed": 42}, "train": {"early_stopping_patience": 3}}

def set_global_seed(seed: int = 42, verbose: bool = True):
    """Ensure reproducibility across Python, NumPy, and TensorFlow."""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if verbose:
        print(f"[INFO] Global seed set to {seed}")

# Load config and apply seed on import
config = get_config()
set_global_seed(config["global"]["seed"])

# ============================================================
# 2. Visualization Utilities
# ============================================================
def plot_training_curves(history, metric="accuracy", save_path=None):
    """Plot training and validation curves."""
    plt.figure(figsize=(6, 4))
    plt.plot(history.history.get(metric, []), label=f"train_{metric}")
    plt.plot(history.history.get(f"val_{metric}", []), label=f"val_{metric}")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.title(f"Training and Validation {metric}")
    plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_roc_curve(y_true, y_probs, save_path=None):
    """Plot ROC curve and compute AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    return auc_score

# ============================================================
# 3. Callback Utilities
# ============================================================
def build_callbacks(task_name, base_dir="Results"):
    """Create Keras callbacks for checkpointing, early stopping, and TensorBoard."""
    os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)

    model_path = os.path.join(base_dir, "models", f"{task_name}_best.h5")
    tensorboard_dir = os.path.join(base_dir, "logs", task_name)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path, save_best_only=True, monitor="val_loss", verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["train"]["early_stopping_patience"],
            restore_best_weights=True,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir),
    ]
    return callbacks, model_path

# ============================================================
# 4. Metrics Summary Utilities
# ============================================================
def summarize_metrics(history, test_metrics):
    """Summarize and print final metrics from training and evaluation."""
    final_train = {k: v[-1] for k, v in history.history.items() if not k.startswith("val_")}
    final_val = {k: v[-1] for k, v in history.history.items() if k.startswith("val_")}
    summary = {
        "train_metrics": final_train,
        "val_metrics": final_val,
        "test_metrics": {"loss": test_metrics[0], "accuracy": test_metrics[1]},
    }
    print("\n=== Final Metrics Summary ===")
    for section, metrics in summary.items():
        print(f"{section}: {metrics}")
    return summary
        
# ============================================================
# 5. Experiment Tracking (MLflow / Weights & Biases)
# ============================================================

import mlflow
import mlflow.tensorflow
import wandb

def init_experiment_tracking(task_name):
    """
    Initialize MLflow or Weights & Biases tracking based on config flags.
    Returns a context handle (MLflow run object or W&B session).
    """
    run = None

    if config["tracking"].get("use_mlflow", False):
        mlflow.set_tracking_uri(config["tracking"]["mlflow_uri"])
        mlflow.set_experiment(config["tracking"]["experiment_name"])

        # --- 🟢 Auto-close any previously active run ---
        if mlflow.active_run():
            mlflow.end_run()

        print(f"[INFO] MLflow tracking initialized → {config['tracking']['mlflow_uri']}")
        run = mlflow.start_run(run_name=task_name)

    elif config["tracking"].get("use_wandb", False):
        wandb.login()
        wandb.init(
            project=config["tracking"]["wandb_project"],
            entity=config["tracking"]["wandb_entity"],
            name=task_name,
            config=config,
        )
        print(f"[INFO] W&B tracking initialized for task: {task_name}")

    else:
        print("[INFO] Experiment tracking disabled.")

    return run


def log_experiment_metrics(history, test_metrics, run=None, task_name=None):
    """
    Log parameters, metrics, and artifacts to MLflow or W&B.
    Does not close the session — call end_experiment_tracking() after.
    """
    train_metrics = {k: v[-1] for k, v in history.history.items() if not k.startswith("val_")}
    val_metrics = {k: v[-1] for k, v in history.history.items() if k.startswith("val_")}
    test_dict = {"test_loss": test_metrics[0], "test_acc": test_metrics[1]}

    # --- MLflow ---
    if config["tracking"].get("use_mlflow", False) and run is not None:
        mlflow.log_params(config.get("train", {}))
        mlflow.log_metrics({**train_metrics, **val_metrics, **test_dict})
        mlflow.log_artifacts("Results/models")
        mlflow.log_artifacts("Results/logs")
        print(f"[INFO] Logged metrics to MLflow: {test_dict}")

    # --- W&B ---
    elif config["tracking"].get("use_wandb", False):
        wandb.log({**train_metrics, **val_metrics, **test_dict})
        print(f"[INFO] Logged metrics to Weights & Biases: {test_dict}")


def end_experiment_tracking():
    """
    Cleanly close MLflow or W&B runs to allow next experiment to start.
    """
    if config["tracking"].get("use_mlflow", False):
        if mlflow.active_run():
            mlflow.end_run()
            print("[INFO] MLflow run ended successfully.")

    if config["tracking"].get("use_wandb", False):
        wandb.finish()
        print("[INFO] W&B session ended successfully.")
        
        
        
        
    
