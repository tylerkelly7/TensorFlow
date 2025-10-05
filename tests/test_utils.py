# tests/test_utils.py

"""
Verify reproducibility, callback creation, and tracking initialization.
"""

import os
import tensorflow as tf
from src import utils

def test_set_global_seed_reproducible():
    utils.set_global_seed(123, verbose=False)
    a = tf.random.uniform((1,))
    utils.set_global_seed(123, verbose=False)
    b = tf.random.uniform((1,))
    assert float(a) == float(b), "Random numbers differ → seed not deterministic"

def test_build_callbacks_creates_files(tmp_path):
    callbacks, model_path = utils.build_callbacks("mnist", base_dir=tmp_path)
    assert any(isinstance(cb, tf.keras.callbacks.ModelCheckpoint) for cb in callbacks)
    assert "mnist_best.h5" in str(model_path)

def test_plot_functions_run(tmp_path):
    # Smoke test for plots (no visual assertion)
    import numpy as np
    history = tf.keras.callbacks.History()
    history.history = {"accuracy": [0.8, 0.9], "val_accuracy": [0.75, 0.88]}
    utils.plot_training_curves(history, save_path=tmp_path / "curve.png")
    utils.plot_confusion_matrix([0,1,1], [0,1,0])
    utils.plot_roc_curve([0,1,1], [0.2,0.9,0.8])

def test_tracking_initialization_and_close(monkeypatch):
    # Disable real MLflow/W&B connections
    utils.config["tracking"]["use_mlflow"] = False
    utils.config["tracking"]["use_wandb"] = False
    run = utils.init_experiment_tracking("mnist_test")
    assert run is None
    utils.end_experiment_tracking()