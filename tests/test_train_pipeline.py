# tests/test_train_pipeline.py
"""
Runs a 1-epoch smoke test on MNIST model to confirm training loop works.
"""

import tensorflow as tf
from src.train import train_task
from src.utils import set_global_seed

def test_train_task_mnist_smoke(monkeypatch):
    # shorten epochs to speed up CI
    from src import train, utils
    utils.config["train"]["epochs"] = 1
    utils.config["train"]["batch_size"] = 32
    utils.config["tracking"]["use_mlflow"] = False
    utils.config["tracking"]["use_wandb"] = False
    set_global_seed(42)
    history, metrics = train_task("mnist")
    loss, acc = metrics
    assert 0 <= acc <= 1
    assert loss >= 0