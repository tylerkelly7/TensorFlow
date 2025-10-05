"""
Centralized seed control for reproducibility across runs.
"""
import os, random, numpy as np, tensorflow as tf
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def set_global_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[INFO] Global seed set to {seed}")

def get_config():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config