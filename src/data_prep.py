"""
src/data.py
-----------
Handles all dataset loading, preprocessing, and batching.
Uses configuration from config/config.yaml to avoid hardcoding.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from config.seeds import set_global_seed, get_config

# ------------------------------------------------------------
# 1. Load configuration and set seeds
# ------------------------------------------------------------
config = get_config()
set_global_seed(config["global"]["seed"])

# ------------------------------------------------------------
# 2. Utility: Build tf.data.Dataset
# ------------------------------------------------------------
def make_tf_dataset(x, y, batch_size=32, shuffle=True, buffer_size=10000):
    """Converts numpy arrays into an efficient tf.data pipeline."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ------------------------------------------------------------
# 3. Dataset-specific loaders
# ------------------------------------------------------------
def load_mnist():
    """Loads MNIST dataset, applies config preprocessing."""
    params = config["data"]["mnist"]
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train.astype("float32"), x_test.astype("float32")

    if params["normalize"]:
        x_train, x_test = x_train / 255.0, x_test / 255.0
    if params["flatten"]:
        x_train = x_train.reshape(-1, 28 * 28)
        x_test = x_test.reshape(-1, 28 * 28)

    return x_train, y_train, x_test, y_test


def load_cifar10():
    """Loads CIFAR-10 dataset."""
    params = config["data"]["cifar10"]
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train.astype("float32"), x_test.astype("float32")
    if params["normalize"]:
        x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = y_train.squeeze(), y_test.squeeze()
    return x_train, y_train, x_test, y_test


def load_imdb():
    """Loads IMDB dataset for sentiment analysis."""
    params = config["data"]["imdb"]
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=params["num_words"]
    )
    x_train = sequence.pad_sequences(x_train, maxlen=params["maxlen"])
    x_test = sequence.pad_sequences(x_test, maxlen=params["maxlen"])
    return x_train, y_train, x_test, y_test

# ------------------------------------------------------------
# 4. Wrapper: Retrieve dataset dynamically
# ------------------------------------------------------------
def get_dataset(name: str):
    """Generic interface to load dataset by name."""
    name = name.lower()
    if name == "mnist":
        return load_mnist()
    elif name == "cifar10":
        return load_cifar10()
    elif name == "imdb":
        return load_imdb()
    else:
        raise ValueError(f"Unsupported dataset: {name}")