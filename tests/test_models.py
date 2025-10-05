#tests/test_models.py
"""
Basic model construction tests to ensure architectures compile correctly.
"""

from src import models
import tensorflow as tf
import numpy as np

def test_dense_mnist():
    model = models.dense_mnist_model()
    x = np.random.rand(8, 784).astype("float32")
    y = model(x)
    assert y.shape == (8, 10)

def test_cnn_cifar10():
    model = models.cnn_cifar10_model()
    x = np.random.rand(4, 32, 32, 3).astype("float32")
    y = model(x)
    assert y.shape == (4, 10)

def test_rnn_imdb():
    model = models.rnn_imdb_model()
    x = np.random.randint(0, 10000, size=(4, 200))
    y = model(x)
    assert y.shape == (4, 1)

def test_transfer_mobilenetv2():
    model = models.mobilenetv2_transfer_model()
    x = np.random.rand(2, 224, 224, 3).astype("float32")
    y = model(x)
    assert y.shape == (2, 2)