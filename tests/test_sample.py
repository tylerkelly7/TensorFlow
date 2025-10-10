"""
tests/test_data.py
Verifies data loading, shapes, and value ranges.
"""

from src import data_prep as data
import numpy as np

def test_mnist_shapes():
    x_train, y_train, x_test, y_test = data.load_mnist()
    assert x_train.ndim in [2, 3]
    assert len(x_train) == len(y_train)
    assert np.all((x_train >= 0) & (x_train <= 1))

def test_cifar_shapes():
    x_train, y_train, x_test, y_test = data.load_cifar10()
    assert x_train.shape[1:] == (32, 32, 3)
    assert len(np.unique(y_train)) == 10

def test_imdb_shapes():
    x_train, y_train, x_test, y_test = data.load_imdb()
    assert x_train.shape[1] == 200  # per config.yaml
    assert set(np.unique(y_train)) <= {0, 1}