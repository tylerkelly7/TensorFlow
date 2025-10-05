"""
src/models.py
--------------
Contains model architecture factories for different TensorFlow tasks.
Each model reads hyperparameters from config/config.yaml.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from src.utils import get_config, set_global_seed

config = get_config()

# ============================================================
# 1. Dense Neural Network for MNIST
# ============================================================
def dense_mnist_model(input_shape=(784,), num_classes=10):
    """
    Simple feedforward neural network for MNIST classification.
    """
    params = config["train"]
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ============================================================
# 2. CNN for CIFAR-10
# ============================================================
def cnn_cifar10_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Convolutional Neural Network for image classification.
    """
    params = config["train"]
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ============================================================
# 3. RNN for IMDB Sentiment
# ============================================================
def rnn_imdb_model(vocab_size=10000, embed_dim=128, maxlen=200):
    """
    Bidirectional LSTM for sentiment classification.
    """
    params = config["train"]
    model = tf.keras.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen, mask_zero=True),
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ============================================================
# 4. Transfer Learning with MobileNetV2
# ============================================================
def mobilenetv2_transfer_model(input_shape=(224, 224, 3), num_classes=2, base_trainable=False):
    """
    Transfer learning using pretrained MobileNetV2.
    """
    params = config["train"]
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = base_trainable

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model