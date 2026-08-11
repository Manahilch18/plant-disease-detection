"""
src/models/custom_cnn.py
========================
Custom CNN architecture for the Plant Disease Detection System.

This module is the single source of truth for the baseline CNN model.
It mirrors the architecture defined in ``notebooks/05_baseline_cnn.ipynb``
exactly — layer order, filter counts, regularisation, and hyperparameters
are intentionally frozen so that notebook experiments and production
training use identical models.

Architecture summary (5 convolutional blocks + classifier head):
    Input → Rescaling → [Conv → BN → ReLU → MaxPool] × 4
          → [Conv → BN → ReLU] × 1 → GAP → Dropout
          → Dense(256) → ReLU → Dropout → Dense(38, softmax)

Usage::

    from src.models.custom_cnn import build_custom_cnn

    model = build_custom_cnn()
    model.compile(...)
    model.fit(train_ds, ...)

Author: Plant Disease Detection Project
Python: 3.11+
Style : PEP 8, Google-style docstrings
"""

from __future__ import annotations

import logging

import tensorflow as tf
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import (BatchNormalization,Conv2D,Dense,Dropout,GlobalAveragePooling2D,Input,MaxPooling2D,ReLU,Rescaling,)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
 
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reusable building block
# ---------------------------------------------------------------------------


def conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: tuple[int, int] = (3, 3),
    pool_size: tuple[int, int] = (2, 2),
    kernel_initializer: HeNormal = None,
    kernel_regularizer: l2 = None,
    name: str = "block",
) -> tf.Tensor:
    """Apply Conv2D → BatchNormalization → ReLU → MaxPooling2D.

    This helper encapsulates the repeated pattern used in blocks 1–4 of the
    network.  Block 5 skips the pooling step and therefore does *not* use
    this function — its layers are constructed inline in
    :func:`build_custom_cnn`.

    Args:
        x: Input tensor from the previous layer.
        filters: Number of convolutional filters (output channels).
        kernel_size: Height and width of the 2-D convolution window.
            Defaults to ``(3, 3)``.
        pool_size: Height and width of the max-pooling window.
            Defaults to ``(2, 2)``.
        kernel_initializer: Weight initialiser for the convolutional kernel.
            Defaults to ``None`` (Keras default).
        kernel_regularizer: Regularisation function applied to the kernel.
            Defaults to ``None``.
        name: Prefix used to name every layer in this block, e.g.
            ``"block1"`` produces ``"block1_conv"``, ``"block1_bn"``, etc.

    Returns:
        Output tensor after pooling, ready for the next block.
    """
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name=f"{name}_conv",
    )(x)
    x = BatchNormalization(name=f"{name}_bn")(x)
    x = ReLU(name=f"{name}_relu")(x)
    x = MaxPooling2D(pool_size=pool_size, name=f"{name}_pool")(x)
    return x


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def build_custom_cnn(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 38,
    dropout_rate: float = 0.30,
    l2_weight_decay: float = 1e-4,
    model_name: str = "baseline_custom_cnn",
) -> Model:
    """Build the baseline custom CNN for plant disease classification.

    The architecture is fixed to match ``notebooks/05_baseline_cnn.ipynb``.
    Do not alter filter counts, layer order, or hyperparameters without
    updating the notebook accordingly.

    Architecture
    ------------
    ::

        Input(224, 224, 3)
        └─ Rescaling(1./255)
           ├─ block1: Conv2D(32)  → BN → ReLU → MaxPool   # 112 × 112 × 32
           ├─ block2: Conv2D(64)  → BN → ReLU → MaxPool   #  56 ×  56 × 64
           ├─ block3: Conv2D(128) → BN → ReLU → MaxPool   #  28 ×  28 × 128
           ├─ block4: Conv2D(256) → BN → ReLU → MaxPool   #  14 ×  14 × 256
           ├─ block5: Conv2D(512) → BN → ReLU             #  14 ×  14 × 512
           ├─ GlobalAveragePooling2D                       #         512
           ├─ Dropout(0.30)
           ├─ Dense(256) → ReLU
           ├─ Dropout(0.30)
           └─ Dense(38, softmax)  [predictions]

    Notes:
        * Pixel rescaling (``1./255``) is baked into the model graph so that
          raw ``uint8`` or ``float32 [0, 255]`` images can be fed directly.
          When this model is used alongside ``preprocess.py`` with
          ``model_type="custom_cnn"``, the ``normalize_cnn`` function in that
          module also divides by 255.  Choose **one** normalisation path and
          use it consistently: either rely on this ``Rescaling`` layer and
          pass un-normalised images, or remove this layer and use the
          preprocessor.  The notebook uses the ``Rescaling`` layer and passes
          un-normalised images — this module mirrors that behaviour.
        * The model is returned **uncompiled**.  Compilation (optimiser, loss,
          metrics) is the responsibility of the training script or notebook.

    Args:
        input_shape: Spatial dimensions and channels of a single input image.
            Defaults to ``(224, 224, 3)``.
        num_classes: Number of output classes.  Defaults to ``38``
            (PlantVillage dataset).
        dropout_rate: Dropout probability applied before and after the dense
            classifier layer.  Defaults to ``0.30``.
        l2_weight_decay: L2 regularisation coefficient applied to all
            convolutional and dense kernels.  Defaults to ``1e-4``.
        model_name: Name assigned to the ``tf.keras.Model`` instance.
            Defaults to ``"baseline_custom_cnn"``.

    Returns:
        An uncompiled ``tf.keras.Model`` instance.
    """
    logger.info(
        "Building '%s': input_shape=%s, num_classes=%d, "
        "dropout_rate=%.2f, l2_weight_decay=%.1e",
        model_name, input_shape, num_classes, dropout_rate, l2_weight_decay,
    )

    initializer = HeNormal()
    regularizer = l2(l2_weight_decay)

    # ------------------------------------------------------------------
    # Input & rescaling
    # ------------------------------------------------------------------
    inputs = Input(shape=input_shape, name="input_layer")
    x = Rescaling(scale=1.0 / 255.0, name="rescaling")(inputs)

    # ------------------------------------------------------------------
    # Convolutional blocks 1–4  (Conv → BN → ReLU → MaxPool)
    # ------------------------------------------------------------------
    x = conv_block(
        x, filters=32,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        name="block1",
    )
    x = conv_block(
        x, filters=64,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        name="block2",
    )
    x = conv_block(
        x, filters=128,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        name="block3",
    )
    x = conv_block(
        x, filters=256,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        name="block4",
    )

    # ------------------------------------------------------------------
    # Convolutional block 5  (Conv → BN → ReLU, no pooling)
    # ------------------------------------------------------------------
    x = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        name="block5_conv",
    )(x)
    x = BatchNormalization(name="block5_bn")(x)
    x = ReLU(name="block5_relu")(x)

    # ------------------------------------------------------------------
    # Classifier head
    # ------------------------------------------------------------------
    x = GlobalAveragePooling2D(name="global_average_pool")(x)

    x = Dropout(rate=dropout_rate, name="dropout_1")(x)

    x = Dense(
        units=256,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        name="classifier",
    )(x)
    x = ReLU(name="classifier_relu")(x)

    x = Dropout(rate=dropout_rate, name="dropout_2")(x)

    outputs = Dense(
        units=num_classes,
        activation="softmax",
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        name="predictions",
    )(x)

    # ------------------------------------------------------------------
    # Assemble model
    # ------------------------------------------------------------------
    model = Model(inputs=inputs, outputs=outputs, name=model_name)

    total_params = model.count_params()
    logger.info(
        "Model '%s' built successfully — total parameters: %s.",
        model_name, f"{total_params:,}",
    )
    return model


# ---------------------------------------------------------------------------
# Local verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_custom_cnn()
    model.summary()
