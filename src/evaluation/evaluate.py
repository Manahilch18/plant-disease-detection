"""
src/models/efficientnet.py
==========================
EfficientNetB0 transfer-learning model for the Plant Disease Detection System.

This module is the single source of truth for the EfficientNetB0 architecture
and mirrors the implementation defined in ``notebooks/07_transfer_learning.ipynb``
exactly.  Layer order, regularisation, and all hyperparameters are intentionally
frozen here so that notebook experiments and production training runs are identical.

Training is structured in two phases:

Phase 1 — Feature extraction
    The EfficientNetB0 backbone is fully frozen.  Only the classification head
    is trained.  Use :func:`build_efficientnet_model` with the backbone frozen
    (default) and compile with a moderate learning rate (e.g. ``1e-3``).

Phase 2 — Fine-tuning
    The last ``config["FINE_TUNE_LAYERS"]`` backbone layers are unfrozen
    (excluding all BatchNormalization layers, which remain frozen throughout).
    Use :func:`configure_fine_tuning` and re-compile with a lower learning rate
    (e.g. ``1e-4`` or ``1e-5``).

Usage::

    from src.models.efficientnet import (
        build_data_augmentation,
        build_efficientnet_model,
        configure_fine_tuning,
        count_model_parameters,
    )

    config = {
        "NUM_CLASSES": 38,
        "INPUT_SHAPE": (224, 224, 3),
        "DROPOUT_RATE": 0.30,
        "DENSE_UNITS": 256,
        "AUGMENTATION_SEED": 42,
        "FINE_TUNE_LAYERS": 30,
        "MODEL_NAME": "plant_efficientnetb0",
    }

    augmentation = build_data_augmentation(config)
    model = build_efficientnet_model(config, augmentation)

    # Phase 1 — train the head only
    model.compile(...)
    model.fit(train_ds, ...)

    # Phase 2 — fine-tune last 30 backbone layers
    configure_fine_tuning(model, config)
    model.compile(...)          # re-compile with lower lr
    model.fit(train_ds, ...)

Author: Plant Disease Detection Project
Python: 3.11+
Style : PEP 8, Google-style docstrings
"""

from __future__ import annotations

import logging
from typing import Any

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Config = dict[str, Any]

# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------


def build_data_augmentation(config: Config) -> tf.keras.Sequential:
    """Build a reusable data-augmentation pipeline as a Keras Sequential model.

    The augmentation layers are applied only during training (``training=True``).
    During inference they are identity-mapped, so the Sequential model can be
    embedded directly in the model graph without branching logic in the training
    script.

    Augmentation operations applied (in order, matching the notebook):
        1. ``RandomFlip("horizontal")``
        2. ``RandomRotation(0.10)``
        3. ``RandomZoom(0.10)``
        4. ``RandomContrast(0.10)``

    Args:
        config: Configuration dictionary.  Recognised keys:

            ``"AUGMENTATION_SEED"`` *(int, default 42)*
                Integer seed forwarded to every augmentation layer for
                reproducibility.

    Returns:
        A ``tf.keras.Sequential`` augmentation pipeline.
    """
    seed: int = config.get("AUGMENTATION_SEED", 42)

    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed, name="aug_random_flip"),
            layers.RandomRotation(factor=0.10, seed=seed, name="aug_random_rotation"),
            layers.RandomZoom(height_factor=0.10, width_factor=0.10, seed=seed, name="aug_random_zoom"),
            layers.RandomContrast(factor=0.10, seed=seed, name="aug_random_contrast"),
        ],
        name="data_augmentation",
    )

    logger.info("Data augmentation pipeline built (seed=%d).", seed)
    return augmentation


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def build_efficientnet_model(
    config: Config,
    augmentation: tf.keras.Sequential,
) -> Model:
    """Build an EfficientNetB0 transfer-learning model with a custom classifier head.

    The EfficientNetB0 backbone is initialised with ImageNet weights and frozen
    for Phase 1 training.  A reference to the backbone is stored as
    ``model.backbone`` so that :func:`configure_fine_tuning` can unfreeze
    specific layers without re-building the model.

    Architecture (mirrors notebooks/07_transfer_learning.ipynb exactly)::

        Input(224, 224, 3)
        └─ DataAugmentation
           └─ Lambda(preprocess_input)       [efficientnet_preprocessing]
              └─ EfficientNetB0(imagenet, frozen)
                 └─ GlobalAveragePooling2D
                    └─ BatchNormalization
                       └─ Dense(256, relu)   [head_dense]
                          └─ Dropout(0.30)
                             └─ Dense(38, softmax)  [predictions]

    Args:
        config: Configuration dictionary.  Required keys:

            ``"INPUT_SHAPE"`` *(tuple[int, int, int])*
                Spatial dimensions and channels, e.g. ``(224, 224, 3)``.

            ``"NUM_CLASSES"`` *(int)*
                Number of output classes (38 for PlantVillage).

            ``"DROPOUT_RATE"`` *(float)*
                Dropout probability applied before the output layer.

            ``"DENSE_UNITS"`` *(int)*
                Number of units in the intermediate dense layer.

            ``"MODEL_NAME"`` *(str, default "plant_efficientnetb0")*
                Name assigned to the assembled ``tf.keras.Model``.

        augmentation: Data-augmentation Sequential model returned by
            :func:`build_data_augmentation`.

    Returns:
        An uncompiled ``tf.keras.Model`` instance with ``model.backbone``
        pointing to the frozen EfficientNetB0 base.

    Raises:
        KeyError: If any required key is missing from *config*.
    """
    required_keys = {"INPUT_SHAPE", "NUM_CLASSES", "DROPOUT_RATE", "DENSE_UNITS"}
    missing = required_keys - config.keys()
    if missing:
        raise KeyError(
            f"Missing required config key(s): {sorted(missing)}. "
            "Provide all required keys before calling build_efficientnet_model()."
        )

    input_shape: tuple[int, int, int] = config["INPUT_SHAPE"]
    num_classes: int = config["NUM_CLASSES"]
    dropout_rate: float = config["DROPOUT_RATE"]
    dense_units: int = config["DENSE_UNITS"]
    model_name: str = config.get("MODEL_NAME", "plant_efficientnetb0")

    # ── Input ──────────────────────────────────────────────────────────
    inputs = layers.Input(shape=input_shape, name="input_layer")

    # ── Augmentation ───────────────────────────────────────────────────
    x = augmentation(inputs)

    # ── EfficientNet preprocessing (mirrors notebook Lambda layer) ─────
    x = layers.Lambda(preprocess_input, name="efficientnet_preprocessing")(x)

    # ── EfficientNetB0 backbone (frozen for Phase 1) ───────────────────
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )
    backbone.trainable = False
    logger.info(
        "EfficientNetB0 backbone loaded (ImageNet weights, frozen). "
        "Total backbone layers: %d.",
        len(backbone.layers),
    )

    # ── Classification head ────────────────────────────────────────────
    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(dense_units, activation="relu", name="head_dense")(x)
    x = layers.Dropout(rate=dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    # ── Assemble model ─────────────────────────────────────────────────
    model = Model(inputs=inputs, outputs=outputs, name=model_name)

    # Attach backbone reference for use in configure_fine_tuning().
    model.backbone = backbone

    trainable_params, non_trainable_params = count_model_parameters(model)
    logger.info(
        "Model '%s' built — trainable params: %s | non-trainable params: %s.",
        model_name,
        f"{trainable_params:,}",
        f"{non_trainable_params:,}",
    )
    return model


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------


def configure_fine_tuning(
    model: Model,
    config: Config,
) -> None:
    """Unfreeze the last N layers of the EfficientNetB0 backbone for Phase 2.

    Mirrors the notebook fine-tuning logic exactly::

        backbone.trainable = True

        for layer in backbone.layers[:-fine_tune_layers]:
            layer.trainable = False

        for layer in backbone.layers[-fine_tune_layers:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

    All :class:`tf.keras.layers.BatchNormalization` layers in the backbone
    remain frozen throughout fine-tuning to preserve running statistics
    accumulated during ImageNet pre-training.

    The model must be **re-compiled** with a lower learning rate after calling
    this function for the updated ``trainable`` flags to take effect.

    Args:
        model: A ``tf.keras.Model`` returned by :func:`build_efficientnet_model`.
            Must expose a ``model.backbone`` attribute.
        config: Configuration dictionary.  Required key:

            ``"FINE_TUNE_LAYERS"`` *(int, default 30)*
                Number of top backbone layers to unfreeze.  Matches the
                notebook value of 30.

    Raises:
        AttributeError: If *model* does not have a ``backbone`` attribute.
    """
    if not hasattr(model, "backbone"):
        raise AttributeError(
            "model.backbone not found. "
            "Pass a model returned by build_efficientnet_model()."
        )

    fine_tune_layers: int = config.get("FINE_TUNE_LAYERS", 30)
    backbone: tf.keras.Model = model.backbone

    # Step 1 — enable gradient flow through backbone.
    backbone.trainable = True

    # Step 2 — re-freeze all layers before the last fine_tune_layers.
    for layer in backbone.layers[:-fine_tune_layers]:
        layer.trainable = False

    # Step 3 — unfreeze the last N layers, keeping BN frozen.
    frozen_bn_count = 0
    trainable_count = 0
    for layer in backbone.layers[-fine_tune_layers:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            frozen_bn_count += 1
        else:
            layer.trainable = True
            trainable_count += 1

    trainable_params, non_trainable_params = count_model_parameters(model)

    logger.info(
        "Fine-tuning configured — last %d backbone layers targeted: "
        "%d trainable, %d BatchNormalization frozen.",
        fine_tune_layers, trainable_count, frozen_bn_count,
    )
    logger.info(
        "Model parameter summary — trainable: %s | non-trainable: %s.",
        f"{trainable_params:,}", f"{non_trainable_params:,}",
    )
    logger.info(
        "Re-compile the model with a reduced learning rate before resuming training."
    )


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------


def count_model_parameters(model: Model) -> tuple[int, int]:
    """Count the trainable and non-trainable parameters of a Keras model.

    Args:
        model: Any ``tf.keras.Model`` instance.

    Returns:
        A 2-tuple of:

        ``trainable_params`` *(int)*
            Total number of trainable scalar parameters.

        ``non_trainable_params`` *(int)*
            Total number of non-trainable (frozen) scalar parameters.
    """
    trainable_params: int = int(
        sum(tf.size(w).numpy() for w in model.trainable_weights)
    )
    non_trainable_params: int = int(
        sum(tf.size(w).numpy() for w in model.non_trainable_weights)
    )
    return trainable_params, non_trainable_params