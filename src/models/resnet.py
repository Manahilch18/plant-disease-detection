"""
src/models/resnet.py
====================
ResNet50 transfer-learning model for the Plant Disease Detection System.

This module is the single source of truth for the ResNet50 architecture and
mirrors the implementation defined in the ResNet50 transfer-learning notebook
exactly.  Every hyperparameter, layer order, and regularisation choice is
intentionally frozen here so that notebook experiments and production training
runs are identical.

Training is structured in two phases:

Phase 1 — Feature extraction
    The ResNet50 backbone is fully frozen.  Only the classification head is
    trained.  Compile with ``config.learning_rate`` (1e-3).

Phase 2 — Fine-tuning
    The last ``config.fine_tune_layers`` backbone layers are unfrozen
    (excluding BatchNormalization layers, which remain frozen throughout).
    Re-compile with ``config.fine_tune_learning_rate`` (1e-5).

Architecture::

    Input(224, 224, 3)
    └─ Rescaling(1./255)
       └─ DataAugmentation [RandomFlip | RandomRotation | RandomZoom]
          └─ ResNet50(imagenet, include_top=False, frozen)
             └─ GlobalAveragePooling2D
                └─ BatchNormalization
                   └─ Dropout(0.30)
                      └─ Dense(512, relu, L2)
                         └─ BatchNormalization
                            └─ Dropout(0.30)
                               └─ Dense(38, softmax)

Author: Plant Disease Detection Project
Python: 3.11+
Style : PEP 8, Google-style docstrings
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import StringIO
from typing import Any

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L2

# ---------------------------------------------------------------------------
# Logging — library module: do NOT call basicConfig here.
# The training script / notebook controls the root logger configuration.
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_IMAGE_SIZE: int = 32
_MAX_IMAGE_SIZE: int = 1024
_MIN_CLASSES: int = 2
_MAX_DROPOUT: float = 1.0
_MIN_DENSE_UNITS: int = 1
_SUPPORTED_WEIGHTS: frozenset = frozenset({"imagenet", None})

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResNetConfig:
    """Immutable configuration for the ResNet50 model — synchronized with notebook.

    All hyperparameters match the completed ResNet50 transfer-learning notebook
    exactly.  Do not change defaults without updating the notebook accordingly.

    Attributes:
        image_size: Spatial resolution as ``(height, width)``.
        num_classes: Number of output classes (38 for PlantVillage).
        dropout_rate: Dropout probability used in both head dropout layers.
        dense_units: Units in the intermediate Dense layer.
        l2_weight_decay: L2 regularisation coefficient for the Dense layer.
        learning_rate: Adam learning rate for Phase 1 (head-only training).
        fine_tune_learning_rate: Adam learning rate for Phase 2 (fine-tuning).
        fine_tune_layers: Number of top backbone layers unfrozen in Phase 2.
        weights: Pre-trained weights for the backbone. ``"imagenet"`` or ``None``.
        include_top: Always ``False`` for transfer learning.
        pooling: Pooling mode when ``include_top=False``. ``None`` delegates
            pooling to the custom head.
        trainable: Initial backbone trainable state. ``False`` for Phase 1.
        seed: Random seed for augmentation layers.
    """

    image_size: tuple[int, int] = (224, 224)
    num_classes: int = 38
    dropout_rate: float = 0.30
    dense_units: int = 512
    l2_weight_decay: float = 1e-4
    learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-5
    fine_tune_layers: int = 30
    weights: str | None = "imagenet"
    include_top: bool = False
    pooling: str | None = None
    trainable: bool = False
    seed: int = 42

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Return the full input shape including the channel dimension.

        Returns:
            3-tuple ``(height, width, 3)``.
        """
        return (*self.image_size, 3)


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------


def _validate_config(config: ResNetConfig) -> None:
    """Validate all fields of *config* and raise on the first violation.

    Args:
        config: Configuration instance to validate.

    Raises:
        TypeError: If *config* is not a :class:`ResNetConfig` instance.
        ValueError: If any field is outside its valid range.
    """
    if not isinstance(config, ResNetConfig):
        raise TypeError(
            f"config must be a ResNetConfig instance, got {type(config).__name__}."
        )
    h, w = config.image_size
    if not (_MIN_IMAGE_SIZE <= h <= _MAX_IMAGE_SIZE and _MIN_IMAGE_SIZE <= w <= _MAX_IMAGE_SIZE):
        raise ValueError(
            f"image_size {config.image_size} out of bounds. "
            f"Each dimension must be in [{_MIN_IMAGE_SIZE}, {_MAX_IMAGE_SIZE}]."
        )
    if config.num_classes < _MIN_CLASSES:
        raise ValueError(
            f"num_classes must be >= {_MIN_CLASSES}, got {config.num_classes}."
        )
    if not (0.0 <= config.dropout_rate < _MAX_DROPOUT):
        raise ValueError(
            f"dropout_rate must be in [0.0, 1.0), got {config.dropout_rate}."
        )
    if config.dense_units < _MIN_DENSE_UNITS:
        raise ValueError(
            f"dense_units must be >= {_MIN_DENSE_UNITS}, got {config.dense_units}."
        )
    if config.learning_rate <= 0.0:
        raise ValueError(
            f"learning_rate must be > 0, got {config.learning_rate}."
        )
    if config.fine_tune_learning_rate <= 0.0:
        raise ValueError(
            f"fine_tune_learning_rate must be > 0, got {config.fine_tune_learning_rate}."
        )
    if config.weights not in _SUPPORTED_WEIGHTS:
        raise ValueError(
            f"weights='{config.weights}' is not supported. "
            f"Choose from: {_SUPPORTED_WEIGHTS}."
        )
    logger.debug("ResNetConfig validated successfully.")


# ---------------------------------------------------------------------------
# Internal layer builders
# ---------------------------------------------------------------------------


def _build_rescaling_layer() -> layers.Rescaling:
    """Return a Rescaling layer that maps [0, 255] pixels to [0, 1].

    Returns:
        ``tf.keras.layers.Rescaling`` instance.

    Notes:
        Do NOT apply additional normalisation in the data pipeline when using
        this model.  The backbone receives [0, 1] values from this layer.
    """
    return layers.Rescaling(1.0 / 255.0, name="rescaling")


def _build_augmentation_block(config: ResNetConfig) -> tf.keras.Sequential:
    """Build the notebook-synchronised augmentation pipeline.

    Augmentation operations (exactly as in notebook):
        1. ``RandomFlip("horizontal")``
        2. ``RandomRotation(0.10)``
        3. ``RandomZoom(0.10)``

    Args:
        config: ResNet configuration (seed used for reproducibility).

    Returns:
        ``tf.keras.Sequential`` augmentation pipeline.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip(
                "horizontal",
                seed=config.seed,
                name="aug_random_flip",
            ),
            layers.RandomRotation(
                0.10,
                seed=config.seed,
                name="aug_random_rotation",
            ),
            layers.RandomZoom(
                0.10,
                seed=config.seed,
                name="aug_random_zoom",
            ),
        ],
        name="data_augmentation",
    )


def _build_resnet50_backbone(
    config: ResNetConfig,
    x: tf.Tensor,
) -> tf.keras.Model:
    """Instantiate the ResNet50 backbone and set its initial trainable state.

    Args:
        config: ResNet configuration.
        x: Preprocessed input tensor wired into the backbone.

    Returns:
        ResNet50 model with ``trainable`` set per ``config.trainable``.
    """
    backbone = tf.keras.applications.ResNet50(
        include_top=config.include_top,
        weights=config.weights,
        input_tensor=x,
        pooling=config.pooling,
    )
    backbone.trainable = config.trainable
    return backbone


def _build_classifier_head(
    x: tf.Tensor,
    config: ResNetConfig,
) -> tf.Tensor:
    """Attach the classification head to the backbone output tensor.

    Head architecture (matches notebook exactly)::

        GlobalAveragePooling2D
        BatchNormalization
        Dropout(0.30)
        Dense(512, relu, L2)
        BatchNormalization
        Dropout(0.30)
        Dense(38, softmax)

    Args:
        x: Output tensor from the backbone.
        config: ResNet configuration.

    Returns:
        Final output tensor with shape ``[batch, num_classes]``.
    """
    regularizer = L2(config.l2_weight_decay)

    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = layers.BatchNormalization(name="head_bn_1")(x)
    x = layers.Dropout(config.dropout_rate, name="head_dropout_1")(x)
    x = layers.Dense(
        config.dense_units,
        activation="relu",
        kernel_regularizer=regularizer,
        name="head_dense",
    )(x)
    x = layers.BatchNormalization(name="head_bn_2")(x)
    x = layers.Dropout(config.dropout_rate, name="head_dropout_2")(x)
    return layers.Dense(
        config.num_classes,
        activation="softmax",
        name="predictions",
    )(x)


# ---------------------------------------------------------------------------
# Public model builder
# ---------------------------------------------------------------------------


def build_resnet_model(
    config: ResNetConfig,
    model_name: str = "ResNet50_TransferLearning",
) -> Model:
    """Build the ResNet50 transfer-learning model for plant disease classification.

    The model is returned **uncompiled**.  Call :func:`compile_resnet_model`
    separately to mirror the notebook's explicit compile step.

    Args:
        config: ResNet configuration dataclass.
        model_name: Name assigned to the assembled ``tf.keras.Model``.
            Defaults to ``"ResNet50_TransferLearning"`` (matches notebook).

    Returns:
        An uncompiled ``tf.keras.Model`` with ``model.backbone`` pointing
        to the ResNet50 base model.

    Raises:
        TypeError: If *config* is not a :class:`ResNetConfig`.
        ValueError: If any config field fails validation.

    Examples:
        >>> config = ResNetConfig()
        >>> model = build_resnet_model(config)
        >>> model.name
        'ResNet50_TransferLearning'
    """
    _validate_config(config)
    logger.info(
        "Building '%s' — input_shape=%s, num_classes=%d.",
        model_name, config.input_shape, config.num_classes,
    )

    inputs = layers.Input(shape=config.input_shape, name="input_layer")

    x = _build_rescaling_layer()(inputs)
    x = _build_augmentation_block(config)(x)

    backbone = _build_resnet50_backbone(config, x)
    logger.info(
        "ResNet50 backbone loaded — weights='%s', trainable=%s, layers=%d.",
        config.weights, config.trainable, len(backbone.layers),
    )

    outputs = _build_classifier_head(backbone.output, config)

    model = Model(inputs=inputs, outputs=outputs, name=model_name)

    # Attach backbone reference; _get_backbone() provides a safe fallback.
    model.backbone = backbone

    trainable_p = count_trainable_parameters(model)
    non_trainable_p = count_non_trainable_parameters(model)
    logger.info(
        "Model '%s' built — trainable: %s | non-trainable: %s.",
        model_name, f"{trainable_p:,}", f"{non_trainable_p:,}",
    )
    return model


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def compile_resnet_model(
    model: Model,
    config: ResNetConfig,
    learning_rate: float | None = None,
) -> None:
    """Compile *model* with Adam, SparseCategoricalCrossentropy, and notebook metrics.

    Metrics compiled:
        * ``SparseCategoricalAccuracy`` (name: ``"accuracy"``)
        * ``SparseTopKCategoricalAccuracy`` k=5 (name: ``"top_5_accuracy"``)

    Args:
        model: An uncompiled or previously compiled ``tf.keras.Model``.
        config: ResNet configuration (provides the default learning rate).
        learning_rate: Override learning rate.  When ``None``, uses
            ``config.learning_rate``.  Pass ``config.fine_tune_learning_rate``
            when re-compiling for Phase 2.

    Raises:
        TypeError: If *model* is not a ``tf.keras.Model``.

    Examples:
        >>> compile_resnet_model(model, config)
        >>> compile_resnet_model(model, config, learning_rate=config.fine_tune_learning_rate)
    """
    if not isinstance(model, Model):
        raise TypeError(
            f"model must be a tf.keras.Model, got {type(model).__name__}."
        )

    lr = learning_rate if learning_rate is not None else config.learning_rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy"),
        ],
    )
    logger.info(
        "Model '%s' compiled — Adam(lr=%.2e), SparseCategoricalCrossentropy.",
        model.name, lr,
    )


# ---------------------------------------------------------------------------
# Backbone resolution helper
# ---------------------------------------------------------------------------


def _get_backbone(model: Model) -> tf.keras.Model:
    """Return the ResNet50 backbone from the assembled model.

    Tries ``model.backbone`` first (set during :func:`build_resnet_model`).
    Falls back to scanning model layers by name, which is robust after
    saving and reloading the model from disk.

    Args:
        model: Assembled ``tf.keras.Model``.

    Returns:
        The ResNet50 backbone ``tf.keras.Model``.

    Raises:
        AttributeError: If the backbone cannot be located.
    """
    if hasattr(model, "backbone"):
        return model.backbone

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "resnet50" in layer.name.lower():
            return layer

    raise AttributeError(
        "ResNet50 backbone could not be located in the model. "
        "Pass a model returned by build_resnet_model(), or ensure the "
        "backbone layer name contains 'resnet50'."
    )


# ---------------------------------------------------------------------------
# Freeze / unfreeze
# ---------------------------------------------------------------------------


def freeze_backbone(model: Model) -> None:
    """Freeze the entire ResNet50 backbone for Phase 1 head-only training.

    Args:
        model: A ``tf.keras.Model`` returned by :func:`build_resnet_model`.

    Raises:
        AttributeError: If the backbone cannot be located.

    Notes:
        Re-compile the model after freezing to propagate the updated
        ``trainable`` flags to the optimiser.

    Examples:
        >>> freeze_backbone(model)
        >>> compile_resnet_model(model, config)
    """
    backbone = _get_backbone(model)
    backbone.trainable = False
    logger.info(
        "ResNet50 backbone frozen — trainable: %s | non-trainable: %s.",
        f"{count_trainable_parameters(model):,}",
        f"{count_non_trainable_parameters(model):,}",
    )


def unfreeze_backbone(model: Model) -> None:
    """Unfreeze the entire ResNet50 backbone.

    Use :func:`fine_tune_model` for partial unfreezing of the last N layers.

    Args:
        model: A ``tf.keras.Model`` returned by :func:`build_resnet_model`.

    Raises:
        AttributeError: If the backbone cannot be located.

    Notes:
        Re-compile the model after unfreezing.

    Examples:
        >>> unfreeze_backbone(model)
        >>> compile_resnet_model(model, config, learning_rate=config.fine_tune_learning_rate)
    """
    backbone = _get_backbone(model)
    backbone.trainable = True
    logger.info(
        "ResNet50 backbone unfrozen — trainable: %s | non-trainable: %s.",
        f"{count_trainable_parameters(model):,}",
        f"{count_non_trainable_parameters(model):,}",
    )


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------


def fine_tune_model(
    model: Model,
    fine_tune_layers: int = 30,
) -> None:
    """Unfreeze the last *fine_tune_layers* backbone layers for Phase 2.

    All BatchNormalization layers remain frozen throughout fine-tuning to
    preserve running statistics accumulated during ImageNet pre-training.

    Mirrors the notebook fine-tuning logic exactly::

        backbone.trainable = True
        for layer in backbone.layers:
            layer.trainable = False
        for layer in backbone.layers[-fine_tune_layers:]:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = True

    Args:
        model: A ``tf.keras.Model`` returned by :func:`build_resnet_model`.
        fine_tune_layers: Number of top backbone layers to unfreeze.
            Defaults to ``30`` (notebook default).

    Raises:
        AttributeError: If the backbone cannot be located.
        ValueError: If *fine_tune_layers* is < 1 or exceeds backbone depth.

    Notes:
        **Re-compile** the model with ``config.fine_tune_learning_rate``
        after calling this function.

    Examples:
        >>> fine_tune_model(model, fine_tune_layers=30)
        >>> compile_resnet_model(model, config, learning_rate=config.fine_tune_learning_rate)
    """
    backbone = _get_backbone(model)

    if fine_tune_layers < 1:
        raise ValueError(
            f"fine_tune_layers must be >= 1, got {fine_tune_layers}."
        )
    if fine_tune_layers > len(backbone.layers):
        raise ValueError(
            f"fine_tune_layers={fine_tune_layers} exceeds total backbone "
            f"layers={len(backbone.layers)}."
        )

    # Step 1 — enable gradient flow through the backbone.
    backbone.trainable = True

    # Step 2 — freeze all backbone layers.
    for layer in backbone.layers:
        layer.trainable = False

    # Step 3 — unfreeze the last N layers; keep BatchNormalization frozen.
    frozen_bn = 0
    trainable_count = 0
    for layer in backbone.layers[-fine_tune_layers:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            frozen_bn += 1
        else:
            layer.trainable = True
            trainable_count += 1

    logger.info(
        "Fine-tuning configured — last %d backbone layers: "
        "%d trainable, %d BatchNormalization kept frozen.",
        fine_tune_layers, trainable_count, frozen_bn,
    )
    logger.info(
        "Model parameters — trainable: %s | non-trainable: %s.",
        f"{count_trainable_parameters(model):,}",
        f"{count_non_trainable_parameters(model):,}",
    )
    logger.info(
        "Re-compile the model with fine_tune_learning_rate before resuming training."
    )


# ---------------------------------------------------------------------------
# Parameter utilities
# ---------------------------------------------------------------------------


def count_trainable_parameters(model: Model) -> int:
    """Return the total number of trainable scalar parameters.

    Args:
        model: Any ``tf.keras.Model`` instance.

    Returns:
        Integer count of trainable parameters.

    Examples:
        >>> n = count_trainable_parameters(model)
        >>> print(f"Trainable: {n:,}")
    """
    return int(
        sum(tf.size(weight).numpy() for weight in model.trainable_weights)
    )


def count_non_trainable_parameters(model: Model) -> int:
    """Return the total number of non-trainable (frozen) scalar parameters.

    Args:
        model: Any ``tf.keras.Model`` instance.

    Returns:
        Integer count of non-trainable parameters.

    Examples:
        >>> n = count_non_trainable_parameters(model)
        >>> print(f"Non-trainable: {n:,}")
    """
    return int(
        sum(tf.size(weight).numpy() for weight in model.non_trainable_weights)
    )


# ---------------------------------------------------------------------------
# Model summary & size utilities
# ---------------------------------------------------------------------------


def model_summary_to_string(model: Model, line_length: int = 100) -> str:
    """Capture ``model.summary()`` as a plain-text string.

    Args:
        model: Any ``tf.keras.Model`` instance.
        line_length: Character width of the summary table.  Defaults to ``100``.

    Returns:
        The full model summary as a single multi-line string.

    Examples:
        >>> summary = model_summary_to_string(model)
        >>> Path("artifacts/model_summary.txt").write_text(summary)
    """
    buffer = StringIO()
    model.summary(
        print_fn=lambda line: buffer.write(line + "\n"),
        line_length=line_length,
        show_trainable=True,
    )
    return buffer.getvalue()


def get_model_size_mb(model: Model) -> float:
    """Estimate the in-memory size of all model weights in megabytes.

    Args:
        model: Any ``tf.keras.Model`` instance.

    Returns:
        Approximate model weight memory in MB, rounded to 3 decimal places.

    Notes:
        Measures parameter memory only.  Activation memory during a forward
        pass is not included.

    Examples:
        >>> size = get_model_size_mb(model)
        >>> print(f"Model size: {size:.2f} MB")
    """
    all_weights = model.trainable_weights + model.non_trainable_weights
    total_bytes = sum(weight.numpy().nbytes for weight in all_weights)
    size_mb = round(total_bytes / (1024 ** 2), 3)
    logger.info("Model '%s' size: %.3f MB.", model.name, size_mb)
    return size_mb


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_model_output(
    model: Model,
    config: ResNetConfig,
    batch_size: int = 2,
) -> dict[str, Any]:
    """Run a forward pass with synthetic data to verify output shape.

    Args:
        model: A compiled or uncompiled ``tf.keras.Model``.
        config: ResNet configuration used to derive the expected output shape.
        batch_size: Number of synthetic samples.  Defaults to ``2``.

    Returns:
        Dictionary with keys:

        ``"input_shape"`` *(tuple)* — shape of the synthetic input.
        ``"output_shape"`` *(tuple)* — actual model output shape.
        ``"expected_output_shape"`` *(tuple)* — expected ``(batch, classes)``.
        ``"passed"`` *(bool)* — ``True`` when shapes match.

    Raises:
        RuntimeError: If the output shape does not match the expected shape.

    Examples:
        >>> result = verify_model_output(model, config)
        >>> assert result["passed"]
    """
    h, w = config.image_size
    synthetic_input = tf.random.uniform(
        shape=(batch_size, h, w, 3),
        minval=0.0,
        maxval=255.0,
        dtype=tf.float32,
    )

    logger.info(
        "Verifying model output — input_shape=%s.",
        tuple(synthetic_input.shape),
    )
    output = model(synthetic_input, training=False)

    expected = (batch_size, config.num_classes)
    actual = tuple(output.shape)
    passed = actual == expected

    if not passed:
        raise RuntimeError(
            f"Output shape mismatch — expected {expected}, got {actual}. "
            "Check num_classes and classifier head configuration."
        )

    logger.info(
        "Verification passed — output_shape=%s matches expected=%s.",
        actual, expected,
    )
    return {
        "input_shape": tuple(synthetic_input.shape),
        "output_shape": actual,
        "expected_output_shape": expected,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Local verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _config = ResNetConfig()
    _model = build_resnet_model(_config)
    compile_resnet_model(_model, _config)

    _result = verify_model_output(_model, _config)
    print(f"Verification passed : {_result['passed']}")
    print(f"Output shape        : {_result['output_shape']}")
    print(f"Trainable params    : {count_trainable_parameters(_model):,}")
    print(f"Non-trainable params: {count_non_trainable_parameters(_model):,}")
    print(f"Model size          : {get_model_size_mb(_model):.2f} MB")
