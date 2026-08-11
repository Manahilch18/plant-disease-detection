"""
src/training/train.py
=====================
Training pipeline for the Plant Disease Detection System.

This module is the single source of truth for model compilation and
``model.fit`` execution.  mirrors the baseline training notebook.
exactly — hyperparameters, optimiser choice, loss function, and metrics are
intentionally frozen here so that notebook experiments and production runs
are identical.

This module does NOT:
    - Load or preprocess datasets (→ ``src/data/preprocess.py``)
    - Define model architecture (→ ``src/models/custom_cnn.py``)
    - Define callbacks (→ ``src/training/callbacks.py``)
    - Evaluate, visualise, or run inference

Usage::

    from src.models.custom_cnn import build_custom_cnn
    from src.training.train import train_model

    config = {
        "LEARNING_RATE": 1e-3,
        "EPOCHS": 50,
        "ARTIFACT_PATH": Path("artifacts/training"),
        "LOG_PATH": Path("logs"),
    }

    model = build_custom_cnn()
    history = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        config=config,
        class_weights=class_weights,
    )

Author: Plant Disease Detection Project
Python: 3.11+
Style : PEP 8, Google-style docstrings
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import tensorflow as tf

 # noqa: F401 — re-exported for convenience
from src.training.callbacks import create_callbacks

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    config: dict[str, Any],
    class_weights: dict[int, float] | None = None
) -> tf.keras.callbacks.History:
    """Compile and train a Keras model using the standard project pipeline.

    Compilation, callback creation, and ``model.fit`` are all handled here so
    that every training run — notebook or production — uses the exact same
    configuration.

    Compilation details:
        * Optimiser : ``Adam(learning_rate=config["LEARNING_RATE"])``
        * Loss      : ``SparseCategoricalCrossentropy()``
        * Metrics   : ``["accuracy", SparseTopKCategoricalAccuracy(k=5)]``

    Callbacks are created via :func:`src.training.callbacks.create_callbacks`
    and written to the paths specified in *config*.

    Args:
        model: An uncompiled ``tf.keras.Model`` instance (e.g. from
            :func:`src.models.custom_cnn.build_custom_cnn`).
        train_ds: Batched and prefetched training ``tf.data.Dataset`` of
            ``(image, label)`` pairs.
        val_ds: Batched validation ``tf.data.Dataset`` of
            ``(image, label)`` pairs.
        config: Training configuration dictionary.  Required keys:

            ``"LEARNING_RATE"`` *(float)*
                Initial learning rate for the Adam optimiser.

            ``"EPOCHS"`` *(int)*
                Maximum number of training epochs.  Early stopping may
                halt training before this limit is reached.

            ``"ARTIFACT_PATH"`` *(Path | str)*
                Directory where the best model checkpoint and CSV history
                are saved.

            ``"LOG_PATH"`` *(Path | str)*
                Root directory for TensorBoard event files.

            Optional keys forwarded to :func:`create_callbacks`:

            ``"MODEL_FILENAME"`` *(str, default "best_model.keras")*
                Filename for the ``ModelCheckpoint`` artefact.

            ``"HISTORY_FILENAME"`` *(str, default "history.csv")*
                Filename for the ``CSVLogger`` output.

        class_weights: Per-class weight mapping ``{class_index: weight}``
            as produced by
            :func:`src.data.preprocess.compute_class_weights`.  Passed
            directly to ``model.fit`` to handle label imbalance.

    Returns:
        The ``tf.keras.callbacks.History`` object returned by ``model.fit``.
        Access per-epoch metrics via ``history.history``.

    Raises:
        KeyError: If any required key is missing from *config*.
        ValueError: If *train_ds* or *val_ds* are not ``tf.data.Dataset``
            instances.
    """
    # ------------------------------------------------------------------
    # Validate required config keys up front — fail fast, fail clearly.
    # ------------------------------------------------------------------
    required_keys = {"LEARNING_RATE", "EPOCHS", "ARTIFACT_PATH", "LOG_PATH"}
    missing = required_keys - config.keys()
    if missing:
        raise KeyError(
            f"Missing required config key(s): {sorted(missing)}. "
            "Ensure all keys are present before calling train_model()."
        )

    if not isinstance(train_ds, tf.data.Dataset):
        raise ValueError(
            f"train_ds must be a tf.data.Dataset, got {type(train_ds).__name__}."
        )
    if not isinstance(val_ds, tf.data.Dataset):
        raise ValueError(
            f"val_ds must be a tf.data.Dataset, got {type(val_ds).__name__}."
        )

    artifact_path = Path(config["ARTIFACT_PATH"])
    log_path = Path(config["LOG_PATH"])
    learning_rate: float = config["LEARNING_RATE"]
    epochs: int = config["EPOCHS"]
    model_filename: str = config.get("MODEL_FILENAME", "best_model.keras")
    history_filename: str = config.get("HISTORY_FILENAME", "history.csv")

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    logger.info(
        "Compiling model '%s' — lr=%.2e, loss=SparseCategoricalCrossentropy, "
        "metrics=[accuracy, top-5 accuracy].",
        model.name, learning_rate,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy"),
        ],
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    logger.info("Creating callbacks...")
    callbacks = create_callbacks(
        artifact_path=artifact_path,
        log_path=log_path,
        model_filename=model_filename,
        history_filename=history_filename,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info(
        "Training started — model='%s', max_epochs=%d, "
        "artifact_path=%s, log_path=%s.",
        model.name, epochs, artifact_path, log_path,
    )

    t_start = time.perf_counter()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    t_elapsed = time.perf_counter() - t_start

    # ------------------------------------------------------------------
    # Post-training summary
    # ------------------------------------------------------------------
    val_acc_history: list[float] = history.history.get("val_accuracy", [])
    val_loss_history: list[float] = history.history.get("val_loss", [])

    best_val_acc: float = max(val_acc_history) if val_acc_history else float("nan")
    best_val_loss: float = min(val_loss_history) if val_loss_history else float("nan")
    epochs_run: int = len(val_acc_history)

    hours, remainder = divmod(int(t_elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info("Training completed.")
    logger.info("  Epochs run          : %d / %d", epochs_run, epochs)
    logger.info("  Total training time : %dh %02dm %02ds", hours, minutes, seconds)
    logger.info("  Best val_accuracy   : %.4f", best_val_acc)
    logger.info("  Best val_loss       : %.4f", best_val_loss)

    return history


# ---------------------------------------------------------------------------
# Entry-point placeholder
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Training is not triggered automatically.
    # Wire up preprocess.py, build_custom_cnn(), and train_model() here
    # or invoke this module from a top-level run script (e.g. run_training.py).
    logger.info(
        "src/training/train.py loaded.  "
        "Call train_model() from a run script or notebook to begin training."
    )
