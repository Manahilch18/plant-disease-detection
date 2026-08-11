"""
src/training/callbacks.py
=========================
Callback factory for the Plant Disease Detection System.

This module is the single source of truth for all Keras training callbacks.
It mirrors the callback configuration defined in the training notebooks so
that notebook experiments and production training runs behave identically.

Callbacks produced
------------------
* EarlyStopping      — halts training when ``val_loss`` stops improving.
* ReduceLROnPlateau  — decays the learning rate on ``val_loss`` plateau.
* ModelCheckpoint    — saves the best model checkpoint by ``val_accuracy``.
* TensorBoard        — writes event files for TensorBoard visualisation.
* CSVLogger          — appends per-epoch metrics to a CSV file.
* TerminateOnNaN     — aborts training immediately on NaN loss.

Usage::

    from src.training.callbacks import create_callbacks

    callbacks = create_callbacks(
        artifact_path=Path("artifacts/training"),
        log_path=Path("logs"),
    )
    model.fit(train_ds, validation_data=val_ds, callbacks=callbacks, ...)

Author: Plant Disease Detection Project
Python: 3.11+
Style : PEP 8, Google-style docstrings
"""

from __future__ import annotations

import logging
from pathlib import Path

import tensorflow as tf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Callback factory
# ---------------------------------------------------------------------------


def create_callbacks(
    artifact_path: Path,
    log_path: Path,
    model_filename: str = "best_model.keras",
    history_filename: str = "history.csv",
) -> list[tf.keras.callbacks.Callback]:
    """Build and return the standard callback stack for model training.

    All callback parameters are frozen to match the notebook configuration.
    Do not alter them without updating the corresponding notebook cell.

    Directory layout created by this function::

        artifact_path/
        ├── best_model.keras   ← ModelCheckpoint (or custom model_filename)
        └── history.csv        ← CSVLogger       (or custom history_filename)

        log_path/              ← TensorBoard event files

    Args:
        artifact_path: Directory where the best model checkpoint and the
            CSV training history are written.  Created automatically if it
            does not exist.
        log_path: Root directory for TensorBoard event files.  Created
            automatically if it does not exist.
        model_filename: Filename for the ``ModelCheckpoint`` artefact.
            Defaults to ``"best_model.keras"``.
        history_filename: Filename for the ``CSVLogger`` output.
            Defaults to ``"history.csv"``.

    Returns:
        An ordered list of ``tf.keras.callbacks.Callback`` instances ready to
        be passed directly to ``model.fit(..., callbacks=callbacks)``.

    Raises:
        OSError: If either directory cannot be created due to permission or
            filesystem errors.
    """
    artifact_path = Path(artifact_path)
    log_path = Path(log_path)

    artifact_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = artifact_path / model_filename
    history_path = artifact_path / history_filename

    logger.info("Initialising callbacks.")
    logger.info("  Checkpoint path : %s", checkpoint_path)
    logger.info("  TensorBoard logs: %s", log_path)
    logger.info("  CSV history path: %s", history_path)

    # ------------------------------------------------------------------
    # 1. EarlyStopping
    # ------------------------------------------------------------------
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    # ------------------------------------------------------------------
    # 2. ReduceLROnPlateau
    # ------------------------------------------------------------------
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # 3. ModelCheckpoint
    # ------------------------------------------------------------------
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # 4. TensorBoard
    # ------------------------------------------------------------------
    from datetime import datetime

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

     
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=str(log_path / run_name),histogram_freq=1,write_graph=True,)
    # ------------------------------------------------------------------
    # 5. CSVLogger
    # ------------------------------------------------------------------
    csv_logger = tf.keras.callbacks.CSVLogger(
        filename=str(history_path),
        append=False,
    )

    # ------------------------------------------------------------------
    # 6. TerminateOnNaN
    # ------------------------------------------------------------------
    terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()

    callbacks: list[tf.keras.callbacks.Callback] = [
        early_stopping,
        reduce_lr,
        model_checkpoint,
        tensorboard,
        csv_logger,
        terminate_on_nan,
    ]

    logger.info("Created %d callbacks.", len(callbacks))
    return callbacks
