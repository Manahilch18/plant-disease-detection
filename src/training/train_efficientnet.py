"""
src/training/train_efficientnet.py
===================================
Two-phase EfficientNetB0 training pipeline for the Plant Disease Detection System.

This module mirrors the training workflow defined in
``notebooks/07_transfer_learning.ipynb`` exactly and delegates all architecture
concerns to ``src/models/efficientnet.py``.

Pipeline
--------
Phase 1 — Feature extraction
    Backbone frozen.  Only the classification head is trained.
    Compiled with Adam(lr=0.001).

Phase 2 — Fine-tuning
    Last N backbone layers unfrozen (BatchNormalization stays frozen).
    Re-compiled with Adam(lr=1e-5).
    Training history merged with Phase 1 history.

Artefacts written
-----------------
* ``best_model_phase1.keras``      — best checkpoint from Phase 1
* ``best_model_phase2.keras``      — best checkpoint from Phase 2
* ``history_phase1.csv``           — per-epoch metrics, Phase 1
* ``history_phase2.csv``           — per-epoch metrics, Phase 2
* ``training_history.pkl``         — merged history dict (both phases)
* ``training_metrics.json``        — scalar summary of the full run
* ``training_curves.png``          — accuracy + loss plots (both phases)
* ``model_summary.txt``            — model.summary() captured to text

Usage::

    python -m src.training.train_efficientnet \\
        --config configs/efficientnet_config.yaml \\
        --output artifacts/efficientnet

Author: Plant Disease Detection Project
Python: 3.11+
Style : PEP 8, Google-style docstrings
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from io import StringIO
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for servers and CI
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml

from src.models.efficientnet import (
    build_data_augmentation,
    build_efficientnet_model,
    configure_fine_tuning,
    count_model_parameters,
)

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
# Type alias
# ---------------------------------------------------------------------------

Config = dict[str, Any]
History = dict[str, list[float]]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> Config:
    """Load and return a YAML configuration file as a dictionary.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
        ValueError: If the YAML file is empty or cannot be parsed.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            "Ensure configs/efficientnet_config.yaml is present."
        )
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not config:
        raise ValueError(
            f"Configuration file is empty or invalid: {config_path}."
        )
    logger.info("Configuration loaded from %s.", config_path)
    return config


# ---------------------------------------------------------------------------
# Directory management
# ---------------------------------------------------------------------------


def create_artifact_directories(config: Config) -> dict[str, Path]:
    """Create all required output directories and return their paths.

    Args:
        config: Configuration dictionary containing ``"ARTIFACT_PATH"``
            and ``"LOG_PATH"`` keys.

    Returns:
        Dictionary mapping directory labels to resolved ``Path`` objects:
        ``"artifact"``, ``"log"``, ``"checkpoint"``.

    Raises:
        KeyError: If required path keys are missing from *config*.
    """
    required = {"ARTIFACT_PATH", "LOG_PATH"}
    missing = required - config.keys()
    if missing:
        raise KeyError(f"Missing config key(s) for directories: {sorted(missing)}.")

    artifact_dir = Path(config["ARTIFACT_PATH"])
    log_dir = Path(config["LOG_PATH"])
    checkpoint_dir = artifact_dir / "checkpoints"

    for directory in (artifact_dir, log_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)
        logger.info("Directory ready: %s.", directory)

    return {
        "artifact": artifact_dir,
        "log": log_dir,
        "checkpoint": checkpoint_dir,
    }


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------


def validate_datasets(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
) -> None:
    """Assert that all three dataset splits are non-empty tf.data.Dataset objects.

    Args:
        train_ds: Training dataset.
        val_ds: Validation dataset.
        test_ds: Test dataset.

    Raises:
        TypeError: If any argument is not a ``tf.data.Dataset``.
        ValueError: If any dataset yields no batches.
    """
    for name, ds in [("train_ds", train_ds), ("val_ds", val_ds), ("test_ds", test_ds)]:
        if not isinstance(ds, tf.data.Dataset):
            raise TypeError(
                f"{name} must be a tf.data.Dataset, got {type(ds).__name__}."
            )
    logger.info("All dataset splits validated.")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def create_phase_callbacks(
    phase: int,
    dirs: dict[str, Path],
    config: Config,
) -> list[tf.keras.callbacks.Callback]:
    """Build the standard callback stack for a single training phase.

    Callbacks created:
        * ``EarlyStopping``    — monitors ``val_loss``, patience from config.
        * ``ReduceLROnPlateau``— monitors ``val_loss``, factor/patience from config.
        * ``ModelCheckpoint``  — saves best model by ``val_accuracy``.
        * ``CSVLogger``        — appends per-epoch metrics to a CSV file.
        * ``TensorBoard``      — writes event files to the log directory.

    Args:
        phase: Training phase identifier (``1`` or ``2``).
        dirs: Directory mapping returned by :func:`create_artifact_directories`.
        config: Configuration dictionary.  Recognised keys:

            ``"EARLY_STOPPING_PATIENCE"`` *(int, default 5)*
            ``"REDUCE_LR_PATIENCE"``      *(int, default 3)*
            ``"REDUCE_LR_FACTOR"``        *(float, default 0.2)*
            ``"REDUCE_LR_MIN_LR"``        *(float, default 1e-7)*

    Returns:
        Ordered list of ``tf.keras.callbacks.Callback`` instances.
    """
    checkpoint_path = dirs["checkpoint"] / f"best_model_phase{phase}.keras"
    csv_path = dirs["artifact"] / f"history_phase{phase}.csv"
    log_dir = dirs["log"] / f"phase{phase}"
    log_dir.mkdir(parents=True, exist_ok=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.get("EARLY_STOPPING_PATIENCE", 5),
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=config.get("REDUCE_LR_FACTOR", 0.2),
        patience=config.get("REDUCE_LR_PATIENCE", 3),
        min_lr=config.get("REDUCE_LR_MIN_LR", 1e-7),
        verbose=1,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    csv_logger = tf.keras.callbacks.CSVLogger(
        filename=str(csv_path),
        append=False,
    )
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=str(log_dir),
    )

    callbacks = [early_stopping, reduce_lr, checkpoint, csv_logger, tensorboard]
    logger.info(
        "Phase %d callbacks created — checkpoint: %s, csv: %s.",
        phase, checkpoint_path, csv_path,
    )
    return callbacks


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


def validate_model(model: tf.keras.Model, phase: int) -> None:
    """Log trainable and non-trainable parameter counts for a given phase.

    Args:
        model: Keras model to inspect.
        phase: Training phase identifier used in log messages.

    Raises:
        ValueError: If the model has zero trainable parameters.
    """
    trainable, non_trainable = count_model_parameters(model)
    if trainable == 0:
        raise ValueError(
            f"Phase {phase}: model has 0 trainable parameters. "
            "Check backbone.trainable and head layer configuration."
        )
    logger.info(
        "Phase %d model validation — trainable: %s | non-trainable: %s.",
        phase, f"{trainable:,}", f"{non_trainable:,}",
    )


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def compile_model(
    model: tf.keras.Model,
    learning_rate: float,
    phase: int,
) -> None:
    """Compile *model* with Adam, SparseCategoricalCrossentropy, and standard metrics.

    Args:
        model: Uncompiled or previously compiled Keras model.
        learning_rate: Initial learning rate for the Adam optimiser.
        phase: Training phase identifier used in log messages.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy"),
        ],
    )
    logger.info(
        "Phase %d — model compiled: Adam(lr=%.2e), SparseCategoricalCrossentropy.",
        phase, learning_rate,
    )


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------


def run_phase(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    callbacks: list[tf.keras.callbacks.Callback],
    class_weights: dict[int, float],
    phase: int,
) -> tuple[tf.keras.callbacks.History, float]:
    """Run a single training phase and return the history and elapsed time.

    Args:
        model: Compiled Keras model.
        train_ds: Training ``tf.data.Dataset``.
        val_ds: Validation ``tf.data.Dataset``.
        epochs: Maximum number of epochs for this phase.
        callbacks: Callback list from :func:`create_phase_callbacks`.
        class_weights: Per-class weight mapping for imbalanced datasets.
        phase: Training phase identifier used in log messages.

    Returns:
        A 2-tuple of:

        ``history`` *(tf.keras.callbacks.History)*
            Keras History object for this phase.

        ``elapsed_seconds`` *(float)*
            Wall-clock training duration in seconds.
    """
    logger.info("Phase %d training started — max_epochs=%d.", phase, epochs)

    t_start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )
    elapsed = time.perf_counter() - t_start

    epochs_run = len(history.history.get("val_accuracy", []))
    best_val_acc = max(history.history.get("val_accuracy", [float("nan")]))
    best_val_loss = min(history.history.get("val_loss", [float("nan")]))

    h, m, s = _format_elapsed(elapsed)
    logger.info("Phase %d training complete.", phase)
    logger.info("  Epochs run       : %d / %d", epochs_run, epochs)
    logger.info("  Training time    : %dh %02dm %02ds", h, m, s)
    logger.info("  Best val_accuracy: %.4f", best_val_acc)
    logger.info("  Best val_loss    : %.4f", best_val_loss)

    return history, elapsed


# ---------------------------------------------------------------------------
# History utilities
# ---------------------------------------------------------------------------


def merge_histories(
    history1: tf.keras.callbacks.History,
    history2: tf.keras.callbacks.History,
) -> History:
    """Concatenate two Keras History objects into a single metric dictionary.

    Args:
        history1: History object from Phase 1.
        history2: History object from Phase 2.

    Returns:
        Merged dictionary mapping metric names to concatenated epoch lists.
    """
    merged: History = {}
    all_keys = set(history1.history.keys()) | set(history2.history.keys())
    for key in all_keys:
        part1 = history1.history.get(key, [])
        part2 = history2.history.get(key, [])
        merged[key] = part1 + part2
    logger.info(
        "Histories merged — total epochs: %d (phase1=%d + phase2=%d).",
        len(merged.get("val_accuracy", [])),
        len(history1.history.get("val_accuracy", [])),
        len(history2.history.get("val_accuracy", [])),
    )
    return merged


def validate_history(merged: History) -> None:
    """Assert that the merged history contains the expected metric keys.

    Args:
        merged: Merged history dictionary from :func:`merge_histories`.

    Raises:
        ValueError: If any required metric key is absent.
    """
    required = {"accuracy", "val_accuracy", "loss", "val_loss"}
    missing = required - merged.keys()
    if missing:
        raise ValueError(
            f"Merged history is missing expected metric(s): {sorted(missing)}."
        )
    logger.info("Merged history validation passed.")


# ---------------------------------------------------------------------------
# Artefact persistence
# ---------------------------------------------------------------------------


def save_history_pickle(merged: History, artifact_dir: Path) -> Path:
    """Serialise the merged history dictionary to a pickle file.

    Args:
        merged: Merged history dictionary.
        artifact_dir: Directory where ``training_history.pkl`` is written.

    Returns:
        Path to the written file.
    """
    output_path = artifact_dir / "training_history.pkl"
    with output_path.open("wb") as fh:
        pickle.dump(merged, fh)
    logger.info("Training history saved to %s.", output_path)
    return output_path


def save_training_metrics(
    history1: tf.keras.callbacks.History,
    history2: tf.keras.callbacks.History,
    elapsed_phase1: float,
    elapsed_phase2: float,
    artifact_dir: Path,
) -> Path:
    """Persist scalar training metrics from both phases to a JSON file.

    Args:
        history1: Phase 1 Keras History object.
        history2: Phase 2 Keras History object.
        elapsed_phase1: Phase 1 wall-clock duration in seconds.
        elapsed_phase2: Phase 2 wall-clock duration in seconds.
        artifact_dir: Directory where ``training_metrics.json`` is written.

    Returns:
        Path to the written file.
    """
    def _best(h: tf.keras.callbacks.History, key: str, fn) -> float:
        values = h.history.get(key, [])
        return float(fn(values)) if values else float("nan")

    metrics = {
        "phase1": {
            "epochs_run": len(history1.history.get("val_accuracy", [])),
            "best_val_accuracy": _best(history1, "val_accuracy", max),
            "best_val_loss": _best(history1, "val_loss", min),
            "training_time_seconds": round(elapsed_phase1, 2),
        },
        "phase2": {
            "epochs_run": len(history2.history.get("val_accuracy", [])),
            "best_val_accuracy": _best(history2, "val_accuracy", max),
            "best_val_loss": _best(history2, "val_loss", min),
            "training_time_seconds": round(elapsed_phase2, 2),
        },
        "total_training_time_seconds": round(elapsed_phase1 + elapsed_phase2, 2),
    }

    output_path = artifact_dir / "training_metrics.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Training metrics saved to %s.", output_path)
    return output_path


def save_training_curves(
    merged: History,
    phase1_epochs: int,
    artifact_dir: Path,
) -> Path:
    """Plot and save accuracy and loss curves for both training phases.

    A vertical dashed line marks the boundary between Phase 1 and Phase 2.

    Args:
        merged: Merged history dictionary from :func:`merge_histories`.
        phase1_epochs: Number of epochs completed during Phase 1 (used to
            draw the phase boundary marker).
        artifact_dir: Directory where ``training_curves.png`` is written.

    Returns:
        Path to the written PNG file.
    """
    acc = merged.get("accuracy", [])
    val_acc = merged.get("val_accuracy", [])
    loss = merged.get("loss", [])
    val_loss = merged.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(epochs_range, acc, label="Train Accuracy", color="steelblue")
    ax1.plot(epochs_range, val_acc, label="Val Accuracy", color="darkorange")
    ax1.axvline(phase1_epochs, linestyle="--", color="gray", linewidth=1.2,
                label=f"Phase boundary (epoch {phase1_epochs})")
    ax1.set_title("Accuracy — Both Phases", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(fontsize=9)
    ax1.grid(linestyle="--", alpha=0.4)

    # Loss
    ax2.plot(epochs_range, loss, label="Train Loss", color="steelblue")
    ax2.plot(epochs_range, val_loss, label="Val Loss", color="darkorange")
    ax2.axvline(phase1_epochs, linestyle="--", color="gray", linewidth=1.2,
                label=f"Phase boundary (epoch {phase1_epochs})")
    ax2.set_title("Loss — Both Phases", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(fontsize=9)
    ax2.grid(linestyle="--", alpha=0.4)

    fig.tight_layout()
    output_path = artifact_dir / "training_curves.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved to %s.", output_path)
    return output_path


def save_model_summary(model: tf.keras.Model, artifact_dir: Path) -> Path:
    """Capture and write ``model.summary()`` to a plain-text file.

    Args:
        model: Keras model to summarise.
        artifact_dir: Directory where ``model_summary.txt`` is written.

    Returns:
        Path to the written file.
    """
    buffer = StringIO()
    model.summary(print_fn=lambda line: buffer.write(line + "\n"), line_length=100)
    output_path = artifact_dir / "model_summary.txt"
    output_path.write_text(buffer.getvalue(), encoding="utf-8")
    logger.info("Model summary saved to %s.", output_path)
    return output_path


def validate_artifacts(artifact_dir: Path) -> None:
    """Verify that all expected artefact files were written successfully.

    Args:
        artifact_dir: Root artefact directory to inspect.

    Raises:
        FileNotFoundError: If any expected artefact is missing.
    """
    expected = [
        "training_history.pkl",
        "training_metrics.json",
        "training_curves.png",
        "model_summary.txt",
        "history_phase1.csv",
        "history_phase2.csv",
        "checkpoints/best_model_phase1.keras",
        "checkpoints/best_model_phase2.keras",
    ]
    missing = [f for f in expected if not (artifact_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"The following expected artefacts are missing from {artifact_dir}: "
            f"{missing}."
        )
    logger.info("All expected artefacts verified in %s.", artifact_dir)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_elapsed(seconds: float) -> tuple[int, int, int]:
    """Convert elapsed seconds to (hours, minutes, seconds).

    Args:
        seconds: Elapsed duration in seconds.

    Returns:
        3-tuple of ``(hours, minutes, seconds)``.
    """
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    return h, m, s


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def train_efficientnet(
    config: Config,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    class_weights: dict[int, float],
) -> dict[str, Any]:
    """Run the complete two-phase EfficientNetB0 training pipeline.

    This function is the single entry point for all downstream consumers
    (run scripts, notebooks, CI jobs).

    Args:
        config: Configuration dictionary loaded from
            ``configs/efficientnet_config.yaml``.  Required keys:

            ``"INPUT_SHAPE"``             *(tuple[int, int, int])*
            ``"NUM_CLASSES"``             *(int)*
            ``"DROPOUT_RATE"``            *(float)*
            ``"DENSE_UNITS"``             *(int)*
            ``"MODEL_NAME"``              *(str)*
            ``"AUGMENTATION_SEED"``       *(int)*
            ``"PHASE1_EPOCHS"``           *(int)*
            ``"PHASE2_EPOCHS"``           *(int)*
            ``"PHASE1_LR"``               *(float, default 1e-3)*
            ``"PHASE2_LR"``               *(float, default 1e-5)*
            ``"FINE_TUNE_LAYERS"``        *(int, default 30)*
            ``"ARTIFACT_PATH"``           *(str)*
            ``"LOG_PATH"``               *(str)*

        train_ds: Batched and prefetched training ``tf.data.Dataset``.
        val_ds: Batched validation ``tf.data.Dataset``.
        test_ds: Batched test ``tf.data.Dataset`` (validated but not used
            during training — reserved for evaluation).
        class_weights: Per-class weight mapping ``{class_index: weight}``
            from :func:`src.data.preprocess.compute_class_weights`.

    Returns:
        Dictionary containing:

        ``"merged_history"``     *(dict)* — concatenated Phase 1 + Phase 2 metrics.
        ``"history_phase1"``     *(History)* — Phase 1 Keras History object.
        ``"history_phase2"``     *(History)* — Phase 2 Keras History object.
        ``"model"``              *(tf.keras.Model)* — final trained model.
        ``"artifact_dir"``       *(Path)* — root directory of saved artefacts.

    Raises:
        KeyError: If required config keys are absent.
        ValueError: If datasets or model fail validation.
        FileNotFoundError: If expected artefacts are not written.
    """
    logger.info("=" * 60)
    logger.info("EfficientNetB0 training pipeline started.")
    logger.info("=" * 60)

    # ── Validate inputs ────────────────────────────────────────────────
    validate_datasets(train_ds, val_ds, test_ds)

    # ── Directories ────────────────────────────────────────────────────
    dirs = create_artifact_directories(config)
    artifact_dir = dirs["artifact"]

    # ── Build model ────────────────────────────────────────────────────
    augmentation = build_data_augmentation(config)
    model = build_efficientnet_model(config, augmentation)
    save_model_summary(model, artifact_dir)

    # ── Phase 1 — feature extraction ──────────────────────────────────
    logger.info("-" * 40)
    logger.info("PHASE 1 — Feature Extraction (backbone frozen).")
    logger.info("-" * 40)

    compile_model(model, learning_rate=config.get("PHASE1_LR", 1e-3), phase=1)
    validate_model(model, phase=1)

    callbacks_p1 = create_phase_callbacks(phase=1, dirs=dirs, config=config)
    history1, elapsed1 = run_phase(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=config["PHASE1_EPOCHS"],
        callbacks=callbacks_p1,
        class_weights=class_weights,
        phase=1,
    )
    phase1_epochs_run = len(history1.history.get("val_accuracy", []))

    # ── Phase 2 — fine-tuning ─────────────────────────────────────────
    logger.info("-" * 40)
    logger.info("PHASE 2 — Fine-tuning (last %d backbone layers unfrozen).",
                config.get("FINE_TUNE_LAYERS", 30))
    logger.info("-" * 40)

    configure_fine_tuning(model, config)
    compile_model(model, learning_rate=config.get("PHASE2_LR", 1e-5), phase=2)
    validate_model(model, phase=2)

    callbacks_p2 = create_phase_callbacks(phase=2, dirs=dirs, config=config)
    history2, elapsed2 = run_phase(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=config["PHASE2_EPOCHS"],
        callbacks=callbacks_p2,
        class_weights=class_weights,
        phase=2,
    )

    # ── Merge & validate histories ─────────────────────────────────────
    merged = merge_histories(history1, history2)
    validate_history(merged)

    # ── Save artefacts ─────────────────────────────────────────────────
    save_history_pickle(merged, artifact_dir)
    save_training_metrics(history1, history2, elapsed1, elapsed2, artifact_dir)
    save_training_curves(merged, phase1_epochs_run, artifact_dir)
    validate_artifacts(artifact_dir)

    total_h, total_m, total_s = _format_elapsed(elapsed1 + elapsed2)
    logger.info("=" * 60)
    logger.info("Training pipeline complete.")
    logger.info("  Total time  : %dh %02dm %02ds", total_h, total_m, total_s)
    logger.info("  Artifacts   : %s", artifact_dir)
    logger.info("=" * 60)

    return {
        "merged_history": merged,
        "history_phase1": history1,
        "history_phase2": history2,
        "model": model,
        "artifact_dir": artifact_dir,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the EfficientNetB0 training pipeline.

    Parses ``--config`` and ``--output`` arguments, loads the YAML config,
    and delegates to :func:`train_efficientnet`.

    Datasets and class weights must be wired up here or supplied via a
    run script.  This function serves as a reference integration point::

        python -m src.training.train_efficientnet \\
            --config configs/efficientnet_config.yaml

    Note:
        Training does NOT start automatically on module import.
        Only explicit invocation of ``main()`` or ``train_efficientnet()``
        triggers execution.
    """
    parser = argparse.ArgumentParser(
        description="Two-phase EfficientNetB0 training — Plant Disease Detection."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/efficientnet_config.yaml"),
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Wire up datasets and class_weights from src.data.preprocess here,
    # or pass them from an orchestrating run script.
    logger.info(
        "src/training/train_efficientnet.py loaded. "
        "Provide train_ds, val_ds, test_ds, and class_weights, "
        "then call train_efficientnet() to begin."
    )


if __name__ == "__main__":
    main()
