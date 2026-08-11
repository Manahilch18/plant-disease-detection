"""
src/evaluation/metrics.py
=========================
Single source of truth for all evaluation metrics in the Plant Disease
Detection System.

This module mirrors the logic implemented in the evaluation notebook.
Every downstream consumer — notebooks, FastAPI, Streamlit, CLI scripts —
imports from here.  No evaluation logic should be duplicated elsewhere.

Responsibilities
----------------
* Compute standard model evaluation metrics (loss, accuracy, top-5, latency).
* Generate per-sample predictions, confidence scores, and class labels.
* Produce classification reports and per-class metric DataFrames.
* Identify misclassified samples and confused class pairs.

This module does NOT:
    - Load datasets or models.
    - Create, save, or display plots.
    - Retrain or fine-tune models.

Author: Plant Disease Detection Project
Python: 3.11+
Style : PEP 8, Google-style docstrings
"""

from __future__ import annotations

import logging
from pyexpat import model
import time
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.testing import verbose
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def compute_evaluation_metrics(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    verbose: int = 0,
) -> dict[str, float]:
    """Evaluate a compiled Keras model on a test dataset.

    Runs ``model.evaluate`` and a separate timed inference pass to measure
    mean per-sample inference latency.

    Args:
        model: A compiled ``tf.keras.Model`` instance.
        test_ds: Batched ``tf.data.Dataset`` of ``(image, label)`` pairs.
        verbose: Verbosity level passed to ``model.evaluate``.  Defaults to
            ``0`` (silent).

    Returns:
        Dictionary with the following keys:

        ``"test_loss"`` *(float)*
            Scalar cross-entropy loss on the test set.

        ``"test_accuracy"`` *(float)*
            Top-1 accuracy on the test set.

        ``"test_top5_accuracy"`` *(float)*
            Top-5 accuracy on the test set.  ``0.0`` when the model was
            compiled without a top-5 metric.

        ``"inference_time_ms"`` *(float)*
            Mean per-sample inference time in milliseconds, measured over
            a single forward pass through *test_ds*.

    Raises:
        ValueError: If *model* has not been compiled.
    """
logger.info("Evaluating model '%s' on test dataset.", model.name)

try:
        results = model.evaluate(
        test_ds,verbose=verbose,
        return_dict=True, )
except Exception as exc:
    raise RuntimeError(
        "Model evaluation failed. Ensure the model is compiled before "
        "calling compute_evaluation_metrics()."
    ) from exc
    test_loss: float = float(results.get("loss", float("nan")))
    test_accuracy: float = float(results.get("accuracy", float("nan")))
    test_top5: float = float(results.get("top_5_accuracy", 0.0))

    # ── Inference latency (single timed forward pass) ──────────────────
    logger.info("Measuring inference latency.")
    sample_count = 0
    t_start = time.perf_counter()
    for images, _ in test_ds:
        model(images, training=False)
        sample_count += images.shape[0]
    t_elapsed = time.perf_counter() - t_start

    inference_time_ms: float = (
        (t_elapsed / sample_count) * 1_000 if sample_count else float("nan")
    )

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_top5_accuracy": test_top5,
        "inference_time_ms": inference_time_ms,
    }

    logger.info(
        "Evaluation complete — loss=%.4f, accuracy=%.4f, top5=%.4f, "
        "inference=%.3f ms/sample.",
        test_loss, test_accuracy, test_top5, inference_time_ms,
    )



# ---------------------------------------------------------------------------
# Prediction generation
# ---------------------------------------------------------------------------


def generate_predictions(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on a dataset and collect probabilities, labels, and confidence.

    Args:
        model: A trained ``tf.keras.Model`` instance.
        test_ds: Batched ``tf.data.Dataset`` of ``(image, label)`` pairs.

    Returns:
        A 4-tuple of NumPy arrays aligned by sample index:

        ``probabilities`` *(shape: [N, num_classes], float32)*
            Softmax output for every sample.

        ``predicted_labels`` *(shape: [N], int64)*
            Argmax of *probabilities* — the predicted class index.

        ``true_labels`` *(shape: [N], int64)*
            Ground-truth integer class labels extracted from *test_ds*.

        ``confidence`` *(shape: [N], float32)*
            Maximum probability value for each prediction.

    Raises:
        ValueError: If *test_ds* yields no samples.
    """
    logger.info("Generating predictions for model '%s'.", model.name)

    probabilities = model.predict(
    test_ds,
    verbose=0,
)

    true_labels = np.concatenate(
    [labels.numpy() for _, labels in test_ds]).astype(np.int64)

    predicted_labels = np.argmax(
    probabilities,
    axis=1,).astype(np.int64)

    confidence = np.max(
    probabilities,
    axis=1,)

    logger.info(
    "Generated predictions for %d samples.",
    len(true_labels),)
    
def build_prediction_dataframe(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    confidence: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    """
    Build a prediction summary DataFrame.

    Args:
        true_labels:
            Ground-truth labels.

        predicted_labels:
            Predicted labels.

        confidence:
            Prediction confidence.

        class_names:
            Ordered class names.

    Returns:
        Prediction DataFrame.
    """

    return pd.DataFrame(
        {
            "True Label": true_labels,
            "Predicted Label": predicted_labels,
            "Actual Class": [
                class_names[i]
                for i in true_labels
            ],
            "Predicted Class": [
                class_names[i]
                for i in predicted_labels
            ],
            "Confidence": confidence,
            "Correct Prediction":
                true_labels == predicted_labels,
        }
    )


# ---------------------------------------------------------------------------
# Classification report
# ---------------------------------------------------------------------------


def generate_classification_report(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: list[str],
    export_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, dict]:
    """Compute a per-class classification report using scikit-learn.

    Args:
        true_labels: 1-D integer array of ground-truth class indices.
        predicted_labels: 1-D integer array of predicted class indices.
        class_names: Ordered list of class name strings where
            ``class_names[i]`` corresponds to integer label ``i``.
        export_path: If provided, the report DataFrame is written to this
            path as a CSV file.  The parent directory is created if it does
            not exist.  Defaults to ``None`` (no export).

    Returns:
        A 2-tuple of:

        ``report_df`` *(pd.DataFrame)*
            DataFrame indexed by class name with columns
            ``precision``, ``recall``, ``f1-score``, ``support``.
            Macro and weighted averages are included as trailing rows.

        ``report_dict`` *(dict)*
            Raw dictionary returned by
            ``sklearn.metrics.classification_report``.

    Raises:
        ValueError: If *true_labels* and *predicted_labels* have different
            lengths.
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError(
            f"Length mismatch: true_labels ({len(true_labels)}) vs "
            f"predicted_labels ({len(predicted_labels)})."
        )

    logger.info("Generating classification report for %d classes.", len(class_names))

    report_dict: dict = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    summary_keys = {"accuracy", "macro avg", "weighted avg"}
    per_class_rows = {
        k: v for k, v in report_dict.items() if k not in summary_keys
    }
    summary_rows = {
        k: v for k, v in report_dict.items()
        if k in summary_keys and isinstance(v, dict)
    }

    report_df = pd.DataFrame(per_class_rows).T
    summary_df = pd.DataFrame(summary_rows).T
    report_df = pd.concat([report_df, summary_df])
    report_df.index.name = "class"
    report_df = report_df.astype(
        {"precision": float, "recall": float, "f1-score": float}
    )

    if export_path is not None:
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(export_path)
        logger.info("Classification report exported to %s.", export_path)

    logger.info("Classification report generated.")
    return report_df, report_dict


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------


def compute_per_class_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    """Build a per-class metrics DataFrame sorted by F1 score (ascending).

    Sorting by F1 ascending surfaces the weakest classes at the top, making
    it straightforward to identify where the model needs improvement.

    Args:
        true_labels: 1-D integer array of ground-truth class indices.
        predicted_labels: 1-D integer array of predicted class indices.
        class_names: Ordered list of class name strings.

    Returns:
        DataFrame with one row per class and the following columns:

        ``Class`` *(str)*
            Human-readable class name.

        ``Precision`` *(float)*
            Fraction of positive predictions that are correct.

        ``Recall`` *(float)*
            Fraction of actual positives that were predicted correctly.

        ``F1 Score`` *(float)*
            Harmonic mean of precision and recall.

        ``Support`` *(int)*
            Number of ground-truth samples for this class.

        ``Accuracy`` *(float)*
            Per-class accuracy (equivalent to recall for multi-class).

        Rows are sorted by ``F1 Score`` ascending and the index is reset.

    Raises:
        ValueError: If *true_labels* and *predicted_labels* have different
            lengths.
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError(
            f"Length mismatch: true_labels ({len(true_labels)}) vs "
            f"predicted_labels ({len(predicted_labels)})."
        )

    logger.info("Computing per-class metrics for %d classes.", len(class_names))

    report_dict: dict = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    rows: list[dict] = []
    for class_name in class_names:
        stats = report_dict.get(class_name, {})
        precision = float(stats.get("precision", 0.0))
        recall = float(stats.get("recall", 0.0))
        f1 = float(stats.get("f1-score", 0.0))
        support = int(stats.get("support", 0))
        rows.append({
            "Class": class_name,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Support": support,
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("F1 Score", ascending=True)
        .reset_index(drop=True)
    )

    logger.info("Per-class metrics computed — mean F1=%.4f.", df["F1 Score"].mean())
    return df


# ---------------------------------------------------------------------------
# Misclassification analysis
# ---------------------------------------------------------------------------


def find_misclassified_indices(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> np.ndarray:
    """Return the indices of samples where prediction does not match truth.

    Args:
        true_labels: 1-D integer array of ground-truth class indices.
        predicted_labels: 1-D integer array of predicted class indices.

    Returns:
        1-D integer NumPy array of sample indices where
        ``predicted_labels[i] != true_labels[i]``.

    Raises:
        ValueError: If *true_labels* and *predicted_labels* have different
            lengths.
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError(
            f"Length mismatch: true_labels ({len(true_labels)}) vs "
            f"predicted_labels ({len(predicted_labels)})."
        )

    indices: np.ndarray = np.where(true_labels != predicted_labels)[0]
    total = len(true_labels)
    error_rate = len(indices) / total if total else 0.0

    logger.info(
        "Misclassified samples: %d / %d (error rate=%.4f).",
        len(indices), total, error_rate,
    )
    return indices


def find_confused_class_pairs(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: list[str],
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """Identify the most common class confusion pairs among misclassified samples.

    Only samples where the prediction is incorrect are considered.  Results
    are sorted by confusion count in descending order.

    Args:
        true_labels: 1-D integer array of ground-truth class indices.
        predicted_labels: 1-D integer array of predicted class indices.
        class_names: Ordered list of class name strings.
        top_n: If provided, only the *top_n* most frequent confusion pairs
            are returned.  Defaults to ``None`` (all pairs).

    Returns:
        DataFrame with the following columns:

        ``Actual`` *(str)*
            Ground-truth class name.

        ``Predicted`` *(str)*
            Incorrectly predicted class name.

        ``Count`` *(int)*
            Number of times this confusion pair occurred.

        Rows are sorted by ``Count`` descending.

    Raises:
        ValueError: If *true_labels* and *predicted_labels* have different
            lengths.
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError(
            f"Length mismatch: true_labels ({len(true_labels)}) vs "
            f"predicted_labels ({len(predicted_labels)})."
        )

    misclassified_idx = find_misclassified_indices(true_labels, predicted_labels)

    if len(misclassified_idx) == 0:
        logger.info("No misclassifications found — returning empty DataFrame.")
        return pd.DataFrame(columns=["Actual", "Predicted", "Count"])

    actual_names = [class_names[true_labels[i]] for i in misclassified_idx]
    predicted_names = [class_names[predicted_labels[i]] for i in misclassified_idx]

    confusion_df = (
        pd.DataFrame({"Actual": actual_names, "Predicted": predicted_names})
        .groupby(["Actual", "Predicted"])
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )

    if top_n is not None:
        confusion_df = confusion_df.head(top_n)

    logger.info(
        "Confused class pairs computed — %d unique confusion pairs.",
        len(confusion_df),
    )
    return confusion_df
