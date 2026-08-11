"""
src/evaluation/plots.py
=======================
Single source of truth for all evaluation visualisations in the Plant Disease
Detection System.

This module mirrors the plotting logic defined in the evaluation notebook.
Every downstream consumer — notebooks, Streamlit, CLI scripts — imports from
here.  No plotting logic should be duplicated elsewhere.

Responsibilities
----------------
* Confusion matrix (raw counts and row-normalised).
* Prediction confidence distributions.
* Correct vs. incorrect confidence comparison.
* Misclassified sample grid.
* Per-class F1, precision, and recall horizontal bar charts.

This module does NOT:
    - Compute metrics  (→ ``src/evaluation/metrics.py``)
    - Load datasets or models.
    - Display figures interactively (``plt.show`` is never called).

All functions return a ``matplotlib.figure.Figure`` object and, when
*save_path* is supplied, write a 300-dpi PNG to disk.

Author: Plant Disease Detection Project
Python: 3.11+
Style : PEP 8, Google-style docstrings
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DPI: int = 300
_FONT_TITLE: int = 14
_FONT_AXIS: int = 11
_FONT_TICK: int = 8

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_figure(fig: plt.Figure, save_path: Optional[Path]) -> None:
    """Persist *fig* to disk at *save_path* if a path is provided.

    Args:
        fig: Matplotlib figure to save.
        save_path: Destination file path.  The parent directory is created
            automatically.  When ``None`` the figure is not saved.
    """
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    logger.info("Figure saved to %s.", save_path)


def _style_axes(
    ax: plt.Axes,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    """Apply consistent title and axis-label styling to *ax*.

    Args:
        ax: Matplotlib axes to style.
        title: Figure / subplot title.
        xlabel: X-axis label.  Empty string hides the label.
        ylabel: Y-axis label.  Empty string hides the label.
    """
    ax.set_title(title, fontsize=_FONT_TITLE, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=_FONT_AXIS)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=_FONT_AXIS)
    ax.tick_params(labelsize=_FONT_TICK)


def _build_horizontal_bar(
    values: Sequence[float],
    labels: Sequence[str],
    title: str,
    xlabel: str,
    color: str,
    save_path: Optional[Path],
    figsize: tuple[float, float] = (10, 14),
) -> plt.Figure:
    """Render a horizontal bar chart for per-class scalar metrics.

    Args:
        values: Metric value for each class (aligned with *labels*).
        labels: Class name for each bar.
        title: Chart title.
        xlabel: X-axis label (e.g. ``"F1 Score"``).
        color: Bar fill colour.
        save_path: Optional path to write the PNG.
        figsize: Figure width and height in inches.

    Returns:
        Rendered ``matplotlib.figure.Figure``.
    """
    values = np.asarray(values, dtype=float)
    y_pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(y_pos, values, color=color, edgecolor="white", height=0.7)

    # Annotate bar ends.
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=6,
            color="dimgray",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=6)
    max_value = float(np.nanmax(values))
    ax.set_xlim(0.0, min(values.max() * 1.15 + 0.01, 1.05))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    _style_axes(ax, title=title, xlabel=xlabel, ylabel="Class")

    fig.tight_layout()
    _save_figure(fig, save_path)
    # Prevent memory accumulation in long-running processes.
    fig.canvas.draw_idle() 
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: list[str],
    save_path: Optional[Path] = None,
    figsize: tuple[float, float] = (20, 18),
    cmap: str = "Blues",
) -> plt.Figure:
    """Render a raw-count confusion matrix as a seaborn heatmap.

    Args:
        confusion_matrix: 2-D integer array of shape ``[num_classes, num_classes]``
            where ``confusion_matrix[i, j]`` is the number of samples with
            true class ``i`` predicted as class ``j``.
        class_names: Ordered list of class name strings.
        save_path: Optional path to write the PNG file.
        figsize: Figure width and height in inches.  Defaults to ``(20, 18)``.
        cmap: Matplotlib / seaborn colour map name.  Defaults to ``"Blues"``.

    Returns:
        Rendered ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If *confusion_matrix* is not square or its size does not
            match ``len(class_names)``.
    """
    if confusion_matrix.ndim != 2 or confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError(
            f"confusion_matrix must be a square 2-D array, "
            f"got shape {confusion_matrix.shape}."
        )
    if confusion_matrix.shape[0] != len(class_names):
        raise ValueError(
            f"confusion_matrix size ({confusion_matrix.shape[0]}) does not match "
            f"len(class_names) ({len(class_names)})."
        )

    logger.info("Plotting raw confusion matrix (%d classes).", len(class_names))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.3,
        linecolor="lightgray",
        ax=ax,
        annot_kws={"size": 6},
        cbar_kws={"shrink": 0.75},
    )
    _style_axes(ax, title="Confusion Matrix", xlabel="Predicted Label", ylabel="True Label")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_normalized_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: list[str],
    save_path: Optional[Path] = None,
    figsize: tuple[float, float] = (20, 18),
    cmap: str = "Blues",
) -> plt.Figure:
    """Render a row-normalised confusion matrix as a seaborn heatmap.

    Each row is divided by its sum so that cell values represent the fraction
    of true-class samples predicted as each class (i.e. recall per cell).

    Args:
        confusion_matrix: 2-D integer array of shape ``[num_classes, num_classes]``.
        class_names: Ordered list of class name strings.
        save_path: Optional path to write the PNG file.
        figsize: Figure width and height in inches.  Defaults to ``(20, 18)``.
        cmap: Matplotlib / seaborn colour map name.  Defaults to ``"Blues"``.

    Returns:
        Rendered ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If *confusion_matrix* is not square or its size does not
            match ``len(class_names)``.
    """
    if confusion_matrix.ndim != 2 or confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError(
            f"confusion_matrix must be a square 2-D array, "
            f"got shape {confusion_matrix.shape}."
        )
    if confusion_matrix.shape[0] != len(class_names):
        raise ValueError(
            f"confusion_matrix size ({confusion_matrix.shape[0]}) does not match "
            f"len(class_names) ({len(class_names)})."
        )

    logger.info("Plotting normalised confusion matrix (%d classes).", len(class_names))

    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    # Guard against zero-support classes.
    normalised = confusion_matrix.astype(np.float32)

    np.divide(confusion_matrix,row_sums,out=normalised, where=row_sums != 0,)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        normalised,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.3,
        linecolor="lightgray",
        ax=ax,
        annot_kws={"size": 6},
        cbar_kws={"shrink": 0.75},
    )
    _style_axes(
        ax,
        title="Normalised Confusion Matrix (Row-wise Recall)",
        xlabel="Predicted Label",
        ylabel="True Label",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Confidence distributions
# ---------------------------------------------------------------------------


def plot_confidence_distribution(
    confidence: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: tuple[float, float] = (9, 5),
    bins: int = 50,
    color: str = "steelblue",
) -> plt.Figure:
    """Plot the distribution of prediction confidence scores.

    Args:
        confidence: 1-D float array of per-sample maximum softmax probabilities.
        save_path: Optional path to write the PNG file.
        figsize: Figure width and height in inches.  Defaults to ``(9, 5)``.
        bins: Number of histogram bins.  Defaults to ``50``.
        color: Bar fill colour.  Defaults to ``"steelblue"``.

    Returns:
        Rendered ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If *confidence* is empty.
    """
    if len(confidence) == 0:
        raise ValueError("confidence array is empty.")

    logger.info(
        "Plotting confidence distribution — mean=%.4f, median=%.4f.",
        float(np.mean(confidence)), float(np.median(confidence)),
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(confidence, bins=bins, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(
        float(np.mean(confidence)),
        color="crimson", linestyle="--", linewidth=1.5,
        label=f"Mean: {np.mean(confidence):.3f}",
    )
    ax.axvline(
        float(np.median(confidence)),
        color="darkorange", linestyle=":", linewidth=1.5,
        label=f"Median: {np.median(confidence):.3f}",
    )
    ax.legend(fontsize=_FONT_TICK)
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _style_axes(
        ax,
        title="Prediction Confidence Distribution",
        xlabel="Confidence (Max Softmax Probability)",
        ylabel="Sample Count",
    )

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_correct_vs_incorrect_confidence(
    correct_confidence: np.ndarray,
    incorrect_confidence: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: tuple[float, float] = (9, 5),
    bins: int = 50,
) -> plt.Figure:
    """Overlay confidence histograms for correct and incorrect predictions.

    Args:
        correct_confidence: 1-D float array of confidence scores for correctly
            classified samples.
        incorrect_confidence: 1-D float array of confidence scores for
            misclassified samples.
        save_path: Optional path to write the PNG file.
        figsize: Figure width and height in inches.  Defaults to ``(9, 5)``.
        bins: Number of histogram bins.  Defaults to ``50``.

    Returns:
        Rendered ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If both arrays are empty.
    """
    if len(correct_confidence) == 0 and len(incorrect_confidence) == 0:
        raise ValueError(
            "Both correct_confidence and incorrect_confidence are empty."
        )

    logger.info(
        "Plotting correct vs incorrect confidence — correct_n=%d, incorrect_n=%d.",
        len(correct_confidence), len(incorrect_confidence),
    )

    fig, ax = plt.subplots(figsize=figsize)

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ax.set_ylim(bottom=0)

    if len(correct_confidence) > 0:
        ax.hist(
            correct_confidence,
            bins=bin_edges,
            alpha=0.65,
            color="seagreen",
            edgecolor="white",
            label=f"Correct  (n={len(correct_confidence):,})",
        )
    if len(incorrect_confidence) > 0:
        ax.hist(
            incorrect_confidence,
            bins=bin_edges,
            alpha=0.65,
            color="crimson",
            edgecolor="white",
            label=f"Incorrect (n={len(incorrect_confidence):,})",
        )

    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=_FONT_TICK)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _style_axes(
        ax,
        title="Confidence Distribution: Correct vs. Incorrect Predictions",
        xlabel="Confidence (Max Softmax Probability)",
        ylabel="Sample Count",
    )

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Misclassified sample grid
# ---------------------------------------------------------------------------


def plot_misclassified_samples(
    images: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    confidence: np.ndarray,
    class_names: list[str],
    save_path: Optional[Path] = None,
    grid_rows: int = 5,
    grid_cols: int = 5,
    figsize: tuple[float, float] = (18, 18),
) -> plt.Figure:
    """Display a grid of misclassified samples with true/predicted annotations.

    Renders up to ``grid_rows × grid_cols`` misclassified images.  Each cell
    shows the image with the true class, predicted class, and confidence score
    in the title.  True-label text is coloured green; predicted-label text is
    coloured red.

    Args:
        images: Float array of shape ``[N, H, W, C]`` containing raw or
            normalised pixel values.  Values are clipped to ``[0, 1]`` before
            rendering.
        true_labels: 1-D integer array of ground-truth class indices.
        predicted_labels: 1-D integer array of predicted class indices.
        confidence: 1-D float array of per-sample maximum softmax probabilities.
        class_names: Ordered list of class name strings.
        save_path: Optional path to write the PNG file.
        grid_rows: Number of grid rows.  Defaults to ``5``.
        grid_cols: Number of grid columns.  Defaults to ``5``.
        figsize: Figure width and height in inches.  Defaults to ``(18, 18)``.

    Returns:
        Rendered ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If there are no misclassified samples to display.
    """
    if not (
    len(images)
    == len(true_labels)
    == len(predicted_labels)
    == len(confidence)):
        raise ValueError(
        "images, labels and confidence arrays must have identical length.")
    misclassified_mask = true_labels != predicted_labels
    mis_idx = np.where(misclassified_mask)[0]

    if len(mis_idx) == 0:
        raise ValueError(
            "No misclassified samples found. "
            "Verify that true_labels and predicted_labels contain errors."
        )

    n_display = min(grid_rows * grid_cols, len(mis_idx))
    selected = mis_idx[:n_display]

    logger.info(
        "Plotting %d misclassified samples (grid %d×%d).",
        n_display, grid_rows, grid_cols,
    )

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=figsize)
    axes_flat = axes.flatten()

    for plot_idx, sample_idx in enumerate(selected):
        ax = axes_flat[plot_idx]
        img = images[sample_idx]

        if img.max() > 1.0:
            img = img / 255.0

        img = np.clip(img, 0.0, 1.0)
        ax.imshow(img)
        ax.axis("off")

        true_name = class_names[true_labels[sample_idx]]
        pred_name = class_names[predicted_labels[sample_idx]]
        conf = confidence[sample_idx]

        ax.set_title(
            f"True: {true_name}\nPred: {pred_name}\nConf: {conf:.2f}",
            fontsize=7,
            color="crimson",
            pad=3,
        )

    # Hide any unused axes.
    for ax in axes_flat[n_display:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Misclassified Samples  (showing {n_display} of {len(mis_idx):,})",
        fontsize=_FONT_TITLE,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Per-class bar charts
# ---------------------------------------------------------------------------


def plot_per_class_f1(
    per_class_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple[float, float] = (10, 14),
    color: str = "steelblue",
) -> plt.Figure:
    """Render a horizontal bar chart of per-class F1 scores.

    Args:
        per_class_df: DataFrame produced by
            :func:`src.evaluation.metrics.compute_per_class_metrics`.
            Must contain ``"Class"`` and ``"F1 Score"`` columns.
        save_path: Optional path to write the PNG file.
        figsize: Figure width and height in inches.
        color: Bar fill colour.

    Returns:
        Rendered ``matplotlib.figure.Figure``.

    Raises:
        KeyError: If *per_class_df* is missing required columns.
    """
    _validate_per_class_df(per_class_df, required_cols=["Class", "F1 Score"])
    logger.info("Plotting per-class F1 scores.")
    return _build_horizontal_bar(
        values=per_class_df["F1 Score"].values,
        labels=per_class_df["Class"].values,
        title="Per-Class F1 Score",
        xlabel="F1 Score",
        color=color,
        save_path=save_path,
        figsize=figsize,
    )


def plot_per_class_precision(
    per_class_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple[float, float] = (10, 14),
    color: str = "darkcyan",
) -> plt.Figure:
    """Render a horizontal bar chart of per-class precision scores.

    Args:
        per_class_df: DataFrame produced by
            :func:`src.evaluation.metrics.compute_per_class_metrics`.
            Must contain ``"Class"`` and ``"Precision"`` columns.
        save_path: Optional path to write the PNG file.
        figsize: Figure width and height in inches.
        color: Bar fill colour.

    Returns:
        Rendered ``matplotlib.figure.Figure``.

    Raises:
        KeyError: If *per_class_df* is missing required columns.
    """
    _validate_per_class_df(per_class_df, required_cols=["Class", "Precision"])
    logger.info("Plotting per-class precision scores.")
    return _build_horizontal_bar(
        values=per_class_df["Precision"].values,
        labels=per_class_df["Class"].values,
        title="Per-Class Precision",
        xlabel="Precision",
        color=color,
        save_path=save_path,
        figsize=figsize,
    )


def plot_per_class_recall(
    per_class_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple[float, float] = (10, 14),
    color: str = "darkorchid",
) -> plt.Figure:
    """Render a horizontal bar chart of per-class recall scores.

    Args:
        per_class_df: DataFrame produced by
            :func:`src.evaluation.metrics.compute_per_class_metrics`.
            Must contain ``"Class"`` and ``"Recall"`` columns.
        save_path: Optional path to write the PNG file.
        figsize: Figure width and height in inches.
        color: Bar fill colour.

    Returns:
        Rendered ``matplotlib.figure.Figure``.

    Raises:
        KeyError: If *per_class_df* is missing required columns.
    """
    _validate_per_class_df(per_class_df, required_cols=["Class", "Recall"])
    logger.info("Plotting per-class recall scores.")
    return _build_horizontal_bar(
        values=per_class_df["Recall"].values,
        labels=per_class_df["Class"].values,
        title="Per-Class Recall",
        xlabel="Recall",
        color=color,
        save_path=save_path,
        figsize=figsize,
    )


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def _validate_per_class_df(df: pd.DataFrame, required_cols: list[str]) -> None:
    """Assert that *df* contains all *required_cols*.

    Args:
        df: DataFrame to validate.
        required_cols: Column names that must be present.

    Raises:
        KeyError: If any column in *required_cols* is absent from *df*.
    """
    if df.empty:
        raise ValueError("per_class_df is empty.")
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        raise KeyError(
            f"per_class_df is missing required column(s): {missing}. "
            "Pass the DataFrame returned by compute_per_class_metrics()."
        )
__all__ = [
    "plot_confusion_matrix",
    "plot_normalized_confusion_matrix",
    "plot_confidence_distribution",
    "plot_correct_vs_incorrect_confidence",
    "plot_misclassified_samples",
    "plot_per_class_precision",
    "plot_per_class_recall",
    "plot_per_class_f1",
]