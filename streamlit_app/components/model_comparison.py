"""
Model Comparison component for the Plant Disease Detection app.

This module is a pure reporting/visualization page:
- It does NOT import tensorflow, load any model weights, or call
  predict()/evaluate() on anything.
- It does NOT touch the Baseline CNN, its preprocessing, the FastAPI
  client, Grad-CAM, disease_info.py, or confidence-threshold logic in
  app.py — none of those are imported here.
- All metrics below are the final evaluation numbers already computed
  offline; this file only formats and displays them.

Usage from app.py:
    from streamlit_app.components.model_comparison import render_model_comparison
    render_model_comparison()
"""

import streamlit as st
import pandas as pd

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


# ============================================================
# EVALUATION DATA
# ============================================================
# Structured once here — nothing computed at runtime, nothing loaded.

DATASET_SUMMARY = {
    "Dataset": "PlantVillage",
    "Classes": "38",
    "Test Images": "8,146",
    "Models Evaluated": "3",
}

MODEL_METRICS = {
    "Baseline CNN": {
        "Test Loss": 0.1588540673,
        "Accuracy": 0.9884606004,
        "Top-5 Accuracy": 0.9996317029,
        "Precision": 0.982484,
        "Recall": 0.986944,
        "Macro F1": 0.984521,
        "Weighted F1": 0.988468,
        "Evaluation Time": 28.60279427,
        "Mean Confidence": 0.985824,
        "Median Confidence": 0.999923,
    },
    "EfficientNetB0": {
        "Test Loss": 0.0686539784,
        "Accuracy": 0.9770439267,
        "Top-5 Accuracy": 0.9996317029,
        "Precision": 0.9745,
        "Recall": 0.9659,
        "Macro F1": 0.9696,
        "Weighted F1": 0.9770,
        "Evaluation Time": 18.25,
        "Mean Confidence": 0.9801,
        "Median Confidence": 1.0000,
    },
    "ResNet50": {
        "Test Loss": 1.5796782970,
        "Accuracy": 0.6551681807,
        "Top-5 Accuracy": 0.9069482088,
        "Precision": 0.7387715091,
        "Recall": 0.6551681807,
        "Macro F1": 0.5806081988,
        "Weighted F1": 0.6434725102,
        "Evaluation Time": 34.13480391,
        "Mean Confidence": 0.8118,
        "Median Confidence": 0.9100,
    },
}

# Rows shown in the comparison table, in display order, with the
# formatter each metric needs. "pct" = 0-1 fraction -> "xx.xx%",
# "loss" = plain 4-decimal, "time" = seconds with 2 decimals.
METRIC_ROWS = [
    ("Test Loss", "loss"),
    ("Accuracy", "pct"),
    ("Top-5 Accuracy", "pct"),
    ("Precision", "pct"),
    ("Recall", "pct"),
    ("Macro F1", "pct"),
    ("Weighted F1", "pct"),
    ("Evaluation Time", "time"),
    ("Mean Confidence", "pct"),
    ("Median Confidence", "pct"),
]

MODEL_ORDER = ["Baseline CNN", "EfficientNetB0", "ResNet50"]

MODEL_COLORS = {
    "Baseline CNN": "#34d8a6",
    "EfficientNetB0": "#5ea8ff",
    "ResNet50": "#b592ff",
}

TRAINING_STRATEGY = [
    {"Model": "Baseline CNN", "Training Approach": "Custom CNN training", "Epochs": "30"},
    {
        "Model": "EfficientNetB0",
        "Training Approach": "Frozen backbone + fine-tuning",
        "Epochs": "25 actual / 30 planned",
    },
    {"Model": "ResNet50", "Training Approach": "Transfer learning + fine-tuning", "Epochs": "45"},
]

MODEL_CARDS = [
    {"icon": "🧠", "name": "Baseline CNN", "subtitle": "Custom CNN", "badge": "Selected Model", "badge_class": "tag-high"},
    {"icon": "⚡", "name": "EfficientNetB0", "subtitle": "ImageNet Transfer Learning", "badge": "Fine-tuned", "badge_class": "tag-neutral"},
    {"icon": "🔬", "name": "ResNet50", "subtitle": "ImageNet Transfer Learning", "badge": "45 epochs", "badge_class": "tag-neutral"},
]


def _fmt(value: float, kind: str) -> str:
    """Format a metric value. Guards against NaN so it never reaches the UI."""
    if value is None or (isinstance(value, float) and value != value):  # NaN check
        return "—"
    if kind == "pct":
        return f"{value * 100:.2f}%"
    if kind == "time":
        return f"{value:.2f} s"
    return f"{value:.4f}"


def _inject_css():
    """
    Additive CSS only — reuses the app's existing --accent / --card-bg /
    --text-muted etc. custom properties (defined once in app.py) rather
    than redefining the theme. Only adds the handful of classes this
    page needs that don't already exist.
    """
    st.markdown(
        """
        <style>
        .mc-subtitle {
            color: var(--text-muted);
            font-size: 1rem;
            line-height: 1.55;
            max-width: 68ch;
            margin-bottom: 1.25rem;
        }

        .mc-summary-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.9rem;
            margin-bottom: 1.25rem;
        }
        @media (max-width: 900px) {
            .mc-summary-grid { grid-template-columns: repeat(2, 1fr); }
        }

        .mc-summary-tile {
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            text-align: left;
        }
        .mc-summary-tile .mc-tile-value {
            font-size: 1.3rem;
            font-weight: 800;
            color: var(--text-primary);
        }
        .mc-summary-tile .mc-tile-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-top: 0.2rem;
        }

        .mc-model-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.9rem;
            margin-bottom: 0.5rem;
        }
        @media (max-width: 900px) {
            .mc-model-grid { grid-template-columns: 1fr; }
        }

        .mc-model-card {
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 0.95rem 1.05rem;
        }
        .mc-model-card .mc-model-name {
            font-size: 1rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.15rem;
        }
        .mc-model-card .mc-model-subtitle {
            font-size: 0.82rem;
            color: var(--text-muted);
            margin-bottom: 0.65rem;
        }

        .mc-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.25rem 0.7rem;
            font-size: 0.72rem;
            font-weight: 700;
            border: 1px solid;
        }
        .tag-neutral {
            background-color: var(--pill-bg);
            color: var(--text-muted);
            border-color: var(--card-border);
        }

        .mc-table-wrap {
            overflow-x: auto;
        }
        table.mc-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }
        table.mc-table th, table.mc-table td {
            padding: 0.55rem 0.8rem;
            border-bottom: 1px solid var(--divider);
            text-align: left;
            color: var(--text-primary);
            white-space: nowrap;
        }
        table.mc-table th {
            color: var(--text-muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 700;
        }
        table.mc-table td.mc-metric-name {
            color: var(--text-muted);
            font-weight: 600;
        }
        table.mc-table tr:last-child td {
            border-bottom: none;
        }
        table.mc-table td.mc-best {
            color: var(--accent);
            font-weight: 700;
        }

        .mc-final-banner {
            background-color: var(--accent-dim);
            border: 1px solid rgba(52, 216, 166, 0.3);
            border-radius: 14px;
            padding: 1.4rem 1.5rem;
            margin: 0.5rem 0 1.25rem 0;
        }
        .mc-final-eyebrow {
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
            color: var(--accent);
            margin-bottom: 0.3rem;
        }
        .mc-final-name {
            font-size: 1.6rem;
            font-weight: 800;
            color: var(--text-primary);
            margin-bottom: 0.4rem;
        }
        .mc-final-reason {
            color: var(--text-muted);
            font-size: 0.92rem;
            line-height: 1.5;
            margin-bottom: 1rem;
            max-width: 70ch;
        }
        .mc-final-metrics {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.9rem;
        }
        @media (max-width: 900px) {
            .mc-final-metrics { grid-template-columns: repeat(2, 1fr); }
        }
        .mc-final-metric-value {
            font-size: 1.35rem;
            font-weight: 800;
            color: var(--accent);
        }
        .mc-final-metric-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.15rem;
        }
        .mc-final-note {
            color: var(--text-muted);
            font-size: 0.82rem;
            margin-top: 1rem;
            padding-top: 0.8rem;
            border-top: 1px solid rgba(52, 216, 166, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_summary_cards():
    tiles = "".join(
        f'<div class="mc-summary-tile">'
        f'<div class="mc-tile-value">{value}</div>'
        f'<div class="mc-tile-label">{label}</div>'
        f"</div>"
        for label, value in DATASET_SUMMARY.items()
    )
    st.markdown(f'<div class="mc-summary-grid">{tiles}</div>', unsafe_allow_html=True)


def _render_model_cards():
    cards = "".join(
        f'<div class="mc-model-card">'
        f'<div class="mc-model-name">{c["icon"]} {c["name"]}</div>'
        f'<div class="mc-model-subtitle">{c["subtitle"]}</div>'
        f'<span class="mc-badge {c["badge_class"]}">{c["badge"]}</span>'
        f"</div>"
        for c in MODEL_CARDS
    )
    st.markdown(f'<div class="mc-model-grid">{cards}</div>', unsafe_allow_html=True)


def _render_comparison_table():
    header = "<tr><th>Metric</th>" + "".join(f"<th>{m}</th>" for m in MODEL_ORDER) + "</tr>"

    rows_html = []
    for metric_name, kind in METRIC_ROWS:
        raw_values = {m: MODEL_METRICS[m].get(metric_name) for m in MODEL_ORDER}

        # "Best" cell highlighting: lower is better for loss/time, higher
        # for everything else. Purely a display aid, computed from the
        # static table above — no live evaluation involved.
        valid = {m: v for m, v in raw_values.items() if v is not None}
        if kind in ("loss", "time"):
            best_model = min(valid, key=valid.get) if valid else None
        else:
            best_model = max(valid, key=valid.get) if valid else None

        cells = [f'<td class="mc-metric-name">{metric_name}</td>']
        for m in MODEL_ORDER:
            css_class = "mc-best" if m == best_model else ""
            cells.append(f'<td class="{css_class}">{_fmt(raw_values[m], kind)}</td>')
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    table_html = (
        '<div class="mc-table-wrap"><table class="mc-table">'
        f"<thead>{header}</thead><tbody>{''.join(rows_html)}</tbody>"
        "</table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def _plotly_layout(fig, title, yaxis_title, y_is_percent=False):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e7ecef")),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8a97a0", size=12),
        margin=dict(t=45, b=30, l=10, r=10),
        height=320,
        legend=dict(orientation="h", y=-0.18, font=dict(color="#8a97a0")),
        yaxis=dict(title=yaxis_title, gridcolor="#1c2730", tickfont=dict(color="#8a97a0")),
        xaxis=dict(tickfont=dict(color="#e7ecef")),
    )
    if y_is_percent:
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
    return fig


def _render_charts():
    if not _PLOTLY_AVAILABLE:
        st.info(
            "Install `plotly` to see the visual comparison charts "
            "(`pip install plotly --break-system-packages`). The table "
            "above already contains the full numeric comparison."
        )
        return

    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]

    chart_col1, chart_col2 = st.columns(2, gap="large")

    with chart_col1:
        acc = [MODEL_METRICS[m]["Accuracy"] * 100 for m in MODEL_ORDER]
        fig = go.Figure(go.Bar(x=MODEL_ORDER, y=acc, marker_color=colors, text=[f"{v:.2f}%" for v in acc], textposition="outside"))
        _plotly_layout(fig, "Accuracy Comparison", "Accuracy", y_is_percent=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with chart_col2:
        top5 = [MODEL_METRICS[m]["Top-5 Accuracy"] * 100 for m in MODEL_ORDER]
        fig = go.Figure(go.Bar(x=MODEL_ORDER, y=top5, marker_color=colors, text=[f"{v:.2f}%" for v in top5], textposition="outside"))
        _plotly_layout(fig, "Top-5 Accuracy Comparison", "Top-5 Accuracy", y_is_percent=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    chart_col3, chart_col4 = st.columns(2, gap="large")

    with chart_col3:
        fig = go.Figure()
        for metric_name in ["Precision", "Recall", "Macro F1"]:
            values = [MODEL_METRICS[m][metric_name] * 100 for m in MODEL_ORDER]
            fig.add_bar(name=metric_name, x=MODEL_ORDER, y=values, text=[f"{v:.1f}%" for v in values], textposition="outside")
        _plotly_layout(fig, "Precision / Recall / Macro F1", "Score", y_is_percent=True)
        fig.update_layout(barmode="group")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with chart_col4:
        evtime = [MODEL_METRICS[m]["Evaluation Time"] for m in MODEL_ORDER]
        fig = go.Figure(go.Bar(x=MODEL_ORDER, y=evtime, marker_color=colors, text=[f"{v:.2f}s" for v in evtime], textposition="outside"))
        _plotly_layout(fig, "Evaluation Time Comparison", "Seconds")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_training_strategy():
    df = pd.DataFrame(TRAINING_STRATEGY)
    header = "<tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
    rows = "".join(
        "<tr>" + "".join(f"<td>{row[c]}</td>" for c in df.columns) + "</tr>"
        for _, row in df.iterrows()
    )
    st.markdown(
        f'<div class="mc-table-wrap"><table class="mc-table">'
        f"<thead>{header}</thead><tbody>{rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="mc-subtitle" style="margin-top:0.9rem;">'
        "Training was planned for 30 epochs. EarlyStopping stopped training "
        "at epoch 25. The best fine-tuning validation result occurred at "
        "epoch 20 with 97.85% validation accuracy and 0.0639 validation loss."
        "</div>",
        unsafe_allow_html=True,
    )


def _render_model_analysis():
    with st.expander("🧠 Baseline CNN"):
        st.markdown(
            "- Strongest overall test accuracy (98.85%) among the models evaluated\n"
            "- Excellent Top-5 accuracy (99.96%)\n"
            "- Strongest Macro F1 and Weighted F1 scores in this evaluation\n"
            "- High recall (98.69%)\n"
            "- **Baseline CNN achieved the strongest overall performance among "
            "the models evaluated in this project**, and was selected as the "
            "final model."
        )

    with st.expander("⚡ EfficientNetB0"):
        st.markdown(
            "- Strong overall classification performance\n"
            "- Strongest evaluation speed among the three models (18.25 s)\n"
            "- Strong transfer-learning representation from ImageNet pretraining\n"
            "- **EfficientNetB0 achieved slightly lower classification metrics "
            "in this evaluation, although it had a shorter evaluation time**\n"
            "- Therefore not selected as the final model for this project"
        )

    with st.expander("🔬 ResNet50"):
        st.markdown(
            "- Substantially lower test accuracy (65.52%)\n"
            "- Lower Macro F1 (58.06%)\n"
            "- Lower Top-5 accuracy (90.69%) relative to the other two models\n"
            "- **ResNet50 achieved substantially lower performance under the "
            "configuration evaluated in this project**\n"
            "- Therefore not selected as the final model"
        )


def _render_final_selection():
    st.markdown(
        f"""
        <div class="mc-final-banner">
            <div class="mc-final-eyebrow">🏆 Final Model</div>
            <div class="mc-final-name">Baseline CNN</div>
            <div class="mc-final-reason">
                Selected based on the strongest overall performance on the
                PlantVillage test set among the evaluated models.
            </div>
            <div class="mc-final-metrics">
                <div>
                    <div class="mc-final-metric-value">98.85%</div>
                    <div class="mc-final-metric-label">Test Accuracy</div>
                </div>
                <div>
                    <div class="mc-final-metric-value">99.96%</div>
                    <div class="mc-final-metric-label">Top-5 Accuracy</div>
                </div>
                <div>
                    <div class="mc-final-metric-value">98.45%</div>
                    <div class="mc-final-metric-label">Macro F1</div>
                </div>
                <div>
                    <div class="mc-final-metric-value">98.85%</div>
                    <div class="mc-final-metric-label">Weighted F1</div>
                </div>
            </div>
            <div class="mc-final-note">
                Model selection was based on comparative evaluation rather
                than a single metric.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_comparison():
    """
    Entry point called from app.py. Renders the full Model Comparison
    page. No model is loaded and no prediction/evaluation runs here —
    every number displayed comes from the static MODEL_METRICS table
    above.
    """
    _inject_css()

    st.markdown('<div class="card-header" style="font-size:1.5rem;">📊 Model Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mc-subtitle">Evidence-based comparison of the three '
        "CNN architectures evaluated on the PlantVillage test set.</div>",
        unsafe_allow_html=True,
    )

    _render_summary_cards()

    st.markdown('<div class="card-header">Model Overview</div>', unsafe_allow_html=True)
    _render_model_cards()

    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Performance Comparison</div>', unsafe_allow_html=True)
    _render_comparison_table()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Visual Comparison</div>', unsafe_allow_html=True)
    _render_charts()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">🧪 Training Strategy</div>', unsafe_allow_html=True)
    _render_training_strategy()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card-header">Model-by-Model Analysis</div>', unsafe_allow_html=True)
    _render_model_analysis()

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    _render_final_selection()