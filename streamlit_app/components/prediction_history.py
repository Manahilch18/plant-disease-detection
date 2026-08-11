"""
Prediction History Utilities

Handles persistent storage of plant disease predictions
using a local JSON file.

Project:
Plant Disease Detection System
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
import streamlit as st

# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

HISTORY_DIR = BASE_DIR / "data" / "metadata"
HISTORY_FILE = HISTORY_DIR / "prediction_history.json"


# ============================================================
# INITIALIZE STORAGE
# ============================================================

def initialize_history_file() -> None:
    """
    Create the history directory and JSON file if they
    do not already exist.
    """

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text(
            "[]",
            encoding="utf-8",
        )


# ============================================================
# LOAD HISTORY
# ============================================================

def load_prediction_history() -> list[dict[str, Any]]:
    """
    Load all saved prediction records from the JSON file.

    Returns
    -------
    list[dict[str, Any]]
        List of prediction history records.
    """

    initialize_history_file()

    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as file:
            history = json.load(file)

        if isinstance(history, list):
            return history

        return []

    except (json.JSONDecodeError, OSError):
        return []


# ============================================================
# SAVE HISTORY
# ============================================================

def save_prediction_history(
    history: list[dict[str, Any]],
) -> None:
    """
    Save prediction history to the JSON file.

    Parameters
    ----------
    history : list[dict[str, Any]]
        Prediction history records.
    """

    initialize_history_file()

    with HISTORY_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# ADD PREDICTION
# ============================================================

def add_prediction_to_history(
    disease: str,
    confidence: float,
    filename: str = "",
    confidence_status: str = "",
) -> dict[str, Any]:
    """
    Add a new prediction to persistent history.

    Parameters
    ----------
    disease : str
        Predicted disease/class name.

    confidence : float
        Model confidence between 0 and 1.

    filename : str, optional
        Uploaded image filename.

    confidence_status : str, optional
        High, Moderate, or Low.

    Returns
    -------
    dict[str, Any]
        The newly created prediction record.
    """

    history = load_prediction_history()

    record = {
        "id": datetime.now().strftime(
            "%Y%m%d%H%M%S%f"
        ),
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "filename": filename,
        "disease": disease,
        "confidence": float(confidence),
        "confidence_percent": round(
            float(confidence) * 100,
            2,
        ),
        "confidence_status": confidence_status,
    }

    history.append(record)

    save_prediction_history(history)

    return record


# ============================================================
# CLEAR HISTORY
# ============================================================

def clear_prediction_history() -> None:
    """
    Remove all saved prediction history.
    """

    save_prediction_history([])


# ============================================================
# HISTORY COUNT
# ============================================================

def get_prediction_count() -> int:
    """
    Return the total number of saved predictions.
    """

    return len(load_prediction_history())
def _format_timestamp(raw_timestamp):
    """
    Formats an ISO timestamp string from the JSON into
    "Aug 08, 2026 • 05:24 PM". Never mutates the underlying record —
    purely a display transform. Falls back to the raw string if it
    can't be parsed (e.g. an older/unexpected format).
    """
    if not raw_timestamp:
        return "—"
    try:
        cleaned = raw_timestamp.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        return dt.strftime("%b %d, %Y • %I:%M %p")
    except (ValueError, AttributeError):
        return str(raw_timestamp)
 
 
def _timestamp_sort_key(record):
    """Sort key for newest-first ordering; unparsable timestamps sink to the bottom."""
    raw_timestamp = record.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(str(raw_timestamp).replace("Z", "+00:00"))
        # Normalize to naive so mixed tz-aware/naive timestamps in the
        # same history file can still be compared/sorted together.
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except (ValueError, AttributeError):
        return datetime.min
 
 
def render_prediction_history():
    """
    Renders the read-only Prediction History page.
 
    Reads existing records via load_prediction_history() and the
    running total via get_prediction_count(). Never writes to
    data/metadata/prediction_history.json — this function has no
    save/clear calls in it at all. Search and confidence filtering
    below operate only on the in-memory list returned by
    load_prediction_history() — the JSON file itself is never touched.
    """
    st.markdown(
        """
        <style>
        .ph-summary-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.9rem;
            margin-bottom: 1.5rem;
        }
        @media (max-width: 900px) {
            .ph-summary-grid { grid-template-columns: repeat(2, 1fr); }
        }
        .ph-summary-tile {
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 0.9rem 1rem;
        }
        .ph-summary-tile .ph-tile-value {
            font-size: 1.4rem;
            font-weight: 800;
            color: var(--text-primary);
        }
        .ph-summary-tile .ph-tile-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-top: 0.2rem;
        }
 
        /* ---- Filter row ---- */
        .ph-filter-row {
            margin-bottom: 0.5rem;
        }
        .ph-filter-row div[data-testid="stTextInput"] input,
        .ph-filter-row div[data-testid="stSelectbox"] > div {
            background-color: var(--card-bg) !important;
            border: 1px solid var(--card-border) !important;
            color: var(--text-primary) !important;
            border-radius: 10px !important;
        }
        .ph-no-match {
            color: var(--text-muted);
            text-align: center;
            padding: 2rem 1rem;
            border: 1px dashed var(--card-border);
            border-radius: 12px;
            margin-top: 0.5rem;
        }
 
        /* ---- Prediction cards ---- */
        .ph-card {
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-left: 3px solid var(--card-border);
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.85rem;
        }
        .ph-card.tag-high    { border-left-color: var(--accent); }
        .ph-card.tag-moderate{ border-left-color: var(--warning); }
        .ph-card.tag-low     { border-left-color: var(--danger); }
 
        .ph-card-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-bottom: 0.7rem;
            padding-bottom: 0.6rem;
            border-bottom: 1px solid var(--divider);
        }
        .ph-card-disease {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text-primary);
        }
        .ph-card-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.88rem;
            padding: 0.2rem 0;
        }
        .ph-card-row-label {
            color: var(--text-muted);
        }
        .ph-card-row-value {
            color: var(--text-primary);
            font-weight: 600;
        }
        .ph-conf-high     { color: var(--accent); }
        .ph-conf-moderate { color: var(--warning); }
        .ph-conf-low      { color: var(--danger); }
 
        .ph-empty-state {
            text-align: center;
            padding: 3rem 1rem;
            color: var(--text-muted);
        }
        .ph-empty-state .ph-empty-icon {
            font-size: 2rem;
            margin-bottom: 0.6rem;
        }
        .ph-empty-state .ph-empty-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.4rem;
        }
        /* Fallback in case this page renders before any tab that
           already defines .tag-neutral elsewhere in the app */
        .tag-neutral {
            background-color: var(--pill-bg);
            color: var(--text-muted);
            border-color: var(--card-border);
        }
 
        /* ---- Clear-history confirmation ---- */
        .ph-confirm-box {
            background-color: var(--danger-dim);
            border: 1px solid rgba(248, 113, 113, 0.35);
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin: 0.5rem 0 1.25rem 0;
        }
        .ph-confirm-title {
            font-size: 1rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.3rem;
        }
        .ph-confirm-desc {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
 
    if "ph_confirm_clear" not in st.session_state:
        st.session_state["ph_confirm_clear"] = False
 
    records = load_prediction_history()
 
    # Show the success message from a clear performed on the *previous*
    # run — st.rerun() below starts a fresh run, so this flag is what
    # carries the message across that boundary. Popped so it only
    # shows once.
    if st.session_state.pop("ph_just_cleared", False):
        st.success("✅ Prediction history cleared.")
 
    header_col, clear_col = st.columns([4, 1])
    with header_col:
        st.markdown(
            '<div class="card-header" style="font-size:1.5rem;">📜 Prediction History</div>',
            unsafe_allow_html=True,
        )
    with clear_col:
        if records and not st.session_state["ph_confirm_clear"]:
            if st.button("🗑 Clear History", key="ph_clear_trigger", use_container_width=True):
                st.session_state["ph_confirm_clear"] = True
                st.rerun()
 
    # --------------------------------------------------------
    # CLEAR-HISTORY CONFIRMATION — nothing is deleted until the user
    # explicitly presses "Clear History" here. clear_prediction_history()
    # is the ONLY call in this whole function that mutates the JSON
    # file; Cancel touches only the session_state flag above.
    # --------------------------------------------------------
    if st.session_state["ph_confirm_clear"]:
        st.markdown(
            """
            <div class="ph-confirm-box">
                <div class="ph-confirm-title">Clear all prediction history?</div>
                <div class="ph-confirm-desc">This action will permanently remove all saved prediction records.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cancel_col, confirm_col = st.columns(2)
        with cancel_col:
            if st.button("Cancel", key="ph_clear_cancel", use_container_width=True):
                st.session_state["ph_confirm_clear"] = False
                st.rerun()
        with confirm_col:
            if st.button("Clear History", key="ph_clear_confirm", type="primary", use_container_width=True):
                clear_prediction_history()
                st.session_state["ph_confirm_clear"] = False
                st.session_state["ph_just_cleared"] = True
                st.rerun()
        # Don't render the (now-stale) list/summary underneath the
        # confirmation prompt — re-fetch happens on the next rerun.
        return
 
    if not records:
        st.markdown(
            """
            <div class="ph-empty-state">
                <div class="ph-empty-icon">📭</div>
                <div class="ph-empty-title">No predictions yet</div>
                <div>Head over to the Analyze tab and upload a leaf image to get started.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
 
    # Summary tiles always reflect the full history, independent of the
    # search/filter below — they describe the dataset, not the current
    # view.
    total = get_prediction_count()
    high = sum(1 for r in records if r.get("confidence_status") == "High Confidence")
    moderate = sum(1 for r in records if r.get("confidence_status") == "Moderate Confidence")
    low = sum(1 for r in records if r.get("confidence_status") == "Low Confidence")
 
    st.markdown(
        f"""
        <div class="ph-summary-grid">
            <div class="ph-summary-tile">
                <div class="ph-tile-value">{total}</div>
                <div class="ph-tile-label">Total Predictions</div>
            </div>
            <div class="ph-summary-tile">
                <div class="ph-tile-value" style="color: var(--accent);">{high}</div>
                <div class="ph-tile-label">High Confidence</div>
            </div>
            <div class="ph-summary-tile">
                <div class="ph-tile-value" style="color: var(--warning);">{moderate}</div>
                <div class="ph-tile-label">Moderate Confidence</div>
            </div>
            <div class="ph-summary-tile">
                <div class="ph-tile-value" style="color: var(--danger);">{low}</div>
                <div class="ph-tile-label">Low Confidence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    # --------------------------------------------------------
    # SEARCH + CONFIDENCE FILTER — operates only on the in-memory
    # `records` list already loaded above. Does not re-read or modify
    # the JSON file in any way.
    # --------------------------------------------------------
    st.markdown('<div class="ph-filter-row">', unsafe_allow_html=True)
    search_col, filter_col = st.columns([2, 1])
    with search_col:
        search_query = st.text_input(
            "🔎 Search",
            placeholder="Search by disease, plant, or filename...",
            key="ph_search_query",
        )
    with filter_col:
        confidence_choice = st.selectbox(
            "Confidence",
            options=["All", "High", "Moderate", "Low"],
            key="ph_confidence_filter",
        )
    st.markdown("</div>", unsafe_allow_html=True)
 
    filtered_records = records
    query = search_query.strip().lower()
    if query:
        filtered_records = [
            r for r in filtered_records
            if query in str(r.get("disease", "")).lower()
            or query in str(r.get("filename", "")).lower()
        ]
 
    status_by_choice = {
        "High": "High Confidence",
        "Moderate": "Moderate Confidence",
        "Low": "Low Confidence",
    }
    if confidence_choice != "All":
        target_status = status_by_choice[confidence_choice]
        filtered_records = [
            r for r in filtered_records if r.get("confidence_status") == target_status
        ]
 
    if not filtered_records:
        st.markdown(
            '<div class="ph-no-match">No predictions match your current filters.</div>',
            unsafe_allow_html=True,
        )
        return
 
    status_tag_class = {
        "High Confidence": "tag-high",
        "Moderate Confidence": "tag-moderate",
        "Low Confidence": "tag-low",
    }
    status_conf_class = {
        "High Confidence": "ph-conf-high",
        "Moderate Confidence": "ph-conf-moderate",
        "Low Confidence": "ph-conf-low",
    }
 
    # Newest first, sorted by parsed timestamp rather than assumed
    # list order — more robust if records are ever loaded out of
    # append order.
    for record in sorted(filtered_records, key=_timestamp_sort_key, reverse=True):
        disease = record.get("disease", "Unknown")
        filename = record.get("filename", "—")
        formatted_time = _format_timestamp(record.get("timestamp"))
        confidence_status = record.get("confidence_status", "—")
        confidence_percent = record.get("confidence_percent")
        if confidence_percent is None:
            confidence_percent = round(record.get("confidence", 0) * 100, 2)
 
        tag_class = status_tag_class.get(confidence_status, "tag-neutral")
        conf_class = status_conf_class.get(confidence_status, "")
 
        st.markdown(
            f"""
            <div class="ph-card {tag_class}">
                <div class="ph-card-top">
                    <div class="ph-card-disease">🌱 {disease}</div>
                    <span class="confidence-tag {tag_class}" style="float:none;">{confidence_status.upper()}</span>
                </div>
                <div class="ph-card-row">
                    <span class="ph-card-row-label">🎯 Confidence</span>
                    <span class="ph-card-row-value {conf_class}">{confidence_percent:.2f}%</span>
                </div>
                <div class="ph-card-row">
                    <span class="ph-card-row-label">📁 Filename</span>
                    <span class="ph-card-row-value">{filename}</span>
                </div>
                <div class="ph-card-row">
                    <span class="ph-card-row-label">📅 Date</span>
                    <span class="ph-card-row-value">{formatted_time}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
 
 