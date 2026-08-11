import io
import base64
import requests
import streamlit as st
from PIL import Image

from src.disease_info import get_disease_info

# Validity / OOD gate — standalone module, does not touch the Baseline
# CNN, its weights, or its [0,255] float32 preprocessing.
from src.validity_check import check_leaf_validity

# Model Comparison — pure reporting page, no model loading, no
# prediction/evaluation logic. Rendered under its own tab below.
from streamlit_app.components.model_comparison import render_model_comparison

# Prediction history — persists successful predictions only.
# Import path follows the existing project structure of
# streamlit_app/components/prediction_history.py.
from components.prediction_history import (
    add_prediction_to_history,
    render_prediction_history,
)

from streamlit_app.config import (
    API_BASE_URL,
    PREDICT_ENDPOINT,
    GRADCAM_ENDPOINT,
)

# ============================================================
# CONFIG
# ============================================================

NUM_CLASSES = 38  # shown in the header badge — update if your model differs

# Margin required between the best "leaf" prompt score and the best
# "non-leaf" prompt score for an upload to be accepted by the validity
# gate. Tune this against a small labeled sample of real uploads —
# it's a plain config value, no retraining involved.
VALIDITY_MARGIN = 0.03

st.set_page_config(
    page_title="PhytoScan AI",
    page_icon="🌿",
    layout="wide",
)

# ============================================================
# DESIGN TOKENS + CSS
# ============================================================
# Dark navy canvas, teal/emerald accent, card-based layout —
# matches the reference "PhytoScan AI" mock.

st.markdown(
    """
    <style>

    :root {
        --bg:            #0a0f16;
        --card-bg:       #101820;
        --card-border:   #1f2b28;
        --accent:        #34d8a6;
        --accent-dim:    #1c3b32;
        --text-primary:  #e7ecef;
        --text-muted:    #8a97a0;
        --pill-bg:       #16211c;
        --divider:       #1c2730;
        --warning:       #f5c451;
        --warning-dim:   #332a12;
        --danger:        #f87171;
        --danger-dim:    #35191a;
    }

    /* App background */
    .stApp {
        background-color: var(--bg);
    }

    section.main > div {
        padding-top: 1.5rem;
    }

    /* Hide default Streamlit chrome for a cleaner product feel */
    #MainMenu, footer, header[data-testid="stHeader"] {
        background: transparent;
    }

    /* ---------- Brand header ---------- */

    .brand-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }

    .brand-logo {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--accent);
        letter-spacing: 0.01em;
    }

    .top-divider {
        border: none;
        border-top: 1px solid var(--divider);
        margin: 0.75rem 0 1.75rem 0;
    }

    /* Eyebrow / model badge pill */
    .model-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background-color: var(--accent-dim);
        color: var(--accent);
        border: 1px solid rgba(52, 216, 166, 0.25);
        border-radius: 999px;
        padding: 0.35rem 0.9rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1.1rem;
    }

    .model-badge .dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: var(--accent);
        display: inline-block;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: var(--text-primary);
        margin-bottom: 0.4rem;
        letter-spacing: -0.01em;
    }

    .subtitle {
        color: var(--text-muted);
        font-size: 1.02rem;
        line-height: 1.55;
        max-width: 62ch;
        margin-bottom: 0.25rem;
    }

    /* ---------- Card shell ---------- */

    .card {
        background-color: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1.25rem;
    }

    .card-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1rem;
    }

    .eyebrow {
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    /* ---------- Upload zone ---------- */

    div[data-testid="stFileUploaderDropzone"] {
        background-color: #0c131a;
        border: 1.5px dashed #2a3944;
        border-radius: 12px;
    }

    div[data-testid="stFileUploaderDropzoneInstructions"] svg {
        display: none;
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        background-color: var(--accent);
        color: #06110d;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        transition: filter 0.15s ease;
    }

    .stButton > button:hover {
        filter: brightness(1.08);
        color: #06110d;
    }

    .stButton > button:active {
        filter: brightness(0.95);
    }

    /* Secondary / reset buttons: give them a muted outline look via key */
    button[kind="secondary"] {
        background-color: transparent !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--card-border) !important;
    }

    /* ---------- Prediction result card ---------- */

    .result-eyebrow {
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    .disease-name {
        font-size: 1.7rem;
        font-weight: 800;
        color: var(--text-primary);
    }

    .confidence-label {
        color: var(--text-muted);
        font-size: 0.92rem;
        margin-top: 0.9rem;
        margin-bottom: 0.35rem;
        display: flex;
        justify-content: space-between;
    }

    .confidence-value {
        color: var(--accent);
        font-weight: 700;
    }

    .pill {
        display: inline-block;
        background-color: var(--pill-bg);
        color: var(--text-muted);
        border: 1px solid var(--card-border);
        border-radius: 8px;
        padding: 0.3rem 0.7rem;
        font-size: 0.82rem;
        margin-right: 0.5rem;
        margin-top: 0.9rem;
    }

    .confidence-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border-radius: 999px;
        padding: 0.3rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 700;
        float: right;
        border: 1px solid;
    }

    /* Confidence-level variants — shared by the result badge and the
       compact status banner below the disease info card. */
    .tag-high {
        background-color: var(--accent-dim);
        color: var(--accent);
        border-color: rgba(52, 216, 166, 0.3);
    }
    .tag-moderate {
        background-color: var(--warning-dim);
        color: var(--warning);
        border-color: rgba(245, 196, 81, 0.35);
    }
    .tag-low {
        background-color: var(--danger-dim);
        color: var(--danger);
        border-color: rgba(248, 113, 113, 0.35);
    }

    /* Compact confidence status banner — replaces the oversized
       st.error/st.warning/st.success blocks. */
    .status-banner {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        border-radius: 10px;
        padding: 0.65rem 0.9rem;
        font-size: 0.88rem;
        line-height: 1.4;
        border: 1px solid;
        margin-top: 0.25rem;
    }

    div[data-testid="stProgress"] > div > div {
        background-color: var(--accent) !important;
    }

    div[data-testid="stProgress"] > div {
        background-color: #1a2530 !important;
    }

    /* ---------- Info grid labels ---------- */

    .field-label {
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 0.35rem;
        margin-top: 0.9rem;
    }

    .field-box {
        background-color: #0c131a;
        border: 1px solid var(--card-border);
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
        color: var(--text-primary);
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .disclaimer {
        color: var(--text-muted);
        font-size: 0.82rem;
        margin-top: 1rem;
    }

    /* ---------- Grad-CAM section text ---------- */

    .gradcam-note {
        color: var(--text-muted);
        font-size: 0.85rem;
        margin-top: 0.85rem;
        line-height: 1.5;
    }

    .gradcam-disclaimer {
        color: var(--text-muted);
        font-size: 0.78rem;
        margin-top: 0.6rem;
        padding-top: 0.6rem;
        border-top: 1px solid var(--divider);
        line-height: 1.5;
    }

    /* Section divider between blocks inside a card */
    .card-divider {
        border: none;
        border-top: 1px solid var(--divider);
        margin: 1.25rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="brand-row"><span class="brand-logo">🌿 PhytoScan AI</span></div>', unsafe_allow_html=True)
st.markdown('<hr class="top-divider" />', unsafe_allow_html=True)

st.markdown(
    f'<div class="model-badge"><span class="dot"></span>Baseline CNN &middot; {NUM_CLASSES} Classes</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Plant Disease Detection</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="subtitle">
    AI-powered plant disease classification using a trained Baseline CNN.
    Upload a clear image of a plant leaf to identify potential pathogens
    or nutritional deficiencies.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ============================================================
# NAVIGATION
# ============================================================
# Two tabs: the existing prediction workflow (untouched below,
# just indented under tab1), and the new reporting-only Model
# Comparison page under tab2.

tab1, tab2, tab3 = st.tabs(["🌿 Analyze", "📊 Model Comparison", "📜 Prediction History"])

with tab1:
    # ============================================================
    # RESET
    # ============================================================
    # Only prediction-related state is cleared here — this keeps the reset
    # safe even if config/session keys are added elsewhere later.

    PREDICTION_STATE_KEYS = ("prediction", "confidence")

    if "uploader_version" not in st.session_state:
        st.session_state["uploader_version"] = 0

    reset_col, _ = st.columns([1, 3])
    with reset_col:
        if st.button("🔄 New Prediction", use_container_width=True):
            # Clears the disease name + confidence score. Everything that is
            # gated on `"prediction" in st.session_state` (Grad-CAM, disease
            # info card, confidence status banner) disappears as a result —
            # nothing else needs to be deleted explicitly.
            for key in PREDICTION_STATE_KEYS:
                st.session_state.pop(key, None)

            # Bumping this forces st.file_uploader to mount as a *new* widget
            # (its key changes below), which is what actually clears the
            # previously uploaded image. Session-state deletion alone does
            # not reset a file_uploader's contents.
            st.session_state["uploader_version"] += 1

            st.rerun()

    st.write("")

    # ============================================================
    # MAIN LAYOUT: upload (left) + result / explainability (right)
    # ============================================================

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">📤 Upload Leaf Image</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload a leaf image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key=f"uploader_{st.session_state['uploader_version']}",
        )

        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="card-header">🖼️ Uploaded Leaf Image '
                f'<span style="color:var(--text-muted); font-weight:400; font-size:0.85rem; margin-left:auto;">'
                f'{uploaded_file.name}</span></div>',
                unsafe_allow_html=True,
            )
            st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            predict_clicked = st.button("🔍 Predict Disease", type="primary", use_container_width=True)

            if predict_clicked:
                # ------------------------------------------------------
                # VALIDITY / OOD GATE — runs before the Baseline CNN.
                # Independent of the CNN's confidence/output: it never
                # looks at the 38-class prediction at all, so it can't be
                # fooled by a confidently-wrong closed-set classification.
                # On failure, the existing prediction code below never
                # runs — no API call, no Grad-CAM, no session_state write.
                # ------------------------------------------------------
                with st.spinner("Checking image..."):
                    try:
                        validity = check_leaf_validity(image, margin=VALIDITY_MARGIN)
                    except Exception as exc:
                        # If the gate itself fails (e.g. missing dependency,
                        # first-run download issue), fail open with a clear
                        # warning rather than silently blocking predictions.
                        validity = None
                        st.warning(
                            f"Validity check unavailable ({exc}); proceeding "
                            "without the plant-leaf gate."
                        )

                if validity is not None and not validity.is_valid:
                    st.error(
                        "This image doesn't look like a plant leaf "
                        f"(closest match: \"{validity.best_negative_label}\"). "
                        "Please upload a clear photo of a plant leaf."
                    )
                else:
                    with st.spinner("Analyzing image..."):
                        try:
                            # IMPORTANT:
                            # Do NOT divide image pixels by 255.
                            image_bytes = io.BytesIO()
                            image.save(image_bytes, format="JPEG")
                            image_bytes.seek(0)

                            response = requests.post(
                                PREDICT_ENDPOINT,
                                files={"file": (uploaded_file.name, image_bytes, "image/jpeg")},
                                timeout=60,
                            )
                            response.raise_for_status()
                            result = response.json()

                            disease = result["disease"]
                            confidence = float(result["confidence"])

                            st.session_state["prediction"] = disease
                            st.session_state["confidence"] = confidence

                            # ------------------------------------------------------
                            # SAVE TO PREDICTION HISTORY — successful predictions only.
                            #
                            # This whole `try` block only executes when the user has
                            # just clicked "Predict Disease" (it's nested inside
                            # `if predict_clicked:`), so it does not re-run on
                            # unrelated Streamlit reruns. The extra guard below is a
                            # second, explicit safeguard: it keys off this exact
                            # (filename, disease, confidence) result and skips the
                            # save if that same result was already persisted, so a
                            # rerun can never append a duplicate row for the same
                            # prediction.
                            # ------------------------------------------------------
                            prediction_key = f"{uploaded_file.name}:{disease}:{confidence}"

                            if st.session_state.get("last_saved_prediction_key") != prediction_key:
                                # Same threshold logic used for the confidence badge
                                # elsewhere in this file — not re-derived, just reused
                                # as plain labels here so the history record matches
                                # what the user was shown on screen.
                                if confidence >= 0.80:
                                    confidence_status = "High Confidence"
                                elif confidence >= 0.60:
                                    confidence_status = "Moderate Confidence"
                                else:
                                    confidence_status = "Low Confidence"

                                add_prediction_to_history(
                                    disease=disease,
                                    confidence=confidence,
                                    filename=uploaded_file.name,
                                    confidence_status=confidence_status,
                                )

                                st.session_state["last_saved_prediction_key"] = prediction_key

                        except requests.exceptions.RequestException as e:
                            st.error(f"❌ Could not connect to FastAPI: {e}")
                        except Exception as e:
                            st.error(f"❌ Prediction failed: {e}")

    with right_col:
        if "prediction" in st.session_state:
            disease = st.session_state["prediction"]
            confidence = st.session_state["confidence"]
            confidence_percent = confidence * 100

            if confidence >= 0.80:
                tag_label, tag_icon, tag_class = "High Confidence", "✓", "tag-high"
            elif confidence >= 0.60:
                tag_label, tag_icon, tag_class = "Moderate Confidence", "!", "tag-moderate"
            else:
                tag_label, tag_icon, tag_class = "Low Confidence", "!", "tag-low"

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="result-eyebrow">Analysis Result</div>
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div class="disease-name">🌱 {disease}</div>
                    <div class="confidence-tag {tag_class}">{tag_icon} {tag_label}</div>
                </div>
                <div class="confidence-label">
                    Prediction Confidence
                    <span class="confidence-value">{confidence_percent:.1f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(confidence)
            st.markdown("</div>", unsafe_allow_html=True)

            # ------------------------------------------------------
            # GRAD-CAM EXPLAINABILITY
            # ------------------------------------------------------
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # Single heading for this section — the API response has no
            # heading of its own. This card-header is the one source of
            # truth for it.
            st.markdown('<div class="card-header">🔥 Model Explainability</div>', unsafe_allow_html=True)
            st.caption(
                "Grad-CAM visualizes the image regions that contributed "
                "to the model's prediction."
            )

            try:
                # Grad-CAM is generated by the FastAPI service.
                # Streamlit does NOT load TensorFlow or the model locally.
                gradcam_response = requests.post(
                    GRADCAM_ENDPOINT,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    },
                    timeout=60,
                )

                gradcam_response.raise_for_status()

                gradcam_data = gradcam_response.json()

                # API returns the Grad-CAM image as a base64 string.
                gradcam_bytes = base64.b64decode(gradcam_data["gradcam_image"])
                gradcam_image = Image.open(io.BytesIO(gradcam_bytes)).convert("RGB")

                # Original image + Grad-CAM side by side.
                col1, col2 = st.columns(2)

                with col1:
                    st.image(image, caption="Original Image", use_container_width=True)

                with col2:
                    st.image(gradcam_image, caption="Grad-CAM Visualization", use_container_width=True)

                st.markdown(
                    '<div class="gradcam-note">Warmer regions indicate stronger '
                    "model attention.</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="gradcam-disclaimer">Scientific note: Grad-CAM '
                    "visualizes image regions that contributed to the model's "
                    "prediction. It should not be interpreted as a precise "
                    "disease-location detector.</div>",
                    unsafe_allow_html=True,
                )

            except requests.exceptions.RequestException as exc:
                st.warning(f"Grad-CAM service could not be reached: {exc}")

            except Exception as exc:
                st.warning(f"Grad-CAM could not be generated: {exc}")

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Lightweight placeholder — no bordered card. An empty card here
            # was exactly the kind of unfinished-looking empty box being
            # removed; plain caption text is enough to guide the user.
            st.caption("Upload a leaf image and click **Predict Disease** to see results here.")

    # ============================================================
    # DISEASE INFORMATION (full width)
    # ============================================================

    if "prediction" in st.session_state:
        disease = st.session_state["prediction"]
        disease_info = get_disease_info(disease)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Disease Information</div>', unsafe_allow_html=True)

        # Row 1 — Plant Species | Identified Disease
        row1_col1, row1_col2 = st.columns(2, gap="large")
        with row1_col1:
            st.markdown('<div class="field-label">Plant Species</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="field-box">{disease_info["plant"]}</div>', unsafe_allow_html=True)
        with row1_col2:
            st.markdown('<div class="field-label">Identified Disease</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="field-box">{disease_info["disease"]}</div>', unsafe_allow_html=True)

        # Row 2 — Description (full width, needs the room for readability)
        st.markdown('<div class="field-label">Description</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="field-box">{disease_info["description"]}</div>', unsafe_allow_html=True)

        # Row 3 — Primary Cause | Management
        row2_col1, row2_col2 = st.columns(2, gap="large")
        with row2_col1:
            st.markdown('<div class="field-label">Primary Cause</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="field-box">{disease_info["cause"]}</div>', unsafe_allow_html=True)
        with row2_col2:
            st.markdown('<div class="field-label">Management</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="field-box">{disease_info["management"]}</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="disclaimer">This information is provided for educational purposes '
            "and should not replace professional agricultural diagnosis.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # --------------------------------------------------------
        # CONFIDENCE STATUS — compact banner, thresholds unchanged
        # --------------------------------------------------------
        confidence = st.session_state["confidence"]

        if confidence < 0.60:
            status_class, status_text = "tag-low", (
                "Low confidence — the model isn't sufficiently confident about "
                "this prediction. Try a clearer image of the leaf."
            )
        elif confidence < 0.80:
            status_class, status_text = "tag-moderate", (
                "Moderate confidence — this prediction may benefit from "
                "verification with another clear image."
            )
        else:
            status_class, status_text = "tag-high", "High confidence prediction."

        st.markdown(
            f'<div class="status-banner {status_class}">{status_text}</div>',
            unsafe_allow_html=True,
        )

with tab2:
    render_model_comparison()

with tab3:
    render_prediction_history()