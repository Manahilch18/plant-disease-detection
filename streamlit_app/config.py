"""
Streamlit application configuration.

Centralized configuration for the Plant Disease Detection UI
and communication with the FastAPI backend.
"""

from __future__ import annotations

import os


# ============================================================
# APPLICATION
# ============================================================

APP_TITLE = "Plant Disease Detection"
APP_ICON = "🌿"


# ============================================================
# FASTAPI BACKEND
# ============================================================

# Local development:
#   http://localhost:8000
#
# Docker Compose:
#   API_BASE_URL=http://api:8000
#
# The environment variable allows the same Streamlit code
# to work both locally and inside Docker Compose.

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://plant-disease-detection-a9062578.fastapicloud.dev"
).rstrip("/")


# ============================================================
# API ENDPOINTS
# ============================================================

PREDICT_ENDPOINT = f"{API_BASE_URL}/predict/"
GRADCAM_ENDPOINT = f"{API_BASE_URL}/gradcam/"


# ============================================================
# REQUEST SETTINGS
# ============================================================

API_TIMEOUT = int(
    os.getenv("API_TIMEOUT", "120")
)


# ============================================================
# MODEL / IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)

# IMPORTANT:
# The Baseline CNN was trained and evaluated using RAW
# pixel values in the range [0, 255].
#
# DO NOT divide input images by 255.
PIXEL_RANGE = "[0, 255]"


# ============================================================
# UI SETTINGS
# ============================================================

LOW_CONFIDENCE_THRESHOLD = 0.60
MODERATE_CONFIDENCE_THRESHOLD = 0.80


# ============================================================
# DEBUGGING
# ============================================================

DEBUG = os.getenv(
    "STREAMLIT_DEBUG",
    "false",
).lower() == "true"