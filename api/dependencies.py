"""
Application dependencies for Plant Disease Detection API.
"""
import json
from pathlib import Path

import tensorflow as tf


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Best trained Baseline CNN
MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "baseline_cnn"
    / "models"
    / "best_model.keras"
)


def load_model() -> tf.keras.Model:
    """
    Load the trained Baseline CNN model.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    return model 

CLASS_MAPPING_PATH = (
        PROJECT_ROOT/ "artifacts"/ "preprocessing"/ "class_mapping.json")

def load_class_mapping() -> dict:
    """
    Load class index to class name mapping.
    """
    if not CLASS_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Class mapping not found: {CLASS_MAPPING_PATH}"
        )

    with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as file:
        return json.load(file)
    
    
    