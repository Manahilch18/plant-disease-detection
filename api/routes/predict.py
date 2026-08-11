"""
Prediction API route.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import io

from api.dependencies import load_model, load_class_mapping
from api.schemas import PredictionResponse
from src.disease_info import get_disease_info


router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


@router.post("/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict plant disease from an uploaded image.
    """

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image.",
        )

    try:
        # Read uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize to model input size
        image = image.resize((224, 224))

        # IMPORTANT:
        # Baseline CNN was trained using RAW [0,255] pixels.
        image_array = np.array(image, dtype=np.float32)

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Load model and class mapping
        model = load_model()
        class_mapping = load_class_mapping()

        # Prediction
        predictions = model.predict(image_array, verbose=0)

        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_index])

        # Convert class index → disease name
        predicted_class = class_mapping[str(predicted_index)]

        # Get disease information
        info = get_disease_info(predicted_class)

        return PredictionResponse(
            disease=predicted_class,
            confidence=confidence,
            description=info["description"],
            cause=info["cause"],
            management=info["management"],
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc