"""
Grad-CAM API route for Plant Disease Detection.
"""

import base64
import io

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from api.dependencies import load_model
from src.explainability.gradcam import (
    make_gradcam_heatmap,
    overlay_gradcam,
)

router = APIRouter(
    prefix="/gradcam",
    tags=["Grad-CAM"],
)


@router.post("/")
async def generate_gradcam(file: UploadFile = File(...)):
    """
    Generate a Grad-CAM visualization for an uploaded plant leaf image.
    """

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg",
    ]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image.",
        )

    try:
        # --------------------------------------------------
        # Read uploaded image
        # --------------------------------------------------
        image_bytes = await file.read()

        original_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        # --------------------------------------------------
        # Prepare model input
        # Baseline CNN expects raw [0, 255] float32 pixels.
        # --------------------------------------------------
        resized_image = original_image.resize((224, 224))

        image_array = np.asarray(
            resized_image,
            dtype=np.float32,
        )

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        # --------------------------------------------------
        # Load trained Baseline CNN
        # --------------------------------------------------
        model = load_model()

        # --------------------------------------------------
        # Generate Grad-CAM heatmap
        # --------------------------------------------------
        heatmap = make_gradcam_heatmap(
            model=model,
            image_array=image_array,
            last_conv_layer_name="conv2d_4",
        )

        # --------------------------------------------------
        # Create overlay
        # --------------------------------------------------
        gradcam_image = overlay_gradcam(
            original_image,
            heatmap,
            alpha=0.4,
        )

        # --------------------------------------------------
        # Convert image → PNG bytes → Base64
        # --------------------------------------------------
        output_buffer = io.BytesIO()

        gradcam_image.save(
            output_buffer,
            format="PNG",
        )

        output_buffer.seek(0)

        encoded_image = base64.b64encode(
            output_buffer.getvalue()
        ).decode("utf-8")

        return {
            "success": True,
            "gradcam_image": encoded_image,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Grad-CAM generation failed: {str(exc)}",
        ) from exc