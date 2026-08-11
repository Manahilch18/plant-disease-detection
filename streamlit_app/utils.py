import os
import requests


def predict_image(image_bytes: bytes, filename: str) -> dict:
    """
    Send an image to the FastAPI prediction endpoint.
    """

    api_base_url = os.getenv(
        "API_BASE_URL",
        "http://localhost:8000",
    )

    files = {
        "file": (
            filename,
            image_bytes,
            "image/jpeg",
        )
    }

    response = requests.post(
        f"{api_base_url}/predict/",
        files=files,
        timeout=60,
    )

    response.raise_for_status()

    return response.json()