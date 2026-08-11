"""
Pydantic schemas for the Plant Disease Detection API.
"""

from pydantic import BaseModel, Field

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    disease: str
    confidence: float
    description: str
    cause: str
    management: str