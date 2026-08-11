"""
Main FastAPI application for Plant Disease Detection.
"""     
from fastapi import FastAPI
from api.routes.predict import router as predict_router
from api.routes.gradcam import router as gradcam_router
app = FastAPI(
    title="Plant Disease Detection API",
    description="API for plant disease classification using a trained Baseline CNN.",
    version="1.0.0",)
# Register prediction routes
app.include_router(predict_router)
app.include_router(gradcam_router)
@app.get("/")
def root():
    """API root endpoint."""
    return { "message": "Plant Disease Detection API is running"    }
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "Baseline CNN",
        "model_input": "224x224 RGB",
        "pixel_range": "[0, 255]",
        "num_classes": 38,}
@app.get("/model-info")
def model_info():
    return {
        "model": "Baseline CNN",
        "input_shape": "(224, 224, 3)",
        "output_classes": 38,
        "preprocessing": "Raw float32 pixels [0, 255]", }