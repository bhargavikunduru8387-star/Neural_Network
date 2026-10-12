
from pathlib import Path

import joblib
import logging
import time
from fastapi import Request
import numpy as np
from collections import Counter
from threading import Lock
from fastapi import FastAPI
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

MODEL_DIR = Path(__file__).resolve().parent / "models"

model = load_model(MODEL_DIR / "iris_neural_network.keras")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

app = FastAPI(title="Iris Flower Prediction API")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iris_api")
metrics = Counter()
metrics_lock = Lock()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        with metrics_lock:
            metrics["requests"] += 1
            if response.status_code >= 400:
                metrics["errors"] += 1
        duration = time.perf_counter() - start
        logger.info(
            "%s %s status=%s duration=%.3fs",
            request.method,
            request.url.path,
            response.status_code,
            duration,
        )
        return response
    except Exception:
        logger.exception("Request failed: %s %s", request.method, request.url.path)
        raise


class FlowerInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


@app.get("/")
def home():
    return {"message": "Iris Prediction API is running - auto-deploy test"}


@app.get("/health")
def health():
    return {"status": "healthy"}
@app.get("/metrics")
def get_metrics():
    with metrics_lock:
        return dict(metrics)


@app.post("/predict")
def predict(flower: FlowerInput):
    features = np.array([[
        flower.sepal_length,
        flower.sepal_width,
        flower.petal_length,
        flower.petal_width,
    ]])

    scaled_features = scaler.transform(features)
    probabilities = model.predict(scaled_features, verbose=0)[0]

    if not np.all(np.isfinite(probabilities)):
        logger.error("Model returned invalid probabilities")
        raise ValueError("Model returned invalid probabilities")

    predicted_index = int(np.argmax(probabilities))

    if not np.isclose(np.sum(probabilities), 1.0, atol=0.01):
        logger.warning("Model probabilities do not sum to approximately 1")

    return {
        "predicted_species": str(encoder.classes_[predicted_index]),
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            str(name): float(probability)
            for name, probability in zip(
                encoder.classes_, probabilities
            )
        },
    }

