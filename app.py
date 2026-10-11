
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

MODEL_DIR = Path(__file__).resolve().parent / "models"

model = load_model(MODEL_DIR / "iris_neural_network.keras")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

app = FastAPI(title="Iris Flower Prediction API")


class FlowerInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


@app.get("/")
def home():
    return {"message": "Iris Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


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

    predicted_index = int(np.argmax(probabilities))

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

