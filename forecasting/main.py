from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
import json
import os

app = FastAPI(title="Finance Forecast API")

# Use absolute paths for Railway reliability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'forecast_model.keras')
SCALER_X_PATH = os.path.join(BASE_DIR, 'models', 'scaler_X.save')
SCALER_Y_PATH = os.path.join(BASE_DIR, 'models', 'scaler_y.save')
JSON_PATH = os.path.join(BASE_DIR, 'models', 'ensemble_forecast_results.json')

# Load Model & Scalers
MODEL = tf.keras.models.load_model(MODEL_PATH)
SCALER_X = joblib.load(SCALER_X_PATH)
SCALER_Y = joblib.load(SCALER_Y_PATH)

with open(JSON_PATH, 'r') as f:
    ENSEMBLE_FORECAST_RESULTS = json.load(f)

class PredictRequest(BaseModel):
    lag_1: float
    lag_2: float
    lag_3: float
    rolling_mean_7: float
    rolling_mean_30: float
    day_of_week: int
    month: int
    is_weekend: int
    mtd_progress: float
    transaction_count: int

@app.get("/")
def home():
    return {"message": "Finance Forecast API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: PredictRequest):
    try:
        return {
            "success": True,
            "message": "OK",
            "forecast_next_month": ENSEMBLE_FORECAST_RESULTS["forecast_next_month"],
            "confidence_lower": ENSEMBLE_FORECAST_RESULTS["confidence_lower"],
            "confidence_upper": ENSEMBLE_FORECAST_RESULTS["confidence_upper"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
