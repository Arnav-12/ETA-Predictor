"""
FastAPI inference service for delivery ETA prediction.
"""

import os
import sys
import pickle
import numpy as np
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="Delivery ETA Predictor API",
    description="Predict delivery time in minutes based on order features",
    version="1.0.0",
)

model_artifact = None


class DeliveryOrder(BaseModel):
    store_lat: float = Field(..., ge=17.0, le=18.0)
    store_lng: float = Field(..., ge=78.0, le=79.0)
    customer_lat: float = Field(..., ge=17.0, le=18.0)
    customer_lng: float = Field(..., ge=78.0, le=79.0)
    haversine_distance_km: float = Field(..., gt=0)
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)
    is_peak_hour: int = Field(..., ge=0, le=1)
    weather_condition: str = Field(...)
    traffic_index: float = Field(..., ge=0, le=1)
    num_items: int = Field(..., ge=1, le=50)
    order_value_inr: float = Field(..., gt=0)
    store_type: str = Field(...)
    rider_experience_days: int = Field(..., ge=0, le=2000)
    concurrent_orders_at_store: int = Field(..., ge=0, le=50)


class PredictionResponse(BaseModel):
    predicted_minutes: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    model_version: str
    model_name: str


@app.on_event("startup")
def load_model():
    global model_artifact
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'models', 'best_model.pkl'
    )
    if not os.path.exists(model_path):
        print(f"WARNING: Model file not found at {model_path}")
        return
    with open(model_path, 'rb') as f:
        model_artifact = pickle.load(f)
    print(f"Loaded model: {model_artifact['model_name']}")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_artifact is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(order: DeliveryOrder):
    if model_artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    from src.features import (
        haversine_vec, add_cyclical_features, add_interaction_features,
        one_hot_encode, apply_log_transforms, add_h3_index,
    )
    import pandas as pd

    row = order.model_dump()
    df = pd.DataFrame([row])

    df['haversine_distance_km'] = haversine_vec(
        df['store_lat'], df['store_lng'], df['customer_lat'], df['customer_lng']
    )
    df = add_h3_index(df)
    df = add_cyclical_features(df)
    df = add_interaction_features(df)
    df = one_hot_encode(df)
    df = apply_log_transforms(df)

    feature_names = model_artifact['feature_names']
    for c in feature_names:
        if c not in df.columns:
            df[c] = 0

    X = df[feature_names].values.astype(np.float64)
    pred = float(model_artifact['model'].predict(X)[0])
    std = model_artifact['train_residuals_std']

    return PredictionResponse(
        predicted_minutes=round(pred, 2),
        confidence_interval_lower=round(max(8, pred - std), 2),
        confidence_interval_upper=round(min(60, pred + std), 2),
        model_version="1.0.0",
        model_name=model_artifact['model_name'],
    )


if __name__ == '__main__':
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
