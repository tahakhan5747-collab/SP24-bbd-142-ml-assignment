"""Load the trained models and provide two prediction routes."""
import json
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

BASE = Path(__file__).resolve().parent
# Run train_models.py first. Only load joblib files you trust.
regression = joblib.load(BASE / "models/regression.joblib")
classification = joblib.load(BASE / "models/classification.joblib")
metrics = json.loads((BASE / "models/metrics.json").read_text())

app = FastAPI(title="Simple Kaggle ML App")


class BillInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    total_bill: float = Field(ge=3, le=51)


class FlowerInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    petal_length: float = Field(ge=1, le=7)
    petal_width: float = Field(ge=0.1, le=2.5)


@app.get("/")
def home():
    return FileResponse(BASE / "static/index.html")


@app.get("/metrics")
def show_metrics():
    return metrics


@app.post("/predict/regression")
def predict_tip(data: BillInput):
    prediction = regression.predict([[data.total_bill]])[0]
    return {"prediction": round(float(prediction), 2), "unit": "USD"}


@app.post("/predict/classification")
def predict_flower(data: FlowerInput):
    prediction = classification.predict([[data.petal_length, data.petal_width]])[0]
    return {"prediction": str(prediction)}
