from __future__ import annotations

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .utils import (
    FEATURE_COLUMN,
    append_prediction_log,
    configure_mlflow,
    load_best_model_metadata,
)


app = FastAPI(title="Salary Prediction API", version="1.0.0")

model = None
model_metadata = None


class PredictionRequest(BaseModel):
    YearsExperience: float = Field(..., ge=0, description="Years of work experience")


@app.on_event("startup")
def load_model() -> None:
    global model, model_metadata
    configure_mlflow()
    model_metadata = load_best_model_metadata()
    model = mlflow.pyfunc.load_model(model_metadata["model_uri"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    return model_metadata or {}


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict:
    if model is None or model_metadata is None:
        raise RuntimeError("Model is not loaded.")

    input_data = pd.DataFrame([{FEATURE_COLUMN: payload.YearsExperience}])
    prediction = float(model.predict(input_data)[0])

    append_prediction_log(
        years_experience=payload.YearsExperience,
        predicted_salary=prediction,
        model_name=model_metadata["model_name"],
        run_id=model_metadata["run_id"],
    )

    return {
        "YearsExperience": payload.YearsExperience,
        "predicted_salary": prediction,
        "model_name": model_metadata["model_name"],
        "run_id": model_metadata["run_id"],
    }
