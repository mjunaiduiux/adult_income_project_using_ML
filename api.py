import json
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

import features  # noqa: F401  (needed so the pickled pipeline can resolve its custom functions)

app = FastAPI(title="Adult Income Predictor", version="1.0")
model = joblib.load("best_model.joblib")
info = json.load(open("model_info.json"))


class Applicant(BaseModel):
    age: int = Field(..., ge=17, le=100, examples=[39])
    workclass: str = Field(..., examples=["Private"])
    education_num: int = Field(..., ge=1, le=16, examples=[13])
    marital_status: str = Field(..., examples=["Married-civ-spouse"])
    occupation: str = Field(..., examples=["Exec-managerial"])
    relationship: str = Field(..., examples=["Husband"])
    race: str = Field(..., examples=["White"])
    sex: str = Field(..., examples=["Male"])
    capital_gain: float = Field(0, ge=0)
    capital_loss: float = Field(0, ge=0)
    hours_per_week: int = Field(..., ge=1, le=99, examples=[45])
    native_country: str = Field(..., examples=["United-States"])


class Prediction(BaseModel):
    prediction: str
    prediction_label: int
    probability_over_50k: Optional[float] = None
    model: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {k: info[k] for k in ["model_name", "test_metrics", "top5_features"]}


@app.post("/predict", response_model=Prediction)
def predict(x: Applicant):
    row = pd.DataFrame([x.model_dump()])
    label = int(model.predict(row)[0])
    proba = float(model.predict_proba(row)[0, 1]) if hasattr(model, "predict_proba") else None
    return Prediction(prediction=">50K" if label == 1 else "<=50K", prediction_label=label,
                      probability_over_50k=proba, model=info["model_name"])
