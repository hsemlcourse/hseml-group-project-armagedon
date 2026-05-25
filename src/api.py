from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.modeling import FEATURE_NAMES


MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models/diabetes_catboost.cbm"
)
UI_PATH = Path(__file__).resolve().parent / "templates/index.html"


class PatientFeatures(BaseModel):
    HighBP: Literal[0, 1]
    HighChol: Literal[0, 1]
    CholCheck: Literal[0, 1]
    BMI: float = Field(ge=10, le=100)
    Smoker: Literal[0, 1]
    Stroke: Literal[0, 1]
    HeartDiseaseorAttack: Literal[0, 1]
    PhysActivity: Literal[0, 1]
    Fruits: Literal[0, 1]
    Veggies: Literal[0, 1]
    HvyAlcoholConsump: Literal[0, 1]
    AnyHealthcare: Literal[0, 1]
    NoDocbcCost: Literal[0, 1]
    GenHlth: int = Field(ge=1, le=5)
    MentHlth: int = Field(ge=0, le=30)
    PhysHlth: int = Field(ge=0, le=30)
    DiffWalk: Literal[0, 1]
    Sex: Literal[0, 1]
    Age: int = Field(ge=1, le=13)
    Education: int = Field(ge=1, le=6)
    Income: int = Field(ge=1, le=8)


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    label: str
    risk_probability: float
    threshold: float = 0.5


def load_model():
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model


def create_app(model_loader: Callable[[], object] = load_model) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = model_loader()
        yield
        app.state.model = None

    app = FastAPI(
        title="Diabetes Risk Prediction API",
        description="Binary risk screening based on BRFSS 2015 indicators.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def interface() -> str:
        return UI_PATH.read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": "catboost"}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(features: PatientFeatures) -> PredictionResponse:
        values = (
            features.model_dump()
            if hasattr(features, "model_dump")
            else features.dict()
        )
        frame = pd.DataFrame(
            [[values[name] for name in FEATURE_NAMES]],
            columns=FEATURE_NAMES,
        )
        probability = float(app.state.model.predict_proba(frame)[0, 1])
        prediction = int(probability >= 0.5)
        return PredictionResponse(
            prediction=prediction,
            label="Есть риск диабета" if prediction else "Низкий риск диабета",
            risk_probability=round(probability, 4),
        )

    return app


app = create_app()
