import numpy as np
from fastapi.testclient import TestClient

from src.api import FEATURE_NAMES, create_app


VALID_PATIENT = {
    "HighBP": 1,
    "HighChol": 1,
    "CholCheck": 1,
    "BMI": 33.0,
    "Smoker": 0,
    "Stroke": 0,
    "HeartDiseaseorAttack": 0,
    "PhysActivity": 0,
    "Fruits": 1,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": 0,
    "GenHlth": 4,
    "MentHlth": 2,
    "PhysHlth": 5,
    "DiffWalk": 0,
    "Sex": 1,
    "Age": 9,
    "Education": 5,
    "Income": 6,
}


class FakeModel:
    def __init__(self):
        self.received_columns = None

    def predict_proba(self, frame):
        self.received_columns = frame.columns.tolist()
        return np.array([[0.23, 0.77]])


def test_health_and_ui_are_available_with_loaded_model():
    app = create_app(model_loader=FakeModel)

    with TestClient(app) as client:
        health = client.get("/health")
        page = client.get("/")

    assert health.json() == {"status": "ok", "model": "catboost"}
    assert page.status_code == 200
    assert "Оценка риска диабета" in page.text
    assert "CatBoost" in page.text
    assert "/docs" in page.text
    assert "18-24 года" in page.text
    assert "80 лет и старше" in page.text
    assert "Оконченное высшее образование" in page.text
    assert "Менее $10 000" in page.text
    assert "$75 000 и более" in page.text


def test_predict_returns_risk_probability_and_uses_training_feature_order():
    model = FakeModel()
    app = create_app(model_loader=lambda: model)

    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_PATIENT)

    assert response.status_code == 200
    assert response.json() == {
        "prediction": 1,
        "label": "Есть риск диабета",
        "risk_probability": 0.77,
        "threshold": 0.5,
    }
    assert model.received_columns == FEATURE_NAMES


def test_predict_rejects_out_of_range_bmi():
    app = create_app(model_loader=FakeModel)
    patient = {**VALID_PATIENT, "BMI": 150}

    with TestClient(app) as client:
        response = client.post("/predict", json=patient)

    assert response.status_code == 422
