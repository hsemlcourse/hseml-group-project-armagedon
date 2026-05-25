from __future__ import annotations

from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier


RANDOM_STATE = 777
TARGET = "Diabetes_Binary"
FEATURE_NAMES = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(csv_path)
    if "Diabetes_012" in data:
        data[TARGET] = data["Diabetes_012"].replace({2: 1})
    elif "Diabetes_binary" in data:
        data[TARGET] = data["Diabetes_binary"].astype(int)
    else:
        raise ValueError(
            "Dataset must contain Diabetes_012 or Diabetes_binary target."
        )
    return data[FEATURE_NAMES], data[TARGET].astype(int)


def build_model() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=500,
        random_seed=RANDOM_STATE,
        auto_class_weights="Balanced",
        verbose=0,
        allow_writing_files=False,
    )


def train_and_export(csv_path: Path, model_path: Path) -> None:
    x, y = load_dataset(csv_path)
    model = build_model().fit(x, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    train_and_export(
        csv_path=project_root
        / "data/raw/diabetes-health-indicators-dataset"
        / "diabetes_012_health_indicators_BRFSS2015.csv",
        model_path=project_root / "models/diabetes_catboost.cbm",
    )
    print("Model saved to models/diabetes_catboost.cbm")
