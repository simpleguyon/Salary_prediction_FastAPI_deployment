from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "Capstone_project_data.csv"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUNTIME_DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
BEST_MODEL_METADATA_PATH = ARTIFACTS_DIR / "best_model.json"
PRODUCTION_REQUESTS_PATH = RUNTIME_DATA_DIR / "production_requests.csv"

FEATURE_COLUMN = "YearsExperience"
TARGET_COLUMN = "Salary"
REGISTERED_MODEL_NAME = "SalaryPredictionModel"
MLFLOW_EXPERIMENT_NAME = "salary_prediction_capstone"


def ensure_project_dirs() -> None:
    for directory in (MLRUNS_DIR, ARTIFACTS_DIR, RUNTIME_DATA_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def configure_mlflow() -> None:
    ensure_project_dirs()
    mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def load_salary_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH)
    data = data.drop(columns=["Unnamed: 0"], errors="ignore")

    expected_columns = {FEATURE_COLUMN, TARGET_COLUMN}
    missing_columns = expected_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    clean_data = data[[FEATURE_COLUMN, TARGET_COLUMN]].dropna().copy()
    clean_data[FEATURE_COLUMN] = clean_data[FEATURE_COLUMN].astype(float)
    clean_data[TARGET_COLUMN] = clean_data[TARGET_COLUMN].astype(float)
    return clean_data


def save_best_model_metadata(metadata: dict[str, Any]) -> None:
    ensure_project_dirs()
    BEST_MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_best_model_metadata() -> dict[str, Any]:
    if not BEST_MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(
            "Best model metadata was not found. Run training first: "
            ".\\mlops_project_env\\Scripts\\python.exe src\\train.py"
        )
    return json.loads(BEST_MODEL_METADATA_PATH.read_text(encoding="utf-8"))


def append_prediction_log(
    years_experience: float,
    predicted_salary: float,
    model_name: str,
    run_id: str,
) -> None:
    ensure_project_dirs()
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        FEATURE_COLUMN: years_experience,
        "prediction": predicted_salary,
        "model_name": model_name,
        "run_id": run_id,
    }

    file_exists = PRODUCTION_REQUESTS_PATH.exists()
    with PRODUCTION_REQUESTS_PATH.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
