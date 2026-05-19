from __future__ import annotations

import sys

import mlflow.pyfunc
import pandas as pd

try:
    from evidently.legacy.metric_preset import DataDriftPreset
    from evidently.legacy.report import Report
except ImportError:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

try:
    from .utils import (
        FEATURE_COLUMN,
        PRODUCTION_REQUESTS_PATH,
        REPORTS_DIR,
        configure_mlflow,
        load_best_model_metadata,
        load_salary_data,
    )
except ImportError:
    from utils import (
        FEATURE_COLUMN,
        PRODUCTION_REQUESTS_PATH,
        REPORTS_DIR,
        configure_mlflow,
        load_best_model_metadata,
        load_salary_data,
    )


REPORT_PATH = REPORTS_DIR / "data_drift_report.html"


def generate_drift_report() -> str:
    reference_data = load_salary_data()[[FEATURE_COLUMN]].copy()

    if not PRODUCTION_REQUESTS_PATH.exists():
        raise FileNotFoundError(
            "No production request data found. Start the API and call /predict first."
        )

    current_data = pd.read_csv(PRODUCTION_REQUESTS_PATH)
    if current_data.empty:
        raise ValueError("Production request log is empty. Call /predict first.")

    required_columns = [FEATURE_COLUMN]
    if "prediction" in current_data.columns:
        configure_mlflow()
        model_metadata = load_best_model_metadata()
        model = mlflow.pyfunc.load_model(model_metadata["model_uri"])
        required_columns.append("prediction")
        reference_data["prediction"] = model.predict(reference_data[[FEATURE_COLUMN]])

    current_data = current_data[required_columns].copy()
    current_data[FEATURE_COLUMN] = current_data[FEATURE_COLUMN].astype(float)
    if "prediction" in current_data.columns:
        current_data["prediction"] = current_data["prediction"].astype(float)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))
    return str(REPORT_PATH)


if __name__ == "__main__":
    try:
        output_path = generate_drift_report()
    except Exception as exc:
        print(f"Monitoring report was not generated: {exc}")
        sys.exit(1)

    print(f"Evidently drift report generated: {output_path}")
