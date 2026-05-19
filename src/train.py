from __future__ import annotations

import math
from typing import Any

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

try:
    from .utils import (
        FEATURE_COLUMN,
        REGISTERED_MODEL_NAME,
        TARGET_COLUMN,
        configure_mlflow,
        load_salary_data,
        save_best_model_metadata,
    )
except ImportError:
    from utils import (
        FEATURE_COLUMN,
        REGISTERED_MODEL_NAME,
        TARGET_COLUMN,
        configure_mlflow,
        load_salary_data,
        save_best_model_metadata,
    )


TEST_SIZE = 0.2
RANDOM_STATE = 42


def build_models() -> dict[str, Any]:
    return {
        "linear_regression": LinearRegression(),
        "decision_tree_regressor": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=4),
        "random_forest_regressor": RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_estimators=100,
            max_depth=4,
        ),
    }


def evaluate_model(model: Any, x_test, y_test) -> dict[str, float]:
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    return {
        "rmse": math.sqrt(mse),
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }


def train_and_log_models() -> dict[str, Any]:
    configure_mlflow()
    data = load_salary_data()

    x = data[[FEATURE_COLUMN]]
    y = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    best_result: dict[str, Any] | None = None

    for model_name, model in build_models().items():
        with mlflow.start_run(run_name=model_name) as run:
            model.fit(x_train, y_train)
            metrics = evaluate_model(model, x_test, y_test)

            params = {
                "model_name": model_name,
                "feature_column": FEATURE_COLUMN,
                "target_column": TARGET_COLUMN,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
            }
            params.update(model.get_params())

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_table(data, artifact_file="cleaned_salary_dataset.json")
            mlflow.sklearn.log_model(model, artifact_path="model")

            result = {
                "model_name": model_name,
                "run_id": run.info.run_id,
                "model_uri": f"runs:/{run.info.run_id}/model",
                "metrics": metrics,
                "params": params,
            }

            if best_result is None or metrics["rmse"] < best_result["metrics"]["rmse"]:
                best_result = result

            print(
                f"{model_name}: "
                f"RMSE={metrics['rmse']:.2f}, "
                f"MAE={metrics['mae']:.2f}, "
                f"R2={metrics['r2']:.4f}"
            )

    if best_result is None:
        raise RuntimeError("No models were trained.")

    registered_model = mlflow.register_model(
        model_uri=best_result["model_uri"],
        name=REGISTERED_MODEL_NAME,
    )

    best_result["registered_model_name"] = REGISTERED_MODEL_NAME
    best_result["registered_model_version"] = registered_model.version
    save_best_model_metadata(best_result)

    print()
    print(
        "Best model: "
        f"{best_result['model_name']} "
        f"(RMSE={best_result['metrics']['rmse']:.2f}, "
        f"run_id={best_result['run_id']})"
    )
    print(f"Registered as: {REGISTERED_MODEL_NAME} v{registered_model.version}")
    return best_result


if __name__ == "__main__":
    train_and_log_models()
