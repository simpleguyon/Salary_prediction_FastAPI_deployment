# Salary Prediction MLOps Project

This project trains three salary prediction models, tracks them with MLflow,
serves the best model with FastAPI, and generates an Evidently AI drift report
from production-like prediction requests.

## Project Layout

- `Capstone_project_data.csv` - source salary dataset
- `src/train.py` - trains and logs Linear Regression, Decision Tree, and Random Forest models
- `src/app.py` - FastAPI production-like prediction service
- `src/monitoring.py` - Evidently drift report generation
- `src/utils.py` - shared paths, data loading, MLflow setup, and request logging
- `artifacts/best_model.json` - generated metadata for the selected model
- `data/production_requests.csv` - generated API request log
- `reports/data_drift_report.html` - generated Evidently report
- `mlruns/` - local MLflow tracking store

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv mlops_project_env
.\mlops_project_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

The generated folders `artifacts/`, `data/`, `reports/`, and `mlruns/` are ignored by Git and are recreated when you train, serve, and monitor the model.

## Train And Register The Best Model

```powershell
.\mlops_project_env\Scripts\python.exe src\train.py
```

The best model is selected by lowest test RMSE and registered as
`SalaryPredictionModel`.

## Run The API

```powershell
.\mlops_project_env\Scripts\python.exe -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Health check:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/health
```

Prediction:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body '{"YearsExperience":5.0}'
```

## Generate Evidently Drift Report

Call the `/predict` endpoint at least once, then run:

```powershell
.\mlops_project_env\Scripts\python.exe src\monitoring.py
```

Open the generated report at:

```text
reports/data_drift_report.html
```
