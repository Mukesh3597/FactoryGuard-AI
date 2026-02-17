# FactoryGuard AI — Predictive Maintenance (Time-Series + MLOps-Ready API)

FactoryGuard AI is a production-style predictive maintenance system that predicts machine failure risk from sensor time-series data (temperature, vibration, pressure).  
It includes time-series feature engineering (rolling stats + lag + EMA), imbalance-aware XGBoost training, PR-AUC evaluation, threshold selection, and a Flask REST API for real-time inference.

---

## Key Features

- **Time-Series Feature Engineering**
  - Lag features (t-1, t-2)
  - Rolling mean/std windows (60, 360, 720 minutes)
  - EMA features (span=720)
  - Time features: hour, day of week
- **Imbalance Handling**
  - Uses `scale_pos_weight` in XGBoost for rare failures
- **Evaluation (Correct for Imbalance)**
  - PR-AUC (Average Precision)
  - Best threshold selection by F1
  - Confusion Matrix + Classification Report
- **Production-Style API**
  - `/predict` endpoint accepts **history** to compute rolling features
  - Optional `?threshold=` override for precision/recall tuning
  - Returns inference latency (`latency_ms`)
- **Clean Repository**
  - `.gitignore` excludes caches, dataset, and binary artifacts

---

## Repository Structure

FactoryGuard-AI/
├── api/
│ └── app.py
├── src/
│ ├── evaluate.py
│ ├── feature_engineering.py
│ ├── train.py
│ └── utils.py
├── scripts/
│ └── predict_test.py
├── models/
│ ├── feature_columns.json
│ └── threshold.json
├── notebooks/
│ ├── data_analysis.ipynb.ipynb
│ └── model_training.ipynb
├── reports/
├── requirements.txt
└── README.md


> Note: `data/sensor_data_v1.csv` and `models/model.pkl` are intentionally excluded from GitHub (see `.gitignore`).

---

## Requirements

- Python 3.10+ (tested on Python 3.13)
- Install dependencies:

```bash
pip install -r requirements.txt


If needed:

python -m pip install -r requirements.txt

Dataset Format

Create a CSV file at:

data/sensor_data_v1.csv


Expected columns:

column	type
timestamp	string (YYYY-MM-DD HH:MM:SS)
temperature	float
vibration	float
pressure	float
failure	0/1

Example:

"timestamp,temperature,vibration,pressure,failure"
"2024-01-01 00:00:00,64.97,0.435,135.8,0"
"2024-01-01 00:01:00,58.62,0.428,180.4,0"


✅ This repo supports both normal CSV and quoted single-column CSV (handled in src/utils.py).

Training

Run training using module mode:

python -m src.train


Outputs (saved locally):

models/model.pkl (generated locally)

models/feature_columns.json

models/threshold.json

During training you will see:

PR-AUC score

Best threshold

Confusion Matrix & Classification Report

Run the API

Start the Flask server:

python -m api.app


Health check:

GET http://127.0.0.1:5000/health

Home:

GET http://127.0.0.1:5000/

Prediction API
Endpoint

POST /predict

This API requires history (multiple rows) because rolling features are computed from recent sensor values.

Request (JSON)
{
  "history": [
    {"timestamp":"2024-01-01 01:39:00","temperature":57.65,"vibration":0.471,"pressure":138.0},
    {"timestamp":"2024-01-01 01:40:00","temperature":45.85,"vibration":0.209,"pressure":155.5},
    {"timestamp":"2024-01-01 01:41:00","temperature":55.79,"vibration":0.321,"pressure":167.4}
  ]
}


Recommended: send 60–200 rows for stable rolling features.

Response (Example)
{
  "failure_probability": 0.000096,
  "prediction": 0,
  "risk_level": "LOW",
  "threshold_default": 0.3081,
  "used_threshold": 0.3081,
  "latency_ms": 3.42
}

Threshold Override

To tune precision/recall:

POST /predict?threshold=0.50

Quick Predict Test

Run API in one terminal:

python -m api.app


Then in a second terminal:

python -m scripts.predict_test

Results (Sample Run)

Example metrics from a sample run:

PR-AUC: ~0.43

Best threshold: ~0.31

Failure recall: ~0.90 (high recall to avoid missing failures)

Threshold can be increased to reduce false alarms.

Future Improvements

SHAP explainability reports (reports/)

Docker containerization for reproducible deployments

Monitoring + drift detection + scheduled retraining (MLOps)

Hyperparameter tuning (Optuna/GridSearchCV)

Author

Mukesh Pratap
GitHub: https://github.com/Mukesh3597
