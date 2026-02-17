# api/app.py
from __future__ import annotations

import json
import time
import joblib
import pandas as pd
from flask import Flask, request, jsonify

from src.feature_engineering import make_features

MODEL_PATH = "models/model.pkl"
FEATURES_PATH = "models/feature_columns.json"
THRESH_PATH = "models/threshold.json"

app = Flask(__name__)

# Load artifacts once at startup (fast inference)
model = joblib.load(MODEL_PATH)
feature_cols = json.load(open(FEATURES_PATH, "r", encoding="utf-8"))
threshold = float(json.load(open(THRESH_PATH, "r", encoding="utf-8"))["threshold"])


def risk_level(p: float) -> str:
    if p >= 0.8:
        return "HIGH"
    if p >= 0.5:
        return "MEDIUM"
    return "LOW"


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "message": "FactoryGuard AI is running",
            "endpoints": {
                "health": "/health",
                "predict": "/predict (POST) | optional query: ?threshold=0.5",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    start = time.perf_counter()

    payload = request.get_json(force=True, silent=True) or {}

    # Optional threshold override: /predict?threshold=0.5
    override_t = request.args.get("threshold", default=None, type=float)
    used_threshold = override_t if override_t is not None else threshold

    # Validate input
    history = payload.get("history", None)
    if not isinstance(history, list) or len(history) < 10:
        return (
            jsonify(
                {
                    "error": "history list required (minimum 10 rows). Recommended: 60-200 rows for rolling features."
                }
            ),
            400,
        )

    df = pd.DataFrame(history)

    required_cols = ["timestamp", "temperature", "vibration", "pressure"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return jsonify({"error": f"missing columns: {missing_cols}"}), 400

    # Build features (same logic as training)
    df = make_features(df)

    # Last row represents current state
    X_last = df[feature_cols].iloc[[-1]]

    prob = float(model.predict_proba(X_last)[0, 1])
    pred = int(prob >= used_threshold)

    lat_ms = (time.perf_counter() - start) * 1000.0

    return jsonify(
        {
            "failure_probability": prob,
            "prediction": pred,
            "risk_level": risk_level(prob),
            "threshold_default": threshold,
            "used_threshold": used_threshold,
            "latency_ms": round(lat_ms, 2),
        }
    )


if __name__ == "__main__":
    # 0.0.0.0 => accessible on LAN via your IP (e.g., http://192.168.x.x:5000)
    app.run(host="0.0.0.0", port=5000, debug=True)
