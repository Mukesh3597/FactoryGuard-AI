# src/train.py
from __future__ import annotations

import json
import os
import joblib
import pandas as pd
from xgboost import XGBClassifier

from src.feature_engineering import make_features, get_feature_columns
from src.utils import load_sensor_csv
from src.evaluate import evaluate_binary_classifier, print_eval

# ----------------------------
# Paths
# ----------------------------
DATA_PATH = "data/sensor_data_v1.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.json")
THRESH_PATH = os.path.join(MODEL_DIR, "threshold.json")


def time_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-series split: first (1-test_size) train, last test_size test."""
    n = len(df)
    cut = int(n * (1 - test_size))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1) Load CSV (handles your quoted single-column format too)
    df = load_sensor_csv(DATA_PATH)

    # 2) Feature engineering (timestamp -> hour/dayofweek + lag/rolling/ema)
    df = make_features(df)

    # 3) Prepare features/target
    feature_cols = get_feature_columns()
    target_col = "failure"

    # Safety check
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after feature engineering: {missing}")

    # 4) Time-based split
    train_df, test_df = time_split(df, test_size=0.2)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(int)

    # 5) Handle class imbalance (failures are rare)
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = (neg / (pos + 1e-12))

    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print(f"Failures in train: {pos} | Non-failures: {neg} | scale_pos_weight: {scale_pos_weight:.2f}")

    # 6) Train XGBoost (good default hyperparams; tune later)
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=1,
        gamma=0.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # 7) Predict probabilities + evaluate with PR-AUC + best threshold
    y_prob = model.predict_proba(X_test)[:, 1]
    res = evaluate_binary_classifier(y_test.values, y_prob, threshold=None)
    print_eval(res)

    # 8) Save artifacts
    joblib.dump(model, MODEL_PATH)

    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    with open(THRESH_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": res.best_threshold,
                "pr_auc": res.pr_auc,
                "best_f1": res.best_f1,
            },
            f,
            indent=2,
        )

    print("\n✅ Saved:")
    print(f" - {MODEL_PATH}")
    print(f" - {FEATURES_PATH}")
    print(f" - {THRESH_PATH}")


if __name__ == "__main__":
    main()
