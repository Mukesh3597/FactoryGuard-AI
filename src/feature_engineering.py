import pandas as pd

SENSOR_COLS = ["temperature", "vibration", "pressure"]

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    return df

def add_rolling_and_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Lag features
    for c in SENSOR_COLS:
        df[f"{c}_lag1"] = df[c].shift(1)
        df[f"{c}_lag2"] = df[c].shift(2)

    # Rolling windows: 60, 360, 720 (minutes)
    windows = {
        "roll_60": 60,
        "roll_360": 360,
        "roll_720": 720
    }

    for c in SENSOR_COLS:
        for name, w in windows.items():
            df[f"{c}_{name}_mean"] = df[c].rolling(w, min_periods=1).mean()
            df[f"{c}_{name}_std"]  = df[c].rolling(w, min_periods=1).std()

        # EMA (span ~ 12 hours = 720 mins)
        df[f"{c}_ema_720"] = df[c].ewm(span=720, adjust=False).mean()

    # Fill NaNs (std of first rows etc.)
    df = df.fillna(method="bfill").fillna(0)
    return df

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_rolling_and_lags(df)
    return df

def get_feature_columns() -> list[str]:
    # final feature list (excluding target)
    base = SENSOR_COLS + ["hour", "dayofweek"]
    lag = [f"{c}_lag1" for c in SENSOR_COLS] + [f"{c}_lag2" for c in SENSOR_COLS]
    roll = []
    for c in SENSOR_COLS:
        for w in ["roll_60", "roll_360", "roll_720"]:
            roll += [f"{c}_{w}_mean", f"{c}_{w}_std"]
        roll += [f"{c}_ema_720"]
    return base + lag + roll
