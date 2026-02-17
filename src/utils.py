import pandas as pd

EXPECTED_COLS = ["timestamp", "temperature", "vibration", "pressure", "failure"]

def load_sensor_csv(path: str) -> pd.DataFrame:
    """
    Handles normal CSV and the 'quoted single-column' CSV format:
    "timestamp,temperature,vibration,pressure,failure"
    "2024-01-01 00:00:00,64.97,0.435,135.8,0"
    """
    df = pd.read_csv(path)

    # If already correct
    if all(c in df.columns for c in EXPECTED_COLS):
        return df

    # If file loaded as single column, split it
    if df.shape[1] == 1:
        col0 = df.columns[0]
        # Combine header + rows into a single Series
        s = pd.concat([pd.Series([col0]), df.iloc[:, 0]], ignore_index=True)

        # Remove surrounding quotes and split by comma
        rows = s.astype(str).str.strip().str.strip('"').str.split(",", expand=True)

        # First row is header
        header = rows.iloc[0].tolist()
        data = rows.iloc[1:].copy()
        data.columns = [h.strip() for h in header]

        # Keep only expected cols (in correct order)
        data = data[EXPECTED_COLS]

        # Convert dtypes
        data["temperature"] = pd.to_numeric(data["temperature"], errors="coerce")
        data["vibration"]   = pd.to_numeric(data["vibration"], errors="coerce")
        data["pressure"]    = pd.to_numeric(data["pressure"], errors="coerce")
        data["failure"]     = pd.to_numeric(data["failure"], errors="coerce").fillna(0).astype(int)

        return data.dropna(subset=["timestamp"])

    raise ValueError(f"CSV columns not recognized: {list(df.columns)}")
