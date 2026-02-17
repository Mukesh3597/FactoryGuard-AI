import requests
from src.utils import load_sensor_csv

df = load_sensor_csv("data/sensor_data_v1.csv")

hist = df[["timestamp", "temperature", "vibration", "pressure"]].tail(200).to_dict(orient="records")

r = requests.post("http://127.0.0.1:5000/predict", json={"history": hist})

print("Status:", r.status_code)
print(r.text)
