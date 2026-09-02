"""
server.py

Runs on your PC (same WiFi network as the ESP32). Loads the trained XGBoost model
and exposes POST /predict, which the ESP32 calls instead of running xgboostPredict()
locally. Returns a bare "0" or "1" as plain text -- easy to parse in Arduino with
no JSON library needed on that leg of the request.

Run:
    pip install fastapi uvicorn xgboost pandas --break-system-packages
    python3 server.py

Then find your PC's LAN IP (e.g. `ipconfig` on Windows / `ifconfig` or `ip a` on
Mac/Linux) and point the ESP32's ML_SERVER_URL at http://<that-ip>:8000/predict
-- the ESP32 and this PC must be on the same WiFi network for this to work.
"""

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
import uvicorn

MODEL_PATH = "irrigation_model.json"
FEATURES = ["soil", "temp", "hum", "leaf", "light", "reservoir", "rain"]

app = FastAPI(title="Irrigation ML Server")

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)


@app.get("/predict", response_class=PlainTextResponse)
def predict(
    soil: float = Query(...),
    temp: float = Query(...),
    hum: float = Query(...),
    leaf: float = Query(...),
    light: float = Query(...),
    reservoir: float = Query(...),
    rain: float = Query(...),
):
    X = np.array([[soil, temp, hum, leaf, light, reservoir, rain]])
    pred = int(model.predict(X)[0])
    # bare "0" or "1" -- the ESP32 side just reads the response body directly
    return str(pred)


@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"


if __name__ == "__main__":
    # host="0.0.0.0" so it's reachable from other devices on the LAN (the ESP32),
    # not just localhost
    uvicorn.run(app, host="0.0.0.0", port=8000)
