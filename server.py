"""
server.py  -  ML server + dashboard backend (run on the laptop, same network as the ESP32)

    pip install fastapi uvicorn xgboost pandas numpy
    python server.py
Dashboard: http://<laptop IP>:8000/      Health: /health
Keep irrigation_model.json and dashboard.html in the same folder as this file.
"""
import json
import math
import os
import threading
import time
import urllib.request
from collections import deque
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from pathlib import Path

import numpy as np
import uvicorn
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

BASE = Path(__file__).parent
MODEL_PATH = BASE / "irrigation_model.json"

# placeholders, same as the sketch; override with environment variables if you like
OWM_API_KEY = os.getenv("OWM_API_KEY", "0b421b2fa8a4e8e9f016f56df19ffe5a")
OWM_LAT = os.getenv("OWM_LAT", "12.97")
OWM_LON = os.getenv("OWM_LON", "79.15")

app = FastAPI(title="Irrigation ML Server")
model = xgb.XGBClassifier()
model.load_model(str(MODEL_PATH))

history = deque(maxlen=500)  # every report from the ESP32
_wc = {"t": 0, "data": None}  # weather cache


# ---------------- light estimate from sun position + cloud cover ----------------
def solar_elevation(lat, lon, when=None):
    """Sun height above the horizon in degrees (negative = night). NOAA formula, UTC input."""
    when = when or datetime.now(timezone.utc)
    n = when.timetuple().tm_yday
    hrs = when.hour + when.minute / 60 + when.second / 3600
    g = 2 * math.pi / 365 * (n - 1 + (hrs - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    ha = math.radians((hrs * 60 + eqtime + 4 * float(lon)) / 4 - 180)
    la = math.radians(float(lat))
    s = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    return math.degrees(math.asin(max(-1, min(1, s))))


def estimate_light():
    """Fake LDR value on the 0-4095 'brighter = higher' scale. Returns (light, sun_elevation)."""
    el = solar_elevation(OWM_LAT, OWM_LON)
    if el <= 0:
        return 0, el
    clouds = (_wc["data"] or {}).get("clouds", 0) / 100   # cached OpenWeatherMap cloud cover
    clear_sky = math.sin(math.radians(el))
    cloud_factor = 1 - 0.75 * clouds ** 3
    return int(4095 * clear_sky * cloud_factor), el


# ---------------- leaf wetness estimate (no leaf sensor connected) ----------------
LEAF_SENSOR = False       # set True only if a real, calibrated leaf sensor is wired up
_rain_seen = {"t": 0.0}   # last time the weather feed reported rain


def dew_point(temp, hum):
    """Magnus formula, degrees C."""
    g = math.log(max(hum, 1) / 100) + 17.62 * temp / (243.12 + temp)
    return 243.12 * g / (17.62 - g)


def estimate_leaf(temp, hum):
    """Guess leaf wetness 0-100 from rain, recent rain and dew (air close to its dew point)."""
    w = _wc["data"] or {}
    if w.get("rain_1h", 0) > 0 or w.get("main") in ("Rain", "Drizzle", "Thunderstorm"):
        _rain_seen["t"] = time.time()
        return 85.0
    gap = temp - dew_point(temp, hum)          # small gap = dew likely
    base = 70.0 if gap <= 1 else 40.0 if gap <= 2.5 else 20.0 if gap <= 4 else 8.0
    hrs = (time.time() - _rain_seen["t"]) / 3600
    if hrs < 2:
        base = max(base, 60.0)                 # leaves stay wet for a while after rain
    elif hrs < 4:
        base = max(base, 35.0)
    return base


# ---------------- ML ----------------
@app.get("/predict", response_class=PlainTextResponse)
def predict(
    soil: float = Query(...), temp: float = Query(...), hum: float = Query(...),
    leaf: float = Query(0.0), light: float = Query(...),
    reservoir: float = Query(...), rain: float = Query(...),
):
    if light <= 0:                      # ESP32 has no LDR, so use the sun estimate
        light = estimate_light()[0]
    if not LEAF_SENSOR:                 # no leaf sensor, so estimate it
        leaf = estimate_leaf(temp, hum)
    X = np.array([[soil, temp, hum, leaf, light, reservoir, rain]])
    return str(int(model.predict(X)[0]))


@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"


# ---------------- reports from the ESP32 ----------------
class Report(BaseModel):
    soil: float
    temp: float
    hum: float
    leaf: float = 0.0                 # not sent any more; the server estimates it
    light: int
    tank: float                       # -1 = HC-SR04 gave no echo
    rain: float                       # next 3-hour block (what the model was trained on)
    rain_max: Optional[float] = None  # highest chance over the next 12 h (safety rule)
    ml: int                           # 1 / 0, or -1 if the model was not asked
    pump: bool
    reason: str                       # the firmware's own reason string
    soil_start: float = 35.0
    soil_stop: float = 55.0
    rain_limit: float
    tank_limit: float
    session_min: float = 0.0
    session_cap_min: float = 20.0
    on_today_min: float = 0.0
    daily_cap_min: float = 60.0
    cool_left_min: float = 0.0
    cycle_s: int


@app.post("/report")
def report(r: Report):
    d = r.model_dump() if hasattr(r, "model_dump") else r.dict()
    d["t"] = time.time()
    if d["rain_max"] is None:
        d["rain_max"] = d["rain"]
    if d["light"] <= 0:
        d["light"], d["sun_elev"] = estimate_light()
        d["light_est"] = True
    if not LEAF_SENSOR:
        d["leaf"], d["leaf_est"] = estimate_leaf(d["temp"], d["hum"]), True
    history.append(d)
    return {"ok": True}


def build_checks(r):
    """Same order as the firmware: tank, rain, daily cap, cooldown, soil, model (+ session length while watering)."""
    tank_err = r["tank"] < 0
    run = r["pump"]
    cool = r["cool_left_min"]
    if run:
        ml_ok, ml_now = True, "not re-checked while watering"
    else:
        ml_ok = r["ml"] == 1
        ml_now = {1: "says irrigate", 0: "says no", -1: "not asked yet"}[r["ml"]]
    checks = [
        dict(key="tank", label="Water tank", ok=(not tank_err) and r["tank"] >= r["tank_limit"],
             now="sensor error" if tank_err else f"{r['tank']:.0f}%", need=f"at least {r['tank_limit']:.0f}%"),
        dict(key="rain", label="Rain, next 12 h (highest chance)", ok=r["rain_max"] < r["rain_limit"],
             now=f"{r['rain_max']:.0f}%", need=f"below {r['rain_limit']:.0f}%"),
        dict(key="daily", label="Pump time today", ok=r["on_today_min"] < r["daily_cap_min"],
             now=f"{r['on_today_min']:.0f} of {r['daily_cap_min']:.0f} min", need="under the daily limit"),
        dict(key="cool", label="Cooldown", ok=cool <= 0,
             now=f"{cool:.0f} min left" if cool > 0 else "clear", need="no recent session"),
        dict(key="soil", label="Soil moisture",
             ok=r["soil"] < (r["soil_stop"] if run else r["soil_start"]), now=f"{r['soil']:.0f}%",
             need=f"below {r['soil_stop']:.0f}% (target)" if run else f"below {r['soil_start']:.0f}% to start"),
        dict(key="ml", label="ML model", ok=ml_ok, now=ml_now, need="says irrigate"),
    ]
    if run:
        checks.append(dict(key="session", label="Session length", ok=r["session_min"] < r["session_cap_min"],
                           now=f"{r['session_min']:.0f} min", need=f"under {r['session_cap_min']:.0f} min"))
    return checks


def explain(r, checks):
    why, soil, stop = r["reason"], r["soil"], r["soil_stop"]
    if r["pump"]:
        if why == "watering in progress":
            return (f"Pump should stay ON: watering until soil reaches {stop:.0f}% (now {soil:.0f}%, "
                    f"session {r['session_min']:.0f} of {r['session_cap_min']:.0f} min).")
        return (f"Pump should be ON: soil is dry ({soil:.0f}% < {r['soil_start']:.0f}%), tank has {r['tank']:.0f}%, "
                f"no heavy rain expected, and the model agrees. It stays on until soil reaches {stop:.0f}%.")
    special = {
        "soil reached target": f"Pump is OFF: soil reached the {stop:.0f}% target ({soil:.0f}%). Watering done.",
        "session time limit": (f"Pump is OFF: it ran for the {r['session_cap_min']:.0f}-minute limit and soil never "
                               f"reached {stop:.0f}%. Check the probe and the pipe."),
        "soil rising fast": f"Pump is OFF: soil is rising quickly ({soil:.0f}%), stopping early so it does not overshoot {stop:.0f}%.",
        "daily limit reached": f"Pump is OFF: it has already run {r['on_today_min']:.0f} min today (limit {r['daily_cap_min']:.0f} min).",
        "cooling down": f"Pump is OFF: cooling down after the last session, {r['cool_left_min']:.0f} min left, so the water can soak in.",
    }
    if why in special:
        return special[why]
    bad = next((c for c in checks if not c["ok"]), None)
    if bad is None:
        return "Pump is off (no blocking rule found)."
    k = bad["key"]
    if k == "tank":
        if r["tank"] < 0:
            return "Pump is OFF: the tank sensor gives no echo, so the level is unknown. The pump stays off for safety."
        return f"Pump is OFF: tank is at {r['tank']:.0f}%, below the {r['tank_limit']:.0f}% safety minimum."
    if k == "rain":
        return (f"Pump is OFF: there is a {r['rain_max']:.0f}% chance of rain in the next 12 h "
                f"(limit {r['rain_limit']:.0f}%), so rain may water the field.")
    if k == "soil":
        return f"Pump is OFF: soil is at {soil:.0f}%, above the {r['soil_start']:.0f}% level that starts watering."
    return "Pump is OFF: soil is dry, but the ML model says not to irrigate right now."


@app.get("/api/status")
def status():
    if not history:
        return {"report": None}
    r = history[-1]
    checks = build_checks(r)
    age = time.time() - r["t"]
    return {
        "report": r,
        "age_s": round(age),
        "online": age < max(60, r["cycle_s"] * 2.5),
        "reason": explain(r, checks),
        "checks": checks,
    }


@app.get("/api/history")
def get_history(n: int = 200):
    return [dict(t=h["t"], soil=h["soil"], tank=h["tank"], temp=h["temp"], pump=h["pump"])
            for h in list(history)[-n:]]


# ---------------- weather (cached 10 min, key stays on the server) ----------------
def _get(url):
    with urllib.request.urlopen(url, timeout=8) as resp:
        return json.loads(resp.read())


@app.get("/api/weather")
def weather():
    now = time.time()
    if _wc["data"] and now - _wc["t"] < 600:
        return _wc["data"]
    try:
        q = f"lat={OWM_LAT}&lon={OWM_LON}&appid={OWM_API_KEY}&units=metric"
        c = _get(f"https://api.openweathermap.org/data/2.5/weather?{q}")
        f = _get(f"https://api.openweathermap.org/data/2.5/forecast?{q}&cnt=4")
        data = {
            "ok": True, "place": c.get("name", ""), "lat": OWM_LAT, "lon": OWM_LON,
            "main": c["weather"][0]["main"], "desc": c["weather"][0]["description"],
            "temp": c["main"]["temp"], "feels": c["main"]["feels_like"],
            "hum": c["main"]["humidity"], "pressure": c["main"]["pressure"],
            "wind": c["wind"]["speed"], "clouds": c["clouds"]["all"],
            "rain_1h": c.get("rain", {}).get("1h", 0),
            "pops": [round(x.get("pop", 0) * 100) for x in f["list"]],
            "updated": now,
        }
        _wc.update(t=now, data=data)
        return data
    except Exception as e:
        return _wc["data"] or {"ok": False, "error": str(e)}


# ---------------- harvest planner (stored in farm.json, no extra hardware) ----------------
FARM = BASE / "farm.json"
# typical days from sowing to harvest (min, max). Starting points only: check with your seed packet / local agri office.
CROPS = {
    "Radish": (25, 40), "Spinach": (40, 50), "Beans (bush)": (50, 60), "Okra": (50, 65),
    "Cucumber": (50, 70), "Cabbage": (70, 100), "Maize": (80, 110), "Tomato": (90, 120),
    "Soybean": (90, 120), "Potato": (90, 120), "Rice": (90, 150), "Onion": (100, 150),
    "Brinjal": (100, 130), "Groundnut": (100, 130), "Wheat": (110, 150), "Cotton": (150, 180),
}


def load_farm():
    try:
        return json.loads(FARM.read_text())
    except Exception:
        return {"crop": None, "sown": None, "custom_days": None, "log": []}


def save_farm(f):
    FARM.write_text(json.dumps(f, indent=2))


def harvest_window(f):
    """(first_day, last_day, source text). Uses the farmer's own number, else the table, then learns from logged harvests."""
    if f.get("custom_days"):
        c = f["custom_days"]
        lo, hi, src = round(c * 0.95), round(c * 1.05), "your own number"
    else:
        lo, hi = CROPS[f["crop"]]
        src = "typical range"
    done = [x["days"] for x in f["log"] if x["crop"] == f["crop"]]
    if done:
        avg = sum(done) / len(done)
        w = max(2, (hi - lo) / 2)
        lo, hi, src = round(avg - w), round(avg + w), f"learned from {len(done)} of your harvests"
    return lo, hi, src


class Plan(BaseModel):
    crop: str
    sown: str                       # YYYY-MM-DD
    custom_days: Optional[int] = None


@app.get("/api/crops")
def crops():
    return CROPS


@app.get("/api/harvest")
def harvest_get():
    f = load_farm()
    if not f["crop"]:
        return {"set": False}
    lo, hi, src = harvest_window(f)
    sown = date.fromisoformat(f["sown"])
    day = (date.today() - sown).days
    return {
        "set": True, "crop": f["crop"], "sown": f["sown"], "day": day, "lo": lo, "hi": hi,
        "ready_from": (sown + timedelta(lo)).isoformat(), "ready_by": (sown + timedelta(hi)).isoformat(),
        "days_left": max(0, lo - day), "pct": max(0, min(100, round(day / lo * 100))),
        "stage": "growing" if day < lo else "window" if day <= hi else "overdue", "source": src,
    }


@app.post("/api/harvest")
def harvest_set(p: Plan):
    try:
        date.fromisoformat(p.sown)
    except ValueError:
        raise HTTPException(400, "Sowing date must look like 2026-09-01")
    if p.crop not in CROPS and not p.custom_days:
        raise HTTPException(400, "Unknown crop: enter the days to harvest")
    if p.custom_days is not None and p.custom_days < 1:
        raise HTTPException(400, "Days to harvest must be at least 1")
    f = load_farm()
    f.update(crop=p.crop, sown=p.sown, custom_days=p.custom_days or None)
    save_farm(f)
    return {"ok": True}


@app.post("/api/harvest/done")
def harvest_done():
    f = load_farm()
    if f["crop"]:
        days = (date.today() - date.fromisoformat(f["sown"])).days
        f["log"].append({"crop": f["crop"], "days": days, "date": date.today().isoformat()})
        f.update(crop=None, sown=None, custom_days=None)
        save_farm(f)
    return {"ok": True}


def _weather_loop():
    """Keep the weather cache fresh so cloud cover and rain are current even if no browser is open."""
    while True:
        try:
            weather()
        except Exception:
            pass
        time.sleep(600)


threading.Thread(target=_weather_loop, daemon=True).start()


@app.get("/")
def dashboard():
    return FileResponse(BASE / "dashboard.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)