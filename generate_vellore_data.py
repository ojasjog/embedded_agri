"""
generate_vellore_data.py

There is no public dataset of real soil-moisture / leaf-wetness / irrigation-decision
readings for Vellore -- that only exists if you've had sensors logging in the field.

What this script does instead: it generates a synthetic dataset for one full year
(hourly-ish samples), grounded in Vellore's REAL climate normals (monthly avg temp,
humidity, rainfall, rainy-day frequency) pulled from climate records. Soil moisture,
leaf wetness, light and reservoir level are then simulated with simple physical
relationships to temp/humidity/rain (dry+hot -> soil dries faster, rain -> soil/leaf
wetness rise, night -> no sunlight, etc.), plus random noise.

The irrigation label is generated from an agronomic rule (not the ML model itself --
that would be circular) with noise, so XGBoost has to learn the underlying pattern
rather than memorize a lookup table.

Swap this out for real logged data once you have hardware deployed -- same column
schema, just replace the CSV.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# Real Vellore monthly climate normals (avg temp C, avg humidity %, avg rainfall mm,
# rainy days/month) -- sourced from climate records (climate-data.org / weather-atlas).
MONTHLY_CLIMATE = {
    1:  dict(temp=22.6, hum=69, rain_mm=12,  rain_days=2),
    2:  dict(temp=24.6, hum=59, rain_mm=8,   rain_days=2),
    3:  dict(temp=27.5, hum=54, rain_mm=14,  rain_days=2),
    4:  dict(temp=29.9, hum=56, rain_mm=23,  rain_days=3),
    5:  dict(temp=31.2, hum=52, rain_mm=64,  rain_days=8),
    6:  dict(temp=29.5, hum=57, rain_mm=89,  rain_days=10),
    7:  dict(temp=28.9, hum=58, rain_mm=93,  rain_days=10),
    8:  dict(temp=28.3, hum=61, rain_mm=112, rain_days=11),
    9:  dict(temp=27.8, hum=65, rain_mm=138, rain_days=12),
    10: dict(temp=26.0, hum=74, rain_mm=177, rain_days=14),
    11: dict(temp=23.9, hum=77, rain_mm=137, rain_days=11),
    12: dict(temp=22.5, hum=74, rain_mm=71,  rain_days=6),
}

N_DAYS = 365
SAMPLES_PER_DAY = 8  # every 3 hours, roughly matching a 5-min-cycle device sampled down
RESERVOIR_HEIGHT_CM = 30.0

rows = []
soil = 55.0        # start at a moderate soil moisture, %
reservoir = 80.0    # start reservoir near full, %

day_of_year = 0
for day in range(N_DAYS):
    month = (day // 30) % 12 + 1
    clim = MONTHLY_CLIMATE[month]

    # does it rain today? probability derived from rain_days / days_in_month
    p_rain_today = clim["rain_days"] / 30.0
    raining_today = np.random.rand() < p_rain_today
    today_rain_total = 0.0
    if raining_today:
        # skew rainfall amount using an exponential-ish draw around the monthly avg/rain_days
        today_rain_total = np.random.exponential(clim["rain_mm"] / max(clim["rain_days"], 1)) 

    for s in range(SAMPLES_PER_DAY):
        hour = s * (24 // SAMPLES_PER_DAY)
        is_daytime = 6 <= hour <= 18

        # temperature: monthly avg +/- diurnal swing +/- noise
        diurnal = 6 * np.sin((hour - 6) / 24 * 2 * np.pi) if is_daytime else -3
        temp = clim["temp"] + diurnal + np.random.normal(0, 1.2)

        # humidity: monthly avg, higher at night/early morning, lower midday
        hum = clim["hum"] + (10 if not is_daytime else -8) + np.random.normal(0, 4)
        hum = float(np.clip(hum, 15, 100))

        # rain forecast probability (%) for the next window -- higher if raining today
        # or if humidity is high (proxy for an approaching front)
        rain_forecast = np.clip(
            (60 if raining_today else 10) + (hum - clim["hum"]) * 1.5 + np.random.normal(0, 12),
            0, 100
        )

        # solar / light intensity (LDR raw-ish scale 0-4095), 0 at night, peaks midday,
        # damped on rainy days (cloud cover)
        if is_daytime:
            sun_curve = np.sin((hour - 6) / 12 * np.pi)
            light = 4095 * sun_curve * (0.4 if raining_today else 1.0)
            light += np.random.normal(0, 150)
        else:
            light = np.random.normal(20, 15)
        light = int(np.clip(light, 0, 4095))

        # soil moisture: dries out with heat/sun, replenished by rain + irrigation
        evap = 0.15 * max(temp - 20, 0) * (1.3 if is_daytime else 0.5)
        soil -= evap
        if raining_today:
            soil += today_rain_total * 0.08  # rain infiltration, spread across samples
        soil += np.random.normal(0, 0.8)
        soil = float(np.clip(soil, 2, 98))

        # leaf wetness: rises sharply with rain/high humidity, dries in sun
        leaf = 0.0
        if raining_today:
            leaf = np.clip(60 + np.random.normal(0, 15), 0, 100)
        elif hum > 80:
            leaf = np.clip(30 + np.random.normal(0, 10), 0, 100)
        else:
            leaf = np.clip(10 + np.random.normal(0, 8), 0, 100)

        # reservoir level: drains a little each sample (evaporation + prior irrigation),
        # refilled by rain; assume periodic manual refills too
        reservoir -= np.random.uniform(0.05, 0.3)
        if raining_today:
            reservoir += today_rain_total * 0.1
        if np.random.rand() < 0.01:  # occasional manual refill
            reservoir = min(100, reservoir + np.random.uniform(20, 40))
        reservoir = float(np.clip(reservoir, 0, 100))

        # ---- label: should the pump turn on? (agronomic rule + noise) ----
        should_irrigate = (
            soil < 45
            and leaf < 50
            and rain_forecast < 70
            and reservoir > 10
            and light < 3000
        )
        label = int(should_irrigate)
        if np.random.rand() < 0.03:  # label noise, mimics imperfect real-world outcomes
            label = 1 - label

        rows.append(dict(
            soil=round(soil, 2),
            temp=round(temp, 2),
            hum=round(hum, 2),
            leaf=round(leaf, 2),
            light=light,
            reservoir=round(reservoir, 2),
            rain=round(rain_forecast, 2),
            irrigate=label,
        ))

df = pd.DataFrame(rows)
df.to_csv("vellore_training_data.csv", index=False)
print(f"Generated {len(df)} rows -> vellore_training_data.csv")
print(df["irrigate"].value_counts(normalize=True))
