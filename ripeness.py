"""
ripeness.py - colour-based ripeness detector for tomato and chili.

CLI:     python ripeness.py photo.jpg --crop tomato
Server:  python ripeness.py --serve        (POST an image to /ripeness?crop=chili)

Install: pip install opencv-python numpy fastapi uvicorn python-multipart

Best results: one or a few fruits, plain background (soil, cloth, paper),
decent daylight. Green leaves in frame will count as unripe fruit.
"""
import argparse
import sys

import cv2
import numpy as np

# OpenCV hue range is 0-179. (low, high) inclusive.
HUES = {
    "red":    [(0, 9), (170, 179)],
    "orange": [(10, 21)],
    "yellow": [(22, 34)],
    "green":  [(35, 85)],
}

# Per-crop thresholds on the "ripeness score" (0 = all green, 1 = all red).
# Chili is often harvested when mostly red, so its bar is a bit lower.
THRESHOLDS = {
    "tomato": {"ripe": 0.75, "turning": 0.25},
    "chili":  {"ripe": 0.65, "turning": 0.25},
}


def _hue_mask(h, ranges):
    m = np.zeros(h.shape, np.uint8)
    for lo, hi in ranges:
        m |= ((h >= lo) & (h <= hi)).astype(np.uint8)
    return m.astype(bool)


def analyze(image_bgr, crop="tomato"):
    if crop not in THRESHOLDS:
        raise ValueError(f"crop must be one of {list(THRESHOLDS)}")

    # Downscale for speed and consistent behaviour
    h0, w0 = image_bgr.shape[:2]
    scale = 640 / max(h0, w0)
    if scale < 1:
        image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale)

    blur = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Fruit = saturated, not-too-dark pixels (drops white/grey/black backgrounds)
    fruit = (s > 80) & (v > 50)
    fruit = cv2.morphologyEx(fruit.astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((5, 5), np.uint8)).astype(bool)

    counts = {name: int((fruit & _hue_mask(h, r)).sum()) for name, r in HUES.items()}
    total = sum(counts.values())
    if total < 0.01 * fruit.size:
        return {"crop": crop, "label": "no_fruit_detected", "score": None,
                "breakdown": counts}

    frac = {k: c / total for k, c in counts.items()}
    # red counts fully, orange/yellow count half, green counts zero
    score = frac["red"] + 0.5 * (frac["orange"] + frac["yellow"])

    t = THRESHOLDS[crop]
    label = "ripe" if score >= t["ripe"] else "turning" if score >= t["turning"] else "unripe"
    return {
        "crop": crop,
        "label": label,
        "score": round(score, 2),
        "breakdown": {k: round(v, 2) for k, v in frac.items()},
    }


def analyze_file(path, crop="tomato"):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"could not read image: {path}")
    return analyze(img, crop)


def serve():
    import uvicorn
    from fastapi import FastAPI, File, HTTPException, UploadFile

    app = FastAPI(title="Ripeness Detector")

    @app.post("/ripeness")
    async def ripeness(crop: str = "tomato", file: UploadFile = File(...)):
        data = np.frombuffer(await file.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "not a valid image")
        try:
            return analyze(img, crop)
        except ValueError as e:
            raise HTTPException(400, str(e))

    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--crop", default="tomato", choices=list(THRESHOLDS))
    ap.add_argument("--serve", action="store_true")
    args = ap.parse_args()

    if args.serve:
        serve()
    elif args.image:
        r = analyze_file(args.image, args.crop)
        print(f"{r['crop']}: {r['label'].upper()} (score {r['score']})  {r['breakdown']}")
    else:
        ap.print_help()
        sys.exit(1)