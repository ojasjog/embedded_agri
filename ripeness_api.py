"""
ripeness_api.py - drop-in FastAPI router that adds POST /api/ripeness.

In the server that already serves dashboard.html and /api/status etc., add:
    from ripeness_api import router as ripeness_router
    app.include_router(ripeness_router)

Needs ripeness.py in the same folder, plus: pip install opencv-python numpy python-multipart
"""
import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from ripeness import analyze

router = APIRouter()
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@router.post("/api/ripeness")
async def ripeness(crop: str = "tomato", file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image too large (max 10 MB)")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "not a valid image")
    try:
        return analyze(img, crop)
    except ValueError as e:
        raise HTTPException(400, str(e))