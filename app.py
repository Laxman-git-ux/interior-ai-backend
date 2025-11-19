from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import os, time, json, requests, base64, uuid
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# 🔥 NEW — YOLO VERSION
REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN")
YOLO_VERSION = os.getenv("REPLICATE_YOLO_VERSION")

REPLICATE_PREDICT_URL = "https://api.replicate.com/v1/predictions"

# ---------------------------
# Helper: Replicate API + polling
# ---------------------------
def replicate_predict(version, input_payload, timeout=60):
    headers = {
        "Authorization": f"Token {REPLICATE_TOKEN}",
        "Content-Type": "application/json"
    }

    body = {"version": version, "input": input_payload}
    resp = requests.post(REPLICATE_PREDICT_URL, json=body, headers=headers)
    data = resp.json()

    # synchronous response
    if data.get("output"):
        return data

    # async polling
    get_url = data["urls"]["get"]
    start = time.time()

    while True:
        result = requests.get(get_url, headers=headers).json()
        if result["status"] == "succeeded":
            return result
        if result["status"] == "failed":
            raise Exception("Replicate job failed")

        if time.time() - start > timeout:
            raise TimeoutError("Prediction timeout")

        time.sleep(1)

# ---------------------------
# YOLOv8 object detection
# ---------------------------
def run_yolo(image_bytes):
    b64 = base64.b64encode(image_bytes).decode()

    payload = {
        "image": f"data:image/jpeg;base64,{b64}",
        "confidence": 0.3
    }

    result = replicate_predict(YOLO_VERSION, payload)

    return result.get("output", [])

# ---------------------------
# ROUTES
# ---------------------------

@app.get("/")
def home():
    return {"message": "Interior AI Backend Running"}

# -------------------------------------------------------------
# PROCESS: upload → detect objects → return bounding boxes
# -------------------------------------------------------------
@app.post("/process")
async def process(payload: dict):

    try:
        base64_img = payload.get("content")
        if not base64_img:
            return JSONResponse({"error": "Missing image"}, status_code=400)

        image_bytes = base64.b64decode(base64_img)

        project_id = str(uuid.uuid4())
        image_path = f"{UPLOAD_DIR}/{project_id}.jpg"

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        # 🔥 DETECT OBJECTS WITH YOLO
        detections = run_yolo(image_bytes)

        objects = []
        for idx, det in enumerate(detections):
            objects.append({
                "id": f"obj_{idx}",
                "name": det.get("class", "object"),
                "score": det.get("confidence", 0),
                "box": det.get("bbox", [])  # [x1,y1,x2,y2]
            })

        return {
            "projectId": project_id,
            "objects": objects,
            "saved_image_url": f"/uploads/{project_id}.jpg"
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# -------------------------------------------------------------
# RECOLOR: uses mask or area
# -------------------------------------------------------------
@app.post("/recolor")
async def recolor(payload: dict):
    # unchanged – your recolor logic remains
    return {"result_base64": None}
