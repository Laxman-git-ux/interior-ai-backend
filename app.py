# app.py — YOLO detection + mask support + bbox & mask recolor + download
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import os, time, json, requests, base64, uuid
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

# FastAPI app + uploads
app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Replicate config from env
REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN")
REPLICATE_PREDICT_URL = "https://api.replicate.com/v1/predictions"
YOLO_VERSION = os.getenv("REPLICATE_YOLO_VERSION")            # e.g. ultralytics/yolov8s-worldv2 version id
SAM_VERSION = os.getenv("REPLICATE_SAM_VERSION")              # optional
CLASSIFY_VERSION = os.getenv("REPLICATE_CLASSIFY_VERSION")    # optional

# helper - generic replicate call with optional polling
def replicate_predict(version: str, input_payload: dict, timeout=120, poll_interval=1.2):
    if not REPLICATE_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN not set")
    headers = {"Authorization": f"Token {REPLICATE_TOKEN}", "Content-Type": "application/json"}
    body = {"version": version, "input": input_payload}
    resp = requests.post(REPLICATE_PREDICT_URL, json=body, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("output") is not None:
        return data
    # async — poll
    get_url = data.get("urls", {}).get("get")
    if not get_url:
        return data
    start = time.time()
    while True:
        r = requests.get(get_url, headers=headers, timeout=30)
        r.raise_for_status()
        pd = r.json()
        status = pd.get("status")
        if status == "succeeded":
            return pd
        if status == "failed":
            raise RuntimeError(f"Replicate prediction failed: {pd}")
        if time.time() - start > timeout:
            raise TimeoutError("Replicate prediction timed out")
        time.sleep(poll_interval)

# utilities
def save_uploaded_image(project_id: str, image_bytes: bytes):
    fname = f"{project_id}.jpg"
    fpath = os.path.join(UPLOAD_DIR, fname)
    with open(fpath, "wb") as f:
        f.write(image_bytes)
    return f"/uploads/{fname}", fpath

def decode_data_uri(data_uri: str) -> bytes:
    if "," in data_uri:
        return base64.b64decode(data_uri.split(",",1)[1])
    return base64.b64decode(data_uri)

# ROUTES

@app.get("/")
def home():
    return {"message":"Interior AI Backend Running"}

@app.post("/process")
async def process(payload: dict):
    """
    Expects: {"filename":"...","content":"<base64 image>"}
    Returns: { projectId, saved_image_url, objects: [{id,name,bbox:[x1,y1,x2,y2],score}] }
    Uses YOLO via Replicate (YOLO output expected as list of detections).
    """
    try:
        content = payload.get("content")
        if not content:
            return JSONResponse({"error":"Missing content"}, status_code=400)

        try:
            image_bytes = base64.b64decode(content)
        except Exception as e:
            return JSONResponse({"error":f"Invalid base64: {e}"}, status_code=400)

        project_id = str(uuid.uuid4())
        saved_url, saved_path = save_uploaded_image(project_id, image_bytes)

        # Call YOLO model on Replicate
        if not YOLO_VERSION:
            # If YOLO not configured, return a small dummy response
            return {
                "projectId": project_id,
                "saved_image_url": saved_url,
                "objects": []
            }

        payload_input = {"image": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}", "conf": 0.25}
        rep = replicate_predict(YOLO_VERSION, payload_input, timeout=120, poll_interval=1.2)

        # Normalize detections
        detections = []
        out = rep.get("output") or []
        # Many Replicate YOLO wrappers return list of dicts: [{"class_name":"sofa","confidence":0.92,"box":[x1,y1,x2,y2]}, ...]
        for item in out:
            # be permissive about key names
            name = item.get("class_name") or item.get("label") or item.get("name") or item.get("class")
            conf = item.get("confidence") or item.get("score") or item.get("conf")
            box = item.get("box") or item.get("bbox") or item.get("coordinates") or item.get("xyxy")
            # Ensure ints
            if box:
                try:
                    box = [int(float(x)) for x in box]
                except Exception:
                    box = None
            detections.append({"id": str(uuid.uuid4()), "name": name or "object", "bbox": box, "score": conf})

        return {"projectId": project_id, "saved_image_url": saved_url, "objects": detections}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/process_masks")
async def process_masks(payload: dict):
    """
    OPTIONAL: run SAM segmentation via Replicate to return masks (data URIs or URLs).
    Input same as /process. Returns objects with mask_url entries.
    (Left as optional — many SAM models return polygons / masks. Use only if needed.)
    """
    try:
        content = payload.get("content")
        if not content:
            return JSONResponse({"error":"Missing content"}, status_code=400)
        try:
            image_bytes = base64.b64decode(content)
        except Exception as e:
            return JSONResponse({"error":f"Invalid base64: {e}"}, status_code=400)

        project_id = str(uuid.uuid4())
        saved_url, saved_path = save_uploaded_image(project_id, image_bytes)

        if not SAM_VERSION:
            return {"projectId": project_id, "saved_image_url": saved_url, "objects": []}

        payload_input = {"image": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"}
        rep = replicate_predict(SAM_VERSION, payload_input, timeout=120, poll_interval=1.2)

        masks = rep.get("output") or rep.get("masks") or []
        objects = []
        for idx, mask_item in enumerate(masks):
            if isinstance(mask_item, str) and mask_item.startswith("data:"):
                mask_bytes = decode_data_uri(mask_item)
                mask_fname = f"{project_id}_mask_{idx}.png"
                mask_path = os.path.join(UPLOAD_DIR, mask_fname)
                with open(mask_path, "wb") as f:
                    f.write(mask_bytes)
                mask_url = f"/uploads/{mask_fname}"
                objects.append({"id": f"mask_{idx}", "name": f"segment_{idx}", "mask_url": mask_url})
            elif isinstance(mask_item, str) and mask_item.startswith("http"):
                # download and save
                r = requests.get(mask_item, timeout=30)
                r.raise_for_status()
                mask_bytes = r.content
                mask_fname = f"{project_id}_mask_{idx}.png"
                mask_path = os.path.join(UPLOAD_DIR, mask_fname)
                with open(mask_path, "wb") as f:
                    f.write(mask_bytes)
                mask_url = f"/uploads/{mask_fname}"
                objects.append({"id": f"mask_{idx}", "name": f"segment_{idx}", "mask_url": mask_url})
        return {"projectId": project_id, "saved_image_url": saved_url, "objects": objects}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/recolor_bbox")
async def recolor_bbox(payload: dict):
    """
    Recolor a bounding-box region.
    Input: {"projectId": "...", "bbox":[x1,y1,x2,y2], "color":"#rrggbb", "alpha":0.6}
    Returns: {"result_base64": "..."}
    """
    try:
        project_id = payload.get("projectId")
        bbox = payload.get("bbox")
        color_hex = payload.get("color", "#ff0000")
        alpha = float(payload.get("alpha", 0.6))

        if not project_id or not bbox:
            return JSONResponse({"error":"Missing projectId or bbox"}, status_code=400)

        image_path = os.path.join(UPLOAD_DIR, f"{project_id}.jpg")
        if not os.path.exists(image_path):
            return JSONResponse({"error":"Original image not found"}, status_code=404)

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(image.shape[1], x2); y2 = min(image.shape[0], y2)
        region = image[y1:y2, x1:x2]
        if region.size == 0:
            return JSONResponse({"error":"Empty bbox region"}, status_code=400)

        try:
            color = tuple(int(color_hex[i:i+2],16) for i in (1,3,5))
        except:
            return JSONResponse({"error":"Invalid color"}, status_code=400)

        color_layer = np.zeros_like(region); color_layer[:] = color
        recolored = (region*(1-alpha) + color_layer*alpha).astype(np.uint8)
        output_image = image.copy(); output_image[y1:y2, x1:x2] = recolored

        _, buf = cv2.imencode(".png", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        result_b64 = base64.b64encode(buf.tobytes()).decode()
        return {"result_base64": result_b64}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/recolor_mask")
async def recolor_mask(payload: dict):
    """
    Recolor using a binary mask file (mask_url should be relative /uploads/<maskfile>).
    Input: {"projectId":"...", "mask_url":"/uploads/xxx.png", "color":"#rrggbb", "alpha":0.6}
    """
    try:
        project_id = payload.get("projectId")
        mask_url = payload.get("mask_url")
        color_hex = payload.get("color", "#ff0000")
        alpha = float(payload.get("alpha", 0.6))

        if not project_id or not mask_url:
            return JSONResponse({"error":"Missing projectId or mask_url"}, status_code=400)

        image_path = os.path.join(UPLOAD_DIR, f"{project_id}.jpg")
        mask_path = mask_url.lstrip("/") if mask_url.startswith("/") else mask_url
        mask_full = os.path.join(os.getcwd(), mask_path)

        if not os.path.exists(image_path) or not os.path.exists(mask_full):
            return JSONResponse({"error":"Image or mask not found"}, status_code=404)

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_full, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return JSONResponse({"error":"Invalid mask"}, status_code=400)

        mask_norm = (mask.astype(float)/255.0).clip(0,1)[..., None]
        try:
            color = tuple(int(color_hex[i:i+2],16) for i in (1,3,5))
        except:
            return JSONResponse({"error":"Invalid color"}, status_code=400)

        color_layer = np.zeros_like(image); color_layer[:] = color
        result = (image*(1 - mask_norm*alpha) + color_layer*(mask_norm*alpha)).astype(np.uint8)

        _, buf = cv2.imencode(".png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        result_b64 = base64.b64encode(buf.tobytes()).decode()
        return {"result_base64": result_b64}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/download/{project_id}")
def download_image(project_id: str):
    """Return original image file so LWC can download it."""
    path = os.path.join(UPLOAD_DIR, f"{project_id}.jpg")
    if not os.path.exists(path):
        return JSONResponse({"error":"Not found"}, status_code=404)
    return FileResponse(path, media_type="image/jpeg", filename=f"{project_id}.jpg")
