# app.py — Full Option B: SAM (Replicate) + CLIP labeling + recolor
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import os, time, json, requests, base64, uuid
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
import cv2

# ---------------------------
# FastAPI app + uploads
# ---------------------------
app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# ---------------------------
# Replicate config from env
# ---------------------------
REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN")
REPLICATE_PREDICT_URL = "https://api.replicate.com/v1/predictions"
SAM_VERSION = os.getenv("REPLICATE_SAM_VERSION")
CLASSIFY_VERSION = os.getenv("REPLICATE_CLASSIFY_VERSION")

# curated labels (expand as needed)
CANDIDATE_LABELS = [
    "wall", "floor", "sofa", "chair", "table", "carpet", "rug",
    "curtain", "window", "bed", "lamp", "plant", "shelf", "mirror"
]

# ---------------------------
# Helpers: Replicate call + polling
# ---------------------------
def replicate_predict(version: str, input_payload: dict, timeout=120, poll_interval=1.5):
    """
    Call Replicate /v1/predictions for a model version.
    Handles both synchronous output and asynchronous predictions (polls the get URL).
    Returns the final JSON result dict.
    """
    if not REPLICATE_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN not set in environment")

    headers = {
        "Authorization": f"Token {REPLICATE_TOKEN}",
        "Content-Type": "application/json",
    }
    body = {"version": version, "input": input_payload}
    resp = requests.post(REPLICATE_PREDICT_URL, json=body, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # if synchronous output present
    if data.get("output") is not None:
        return data

    # otherwise, poll the provided GET url
    urls = data.get("urls") or {}
    get_url = urls.get("get")
    if not get_url:
        # some wrappers might return output later in same payload; return what we have
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

# ---------------------------
# Helpers: image/mask utilities
# ---------------------------
def download_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

def decode_data_uri(data_uri: str) -> bytes:
    # data:image/png;base64,AAAA...
    if "," in data_uri:
        return base64.b64decode(data_uri.split(",", 1)[1])
    return base64.b64decode(data_uri)

def save_bytes_to_uploads(prefix: str, idx: int, content: bytes, ext="png") -> str:
    fname = f"{prefix}_mask_{idx}.{ext}"
    fpath = os.path.join(UPLOAD_DIR, fname)
    with open(fpath, "wb") as f:
        f.write(content)
    # return relative URL for LWC: /uploads/<fname>
    return f"/uploads/{fname}"

# crop masked region and run classifier
def crop_mask_and_label(image_bytes: bytes, mask_bytes: bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    mask_img = Image.open(BytesIO(mask_bytes)).convert("L")
    img_np = np.array(img)
    mask_np = np.array(mask_img)

    ys, xs = np.where(mask_np > 127)
    if ys.size == 0 or xs.size == 0:
        return None

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    crop = img_np[y0:y1+1, x0:x1+1]

    # encode crop to JPEG bytes
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    crop_bytes = buf.tobytes()

    # classify via Replicate CLIP model
    label = run_classify_api(crop_bytes, CANDIDATE_LABELS)
    return label

def run_classify_api(image_bytes: bytes, candidates: list):
    """
    Call CLIP-like classifier on Replicate.
    Expect payload: {"image": "data:image/jpeg;base64,...", "candidates": [...]}
    The wrapper may return output as list (best-first) or dict with labels/scores.
    """
    b64 = base64.b64encode(image_bytes).decode()
    payload = {"image": f"data:image/jpeg;base64,{b64}", "candidates": candidates}
    data = replicate_predict(CLASSIFY_VERSION, payload, timeout=60)
    out = data.get("output")
    if isinstance(out, list) and len(out):
        return out[0]
    if isinstance(out, dict):
        labels = out.get("labels") or []
        scores = out.get("scores") or []
        if labels and scores:
            return labels[int(np.argmax(scores))]
    # fallback
    return candidates[0]

# ---------------------------
# Routes
# ---------------------------

@app.get("/")
def home():
    return {"message": "Interior AI Backend Running"}

@app.post("/process")
async def process(payload: dict):
    """
    Input: {"filename":"...","content":"<base64 image>"}
    Returns: {"projectId": "...", "objects":[{id,name,mask_url},...], "saved_image_url": "/uploads/.."}
    """
    try:
        if not isinstance(payload, dict):
            return JSONResponse({"error": "Invalid payload"}, status_code=400)

        content = payload.get("content")
        filename = payload.get("filename", "uploaded_image")

        if not content:
            return JSONResponse({"error": "No content provided"}, status_code=400)

        # decode and save original image
        try:
            image_bytes = base64.b64decode(content)
        except Exception as e:
            return JSONResponse({"error": f"Base64 decode error: {str(e)}"}, status_code=400)

        project_id = str(uuid.uuid4())
        saved_name = f"{project_id}.jpg"
        saved_path = os.path.join(UPLOAD_DIR, saved_name)
        with open(saved_path, "wb") as f:
            f.write(image_bytes)

        # Call Replicate SAM model (may be async) to get masks
        try:
            sam_payload = {"image": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"}
            sam_result = replicate_predict(SAM_VERSION, sam_payload, timeout=120, poll_interval=1.5)
        except Exception as e:
            return JSONResponse({"error": f"SAM prediction error: {str(e)}"}, status_code=500)

        # Normalize masks from model output
        masks_output = sam_result.get("output") or sam_result.get("masks") or []
        objects = []
        for idx, item in enumerate(masks_output):
            try:
                # item might be URL (http...) or data URI (data:...) or base64 string
                if isinstance(item, str) and item.startswith("http"):
                    mask_bytes = download_bytes(item)
                elif isinstance(item, str) and item.startswith("data:"):
                    mask_bytes = decode_data_uri(item)
                else:
                    # unknown: try decode if base64
                    try:
                        mask_bytes = base64.b64decode(item)
                    except Exception:
                        continue

                # optional: convert mask to clean binary PNG if needed (ensure 0/255)
                # label crop region
                label = crop_mask_and_label(image_bytes, mask_bytes) or f"segment_{idx}"

                # save mask for LWC preview
                mask_url = save_bytes_to_uploads(project_id, idx, mask_bytes, ext="png")

                objects.append({"id": f"mask_{idx}", "name": label, "mask_url": mask_url})
            except Exception as e:
                # skip mask on error
                print("mask handling error:", e)
                continue

        return {"projectId": project_id, "objects": objects, "saved_image_url": f"/uploads/{saved_name}"}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/recolor")
async def recolor(payload: dict):
    """
    Input: {"projectId":"...","mask_url":"/uploads/..png","color":"#rrggbb"}
    Returns: {"result_base64":"..."}
    """
    try:
        project_id = payload.get("projectId")
        mask_url = payload.get("mask_url")
        color_hex = payload.get("color", "#ff0000")

        if not project_id or not mask_url:
            return JSONResponse({"error":"Missing projectId or mask_url"}, status_code=400)

        image_path = os.path.join(UPLOAD_DIR, f"{project_id}.jpg")

        # mask_url is relative like /uploads/<file>
        if mask_url.startswith("/"):
            mask_path = mask_url.lstrip("/")
        else:
            mask_path = mask_url
        mask_fullpath = os.path.join(os.getcwd(), mask_path)

        if not os.path.exists(image_path):
            return JSONResponse({"error":"Original image not found"}, status_code=404)
        if not os.path.exists(mask_fullpath):
            return JSONResponse({"error":"Mask file not found"}, status_code=404)

        # load image and mask
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_fullpath, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return JSONResponse({"error":"Invalid mask"}, status_code=400)

        mask_norm = (mask.astype(float) / 255.0).clip(0,1)

        # parse hex color (#rrggbb)
        try:
            color = tuple(int(color_hex[i:i+2],16) for i in (1,3,5))
        except Exception:
            return JSONResponse({"error":"Invalid color"}, status_code=400)

        # create color layer and blend preserving texture
        color_layer = np.zeros_like(image)
        color_layer[:] = color
        alpha = 0.6
        result = (image * (1 - mask_norm[...,None]*alpha) + color_layer * (mask_norm[...,None]*alpha)).astype(np.uint8)

        # encode to PNG base64
        _, buf = cv2.imencode(".png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        result_b64 = base64.b64encode(buf.tobytes()).decode()

        return {"result_base64": result_b64}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
