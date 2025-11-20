# app.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import os, time, json, requests, base64, uuid, logging
from io import BytesIO
from PIL import Image

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("interior-ai")

app = FastAPI()

# CORS (allow Salesforce/dev clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Replicate config from env
REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
YOLO_VERSION = os.getenv("REPLICATE_YOLO_VERSION", "")  # expect the version id/hash
REPLICATE_PREDICT_URL = "https://api.replicate.com/v1/predictions"

if not REPLICATE_TOKEN:
    log.warning("REPLICATE_API_TOKEN is not set. Replicate calls will fail.")
if not YOLO_VERSION:
    log.warning("REPLICATE_YOLO_VERSION is not set. Replicate calls will fail.")

def replicate_predict(version: str, input_payload: dict, timeout: int = 120):
    """
    Post to Replicate /v1/predictions and poll until finished.
    Returns the final response JSON.
    """
    headers = {
        "Authorization": f"Token {REPLICATE_TOKEN}",
        "Content-Type": "application/json"
    }

    body = {"version": version, "input": input_payload}
    log.info("Creating replicate prediction...")
    resp = requests.post(REPLICATE_PREDICT_URL, json=body, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Synchronous output returned directly
    if data.get("output"):
        return data

    # Poll async get URL
    get_url = data.get("urls", {}).get("get")
    if not get_url:
        raise Exception("Replicate response missing get URL for polling.")

    start = time.time()
    while True:
        result = requests.get(get_url, headers=headers, timeout=30)
        result.raise_for_status()
        result_json = result.json()

        status = result_json.get("status")
        if status == "succeeded":
            return result_json
        if status == "failed":
            raise Exception("Replicate job failed: " + json.dumps(result_json))

        if time.time() - start > timeout:
            raise TimeoutError("Replicate prediction timed out")

        time.sleep(1)

def run_yolo(image_bytes: bytes):
    """
    Calls replicate YOLO model and normalizes detections into:
    [{ "class": "...", "confidence": 0.9, "bbox": [x1,y1,x2,y2] }, ...]
    Handles outputs that either provide bbox directly or x,y,width,height.
    """
    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "image": f"data:image/jpeg;base64,{b64}",
        "confidence": 0.3
    }

    result = replicate_predict(YOLO_VERSION, payload)
    outputs = result.get("output") or []

    detections = []

    # Some Replicate models produce a list of lists or list of dicts; normalize
    # We try to be permissive: loop through outputs and flatten if needed
    flat = []
    for o in outputs:
        if isinstance(o, list):
            flat.extend(o)
        else:
            flat.append(o)

    for det in flat:
        try:
            # If the model returns bbox already
            if isinstance(det.get("bbox"), (list, tuple)) and len(det.get("bbox")) >= 4:
                bbox = det.get("bbox")
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            else:
                # Try x,y,width,height (center or top-left)
                x = det.get("x")
                y = det.get("y")
                w = det.get("width") or det.get("w") or det.get("box_width")
                h = det.get("height") or det.get("h") or det.get("box_height")
                # If all present, assume x,y are center coords (common in YOLO)
                if all(v is not None for v in (x, y, w, h)):
                    x1 = x - w / 2
                    y1 = y - h / 2
                    x2 = x + w / 2
                    y2 = y + h / 2
                else:
                    # fallback try top/left/right/bottom keys
                    x1 = det.get("left") or det.get("x1") or 0
                    y1 = det.get("top") or det.get("y1") or 0
                    x2 = det.get("right") or det.get("x2") or x1
                    y2 = det.get("bottom") or det.get("y2") or y1
        except Exception as ex:
            log.exception("Failed to parse detection: %s", ex)
            continue

        detections.append({
            "class": det.get("class") or det.get("label") or det.get("name") or "object",
            "confidence": float(det.get("confidence") or det.get("score") or 0),
            "bbox": [float(x1), float(y1), float(x2), float(y2)]
        })

    return detections

@app.get("/")
def home():
    return {"message": "Interior AI Backend Running"}

@app.post("/process")
async def process(payload: dict):
    """
    Expects: { "content": "<base64 image data (without data:image/... prefix)>" }
    Returns: { projectId, saved_image_url, image_width, image_height, objects: [{id,name,score,box:[x1,y1,x2,y2]}] }
    """
    try:
        base64_img = payload.get("content")
        if not base64_img:
            return JSONResponse({"error": "Missing image in payload 'content' field"}, status_code=400)

        # Accept both pure base64 or data URL
        if base64_img.startswith("data:"):
            base64_img = base64_img.split(",", 1)[1]

        image_bytes = base64.b64decode(base64_img)
        project_id = str(uuid.uuid4())
        image_path = f"{UPLOAD_DIR}/{project_id}.jpg"

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        # get image dimensions
        image = Image.open(BytesIO(image_bytes))
        width, height = image.size

        # Run YOLO detection
        detections = run_yolo(image_bytes)

        objects = []
        for idx, det in enumerate(detections):
            objects.append({
                "id": f"obj_{idx}",
                "name": det.get("class", "object"),
                "score": det.get("confidence", 0),
                "box": det.get("bbox", [])  # [x1,y1,x2,y2]
            })

        response = {
            "projectId": project_id,
            "saved_image_url": f"/uploads/{project_id}.jpg",
            "image_width": width,
            "image_height": height,
            "objects": objects
        }

        return JSONResponse(response, status_code=200)

    except Exception as e:
        log.exception("Processing failed")
        return JSONResponse({"error": str(e)}, status_code=500)


# Keep recolor endpoint placeholder (if you have recolor logic add here)
@app.post("/recolor")
async def recolor(payload: dict):
    return {"result_base64": None}
