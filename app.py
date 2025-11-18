from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uuid, os, base64
from PIL import Image, ImageDraw
from io import BytesIO

app = FastAPI()

# directory to store uploaded images
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# serve uploaded images
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/")
def home():
    return {"message": "Interior AI Backend Running"}


# -----------------------------------------------------------
# PROCESS ENDPOINT — receives base64, saves the file, returns masks
# -----------------------------------------------------------
@app.post("/process")
async def process(payload: dict):
    try:
        if not isinstance(payload, dict):
            return JSONResponse({"error": "Invalid payload"}, status_code=400)

        filename = payload.get("filename", "uploaded_image")
        content = payload.get("content")

        if not content:
            return JSONResponse({"error": "No content provided"}, status_code=400)

        # decode base64 image
        try:
            image_bytes = base64.b64decode(content)
        except Exception as e:
            return JSONResponse({"error": f"Base64 decode error: {str(e)}"}, status_code=400)

        # save image
        file_id = str(uuid.uuid4())
        saved_name = f"{file_id}.jpg"
        filepath = os.path.join(UPLOAD_DIR, saved_name)

        with open(filepath, "wb") as f:
            f.write(image_bytes)

        # return dummy detected objects (placeholder masks)
        return {
            "projectId": file_id,
            "objects": [
                {
                    "id": "mask_wall",
                    "name": "wall",
                    "mask_url": "https://placehold.co/200x200/CCCCCC/000000?text=Wall+Mask"
                },
                {
                    "id": "mask_floor",
                    "name": "floor",
                    "mask_url": "https://placehold.co/200x200/999999/000000?text=Floor+Mask"
                }
            ],
            "saved_image_url": f"/uploads/{saved_name}"
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# -----------------------------------------------------------
# RECOLOR ENDPOINT — returns BASE64 image (works in Salesforce)
# -----------------------------------------------------------
@app.post("/recolor")
async def recolor(payload: dict):
    try:
        if not isinstance(payload, dict):
            return JSONResponse({"error": "Invalid payload"}, status_code=400)

        color = payload.get("color", "#ff0000")

        # Create recolor image dynamically for demo
        img = Image.new("RGB", (800, 600), color=color)
        draw = ImageDraw.Draw(img)
        draw.text((200, 250), "Recolored Demo", fill="white")

        # convert to base64
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        base64_img = base64.b64encode(buffer.getvalue()).decode()

        return {
            "result_base64": base64_img
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
