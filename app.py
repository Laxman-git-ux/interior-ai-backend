from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uuid, os, base64

app = FastAPI()

# ensure upload directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# serve uploaded files
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/")
def home():
    return {"message": "Interior AI Backend Running"}


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

        file_id = str(uuid.uuid4())
        saved_name = f"{file_id}.jpg"
        filepath = os.path.join(UPLOAD_DIR, saved_name)

        # save image
        with open(filepath, "wb") as f:
            f.write(image_bytes)

        # dummy response
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


@app.post("/recolor")
async def recolor(payload: dict):
    try:
        if not isinstance(payload, dict):
            return JSONResponse({"error": "Invalid payload"}, status_code=400)

        return {
            "result_image_url": "https://placehold.co/600x400.png?text=Recolored+Image"
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
