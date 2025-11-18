from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uuid, os

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Interior AI Backend Running"}

@app.post("/process")
async def process(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    ext = file.filename.split(".")[-1]
    filepath = f"{UPLOAD_DIR}/{file_id}.{ext}"

    # Save uploaded image
    with open(filepath, "wb") as f:
        f.write(await file.read())

    # Dummy masks (replace with real masks later)
    response = {
        "projectId": file_id,
        "objects": [
            {
                "id": "mask_wall",
                "name": "wall",
                "mask_url": f"https://placehold.co/200x200/CCCCCC/000000?text=Wall+Mask"
            },
            {
                "id": "mask_floor",
                "name": "floor",
                "mask_url": f"https://placehold.co/200x200/999999/000000?text=Floor+Mask"
            }
        ]
    }
    return JSONResponse(response)

@app.post("/recolor")
async def recolor(payload: dict):
    return {
        "result_image_url": "https://placehold.co/600x400.png?text=Recolored+Image"
    }
