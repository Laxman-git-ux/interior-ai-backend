from fastapi import FastAPI
from fastapi.responses import JSONResponse
import base64, uuid, os

@app.post("/process")
async def process(payload: dict):
    filename = payload.get("filename")
    content = payload.get("content")  # base64 string

    # Decode base64
    image_bytes = base64.b64decode(content)

    file_id = str(uuid.uuid4())
    filepath = f"uploads/{file_id}.jpg"

    # Save image
    with open(filepath, "wb") as f:
        f.write(image_bytes)

    # Dummy mask response
    return {
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
