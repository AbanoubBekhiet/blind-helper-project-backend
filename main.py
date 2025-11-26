from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import torch
import numpy as np
import cv2
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# ---------------------------
# App & CORS setup
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Load YOLO model
# ---------------------------
yolo = YOLO("models/yoloe-v8l-seg-pf.pt")

# ---------------------------
# Detection endpoint (WITHOUT distance)
# ---------------------------
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    # ---------------------------
    # YOLO Detection ONLY
    # ---------------------------
    results = yolo(frame)
    detections = results[0].boxes.data.tolist() if results[0].boxes.data is not None else []

    objects_info = []
    description_texts = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = yolo.model.names[int(cls)]

        objects_info.append({
            "label": label,
            "confidence": float(round(conf, 2)),
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        })

        description_texts.append(label)

    return JSONResponse({
        "objects": objects_info,
        "text": ", ".join(description_texts) if description_texts else "لا توجد أشياء مكتشفة"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
