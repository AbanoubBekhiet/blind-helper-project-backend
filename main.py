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
# App & CORS
# ---------------------------
app = FastAPI(title="YOLOE Object Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Load YOLOE model
# ---------------------------
yolo = YOLO("models/yoloe-v8l-seg-pf.pt")  
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo.to(device)

# ---------------------------
# Detection endpoint using YOLOE predict()
# ---------------------------
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    # Convert BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---------------------------
    # YOLOE Prediction
    # ---------------------------
    results = yolo.predict(
        source=frame_rgb,
        conf=0.1,
        iou=0.5,
        device=device,
        verbose=False
    )

    # Extract detections
    objects_info = []
    description_texts = []

    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                conf = float(round(scores[i], 2))
                cls = int(classes[i])
                label = yolo.model.names[cls]

                objects_info.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })
                description_texts.append(label)

    return JSONResponse({
        "objects": objects_info,
        "text": ", ".join(description_texts) if description_texts else "لا توجد أشياء مكتشفة",
        "yolo_benefits": "YOLOE provides fast real-time object detection with proper confidence and NMS handling."
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
