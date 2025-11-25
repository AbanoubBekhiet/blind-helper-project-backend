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
yolo = YOLO("models/yolo12x.pt")

# ---------------------------
# Load MiDaS model via torch.hub
# ---------------------------
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform


# ---------------------------
# Distance classifier (best version)
# ---------------------------
def classify_distance(avg_depth: float, min_depth: float, max_depth: float) -> str:
    # Normalize depth 0 → near, 1 → far
    normalized = (avg_depth - min_depth) / (max_depth - min_depth + 1e-6)
    print(normalized)
    if normalized < 0.25:
        return "قريب جدًا"
    elif normalized < 0.45:
        return "قريب"
    elif normalized < 0.7:
        return "بعيد"
    else:
        return "بعيد جدًا"




# ---------------------------
# Detection endpoint
# ---------------------------
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Read image
    
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # ---------------------------
    # YOLO Detection
    # ---------------------------
    results = yolo(frame)
    detections = results[0].boxes.data.tolist() if results[0].boxes.data is not None else []

    # ---------------------------
    # MiDaS Depth Estimation
    # ---------------------------
    input_batch = transform(frame).to("cpu")
    with torch.no_grad():
        prediction = midas(input_batch)
    depth_map = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=frame.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze().cpu().numpy()




    # ---------------------------
    # Collect object info
    # ---------------------------
    objects_info = []
    description_texts = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = yolo.model.names[int(cls)]
        pixel_height = y2 - y1
        avg_depth = np.median(depth_map[int(y1):int(y2), int(x1):int(x2)])
        distance_label = classify_distance(pixel_height, avg_depth)

        objects_info.append({
            "label": label,
            "confidence": float(round(conf, 2)),
            "distance_label": distance_label,
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        })
        description_texts.append(f"{label} {distance_label}")

    return JSONResponse({
        "objects": objects_info,
        "text": ", ".join(description_texts) if description_texts else ""
    })



if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
