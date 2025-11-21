from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import torch
import numpy as np
import cv2
from fastapi.middleware.cors import CORSMiddleware

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
yolo = YOLO("models/yolov8m.pt")  # place your weights here

# ---------------------------
# Load MiDaS model via torch.hub
# ---------------------------
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# ---------------------------
# Distance classifier
# ---------------------------
def classify_distance(pixel_height: float, avg_depth: float) -> str:
    score = pixel_height * avg_depth
    if score > 200:
        return "قريب جدًا"
    elif score > 150:
        return "قريب"
    elif score > 100:
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
