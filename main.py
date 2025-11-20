from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import torch, cv2, numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO
yolo = YOLO("models/yolov8m.pt")

# Load MiDaS
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform


def classify_distance(pixel_height, avg_depth):
    score = pixel_height * avg_depth

    if score > 200:
        return "قريب جدًا"
    elif score > 150:
        return "قريب"
    elif score > 100:
        return "بعيد"
    else:
        return "بعيد جدًا"


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # YOLO detection
    results = yolo(frame)
    detections = results[0].boxes.data.tolist()

    # Depth (MiDaS)
    input_batch = transform(frame).to("cpu")
    with torch.no_grad():
        prediction = midas(input_batch)
    depth_map = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=frame.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

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
