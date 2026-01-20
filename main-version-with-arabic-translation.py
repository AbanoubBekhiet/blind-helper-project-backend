from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# ---------------------------
# Arabic Translation (Argos Translate)
# ---------------------------
import argostranslate.package
import argostranslate.translate

def prepare_argos_en_ar():
    """Download and install English → Arabic Argos model dynamically."""
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()

    # Find English → Arabic package
    package_to_install = next(
        filter(lambda x: x.from_code == "en" and x.to_code == "ar", available_packages),
        None
    )

    if package_to_install is None:
        raise Exception("EN → AR Argos model not found!")

    # Download + install
    path = package_to_install.download()
    argostranslate.package.install_from_path(path)

# Install model once at startup
prepare_argos_en_ar()

def translate_to_ar(text: str) -> str:
    """Translate any English text to Arabic using Argos."""
    try:
        # Fixed: pass both from_code and to_code
        return argostranslate.translate.translate(text, "en", "ar")
    except Exception as e:
        print(f"Translation error: {e}")
        return text

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
# Detection endpoint
# ---------------------------
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    # YOLO inference
    results = yolo(frame)
    detections = results[0].boxes.data.tolist() if results[0].boxes.data is not None else []

    objects_info = []
    description_texts = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label_en = yolo.model.names[int(cls)]
        label_ar = translate_to_ar(label_en)

        objects_info.append({
            "label_en": label_en,
            "label_ar": label_ar,
            "confidence": float(round(conf, 2)),
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        })

        description_texts.append(label_ar)

    return JSONResponse({
        "objects": objects_info,
        "text": ", ".join(description_texts) if description_texts else "لا توجد أشياء مكتشفة"
    })

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
