from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
import uvicorn
import os
from collections import Counter



# ---------------------------
# Arabic Translation (Argos Translate)
# ---------------------------
import argostranslate.package
import argostranslate.translate

def prepare_argos_en_ar():
    argostranslate.package.update_package_index()
    packages = argostranslate.package.get_available_packages()
    pkg = next(filter(lambda x: x.from_code=="en" and x.to_code=="ar", packages), None)
    if pkg:
        path = pkg.download()
        argostranslate.package.install_from_path(path)

prepare_argos_en_ar()

def translate_to_ar(text: str) -> str:
    try:
        return argostranslate.translate.translate(text, "en", "ar")
    except:
        return text


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


yolo = YOLO("models/yoloe-v8l-seg-pf.pt") 
currency_detection_model = YOLO("./models/best.pt") 


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

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



@app.post("/detect-currency")
async def detect_currency(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    # Run YOLO inference
    results = currency_detection_model(frame)
    detections = results[0].boxes.data.tolist() if results[0].boxes.data is not None else []

    # Prepare response
    objects_info = []
    summary_ar_list = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label_en = currency_detection_model.model.names[int(cls)]
        label_ar = translate_to_ar(label_en)

        objects_info.append({
            "denomination_en": label_en,
            "denomination_ar": label_ar,
            "confidence": float(round(conf, 2)),
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        })

        summary_ar_list.append(label_ar)

    if not objects_info:
        return JSONResponse({"detected_text": "No Egyptian currency detected", "currencies": []})

    # Count each denomination (English)
    from collections import Counter
    counts = Counter([obj["denomination_en"] for obj in objects_info])
    summary_en = ", ".join(f"{k} x {v}" for k, v in counts.items())

    # Count each denomination (Arabic)
    counts_ar = Counter(summary_ar_list)
    summary_ar = ", ".join(f"{k} x {v}" for k, v in counts_ar.items())

    return JSONResponse({
        "detected_text_en": summary_en,
        "detected_text_ar": summary_ar,
        "currencies": objects_info
    })




# ---------------------------
# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
