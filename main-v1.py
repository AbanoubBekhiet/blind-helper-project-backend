from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO
import torch, cv2, numpy as np, tempfile
from gtts import gTTS
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import hashlib

app = FastAPI()

# السماح للفرونت بالوصول
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل موديل YOLO مرة واحدة
yolo = YOLO("models/yolov8m.pt")  

# تحميل MiDaS لتقدير العمق النسبي
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# مجلد مؤقت للملفات الصوتية
TEMP_DIR = Path(tempfile.gettempdir()) / "blind_project_audio"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# تصنيف المسافات
def classify_distance(pixel_height, avg_depth):
    """
    pixel_height: ارتفاع الجسم بالبكسل
    avg_depth: قيمة عمق MiDaS (0-1)
    """
    # كل جسم له تقدير نسبي للعمق: أصغر pixel_height = بعيد
    # نستخدم avg_depth كعامل تصحيح
    score = pixel_height * avg_depth
    if score > 200:
        return "قريب جدًا"
    elif score > 150:
        return "قريب"
    elif score > 100:
        return "بعيد"
    else:
        return "بعيد جدًا"

def get_cached_audio(text: str):
    """ينشئ ملف صوتي للنص لو مش موجود مسبقًا"""
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    audio_path = TEMP_DIR / f"{text_hash}.mp3"
    if not audio_path.exists():
        tts = gTTS(text=text, lang="ar", slow=False)
        tts.save(audio_path)
    return audio_path

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # كشف الأجسام
    results = yolo(frame)
    detections = results[0].boxes.data.tolist()

    # تقدير العمق النسبي
    input_batch = transform(frame).to("cpu")
    with torch.no_grad():
        prediction = midas(input_batch)
    depth_map = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=frame.shape[:2],
        mode="bicubic",
        align_corners=False
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

        description_texts.append(f"{label}  {distance_label}")

    audio_url = None
    if description_texts:
        description = ", ".join(description_texts)
        audio_path = get_cached_audio(description)
        audio_url = f"/audio/{audio_path.name}"

    if objects_info:
        return JSONResponse({
            "objects": objects_info,
            "audio_url": audio_url
        })
    else:
        return JSONResponse({"message": "لا توجد أشياء مكتشفة"})

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    path = TEMP_DIR / filename
    if path.exists():
        return FileResponse(path, media_type="audio/mpeg", filename=filename)
    return JSONResponse({"error": "ملف الصوت غير موجود"}, status_code=404)
