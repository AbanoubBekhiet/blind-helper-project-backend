import base64

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
import uvicorn
import os
import easyocr
import PIL.Image
import io
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from groq import Groq

load_dotenv()
reader = easyocr.Reader(['ar', 'en'], gpu=False)  

import argostranslate.package
import argostranslate.translate
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

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


currency_detection_model = YOLO("./models/best.pt") 


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        # 1. قراءة محتوى الصورة
        contents = await file.read()
        
        # 2. تحويل الصورة إلى Base64 (متطلب أساسي لـ Groq Vision)
        base64_image = base64.b64encode(contents).decode('utf-8')

        # 3. إرسال الطلب لموديل Llama 3.2 Vision الحديث
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "صف هذه الصورة بدقة وبإيجاز لشخص كفيف باللغة العربية، ركز على الأشياء وأماكنها."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
        )

        # 4. استخراج النص
        description_ar = chat_completion.choices[0].message.content

        return JSONResponse({
            "status": "success",
            "description_ar": description_ar.strip(),
            "provider": "groq_cloud"
        })

    except Exception as e:
        print(f"Error logic: {str(e)}")
        return JSONResponse({"error": f"Groq API Error: {str(e)}"}, status_code=500)

@app.post("/detect-currency")
async def detect_currency(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    results = currency_detection_model(frame)
    detections = results[0].boxes.data.tolist() if results[0].boxes.data is not None else []

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

    from collections import Counter
    counts = Counter([obj["denomination_en"] for obj in objects_info])
    summary_en = ", ".join(f"{k} x {v}" for k, v in counts.items())

    counts_ar = Counter(summary_ar_list)
    summary_ar = ", ".join(f"{k} x {v}" for k, v in counts_ar.items())

    return JSONResponse({
        "detected_text_en": summary_en,
        "detected_text_ar": summary_ar,
        "currencies": objects_info
    })


@app.post("/detect-currency-new")
async def detect_currency(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    try:
        result = CLIENT.infer(frame, model_id="eg-currency-rjbjs/1")
    except Exception as e:
        return JSONResponse({"error": f"Roboflow API error: {str(e)}"}, status_code=500)

    predictions = result.get("predictions", [])

    objects_info = []
    summary_ar_list = []
    total_amount = 0

    amounts_map = {
        "200 pounds": 200,
        "100 pounds": 100,
        "50 pounds": 50,
        "20 pounds": 20,
        "10 pound": 10,
        "5 pounds": 5,
        "1 pound": 1
    }

    translations_map = {
        "200 pounds": "٢٠٠ جنيه",
        "100 pounds": "١٠٠ جنيه",
        "50 pounds": "٥٠ جنيهاً",
        "20 pounds": "٢٠ جنيهاً",
        "10 pound": "١٠ جنيهات",
        "5 pounds": "٥ جنيهات",
        "1 pound": "جنيه واحد"
    }

    for pred in predictions:
        label_en = pred["class"]
        label_ar = translations_map.get(label_en, label_en)
        val = amounts_map.get(label_en, 0)

        total_amount += val

        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)

        objects_info.append({
            "denomination_en": label_en,
            "denomination_ar": label_ar,
            "confidence": float(round(pred["confidence"], 2)),
            "bbox": [x1, y1, x2, y2]
        })
        summary_ar_list.append(label_ar)

    # Prepare summary strings from the model predictions.
    from collections import Counter
    counts_en = Counter([obj["denomination_en"] for obj in objects_info])
    summary_en = f"Total: {total_amount} EGP (" + ", ".join(f"{k} x {v}" for k, v in counts_en.items()) + ")"
    counts_ar = Counter(summary_ar_list)
    summary_manual_ar = f"الإجمالي: {total_amount} جنيه " + ", ".join(f"{k} x {v}" for k, v in counts_ar.items())

    # Use Groq to generate a more natural Arabic currency summary from the image.
    base64_image = base64.b64encode(contents).decode('utf-8')
    groq_summary_ar = summary_manual_ar
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "أحصر العملات الورقية والقطع النقدية المصرية الموجودة في هذه الصورة واطرحها باختصار باللغة العربية. إذا لم توجد عملات، أجب: لا توجد عملات مكتشفة."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
        )

        groq_summary_ar = chat_completion.choices[0].message.content.strip()
    except Exception:
        groq_summary_ar = summary_manual_ar

    if not objects_info:
        return JSONResponse({
            "total_amount": 0,
            "detected_text_en": "",
            "detected_text_ar": "لا توجد عملات مكتشفة",
            "currencies": [],
            "groq_summary_ar": "لا توجد عملات مكتشفة"
        })

    return JSONResponse({
        "total_amount": total_amount,
        "detected_text_en": summary_en,
        "detected_text_ar": groq_summary_ar,
        "currencies": objects_info,
        "groq_summary_ar": groq_summary_ar
    })


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)

        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        _, processed = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        results = reader.readtext(processed)

        results_filtered = [r for r in results if r[2] > 0.3]
        results_sorted = sorted(results_filtered, key=lambda r: (r[0][0][1], r[0][0][0]))

        lines_dict = {}
        for res in results_sorted:
            y = int(res[0][0][1] // 10)  
            lines_dict.setdefault(y, []).append(res[1])

        text_lines = [" ".join(line) for line in lines_dict.values()]
        final_text = "\n".join(text_lines)

        return JSONResponse(content={"text": final_text})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



# ---------------------------
# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
