
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO(r"C:\Users\Kelly\MAKIT\best.pt")

@app.get("/")
def root():
    return {"status": "ElectroCom API running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = model.predict(source=frame, conf=0.7, verbose=False)
    detections = []
    for det in results[0].boxes:
        class_id = int(det.cls[0])
        confidence = float(det.conf[0])
        bbox = det.xyxy[0].tolist()
        class_name = model.names.get(class_id, f"class_{class_id}")
        detections.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(confidence, 3),
            "bbox": bbox
        })
    return {"total_detections": len(detections), "detections": detections}
