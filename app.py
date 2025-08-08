# app.py
import os
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from utils import preprocess_image_bytes, get_labels

MODEL_DIR = os.environ.get("MODEL_DIR", "saved_model")

app = FastAPI(title="DL Classification API")

@app.on_event("startup")
def load_model():
    global model, labels
    print(f"Loading model from {MODEL_DIR} ...")
    MODEL_PATH = os.environ.get("MODEL_PATH", "model.keras")
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = get_labels()
    print("Model loaded.")

@app.get("/")
def root():
    return {"status": "ok", "message": "DL Classification API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        arr = preprocess_image_bytes(contents)  # (1,32,32,3)
        preds = model.predict(arr)
        pred_idx = int(np.argmax(preds, axis=-1)[0])
        confidence = float(np.max(preds))
        label = labels[pred_idx]
        return JSONResponse({
            "label": label,
            "label_id": pred_idx,
            "confidence": round(confidence, 4)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
