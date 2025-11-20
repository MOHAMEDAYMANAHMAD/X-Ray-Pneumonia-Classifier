from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from Prediction import predict_image
from Load_model import all_models


app = FastAPI(title="X-Ray Pneumonia Classifier")

@app.post("/predict")
async def predict(file: UploadFile, model: str = Form("efficientnet_b0")):
    if model not in all_models:
        raise HTTPException(status_code=400, detail=f"Model '{model}' not found")

    img = Image.open(file.file).convert("RGB")
    selected_model = all_models[model]

    prediction, confidence = predict_image(selected_model, img)
    return JSONResponse({
        "model": model,
        "prediction": prediction,
        "confidence": confidence
    })

