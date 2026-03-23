import os
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from inference import load_model, predict

app = FastAPI()

# Mount static files
os.makedirs("uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global model — loaded lazily on first request
model = None

def get_model():
    global model
    if model is None:
        model_path = "model.pth"
        if not os.path.exists(model_path):
            model_path = "../model.pth"
        model = load_model(model_path)
        print("Model loaded successfully.")
    return model

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def run_prediction(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    try:
        contents = await file.read()
        file_location = f"uploads/{file.filename}"
        with open(file_location, "wb+") as f:
            f.write(contents)
        m = get_model()
        label, confidence, all_scores = predict(m, contents)
        return JSONResponse(content={
            "label": label,
            "confidence": confidence,
            "all_scores": all_scores
        })
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    try:
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
        return JSONResponse(content=metrics)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Metrics file not found.")
