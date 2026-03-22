import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from inference import load_model, predict

# Global model variable
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # Look for model.pth in current dir or parent dir
    model_path = "model.pth"
    if not os.path.exists(model_path):
        model_path = "../model.pth"
        if not os.path.exists(model_path):
            # Also try the user's saved path name in their training script
            model_path = "../best_brain_tumor_model.pth"
            if not os.path.exists(model_path):
                print("WARNING: model.pth not found! Please place it in the app directory.")
    
    try:
        model = load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
    
    yield
    # Clean up (if any) at shutdown
    model = None

app = FastAPI(lifespan=lifespan)

# Mount static files
os.makedirs("uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def run_prediction(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image (jpg, jpeg, png).")
    
    try:
        contents = await file.read()
        
        # Save to uploads folder as requested (temp storage)
        file_location = f"uploads/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(contents)
            
        if model is None:
            raise HTTPException(status_code=500, detail="Model is not loaded.")
            
        label, confidence, all_scores = predict(model, contents)
        
        return JSONResponse(content={
            "label": label,
            "confidence": confidence,
            "all_scores": all_scores
        })
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Model inference failed.")

@app.get("/metrics")
async def get_metrics():
    try:
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
        return JSONResponse(content=metrics)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Metrics file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to load metrics.")
