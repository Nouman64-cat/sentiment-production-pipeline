from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import time
import os
import mlflow
from contextlib import asynccontextmanager
from src.preprocessing.clean import clean_text
from src.api.database import init_db, log_prediction

# Path configuration
MODEL_PATH = os.path.join("models", "ml_model.joblib")
ml_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and DB on startup"""
    global ml_model
    init_db()  # Create table if not exists
    
    if os.path.exists(MODEL_PATH):
        ml_model = joblib.load(MODEL_PATH)
        print(f"✅ Loaded ML Model from {MODEL_PATH}")
    else:
        print(f"⚠️ Warning: Model not found at {MODEL_PATH}")
    
    yield
    print("Shutting down...")

app = FastAPI(title="Sentiment API", lifespan=lifespan)

class PredictionRequest(BaseModel):
    text: str

@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok", "model_loaded": ml_model is not None}

@app.post("/predict-ml")
@mlflow.trace(name="predict_endpoint")
def predict_ml(request: PredictionRequest):
    if not ml_model:
        raise HTTPException(status_code=503, detail="ML Model not available")
    
    start_time = time.time()
    
    # 1. Preprocess
    clean_input = clean_text(request.text)
    
    # 2. Predict (returns [0] or [1])
    try:
        pred_val = ml_model.predict([clean_input])[0]
        label = "positive" if pred_val == 1 else "negative"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    # 3. Log
    duration = (time.time() - start_time) * 1000
    log_prediction(request.text, label, "Classical_ML", duration)
    
    return {
        "prediction": label, 
        "inference_time_ms": round(duration, 2),
        "model": "LogisticRegression"
    }