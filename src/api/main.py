from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import time
import os
import torch
import mlflow
from contextlib import asynccontextmanager
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from src.preprocessing.clean import clean_text
from src.api.database import init_db, log_prediction

ML_MODEL_PATH = os.path.join("models", "ml_model.joblib")
DL_MODEL_PATH = os.path.join("models", "dl_model.pth")

ml_pipeline = None
dl_model = None
dl_tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlflow.set_experiment("Sentiment_Analysis_Production")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_pipeline, dl_model, dl_tokenizer
    
    init_db()
    
    if os.path.exists(ML_MODEL_PATH):
        ml_pipeline = joblib.load(ML_MODEL_PATH)
        print(f"✅ Loaded ML Model (Scikit-Learn)")
    else:
        print(f"⚠️ Warning: ML Model missing at {ML_MODEL_PATH}")

    if os.path.exists(DL_MODEL_PATH):
        print("⏳ Loading DL Model (DistilBERT)... this takes a moment.")
        try:
            dl_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            dl_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
            
            dl_model.load_state_dict(torch.load(DL_MODEL_PATH, map_location=device))
            dl_model.to(device)
            dl_model.eval() 
            print(f"✅ Loaded DL Model (PyTorch) on {device}")
        except Exception as e:
            print(f"❌ Failed to load DL Model: {e}")
    else:
        print(f"⚠️ Warning: DL Model missing at {DL_MODEL_PATH}")

    yield
    print("Shutting down...")

app = FastAPI(title="Sentiment API", lifespan=lifespan)

class PredictionRequest(BaseModel):
    text: str

@app.get("/healthcheck")
def healthcheck():
    return {
        "status": "ok", 
        "ml_model": ml_pipeline is not None,
        "dl_model": dl_model is not None
    }

@app.post("/predict-ml")
@mlflow.trace(name="predict_ml")
def predict_ml(request: PredictionRequest):
    if not ml_pipeline:
        raise HTTPException(status_code=503, detail="ML Model not available")
    
    start_time = time.time()
    
    clean_input = clean_text(request.text)
    pred_val = ml_pipeline.predict([clean_input])[0]
    label = "positive" if pred_val == 1 else "negative"
    
    duration = (time.time() - start_time) * 1000
    log_prediction(request.text, label, "Classical_ML", duration)
    
    return {
        "prediction": label, 
        "inference_time_ms": round(duration, 2),
        "model": "LogisticRegression"
    }

@app.post("/predict-dl")
@mlflow.trace(name="predict_dl")
def predict_dl(request: PredictionRequest):
    if not dl_model or not dl_tokenizer:
        raise HTTPException(status_code=503, detail="DL Model not available")
    
    start_time = time.time()
    clean_input = clean_text(request.text)

    inputs = dl_tokenizer(
        clean_input, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = dl_model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    
    label = "positive" if prediction == 1 else "negative"
    
    duration = (time.time() - start_time) * 1000
    log_prediction(request.text, label, "DistilBERT_DL", duration)
    
    return {
        "prediction": label, 
        "inference_time_ms": round(duration, 2),
        "model": "DistilBERT"
    }