from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
import uvicorn
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Improved Latency Predictor Service")

# Global variables for loaded model
model_data = None
models = {}

class PodFeatures(BaseModel):
    # Required fields
    selected_pod: str  # This is the pod IP
    input_tokens: int
    output_tokens: int
    total_tokens: int
    
    # Optional pod metrics - directly matching the training data columns
    kv_hit_ratio: Optional[int] = None
    inflight_requests: Optional[int] = None
    gpu_kv_cache: Optional[float] = None
    cpu_kv_cache: Optional[float] = None
    running_requests: Optional[int] = None
    waiting_requests: Optional[int] = None
    prefill_tokens: Optional[int] = None
    decode_tokens: Optional[int] = None
    gpu_model: Optional[str] = None
    last_second_avg_ttft_ms: Optional[float] = None
    last_second_avg_tpot_ms: Optional[float] = None
    last_second_p99_ttft_ms: Optional[int] = None
    last_second_p99_tpot_ms: Optional[int] = None
    last_second_total_requests: Optional[int] = None
    last_second_total_tokens: Optional[int] = None
    last_second_total_decode_tokens: Optional[int] = None
    last_second_total_prefill_tokens: Optional[int] = None
    
    # Allow additional fields that might be pod-specific
    additional_metrics: Optional[Dict[str, Union[int, float, str]]] = None

class PredictionRequest(BaseModel):
    pods: List[PodFeatures]

class PredictionResponse(BaseModel):
    predictions: Dict[str, Dict[str, float]]

@app.on_event("startup")
async def load_model():
    global model_data, models
    
    model_path = "./latency_predictor.joblib"
    
    try:
        logger.info(f"Loading model from {model_path}")
        model_data = joblib.load(model_path)
        models = model_data.get('models', {})
        
        if not models:
            logger.error("No models found in model data")
            return
            
        logger.info(f"Loaded models for targets: {list(models.keys())}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")

@app.get("/health")
async def health_check():
    if not models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "models": list(models.keys())}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Result dictionary: pod_ip -> {target -> prediction}
    predictions = {}
    
    for pod in request.pods:
        pod_ip = pod.selected_pod
        predictions[pod_ip] = {}
        
        # Create a record with all pod features
        record = {
            'selected_pod': pod_ip,
            'input_tokens': pod.input_tokens,
            'output_tokens': pod.output_tokens,
            'total_tokens': pod.total_tokens,
        }
        
        # Add all optional fields if they exist
        if pod.kv_hit_ratio is not None:
            record['kv_hit_ratio'] = pod.kv_hit_ratio
        if pod.inflight_requests is not None:
            record['inflight_requests'] = pod.inflight_requests
        if pod.gpu_kv_cache is not None:
            record['gpu_kv_cache'] = pod.gpu_kv_cache
        if pod.cpu_kv_cache is not None:
            record['cpu_kv_cache'] = pod.cpu_kv_cache
        if pod.running_requests is not None:
            record['running_requests'] = pod.running_requests
        if pod.waiting_requests is not None:
            record['waiting_requests'] = pod.waiting_requests
        if pod.prefill_tokens is not None:
            record['prefill_tokens'] = pod.prefill_tokens
        if pod.decode_tokens is not None:
            record['decode_tokens'] = pod.decode_tokens
        if pod.gpu_model is not None:
            record['gpu_model'] = pod.gpu_model
        if pod.last_second_avg_ttft_ms is not None:
            record['last_second_avg_ttft_ms'] = pod.last_second_avg_ttft_ms
        if pod.last_second_avg_tpot_ms is not None:
            record['last_second_avg_tpot_ms'] = pod.last_second_avg_tpot_ms
        if pod.last_second_p99_ttft_ms is not None:
            record['last_second_p99_ttft_ms'] = pod.last_second_p99_ttft_ms
        if pod.last_second_p99_tpot_ms is not None:
            record['last_second_p99_tpot_ms'] = pod.last_second_p99_tpot_ms
        if pod.last_second_total_requests is not None:
            record['last_second_total_requests'] = pod.last_second_total_requests
        if pod.last_second_total_tokens is not None:
            record['last_second_total_tokens'] = pod.last_second_total_tokens
        if pod.last_second_total_decode_tokens is not None:
            record['last_second_total_decode_tokens'] = pod.last_second_total_decode_tokens
        if pod.last_second_total_prefill_tokens is not None:
            record['last_second_total_prefill_tokens'] = pod.last_second_total_prefill_tokens
        
        # Add any additional metrics
        if pod.additional_metrics:
            for key, value in pod.additional_metrics.items():
                record[key] = value
                
        # Create a DataFrame with just this pod's data
        df = pd.DataFrame([record])
        
        # Log what we're sending to the model
        logger.info(f"Prediction features for pod {pod_ip}: {record}")
        
        # Make predictions for each target model
        for target, model in models.items():
            try:
                # Predict using the model
                pred = model.predict(df)[0]  # Get first row value
                predictions[pod_ip][target] = float(pred)
            except Exception as e:
                logger.error(f"Error predicting {target} for pod {pod_ip}: {e}")
                logger.error(f"DataFrame columns: {df.columns.tolist()}")
                # Set to -1 to indicate prediction failure
                predictions[pod_ip][target] = -1
    
    return {"predictions": predictions}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)