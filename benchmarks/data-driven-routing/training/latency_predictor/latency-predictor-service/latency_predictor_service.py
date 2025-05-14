import threading
import orjson
from fastapi.responses import ORJSONResponse
from fastapi import FastAPI, HTTPException, Response
import joblib
import pandas as pd
import numpy as np
import uvicorn
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union
import os
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

thread_pool = ThreadPoolExecutor(max_workers=8)  # Adjust worker count based on your CPU cores

pending_requests = 0  # Requests waiting + active
active_requests = 0   # Currently processing
completed_requests = 0  # Total completed for stats
request_counter_lock = threading.Lock()

# Check for GPU availability 
try:
    import torch
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        # Set this environment variable to enable GPU acceleration in XGBoost
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        logger.info("CUDA is not available. Using CPU only.")
except ImportError:
    logger.info("PyTorch not available. Unable to check for GPU support.")

# Import XGBoost with potential GPU support
try:
    import xgboost as xgb
    logger.info(f"XGBoost version: {xgb.__version__}")
    # Check if XGBoost is built with GPU support
    try:
        xgb_params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
        dummy_model = xgb.XGBRegressor(**xgb_params)
        logger.info("XGBoost is configured to use GPU acceleration.")
    except Exception as e:
        logger.warning(f"XGBoost GPU test failed: {e}. Will use CPU version.")
except ImportError:
    logger.warning("XGBoost import failed.")

# app = FastAPI(title="GPU-Accelerated Latency Predictor Service")
app = FastAPI(
    title="GPU-Accelerated Latency Predictor Service",
    default_response_class=ORJSONResponse
)

# Global variables for loaded model
model_data = None
models = {}
model_types = {}  # Track model types to apply correct GPU acceleration

class PodFeatures(BaseModel):
    # Required fields
    request_id: str
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


def monitor_gpu_utilization():
    """Background thread to monitor and print GPU utilization every second"""
    logger.info("Starting GPU utilization monitoring thread")
    
    while True:
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            )
            gpu_util, mem_util = result.stdout.strip().split(',')
            logger.info(f"GPU Utilization: {gpu_util.strip()}%, Memory Utilization: {mem_util.strip()}%")
        except Exception as e:
            logger.error(f"Failed to get GPU utilization: {e}")
        
        # Sleep for one second
        time.sleep(1)

@app.on_event("startup")
async def start_gpu_monitoring():
    # Start GPU monitoring thread as a daemon (will exit when main thread exits)
    gpu_thread = threading.Thread(target=monitor_gpu_utilization, daemon=True)
    gpu_thread.start()
    logger.info("GPU monitoring thread started")

    
@app.on_event("startup")
async def load_model():
    global model_data, models, model_types
    
    model_path = os.environ.get("MODEL_PATH", "./latency_predictor.joblib")
    
    try:
        logger.info(f"Loading model from {model_path}")
        model_data = joblib.load(model_path)
        models = model_data.get('models', {})
        
        if not models:
            logger.error("No models found in model data")
            return
            
        logger.info(f"Loaded models for targets: {list(models.keys())}")
        
        # Determine model types and configure GPU acceleration if possible
        for target, model in models.items():
            model_type = type(model).__name__
            model_types[target] = model_type
            logger.info(f"Model for target '{target}' is type: {model_type}")
            
            # For Pipeline models, check the final estimator
            if model_type == 'Pipeline':
                steps = model.steps
                final_estimator_name, final_estimator = steps[-1]
                final_estimator_type = type(final_estimator).__name__
                logger.info(f"Pipeline for '{target}' has final estimator: {final_estimator_type}")
                
                # If the final estimator is XGBoost, enable GPU
                if 'XGB' in final_estimator_type:
                    try:
                        if hasattr(final_estimator, 'get_booster'):
                            booster = final_estimator.get_booster()
                            # Try to set tree_method and predictor to use GPU
                            booster.set_param({'tree_method': 'hist', 'device': 'cuda'})
                            logger.info(f"Enabled GPU acceleration for {target} XGBoost estimator")
                    except Exception as e:
                        logger.warning(f"Could not configure GPU for {target} XGBoost estimator: {e}")
            # Check for direct XGBoost models (not in a pipeline)
            elif 'XGB' in model_type:
                try:
                    if hasattr(model, 'get_booster'):
                        booster = model.get_booster()
                        booster.set_param({'tree_method': 'hist', 'device': 'cuda'})
                        logger.info(f"Enabled GPU acceleration for {target} XGBoost model")
                except Exception as e:
                    logger.warning(f"Could not configure GPU for {target} XGBoost model: {e}")
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")

@app.get("/health")
async def health_check():
    if not models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check GPU status with detailed metrics
    gpu_info = {"available": False}
    try:
        import torch
        if torch.cuda.is_available():
            # Get current GPU utilization
            gpu_info = {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
                "memory": {
                    "allocated": f"{torch.cuda.memory_allocated(0) / (1024**2):.2f} MB",
                    "reserved": f"{torch.cuda.memory_reserved(0) / (1024**2):.2f} MB",
                    "max": f"{torch.cuda.get_device_properties(0).total_memory / (1024**2):.2f} MB"
                }
            }
            
            # Try to get NVIDIA GPU utilization if possible
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, check=True)
                gpu_util, mem_util = result.stdout.strip().split(',')
                gpu_info["utilization"] = {
                    "gpu": f"{gpu_util.strip()}%",
                    "memory": f"{mem_util.strip()}%"
                }
            except:
                pass
    except:
        pass
        
    return {
        "status": "healthy", 
        "models": list(models.keys()),
        "model_types": model_types,
        "gpu": gpu_info
    }


@app.get("/gpu-status")
async def gpu_status():
    """Return GPU compute utilization"""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        gpu_utilization = result.stdout.strip()
        return {"gpu_utilization_percent": gpu_utilization}
    except Exception as e:
        return {"error": str(e), "gpu_available": False}

@app.get("/benchmark")
async def benchmark():
    """Run a performance test to compare CPU vs GPU inference"""
    if not models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Create a sample batch of requests
    batch_size = 1000
    sample_records = []
    
    # Create sample data similar to your production data
    for i in range(batch_size):
        record = {
            'selected_pod': f'pod-{i}',
            'input_tokens': 100 + i % 500,
            'output_tokens': 50 + i % 200,
            'total_tokens': 150 + i % 700,
            # Add other necessary fields...
        }
        sample_records.append(record)
    
    sample_df = pd.DataFrame(sample_records)
    
    results = {}
    for target, model in models.items():
        start_time = time.time()
        predictions = model.predict(sample_df)
        elapsed = time.time() - start_time
        
        results[target] = {
            "batch_size": batch_size,
            "prediction_time_ms": elapsed * 1000,
            "predictions_per_second": batch_size / elapsed,
            "avg_prediction_ms": (elapsed * 1000) / batch_size
        }
    
    return {
        "benchmark_results": results,
        "gpu_info": {
            "available": torch.cuda.is_available() if 'torch' in sys.modules else False,
            "device": torch.cuda.get_device_name(0) if 'torch' in sys.modules and torch.cuda.is_available() else "None"
        }
    }

@app.head("/")
@app.head("/predict")
async def head_endpoint():
    try:
        return Response(status_code=200)
    except Exception as e:
        logger.error(f"Error in HEAD endpoint: {e}")
        return Response(status_code=500)

# Function to process a single pod prediction in a separate thread
def process_pod_prediction(pod, models):
    request_id = pod.request_id
    pod_ip = pod.selected_pod
    pod_predictions = {}
    
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
    
    df = pd.DataFrame([record])
    logger.debug(f"Prediction features for pod {pod_ip}: {record}")
    from xgboost import DMatrix
    # Create a DMatrix with the device parameter
    try:
        dmatrix = DMatrix(df, device='cuda:0')
        use_dmatrix = True
    except Exception as e:
        use_dmatrix = False
        logger.debug(f"Unable to create CUDA DMatrix: {e}, falling back to DataFrame")
    
    for target, model in models.items():
        try:
            # Try to get the underlying booster
            if hasattr(model, 'get_booster'):
                booster = model.get_booster()
                if use_dmatrix:
                    pred = booster.predict(dmatrix)[0]
                else:
                    pred = model.predict(df)[0]
            else:
                # Directly use the model as fallback
                pred = model.predict(df)[0]
                
            pod_predictions[target] = float(pred)
            logger.info(f"SUCCESS - prediction - requestID: {request_id}, Pod features for: {pod_ip} target {target}. prediction: {int(pred)}ms")
        except Exception as e:
            logger.error(f"FAIL - Error predicting - requestID: {request_id}, Pod features for: {pod_ip}. target {target}. error: {e}")
            logger.error(f"DataFrame columns: {df.columns.tolist()}")
            pod_predictions[target] = -1
    for target, model in models.items():
        try:
            pred = model.predict(df)[0]  # Get first row value
            pod_predictions[target] = float(pred)
            logger.info(f"SUCCESS - prediction - requestID: {request_id}, Pod features for: {pod_ip} target {target}. prediction: {int(pred)}ms")
        except Exception as e:
            logger.error(f"FAIL - Error predicting - requestID: {request_id}, Pod features for: {pod_ip}. target {target}. error: {e}")
            logger.error(f"DataFrame columns: {df.columns.tolist()}")
            pod_predictions[target] = -1
    
    return pod_ip, pod_predictions

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    global pending_requests, active_requests, completed_requests
    
    # Track a new request arrival
    with request_counter_lock:
        pending_requests += 1
        queue_position = pending_requests
        queue_depth = pending_requests - active_requests  # Actual queue (waiting, not processing)
    
    ts = time.time()

    if not models:
        with request_counter_lock:
            pending_requests -= 1
            completed_requests += 1
        raise HTTPException(status_code=503, detail="Model not loaded")
    

    request_id = request.pods[0].request_id
    logger.info(f"Request ID: {request_id}, Queue position: {queue_position}, Queue depth: {queue_depth}")
    
    # Now mark this request as active (it's being processed, not just queued)
    with request_counter_lock:
        active_requests += 1

    # Result dictionary: pod_ip -> {target -> prediction}
    predictions = {}
    
    # Submit all pod prediction tasks to the thread pool
    tasks = []
    request_id = request.pods[0].request_id
    for pod in request.pods:
        # Use asyncio to run the prediction in a thread pool
        task = asyncio.create_task(
            asyncio.to_thread(process_pod_prediction, pod, models)
        )
        tasks.append(task)
    
    # Wait for all predictions to complete
    results = await asyncio.gather(*tasks)
    
    # Collect results
    for pod_ip, pod_predictions in results:
        predictions[pod_ip] = pod_predictions
    
    with request_counter_lock:
        pending_requests -= 1
        active_requests -= 1
        completed_requests += 1
        current_queue_depth = pending_requests - active_requests

    logger.info(f"requestID: {request_id}, parallel predictions took {int((time.time() - ts)*1000)}ms, queue depth: {current_queue_depth}")

    return ORJSONResponse(content={"predictions": predictions})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)