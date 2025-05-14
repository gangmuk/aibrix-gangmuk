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
import concurrent.futures
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

def process_pod_prediction(pod, models):
    overall_ts = time.time()
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
    
    # Create DataFrame for prediction
    df_start = time.time()
    df = pd.DataFrame([record])
    df_time = time.time() - df_start
    
    # Try to create a GPU-backed DMatrix
    dmatrix_start = time.time()
    try:
        from xgboost import DMatrix
        dmatrix = DMatrix(df, device='cuda:0')
        use_dmatrix = True
        logger.debug(f"Successfully created CUDA DMatrix for prediction")
    except Exception as e:
        use_dmatrix = False
        logger.debug(f"Unable to create CUDA DMatrix: {e}, falling back to CPU DataFrame")
    dmatrix_time = time.time() - dmatrix_start

    # SINGLE prediction loop for all targets
    for target, model in models.items():
        pred_start = time.time()
        try:
            # For Pipeline models, we need to use the pipeline's predict method
            # which handles preprocessing steps
            if model_types.get(target) == 'Pipeline':
                # Pipelines need to use the standard predict method
                pred = model.predict(df)[0]
            else:
                # For direct XGBoost models, we can try to use the GPU directly
                if use_dmatrix and hasattr(model, 'get_booster'):
                    booster = model.get_booster()
                    pred = booster.predict(dmatrix)[0]
                else:
                    # Fallback to standard prediction
                    pred = model.predict(df)[0]
            pod_predictions[target] = float(pred)
            pred_time = time.time() - pred_start
            logger.info(f"SUCCESS - prediction - requestID: {request_id}, Pod: {pod_ip}, target: {target}, prediction: {int(pred)}ms, took {int(pred_time * 1000)}ms")
        except Exception as e:
            logger.error(f"FAIL - Error predicting - requestID: {request_id}, Pod: {pod_ip}, target: {target}, error: {e}")
            logger.error(f"DataFrame columns: {df.columns.tolist()}")
            pod_predictions[target] = -1
    
    total_time = time.time() - overall_ts
    logger.info(f"process_pod_prediction, requestID: {request_id}, breakdown took - DataFrame: {int(df_time*1000)}ms, DMatrix: {int(dmatrix_time*1000)}ms, Total: {int(total_time*1000)}ms")
    return pod_ip, pod_predictions

# @app.post("/predict", response_model=PredictionResponse)
# async def predict(request: PredictionRequest):
#     global pending_requests, active_requests, completed_requests
    
#     # Track a new request arrival
#     with request_counter_lock:
#         pending_requests += 1
#         queue_position = pending_requests
#         queue_depth = pending_requests - active_requests  # Actual queue (waiting, not processing)
    
#     ts = time.time()

#     if not models:
#         with request_counter_lock:
#             pending_requests -= 1
#             completed_requests += 1
#         raise HTTPException(status_code=503, detail="Model not loaded")
    

#     request_id = request.pods[0].request_id
#     logger.info(f"Request ID: {request_id}, Queue position: {queue_position}, Queue depth: {queue_depth}")
    
#     # Now mark this request as active (it's being processed, not just queued)
#     with request_counter_lock:
#         active_requests += 1

#     # Result dictionary: pod_ip -> {target -> prediction}
#     predictions = {}
    
#     # Submit all pod prediction tasks to the thread pool
#     taskcreation_start = time.time()
#     tasks = []
#     request_id = request.pods[0].request_id
#     for pod in request.pods:
#         # Use asyncio to run the prediction in a thread pool
#         task = asyncio.create_task(
#             asyncio.to_thread(process_pod_prediction, pod, models)
#         )
#         tasks.append(task)
#     task_creation_time = time.time() - taskcreation_start
    
#     # Wait for all predictions to complete
#     gather_start = time.time()
#     results = await asyncio.gather(*tasks)
#     gather_time = time.time() - gather_start

#     # Collect results
#     for pod_ip, pod_predictions in results:
#         predictions[pod_ip] = pod_predictions
    
#     with request_counter_lock:
#         pending_requests -= 1
#         active_requests -= 1
#         completed_requests += 1
#         current_queue_depth = pending_requests - active_requests

#     logger.info(f"requestID: {request_id}, parallel predictions took {int((time.time() - ts)*1000)}ms, task creation took {int(task_creation_time*1000)}ms, gather took {int(gather_time*1000)}ms, current queue depth: {current_queue_depth}")

#     return ORJSONResponse(content={"predictions": predictions})









@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    request_id = request.pods[0].request_id
    ts_start = time.time()
    logger.info(f"requestID: {request_id}, TS_START: {ts_start:.6f}, starting batch prediction request")
    
    # Phase 1: Record creation
    ts_before_records = time.time()
    logger.info(f"requestID: {request_id}, TS_BEFORE_RECORDS: {ts_before_records:.6f}, delta: {(ts_before_records-ts_start)*1000:.2f}ms")
    
    records = []
    pod_map = {}  # To track which index maps to which pod
    
    for i, pod in enumerate(request.pods):
        pod_ip = pod.selected_pod
        pod_map[i] = pod_ip
        
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
        
        records.append(record)
    
    ts_after_records = time.time()
    logger.info(f"requestID: {request_id}, TS_AFTER_RECORDS: {ts_after_records:.6f}, delta: {(ts_after_records-ts_before_records)*1000:.2f}ms, created {len(records)} records")
    
    # Phase 2: DataFrame creation
    ts_before_df = time.time()
    logger.info(f"requestID: {request_id}, TS_BEFORE_DF: {ts_before_df:.6f}, delta: {(ts_before_df-ts_after_records)*1000:.2f}ms")
    
    batch_df = pd.DataFrame(records)
    
    ts_after_df = time.time()
    logger.info(f"requestID: {request_id}, TS_AFTER_DF: {ts_after_df:.6f}, delta: {(ts_after_df-ts_before_df)*1000:.2f}ms, DataFrame shape: {batch_df.shape}")
    
    # Phase 3: Create DMatrix (optional, depending on model)
    ts_before_dmatrix = time.time()
    logger.info(f"requestID: {request_id}, TS_BEFORE_DMATRIX: {ts_before_dmatrix:.6f}, delta: {(ts_before_dmatrix-ts_after_df)*1000:.2f}ms")
    
    try:
        from xgboost import DMatrix
        dmatrix = DMatrix(batch_df, device='cuda:0')
        use_dmatrix = True
        logger.info(f"requestID: {request_id}, Created CUDA DMatrix for batch prediction")
    except Exception as e:
        use_dmatrix = False
        logger.info(f"requestID: {request_id}, Unable to create CUDA DMatrix: {str(e)}, falling back to DataFrame")
    
    ts_after_dmatrix = time.time()
    logger.info(f"requestID: {request_id}, TS_AFTER_DMATRIX: {ts_after_dmatrix:.6f}, delta: {(ts_after_dmatrix-ts_before_dmatrix)*1000:.2f}ms")
    
    # Phase 4: Initialize results dict
    ts_before_init = time.time()
    logger.info(f"requestID: {request_id}, TS_BEFORE_INIT: {ts_before_init:.6f}, delta: {(ts_before_init-ts_after_dmatrix)*1000:.2f}ms")
    
    predictions = {pod_ip: {} for pod_ip in pod_map.values()}
    
    ts_after_init = time.time()
    logger.info(f"requestID: {request_id}, TS_AFTER_INIT: {ts_after_init:.6f}, delta: {(ts_after_init-ts_before_init)*1000:.2f}ms")
    
    # Phase 5: Batch prediction for each target
    for target, model in models.items():
        ts_before_target = time.time()
        logger.info(f"requestID: {request_id}, TS_BEFORE_TARGET_{target}: {ts_before_target:.6f}")
        
        try:
            # Choose prediction method based on model type and DMatrix availability
            if model_types.get(target) == 'Pipeline':
                # For Pipeline models, use the standard prediction method
                batch_predictions = model.predict(batch_df)
                logger.info(f"requestID: {request_id}, Used Pipeline predict for {target}")
            else:
                # For direct XGBoost models
                if use_dmatrix and hasattr(model, 'get_booster'):
                    booster = model.get_booster()
                    batch_predictions = booster.predict(dmatrix)
                    logger.info(f"requestID: {request_id}, Used GPU DMatrix predict for {target}")
                else:
                    batch_predictions = model.predict(batch_df)
                    logger.info(f"requestID: {request_id}, Used standard predict for {target}")
            
            # Distribute results to corresponding pods
            for i, pred in enumerate(batch_predictions):
                pod_ip = pod_map[i]
                predictions[pod_ip][target] = float(pred)
            
            ts_after_target = time.time()
            target_time = (ts_after_target - ts_before_target) * 1000
            logger.info(f"requestID: {request_id}, TS_AFTER_TARGET_{target}: {ts_after_target:.6f}, delta: {target_time:.2f}ms, predicted {len(batch_predictions)} values")
            
        except Exception as e:
            ts_after_target = time.time()
            logger.error(f"requestID: {request_id}, Error in batch prediction for {target}: {str(e)}")
            logger.error(f"requestID: {request_id}, DataFrame columns: {batch_df.columns.tolist()}")
            
            # Set default prediction value for error cases
            for i in range(len(records)):
                pod_ip = pod_map[i]
                predictions[pod_ip][target] = -1
            
            logger.info(f"requestID: {request_id}, TS_AFTER_TARGET_{target}_ERROR: {ts_after_target:.6f}, delta: {(ts_after_target-ts_before_target)*1000:.2f}ms")
    
    # Phase 6: Finalization
    ts_before_final = time.time()
    logger.info(f"requestID: {request_id}, TS_BEFORE_FINAL: {ts_before_final:.6f}, delta: {(ts_before_final-ts_after_target)*1000:.2f}ms")
    
    # Any final processing here
    
    ts_after_final = time.time()
    logger.info(f"requestID: {request_id}, TS_AFTER_FINAL: {ts_after_final:.6f}, delta: {(ts_after_final-ts_before_final)*1000:.2f}ms")
    
    # Phase 7: Response creation
    elapsed = int((time.time() - ts_start) * 1000)
    logger.info(f"requestID: {request_id}, TS_END: {time.time():.6f}, batch predictions completed in {elapsed}ms for {len(request.pods)} pods")
    
    return {"predictions": predictions}

# @app.post("/predict", response_model=PredictionResponse)
# def predict(request: PredictionRequest):
#     ts = time.time()
    
#     # Use direct threading without asyncio
#     with ThreadPoolExecutor(max_workers=16) as executor:
#         # Submit all tasks
#         future_to_pod = {
#             executor.submit(process_pod_prediction, pod, models): pod.selected_pod
#             for pod in request.pods
#         }
        
#         # Collect results as they complete
#         predictions = {}
#         for future in concurrent.futures.as_completed(future_to_pod):
#             pod_ip, pod_predictions = future.result()
#             predictions[pod_ip] = pod_predictions
    
#     elapsed = int((time.time() - ts) * 1000)
#     logger.info(f"requestID: {request.pods[0].request_id}, parallel predictions took {elapsed}ms")
    
#     return {"predictions": predictions}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)