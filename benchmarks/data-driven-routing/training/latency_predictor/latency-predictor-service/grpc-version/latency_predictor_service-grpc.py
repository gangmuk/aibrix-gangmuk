import grpc
from concurrent import futures
import joblib
import pandas as pd
import numpy as np
import os
import logging
import time
import sys
from concurrent.futures import ThreadPoolExecutor

# Import the generated protobuf modules
import prediction_service_pb2 as prediction_pb2
import prediction_service_pb2_grpc as prediction_pb2_grpc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for loaded model
model_data = None
models = {}
model_types = {}

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

# gRPC service implementation
class PredictionServicer(prediction_pb2_grpc.PredictionServiceServicer):
    def PredictLatency(self, request, context):
        start_time = time.time()
        
        # Create response object
        response = prediction_pb2.PredictionResponse(predictions={})
        
        # Process each pod in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all predictions
            futures = {}
            for pod in request.pods:
                future = executor.submit(self._predict_pod, pod)
                futures[future] = pod.selected_pod
            
            # Collect results
            for future in futures.as_completed(list(futures.keys())):
                pod_ip = futures[future]
                pod_results = future.result()
                
                # Create PodPrediction with metrics map
                pod_pred = prediction_pb2.PodPrediction(metrics={})
                
                # Add each target prediction to the metrics map
                for target, value in pod_results.items():
                    pod_pred.metrics[target] = value
                
                # Add to response
                response.predictions[pod_ip] = pod_pred
        
        request_id = request.pods[0].request_id if request.pods else "unknown"
        logger.info(f"gRPC requestID: {request_id}, PredictLatency took {(time.time() - start_time)*1000:.1f}ms")
        return response
    
    def _predict_pod(self, pod):
        # Convert protobuf message to pandas DataFrame
        record = {
            'selected_pod': pod.selected_pod,
            'input_tokens': pod.input_tokens,
            'output_tokens': pod.output_tokens,
            'total_tokens': pod.total_tokens,
        }
        
        # Add all optional fields
        if pod.kv_hit_ratio != 0:
            record['kv_hit_ratio'] = pod.kv_hit_ratio
        if pod.inflight_requests != 0:
            record['inflight_requests'] = pod.inflight_requests
        if pod.gpu_kv_cache != 0:
            record['gpu_kv_cache'] = pod.gpu_kv_cache
        if pod.cpu_kv_cache != 0:
            record['cpu_kv_cache'] = pod.cpu_kv_cache
        if pod.running_requests != 0:
            record['running_requests'] = pod.running_requests
        if pod.waiting_requests != 0:
            record['waiting_requests'] = pod.waiting_requests
        if pod.prefill_tokens != 0:
            record['prefill_tokens'] = pod.prefill_tokens
        if pod.decode_tokens != 0:
            record['decode_tokens'] = pod.decode_tokens
        if pod.gpu_model:
            record['gpu_model'] = pod.gpu_model
        if pod.last_second_avg_ttft_ms != 0:
            record['last_second_avg_ttft_ms'] = pod.last_second_avg_ttft_ms
        if pod.last_second_avg_tpot_ms != 0:
            record['last_second_avg_tpot_ms'] = pod.last_second_avg_tpot_ms
        if pod.last_second_p99_ttft_ms != 0:
            record['last_second_p99_ttft_ms'] = pod.last_second_p99_ttft_ms
        if pod.last_second_p99_tpot_ms != 0:
            record['last_second_p99_tpot_ms'] = pod.last_second_p99_tpot_ms
        if pod.last_second_total_requests != 0:
            record['last_second_total_requests'] = pod.last_second_total_requests
        if pod.last_second_total_tokens != 0:
            record['last_second_total_tokens'] = pod.last_second_total_tokens
        if pod.last_second_total_decode_tokens != 0:
            record['last_second_total_decode_tokens'] = pod.last_second_total_decode_tokens
        if pod.last_second_total_prefill_tokens != 0:
            record['last_second_total_prefill_tokens'] = pod.last_second_total_prefill_tokens
        
        df = pd.DataFrame([record])
        
        # Make predictions
        pod_predictions = {}
        for target, model in models.items():
            try:
                pred = model.predict(df)[0]
                pod_predictions[target] = float(pred)
                logger.info(f"SUCCESS - gRPC prediction - requestID: {pod.request_id}, Pod: {pod.selected_pod}, target {target}, prediction: {int(pred)}ms")
            except Exception as e:
                logger.error(f"FAIL - Error in gRPC prediction - requestID: {pod.request_id}, Pod: {pod.selected_pod}, target {target}, error: {e}")
                pod_predictions[target] = -1.0
        
        return pod_predictions

def load_model():
    """Load the model from disk"""
    global model_data, models, model_types
    
    model_path = os.environ.get("MODEL_PATH", "./latency_predictor.joblib")
    
    try:
        logger.info(f"Loading model from {model_path}")
        model_data = joblib.load(model_path)
        models = model_data.get('models', {})
        
        if not models:
            logger.error("No models found in model data")
            return False
            
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
                            booster.set_param({'predictor': 'gpu_predictor'})
                            logger.info(f"Enabled GPU acceleration for {target} XGBoost estimator")
                    except Exception as e:
                        logger.warning(f"Could not configure GPU for {target} XGBoost estimator: {e}")
        return True
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def serve():
    """Start the gRPC server"""
    # Load model first
    if not load_model():
        logger.error("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Configure and start the gRPC server
    grpc_port = int(os.environ.get("GRPC_PORT", 8081))
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ('grpc.keepalive_time_ms', 10000),  # 10 seconds
            ('grpc.keepalive_timeout_ms', 5000),  # 5 seconds
            ('grpc.keepalive_permit_without_calls', 1),  # Allow pings without active calls
            ('grpc.http2.max_pings_without_data', 0),  # Allow unlimited pings
            ('grpc.http2.min_time_between_pings_ms', 10000),  # 10 seconds between pings
            ('grpc.http2.min_ping_interval_without_data_ms', 5000),  # 5 seconds if there's no data
        ]
    )
    prediction_pb2_grpc.add_PredictionServiceServicer_to_server(
        PredictionServicer(), server)
    
    server.add_insecure_port(f'[::]:{grpc_port}')
    server.start()
    logger.info(f"gRPC server started on port {grpc_port}")
    
    # Keep server running
    try:
        while True:
            time.sleep(86400)  # Sleep for a day
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server...")
        server.stop(0)

if __name__ == "__main__":
    serve()