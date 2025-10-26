## routing_agent_service.py

# import threading
# import joblib
import pandas as pd
import numpy as np
# import uvicorn
# from pydantic import BaseModel
import os
import time
# import asyncio
# from concurrent.futures import ThreadPoolExecutor
import sys
# import concurrent.futures
import encoding
# import sac
# import ppo
# import contextual_bandit
import simpler_contextual_bandit
import latency_predictor
from rl_routing_agent_sb3 import create_rl_routing_agent_sb3, infer_rl_agent
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import preprocess
import pickle
import threading
from kubernetes import client, config
import ast
import json
import data_normalizer
import signal
import sys
import socket
import utils as utils
from kubernetes import client, config
from logger import logger, INCLUDE_GPU_IN_FEATURE
import queue
from collections import deque
from rwlock import RWLock


## colors for logging
BLUE_COLOR = "\033[94m"
RED_COLOR = "\033[91m"
GREEN_COLOR = "\033[92m"
PURPLE_COLOR = "\033[95m"
CYAN_COLOR = "\033[96m"
MAGENTA_COLOR = "\033[95m"
RESET_COLOR = "\033[0m" 

# INCLUDE_GPU_IN_FEATURE = True

app = Flask(__name__)
hyperparameter_file_path = '/app/final_model/model_config.json'
NUM_FLUSH = 0
ENCODED_DATA_DIR = "encoded_data"
final_model_dir = "/app/final_model"
feature_normalization_stats_file = f"{final_model_dir}/feature_normalization_statistics.csv"  # Add this near the top with your other constants;
NUM_TRAINS = 0
MODEL_UPDATED = True
LOCK_TRAINING_DATA = threading.Lock()
first_request_starting_time = None
stats_instance = None
TOTAL_NUM_DATA = 0
NUM_NEW_DATA = 0
TRAINING_RIGHT_NOW = False

# Training data accumulation (offline + online)
TRAINING_DF = None  # Holds all training data (offline CSV + online appended data)
TRAINING_DF_LOCK = threading.Lock()  # Thread safety for concurrent flush/train
PRINT_ONCE_AT_THE_FIRST_REQUEST = True
# RL agent globals
RL_AGENT = None  # Old RL agent (entire cluster as input) - for 'rl_agent' subAlgorithm
SCALABLE_RL_AGENT = None  # New scalable RL agent (pod-independent) - for 'scalable_rl_agent' subAlgorithm
LATENCY_PREDICTOR = None  # Latency predictor model - for 'latency_predictor' subAlgorithm
# RWLock enables concurrent predictions (readers) with exclusive updates (writer)
# - Predictions: use rwlock.read() for high concurrency
# - Updates: use rwlock.write() for exclusive access
# - Initialization: use rwlock.write() for exclusive access
RL_AGENT_LOCK = RWLock()
SCALABLE_RL_AGENT_LOCK = RWLock()
LATENCY_PREDICTOR_LOCK = RWLock()

# Multi-model registry for per-workload, per-GPU routing models
# Key: (workload, gpu_type) tuple, Value: model instance
MODEL_REGISTRY = {}
MODEL_REGISTRY_LOCK = RWLock()
WORKLOAD = os.getenv("WORKLOAD", "A") # default workload
logger.info(f"WORKLOAD: {WORKLOAD}")

# Predictor instance cache for per-workload, per-GPU latency predictors
# Key: (workload, gpu_type) tuple, Value: LatencyPredictor instance
PREDICTOR_REGISTRY = {}
PREDICTOR_REGISTRY_LOCK = RWLock()

# Scalable RL agent training thread
SCALABLE_RL_TRAINING_THREAD = None
SCALABLE_RL_TRAINING_SHUTDOWN = threading.Event()


# RL agent async update queue
RL_UPDATE_QUEUE = queue.Queue(maxsize=1000)  # Bounded queue to prevent memory issues
RL_UPDATE_THREAD = None
RL_UPDATE_SHUTDOWN = threading.Event()

# Request completion tracking (for scalable RL async completion)
PENDING_REQUESTS = {}  # request_id → (route_time, selected_pod_idx)
PENDING_REQUESTS_LOCK = threading.Lock()

MIN_NUM_TRAINING_DATA = int(os.getenv("MIN_NUM_TRAINING_DATA", 1000))
POD_LABEL_SELECTOR = os.getenv("POD_LABEL_SELECTOR", "model.aibrix.ai/name=llama3-1-8b")
logger.info(f"POD_LABEL_SELECTOR: {POD_LABEL_SELECTOR}")
if POD_LABEL_SELECTOR == "":
    logger.error(f"POD_LABEL_SELECTOR is empty")
    assert False
ENABLE_ONLINE_LEARNING = int(os.getenv("ENABLE_ONLINE_LEARNING", 0))
EXPLORATION_ENABLED = int(os.getenv("EXPLORATION_ENABLED", 0))
TTFT_REWARD_WEIGHT = float(os.getenv("TTFT_REWARD_WEIGHT", 0.5))
RL_MODEL_HYPERPARAMETERS = None

# Multi-model configuration: workload types for per-workload-per-GPU models
WORKLOAD_TYPES = [w.strip() for w in os.getenv("WORKLOAD_TYPES", "A").split(",")]
logger.info(f"WORKLOAD_TYPES: {WORKLOAD_TYPES}")

BROKER_LOCK = RWLock()

request_features_train = ['input_tokens', 'output_tokens', 'total_tokens']
# request_features_reward = ['ttft', 'avg_tpot', 'e2e_latency']

# @app.route("/request_complete", methods=["POST"])
# def handle_request_complete():
#     """
#     Endpoint for async request completion notifications (for scalable_rl_agent).
    
#     Expected payload:
#     {
#         "request_id": "req_12345",
#         "ttft": 45.6,           # milliseconds
#         "tpot": 12.3,           # milliseconds  
#         "selected_pod": "10.0.1.30"
#     }
#     """
#     global SCALABLE_RL_AGENT, RL_MODEL_HYPERPARAMETERS
    
#     try:
#         data = request.json
#         request_id = data.get('request_id')
#         ttft = data.get('ttft')
#         tpot = data.get('tpot')
#         selected_pod = data.get('selected_pod')
        
#         if not request_id or ttft is None or tpot is None:
#             logger.error(f"Missing required fields in request completion: {data}")
#             return jsonify({"error": "Missing required fields"}), 400
        
#         if SCALABLE_RL_AGENT is None:
#             logger.debug(f"Scalable RL agent not initialized, ignoring completion for {request_id}")
#             return jsonify({"status": "ok", "message": "agent not initialized"}), 200
        
#         # Get current cluster state (after completion)
#         try:
#             pod_features, kv_hit_ratios, request_features = get_current_cluster_features()
#             current_state = (pod_features, kv_hit_ratios, request_features)
            
#             # Complete the experience
#             on_request_complete_callback(
#                 rl_agent=SCALABLE_RL_AGENT,
#                 request_id=request_id,
#                 current_cluster_state=current_state,
#                 ttft=ttft,
#                 tpot=tpot,
#                 hyperparameters=RL_MODEL_HYPERPARAMETERS
#             )
            
#             logger.debug(f"✅ Completed experience for request {request_id} (ttft={ttft}ms, tpot={tpot}ms)")
#             return jsonify({"status": "ok"}), 200
            
#         except NotImplementedError:
#             logger.debug(f"⚠️  get_current_cluster_features() not implemented, skipping completion for {request_id}")
#             return jsonify({"status": "ok", "message": "cluster state fetch not implemented"}), 200
            
#     except Exception as e:
#         logger.error(f"Error in request completion handler: {e}")
#         import traceback
#         logger.error(traceback.format_exc())
#         return jsonify({"error": str(e)}), 500


def slice_tensor_data_for_pods(tensor_data, pod_indices, sorted_all_pod_ids):
    """
    Slice tensor_data to include only specific pods by their indices.
    
    Args:
        tensor_data: Dict containing tensors with pod dimension
        pod_indices: List of pod indices to keep
        sorted_all_pod_ids: List of all pod IDs (for reference)
        
    Returns:
        New tensor_data dict with sliced tensors
    """
    import torch
    
    sliced_data = {}
    
    # Slice pod_features (shape: [batch, num_pods, features])
    if 'pod_features' in tensor_data:
        sliced_data['pod_features'] = tensor_data['pod_features'][:, pod_indices, :]
    
    if 'pod_features_with_staleness' in tensor_data:
        sliced_data['pod_features_with_staleness'] = tensor_data['pod_features_with_staleness'][:, pod_indices, :]
    
    # Slice kv_hit_ratios (shape: [batch, num_pods, kv_dim])
    if 'kv_hit_ratios' in tensor_data:
        sliced_data['kv_hit_ratios'] = tensor_data['kv_hit_ratios'][:, pod_indices, :]
    
    # Request features don't need slicing (same for all pods)
    if 'request_features' in tensor_data:
        sliced_data['request_features'] = tensor_data['request_features']
    
    return sliced_data


def extract_workload_from_log(log_message, default_workload='A'):
    """
    Extract workload identifier from log message.
    
    Expected format: ...workload@<workload_id>@...
    
    Args:
        log_message: Raw log message string
        default_workload: Default workload if not found in message
        
    Returns:
        Workload identifier string
    """
    try:
        parts = log_message.split("workload@")
        if len(parts) > 1:
            workload_parts = parts[1].split("@")
            if workload_parts:
                return workload_parts[0].strip()
    except Exception as e:
        logger.debug(f"Error extracting workload: {e}")
    
    return default_workload


def group_pods_by_gpu(sorted_all_pod_ids, generalpodid_to_gpu_model):
    """
    Group pod indices by their GPU type.
    
    Args:
        sorted_all_pod_ids: List of pod IDs in sorted order
        generalpodid_to_gpu_model: Mapping from pod ID to GPU model
        
    Returns:
        Dict {gpu_type: [pod_indices]}
    """
    pods_by_gpu = {}
    
    for idx, pod_id in enumerate(sorted_all_pod_ids):
        if pod_id in generalpodid_to_gpu_model:
            gpu_type = generalpodid_to_gpu_model[pod_id]
            if gpu_type not in pods_by_gpu:
                pods_by_gpu[gpu_type] = []
            pods_by_gpu[gpu_type].append(idx)
        else:
            logger.warning(f"Pod {pod_id} not found in GPU mapping, skipping")
    
    return pods_by_gpu


def infer_for_gpu_batch(workload, gpu_type, pod_indices, tensor_data, sorted_all_pod_ids, 
                        request_id, model_type, hyperparameters):
    """
    Run inference for a batch of pods with the same GPU type using workload-specific model.
    
    This function is designed for PARALLEL execution across multiple GPU types.
    Each GPU batch uses its own cached predictor instance to avoid lock contention.
    
    Args:
        workload: Workload identifier
        gpu_type: GPU model type
        pod_indices: List of pod indices to infer for
        tensor_data: Full tensor data (will be sliced)
        sorted_all_pod_ids: All pod IDs in order
        request_id: Request identifier
        model_type: Model type ('contextual_bandit', 'latency_predictor', etc.)
        hyperparameters: Model hyperparameters
        
    Returns:
        Dict {pod_idx: {score, probability, latency}} for pods in this GPU batch
    """
    
    overhead_summary = {}
    start_time = time.time()
    
    try:
        # Get model for this (workload, gpu_type)
        global MODEL_REGISTRY, MODEL_REGISTRY_LOCK, PREDICTOR_REGISTRY, PREDICTOR_REGISTRY_LOCK
        
        model_info = None
        with MODEL_REGISTRY_LOCK.read():
            model_key = (workload, gpu_type)
            if model_key in MODEL_REGISTRY:
                model_info = MODEL_REGISTRY[model_key]
            else:
                logger.warning(f"No model found for (workload={workload}, gpu={gpu_type}), falling back to default")
                # Try to find any model for this workload
                for key in MODEL_REGISTRY.keys():
                    if key[0] == workload:
                        model_info = MODEL_REGISTRY[key]
                        logger.info(f"Using fallback model from {key}")
                        break
        
        if model_info is None:
            logger.error(f"No model available for workload={workload}, gpu={gpu_type}")
            # Return uniform scores
            return {idx: {'score': 1.0/len(pod_indices), 'probability': 1.0/len(pod_indices), 'latency': -1} 
                    for idx in pod_indices}
        
        overhead_summary['get_model'] = time.time() - start_time
        
        # Slice tensor data for this GPU's pods
        slice_start = time.time()
        sliced_tensor_data = slice_tensor_data_for_pods(tensor_data, pod_indices, sorted_all_pod_ids)
        overhead_summary['slice_tensors'] = time.time() - slice_start
        
        # Run inference with the appropriate model
        infer_start = time.time()
        
        if model_type == 'latency_predictor':
            # Use latency predictor with cached instance per (workload, gpu_type)
            import latency_predictor
            
            predictor_key = (workload, gpu_type)
            predictor = None
            
            # Try to get existing predictor (read lock for common case)
            with PREDICTOR_REGISTRY_LOCK.read():
                predictor = PREDICTOR_REGISTRY.get(predictor_key)
            
            # Create predictor if needed (upgrade to write lock)
            if predictor is None:
                with PREDICTOR_REGISTRY_LOCK.write():
                    # Double-check after acquiring write lock (another thread might have created it)
                    if predictor_key not in PREDICTOR_REGISTRY:
                        state_dims = {
                            'pod_features': sliced_tensor_data['pod_features_with_staleness'].shape[2],
                            'kv_hit_ratios': sliced_tensor_data['kv_hit_ratios'].shape[2],
                            'request_features': sliced_tensor_data['request_features'].shape[1],
                            'num_pods': len(pod_indices)
                        }
                        
                        new_predictor = latency_predictor.LatencyPredictor(
                            state_dims, hyperparameters, model_info['model_dir']
                        )
                        
                        # Load model
                        try:
                            new_predictor.load(model_info['model_dir'])
                            logger.info(f"Loaded latency predictor for (workload={workload}, gpu={gpu_type})")
                            # Cache the predictor only if load succeeds
                            PREDICTOR_REGISTRY[predictor_key] = new_predictor
                        except Exception as e:
                            logger.error(f"Failed to load latency predictor: {e}")
                            # Return uniform fallback (don't cache failed predictor)
                            return {idx: {'score': 1.0/len(pod_indices), 'probability': 1.0/len(pod_indices), 'latency': -1} 
                                    for idx in pod_indices}
                    
                    # Get predictor reference while still holding write lock (now guaranteed to exist)
                    predictor = PREDICTOR_REGISTRY[predictor_key]
            
            # Run inference (NO LOCK needed - predictor.predict() is thread-safe for read-only operations)
            # This allows true parallel inference across different GPU batches
            gpu_batch_pod_ids = [sorted_all_pod_ids[idx] for idx in pod_indices]
            result, _ = latency_predictor.infer_latency_predictor_with_model(
                predictor, sliced_tensor_data, request_id, gpu_batch_pod_ids
            )
            
            # Map results back to original indices
            results = {}
            predicted_latencies = result.get('predicted_latencies', {})
            for local_idx, global_idx in enumerate(pod_indices):
                pod_id = sorted_all_pod_ids[global_idx]
                latency = predicted_latencies.get(pod_id, -1)
                # Lower latency = higher score
                score = 1.0 / (latency + 1e-6) if latency > 0 else 1.0
                results[global_idx] = {
                    'score': score,
                    'probability': score,  # Will be normalized later
                    'latency': latency
                }
        
        elif model_type == 'contextual_bandit':
            # Use contextual bandit
            import simpler_contextual_bandit
            
            result, _ = simpler_contextual_bandit.infer_from_tensor(
                sliced_tensor_data, request_id, True, hyperparameters, model_info['model_dir']
            )
            
            # Map results back to original indices
            selected_local_idx = result['selected_pod_index']
            probabilities = result.get('pod_probabilities', [])
            
            results = {}
            for local_idx, global_idx in enumerate(pod_indices):
                prob = probabilities[local_idx] if isinstance(probabilities, list) and local_idx < len(probabilities) else 1.0/len(pod_indices)
                results[global_idx] = {
                    'score': prob,
                    'probability': prob,
                    'latency': -1
                }
        
        else:
            logger.warning(f"Unknown model type: {model_type}")
            results = {idx: {'score': 1.0/len(pod_indices), 'probability': 1.0/len(pod_indices), 'latency': -1} 
                      for idx in pod_indices}
        
        overhead_summary['model_inference'] = time.time() - infer_start
        overhead_summary['total'] = time.time() - start_time
        
        logger.debug(f"GPU batch inference for {gpu_type}: {len(pod_indices)} pods, {overhead_summary['total']*1000:.1f}ms")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in GPU batch inference for {gpu_type}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Return uniform fallback
        return {idx: {'score': 1.0/len(pod_indices), 'probability': 1.0/len(pod_indices), 'latency': -1} 
                for idx in pod_indices}


# Fixed handle_flush function
@app.route("/flush", methods=["POST"])
def handle_flush():
    global NUM_FLUSH, ENCODED_DATA_DIR, TOTAL_NUM_DATA, NUM_NEW_DATA, feature_normalization_stats_file, stats_instance, TRAINING_DF
    NUM_FLUSH += 1
    flush_start_time = time.time()
    log_data = request.json
    try:
        logger.info(f"Received log data with {len(log_data) if log_data else 0} entries")
        if not os.path.exists("raw_training_data"):
            os.mkdir("raw_training_data")
        raw_data_path = f"raw_training_data/batch_{NUM_FLUSH}.csv"

        # Write raw data to file
        ts_write_raw_data = time.time()
        ##################################################
        ## Write raw data to file
        utils.write_to_file(log_data, raw_data_path)
        ##################################################
        logger.info(f"wrote {len(log_data)} entries to {raw_data_path}, took {time.time() - ts_write_raw_data} seconds")

        podip_replaced_data_path = utils.replace_pod_ip_with_generalpodid(raw_data_path)
        ts_preprocess = time.time()
        ##################################################
        ## Preprocess
        processed_df, sorted_all_pod_ids, _ = preprocess.main(podip_replaced_data_path, "", RL_MODEL_HYPERPARAMETERS)
        ##################################################
        logger.info(f"Successfully parsed data, took {time.time() - ts_preprocess} seconds")

        # Append preprocessed data to TRAINING_DF for online learning
        if ENABLE_ONLINE_LEARNING:
            with TRAINING_DF_LOCK:
                if TRAINING_DF is None:
                    TRAINING_DF = processed_df.copy()
                    logger.info(f"Initialized TRAINING_DF with {len(processed_df)} samples")
                else:
                    old_size = len(TRAINING_DF)
                    TRAINING_DF = pd.concat([TRAINING_DF, processed_df], ignore_index=True)
                    logger.info(f"Appended {len(processed_df)} samples to TRAINING_DF (total: {old_size} → {len(TRAINING_DF)})")

        logger.info(f"Successfully flushed {len(log_data)} log messages, took {time.time() - flush_start_time} seconds")
        TOTAL_NUM_DATA += len(log_data)
        NUM_NEW_DATA += len(log_data)

        return jsonify({"status": "success", "message": f"Successfully processed {len(log_data)} log messages"}), 200

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        return jsonify({"status": "error", "message": str(e), "traceback": error_traceback}), 500


@app.route("/infer", methods=["POST"])
def handle_infer():
    global NUM_TRAINS, MODEL_UPDATED, first_request_starting_time, stats_instance, RL_MODEL_HYPERPARAMETERS, PRINT_ONCE_AT_THE_FIRST_REQUEST
    handle_infer_overhead_summary = {}
    if first_request_starting_time == None:
        first_request_starting_time = time.time()
        logger.info(f"First request starting time set to {first_request_starting_time}")
    # if NUM_TRAINS == 0:
    #     logger.warning("No trained model available, please call /flush to train first")
    #     return jsonify({"error": "No trained model available, please call /flush to train first"}), 503
    
    handle_infer_start_time = time.time()
    try:
        # Get the log message as a string from the request body
        request_prepare_start_time = time.time()
        log_data = request.json
        if isinstance(log_data, str):
            log_message = log_data
        else:
            # Handle dict input (original logic)
            if len(list(log_data.keys())) != 1:
                logger.error(f"There must be only one request for inference, but got {len(list(log_data.keys()))} requests")
                return jsonify({"error": "Invalid request format"}), 400
            
            first_key = list(log_data.keys())[0]
            log_message = log_data[first_key]
        logger.debug(f"Received inference request in handle_infer, log_message:\n{log_message}")

        # Extract request ID for logging purposes
        request_id = "default"  # or extract from log_message
        replace_podid_start_time = time.time()
        log_message = utils.replace_pod_ip_with_generalpodid(log_message)
        # logger.info(f"log_message_with_replaced_pod_id: {log_message}")
        handle_infer_overhead_summary["replace_podid_overhead"] = time.time() - replace_podid_start_time
        parts = log_message.split("requestID@")
        if len(parts) > 1:
            request_id_parts = parts[1].split("@")
            if request_id_parts:
                request_id = request_id_parts[0]
        else:
            logger.warning("No request ID found in log message, using default 'default'")
        handle_infer_overhead_summary["request_prepare"] = time.time() - request_prepare_start_time
        
        # Use the existing preprocessing function to parse the log
        preprocess_start_time = time.time()
        processed_df, sorted_all_pod_ids, preprocess_dataset_overhead_summary = preprocess.main(None, log_message, RL_MODEL_HYPERPARAMETERS)
        if PRINT_ONCE_AT_THE_FIRST_REQUEST:
            logger.info(f"processed_df.columns: {list(processed_df.columns)}")
            logger.info(f"sorted_all_pod_ids: {sorted_all_pod_ids}")
            PRINT_ONCE_AT_THE_FIRST_REQUEST = False
        handle_infer_overhead_summary["preprocess_overhead"] = time.time() - preprocess_start_time

        normalize_start = time.time()
        if stats_instance is None:
            logger.error(f"No running statistics available, stats_instance: {stats_instance}")
            logger.error("Cannot perform inference without normalization statistics")
            return jsonify({"error": "No normalization statistics available"}), 500
            assert False
        if stats_instance.get_max_count() == 0:
            logger.error(f"Stats instance count is 0, no data available for normalization")
            assert False

        normalizable_features, non_normalizable_features = data_normalizer._get_normalizable_features(processed_df, RL_MODEL_HYPERPARAMETERS.get('NO_NORMALIZE_FEATURES', []))
        if stats_instance.get_max_count() == 0:
            logger.error(f"request_id,{request_id},No normalization statistics available for inference")
            assert False
            
        non_interest = ['request_id', 'requestID', 'ttft', 'avg_tpot', 'e2e_latency', 'selected_pod', 'request_start_time', 'request_end_time']
        features_must_exist_in_stats_instance = []
        for feature in processed_df.columns:
            # NOTE: ignoring last_second_* features
            if "last_second_" not in feature and feature not in non_interest and feature in normalizable_features:
                features_must_exist_in_stats_instance.append(feature)
        for feature in features_must_exist_in_stats_instance:
            if feature not in stats_instance.feature_stats:
                logger.error(f"Feature {feature} not found in stats_instance")
                # logger.error(f"processed_df.columns: {list(processed_df.columns)}")
                # logger.error(f"features_must_exist_in_stats_instance: {features_must_exist_in_stats_instance}")
                # logger.error(f"Available stats features: {list(stats_instance.feature_stats.keys())}")
                assert False
                
        for feature in normalizable_features:
            ##################################################
            data_normalizer._normalize_single_feature(processed_df, feature, stats_instance, is_training=False, request_id=request_id)
            ##################################################
        handle_infer_overhead_summary["normalize"] = time.time() - normalize_start

        ## Encode data (normalization already done)
        encode_start_time = time.time()
        tensor_data, encode_for_inference_overhead_summary = encoding.encode_for_inference(sorted_all_pod_ids, processed_df, request_features_train, RL_MODEL_HYPERPARAMETERS)
        handle_infer_overhead_summary["encode"] = time.time() - encode_start_time

        infer_from_tensor_start_time = time.time()
        
        # ========================================
        # Multi-Model GPU-Specific Inference
        # ========================================
        global MODEL_REGISTRY, MODEL_REGISTRY_LOCK
        
        # Extract workload identifier from request
        workload = extract_workload_from_log(log_message, default_workload=WORKLOAD)
        logger.debug(f"Request {request_id}: workload={workload}")
        
        # Check if we should use multi-model GPU-specific inference
        use_multi_model = False
        with MODEL_REGISTRY_LOCK.read():
            use_multi_model = len(MODEL_REGISTRY) > 0
        
        if use_multi_model:
            logger.info(f"{CYAN_COLOR}Using multi-model GPU-specific inference for workload={workload}{RESET_COLOR}")
            
            # Group pods by GPU type
            group_start = time.time()
            generalpodid_to_gpu_model = RL_MODEL_HYPERPARAMETERS.get('generalpodid_to_gpu_model', {})
            pods_by_gpu = group_pods_by_gpu(sorted_all_pod_ids, generalpodid_to_gpu_model)
            handle_infer_overhead_summary["group_pods_by_gpu"] = time.time() - group_start
            
            logger.debug(f"Pods grouped by GPU: {[(gpu, len(indices)) for gpu, indices in pods_by_gpu.items()]}")
            
            # Parallel inference for each GPU type
            parallel_infer_start = time.time()
            model_type = RL_MODEL_HYPERPARAMETERS.get('MODEL_TYPE', 'latency_predictor')
            
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            all_results = {}  # {pod_idx: {score, probability, latency}}
            
            with ThreadPoolExecutor(max_workers=len(pods_by_gpu)) as executor:
                # Submit inference tasks for each GPU batch
                future_to_gpu = {}
                for gpu_type, pod_indices in pods_by_gpu.items():
                    future = executor.submit(
                        infer_for_gpu_batch,
                        workload,
                        gpu_type,
                        pod_indices,
                        tensor_data,
                        sorted_all_pod_ids,
                        request_id,
                        model_type,
                        RL_MODEL_HYPERPARAMETERS
                    )
                    future_to_gpu[future] = gpu_type
                
                # Collect results as they complete
                for future in as_completed(future_to_gpu):
                    gpu_type = future_to_gpu[future]
                    try:
                        gpu_results = future.result()
                        all_results.update(gpu_results)
                        logger.debug(f"GPU {gpu_type}: got {len(gpu_results)} pod results")
                    except Exception as e:
                        logger.error(f"Error in GPU batch for {gpu_type}: {e}")
            
            handle_infer_overhead_summary["parallel_gpu_inference"] = time.time() - parallel_infer_start
            
            # Combine results and select best pod
            combine_start = time.time()
            
            if len(all_results) == 0:
                logger.error("No results from any GPU batch, falling back to single-model")
                use_multi_model = False
            else:
                # Normalize scores across all pods
                all_scores = [res['score'] for res in all_results.values()]
                total_score = sum(all_scores) + 1e-9
                
                # Select pod with highest score
                best_pod_idx = max(all_results.keys(), key=lambda idx: all_results[idx]['score'])
                best_result = all_results[best_pod_idx]
                
                # Build result dict
                pod_probabilities = {}
                predicted_latencies = {}
                for idx in range(len(sorted_all_pod_ids)):
                    pod_id = sorted_all_pod_ids[idx]
                    if idx in all_results:
                        pod_probabilities[pod_id] = all_results[idx]['score'] / total_score
                        predicted_latencies[pod_id] = all_results[idx]['latency']
                    else:
                        pod_probabilities[pod_id] = 0.0
                        predicted_latencies[pod_id] = -1
                
                result = {
                    'selected_pod_index': best_pod_idx,
                    'confidence': best_result['score'] / total_score,
                    'pod_probabilities': pod_probabilities,
                    'explore_mask': 0,  # No exploration in multi-model mode
                    'predicted_latencies': predicted_latencies,
                    'chosen_pod_predicted_latency': best_result['latency']
                }
                
                infer_from_tensor_overhead_summary = {
                    'multi_model_combine': time.time() - combine_start,
                    'multi_model_total': time.time() - infer_from_tensor_start_time
                }
                
                logger.info(f"{GREEN_COLOR}Multi-model inference selected pod {best_pod_idx} ({sorted_all_pod_ids[best_pod_idx]}) "
                           f"with confidence {result['confidence']:.3f}{RESET_COLOR}")
        
        # Fallback to single-model inference if multi-model not available or failed
        if not use_multi_model:
            logger.debug("Using single-model inference (fallback)")
            
            # Route to appropriate model based on model type
            # model_type = RL_MODEL_HYPERPARAMETERS.get('MODEL_TYPE', 'contextual_bandit')
            subAlgorithm = processed_df['subAlgorithm'].iloc[0]
            if subAlgorithm == 'latency_predictor':
                logger.info(f"requestID: {request_id}, subAlgorithm: {subAlgorithm}")

                global LATENCY_PREDICTOR

                # Check if initialization needed without blocking
                if LATENCY_PREDICTOR is None:
                    with LATENCY_PREDICTOR_LOCK.write():
                        # Double-check after acquiring lock
                        if LATENCY_PREDICTOR is None:
                            state_dims = {
                                'pod_features': tensor_data['pod_features_with_staleness'].shape[2],
                                'kv_hit_ratios': tensor_data['kv_hit_ratios'].shape[2],
                                'request_features': tensor_data['request_features'].shape[1],
                                'num_pods': tensor_data['pod_features_with_staleness'].shape[1]
                            }
                            
                            logger.info(f"Initializing latency predictor with state_dims={state_dims}")
                            LATENCY_PREDICTOR = latency_predictor.LatencyPredictor(state_dims, RL_MODEL_HYPERPARAMETERS, final_model_dir)

                            # Load pretrained model
                            model_path = os.path.join(final_model_dir, 'latency_predictor.pth')
                            if os.path.exists(model_path):
                                try:
                                    LATENCY_PREDICTOR.load(final_model_dir)
                                    logger.info(f"Loaded latency predictor from {final_model_dir}")
                                except Exception as e:
                                    logger.error(f"Failed to load latency predictor: {e}")
                            else:
                                logger.warning(f"No pretrained latency predictor found at {model_path}, using untrained model")

                # Inference with read lock (allows concurrent requests)
                with LATENCY_PREDICTOR_LOCK.read():
                    result, infer_from_tensor_overhead_summary = latency_predictor.infer_latency_predictor_with_model(
                        predictor=LATENCY_PREDICTOR,
                        tensor_data=tensor_data,
                        request_id=request_id,
                        sorted_all_pod_ids=sorted_all_pod_ids
                    )
            elif subAlgorithm == 'contextual_bandit' or subAlgorithm == 'rl_naive':
                logger.info(f"subAlgorithm: {subAlgorithm}, Using contextual bandit model for inference (request_id: {request_id})")
                result, infer_from_tensor_overhead_summary = simpler_contextual_bandit.infer_from_tensor(tensor_data, request_id, MODEL_UPDATED, RL_MODEL_HYPERPARAMETERS, final_model_dir)
                result['predicted_latencies'] = {pod_id: -1 for pod_id in sorted_all_pod_ids}
                result['chosen_pod_predicted_latency'] = -1
            elif subAlgorithm == 'rl_agent':
                # === OLD RL AGENT (entire cluster as input state) ===
                logger.info(f"requestID: {request_id}, subAlgorithm: {subAlgorithm}, Using OLD RL agent (entire cluster) for inference")
                
                global RL_AGENT
                
                with RL_AGENT_LOCK.write():
                    # Check if initialization needed
                    pod_features_t = tensor_data['pod_features']
                    n_pods = int(pod_features_t.shape[1])
                    per_pod_dim = int(pod_features_t.shape[2])
                    
                    if (RL_AGENT is None or 
                        RL_AGENT.action_dim != n_pods or
                        RL_AGENT.state_dim.get('pod_features') != per_pod_dim):
                        # Initialize new agent
                        kv_hit_t = tensor_data['kv_hit_ratios']
                        req_features_t = tensor_data['request_features']
                        state_dim = {
                            'pod_features': per_pod_dim,
                            'kv_hit_ratios': int(kv_hit_t.shape[2]),
                            'request_features': int(req_features_t.shape[1]),
                        }
                        RL_AGENT = create_rl_routing_agent_sb3(
                            state_dim=state_dim,
                            action_dim=n_pods,
                            **RL_MODEL_HYPERPARAMETERS
                        )
                        ckpt_path = RL_MODEL_HYPERPARAMETERS.get('RL_CHECKPOINT_PATH')
                        if ckpt_path and os.path.exists(ckpt_path):
                            try:
                                RL_AGENT.load(ckpt_path)
                                logger.info(f"Loaded RL checkpoint from {ckpt_path}")
                            except Exception as e:
                                logger.error(f"Failed to load RL checkpoint {ckpt_path}: {e}")
                        logger.info(f"Initialized OLD RL agent with state_dim={state_dim}, action_dim={n_pods}")
                    
                    # Get agent reference under write lock
                    current_agent = RL_AGENT
                
                # Inference uses read lock for predictions (allows concurrency)
                current_agent, result, infer_from_tensor_overhead_summary = infer_rl_agent(
                    tensor_data=tensor_data,
                    request_id=request_id,
                    sorted_all_pod_ids=sorted_all_pod_ids,
                    processed_df=processed_df,
                    rl_agent=current_agent,
                    hyperparameters=RL_MODEL_HYPERPARAMETERS,
                    agent_lock=RL_AGENT_LOCK  # RWLock for read (predict) and write (buffer)
                )
                
                # Queue async update if online learning enabled
                update_overhead = 0.0
                if ENABLE_ONLINE_LEARNING:
                    update_start = time.time()
                    with RL_AGENT_LOCK.read():
                        if RL_AGENT is not None:
                            buffer_size = len(RL_AGENT.experience_buffer)
                            batch_size = RL_AGENT.hyperparameters.get('batch_size', 64)
                            
                            if buffer_size >= batch_size:
                                queue_rl_update(n_steps=batch_size)
                                logger.debug(f"Queued RL update: buffer_size={buffer_size}, batch_size={batch_size}")
                    update_overhead = time.time() - update_start
                
                infer_from_tensor_overhead_summary['online_update'] = update_overhead
            ####################################################################################
            ####################################################################################
            elif subAlgorithm == 'scalable_rl_agent':
                from scalable_rl_routing_agent import BROKER, infer
                
                # === NEW SCALABLE RL AGENT (pod-count independent) ===
                logger.info(f"scalable_rl_routing_agent, requestID: {request_id}, subAlgorithm: {subAlgorithm}, Using SCALABLE RL agent (pod-independent) for inference")
                
                # Extract features from tensor_data
                pod_features = tensor_data['pod_features'].cpu().numpy()[0]  # [num_pods, 10]
                kv_hit_ratios = tensor_data['kv_hit_ratios'].cpu().numpy()[0]  # [num_pods, 1]
                request_features = tensor_data['request_features'].cpu().numpy()[0]  # [3]
                temporal_features = np.array([1], dtype=np.float32)  # Empty for now
                
                # Get previous reward from processed_df (gateway provides this)
                if 'prev_reward' in processed_df.columns:
                    prev_reward = float(processed_df['prev_reward'].iloc[0])
                else:
                    logger.error(f"scalable_rl_routing_agent, prev_reward not found in processed_df for requestID: {request_id}")
                    assert False
                
                # Call infer function from scalable_rl_routing_agent
                infer_start = time.time()
                timeout_in_seconds = 5.0  # 5 second timeout for inference
                pod_idx, infer_from_tensor_overhead_summary = infer(request_id, prev_reward, pod_features, kv_hit_ratios, request_features, temporal_features, BROKER, timeout_in_seconds)
                infer_from_tensor_overhead_summary['scalable_rl_infer'] = time.time() - infer_start
                
                # Build result with actual probabilities
                num_pods = len(sorted_all_pod_ids)
                
                ## TODO: we need action probabilities for debugging
                # if action_probs is not None:
                #     # Use actual probabilities from policy
                #     pod_probabilities = {sorted_all_pod_ids[i]: float(action_probs[i]) for i in range(min(num_pods, len(action_probs)))}
                #     confidence = float(action_probs[pod_idx])
                # else:
                #     # Fallback to uniform
                #     pod_probabilities = {sorted_all_pod_ids[i]: 1.0/num_pods for i in range(num_pods)}
                #     confidence = 1.0/num_pods
                
                # TODO: these are placeholder. we need actual probabilities from the model.
                pod_probabilities = {sorted_all_pod_ids[i]: 1.0/num_pods for i in range(num_pods)}
                confidence = 1.0/num_pods
                result = {
                    'selected_pod_index': int(pod_idx),
                    'pod_probabilities': pod_probabilities,
                    'confidence': confidence,
                    'explore_mask': 1,  # RL always explores
                    'predicted_latencies': {pod_id: -1 for pod_id in sorted_all_pod_ids},
                    'chosen_pod_predicted_latency': -1,
                }
                
                logger.info(f"scalable_rl_routing_agent, requestID: {request_id}, action={pod_idx}, prev_reward={prev_reward:.2f}, confidence={confidence:.3f}, num_pods={num_pods}")
            
        ####################################################################################
        ####################################################################################
        # elif subAlgorithm == 'scalable_rl_agent_old':
        #     # === NEW SCALABLE RL AGENT (pod-count independent) ===
        #     logger.info(f"requestID: {request_id}, subAlgorithm: {subAlgorithm}, Using SCALABLE RL agent (pod-independent) for inference")
            
        #     global SCALABLE_RL_AGENT
            
        #     with SCALABLE_RL_AGENT_LOCK.write():
        #         if SCALABLE_RL_AGENT is None:
        #             # Initialize ONCE - works for any number of pods!
        #             pod_features_t = tensor_data['pod_features']
        #             per_pod_dim = int(pod_features_t.shape[2])  # e.g., 10
        #             kv_hit_t = tensor_data['kv_hit_ratios']
        #             kv_dim = int(kv_hit_t.shape[2])  # e.g., 1
        #             req_features_t = tensor_data['request_features']
        #             req_dim = int(req_features_t.shape[1])  # e.g., 3
                    
        #             # Per-pod dimension = pod_features + kv_hit_ratios
        #             total_per_pod_dim = per_pod_dim + kv_dim
                    
        #             SCALABLE_RL_AGENT = create_scalable_rl_agent(
        #                 per_pod_dim=total_per_pod_dim,  # 11 (10 pod + 1 kv)
        #                 request_dim=req_dim,             # 3
        #                 max_pods=100,                    # Max expected pods
        #                 **RL_MODEL_HYPERPARAMETERS
        #             )
                    
        #             # Load checkpoint if available
        #             ckpt_path = RL_MODEL_HYPERPARAMETERS.get('RL_CHECKPOINT_PATH')
        #             if ckpt_path and os.path.exists(ckpt_path):
        #                 try:
        #                     SCALABLE_RL_AGENT.load(ckpt_path)
        #                     logger.info(f"✅ Loaded scalable RL checkpoint from {ckpt_path}")
        #                 except Exception as e:
        #                     logger.warning(f"⚠️  Failed to load RL checkpoint {ckpt_path}: {e}")
                    
        #             logger.info(f"🚀 Initialized SCALABLE RL agent: per_pod_dim={total_per_pod_dim}, "
        #                       f"request_dim={req_dim}, max_pods=100 (works with ANY #pods!)")
                
        #         current_agent = SCALABLE_RL_AGENT
            
        #     # Inference (no lock needed - thread-safe in new design)
        #     current_agent, result, infer_from_tensor_overhead_summary = infer_scalable_rl_agent(
        #         tensor_data=tensor_data,
        #         request_id=request_id,
        #         sorted_all_pod_ids=sorted_all_pod_ids,
        #         processed_df=processed_df,
        #         rl_agent=current_agent,
        #         hyperparameters=RL_MODEL_HYPERPARAMETERS,
        #         agent_lock=None  # New agent doesn't need lock for prediction
        #     )
        ####################################################################################
            else:
                logger.info(f"requestID: {request_id}, contextual bandit model for inference")
                result, infer_from_tensor_overhead_summary = simpler_contextual_bandit.infer_from_tensor(tensor_data, request_id, MODEL_UPDATED, RL_MODEL_HYPERPARAMETERS, final_model_dir)
                result['predicted_latencies'] = {pod_id: -1 for pod_id in sorted_all_pod_ids}
                result['chosen_pod_predicted_latency'] = -1
        handle_infer_overhead_summary["calling_infer_from_tensor"] = time.time() - infer_from_tensor_start_time
        
        
        remaining_work_start = time.time()
        result["requestID"] = request_id
        result["num_trains"] = NUM_TRAINS
        result["request_timestamp"] = time.time() - first_request_starting_time
        logger.info(f"requestID: {request_id}, inference result: {result}")
        
        # Map the pod index back to the actual pod ID
        selected_pod_index = result['selected_pod_index']
        if selected_pod_index >= len(sorted_all_pod_ids):
            logger.error(f"Selected pod index {selected_pod_index} out of range, defaulting to first pod")
            assert False
        selected_pod_generalpodid = sorted_all_pod_ids[selected_pod_index]
        if selected_pod_generalpodid not in RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip']:
            logger.error(f"selected_pod_generalpodid: {selected_pod_generalpodid} not found in RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip']")
            logger.error(f"RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip']: {RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip']}")
            assert False
        selected_pod_ip = RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip'][selected_pod_generalpodid]
            
        logger.debug(f"selected_pod_generalpodid: {selected_pod_generalpodid}, selected_pod_ip: {selected_pod_ip}, pod_probability: {result['pod_probabilities']}")
        
        handle_infer_overhead_summary['remaining_work'] = time.time() - remaining_work_start
        handle_infer_overhead_summary["end_to_end"] = time.time() - handle_infer_start_time
        
        overhead_log = "oh"
        for key, value in handle_infer_overhead_summary.items():
            overhead_log += f", handle_infer_{key}: {value*1000:.0f}ms"
        for key, value in encode_for_inference_overhead_summary.items():
            overhead_log += f", encode_{key}: {value*1000:.0f}ms"
        for key, value in preprocess_dataset_overhead_summary.items():
            overhead_log += f", preprocess_{key}: {value*1000:.0f}ms"
        for key, value in infer_from_tensor_overhead_summary.items():
            overhead_log += f", infer_from_tensor_{key}: {value*1000:.0f}ms"
            
        response = {
            "num_trains": NUM_TRAINS,
            "num_flush": NUM_FLUSH,
            "request_timestamp": time.time() - first_request_starting_time,
            "selected_pod": selected_pod_ip,
            "selected_pod_generalpodid": selected_pod_generalpodid,
            "confidence": result['confidence'],
            "exploration": result['explore_mask'],
            "exploration_enabled": EXPLORATION_ENABLED,
            "request_id": request_id,
            "overhead_log": overhead_log,
            "predicted_latencies": result['predicted_latencies'],
            "chosen_pod_predicted_latency": result['chosen_pod_predicted_latency'],
        }
        return jsonify(response), 200
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error in handle_infer: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 500


def online_train_routine():
    global NUM_TRAINS, MODEL_UPDATED, TOTAL_NUM_DATA, final_model_dir, NUM_NEW_DATA, RL_MODEL_HYPERPARAMETERS, TRAINING_RIGHT_NOW, LATENCY_PREDICTOR, TRAINING_DF, stats_instance
    global MODEL_REGISTRY, MODEL_REGISTRY_LOCK, PREDICTOR_REGISTRY, PREDICTOR_REGISTRY_LOCK, WORKLOAD_TYPES, WORKLOAD
    
    if TRAINING_RIGHT_NOW:
        logger.info(f"Previous training still in progress, skipping training")
        return
    if NUM_NEW_DATA < MIN_NUM_TRAINING_DATA:
        logger.info(f"Not enough training data available, NUM_NEW_DATA: {NUM_NEW_DATA} < {MIN_NUM_TRAINING_DATA}, wait until enough data are added. NUM_TRAINS: {NUM_TRAINS}, TOTAL_NUM_DATA: {TOTAL_NUM_DATA}")
        return
    TRAINING_RIGHT_NOW = True
    training_start_time = time.time()
    logger.info(f"online_train_routine start, {NUM_TRAINS}th online training with {NUM_NEW_DATA} new training data")
    try:
        # Route to appropriate training function based on model type
        model_type = RL_MODEL_HYPERPARAMETERS['MODEL_TYPE']
        if model_type == 'latency_predictor':
            logger.info(f"Training with latency predictor model on entire dataset (offline + online)")

            # Get a copy of TRAINING_DF for training (thread-safe)
            with TRAINING_DF_LOCK:
                if TRAINING_DF is None or len(TRAINING_DF) == 0:
                    logger.error("TRAINING_DF is empty, cannot train")
                    TRAINING_RIGHT_NOW = False
                    return
                training_df_copy = TRAINING_DF.copy()
                total_samples = len(training_df_copy)

            logger.info(f"Training on total data: {total_samples}")

            # Drop non-numeric metadata columns from offline CSV that are absent online
            metadata_cols_to_drop = ['source_file', 'reward_function_used']
            cols_present_to_drop = [c for c in metadata_cols_to_drop if c in training_df_copy.columns]
            if cols_present_to_drop:
                logger.info(f"Dropping metadata columns not used for training: {cols_present_to_drop}")
                training_df_copy = training_df_copy.drop(columns=cols_present_to_drop)

            ############################################################################
            # Handle missing values proactively to avoid intermittent failures due to dynamic pod columns
            try:
                import numpy as np
                numeric_columns = training_df_copy.select_dtypes(include=[np.number]).columns
                total_missing_numeric = int(training_df_copy[numeric_columns].isna().sum().sum()) if len(numeric_columns) > 0 else 0
                if total_missing_numeric > 0:
                    logger.warning(f"Filling {total_missing_numeric} missing numeric values with 0 for training consistency")
                    training_df_copy[numeric_columns] = training_df_copy[numeric_columns].fillna(0)
                # Optional: warn about numeric columns with high missing rates (diagnostics only)
                missing_pct = training_df_copy[numeric_columns].isnull().mean() * 100 if len(numeric_columns) > 0 else pd.Series()
                high_missing = missing_pct[missing_pct > 5].sort_values(ascending=False)
                if len(high_missing) > 0:
                    topk = list(high_missing.head(10).items())
                    logger.error(f"High-missing columns (>5%) detected (top 10): {[(k, round(v,1)) for k,v in topk]}")
            except Exception as e:
                logger.warning(f"NaN handling diagnostics failed: {e}")
            ############################################################################
            
            # Normalize the entire dataset
            normalizable_features, non_normalizable_features = data_normalizer._get_normalizable_features(
                training_df_copy, RL_MODEL_HYPERPARAMETERS.get('NO_NORMALIZE_FEATURES', []))

            for feature in normalizable_features:
                data_normalizer._normalize_single_feature(training_df_copy, feature, stats_instance, is_training=False)

            # Get sorted pod IDs and GPU mapping
            sorted_all_pod_ids = utils.get_sorted_all_pod_ids('processed_csv_columns', training_df_copy.columns.tolist())
            generalpodid_to_gpu_model = RL_MODEL_HYPERPARAMETERS.get('generalpodid_to_gpu_model', {})
            logger.info(f"Training with pods: {sorted_all_pod_ids}")

            # ========================================
            # Multi-Model Training: Train per (workload, GPU) combination
            # ========================================
            
            # Check if multi-model training is enabled
            use_multi_model_training = False
            with MODEL_REGISTRY_LOCK.read():
                use_multi_model_training = len(MODEL_REGISTRY) > 0
            
            if use_multi_model_training:
                logger.info(f"{GREEN_COLOR}Multi-model training enabled: training models per GPU type for workload={WORKLOAD}{RESET_COLOR}")
                
                # NOTE: Currently trains models only for the configured WORKLOAD (from env var)
                # TODO: Future enhancement - extract workload from training data and split by it
                # This would enable training models for multiple workloads simultaneously
                if len(WORKLOAD_TYPES) > 1:
                    logger.warning(f"Multiple workload types configured ({WORKLOAD_TYPES}), but training only for WORKLOAD={WORKLOAD}. "
                                 f"Multi-workload training not yet implemented.")
                
                # Get unique GPU types from pods
                unique_gpu_types = set()
                for pod_id in sorted_all_pod_ids:
                    if pod_id in generalpodid_to_gpu_model:
                        unique_gpu_types.add(generalpodid_to_gpu_model[pod_id])
                
                logger.info(f"Training models for GPU types: {unique_gpu_types}")
                
                # For each GPU type, filter data and train model
                for gpu_type in unique_gpu_types:
                    # Get pods with this GPU type
                    gpu_pod_ids = [pid for pid in sorted_all_pod_ids 
                                   if pid in generalpodid_to_gpu_model and 
                                   generalpodid_to_gpu_model[pid] == gpu_type]
                    
                    if len(gpu_pod_ids) == 0:
                        logger.warning(f"No pods found for GPU type {gpu_type}, skipping")
                        continue
                    
                    logger.info(f"Training model for (workload={WORKLOAD}, gpu={gpu_type}) with {len(gpu_pod_ids)} pods")
                    
                    # Create output directory for this (workload, GPU) combination
                    model_output_dir = os.path.join(final_model_dir, f"{WORKLOAD}_{gpu_type}")
                    os.makedirs(model_output_dir, exist_ok=True)
                    
                    # Encode data for this GPU type
                    encode_start_time = time.time()
                    encoded_training_dir = os.path.join(ENCODED_DATA_DIR, f"{WORKLOAD}_{gpu_type}")
                    encoding.encode_for_train(gpu_pod_ids, training_df_copy, encoded_training_dir, request_features_train, RL_MODEL_HYPERPARAMETERS)
                    logger.info(f"Encoded {total_samples} samples for {gpu_type} to {encoded_training_dir}, took {time.time() - encode_start_time:.2f}s")
                    
                    # Train model for this GPU type
                    train_start_time = time.time()
                    latency_predictor.train_latency_predictor(encoded_training_dir, model_output_dir, RL_MODEL_HYPERPARAMETERS)
                    logger.info(f"Trained model for (workload={WORKLOAD}, gpu={gpu_type}), took {time.time() - train_start_time:.2f}s")
                
                # Clear predictor registry to force reload on next inference
                logger.info(f"{GREEN_COLOR}Clearing predictor registry to reload updated models{RESET_COLOR}")
                with PREDICTOR_REGISTRY_LOCK.write():
                    PREDICTOR_REGISTRY.clear()
                
            else:
                # ========================================
                # Single-Model Training (original behavior)
                # ========================================
                logger.info("Single-model training (multi-model disabled or single workload)")
                
                # Encode the entire dataset
                encode_start_time = time.time()
                os.makedirs(ENCODED_DATA_DIR, exist_ok=True)
                encoded_training_dir = os.path.join(ENCODED_DATA_DIR, "full_training_data")
                encoding.encode_for_train(sorted_all_pod_ids, training_df_copy, encoded_training_dir, request_features_train, RL_MODEL_HYPERPARAMETERS)
                logger.info(f"Encoded {total_samples} samples to {encoded_training_dir}, encode time: {time.time() - encode_start_time} seconds")

                # Train on the encoded dataset
                train_start_time = time.time()
                latency_predictor.train_latency_predictor(encoded_training_dir, final_model_dir, RL_MODEL_HYPERPARAMETERS)
                logger.info(f"train_latency_predictor done, train time: {time.time() - train_start_time} seconds")

                # Reload global LATENCY_PREDICTOR (for single-model fallback path)
                with LATENCY_PREDICTOR_LOCK.write():
                    if LATENCY_PREDICTOR is not None:
                        load_start_time = time.time()
                        LATENCY_PREDICTOR.load(final_model_dir)
                        logger.info(f"Reloaded latency predictor after training, load time: {time.time() - load_start_time} seconds")
        else:
            logger.info(f"Training with contextual bandit model")
            simpler_contextual_bandit.train(ENCODED_DATA_DIR, final_model_dir, RL_MODEL_HYPERPARAMETERS, ENABLE_ONLINE_LEARNING)
            logger.info(f"train_contextual_bandit done")
    except Exception as e:
        import traceback
        logger.error(f"Error during training: {e}")
        logger.error(traceback.format_exc())
        TRAINING_RIGHT_NOW = False
        return
    logger.info(f"Successfully completed online_train_routine done, {NUM_TRAINS}th online training with {NUM_NEW_DATA} new training data, took {time.time() - training_start_time} seconds")
    MODEL_UPDATED = True
    TRAINING_RIGHT_NOW = False
    NUM_TRAINS += 1
    NUM_NEW_DATA = 0


def test_kubernetes_permissions():
    """Test if we have the required Kubernetes permissions"""
    try:
        config.load_incluster_config()
        
        v1 = client.CoreV1Api()
        
        # Test 1: Can we list pods?
        try:
            pods = v1.list_pod_for_all_namespaces(label_selector=POD_LABEL_SELECTOR, limit=1)
            logger.info("✅ Successfully listed pods - pod permissions OK")
        except Exception as e:
            logger.error(f"❌ Cannot list pods: {e}")
            return False
            
        # Test 2: Can we read nodes?
        try:
            nodes = v1.list_node(limit=1)
            logger.info("✅ Successfully listed nodes - node permissions OK")
        except Exception as e:
            logger.error(f"❌ Cannot list nodes: {e}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Kubernetes API access failed: {e}")
        return False

def rl_update_worker():
    """Background worker thread for RL agent updates"""
    global RL_AGENT, RL_AGENT_LOCK
    logger.info("RL update worker thread started")
    
    while not RL_UPDATE_SHUTDOWN.is_set():
        try:
            # Wait for update request with timeout
            update_request = RL_UPDATE_QUEUE.get(timeout=1.0)
            
            if update_request is None:  # Shutdown signal
                break
                
            # Perform the update with WRITE lock (exclusive access)
            # This blocks concurrent predictions to prevent PyTorch read+write races
            with RL_AGENT_LOCK.write():
                if RL_AGENT is not None:
                    try:
                        n_steps = update_request.get('n_steps', 32)
                        RL_AGENT.update_online(n_steps=n_steps)
                        logger.debug(f"RL agent updated with {n_steps} steps")
                    except Exception as e:
                        logger.error(f"Error updating RL agent: {e}")
            
            RL_UPDATE_QUEUE.task_done()
            
        except queue.Empty:
            continue  # Timeout, check shutdown flag
        except Exception as e:
            logger.error(f"Error in RL update worker: {e}")
    
    logger.info("RL update worker thread stopped")


def start_rl_update_worker():
    """Start the RL update worker thread"""
    global RL_UPDATE_THREAD
    if RL_UPDATE_THREAD is None or not RL_UPDATE_THREAD.is_alive():
        RL_UPDATE_THREAD = threading.Thread(target=rl_update_worker, daemon=True)
        RL_UPDATE_THREAD.start()
        logger.info("Started RL update worker thread")


def stop_rl_update_worker():
    """Stop the RL update worker thread"""
    global RL_UPDATE_THREAD
    if RL_UPDATE_THREAD and RL_UPDATE_THREAD.is_alive():
        logger.info("Stopping RL update worker thread...")
        RL_UPDATE_SHUTDOWN.set()
        
        # Send shutdown signal
        try:
            RL_UPDATE_QUEUE.put(None, timeout=1.0)
        except queue.Full:
            pass
        
        # Wait for thread to finish
        RL_UPDATE_THREAD.join(timeout=5.0)
        if RL_UPDATE_THREAD.is_alive():
            logger.warning("RL update worker thread did not stop gracefully")
        else:
            logger.info("RL update worker thread stopped successfully")


def scalable_rl_training_worker():
    """
    Background worker thread for scalable RL agent training.
    
    This continuously runs the RL training loop, which:
    1. Pulls requests from the environment (blocks until request available from BROKER)
    2. Predicts action (pod selection)
    3. Routes request (sets decision in BROKER, unblocking /infer)
    4. Collects experience and updates policy
    """
    global SCALABLE_RL_AGENT, SCALABLE_RL_TRAINING_SHUTDOWN
    logger.info("🏋️  Scalable RL training worker thread started")
    
    try:
        # Training loop - runs until shutdown
        while not SCALABLE_RL_TRAINING_SHUTDOWN.is_set():
            if SCALABLE_RL_AGENT is not None:
                try:
                    # Run training for a batch of steps
                    # The agent's learn() will internally call env.step() which pulls from BROKER
                    logger.info(f"{PURPLE_COLOR}Training scalable RL agent...{RESET_COLOR}")
                    # Create proper checkpoint file path (not just directory)
                    checkpoint_file = os.path.join(RL_MODEL_HYPERPARAMETERS['CHECKPOINT_DIR'], 'scalable_rl_agent')
                    
                    eval_freq = (RL_MODEL_HYPERPARAMETERS['num_requests_per_episode'] + 1) * RL_MODEL_HYPERPARAMETERS['num_episodes_per_iteration'] # how often to evaluate the model, the eval will be triggered every eval_freq steps. Hence, eval_freq should be the number of requests per iteration
                    SCALABLE_RL_AGENT.train(
                        save_path=checkpoint_file,
                        eval_freq=eval_freq,
                        n_eval_episodes=RL_MODEL_HYPERPARAMETERS['n_eval_episodes'],
                        )
                except Exception as e:
                    if not SCALABLE_RL_TRAINING_SHUTDOWN.is_set():
                        logger.error(f"Error in scalable RL training loop: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        time.sleep(1)  # Avoid tight error loop
            else:
                time.sleep(1)  # Wait for agent initialization
    except Exception as e:
        logger.error(f"Fatal error in scalable RL training worker: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("Scalable RL training worker thread stopped")


def start_scalable_rl_training_worker():
    """Start the scalable RL training worker thread"""
    global SCALABLE_RL_TRAINING_THREAD
    if SCALABLE_RL_TRAINING_THREAD is None or not SCALABLE_RL_TRAINING_THREAD.is_alive():
        SCALABLE_RL_TRAINING_THREAD = threading.Thread(
            target=scalable_rl_training_worker,
            daemon=True,
            name="ScalableRLTraining"
        )
        SCALABLE_RL_TRAINING_THREAD.start()
        logger.info("✅ Started scalable RL training worker thread")


def stop_scalable_rl_training_worker():
    """Stop the scalable RL training worker thread"""
    global SCALABLE_RL_TRAINING_THREAD
    if SCALABLE_RL_TRAINING_THREAD and SCALABLE_RL_TRAINING_THREAD.is_alive():
        logger.info("Stopping scalable RL training worker thread...")
        SCALABLE_RL_TRAINING_SHUTDOWN.set()
        
        # Wait for thread to finish
        SCALABLE_RL_TRAINING_THREAD.join(timeout=5.0)
        if SCALABLE_RL_TRAINING_THREAD.is_alive():
            logger.warning("Scalable RL training worker thread did not stop gracefully")
        else:
            logger.info("Scalable RL training worker thread stopped successfully")


def queue_rl_update(n_steps=32):
    """Queue an RL agent update request (non-blocking)"""
    if ENABLE_ONLINE_LEARNING and RL_AGENT is not None:
        try:
            update_request = {'n_steps': n_steps}
            RL_UPDATE_QUEUE.put_nowait(update_request)
            logger.debug(f"Queued RL update request with {n_steps} steps")
        except queue.Full:
            logger.warning("RL update queue is full, skipping update request")


def get_current_cluster_features():
    """
    Fetch real-time cluster state for experience completion (next_obs).
    
    This mirrors the feature extraction done in /infer, but fetches CURRENT state.
    
    Returns:
        pod_features: [num_pods, 10] - Current pod metrics
        kv_hit_ratios: [num_pods, 1] - Current cache hit ratios (zeros for now)
        request_features: [3] - Dummy request features (not used for next_obs)
    """
    global RL_MODEL_HYPERPARAMETERS, stats_instance
    
    try:
        # Get current running pods (same as init())
        running_pods = utils.get_running_pods_by_label(POD_LABEL_SELECTOR)
        sorted_pod_ips = utils.fetch_running_pod_ips(running_pods)
        num_pods = len(sorted_pod_ips)
        
        if num_pods == 0:
            logger.warning("No running pods found for cluster state fetch")
            # Return dummy state
            return (
                np.zeros((1, 10), dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32),
                np.zeros(3, dtype=np.float32)
            )
        
        # Initialize arrays for pod features
        pod_features = np.zeros((num_pods, 10), dtype=np.float32)
        
        # Get pod_ip to general_pod_id mapping
        pod_ip_to_generalpodid = RL_MODEL_HYPERPARAMETERS.get('pod_ip_to_generalpodid', {})
        pod_ip_to_gpu_model_encoded = RL_MODEL_HYPERPARAMETERS.get('pod_ip_to_gpu_model_encoded', {})
        
        # For each pod, fetch current metrics
        for i, pod_ip in enumerate(sorted_pod_ips):
            try:
                # These are the same features used in preprocess.py
                # The exact column order depends on your feature extraction
                # Adjust indices based on your actual feature schema
                
                # Example feature extraction (adjust to your schema):
                # Column 0: running_requests (from inflight)
                inflight_requests = utils.GetInflightRequestsForPod(pod_ip) if hasattr(utils, 'GetInflightRequestsForPod') else 0
                pod_features[i, 0] = float(inflight_requests)
                
                # Column 1: queue_length (from waiting requests)
                # Use stored metrics or default to 0
                pod_features[i, 1] = 0  # Placeholder - implement if you track queue length
                
                # Column 2-3: GPU/CPU KV cache usage
                # These would come from vLLM metrics if available
                pod_features[i, 2] = 0  # GPU cache usage (implement if tracked)
                pod_features[i, 3] = 0  # CPU cache usage (implement if tracked)
                
                # Column 4-5: Num requests running/waiting
                pod_features[i, 4] = 0  # Running requests (implement if tracked)
                pod_features[i, 5] = 0  # Waiting requests (implement if tracked)
                
                # Column 6-7: Prefill/decode token counts
                prefill_tokens = utils.GetNumPrefillTokensForPod(pod_ip) if hasattr(utils, 'GetNumPrefillTokensForPod') else 0
                decode_tokens = utils.GetNumDecodeTokensForPod(pod_ip) if hasattr(utils, 'GetNumDecodeTokensForPod') else 0
                pod_features[i, 6] = float(prefill_tokens)
                pod_features[i, 7] = float(decode_tokens)
                
                # Column 8: GPU type (encoded)
                if pod_ip in pod_ip_to_gpu_model_encoded:
                    pod_features[i, 8] = float(pod_ip_to_gpu_model_encoded[pod_ip])
                else:
                    pod_features[i, 8] = 0.0
                
                # Column 9: Availability (1.0 = available)
                pod_features[i, 9] = 1.0  # Assume available (or check pod status)
                
            except Exception as e:
                logger.warning(f"Error fetching metrics for pod {pod_ip}: {e}")
                # Keep zeros for this pod
        
        # Normalize features using stats_instance (same as /infer)
        if stats_instance is not None and stats_instance.get_max_count() > 0:
            # Get normalizable feature names (adjust to your schema)
            # This should match the features in your processed_df
            feature_names = [
                'running_requests', 'queue_length', 'gpu_cache', 'cpu_cache',
                'num_running', 'num_waiting', 'prefill_tokens', 'decode_tokens',
                'gpu_type', 'availability'
            ]
            
            for col_idx, feature_name in enumerate(feature_names):
                if feature_name in stats_instance.feature_stats:
                    stats = stats_instance.feature_stats[feature_name]
                    # Z-score normalization: (x - mean) / std
                    if stats.std > 0:
                        pod_features[:, col_idx] = (pod_features[:, col_idx] - stats.mean) / stats.std
        
        # KV hit ratios - for next_obs, we don't have per-request cache hit info
        # So we use zeros (or could use average cache hit rates if tracked)
        kv_hit_ratios = np.zeros((num_pods, 1), dtype=np.float32)
        
        # Request features - dummy (not used for next_obs, but needed for consistency)
        request_features = np.zeros(3, dtype=np.float32)
        
        logger.debug(f"Fetched current cluster state: {num_pods} pods")
        return pod_features, kv_hit_ratios, request_features
        
    except Exception as e:
        logger.error(f"Error in get_current_cluster_features: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Return minimal valid state
        return (
            np.zeros((1, 10), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros(3, dtype=np.float32)
        )


def load_model_for_workload_gpu(workload, gpu_type, model_type, model_base_dir, hyperparameters):
    """
    Load a routing model for a specific (workload, GPU_type) combination.
    
    Args:
        workload: Workload identifier (e.g., 'A', 'B', 'C')
        gpu_type: GPU model type (e.g., 'NVIDIA-L20', 'NVIDIA-A10')
        model_type: Type of model ('contextual_bandit', 'latency_predictor', etc.)
        model_base_dir: Base directory containing model subdirectories
        hyperparameters: Model hyperparameters dict
        
    Returns:
        Loaded model instance or None if loading fails
    """
    try:
        # Construct model directory path
        model_dir = os.path.join(model_base_dir, f"{workload}_{gpu_type}")
        
        if not os.path.exists(model_dir):
            logger.warning(f"Model directory not found: {model_dir}, skipping")
            return None
            
        logger.info(f"Loading model for workload={workload}, gpu_type={gpu_type} from {model_dir}")
        
        if model_type == 'latency_predictor':
            # Load latency predictor model
            model_path = os.path.join(model_dir, 'latency_predictor.pth')
            if not os.path.exists(model_path):
                logger.warning(f"Latency predictor not found at {model_path}")
                return None
                
            # We'll initialize with dummy dims - will be properly initialized on first inference
            # Store just the model path for now, actual initialization happens in handle_infer
            return {'type': 'latency_predictor', 'model_dir': model_dir, 'model_path': model_path}
            
        elif model_type == 'contextual_bandit':
            # Load contextual bandit model
            model_path = os.path.join(model_dir, 'policy_network.pth')
            if not os.path.exists(model_path):
                logger.warning(f"Contextual bandit model not found at {model_path}")
                return None
                
            # Store model directory for lazy loading during inference
            return {'type': 'contextual_bandit', 'model_dir': model_dir, 'model_path': model_path}
            
        else:
            logger.warning(f"Unknown model type: {model_type}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to load model for workload={workload}, gpu_type={gpu_type}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def graceful_shutdown(sig=None, frame=None):
    """Handle graceful shutdown when receiving SIGTERM or SIGINT"""
    logger.info(f"Received signal {sig if sig else 'shutdown'}, shutting down gracefully...")
    
    # Stop RL update worker
    stop_rl_update_worker()
    
    # Stop scalable RL training worker
    stop_scalable_rl_training_worker()
    
    # Shutdown the scheduler if it exists
    if 'scheduler' in globals() and scheduler:
        try:
            scheduler.shutdown(wait=False)
            logger.info("Background scheduler shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down scheduler: {e}")
    
    # Any other cleanup you need can go here
    logger.info("Graceful shutdown completed")
    sys.exit(0)


def init():
    global RL_MODEL_HYPERPARAMETERS, stats_instance, WORKLOAD, WORKLOAD_TYPES
    if RL_MODEL_HYPERPARAMETERS is None:

        logger.info(f"{GREEN_COLOR}RL_MODEL_HYPERPARAMETERS is None{RESET_COLOR}")
        
        # Validate workload configuration
        if WORKLOAD not in WORKLOAD_TYPES:
            logger.error(f"Configuration error: WORKLOAD='{WORKLOAD}' is not in WORKLOAD_TYPES={WORKLOAD_TYPES}")
            logger.error(f"Training will create models for '{WORKLOAD}', but inference will try to load models for {WORKLOAD_TYPES}")
            logger.error(f"Please ensure WORKLOAD is included in WORKLOAD_TYPES")
            assert False, f"WORKLOAD '{WORKLOAD}' must be in WORKLOAD_TYPES {WORKLOAD_TYPES}"

        RL_MODEL_HYPERPARAMETERS = {}
        RL_MODEL_HYPERPARAMETERS['TTFT_REWARD_WEIGHT'] = TTFT_REWARD_WEIGHT
        RL_MODEL_HYPERPARAMETERS['EXPLORATION_ENABLED'] = EXPLORATION_ENABLED
        RL_MODEL_HYPERPARAMETERS['ENABLE_ONLINE_LEARNING'] = ENABLE_ONLINE_LEARNING
        logger.info("Loading RL hyperparameters from model_config.json")
        utils.load_rl_hyperparameters(hyperparameter_file_path, RL_MODEL_HYPERPARAMETERS)
        model_type = RL_MODEL_HYPERPARAMETERS.get('MODEL_TYPE', 'contextual_bandit')
        logger.info(f"Model type configured: {model_type}")
        if model_type == 'latency_predictor':
            latency_metric = RL_MODEL_HYPERPARAMETERS.get('LATENCY_METRIC', 'ttft')
            logger.info(f"Latency metric for prediction: {latency_metric}")
        else:
            logger.info(f"Using contextual bandit with exploration rate: {RL_MODEL_HYPERPARAMETERS.get('exploration_rate', 0)}")
        # Test permissions first
        logger.info("Testing Kubernetes API permissions...")
        if not test_kubernetes_permissions():
            logger.error("Insufficient Kubernetes permissions - using fallback GPU mapping")
            assert False
        running_vllm_pods = utils.get_running_pods_by_label(POD_LABEL_SELECTOR)
        sorted_running_pod_ips = utils.fetch_running_pod_ips(running_vllm_pods)
        pod_ip_to_generalpodid = utils.create_pod_ip_to_generalpodid_mapping(sorted_running_pod_ips)
        generalpodid_to_pod_ip = {}
        for pod_ip, generalpodid in pod_ip_to_generalpodid.items():
            generalpodid_to_pod_ip[generalpodid] = pod_ip
        generalpodid_to_gpu_model = utils.fetch_generalpodid_to_gpu_model(running_vllm_pods, pod_ip_to_generalpodid)
        pod_ip_to_gpu_model, pod_ip_to_gpu_model_encoded = utils.create_pod_ip_to_gpu_model_mapping(generalpodid_to_gpu_model, pod_ip_to_generalpodid)
        
        logger.info(f"POD_LABEL_SELECTOR: {POD_LABEL_SELECTOR}")
        logger.info(f"len(sorted_running_pod_ips): {len(sorted_running_pod_ips)}, sorted_running_pod_ips: {sorted_running_pod_ips}")
        logger.info(f"pod_ip_to_generalpodid: {pod_ip_to_generalpodid}")
        logger.info(f"generalpodid_to_gpu_model: {generalpodid_to_gpu_model}")
        logger.info(f"pod_ip_to_gpu_model: {pod_ip_to_gpu_model}")
        logger.info(f"pod_ip_to_gpu_model_encoded: {pod_ip_to_gpu_model_encoded}")

        RL_MODEL_HYPERPARAMETERS['pod_ip_to_generalpodid'] = pod_ip_to_generalpodid
        RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip'] = generalpodid_to_pod_ip
        logger.info(f"RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip']: {RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip']}")
        RL_MODEL_HYPERPARAMETERS['sorted_running_pod_ips'] = sorted_running_pod_ips
        RL_MODEL_HYPERPARAMETERS['pod_ip_to_gpu_model'] = pod_ip_to_gpu_model
        RL_MODEL_HYPERPARAMETERS['pod_ip_to_gpu_model_encoded'] = pod_ip_to_gpu_model_encoded
        RL_MODEL_HYPERPARAMETERS['generalpodid_to_gpu_model'] = generalpodid_to_gpu_model
        # Additional mappings for GPU features expected by preprocess/encoding
        RL_MODEL_HYPERPARAMETERS['pod_gpu_mapping'] = generalpodid_to_gpu_model
        GPU_MODEL_TO_ENCODE = {
            'NVIDIA-L20': 0,
            'NVIDIA-L40': 1,
            'NVIDIA-A10': 2,
            'NVIDIA-A100': 3,
            'NVIDIA-H100': 4,
        }
        pod_gpu_id_mapping = {}
        for generalpodid, gpu_model in generalpodid_to_gpu_model.items():
            if gpu_model in GPU_MODEL_TO_ENCODE:
                pod_gpu_id_mapping[generalpodid] = GPU_MODEL_TO_ENCODE[gpu_model]
            else:
                logger.error(f"Unknown GPU model for {generalpodid}: {gpu_model}")
                assert False
        RL_MODEL_HYPERPARAMETERS['pod_gpu_id_mapping'] = pod_gpu_id_mapping
        
        # ========================================
        # Parallel model loading for multi-workload, multi-GPU support
        # ========================================
        
        # Get unique GPU types from running pods
        unique_gpu_types = list(set(generalpodid_to_gpu_model.values()))
        logger.info(f"{GREEN_COLOR}Starting parallel model loading for workloads={WORKLOAD_TYPES}, gpu_types={unique_gpu_types}{RESET_COLOR}")
        
        # Import ThreadPoolExecutor for parallel loading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Parallel model loading
        model_loading_start = time.time()
        with MODEL_REGISTRY_LOCK.write():
            with ThreadPoolExecutor(max_workers=len(WORKLOAD_TYPES) * len(unique_gpu_types)) as executor:
                # Submit loading tasks for all (workload, gpu_type) combinations
                future_to_key = {}
                for workload in WORKLOAD_TYPES:
                    for gpu_type in unique_gpu_types:
                        future = executor.submit(
                            load_model_for_workload_gpu,
                            workload,
                            gpu_type,
                            model_type,
                            final_model_dir,
                            RL_MODEL_HYPERPARAMETERS
                        )
                        future_to_key[future] = (workload, gpu_type)
                
                # Collect results as they complete
                for future in as_completed(future_to_key):
                    workload, gpu_type = future_to_key[future]
                    try:
                        model_info = future.result()
                        if model_info is not None:
                            MODEL_REGISTRY[(workload, gpu_type)] = model_info
                            logger.info(f"{GREEN_COLOR}✓ Loaded model for (workload={workload}, gpu={gpu_type}){RESET_COLOR}")
                        else:
                            logger.warning(f"{RED_COLOR}✗ Failed to load model for (workload={workload}, gpu={gpu_type}){RESET_COLOR}")
                    except Exception as e:
                        logger.error(f"Error loading model for (workload={workload}, gpu={gpu_type}): {e}")
        
        model_loading_time = time.time() - model_loading_start
        logger.info(f"{GREEN_COLOR}Parallel model loading completed in {model_loading_time:.2f}s, loaded {len(MODEL_REGISTRY)} models{RESET_COLOR}")
        
        if len(MODEL_REGISTRY) == 0:
            logger.warning(f"{RED_COLOR}No models loaded! Falling back to single-model mode{RESET_COLOR}")
        else:
            logger.info(f"Model registry keys: {list(MODEL_REGISTRY.keys())}")
        
        # Load normalization statistics from CSV file
        if os.path.exists(feature_normalization_stats_file):
            logger.info(f"Loading normalization statistics from: {feature_normalization_stats_file}")
            try:
                stats_instance = data_normalizer.FeatureStats.load_from_csv(feature_normalization_stats_file)
                if stats_instance is not None:
                    logger.info(f"Successfully loaded stats for {len(stats_instance.feature_stats)} features")
                else:
                    logger.error("Failed to load normalization statistics")
                    assert False
            except Exception as e:
                logger.error(f"Failed to load normalization statistics: {e}")
                assert False
        else:
            logger.error(f"Normalization statistics file not found: {feature_normalization_stats_file}")
            assert False
    
    # Print feature statistics if available
    if stats_instance is not None:
        logger.info("Per-feature statistics loaded:")
        for feature_name, stats in stats_instance.feature_stats.items():
            logger.info(f"stats_instance, {feature_name}: count={stats.count}, mean={stats.mean}, std={stats.std}")
    else:
        logger.warning("No normalization statistics available - inference will fail")
        assert False
    
    # Add checkpointing configuration to hyperparameters
    RL_MODEL_HYPERPARAMETERS['CHECKPOINT_INTERVAL_STEPS'] = 100
    RL_MODEL_HYPERPARAMETERS['CHECKPOINT_DIR'] = os.path.join(final_model_dir, 'checkpoints')
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(RL_MODEL_HYPERPARAMETERS['CHECKPOINT_DIR'], exist_ok=True)
    logger.info(f"Checkpoint directory: {RL_MODEL_HYPERPARAMETERS['CHECKPOINT_DIR']}")
    logger.info(f"Checkpointing every {RL_MODEL_HYPERPARAMETERS['CHECKPOINT_INTERVAL_STEPS']} steps")

    # Load offline training data for online learning
    global TRAINING_DF
    if ENABLE_ONLINE_LEARNING:
        offline_csv_path = "/app/offline_training_data.csv"
        if os.path.exists(offline_csv_path):
            try:
                with TRAINING_DF_LOCK:
                    TRAINING_DF = pd.read_csv(offline_csv_path)
                    logger.info(f"✅ Loaded offline training data: {len(TRAINING_DF)} samples from {offline_csv_path}")
                    logger.info(f"   Columns: {list(TRAINING_DF.columns[:10])}...")  # Show first 10 columns
            except Exception as e:
                logger.error(f"Failed to load offline training data: {e}")
                TRAINING_DF = pd.DataFrame()
                logger.warning("Starting with empty training dataframe")
        else:
            logger.warning(f"Offline training data not found at {offline_csv_path}")
            logger.warning("Online learning will start from scratch with only new data")
            TRAINING_DF = pd.DataFrame()
    else:
        logger.info("Online learning disabled, skipping offline data load")
    
    # Initialize scalable RL agent if configured

    logger.info(f"{BLUE_COLOR}model_type: {RL_MODEL_HYPERPARAMETERS['MODEL_TYPE']}{RESET_COLOR}")


    # if RL_MODEL_HYPERPARAMETERS['MODEL_TYPE'] == 'scalable_rl_agent':
    if RL_MODEL_HYPERPARAMETERS['MODEL_TYPE'] == 'scalable_rl_agent':
        import scalable_rl_routing_agent
        from scalable_rl_routing_agent import BROKER
        
        logger.info("Initializing scalable_rl_agent, scalable RL agent...")
        global SCALABLE_RL_AGENT, BROKER_LOCK
        with BROKER_LOCK.write():
            # Get number of pods from current running pods
            num_pods = len(sorted_running_pod_ips)
            logger.info(f"Creating scalable RL agent with {num_pods} pods")
            
            # # Create agent with hyperparameters
            # SCALABLE_RL_AGENT = create_scalable_rl_agent(
            #     per_pod_dim=RL_MODEL_HYPERPARAMETERS.get('per_pod_dim', 8),
            #     request_dim=RL_MODEL_HYPERPARAMETERS.get('request_dim', 3),
            #     max_pods=RL_MODEL_HYPERPARAMETERS.get('max_pods', 100),
            #     learning_rate=RL_MODEL_HYPERPARAMETERS.get('learning_rate', 3e-4),
            #     reward_decay_factor=RL_MODEL_HYPERPARAMETERS.get('reward_decay_factor', 1.0),
            #     gae_lambda=RL_MODEL_HYPERPARAMETERS.get('gae_lambda', 0.95),
            #     n_steps=RL_MODEL_HYPERPARAMETERS.get('n_steps', 256),
            #     horizon=RL_MODEL_HYPERPARAMETERS.get('horizon', 1024),
            #     batch_size=RL_MODEL_HYPERPARAMETERS.get('batch_size', 64),
            #     last_layer_dim_vf=RL_MODEL_HYPERPARAMETERS.get('last_layer_dim_vf', 1),
            #     rl=RL_MODEL_HYPERPARAMETERS.get('rl_algorithm', 'PPO'),
            #     static_num_pods=True,
            # )
            
            SCALABLE_RL_AGENT = scalable_rl_routing_agent.ScalableRLRoutingAgent(
                per_pod_dim=8, 
                # per_pod_dim=11, 
                request_dim=3, 
                max_pods=100, 
                num_requests_per_episode=RL_MODEL_HYPERPARAMETERS['num_requests_per_episode'], 
                num_episodes_per_iteration=RL_MODEL_HYPERPARAMETERS['num_episodes_per_iteration'],
                num_iterations=RL_MODEL_HYPERPARAMETERS['num_iterations'],   
                rl=RL_MODEL_HYPERPARAMETERS['rl_algorithm'], 
                static_num_pods=True, 
                learning_rate=RL_MODEL_HYPERPARAMETERS['rl_learning_rate'], 
                hidden_dim=RL_MODEL_HYPERPARAMETERS['hidden_dim'], 
                gamma=RL_MODEL_HYPERPARAMETERS['gamma'], 
                gae_lambda=RL_MODEL_HYPERPARAMETERS['gae_lambda'], 
                tb_log_dir=os.path.join(RL_MODEL_HYPERPARAMETERS['CHECKPOINT_DIR'], 'tb_logs'), 
                batch_size=RL_MODEL_HYPERPARAMETERS['batch_size'], 
                n_epochs=RL_MODEL_HYPERPARAMETERS['training_epochs'], 
                clip_range=RL_MODEL_HYPERPARAMETERS['clip_range'], 
                entropy_coeff=RL_MODEL_HYPERPARAMETERS['entropy_coeff'], 
                vf_coef=RL_MODEL_HYPERPARAMETERS['vf_coef'], 
                max_grad_norm=RL_MODEL_HYPERPARAMETERS['max_grad_norm'], 
                last_layer_dim_pi=RL_MODEL_HYPERPARAMETERS['last_layer_dim_pi'], 
                last_layer_dim_vf=RL_MODEL_HYPERPARAMETERS['last_layer_dim_vf'], 
                use_prioritized_replay=False, 
                buffer_size=RL_MODEL_HYPERPARAMETERS['buffer_size'], 
                priority_alpha=RL_MODEL_HYPERPARAMETERS['priority_alpha'], 
                priority_beta=RL_MODEL_HYPERPARAMETERS['priority_beta'],
                lr_scheduler_type=RL_MODEL_HYPERPARAMETERS['lr_scheduler_type'],
                load_tb_best='/app/final_model/init_model/best_model.zip',
                )
            
            logger.info(f"scalable_rl_routing_agent, Scalable RL agent created successfully")
            
            # Load checkpoint if available
            checkpoint_path = RL_MODEL_HYPERPARAMETERS.get('RL_CHECKPOINT_PATH')
            if checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    SCALABLE_RL_AGENT.load(checkpoint_path)
                    logger.info(f"scalable_rl_routing_agent, Loaded scalable RL checkpoint from {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"scalable_rl_routing_agent, Failed to load checkpoint: {e}")
        
        # Start training thread
        logger.info(f"{GREEN_COLOR}Starte scalable rl training worker in init()...{RESET_COLOR}")
        start_scalable_rl_training_worker()
        logger.info(f"{GREEN_COLOR}Scalable RL agent initialized and training thread started{RESET_COLOR}")
        logger.info("scalable_rl_routing_agent, Scalable RL agent initialized and training thread started")


def periodic_checkpoint_scalable_rl():
    """
    Periodically checkpoint the scalable RL agent with comprehensive metadata.
    
    This runs in a background thread and saves:
    - Model weights
    - Training progress (steps, episodes)
    - Performance metrics (rewards, success rate)
    - Buffer statistics
    - Human-readable JSON metadata
    """
    global SCALABLE_RL_AGENT, RL_MODEL_HYPERPARAMETERS
    
    try:
        if SCALABLE_RL_AGENT is None:
            return  # Agent not initialized yet
        
        with SCALABLE_RL_AGENT_LOCK.read():
            # Check if it's time to checkpoint
            checkpoint_interval = RL_MODEL_HYPERPARAMETERS.get('CHECKPOINT_INTERVAL_STEPS', 1000)
            total_steps = SCALABLE_RL_AGENT.total_steps
            
            # Checkpoint at regular intervals
            if total_steps > 0 and total_steps % checkpoint_interval < 64:  # 64 = typical batch size
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                checkpoint_name = f"scalable_rl_step_{total_steps}_{timestamp}"
                checkpoint_path = os.path.join(RL_MODEL_HYPERPARAMETERS['CHECKPOINT_DIR'], checkpoint_name)
                
                try:
                    # Upgrade to write lock for saving
                    with SCALABLE_RL_AGENT_LOCK.write():
                        logger.info(f"scalable_rl_routing_agent, Checkpointing scalable RL agent at step {total_steps}")
                        
                        # Save with comprehensive metadata
                        SCALABLE_RL_AGENT.save(
                            checkpoint_path,
                            save_buffer=False  # Don't save buffer by default (can be large)
                        )
                        
                        # Log metrics
                        metrics = SCALABLE_RL_AGENT.get_metrics()
                        if metrics.get('reward_stats'):
                            logger.info(f"scalable_rl_routing_agent, Avg reward (recent 100): {metrics['reward_stats']['avg_reward_recent']:.3f}")
                            logger.info(f"scalable_rl_routing_agent, Success rate: {metrics['success_rate']:.2%}")
                        
                        # Clean up old checkpoints (keep only last 5)
                        # cleanup_old_checkpoints(RL_MODEL_HYPERPARAMETERS['CHECKPOINT_DIR'], keep_latest=5)
                        
                except Exception as e:
                    logger.error(f"scalable_rl_routing_agent, Failed to save checkpoint: {e}")
    
    except Exception as e:
        logger.error(f"scalable_rl_routing_agent, Error in periodic_checkpoint_scalable_rl: {e}")


def cleanup_old_checkpoints(checkpoint_dir, keep_latest=5):
    """
    Remove old checkpoints, keeping only the most recent ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_latest: Number of recent checkpoints to keep
    """
    try:
        # Find all checkpoint files (main model file, not metadata)
        import glob
        checkpoint_files = []
        
        # Look for .zip files (SB3 PPO saves as .zip)
        for f in glob.glob(os.path.join(checkpoint_dir, "scalable_rl_step_*.zip")):
            # Get modification time
            mtime = os.path.getmtime(f)
            checkpoint_files.append((f, mtime))
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: x[1], reverse=True)
        
        # Delete old ones
        if len(checkpoint_files) > keep_latest:
            for checkpoint_path, _ in checkpoint_files[keep_latest:]:
                try:
                    # Remove checkpoint and associated files
                    base_path = checkpoint_path.replace('.zip', '')
                    
                    # Remove .zip, _metadata.pkl, _metadata.json
                    for ext in ['.zip', '_metadata.pkl', '_metadata.json', '_buffer.pkl']:
                        file_to_remove = base_path + ext if ext != '.zip' else checkpoint_path
                        if os.path.exists(file_to_remove):
                            os.remove(file_to_remove)
                            logger.info(f"scalable_rl_routing_agent, Removed old checkpoint: {os.path.basename(file_to_remove)}")
                except Exception as e:
                    logger.warning(f"scalable_rl_routing_agent, Failed to remove old checkpoint {checkpoint_path}: {e}")
    except Exception as e:
        logger.error(f"scalable_rl_routing_agent, Error cleaning up old checkpoints: {e}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)
    atexit.register(graceful_shutdown)
    
    
    port = int(os.environ.get("PORT", 8080))
    if not utils.wait_for_port_available(port, max_wait=5):
        logger.error(f"Cannot start Flask app - port {port} is not available")
        sys.exit(1)
        
    logger.info(f"Port {port} is available, starting Flask app properly!")
    
    init()

    logger.info(f"{RED_COLOR}init() finished in main()...{RESET_COLOR}")

    scheduler = BackgroundScheduler()
    # If online learning is disabled, just use the pretrained model
    if ENABLE_ONLINE_LEARNING:
        scheduler.add_job(func=online_train_routine, trigger="interval", seconds=30)
    else:
        logger.info("Online learning disabled. online_train_routine will not be invoked at all - using pretrained model only in inference")
    
    # # Add periodic checkpointing for scalable RL agent (every 2 minutes)
    # scheduler.add_job(func=periodic_checkpoint_scalable_rl, trigger="interval", seconds=120)
    # logger.info("Periodic checkpointing scheduled (every 2 minutes)")
    
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())
    
    # Start RL update worker thread
    logger.info(f"{GREEN_COLOR}Starting RL update worker in main()...{RESET_COLOR}")
    start_rl_update_worker()
        
    # NEW CODE: Add error handling around app.run()
    try:
        logger.info(f"Starting Flask app on port {port}")
        app.run(host="0.0.0.0", port=port, debug=False)
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"Port {port} is still in use. Trying to wait and retry...")
            time.sleep(5)
            if utils.wait_for_port_available(port, max_wait=30):
                app.run(host="0.0.0.0", port=port, debug=False)
            else:
                logger.error("Failed to start Flask app - port conflict persists")
                sys.exit(1)
        else:
            raise