## routing_agent_service.py

# import threading
# import joblib
import pandas as pd
import numpy as np
import random
# import uvicorn
# from pydantic import BaseModel
import os
import time
# import asyncio
# from concurrent.futures import ThreadPoolExecutor
import sys
import concurrent.futures
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

BROKER_LOCK = RWLock()

## colors for logging
BLUE_COLOR = "\033[94m"
RED_COLOR = "\033[91m"
GREEN_COLOR = "\033[92m"
PURPLE_COLOR = "\033[95m"
CYAN_COLOR = "\033[96m"
MAGENTA_COLOR = "\033[95m"
RESET_COLOR = "\033[0m" 

# INCLUDE_GPU_IN_FEATURE = True

unique_gpu_types = ['GPU-L3c', 'NVIDIA-A30']
use_multi_model = True
MAX_TOTAL_DATA = int(os.getenv("MAX_TOTAL_DATA", 20000))


app = Flask(__name__)
hyperparameter_file_path = '/app/final_model/model_config.json'
NUM_FLUSH = 0


ENCODED_DATA_DIR = "encoded_data"
ENCODED_DATA_DIR_GPU_L3c = "encoded_data_gpu_l3c"
ENCODED_DATA_DIR_NVIDIA_A30 = "encoded_data_nvidia_a30"


FINAL_MODEL_DIR = "/app/final_model"
FINAL_MODEL_DIR_GPU_L3c = "/app/final_model/A_GPU-L3c"
FINAL_MODEL_DIR_NVIDIA_A30 = "/app/final_model/A_NVIDIA-A30"
feature_normalization_stats_file = f"{FINAL_MODEL_DIR}/feature_normalization_statistics.csv"  # Add this near the top with your other constants;
feature_normalization_stats_file_gpu_l3c = f"{FINAL_MODEL_DIR_GPU_L3c}/feature_normalization_statistics.csv"
feature_normalization_stats_file_nvidia_a30 = f"{FINAL_MODEL_DIR_NVIDIA_A30}/feature_normalization_statistics.csv"


NUM_TRAINS = 0
NUM_TRAINS_GPU_L3c = 0
NUM_TRAINS_NVIDIA_A30 = 0

MODEL_UPDATED = True
MODEL_UPDATED_GPU_L3c = True
MODEL_UPDATED_NVIDIA_A30 = True
first_request_starting_time = None
STATS_INSTANCE = None
STATS_INSTANCE_GPU_L3c = None
STATS_INSTANCE_NVIDIA_A30 = None

TOTAL_NUM_DATA = 0
TOTAL_NUM_DATA_GPU_L3c = 0
TOTAL_NUM_DATA_NVIDIA_A30 = 0

NUM_NEW_DATA = 0
NUM_NEW_DATA_GPU_L3c = 0
NUM_NEW_DATA_NVIDIA_A30 = 0

TOTAL_NUM_NEW_DATA = 0
TOTAL_NUM_NEW_DATA_GPU_L3c = 0
TOTAL_NUM_NEW_DATA_NVIDIA_A30 = 0

TRAINING_RIGHT_NOW = False
TRAINING_RIGHT_NOW_GPU_L3c = False
TRAINING_RIGHT_NOW_NVIDIA_A30 = False

# Training data accumulation (offline + online)
TRAINING_DF = None  # Holds all training data (offline CSV + online appended data)
TRAINING_DF_NVIDIA_A30 = None
TRAINING_DF_GPU_L3c = None

TRAINING_DF_LOCK = threading.Lock()  # Thread safety for concurrent flush/train
TRAINING_DF_NVIDIA_A30_LOCK = threading.Lock()
TRAINING_DF_GPU_L3c_LOCK = threading.Lock()

OFFLINE_DATA_SIZE = 0  # Tracks the size of offline data portion in TRAINING_DF (shrinks as we remove overflow)
OFFLINE_DATA_SIZE_GPU_L3c = 0
OFFLINE_DATA_SIZE_NVIDIA_A30 = 0

PRINT_ONCE_AT_THE_FIRST_REQUEST = True
# RL agent globals
RL_AGENT = None  # Old RL agent (entire cluster as input) - for 'rl_agent' subAlgorithm
SCALABLE_RL_AGENT = None  # New scalable RL agent (pod-independent) - for 'scalable_rl_agent' subAlgorithm
LATENCY_PREDICTOR = None  # Latency predictor model - for 'latency_predictor' subAlgorithm
LATENCY_PREDICTOR_LOCK = RWLock()

LATENCY_PREDICTOR_GPU_L3c = None
LATENCY_PREDICTOR_GPU_L3c_LOCK = RWLock()
LATENCY_PREDICTOR_NVIDIA_A30 = None
LATENCY_PREDICTOR_NVIDIA_A30_LOCK = RWLock()


# RWLock enables concurrent predictions (readers) with exclusive updates (writer)
# - Predictions: use rwlock.read() for high concurrency
# - Updates: use rwlock.write() for exclusive access
# - Initialization: use rwlock.write() for exclusive access
RL_AGENT_LOCK = RWLock()
SCALABLE_RL_AGENT_LOCK = RWLock()

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

MIN_NUM_TRAINING_DATA = int(os.getenv("MIN_NUM_TRAINING_DATA", 5000))
MIN_NUM_UPDATE_DATA = int(os.getenv("MIN_NUM_UPDATE_DATA", 500))
POD_LABEL_SELECTOR = os.getenv("POD_LABEL_SELECTOR", "model.aibrix.ai/name=llama3-1-8b")
logger.info(f"POD_LABEL_SELECTOR: {POD_LABEL_SELECTOR}")
if POD_LABEL_SELECTOR == "":
    logger.error(f"POD_LABEL_SELECTOR is empty")
    assert False
ENABLE_ONLINE_LEARNING = int(os.getenv("ENABLE_ONLINE_LEARNING", 0))
EXPLORATION_ENABLED = int(os.getenv("EXPLORATION_ENABLED", 0))
TTFT_REWARD_WEIGHT = float(os.getenv("TTFT_REWARD_WEIGHT", 0.5))
EXPLORATION_RATE = float(os.getenv("EXPLORATION_RATE", 0.1))
SMOOTHING_ENABLED = int(os.getenv("SMOOTHING_ENABLED", 1))  # Enable smoothing for latency predictor
SMOOTHING_THRESHOLD = float(os.getenv("SMOOTHING_THRESHOLD", 0.1))  # 10% threshold by default
logger.info(f"Routing configuration: EXPLORATION_ENABLED={EXPLORATION_ENABLED}, EXPLORATION_RATE={EXPLORATION_RATE}, "
           f"SMOOTHING_ENABLED={SMOOTHING_ENABLED}, SMOOTHING_THRESHOLD={SMOOTHING_THRESHOLD}")
RL_MODEL_HYPERPARAMETERS = None
INIT_DONE = False




request_features_train = ['input_tokens', 'output_tokens', 'total_tokens']

# Fixed handle_flush function
@app.route("/flush", methods=["POST"])
def handle_flush():
    global NUM_FLUSH, ENCODED_DATA_DIR, ENCODED_DATA_DIR_GPU_L3c, ENCODED_DATA_DIR_NVIDIA_A30, TOTAL_NUM_DATA, NUM_NEW_DATA, TOTAL_NUM_NEW_DATA, NUM_NEW_DATA_GPU_L3c, NUM_NEW_DATA_NVIDIA_A30, TOTAL_NUM_NEW_DATA_GPU_L3c, TOTAL_NUM_NEW_DATA_NVIDIA_A30, feature_normalization_stats_file, TRAINING_DF, OFFLINE_DATA_SIZE, OFFLINE_DATA_SIZE_GPU_L3c, OFFLINE_DATA_SIZE_NVIDIA_A30, TRAINING_DF_GPU_L3c_LOCK, TRAINING_DF_NVIDIA_A30_LOCK, TRAINING_DF_GPU_L3c, TRAINING_DF_NVIDIA_A30, INIT_DONE

    if not INIT_DONE:
        return jsonify({"status": "error", "message": "Not initialized"}), 500

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
        ## seperate processed_df by GPU type
        processed_df_gpu_l3c = processed_df[processed_df["selectedPodGPU"] == 'GPU-L3c']
        processed_df_nvidia_a30 = processed_df[processed_df["selectedPodGPU"] == 'NVIDIA-A30']
        ##################################################
        logger.info(f"Successfully parsed data, took {time.time() - ts_preprocess} seconds")

        # Append preprocessed data to TRAINING_DF for online learning
        if ENABLE_ONLINE_LEARNING:
            if use_multi_model:
                with TRAINING_DF_GPU_L3c_LOCK:
                    if TRAINING_DF_GPU_L3c is None:
                        TRAINING_DF_GPU_L3c = processed_df_gpu_l3c.copy()
                        logger.info(f"Initialized TRAINING_DF_GPU_L3c with {len(processed_df_gpu_l3c)} samples")
                    else:
                        old_size = len(TRAINING_DF_GPU_L3c)
                        TRAINING_DF_GPU_L3c = pd.concat([TRAINING_DF_GPU_L3c, processed_df_gpu_l3c], ignore_index=True)
                        logger.info(f"Appended {len(processed_df_gpu_l3c)} samples to TRAINING_DF_GPU_L3c (total: {old_size} → {len(TRAINING_DF_GPU_L3c)})")
                with TRAINING_DF_NVIDIA_A30_LOCK:
                    if TRAINING_DF_NVIDIA_A30 is None:
                        TRAINING_DF_NVIDIA_A30 = processed_df_nvidia_a30.copy()
                        logger.info(f"Initialized TRAINING_DF_NVIDIA_A30 with {len(processed_df_nvidia_a30)} samples")
                    else:
                        old_size = len(TRAINING_DF_NVIDIA_A30)
                        TRAINING_DF_NVIDIA_A30 = pd.concat([TRAINING_DF_NVIDIA_A30, processed_df_nvidia_a30], ignore_index=True)
                        logger.info(f"Appended {len(processed_df_nvidia_a30)} samples to TRAINING_DF_NVIDIA_A30 (total: {old_size} → {len(TRAINING_DF_NVIDIA_A30)})")
            else:
                with TRAINING_DF_LOCK:
                    if TRAINING_DF is None:
                        TRAINING_DF = processed_df.copy()
                        logger.info(f"Initialized TRAINING_DF with {len(processed_df)} samples")
                    else:
                        old_size = len(TRAINING_DF)
                        TRAINING_DF = pd.concat([TRAINING_DF, processed_df], ignore_index=True)
                        logger.info(f"Appended {len(processed_df)} samples to TRAINING_DF (total: {old_size} → {len(TRAINING_DF)})")

        logger.info(f"Successfully flushed {len(log_data)} log messages, took {time.time() - flush_start_time} seconds")

        TOTAL_NUM_DATA += len(processed_df)
        NUM_NEW_DATA += len(processed_df)
        TOTAL_NUM_NEW_DATA += len(processed_df)

        TOTAL_NUM_DATA_GPU_L3c += len(processed_df_gpu_l3c)
        NUM_NEW_DATA_GPU_L3c += len(processed_df_gpu_l3c)
        TOTAL_NUM_NEW_DATA_GPU_L3c += len(processed_df_gpu_l3c)
        
        TOTAL_NUM_DATA_NVIDIA_A30 += len(processed_df_nvidia_a30)
        NUM_NEW_DATA_NVIDIA_A30 += len(processed_df_nvidia_a30)
        TOTAL_NUM_NEW_DATA_NVIDIA_A30 += len(processed_df_nvidia_a30)

        return jsonify({"status": "success", "message": f"Successfully processed {len(log_data)} log messages"}), 200

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        return jsonify({"status": "error", "message": str(e), "traceback": error_traceback}), 500

def infer_wrapper(tensor_data, predictor_instance, latency_predictor_lock, final_model_dir, request_id, sorted_all_pod_ids):
    global EXPLORATION_RATE, SMOOTHING_ENABLED, SMOOTHING_THRESHOLD, RL_MODEL_HYPERPARAMETERS
    #  TODO: if predictor_instance is None, should return it for global reuse
    if predictor_instance is None:
        with latency_predictor_lock.write():
            if predictor_instance is None:
                state_dims = {
                    'pod_features': tensor_data['pod_features_with_staleness'].shape[2],
                    'kv_hit_ratios': tensor_data['kv_hit_ratios'].shape[2],
                    'request_features': tensor_data['request_features'].shape[1],
                    'num_pods': tensor_data['pod_features_with_staleness'].shape[1]
                }
                
                logger.info(f"Initializing latency predictor with state_dims={state_dims}")
                predictor_instance = latency_predictor.LatencyPredictor(state_dims, RL_MODEL_HYPERPARAMETERS, final_model_dir)

                # Load pretrained model
                model_path = os.path.join(final_model_dir, 'latency_predictor.pth')
                if os.path.exists(model_path):
                    try:
                        predictor_instance.load(final_model_dir)
                        logger.info(f"Loaded latency predictor from {final_model_dir}")
                    except Exception as e:
                        logger.error(f"Failed to load latency predictor: {e}")
                else:
                    logger.warning(f"No pretrained latency predictor found at {model_path}, using untrained model")

    # Inference with read lock (allows concurrent requests)
    with latency_predictor_lock.read():
        # Get exploration and smoothing parameters
        result, infer_from_tensor_overhead_summary = latency_predictor.infer_latency_predictor_with_model(
            predictor=predictor_instance,
            tensor_data=tensor_data,
            request_id=request_id,
            sorted_all_pod_ids=sorted_all_pod_ids,
            exploration_rate=EXPLORATION_RATE,
            smoothing=bool(SMOOTHING_ENABLED),
            smoothing_threshold=SMOOTHING_THRESHOLD
        )
    return result, infer_from_tensor_overhead_summary


def slice_tensor_data_for_pods(tensor_data, pod_indices):
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


@app.route("/infer", methods=["POST"])
def handle_infer():
    global NUM_TRAINS, MODEL_UPDATED, first_request_starting_time, STATS_INSTANCE, STATS_INSTANCE_GPU_L3c, STATS_INSTANCE_NVIDIA_A30, RL_MODEL_HYPERPARAMETERS, PRINT_ONCE_AT_THE_FIRST_REQUEST, LATENCY_PREDICTOR_LOCK, LATENCY_PREDICTOR_GPU_L3c_LOCK, LATENCY_PREDICTOR_NVIDIA_A30_LOCK, LATENCY_PREDICTOR, LATENCY_PREDICTOR_GPU_L3c, LATENCY_PREDICTOR_NVIDIA_A30, FINAL_MODEL_DIR_GPU_L3c, FINAL_MODEL_DIR_NVIDIA_A30
    global use_multi_model

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
        if use_multi_model:
            if STATS_INSTANCE_GPU_L3c is None:
                logger.error(f"No running statistics available, STATS_INSTANCE_GPU_L3c: {STATS_INSTANCE_GPU_L3c}")
                logger.error("Cannot perform inference without normalization statistics")
                return jsonify({"error": "No normalization statistics available"}), 500
                assert False
            if STATS_INSTANCE_GPU_L3c.get_max_count() == 0:
                logger.error(f"Stats instance count is 0, no data available for normalization")
                logger.error(f"request_id,{request_id},No normalization statistics available for inference")
                assert False
            
            if STATS_INSTANCE_NVIDIA_A30 is None:
                logger.error(f"No running statistics available, STATS_INSTANCE_NVIDIA_A30: {STATS_INSTANCE_NVIDIA_A30}")
                logger.error("Cannot perform inference without normalization statistics")
                return jsonify({"error": "No normalization statistics available"}), 500
                assert False
            if STATS_INSTANCE_NVIDIA_A30.get_max_count() == 0:
                logger.error(f"Stats instance count is 0, no data available for normalization")
                logger.error(f"request_id,{request_id},No normalization statistics available for inference")
                assert False
                
        else:
            if STATS_INSTANCE is None:
                logger.error(f"No running statistics available, STATS_INSTANCE: {STATS_INSTANCE}")
                logger.error("Cannot perform inference without normalization statistics")
                return jsonify({"error": "No normalization statistics available"}), 500
                assert False
            if STATS_INSTANCE.get_max_count() == 0:
                logger.error(f"Stats instance count is 0, no data available for normalization")
                logger.error(f"request_id,{request_id},No normalization statistics available for inference")
                assert False

        normalizable_features, non_normalizable_features = data_normalizer._get_normalizable_features(processed_df, RL_MODEL_HYPERPARAMETERS.get('NO_NORMALIZE_FEATURES', []))

        non_interest = ['request_id', 'requestID', 'ttft', 'avg_tpot', 'e2e_latency', 'selected_pod', 'request_start_time', 'request_end_time']
        features_must_exist_in_stats_instance = []
        for feature in processed_df.columns:
            # NOTE: ignoring last_second_* features
            if "last_second_" not in feature and feature not in non_interest and feature in normalizable_features:
                features_must_exist_in_stats_instance.append(feature)
        for feature in features_must_exist_in_stats_instance:
            if use_multi_model:
                if feature not in STATS_INSTANCE_GPU_L3c.feature_stats:
                    logger.error(f"Feature {feature} not found in STATS_INSTANCE_GPU_L3c")
                    assert False
                if feature not in STATS_INSTANCE_NVIDIA_A30.feature_stats:
                    logger.error(f"Feature {feature} not found in STATS_INSTANCE_NVIDIA_A30")
                    assert False
            else:
                if feature not in STATS_INSTANCE.feature_stats:
                    logger.error(f"Feature {feature} not found in STATS_INSTANCE")
                    assert False
                
        for feature in normalizable_features:
            if use_multi_model:
                data_normalizer._normalize_single_feature(processed_df, feature, STATS_INSTANCE_GPU_L3c, is_training=False, request_id=request_id)
                data_normalizer._normalize_single_feature(processed_df, feature, STATS_INSTANCE_NVIDIA_A30, is_training=False, request_id=request_id)
            else:
                data_normalizer._normalize_single_feature(processed_df, feature, STATS_INSTANCE, is_training=False, request_id=request_id)
        handle_infer_overhead_summary["normalize"] = time.time() - normalize_start

        ## Encode data (normalization already done)
        encode_start_time = time.time()
        tensor_data, encode_for_inference_overhead_summary = encoding.encode_for_inference(sorted_all_pod_ids, processed_df, request_features_train, RL_MODEL_HYPERPARAMETERS)
        handle_infer_overhead_summary["encode"] = time.time() - encode_start_time

        infer_from_tensor_start_time = time.time()
        
        subAlgorithm = processed_df['subAlgorithm'].iloc[0]
        logger.info(f"requestID: {request_id}, subAlgorithm: {subAlgorithm}")

        selected_pod_generalpodid = None
    
        if use_multi_model:
            parallel_infer_start = time.time()
            # Slide tensor_data by gpu type
            podid_to_gpu_type = {pod_id: processed_df[f"{pod_id}-GPU"].iloc[0] for pod_id in sorted_all_pod_ids}
            gpu_type_to_pod_ids = {gpu_type: [pod_id for pod_id in sorted_all_pod_ids if podid_to_gpu_type[pod_id] == gpu_type] for gpu_type in set(podid_to_gpu_type.values())}

            logger.error(f"gpu_type_to_pod_ids: {gpu_type_to_pod_ids}")

            l3c_pods = gpu_type_to_pod_ids['GPU-L3c']
            nvidia_a30_pods = gpu_type_to_pod_ids['NVIDIA-A30']

            logger.info(f"l3c_pods: {l3c_pods}")
            logger.info(f"nvidia_a30_pods: {nvidia_a30_pods}")

            # TODO: should call parallely
            tensor_data_gpu_l3c = slice_tensor_data_for_pods(tensor_data, l3c_pods)
            tensor_data_nvidia_a30 = slice_tensor_data_for_pods(tensor_data, nvidia_a30_pods)
            sorted_all_pod_ids_gpu_l3c = [sorted_all_pod_ids[i] for i in l3c_pods]
            sorted_all_pod_ids_nvidia_a30 = [sorted_all_pod_ids[i] for i in nvidia_a30_pods]

            result_gpu_l3c, infer_from_tensor_overhead_summary_gpu_l3c = infer_wrapper(tensor_data_gpu_l3c, LATENCY_PREDICTOR_GPU_L3c, LATENCY_PREDICTOR_GPU_L3c_LOCK, FINAL_MODEL_DIR_GPU_L3c, request_id, sorted_all_pod_ids_gpu_l3c)
            result_nvidia_a30, infer_from_tensor_overhead_summary_nvidia_a30 = infer_wrapper(tensor_data_nvidia_a30, LATENCY_PREDICTOR_NVIDIA_A30, LATENCY_PREDICTOR_NVIDIA_A30_LOCK, FINAL_MODEL_DIR_NVIDIA_A30, request_id, sorted_all_pod_ids_nvidia_a30)

            handle_infer_overhead_summary["parallel_infer"] = time.time() - parallel_infer_start

            # Combine results from both GPUs (l3c and nvidia_a30)
            combine_start = time.time()
            predicted_latencies_gpu_l3c = {sorted_all_pod_ids_gpu_l3c[i]: result_gpu_l3c['predicted_latencies'][i] for i in range(len(sorted_all_pod_ids_gpu_l3c))}
            predicted_latencies_nvidia_a30 = {sorted_all_pod_ids_nvidia_a30[i]: result_nvidia_a30['predicted_latencies'][i] for i in range(len(sorted_all_pod_ids_nvidia_a30))}
            predicted_latencies = {**predicted_latencies_gpu_l3c, **predicted_latencies_nvidia_a30}
            if result_gpu_l3c['chosen_pod_predicted_latency'] < result_nvidia_a30['chosen_pod_predicted_latency']:
                chosen_pod_predicted_latency = result_gpu_l3c['chosen_pod_predicted_latency']
                selected_pod_generalpodid = sorted_all_pod_ids_gpu_l3c[result_gpu_l3c['selected_pod_index']]
            else:
                chosen_pod_predicted_latency = result_nvidia_a30['chosen_pod_predicted_latency']
                selected_pod_generalpodid = sorted_all_pod_ids_nvidia_a30[result_nvidia_a30['selected_pod_index']]

            result = {
                'selected_pod_index': None,
                'predicted_latencies': predicted_latencies,
                'chosen_pod_predicted_latency': chosen_pod_predicted_latency,
                'confidence': None,
                'pod_probabilities': None,
                'explore_mask': None,
                'smoothing_mask': None,
            }
            infer_from_tensor_overhead_summary = {f"{key}_gpu_l3c": value for key, value in infer_from_tensor_overhead_summary_gpu_l3c.items()}
            infer_from_tensor_overhead_summary.update({f"{key}_nvidia_a30": value for key, value in infer_from_tensor_overhead_summary_nvidia_a30.items()})

            handle_infer_overhead_summary["combine_parallel_results"] = time.time() - combine_start

        else:
            if subAlgorithm == 'latency_predictor':
                result, infer_from_tensor_overhead_summary = infer_wrapper(tensor_data, LATENCY_PREDICTOR, LATENCY_PREDICTOR_LOCK, FINAL_MODEL_DIR, request_id, sorted_all_pod_ids)
            elif subAlgorithm == 'contextual_bandit' or subAlgorithm == 'rl_naive':
                logger.info(f"subAlgorithm: {subAlgorithm}, Using contextual bandit model for inference (request_id: {request_id})")
                result, infer_from_tensor_overhead_summary = simpler_contextual_bandit.infer_from_tensor(tensor_data, request_id, MODEL_UPDATED, RL_MODEL_HYPERPARAMETERS, FINAL_MODEL_DIR)
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
                
                # global RL_AGENT
                
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
            elif subAlgorithm == 'scalable_rl_agent_old':
                # === NEW SCALABLE RL AGENT (pod-count independent) ===
                logger.info(f"requestID: {request_id}, subAlgorithm: {subAlgorithm}, Using SCALABLE RL agent (pod-independent) for inference")
            
                global SCALABLE_RL_AGENT
                
                with SCALABLE_RL_AGENT_LOCK.write():
                    if SCALABLE_RL_AGENT is None:
                        # Initialize ONCE - works for any number of pods!
                        pod_features_t = tensor_data['pod_features']
                        per_pod_dim = int(pod_features_t.shape[2])  # e.g., 10
                        kv_hit_t = tensor_data['kv_hit_ratios']
                        kv_dim = int(kv_hit_t.shape[2])  # e.g., 1
                        req_features_t = tensor_data['request_features']
                        req_dim = int(req_features_t.shape[1])  # e.g., 3
                        
                        # Per-pod dimension = pod_features + kv_hit_ratios
                        total_per_pod_dim = per_pod_dim + kv_dim
                        
                        SCALABLE_RL_AGENT = create_scalable_rl_agent(
                            per_pod_dim=total_per_pod_dim,  # 11 (10 pod + 1 kv)
                            request_dim=req_dim,             # 3
                            max_pods=100,                    # Max expected pods
                            **RL_MODEL_HYPERPARAMETERS
                        )
                        
                        # Load checkpoint if available
                        ckpt_path = RL_MODEL_HYPERPARAMETERS.get('RL_CHECKPOINT_PATH')
                        if ckpt_path and os.path.exists(ckpt_path):
                            try:
                                SCALABLE_RL_AGENT.load(ckpt_path)
                                logger.info(f"✅ Loaded scalable RL checkpoint from {ckpt_path}")
                            except Exception as e:
                                logger.warning(f"⚠️  Failed to load RL checkpoint {ckpt_path}: {e}")
                        
                        logger.info(f"🚀 Initialized SCALABLE RL agent: per_pod_dim={total_per_pod_dim}, "
                                f"request_dim={req_dim}, max_pods=100 (works with ANY #pods!)")
                    
                    current_agent = SCALABLE_RL_AGENT
                
                # Inference (no lock needed - thread-safe in new design)
                current_agent, result, infer_from_tensor_overhead_summary = infer_scalable_rl_agent(
                    tensor_data=tensor_data,
                    request_id=request_id,
                    sorted_all_pod_ids=sorted_all_pod_ids,
                    processed_df=processed_df,
                    rl_agent=current_agent,
                    hyperparameters=RL_MODEL_HYPERPARAMETERS,
                    agent_lock=None  # New agent doesn't need lock for prediction
                )
            else:
                logger.info(f"requestID: {request_id}, contextual bandit model for inference")
                result, infer_from_tensor_overhead_summary = simpler_contextual_bandit.infer_from_tensor(tensor_data, request_id, MODEL_UPDATED, RL_MODEL_HYPERPARAMETERS, FINAL_MODEL_DIR)
                result['predicted_latencies'] = {pod_id: -1 for pod_id in sorted_all_pod_ids}
                result['chosen_pod_predicted_latency'] = -1
        handle_infer_overhead_summary["calling_infer_from_tensor"] = time.time() - infer_from_tensor_start_time
        
        remaining_work_start = time.time()
        result["requestID"] = request_id
        result["num_trains"] = NUM_TRAINS
        result["request_timestamp"] = time.time() - first_request_starting_time
        logger.info(f"requestID: {request_id}, inference result: {result}")
        
        if selected_pod_generalpodid is None:
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
            "smoothing": result.get('smoothing_mask', 0),  # Include smoothing indicator
        }
        
        # if subAlgorithm == 'latency_predictor' and NUM_TRAINS <= 0:
        #     random_generalpodid = sorted_all_pod_ids[random.randint(0, len(sorted_all_pod_ids)-1)]
        #     random_pod_ip = RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip'][random_generalpodid]
        #     response['selected_pod'] = random_pod_ip
        #     response['selected_pod_generalpodid'] = random_generalpodid
        #     logger.info(f"Training never done yet, selected random pod for exploration! selected_pod_generalpodid: {random_generalpodid}, selected_pod_ip: {random_pod_ip}")
            
        return jsonify(response), 200
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error in handle_infer: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 500


def online_train_helper(training_df, training_df_lock, training_right_now, total_num_new_data, num_new_data, min_num_training_data, min_num_update_data, offline_data_size, num_trains, encoded_data_dir, final_model_dir, latency_predictor, latency_predictor_lock, stats_instance):
    global RL_MODEL_HYPERPARAMETERS, MAX_TOTAL_DATA
    if training_right_now:
        logger.info(f"Previous training still in progress, skipping training")
        return training_right_now, None, num_new_data, num_trains
    if total_num_new_data < min_num_training_data:
        logger.info(f"Not enough total training data available, num_new_data: {total_num_new_data} < {min_num_training_data}, wait until enough data are added. num_trains: {num_trains}, TOTAL_NUM_DATA: {TOTAL_NUM_DATA}")
        return training_right_now, None, num_new_data, num_trains
    if num_new_data < min_num_update_data:
        logger.info(f"Not enough new training data available, num_new_data: {num_new_data} < {min_num_update_data}, wait until enough data are added. num_trains: {num_trains}, TOTAL_NUM_DATA: {TOTAL_NUM_DATA}")
        return training_right_now, None, num_new_data, num_trains
    training_right_now = True
    training_start_time = time.time()
    logger.info(f"online_train_routine start, {num_trains}th online training with {num_new_data} new training data")
    try:
        # Route to appropriate training function based on model type
        model_type = RL_MODEL_HYPERPARAMETERS['MODEL_TYPE']
        if model_type == 'latency_predictor':
            logger.info(f"Training with latency predictor model on entire dataset (offline({len(training_df)}) + online({num_new_data}))")

            # Get a copy of training_df for training (thread-safe)
            with training_df_lock:
                if training_df is None or len(training_df) == 0:
                    logger.error("training_df is empty, cannot train")
                    training_right_now = False
                    return training_right_now, None, num_new_data, num_trains
                training_df_copy = training_df.copy()
                total_samples = len(training_df_copy)
                current_offline_size = offline_data_size

            logger.info(f"Training on total data: {total_samples} (offline: {current_offline_size}, online: {total_samples - current_offline_size})")

            if total_samples > MAX_TOTAL_DATA and current_offline_size > 0:
                overflow = total_samples - MAX_TOTAL_DATA
                to_remove = min(overflow, current_offline_size)
                training_df_copy = training_df_copy.iloc[to_remove:].reset_index(drop=True)
                logger.info(f"⚠️  Total data ({total_samples}) exceeds limit ({MAX_TOTAL_DATA}). Removed {to_remove} oldest offline samples (overflow={overflow}, offline_size={current_offline_size})")
                with training_df_lock:
                    training_df = training_df_copy.copy()
                    offline_data_size = max(0, current_offline_size - to_remove)
                    logger.info(f"✅ Updated training_df: new size={len(training_df)} (offline: {offline_data_size}, online: {len(training_df) - offline_data_size})")
                total_samples = len(training_df_copy)
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
            normalizable_features, non_normalizable_features = data_normalizer._get_normalizable_features(training_df_copy, RL_MODEL_HYPERPARAMETERS.get('NO_NORMALIZE_FEATURES', []))

            for feature in normalizable_features:
                # FIX: Changed from is_training=False to is_training=True
                # This allows normalization stats to update during online training,
                # preventing outlier creation when new data has different distributions
                data_normalizer._normalize_single_feature(training_df_copy, feature, stats_instance, is_training=True)
        
            # Get sorted pod IDs and GPU mapping
            sorted_all_pod_ids = utils.get_sorted_all_pod_ids('processed_csv_columns', training_df_copy.columns.tolist())
            # generalpodid_to_gpu_model = RL_MODEL_HYPERPARAMETERS.get('generalpodid_to_gpu_model', {})
            logger.info(f"Training with pods: {sorted_all_pod_ids}")
            
            # Encode the entire dataset
            encode_start_time = time.time()
            os.makedirs(encoded_data_dir, exist_ok=True)
            encoded_training_dir = os.path.join(encoded_data_dir, "full_training_data")
            encoding.encode_for_train(sorted_all_pod_ids, training_df_copy, encoded_training_dir, request_features_train, RL_MODEL_HYPERPARAMETERS)
            logger.info(f"Encoded {total_samples} samples to {encoded_training_dir}, encode time: {time.time() - encode_start_time} seconds")

            # Train on the encoded dataset
            train_start_time = time.time()
            latency_predictor.train_latency_predictor(encoded_training_dir, final_model_dir, RL_MODEL_HYPERPARAMETERS, num_trains)
            logger.info(f"train_latency_predictor done, train time: {time.time() - train_start_time} seconds")

            # Reload global latency_predictor (for single-model fallback path)
            with latency_predictor_lock.write():
                if latency_predictor is not None:
                    load_start_time = time.time()
                    latency_predictor.load(final_model_dir)
                    logger.info(f"Reloaded latency predictor after training, load time: {time.time() - load_start_time} seconds")
        else:
            logger.info(f"Training with contextual bandit model")
            simpler_contextual_bandit.train(encoded_data_dir, final_model_dir, RL_MODEL_HYPERPARAMETERS, ENABLE_ONLINE_LEARNING)
            logger.info(f"train_contextual_bandit done")
    except Exception as e:
        import traceback
        logger.error(f"Error during training: {e}")
        logger.error(traceback.format_exc())
        training_right_now = False
        return training_right_now, None, num_new_data, num_trains
    logger.info(f"Successfully completed online_train_routine done, {num_trains}th online training with {num_new_data} new training data, took {time.time() - training_start_time} seconds")
    model_updated = True
    training_right_now = False
    num_trains += 1
    num_new_data = 0
    return training_right_now, model_updated, num_new_data, num_trains

    
def online_train_routine():
    global use_multi_model
    global NUM_TRAINS, MODEL_UPDATED, TOTAL_NUM_DATA, FINAL_MODEL_DIR, NUM_NEW_DATA, TOTAL_NUM_NEW_DATA, RL_MODEL_HYPERPARAMETERS, TRAINING_RIGHT_NOW, TRAINING_RIGHT_NOW_GPU_L3c, TRAINING_RIGHT_NOW_NVIDIA_A30, MODEL_UPDATED_GPU_L3c, MODEL_UPDATED_NVIDIA_A30, NUM_NEW_DATA_GPU_L3c, NUM_NEW_DATA_NVIDIA_A30, NUM_TRAINS_GPU_L3c, NUM_TRAINS_NVIDIA_A30, LATENCY_PREDICTOR, TRAINING_DF, STATS_INSTANCE, STATS_INSTANCE_GPU_L3c, STATS_INSTANCE_NVIDIA_A30, OFFLINE_DATA_SIZE, OFFLINE_DATA_SIZE_GPU_L3c, OFFLINE_DATA_SIZE_NVIDIA_A30, TOTAL_NUM_DATA_GPU_L3c, TOTAL_NUM_DATA_NVIDIA_A30

    if use_multi_model:
        TRAINING_RIGHT_NOW_GPU_L3c, MODEL_UPDATED_GPU_L3c, NUM_NEW_DATA_GPU_L3c, NUM_TRAINS_GPU_L3c = online_train_helper(TRAINING_DF_GPU_L3c, TRAINING_DF_GPU_L3c_LOCK, TRAINING_RIGHT_NOW_GPU_L3c, TOTAL_NUM_NEW_DATA_GPU_L3c, NUM_NEW_DATA_GPU_L3c, MIN_NUM_TRAINING_DATA, MIN_NUM_UPDATE_DATA, OFFLINE_DATA_SIZE_GPU_L3c, NUM_TRAINS_GPU_L3c, ENCODED_DATA_DIR_GPU_L3c, FINAL_MODEL_DIR_GPU_L3c, LATENCY_PREDICTOR_GPU_L3c, LATENCY_PREDICTOR_GPU_L3c_LOCK, STATS_INSTANCE_GPU_L3c)
        TRAINING_RIGHT_NOW_NVIDIA_A30, MODEL_UPDATED_NVIDIA_A30, NUM_NEW_DATA_NVIDIA_A30, NUM_TRAINS_NVIDIA_A30 = online_train_helper(TRAINING_DF_NVIDIA_A30, TRAINING_DF_NVIDIA_A30_LOCK, TRAINING_RIGHT_NOW_NVIDIA_A30, TOTAL_NUM_NEW_DATA_NVIDIA_A30, NUM_NEW_DATA_NVIDIA_A30, MIN_NUM_TRAINING_DATA, MIN_NUM_UPDATE_DATA, OFFLINE_DATA_SIZE_NVIDIA_A30, NUM_TRAINS_NVIDIA_A30, ENCODED_DATA_DIR_NVIDIA_A30, FINAL_MODEL_DIR_NVIDIA_A30, LATENCY_PREDICTOR_NVIDIA_A30, LATENCY_PREDICTOR_NVIDIA_A30_LOCK, STATS_INSTANCE_NVIDIA_A30)
    else:
        TRAINING_RIGHT_NOW, MODEL_UPDATED, NUM_NEW_DATA, NUM_TRAINS = online_train_helper(TRAINING_DF, TRAINING_DF_LOCK, TRAINING_RIGHT_NOW, TOTAL_NUM_NEW_DATA, NUM_NEW_DATA, MIN_NUM_TRAINING_DATA, MIN_NUM_UPDATE_DATA, OFFLINE_DATA_SIZE, NUM_TRAINS, ENCODED_DATA_DIR, FINAL_MODEL_DIR, LATENCY_PREDICTOR, LATENCY_PREDICTOR_LOCK, STATS_INSTANCE)


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
    global RL_MODEL_HYPERPARAMETERS, STATS_INSTANCE, STATS_INSTANCE_GPU_L3c, STATS_INSTANCE_NVIDIA_A30, use_multi_model, INIT_DONE
    logger.info(f"start init(), INIT_DONE: {INIT_DONE}")
    if RL_MODEL_HYPERPARAMETERS is None:
        logger.info(f"{GREEN_COLOR}RL_MODEL_HYPERPARAMETERS is None{RESET_COLOR}")

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
        # generalpodid_to_gpu_model = utils.fetch_generalpodid_to_gpu_model(running_vllm_pods, pod_ip_to_generalpodid)
        # pod_ip_to_gpu_model, pod_ip_to_gpu_model_encoded = utils.create_pod_ip_to_gpu_model_mapping(generalpodid_to_gpu_model, pod_ip_to_generalpodid)
        
        logger.info(f"POD_LABEL_SELECTOR: {POD_LABEL_SELECTOR}")
        logger.info(f"len(sorted_running_pod_ips): {len(sorted_running_pod_ips)}, sorted_running_pod_ips: {sorted_running_pod_ips}")
        logger.info(f"pod_ip_to_generalpodid: {pod_ip_to_generalpodid}")
        # logger.info(f"generalpodid_to_gpu_model: {generalpodid_to_gpu_model}")
        # logger.info(f"pod_ip_to_gpu_model: {pod_ip_to_gpu_model}")
        # logger.info(f"pod_ip_to_gpu_model_encoded: {pod_ip_to_gpu_model_encoded}")

        RL_MODEL_HYPERPARAMETERS['pod_ip_to_generalpodid'] = pod_ip_to_generalpodid
        RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip'] = generalpodid_to_pod_ip
        logger.info(f"RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip']: {RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip']}")
        RL_MODEL_HYPERPARAMETERS['sorted_running_pod_ips'] = sorted_running_pod_ips
        # RL_MODEL_HYPERPARAMETERS['pod_ip_to_gpu_model'] = pod_ip_to_gpu_model
        # RL_MODEL_HYPERPARAMETERS['pod_ip_to_gpu_model_encoded'] = pod_ip_to_gpu_model_encoded
        # RL_MODEL_HYPERPARAMETERS['generalpodid_to_gpu_model'] = generalpodid_to_gpu_model
        # Additional mappings for GPU features expected by preprocess/encoding
        # RL_MODEL_HYPERPARAMETERS['pod_gpu_mapping'] = generalpodid_to_gpu_model
        # pod_gpu_id_mapping = {}
        # for generalpodid, gpu_model in generalpodid_to_gpu_model.items():
        #     if gpu_model in utils.GPU_MODEL_TO_ENCODE:
        #         pod_gpu_id_mapping[generalpodid] = utils.GPU_MODEL_TO_ENCODE[gpu_model]
        #     else:
        #         logger.error(f"Unknown GPU model for {generalpodid}: {gpu_model}")
        #         assert False
        # RL_MODEL_HYPERPARAMETERS['pod_gpu_id_mapping'] = pod_gpu_id_mapping
        
        
        # Load normalization statistics from CSV file
        if use_multi_model:
            feature_normalization_stats_file_gpu_l3c = f"{FINAL_MODEL_DIR_GPU_L3c}/feature_normalization_statistics.csv"
            feature_normalization_stats_file_nvidia_a30 = f"{FINAL_MODEL_DIR_NVIDIA_A30}/feature_normalization_statistics.csv"
            if os.path.exists(feature_normalization_stats_file_gpu_l3c):
                logger.info(f"Loading normalization statistics from: {feature_normalization_stats_file_gpu_l3c}")
                try:
                    STATS_INSTANCE_GPU_L3c = data_normalizer.FeatureStats.load_from_csv(feature_normalization_stats_file_gpu_l3c)
                except Exception as e:
                    logger.error(f"Failed to load normalization statistics: {e}")
                    assert False
            else:
                logger.error(f"Normalization statistics file not found: {feature_normalization_stats_file_gpu_l3c}")
                assert False
            if STATS_INSTANCE_GPU_L3c is not None:
                logger.info(f"Successfully loaded stats for {len(STATS_INSTANCE_GPU_L3c.feature_stats)} features")
                for feature_name, stats in STATS_INSTANCE_GPU_L3c.feature_stats.items():
                    logger.info(f"STATS_INSTANCE_GPU_L3c, {feature_name}: count={stats.count}, mean={stats.mean}, std={stats.std}")


            if os.path.exists(feature_normalization_stats_file_nvidia_a30):
                logger.info(f"Loading normalization statistics from: {feature_normalization_stats_file_nvidia_a30}")
                try:
                    STATS_INSTANCE_NVIDIA_A30 = data_normalizer.FeatureStats.load_from_csv(feature_normalization_stats_file_nvidia_a30)
                except Exception as e:
                    logger.error(f"Failed to load normalization statistics: {e}")
                    assert False
            else:
                logger.error(f"Normalization statistics file not found: {feature_normalization_stats_file_nvidia_a30}")
                assert False
            if STATS_INSTANCE_NVIDIA_A30 is not None:
                logger.info(f"Successfully loaded stats for {len(STATS_INSTANCE_NVIDIA_A30.feature_stats)} features")
                for feature_name, stats in STATS_INSTANCE_NVIDIA_A30.feature_stats.items():
                    logger.info(f"STATS_INSTANCE_NVIDIA_A30, {feature_name}: count={stats.count}, mean={stats.mean}, std={stats.std}")
        else:
            if os.path.exists(feature_normalization_stats_file):
                logger.info(f"Loading normalization statistics from: {feature_normalization_stats_file}")
                try:
                    STATS_INSTANCE = data_normalizer.FeatureStats.load_from_csv(feature_normalization_stats_file)
                    if STATS_INSTANCE is not None:
                        logger.info(f"Successfully loaded stats for {len(STATS_INSTANCE.feature_stats)} features")
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
            if STATS_INSTANCE is not None:
                logger.info("Per-feature statistics loaded:")
                for feature_name, stats in STATS_INSTANCE.feature_stats.items():
                    logger.info(f"STATS_INSTANCE, {feature_name}: count={stats.count}, mean={stats.mean}, std={stats.std}")
            else:
                logger.warning("No normalization statistics available - inference will fail")
                assert False
    
    # Add checkpointing configuration to hyperparameters
    RL_MODEL_HYPERPARAMETERS['CHECKPOINT_INTERVAL_STEPS'] = 100
    RL_MODEL_HYPERPARAMETERS['CHECKPOINT_DIR'] = os.path.join(FINAL_MODEL_DIR, 'checkpoints')
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(RL_MODEL_HYPERPARAMETERS['CHECKPOINT_DIR'], exist_ok=True)
    logger.info(f"Checkpoint directory: {RL_MODEL_HYPERPARAMETERS['CHECKPOINT_DIR']}")
    logger.info(f"Checkpointing every {RL_MODEL_HYPERPARAMETERS['CHECKPOINT_INTERVAL_STEPS']} steps")

    # Load offline training data for online learning
    global TRAINING_DF, OFFLINE_DATA_SIZE, TRAINING_DF_GPU_L3c, TRAINING_DF_NVIDIA_A30, TRAINING_DF_GPU_L3c_LOCK, TRAINING_DF_NVIDIA_A30_LOCK, OFFLINE_DATA_SIZE_GPU_L3c, OFFLINE_DATA_SIZE_NVIDIA_A30, TOTAL_NUM_NEW_DATA, TOTAL_NUM_NEW_DATA_GPU_L3c, TOTAL_NUM_NEW_DATA_NVIDIA_A30
    if ENABLE_ONLINE_LEARNING:
        if use_multi_model:
            offline_csv_path_GPU_L3c = "/app/offline_training_data_GPU-L3c.csv"
            offline_csv_path_NVIDIA_A30 = "/app/offline_training_data_NVIDIA-A30.csv"
            ###################################################################### 
            ## GPU-L3c
            ###################################################################### 
            if os.path.exists(offline_csv_path_GPU_L3c):
                try:
                    with TRAINING_DF_GPU_L3c_LOCK:
                        TRAINING_DF_GPU_L3c = pd.read_csv(offline_csv_path_GPU_L3c)
                        # shuffle the training data
                        TRAINING_DF_GPU_L3c = TRAINING_DF_GPU_L3c.sample(frac=1).reset_index(drop=True)
                        OFFLINE_DATA_SIZE_GPU_L3c = len(TRAINING_DF_GPU_L3c)
                        logger.info(f"✅ Loaded offline training data: {len(TRAINING_DF_GPU_L3c)} samples from {offline_csv_path_GPU_L3c}")
                        logger.info(f"   Columns: {list(TRAINING_DF_GPU_L3c.columns[:10])}...")  # Show first 10 columns
                except Exception as e:
                    logger.error(f"Failed to load offline training data: {e}")
                    TRAINING_DF_GPU_L3c = pd.DataFrame()
                    OFFLINE_DATA_SIZE_GPU_L3c = 0
                    logger.warning("Starting with empty training dataframe")
            else:
                logger.warning(f"Offline training data not found at {offline_csv_path_GPU_L3c}")
                logger.warning("offline_csv_path_GPU_L3c, Online learning will start from scratch with only new data")
                TRAINING_DF_GPU_L3c = pd.DataFrame()
                OFFLINE_DATA_SIZE_GPU_L3c = 0
            TOTAL_NUM_NEW_DATA_GPU_L3c = len(TRAINING_DF_GPU_L3c)
            ###################################################################### 
            ## NVIDIA-A30
            ######################################################################
            if os.path.exists(offline_csv_path_NVIDIA_A30):
                try:
                    with TRAINING_DF_NVIDIA_A30_LOCK:
                        TRAINING_DF_NVIDIA_A30 = pd.read_csv(offline_csv_path_NVIDIA_A30)
                        # shuffle the training data
                        TRAINING_DF_NVIDIA_A30 = TRAINING_DF_NVIDIA_A30.sample(frac=1).reset_index(drop=True)
                        OFFLINE_DATA_SIZE_NVIDIA_A30 = len(TRAINING_DF_NVIDIA_A30)
                        logger.info(f"✅ Loaded offline training data: {len(TRAINING_DF_NVIDIA_A30)} samples from {offline_csv_path_NVIDIA_A30}")
                        logger.info(f"   Columns: {list(TRAINING_DF_NVIDIA_A30.columns[:10])}...")  # Show first 10 columns
                except Exception as e:
                    logger.error(f"Failed to load offline training data: {e}")
                    TRAINING_DF_NVIDIA_A30 = pd.DataFrame()
                    OFFLINE_DATA_SIZE_NVIDIA_A30 = 0
                    logger.warning("Starting with empty training dataframe")
            else:
                logger.warning(f"Offline training data not found at {offline_csv_path_NVIDIA_A30}")
                logger.warning("offline_csv_path_NVIDIA_A30, Online learning will start from scratch with only new data")
                TRAINING_DF_NVIDIA_A30 = pd.DataFrame()
                OFFLINE_DATA_SIZE_NVIDIA_A30 = 0
            TOTAL_NUM_NEW_DATA_NVIDIA_A30 = len(TRAINING_DF_NVIDIA_A30)
            TOTAL_NUM_NEW_DATA = TOTAL_NUM_NEW_DATA_GPU_L3c + TOTAL_NUM_NEW_DATA_NVIDIA_A30
        else:
            offline_csv_path = "/app/offline_training_data.csv"
            if os.path.exists(offline_csv_path):
                try:
                    with TRAINING_DF_LOCK:
                        TRAINING_DF = pd.read_csv(offline_csv_path)
                        # shuffle the training data
                        TRAINING_DF = TRAINING_DF.sample(frac=1).reset_index(drop=True)
                        OFFLINE_DATA_SIZE = len(TRAINING_DF)
                        logger.info(f"✅ Loaded offline training data: {len(TRAINING_DF)} samples from {offline_csv_path}")
                        logger.info(f"   Columns: {list(TRAINING_DF.columns[:10])}...")  # Show first 10 columns
                except Exception as e:
                    logger.error(f"Failed to load offline training data: {e}")
                    TRAINING_DF = pd.DataFrame()
                    OFFLINE_DATA_SIZE = 0
                    logger.warning("Starting with empty training dataframe")
            else:
                logger.warning(f"Offline training data not found at {offline_csv_path}")
                logger.warning("Online learning will start from scratch with only new data")
                TRAINING_DF = pd.DataFrame()
                OFFLINE_DATA_SIZE = 0
            TOTAL_NUM_NEW_DATA = len(TRAINING_DF)
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
    INIT_DONE = True

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
    if RL_MODEL_HYPERPARAMETERS['MODEL_TYPE'] == 'scalable_rl_agent':
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