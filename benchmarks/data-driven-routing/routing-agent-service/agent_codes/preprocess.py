#!/usr/bin/env python3

# preprocess.py

import pandas as pd
import numpy as np
import json
import ast
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import argparse
import sys
import time
from logger import logger, INCLUDE_GPU_IN_FEATURE
import utils as utils
# INCLUDE_GPU_IN_FEATURE = True

def parse_json_columns(df, json_columns):
    for col in json_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    return df

def parse_log_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Check if this is a metrics line
            if "latency_metrics" not in line:
                logger.error(f"Invalid line. {line}")
                assert False
            if "**@" in line:
                line = line.split("**@latency_metrics@")[1]
            parts = line.split('@')
            row = {}
            json_columns = list()
            column_names = list()
            for i in range(0, len(parts), 2):
                if i + 1 >= len(parts):
                    break
                column_name = parts[i]
                column_names.append(column_name)
                value = parts[i+1]
                if value.startswith('{') and value.endswith('}'):
                    try:
                        # NEW: Fix escaped quotes issue - replace \" with " before parsing
                        fixed_value = value.replace('\\"', '"')
                        json_columns.append(column_name)
                        row[column_name] = json.loads(fixed_value)
                    except Exception as e:
                        logger.error(f"Error decoding JSON, column: {column_name}, value: {value}")
                        logger.error(f"Error: {e}")
                        # Since we can't parse it, store as string to avoid losing data
                        row[column_name] = value
                else:
                    try:
                        row[column_name] = int(value)
                    except ValueError:
                        try:
                            row[column_name] = float(value)
                        except ValueError:
                            row[column_name] = value
            data.append(row)
    parsed_df = pd.DataFrame(data, columns=column_names)
    if len(parsed_df) == 0:
        logger.error("No data found in the log file.")
        assert False
    return parsed_df, json_columns

def safe_parse_json(json_str):
    """Safely parse Python dictionary-like strings or JSON strings"""
    # If already a dictionary, return as is
    if isinstance(json_str, dict):
        return json_str
    if pd.isna(json_str) or not json_str:
        logger.error(f"ERROR: Empty or NaN JSON string: {str(json_str)}...")
        assert False
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        if isinstance(json_str, str):
            return json.loads(json_str.replace("'", '"'))
        else:
            logger.error(f"ERROR: Invalid JSON string: {str(json_str)}...")
            assert False

def extract_key_pod_metrics(pod_metrics, pod_id):
    """Extract the most relevant metrics for a pod from the pod metrics"""
    if pod_id not in pod_metrics:
        logger.error(f"Error: Pod ID {pod_id} not found in pod metrics.")
        assert False
    return {
        'last_second_avg_ttft_ms': pod_metrics[pod_id]['last_second_avg_ttft_ms'],
        'last_second_avg_tpot_ms': pod_metrics[pod_id]['last_second_avg_tpot_ms'],
        'last_second_p99_ttft_ms': pod_metrics[pod_id]['last_second_p99_ttft_ms'],
        'last_second_p99_tpot_ms': pod_metrics[pod_id]['last_second_p99_tpot_ms'],
        'last_second_total_requests': pod_metrics[pod_id]['last_second_total_requests'],
        'last_second_total_tokens': pod_metrics[pod_id]['last_second_total_tokens'],
        'last_second_total_decode_tokens': pod_metrics[pod_id]['last_second_total_decode_tokens'],
        'last_second_total_prefill_tokens': pod_metrics[pod_id]['last_second_total_prefill_tokens'],
    }
    
def calculate_combined_rewards(ttft_rewards, tpot_rewards, ttft_reward_weight):
    return ttft_reward_weight*ttft_rewards + max(0, (1-ttft_reward_weight))*tpot_rewards

def calculate_rewards_simple(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight):
        ttft_rewards = np.where(
            ttft_values <= 0, 
            0.5,  # Maximum reward for perfect performance
            np.where(
                ttft_values <= ttft_slo,
                0.5 - (0.4 * ttft_values / ttft_slo),  # Linear scaling
                -0.1 - (0.4 * np.minimum(1.0, (ttft_values - ttft_slo) / ttft_slo))  # Negative reward
            )
        )

        tpot_rewards = np.where(
            tpot_values <= 0,
            -0.5,  # Penalize invalid values
            np.where(
                tpot_values <= avg_tpot_slo,
                0.1 + (0.4 * (1 - tpot_values / avg_tpot_slo)),  # Linear scaling
                -0.1 - (0.4 * np.minimum(1.0, (tpot_values - avg_tpot_slo) / avg_tpot_slo))  # Negative reward
            )
        )
        return {
            'ttft_rewards': ttft_rewards,
            'tpot_rewards': tpot_rewards,
            'combined_rewards': calculate_combined_rewards(ttft_rewards, tpot_rewards, ttft_reward_weight),
        }


def calculate_rewards_simple_extended(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight):
    """
    expressible TTFT range: 0-5000ms, 
    expressible TPOT: 0-200ms, 
    ttft reward range: -1.7 ~ 0.5
    ttft reward range: -1.3 ~ 0.5
    combined reward range: -3.0 ~ 1.0
    """
    ttft_rewards = np.where(
        ttft_values <= 0, 
        0.5,  # Maximum reward for perfect performance
        np.where(
            ttft_values <= ttft_slo,
            0.5 - (0.4 * ttft_values / ttft_slo),  # Linear scaling
            # CHANGE: 1.0 -> 4.0 for TTFT to cover 0-5000ms (5x SLO)
            -0.1 - (0.4 * np.minimum(4.0, (ttft_values - ttft_slo) / ttft_slo))  # Extended penalty
        )
    )

    tpot_rewards = np.where(
        tpot_values <= 0,
        -0.5,  # Penalize invalid values
        np.where(
            tpot_values <= avg_tpot_slo,
            0.1 + (0.4 * (1 - tpot_values / avg_tpot_slo)),  # Linear scaling
            # CHANGE: 1.0 -> 3.0 for TPOT to cover 0-200ms (4x SLO)  
            -0.1 - (0.4 * np.minimum(3.0, (tpot_values - avg_tpot_slo) / avg_tpot_slo))  # Extended penalty
        )
    )
    return {
        'ttft_rewards': ttft_rewards,
        'tpot_rewards': tpot_rewards, 
        'combined_rewards': calculate_combined_rewards(ttft_rewards, tpot_rewards, ttft_reward_weight),
    }
    
def calculate_rewards_piecewise_linear_steeper_gradient(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight):
    """
    Maximum performance reward function with steeper optimization gradient within the good performance region
    
    Expressible ranges:
    - TTFT: 0-5000ms 
    - TPOT: 0-200ms
    
    Reward ranges:
    - TTFT: -1.7 to +2.1 (range: 3.8)
    - TPOT: -1.3 to +1.7 (range: 3.0)  
    - Combined: -3.0 to +3.8
    
    This creates strong incentives for optimal performance while maintaining
    harsh penalties for SLO violations.
    """
    
    ttft_rewards = np.where(
        ttft_values <= 0, 
        2.1,  # Maximum reward for perfect TTFT
        np.where(
            ttft_values <= ttft_slo,
            # Balanced positive scaling: 0.5 to 2.1 (matches penalty range of 1.6)
            0.5 + (1.6 * (1 - ttft_values / ttft_slo)),  
            # Extended harsh penalty: -0.1 to -1.7
            -0.1 - (0.4 * np.minimum(4.0, (ttft_values - ttft_slo) / ttft_slo))
        )
    )

    tpot_rewards = np.where(
        tpot_values <= 0,
        -0.5,  # Still penalize invalid values
        np.where(
            tpot_values <= avg_tpot_slo,
            # Balanced positive scaling: 0.5 to 1.7 (matches penalty range of 1.2)  
            0.5 + (1.2 * (1 - tpot_values / avg_tpot_slo)),
            # Extended harsh penalty: -0.1 to -1.3
            -0.1 - (0.4 * np.minimum(3.0, (tpot_values - avg_tpot_slo) / avg_tpot_slo))
        )
    )
    
    return {
        'ttft_rewards': ttft_rewards,
        'tpot_rewards': tpot_rewards,
        'combined_rewards': calculate_combined_rewards(ttft_rewards, tpot_rewards, ttft_reward_weight),
    }



def calculate_rewards_latency_optimization(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight):
    """
    Latency optimization reward function that continuously rewards lower latency.
    
    This function provides meaningful reward differences across the entire range,
    encouraging the system to minimize latency even within SLO boundaries.
    
    Expressible ranges:
    - TTFT: 0-10000ms 
    - TPOT: 0-200ms
    
    Reward ranges:
    - TTFT: -2.0 to +2.0 (range: 4.0)
    - TPOT: -1.5 to +1.5 (range: 3.0)  
    - Combined: -3.5 to +3.5
    
    Key features:
    - Linear scaling across entire range for meaningful differences
    - Strong incentives for lower latency (0ms = max reward)
    - Harsh penalties for SLO violations
    - Continuous optimization signal
    """
    
    # TTFT rewards with linear scaling across entire range for latency optimization
    ttft_rewards = np.where(
        ttft_values <= 0, 
        2.0,  # Maximum reward for perfect TTFT
        np.where(
            ttft_values <= ttft_slo,
            # Linear scaling within SLO: 2.0 to 0.0 (meaningful differences)
            2.0 - (2.0 * ttft_values / ttft_slo),  
            # Extended harsh penalty for SLO violations: -0.5 to -2.0
            -0.5 - (0.4 * np.minimum(3.75, (ttft_values - ttft_slo) / ttft_slo))
        )
    )

    # TPOT rewards with linear scaling across entire range for latency optimization
    tpot_rewards = np.where(
        tpot_values <= 0,
        -0.5,  # Penalize invalid values
        np.where(
            tpot_values <= avg_tpot_slo,
            # Linear scaling within SLO: 1.5 to 0.0 (meaningful differences)
            1.5 - (1.5 * tpot_values / avg_tpot_slo),
            # Extended harsh penalty for SLO violations: -0.5 to -1.5
            -0.5 - (0.4 * np.minimum(2.5, (tpot_values - avg_tpot_slo) / avg_tpot_slo))
        )
    )
    
    return {
        'ttft_rewards': ttft_rewards,
        'tpot_rewards': tpot_rewards, 
        'combined_rewards': calculate_combined_rewards(ttft_rewards, tpot_rewards, ttft_reward_weight),
    }


## new - unified preprocessing function
def preprocess_data_unified(parsed_df, RL_MODEL_HYPERPARAMETERS, sorted_all_pod_ids, is_training):
    num_rows = len(parsed_df)
    processing_type = "batch" if num_rows > 1 else "single row"
    logger.debug(f"Processing {num_rows} rows ({processing_type}) with is_training={is_training}")
    
    # Pre-parse all JSON columns once to avoid repeated parsing
    json_columns = [
        'allPodsKvCacheHitRatios', 
        'numInflightRequestsAllPods', 
        'vllmGPUKVCacheUsage', 
        'vllmCPUKVCacheUsage', 
        'vllmNumRequestsRunning', 
        'vllmNumRequestsWaiting', 
        # 'podMetricsLastSecond',  # Made optional - will be handled separately
        'numPrefillTokensForAllPods', 
        'numDecodeTokensForAllPods',
        'GPU',  # GPU model mapping per pod (required for heterogeneous GPU support)
    ]
    
    json_parse_start_time = time.time()
    for col in json_columns:
        if col not in parsed_df.columns:
            logger.debug(f"Column {col} not in dataframe, skipping")
            continue
        try:
            sample_val = parsed_df[col].iloc[0]
            if isinstance(sample_val, str):
                logger.debug(f"Parsing JSON column: {col}")
                parsed_df[col] = parsed_df[col].apply(safe_parse_json)
                logger.debug(f"Successfully parsed {col}, sample: {parsed_df[col].iloc[0]}")
        except Exception as e:
            logger.error(f"Error parsing JSON column {col}: {e}")
            logger.error(f"Sample value: {sample_val}")
            raise
    
    # Handle podMetricsLastSecond separately (optional column)
    if 'podMetricsLastSecond' in parsed_df.columns:
        sample_val = parsed_df['podMetricsLastSecond'].iloc[0]
        if isinstance(sample_val, str):
            parsed_df['podMetricsLastSecond'] = parsed_df['podMetricsLastSecond'].apply(safe_parse_json)
        logger.info("Found podMetricsLastSecond column - will be ignored for feature extraction")
    else:
        logger.info("podMetricsLastSecond column not found - this is fine, features from this column are not used")
    
    json_parse_overhead = time.time() - json_parse_start_time

    # Collect all unique pod IDs in a single pass
    logger.debug("Collecting all unique pod IDs across the dataset...")
    logger.debug(f"Original dataset shape: {parsed_df.shape}")
    logger.debug(f"Columns: {parsed_df.columns.tolist()}")
    expected_columns = [
        'requestID',
        'selectedpod',
        'ttft',
        'avg_tpot',
        'total_decode_time',
        'e2e',
        'numInputTokens',
        # 'expectedNumOutputTokens',
        'numOutputTokens',
        'numTotalTokens',
        'request_start_time',  # NEW: Add request timing columns
        'request_end_time',    # NEW: Add request timing columns
        'allPodsKvCacheHitRatios',
        'numInflightRequestsAllPods',
        'vllmGPUKVCacheUsage',
        'vllmCPUKVCacheUsage',
        'vllmNumRequestsRunning',
        'vllmNumRequestsWaiting',
        # 'podMetricsLastSecond',  # Optional column - may be empty or missing
        'numPrefillTokensForAllPods',
        'numDecodeTokensForAllPods',
        'GPU',  # GPU model mapping per pod
        'subAlgorithm', # old training data does not have it... so...
        # 'prev_reward', ## uncomment it for scalable RL agent training
    ]
    
    ###########################################
    ## HARDCODE TEMPORARY FIX FOR OLD TRAINING DATA
    if 'subAlgorithm' not in parsed_df.columns:
        parsed_df['subAlgorithm'] = None
    ###########################################

    # if INCLUDE_GPU_IN_FEATURE:
    #     def get_gpu_model_encoded(selected_pod):
    #         selected_pod_generalpodid = RL_MODEL_HYPERPARAMETERS['pod_ip_to_generalpodid'][selected_pod]
    #         return RL_MODEL_HYPERPARAMETERS['pod_ip_to_gpu_model_encoded'][selected_pod_generalpodid]
    #     parsed_df['gpu_model_encoded'] = parsed_df['selectedpod'].apply(get_gpu_model_encoded)
    #     parsed_df['gpu_model_encoded'] = parsed_df['gpu_model_encoded'].astype(int)
    ## TODO: assume parsed_df has a column 'GPU'
    
    # Check for missing expected columns
    missing_columns = [col for col in expected_columns if col not in parsed_df.columns]
    if missing_columns:
        logger.error(f"Error: Missing expected columns: {missing_columns}")
        logger.error(f"parsed_df.columns: {parsed_df.columns}")
        logger.error(f"expected_columns: {expected_columns}")
        assert False
    
    # Check for unknown columns
    unknown_columns = [col for col in parsed_df.columns if col not in expected_columns]
    if unknown_columns:
        logger.warning(f"Warning: Unused columns: {unknown_columns}")

    numeric_conversion_start_time = time.time()
    # Convert string columns to appropriate types - vectorized
    numeric_columns = [
        'ttft',
        'avg_tpot',
        'total_decode_time',
        'e2e',
        'numInputTokens',
        # 'expectedNumOutputTokens',
        'numOutputTokens',
        'numTotalTokens',
        'request_start_time',
        'request_end_time',
        'prev_reward',
    ]
    
    for col in numeric_columns:
        if col in parsed_df.columns:
            parsed_df[col] = pd.to_numeric(parsed_df[col], errors='coerce')
    numeric_conversion_overhead = time.time() - numeric_conversion_start_time # 0-1ms
    
    # Vectorized processing using pandas operations
    logger.info("Processing records in vectorized manner...")
    
    get_value_start_time = time.time()
    # Extract base features
    base_data = {
        'request_id': parsed_df['requestID'].values,
        'selected_pod': parsed_df['selectedpod'].values,
        'input_tokens': parsed_df['numInputTokens'].values,
        # 'output_tokens': parsed_df['expectedNumOutputTokens'].values,
        'output_tokens': parsed_df['numOutputTokens'].values,
        'total_tokens': parsed_df['numTotalTokens'].values,
        'ttft': parsed_df['ttft'].values,
        'avg_tpot': parsed_df['avg_tpot'].values,
        'e2e_latency': parsed_df['e2e'].values,
        'request_start_time': parsed_df['request_start_time'].values,
        'request_end_time': parsed_df['request_end_time'].values, 
        'subAlgorithm': parsed_df['subAlgorithm'].values,
        # 'prev_reward': parsed_df['prev_reward'].values,
    }
    # if INCLUDE_GPU_IN_FEATURE:
    #     base_data['gpu_model_encoded'] = parsed_df['gpu_model_encoded'].values
    #     # Fix 2: Use proper GPU mapping instead of hardcoding
    #     if 'pod_gpu_mapping' not in RL_MODEL_HYPERPARAMETERS:
    #         logger.error("Error: pod_gpu_mapping not found in RL_MODEL_HYPERPARAMETERS")
    #         assert False
    #     pod_gpu_models = {pod_id: RL_MODEL_HYPERPARAMETERS['pod_gpu_mapping'][pod_id] for pod_id in sorted_all_pod_ids}
    
    # Pre-extract all JSON data to avoid repeated parsing
    all_kv_cache = parsed_df['allPodsKvCacheHitRatios'].values
    all_inflight = parsed_df['numInflightRequestsAllPods'].values  
    all_gpu_cache = parsed_df['vllmGPUKVCacheUsage'].values
    all_cpu_cache = parsed_df['vllmCPUKVCacheUsage'].values
    all_running = parsed_df['vllmNumRequestsRunning'].values
    all_waiting = parsed_df['vllmNumRequestsWaiting'].values
    all_prefill = parsed_df['numPrefillTokensForAllPods'].values
    all_decode = parsed_df['numDecodeTokensForAllPods'].values
    all_gpu = parsed_df['GPU'].values
    # NOTE: podMetricsLastSecond features are not used in training anymore
    # all_pod_metrics = parsed_df['podMetricsLastSecond'].values
    
    # Process pod features for all rows at once
    excluded_pod_features = set(RL_MODEL_HYPERPARAMETERS.get('EXCLUDED_POD_FEATURES', []))
    if 'none' in excluded_pod_features or 'None' in excluded_pod_features:
        excluded_pod_features = set()

    for pod_id in sorted_all_pod_ids:
        # Vectorized extraction for each pod across all rows
        if 'kv_hit_ratio' not in excluded_pod_features:
            base_data[f"{pod_id}-kv_hit_ratio"] = [data.get(pod_id, 0) for data in all_kv_cache]
        if 'inflight_requests' not in excluded_pod_features:
            base_data[f"{pod_id}-inflight_requests"] = [data.get(pod_id, 0) for data in all_inflight]
        if 'gpu_kv_cache' not in excluded_pod_features:
            base_data[f"{pod_id}-gpu_kv_cache"] = [data.get(pod_id, 0) for data in all_gpu_cache]
        if 'cpu_kv_cache' not in excluded_pod_features:
            base_data[f"{pod_id}-cpu_kv_cache"] = [data.get(pod_id, 0) for data in all_cpu_cache]
        if 'running_requests' not in excluded_pod_features:
            base_data[f"{pod_id}-running_requests"] = [data.get(pod_id, 0) for data in all_running]
        if 'waiting_requests' not in excluded_pod_features:
            base_data[f"{pod_id}-waiting_requests"] = [data.get(pod_id, 0) for data in all_waiting]
        if 'prefill_tokens' not in excluded_pod_features:
            base_data[f"{pod_id}-prefill_tokens"] = [data.get(pod_id, 0) for data in all_prefill]
        if 'decode_tokens' not in excluded_pod_features:
            base_data[f"{pod_id}-decode_tokens"] = [data.get(pod_id, 0) for data in all_decode]
        if 'GPU' not in excluded_pod_features:
            try:
                base_data[f"{pod_id}-GPU"] = [data.get(pod_id, 0) for data in all_gpu]
            except Exception as e:
                logger.error(f"all_gpu: {all_gpu}")
                logger.error(f"pasred_df.iloc[0]: {parsed_df.iloc[0]}")
                logger.error(f"Error: {e}")
                assert False
        
        # if INCLUDE_GPU_IN_FEATURE:
        #     if pod_id not in RL_MODEL_HYPERPARAMETERS['pod_gpu_mapping']:
        #         logger.error(f"Error: Pod ID {pod_id} not found in RL_MODEL_HYPERPARAMETERS['pod_gpu_mapping']")
        #         assert False
        #     gpu_model = RL_MODEL_HYPERPARAMETERS['pod_gpu_mapping'][pod_id]
        #     base_data[f"{pod_id}-gpu_model"] = [gpu_model] * len(parsed_df)
    get_value_overhead = time.time() - get_value_start_time # 0ms
    num_rows = len(base_data['request_id'])
    pod_index_start_time = time.time()
    if is_training:
        unique_pods = np.unique(base_data['selected_pod'])
        pod_to_index = {str(pod): idx for idx, pod in enumerate(unique_pods)}
        index_to_pod = {int(idx): str(pod) for pod, idx in pod_to_index.items()}
        selected_pods_array = np.array(base_data['selected_pod'])
        action_values = np.array([pod_to_index[str(pod)] for pod in selected_pods_array])
    else:
        pod_to_index = {}
        index_to_pod = {}
        action_values = None

    ttft_values = np.array(base_data['ttft'], dtype=np.float64)
    tpot_values = np.array(base_data['avg_tpot'], dtype=np.float64)
    pod_index_overhead = time.time() - pod_index_start_time
        
    # Training-specific calculations (rewards and action mapping)
    if is_training:
        # Calculate rewards for training
        ttft_slo = RL_MODEL_HYPERPARAMETERS['TTFT_SLO']
        avg_tpot_slo = RL_MODEL_HYPERPARAMETERS['AVG_TPOT_SLO']
        ttft_reward_weight = RL_MODEL_HYPERPARAMETERS['TTFT_REWARD_WEIGHT']
        if RL_MODEL_HYPERPARAMETERS['REWARD_FUNCTION'] == "linear_simple":
            reward = calculate_rewards_simple(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight)
        elif RL_MODEL_HYPERPARAMETERS['REWARD_FUNCTION'] == "linear_simple_extended":
            reward = calculate_rewards_simple_extended(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight)
        elif RL_MODEL_HYPERPARAMETERS['REWARD_FUNCTION'] == "piecewise_linear_steeper_gradient":
            reward = calculate_rewards_piecewise_linear_steeper_gradient(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight)
        elif RL_MODEL_HYPERPARAMETERS['REWARD_FUNCTION'] == "latency_optimized":
            reward = calculate_rewards_latency_optimization(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight)
        else:
            logger.error(f"Unknown reward function: {RL_MODEL_HYPERPARAMETERS['REWARD_FUNCTION']}")
            assert False
        # Add training-specific columns
        base_data.update({
            'action': action_values,
            'avg_tpot_slo_satisfied': tpot_values <= RL_MODEL_HYPERPARAMETERS['AVG_TPOT_SLO'],
            'avg_ttft_slo_satisfied': ttft_values <= RL_MODEL_HYPERPARAMETERS['TTFT_SLO'],
            'ttft_reward': reward['ttft_rewards'],
            'tpot_reward': reward['tpot_rewards'],
            'reward': reward['combined_rewards'],
        })
    create_df_start_time = time.time()
    processed_df = pd.DataFrame(base_data)
    create_df_overhead = time.time() - create_df_start_time

    # Replace fillna(0) with a more targeted approach since most values should already be handled
    # Only fill NaN values in specific columns that might have them
    nan_columns = processed_df.columns[processed_df.isnull().any()].tolist()
    if nan_columns:
        processed_df[nan_columns] = processed_df[nan_columns].fillna(0)

    logger.debug(f"Processed dataset shape: {processed_df.shape}")
    logger.debug(f"Processed columns: {processed_df.columns[:10].tolist()}...")

    # Prepare overhead summary
    preprocess_overhead_summary = {
        'json_parse_overhead': json_parse_overhead,
        'column_check_overhead': -1,
        'podmetrics_parse_overhead': -1,
        'numeric_conversion_overhead': numeric_conversion_overhead,
        'get_value_overhead': get_value_overhead,
        'create_df_overhead': create_df_overhead,
        'pod_index_overhead': pod_index_overhead,
        'reward_calc_overhead': -1,
        'slo_update_overhead': -1,
    }
    
    # if is_training:
    #     # Training mode: return mapping info for action space creation
    #     if INCLUDE_GPU_IN_FEATURE:
    #         mapping_info = {
    #             'pod_to_index': pod_to_index,
    #             'index_to_pod': index_to_pod,
    #         }
    #         mapping_info['pod_gpu_models'] = pod_gpu_models
    #         logger.debug("\nPod GPU model mapping:")
    #         for pod_id, gpu_model in pod_gpu_models.items():
    #             logger.debug(f"  Pod {pod_id} -> GPU model {gpu_model}")
        
    return processed_df, sorted_all_pod_ids, preprocess_overhead_summary
    # else:
    #     # Inference mode: simplified return for speed
    #     return processed_df, sorted_all_pod_ids, preprocess_overhead_summary


def parse_log_message(log_message):
    # logger.info(f"log_message: {log_message}")
    if "latency_metrics" not in log_message:
        logger.error(f"Invalid line. {log_message}")
        return pd.DataFrame(), []
    # Find start position more efficiently
    start_idx = log_message.find("latency_metrics@") + 16
    if start_idx == 15:  # find returned -1
        return pd.DataFrame(), []
    # Split only the relevant part
    parts = log_message[start_idx:].split('@')
    row = {}
    json_columns = []
    # Process pairs directly
    i = 0
    while i < len(parts) - 1:
        key = parts[i]
        value = parts[i + 1]
        # Fast JSON detection and parsing
        if value and value[0] == '{' and value[-1] == '}':
            try:
                # Only fix quotes if needed
                if '\\"' in value:
                    value = value.replace('\\"', '"')
                row[key] = json.loads(value)
                json_columns.append(key)
            except Exception as e:
                logger.error(f"Error decoding JSON, column: {key}, value: {value}")
                logger.error(f"Error: {e}")
                row[key] = value
        else:
            # Fast type conversion with better float detection
            if value.isdigit():
                row[key] = int(value)
            elif value.replace('.', '').replace('-', '').isdigit() and value.count('.') == 1:
                # Only convert to float if there's exactly one decimal point
                row[key] = float(value)
            else:
                row[key] = value
        i += 2
    # Create DataFrame only if we have data
    if row:
        df = pd.DataFrame([row])
        # logger.info(f"df: {df.to_csv(index=False)}")
        return df, json_columns
    else:
        return pd.DataFrame(), []


def main(input_file, log_message, RL_MODEL_HYPERPARAMETERS):
    preprocess_dataset_overhead_summary = {}
    if input_file == None and (log_message == "" or log_message is None):
        logger.error("Error: Both input_file and log_message are empty or None.")
        assert False
    if input_file is not None and log_message != "":
        logger.error("Error: Both input_file and log_message are provided. Please provide only one.")
        assert False
    if input_file is not None:  # Training path
        parsed_df, json_columns = parse_log_file(input_file)
    else:  # Inference path
        parse_start_time = time.time()
        parsed_df, _ = parse_log_message(log_message)
        preprocess_dataset_overhead_summary["parse_log_message"] = time.time() - parse_start_time
    if len(parsed_df) == 0:
        logger.error("No data found after parsing JSON columns.")
        logger.error(f"Log message: {log_message}")
        assert False
    
    # Unified preprocessing for both single row (inference) and batch (training)
    sorted_all_pod_ids = utils.get_sorted_all_pod_ids('batch_dataframe', parsed_df)
    if len(parsed_df) == 1 and input_file is None:
        # Inference mode: single row, no training-specific features needed
        preprocess_start_time = time.time()
        is_training = False
        processed_df, sorted_all_pod_ids, preprocess_dataset_overhead_summary = preprocess_data_unified(parsed_df, RL_MODEL_HYPERPARAMETERS, sorted_all_pod_ids, is_training)
        preprocess_dataset_overhead_summary["preprocess_unified_inference"] = time.time() - preprocess_start_time
        mapping_info = None  # No mapping info needed for inference
    else:
        # Training mode: batch processing with full features
        preprocess_start_time = time.time()
        is_training = True
        processed_df, sorted_all_pod_ids, preprocess_dataset_overhead_summary = preprocess_data_unified(parsed_df, RL_MODEL_HYPERPARAMETERS, sorted_all_pod_ids, is_training)
        preprocess_dataset_overhead_summary["preprocess_unified_training"] = time.time() - preprocess_start_time
    return processed_df, sorted_all_pod_ids, preprocess_dataset_overhead_summary
