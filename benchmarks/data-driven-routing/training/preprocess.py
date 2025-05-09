import pandas as pd
import numpy as np
import json
import ast
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import argparse
import sys


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
                column_name = parts[i]
                column_names.append(column_name)
                value = parts[i+1]
                if value.startswith('{') and value.endswith('}'):
                    try:
                        json_columns.append(column_name)
                        row[column_name] = json.loads(value) # this is going to be dictionary
                    except json.JSONDecodeError:
                        logger.error(f"Error decoding JSON: {value}")
                else:
                    try:
                        row[column_name] = int(value)
                    except ValueError:
                        try:
                            row[column_name] = float(value)
                        except ValueError:
                            row[column_name] = value
            data.append(row)
    df = pd.DataFrame(data, columns=column_names)
    if len(df) == 0:
        logger.error("No data found in the log file.")
        assert False
    return df, json_columns

def normalize_time(df):
    cutoff_time=0
    first_request_start_time = df['request_start_time'].min()
    df['normalized_start_time'] = df['request_start_time'] - first_request_start_time
    df['normalized_end_time'] = df['request_end_time'] - first_request_start_time
    df['normalized_start_time'] /= 1_000_000
    df['normalized_end_time'] /= 1_000_000
    df['log_window_start_time'] = df['log_window_start_time'] - first_request_start_time
    df['log_window_start_time'] /= 1_000_000
    df['log_window_end_time'] = df['log_window_end_time'] - first_request_start_time
    df['log_window_end_time'] /= 1_000_000
    df = df[df['normalized_start_time'] > cutoff_time]
    # df['normalized_start_time'] = df['normalized_start_time'] - df['normalized_start_time'].min()
    df.loc[:, 'normalized_start_time'] = df['normalized_start_time'] - df['normalized_start_time'].min()
    # df['normalized_end_time'] = df['normalized_end_time'] - df['normalized_start_time'].min()
    df.loc[:, 'normalized_end_time'] = df['normalized_end_time'] - df['normalized_start_time'].min()
    df = df.sort_values(by='normalized_start_time', ascending=True)
    df['time_bucket'] = df['normalized_start_time'].astype(int)
    df = df[['normalized_start_time', 'time_bucket', 'normalized_end_time'] + [col for col in df.columns if col != 'normalized_start_time' and col != 'normalized_end_time' and col != 'time_bucket']]
    df.reset_index(drop=True, inplace=True)
    return df

def safe_parse_json(json_str):
    """Safely parse Python dictionary-like strings or JSON strings"""
    # If already a dictionary, return as is
    if isinstance(json_str, dict):
        return json_str
        
    if pd.isna(json_str) or not json_str:
        return {}
    
    try:
        # Try standard JSON parsing
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        try:
            # Try replacing single quotes with double quotes
            if isinstance(json_str, str):
                return json.loads(json_str.replace("'", '"'))
            else:
                return {}
        except (json.JSONDecodeError, TypeError):
            try:
                # Try using ast.literal_eval for Python dict literals
                if isinstance(json_str, str):
                    return ast.literal_eval(json_str)
                else:
                    return {}
            except (SyntaxError, ValueError, TypeError):
                print(f"Warning: Could not parse JSON: {str(json_str)[:50]}...")
                return {}

def calculate_slo_satisfaction_for_avg_tpot(row, avg_tpot_slo_threshold=25):
    """
    Calculate if SLO was satisfied based on average tokens per second (TPOT)
    
    Args:
        row: DataFrame row
        avg_tpot_slo_threshold: SLO threshold in milliseconds
    
    Returns:
        True if SLO was satisfied, False otherwise
    """
    try:
        avg_tpot = float(row['avg_tpot'])
        return avg_tpot <= avg_tpot_slo_threshold
    except (ValueError, TypeError):
        return False

def calculate_slo_satisfaction_for_avg_ttft(row, avg_ttft_slo_threshold=500):
    """
    Calculate if SLO was satisfied based on average time to first token (TTFT)
    
    Args:
        row: DataFrame row
        avg_ttft_slo_threshold: SLO threshold in milliseconds
    
    Returns:
        True if SLO was satisfied, False otherwise
    """
    try:
        ttft = float(row['ttft'])
        return ttft <= avg_ttft_slo_threshold
    except (ValueError, TypeError):
        return False

def calculate_ttft_reward(row, ttft_slo_threshold=500):
    """
    Calculate sophisticated reward for TTFT (time to first token)
    
    Args:
        row: DataFrame row with 'ttft' value
        ttft_slo_threshold: SLO threshold for TTFT in milliseconds
    
    Returns:
        TTFT reward component
    """
    try:
        ttft = float(row['ttft'])
        
        if ttft <= 0:
            return 0.5  # Maximum reward for perfect performance
        elif ttft <= ttft_slo_threshold:
            # Linear scaling from 0.5 (best) to 0.1 (at threshold)
            return 0.5 - (0.4 * ttft / ttft_slo_threshold)
        else:
            # Negative reward scaling with how much it exceeds threshold
            excess_factor = min(1.0, (ttft - ttft_slo_threshold) / ttft_slo_threshold)
            return -0.1 - (0.4 * excess_factor)
    except (ValueError, TypeError, ZeroDivisionError):
        return -0.5  # Default penalty for invalid data

def calculate_tpot_reward(row, avg_tpot_slo_threshold=25):
    """
    Calculate sophisticated reward for TPOT (tokens per output time)
    
    Args:
        row: DataFrame row with 'avg_tpot' value
        avg_tpot_slo_threshold: SLO threshold for TPOT in milliseconds
    
    Returns:
        TPOT reward component
    """
    try:
        avg_tpot = float(row['avg_tpot'])
        
        if avg_tpot <= 0:
            return -0.5  # Penalize invalid values
        elif avg_tpot <= avg_tpot_slo_threshold:
            # Linear scaling from 0.5 (best) to 0.1 (at threshold)
            return 0.1 + (0.4 * (1 - avg_tpot / avg_tpot_slo_threshold))
        else:
            # Negative reward scaling with how much it exceeds threshold
            excess_factor = min(1.0, (avg_tpot - avg_tpot_slo_threshold) / avg_tpot_slo_threshold)
            return -0.1 - (0.4 * excess_factor)
    except (ValueError, TypeError, ZeroDivisionError):
        return -0.5  # Default penalty for invalid data


def calculate_sophisticated_reward(row, ttft_slo_threshold=500, avg_tpot_slo_threshold=25):
    """
    Calculate a sophisticated reward based on how close/far metrics are from their SLO thresholds
    
    Args:
        row: DataFrame row with 'ttft' and 'avg_tpot' values
        ttft_slo_threshold: SLO threshold for time to first token in milliseconds
        avg_tpot_slo_threshold: SLO threshold for average tokens per second in milliseconds
    
    Returns:
        Combined reward value
    """
    try:
        # Get the metrics
        ttft = float(row['ttft'])
        avg_tpot = float(row['avg_tpot'])
        
        # Calculate TTFT reward component (higher is better)
        # If TTFT is 0 or less, max reward
        # If TTFT is at the threshold, small positive reward
        # If TTFT exceeds threshold, negative reward scaling with how much it exceeds
        if ttft <= 0:
            ttft_reward = 0.5
        elif ttft <= ttft_slo_threshold:
            # Linear scaling from 0.5 (best) to 0.1 (at threshold)
            ttft_reward = 0.5 - (0.4 * ttft / ttft_slo_threshold)
        else:
            # Negative reward scaling with how much it exceeds threshold
            # -0.1 at threshold, down to -0.5 at 2x threshold or worse
            excess_factor = min(1.0, (ttft - ttft_slo_threshold) / ttft_slo_threshold)
            ttft_reward = -0.1 - (0.4 * excess_factor)
        
        # Calculate TPOT reward component (lower is better for latency)
        # If TPOT is 0, it's an error case, so penalize
        # If TPOT is below threshold, positive reward scaling with how far below
        # If TPOT exceeds threshold, negative reward scaling with how much it exceeds
        if avg_tpot <= 0:
            tpot_reward = -0.5  # Penalize invalid values
        elif avg_tpot <= avg_tpot_slo_threshold:
            # Linear scaling from 0.5 (best) to 0.1 (at threshold)
            # Note: Lower TPOT is better (faster token generation)
            tpot_reward = 0.1 + (0.4 * (1 - avg_tpot / avg_tpot_slo_threshold))
        else:
            # Negative reward scaling with how much it exceeds threshold
            # -0.1 at threshold, down to -0.5 at 2x threshold or worse
            excess_factor = min(1.0, (avg_tpot - avg_tpot_slo_threshold) / avg_tpot_slo_threshold)
            tpot_reward = -0.1 - (0.4 * excess_factor)
        
        # Combine rewards with equal weight
        combined_reward = ttft_reward + tpot_reward
        
        return combined_reward
    
    except (ValueError, TypeError, ZeroDivisionError):
        return -1.0  # Default penalty for invalid data

def calculate_reward(row):
    """
    Calculate reward based on multiple SLO satisfaction metrics and latency
    
    Args:
        row: DataFrame row with SLO satisfaction flags and latency metrics
    
    Returns:
        Reward value
    """
    # Base rewards for each SLO type
    tpot_reward = 0.5 if row['avg_tpot_slo_satisfied'] else -0.5
    ttft_reward = 0.5 if row['avg_ttft_slo_satisfied'] else -0.5
    
    # Combined base reward
    base_reward = tpot_reward + ttft_reward
    
    # Latency component: normalize between 0 and 1, where lower is better
    try:
        # TTFT latency component
        ttft = float(row['ttft'])
        max_ttft = 2000  # Adjust based on typical range
        normalized_ttft = max(0, min(1, 1.0 - ttft / max_ttft))

        # TPOT latency component
        avg_tpot = float(row['avg_tpot'])
        max_tpot = 100  # Adjust based on typical range
        normalized_tpot = max(0, min(1, 1.0 - avg_tpot / max_tpot))
        
        # Weighted latency reward
        latency_reward = 0.5 * normalized_ttft + 0.5 * normalized_tpot
    except (ValueError, TypeError, ZeroDivisionError):
        latency_reward = 0.0
    
    return base_reward + latency_reward
        

def extract_key_pod_metrics(pod_metrics, pod_id):
    """Extract the most relevant metrics for a pod from the pod metrics"""
    if pod_id not in pod_metrics:
        print(f"Error: Pod ID {pod_id} not found in pod metrics.")
        assert False
        # return {
        #     'last_second_avg_ttft_ms': 0,
        #     'last_second_avg_tpot_ms': 0, 
        #     'last_second_p99_ttft_ms': 0,
        #     'last_second_p99_tpot_ms': 0,
        #     'last_second_total_requests': 0,
        #     'last_second_total_tokens': 0
        # }
    print(f"pod_metrics[{pod_id}]: {pod_metrics[pod_id].keys()}")
    return {
        'last_second_avg_ttft_ms': pod_metrics[pod_id]['last_second_avg_ttft_ms'],
        'last_second_avg_tpot_ms': pod_metrics[pod_id]['last_second_avg_tpot_ms'],
        'last_second_p99_ttft_ms': pod_metrics[pod_id]['last_second_p99_ttft_ms'],
        'last_second_p99_tpot_ms': pod_metrics[pod_id]['last_second_p99_tpot_ms'],
        'last_second_total_requests': pod_metrics[pod_id]['last_second_total_requests'],
        'last_second_total_tokens': pod_metrics[pod_id]['last_second_total_tokens'],
    }

def preprocess_dataset(df, output_file=None):
    all_pods_set = set()
    print("Collecting all unique pod IDs across the dataset...")
    for _, row in df.iterrows():
        kv_cache_hit_ratios = safe_parse_json(row['allPodsKvCacheHitRatios'])
        if kv_cache_hit_ratios:
            all_pods_set.update(kv_cache_hit_ratios.keys())
        inflight_requests = safe_parse_json(row['numInflightRequestsAllPods'])
        if inflight_requests:
            all_pods_set.update(inflight_requests.keys())
        pod_metrics = safe_parse_json(row['podMetricsLastSecond'])
        if pod_metrics:
            all_pods_set.update(pod_metrics.keys())
        if len(all_pods_set) == 8:
            print(f"Found all EIGHT pods: {all_pods_set}. break")
            break
    all_pods = list(all_pods_set)
    print(f"Identified {len(all_pods)} pods: {all_pods}")


    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    expected_columns = [
        'normalized_start_time', 'time_bucket', 'normalized_end_time', 
        'requestID', 'request_start_time', 'request_end_time', 
        'selectedpod', 'ttft', 'avg_tpot', 'total_decode_time', 'e2e',
        'numInputTokens', 'numOutputTokens', 'numTotalTokens',
        'allPodsKvCacheHitRatios', 'numInflightRequestsAllPods',
        'vllmGPUKVCacheUsage', 'vllmCPUKVCacheUsage',
        'vllmNumRequestsRunning', 'vllmNumRequestsWaiting',
        'podMetricsLastSecond', 'log_window_start_time',
        'log_window_end_time', 'numPrefillTokensForAllPods',
        'numDecodeTokensForAllPods'
    ]

    expected_last_second_pod_metrics_keys = [
        'last_second_avg_ttft_ms', 
        'last_second_min_ttft_ms', 
        'last_second_max_ttft_ms', 
        'last_second_p50_ttft_ms', 
        'last_second_p90_ttft_ms', 
        'last_second_p95_ttft_ms', 
        'last_second_p99_ttft_ms', 
        'last_second_ttft_samples', 
        'last_second_avg_tpot_ms', 
        'last_second_min_tpot_ms', 
        'last_second_max_tpot_ms', 
        'last_second_p50_tpot_ms', 
        'last_second_p90_tpot_ms', 
        'last_second_p95_tpot_ms', 
        'last_second_p99_tpot_ms', 
        'last_second_tpot_samples', 
        'last_second_early_tokens_tpot_ms', 
        'last_second_mid_tokens_tpot_ms', 
        'last_second_late_tokens_tpot_ms', 
        'last_second_total_requests', 
        'last_second_total_decode_tokens', 
        'last_second_total_prefill_tokens', 
        'last_second_total_tokens'
    ]
    
    # Check for missing expected columns
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing expected columns: {missing_columns}")
        assert False
    
    # Check for unknown columns
    unknown_columns = [col for col in df.columns if col not in expected_columns]
    if unknown_columns:
        print(f"Error: Found unknown columns: {unknown_columns}")
        assert False

    # Process first row to check podMetricsLastSecond structure
    if 'podMetricsLastSecond' in df.columns and len(df) > 0:
        first_row = df.iloc[0]
        print(f"WARNING: We are using the first row only to check podMetricsLastSecond structure")
        pod_metrics = safe_parse_json(first_row['podMetricsLastSecond'])
        # print(f"features in pod_metrics: {pod_metrics.keys()}")
        print(f"features in pod_metrics: {pod_metrics[list(pod_metrics.keys())[0]].keys()}")
        if pod_metrics:
            # Check structure for each pod
            for pod_id, metrics in pod_metrics.items():
                # print(f"Checking metrics for pod {pod_id}")
                print(f"metrics: {metrics}")
                # Check for missing expected keys
                missing_keys = [key for key in expected_last_second_pod_metrics_keys if key not in metrics]
                if missing_keys:
                    print(f"Error: Missing expected keys in podMetricsLastSecond for pod {pod_id}: {missing_keys}")
                    assert False
                
                # Check for unknown keys
                unknown_keys = [key for key in metrics.keys() if key not in expected_last_second_pod_metrics_keys]
                if unknown_keys:
                    print(f"Error: Found unknown keys in podMetricsLastSecond for pod {pod_id}: {unknown_keys}")
                    assert False
    
    
    # Convert string columns to appropriate types
    numeric_columns = ['normalized_start_time', 'normalized_end_time', 'ttft', 
                      'avg_tpot', 'total_decode_time', 'e2e', 
                      'numInputTokens', 'numOutputTokens', 'numTotalTokens']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Parse JSON fields if they are strings
    json_columns = ['allPodsKvCacheHitRatios', 'numInflightRequestsAllPods', 
                   'vllmGPUKVCacheUsage', 'vllmCPUKVCacheUsage', 
                   'vllmNumRequestsRunning', 'vllmNumRequestsWaiting', 
                   'podMetricsLastSecond', 'numPrefillTokensForAllPods', 'numDecodeTokensForAllPods']
    
    for col in json_columns:
        if col in df.columns:
            # print(f"Processing column: {col}")
            sample_val = df[col].iloc[0]
            if isinstance(sample_val, str):
                # print(f"  Parsing as JSON string")
                df[col] = df[col].apply(safe_parse_json)
            elif isinstance(sample_val, dict):
                print(f"  Already parsed as dictionary")
            else:
                print(f"  Unknown type: {type(sample_val)}")
                # Try to parse anyway
                df[col] = df[col].apply(safe_parse_json)
    
    # Create a new list to store processed records
    processed_records = []
    
    pod_gpu_models = {pod_id: "NVIDIA-L20" for pod_id in all_pods}
    for _, row in df.iterrows():
        base_features = {
            'request_id': row['requestID'],
            'request_start_time': row['request_start_time'],
            'request_end_time': row['request_end_time'],
            'selected_pod': row['selectedpod'],
            'input_tokens': row['numInputTokens'],
            'output_tokens': row['numOutputTokens'],
            'total_tokens': row['numTotalTokens'],
            'ttft': row['ttft'],
            'avg_tpot': row['avg_tpot'],
            'e2e_latency': row['e2e'],
        }
        pod_metrics = safe_parse_json(row['podMetricsLastSecond'])
        
        # Here's the fixed part - getting pod-specific dictionaries for each metric
        all_pods_kv_cache_hit_ratios = safe_parse_json(row['allPodsKvCacheHitRatios'])
        all_pods_inflight_requests = safe_parse_json(row['numInflightRequestsAllPods'])
        all_pods_gpu_kv_cache = safe_parse_json(row['vllmGPUKVCacheUsage'])
        all_pods_cpu_kv_cache = safe_parse_json(row['vllmCPUKVCacheUsage'])
        all_pods_running_requests = safe_parse_json(row['vllmNumRequestsRunning'])
        all_pods_waiting_requests = safe_parse_json(row['vllmNumRequestsWaiting'])
        all_pods_prefill_tokens = safe_parse_json(row['numPrefillTokensForAllPods'])
        all_pods_decode_tokens = safe_parse_json(row['numDecodeTokensForAllPods'])
        
        # Get pod-specific features for each pod
        pod_features = {}
        for pod_id in all_pods:
            # Now extract just the value for this pod from each dictionary
            pod_features[pod_id] = {
                'kv_hit_ratio': all_pods_kv_cache_hit_ratios.get(pod_id, 0),
                'inflight_requests': all_pods_inflight_requests.get(pod_id, 0),
                'gpu_kv_cache': all_pods_gpu_kv_cache.get(pod_id, 0),
                'cpu_kv_cache': all_pods_cpu_kv_cache.get(pod_id, 0),
                'running_requests': all_pods_running_requests.get(pod_id, 0),
                'waiting_requests': all_pods_waiting_requests.get(pod_id, 0),
                'prefill_tokens': all_pods_prefill_tokens.get(pod_id, 0),
                'decode_tokens': all_pods_decode_tokens.get(pod_id, 0),
                'gpu_model': pod_gpu_models[pod_id],
            }
            
            # Add key metrics from pod_metrics
            key_metrics = extract_key_pod_metrics(pod_metrics, pod_id)
            pod_features[pod_id].update(key_metrics)
        
        # Create a flattened record with all features
        record = base_features.copy()
        
        # Add pod features with prefixed column names
        for pod_id, features in pod_features.items():
            pod_prefix = f"pod_{pod_id.replace('.', '_')}"
            for feature_name, feature_value in features.items():
                record[f"{pod_prefix}-{feature_name}"] = feature_value
        
        processed_records.append(record)
    
    # Create a new DataFrame with processed records
    processed_df = pd.DataFrame(processed_records)
    
    all_pod_ids = [pod_id.replace('.', '_') for pod_id in all_pods]
    # processed_df = create_essential_relative_features(
    #     processed_df,
    #     all_pod_ids,
    #     drop_raw=False,
    #     drop_pct=True,
    #     drop_rank=True,
    #     drop_norm_rank=True,
    # )

    # Remove rows with NaN values or replace them with default values
    processed_df = processed_df.fillna(0)
    
    # Map pod IDs to integer indices for the action space
    unique_pods = processed_df['selected_pod'].unique()
    pod_to_index = {pod: idx for idx, pod in enumerate(unique_pods)}
    
    # Add the action column (the pod index)
    processed_df['action'] = processed_df['selected_pod'].map(pod_to_index)

    processed_df['avg_tpot_slo_satisfied'] = processed_df['avg_tpot'].apply(
        lambda x: x <= avg_tpot_slo_threshold)
    processed_df['avg_ttft_slo_satisfied'] = processed_df['ttft'].apply(
        lambda x: x <= avg_ttft_slo_threshold)

    processed_df['ttft_reward'] = processed_df.apply(
        lambda row: calculate_ttft_reward(row, ttft_slo_threshold=avg_ttft_slo_threshold), axis=1)
    
    processed_df['tpot_reward'] = processed_df.apply(
        lambda row: calculate_tpot_reward(row, avg_tpot_slo_threshold=avg_tpot_slo_threshold), axis=1)
    
    # Combined reward
    processed_df['reward'] = processed_df['ttft_reward'] + processed_df['tpot_reward']
    
    # Add normalized metrics for analysis
    processed_df['ttft_normalized'] = processed_df['ttft'].apply(
        lambda x: min(1.0, max(0.0, float(x) / 500)) if x > 0 else 0)
    
    processed_df['tpot_normalized'] = processed_df['avg_tpot'].apply(
        lambda x: min(1.0, max(0.0, float(x) / 25)) if x > 0 else 0)

    # Save mapping information
    mapping_info = {
        'pod_to_index': pod_to_index,
        'index_to_pod': {idx: pod for pod, idx in pod_to_index.items()},
        'pod_gpu_models': pod_gpu_models,
    }
    
    # Save the processed dataset
    print(f"Saving processed dataset to {output_file}...")
    processed_df.to_csv(output_file, index=False)
    
    # Save mapping information
    mapping_file = output_file.replace('.csv', '_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(mapping_info, f, indent=2)
    
    print(f"Processed dataset shape: {processed_df.shape}")
    print(f"Processed columns: {processed_df.columns[:10].tolist()}...")
    print(f"Mapping information saved to {mapping_file}")

    # Print GPU model mapping
    print("\nPod GPU model mapping:")
    for pod_id, gpu_model in pod_gpu_models.items():
        print(f"  Pod {pod_id} -> GPU model {gpu_model}")
    
    return processed_df, mapping_info


## NOTE: create_essential_relative_features is done in training2.py
# def create_essential_relative_features(df, pod_ids, drop_raw, drop_pct, drop_rank, drop_norm_rank, metrics=None):
#     if metrics is None:
#         # Detect metrics by finding columns that appear for multiple pods
#         all_cols = set(df.columns)
#         metrics = set()
        
#         for pod_id in pod_ids:
#             pod_prefix = f"pod_{pod_id}"
#             pod_cols = [col[len(pod_prefix)+1:] for col in all_cols if col.startswith(pod_prefix)]
            
#             if not metrics:
#                 metrics = set(pod_cols)
#             else:
#                 metrics = metrics.intersection(pod_cols)
        
#         metrics = list(metrics)
#         print(f"Detected {len(metrics)} common metrics across pods: {metrics}")
    
#     # Track columns to potentially drop
#     columns_to_drop = []
    
#     # Create dictionaries to hold new columns
#     new_columns = {}
    
#     cluster_wise_total_features = []

#     # Process each metric
#     for metric in metrics:
#         # Get all pod columns for this metric
#         pod_metric_cols = [f"pod_{p}_{metric}" for p in pod_ids if f"pod_{p}_{metric}" in df.columns]
        
#         if len(pod_metric_cols) <= 1:
#             continue  # Skip if not enough pods have this metric
        
#         # Track raw columns for potential dropping
#         if drop_raw:
#             columns_to_drop.extend(pod_metric_cols)
        
#         # Make sure all columns are numeric before calculations
#         for col in pod_metric_cols:
#             if df[col].dtype == 'object':
#                 df[col] = pd.to_numeric(df[col], errors='coerce')
#                 df[col] = df[col].fillna(0)
        
#         # Calculate total value for this metric
#         cluster_wise_total_col_name = f"cluster_total_{metric}"
#         cluster_wise_total_features.append(cluster_wise_total_col_name)
#         new_columns[cluster_wise_total_col_name] = df[pod_metric_cols].sum(axis=1)
        
#         # Calculate percentage columns
#         if not drop_pct:
#             for col in pod_metric_cols:
#                 # Extract the pod ID
#                 parts = col.split('_')
#                 pod_id_parts = parts[1:-1]
#                 pod_id = "_".join(pod_id_parts)
                
#                 pct_col_name = f"pct_{pod_id}_{metric}"
#                 # Reference the total from new_columns to handle dependencies correctly
#                 total_values = new_columns[cluster_wise_total_col_name]
#                 new_columns[pct_col_name] = df[col] / (total_values + 1e-6)
        
#         # Calculate ranks if needed
#         if not drop_rank or not drop_norm_rank:
#             ranks = df[pod_metric_cols].rank(axis=1)
            
#             for col in pod_metric_cols:
#                 # Extract the pod ID
#                 parts = col.split('_')
#                 pod_id_parts = parts[1:-1]
#                 pod_id = "_".join(pod_id_parts)
                
#                 # Add rank column if needed
#                 if not drop_rank:
#                     rank_col_name = f"rank_{pod_id}_{metric}"
#                     new_columns[rank_col_name] = ranks[col]
                
#                 # Add normalized rank if needed
#                 if not drop_norm_rank:
#                     norm_rank_col_name = f"norm_rank_{pod_id}_{metric}"
#                     new_columns[norm_rank_col_name] = (ranks[col] - 1) / (len(pod_metric_cols) - 1)
    
#     print(f"cluster_wise_total_features: {cluster_wise_total_features}")

#     # Create a new DataFrame with all the new columns at once
#     new_df = pd.DataFrame(new_columns, index=df.index)
    
#     # Count columns by type for reporting
#     rank_columns = [col for col in new_df.columns if col.startswith('rank_')]
#     norm_rank_columns = [col for col in new_df.columns if col.startswith('norm_rank_')]
#     total_raw_columns = len(columns_to_drop) if drop_raw else 0
    
#     # Report on dropping columns
#     if columns_to_drop or drop_pct or drop_rank or drop_norm_rank:
#         print(f"Dropping {total_raw_columns} columns: {len(rank_columns)} ranks, "
#               f"{len(norm_rank_columns)} norm_ranks, {total_raw_columns} raw metrics")
        
#     # Drop columns from original DataFrame if requested
#     if columns_to_drop:
#         df = df.drop(columns=columns_to_drop)
    
#     # Combine original DataFrame (minus dropped columns) with new columns
#     result_df = pd.concat([df, new_df], axis=1)
    
#     return result_df

def main(df, output_file):
    
    try:
        processed_df, mapping_info = preprocess_dataset(df, output_file)
        print("\nPod mapping (for action space):")
        for pod, idx in mapping_info['pod_to_index'].items():
            print(f"  Pod {pod} -> Action {idx}")
            
    except Exception as e:
        print(f"Error processing dataset: {e}")
        assert False

if __name__ == "__main__":
    global avg_tpot_slo_threshold, avg_ttft_slo_threshold
    avg_tpot_slo_threshold = 50
    avg_ttft_slo_threshold = 1000

    if len(sys.argv) < 2:
        print("Usage: python reorg.py <input_dir>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print("Input file does not exist. exiting...")
        exit()
    input_dir = os.path.dirname(input_file)
    df, json_columns = parse_log_file(input_file)
    if len(df) == 0:
        logger.error("No data found in the log file.")

    df = parse_json_columns(df, json_columns)
    df = normalize_time(df)
    output_file = os.path.join(input_dir, "processed_dataset.csv")
    main(df, output_file)

    print(f"* avg_tpot_slo_threshold: {avg_tpot_slo_threshold} ms")
    print(f"* avg_ttft_slo_threshold: {avg_ttft_slo_threshold} ms")
    print(f"Processed dataset saved to {output_file}")