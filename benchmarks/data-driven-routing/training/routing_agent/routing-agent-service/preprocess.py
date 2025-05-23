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
from logger import logger
import time

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
    df = pd.DataFrame(data, columns=column_names)
    if len(df) == 0:
        logger.error("No data found in the log file.")
        assert False
    return df, json_columns

# def parse_log_message(log_message):
#     """
#     Parse log message with format: **@latency_metrics@key1@value1@key2@value2@...
#     Based on working reference implementation.
    
#     Args:
#         log_message (str): Log message to parse
        
#     Returns:
#         tuple: (DataFrame with parsed data, list of JSON column names)
#     """
#     # Check if this is a metrics line
#     if "latency_metrics" not in log_message:
#         logging.error(f"Invalid line. {log_message}")
#         return pd.DataFrame(), []
    
#     # Split on the prefix to get clean key-value pairs
#     if "**@" in log_message:
#         line = log_message.split("**@latency_metrics@")[1]
#     else:
#         line = log_message
    
#     parts = line.split('@')
#     row = {}
#     json_columns = []
#     column_names = []
    
#     for i in range(0, len(parts), 2):
#         if i + 1 >= len(parts):
#             break
            
#         column_name = parts[i]
#         column_names.append(column_name)
#         value = parts[i + 1]
        
#         if value.startswith('{') and value.endswith('}'):
#             try:
#                 # Fix escaped quotes issue - replace \" with " before parsing
#                 fixed_value = value.replace('\\"', '"')
#                 json_columns.append(column_name)
#                 row[column_name] = json.loads(fixed_value)
#             except Exception as e:
#                 logging.error(f"Error decoding JSON, column: {column_name}, value: {value}")
#                 logging.error(f"Error: {e}")
                
#                 # Since we can't parse it, store as string to avoid losing data
#                 row[column_name] = value
#         else:
#             # Try to convert to appropriate data type
#             try:
#                 row[column_name] = int(value)
#             except ValueError:
#                 try:
#                     row[column_name] = float(value)
#                 except ValueError:
#                     row[column_name] = value
    
#     # Create DataFrame with single row
#     df = pd.DataFrame([row], columns=column_names)
    
#     return df, json_columns


def normalize_time(df):
    first_request_start_time = df['request_start_time'].min()
    df['normalized_start_time'] = df['request_start_time'] - first_request_start_time
    df['normalized_end_time'] = df['request_end_time'] - first_request_start_time
    df['normalized_start_time'] /= 1_000_000
    df['normalized_end_time'] /= 1_000_000
    
    
    if 'log_window_start_time' in df.columns:
        df['log_window_start_time'] = df['log_window_start_time'] - first_request_start_time
        df['log_window_start_time'] /= 1_000_000
    if 'log_window_end_time' in df.columns:
        df['log_window_end_time'] = df['log_window_end_time'] - first_request_start_time
        df['log_window_end_time'] /= 1_000_000

    df.loc[:, 'normalized_start_time'] = df['normalized_start_time'] - df['normalized_start_time'].min()
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
        logger.error(f"ERROR: Empty or NaN JSON string: {str(json_str)}...")
        assert False
    try:
        # Try standard JSON parsing
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        try:
            # Try replacing single quotes with double quotes
            if isinstance(json_str, str):
                return json.loads(json_str.replace("'", '"'))
            else:
                logger.error(f"ERROR: Invalid JSON string: {str(json_str)}...")
                assert False
        except (json.JSONDecodeError, TypeError):
            try:
                # Try using ast.literal_eval for Python dict literals
                if isinstance(json_str, str):
                    return ast.literal_eval(json_str)
                else:
                    logger.error(f"ERROR: Invalid JSON string: {str(json_str)}...")
                    assert False
            except (SyntaxError, ValueError, TypeError):
                logger.error(f"ERROR: Could not parse JSON: {str(json_str)}...")
                assert False

def calculate_ttft_reward(row, ttft_slo):
    try:
        ttft = float(row['ttft'])
        
        if ttft <= 0:
            return 0.5  # Maximum reward for perfect performance
        elif ttft <= ttft_slo:
            # Linear scaling from 0.5 (best) to 0.1 (at threshold)
            return 0.5 - (0.4 * ttft / ttft_slo)
        else:
            # Negative reward scaling with how much it exceeds threshold
            excess_factor = min(1.0, (ttft - ttft_slo) / ttft_slo)
            return -0.1 - (0.4 * excess_factor)
    except (ValueError, TypeError, ZeroDivisionError):
        return -0.5  # Default penalty for invalid data

def calculate_tpot_reward(row, avg_tpot_slo):
    try:
        avg_tpot = float(row['avg_tpot'])
        
        if avg_tpot <= 0:
            return -0.5  # Penalize invalid values
        elif avg_tpot <= avg_tpot_slo:
            # Linear scaling from 0.5 (best) to 0.1 (at threshold)
            return 0.1 + (0.4 * (1 - avg_tpot / avg_tpot_slo))
        else:
            # Negative reward scaling with how much it exceeds threshold
            excess_factor = min(1.0, (avg_tpot - avg_tpot_slo) / avg_tpot_slo)
            return -0.1 - (0.4 * excess_factor)
    except (ValueError, TypeError, ZeroDivisionError):
        return -0.5  # Default penalty for invalid data

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


## old
# def preprocess_dataset(df, ttft_slo, avg_tpot_slo):
#     all_pods_set = set()
#     logger.info("Collecting all unique pod IDs across the dataset...")
#     for _, row in df.iterrows():
#         try:
#             kv_cache_hit_ratios = safe_parse_json(row['allPodsKvCacheHitRatios'])
#             if kv_cache_hit_ratios:
#                 all_pods_set.update(kv_cache_hit_ratios.keys())
#             inflight_requests = safe_parse_json(row['numInflightRequestsAllPods'])
#             if inflight_requests:
#                 all_pods_set.update(inflight_requests.keys())
#             pod_metrics = safe_parse_json(row['podMetricsLastSecond'])
#             if pod_metrics:
#                 all_pods_set.update(pod_metrics.keys())
#         except Exception as e:
#             # logger.error(f"Error processing row: {row}")
#             logger.error(f"df.columns: {list(df.columns)}")
#             logger.error(f"Error: {e}")
#             assert False
#     all_pods = list(all_pods_set)
#     logger.info(f"Identified {len(all_pods)} pods: {all_pods}")


#     logger.info(f"Original dataset shape: {df.shape}")
#     logger.info(f"Columns: {df.columns.tolist()}")
    
#     expected_columns = [
#         # 'normalized_start_time', 
#         # 'time_bucket', 
#         # 'normalized_end_time', 
#         'requestID', 
#         # 'request_start_time', 
#         # 'request_end_time', 
#         'selectedpod', 
#         'ttft', 
#         'avg_tpot', 
#         'total_decode_time', 
#         'e2e',
#         'numInputTokens', 
#         'numOutputTokens', 
#         'numTotalTokens',
#         'allPodsKvCacheHitRatios', 
#         'numInflightRequestsAllPods',
#         'vllmGPUKVCacheUsage', 
#         'vllmCPUKVCacheUsage',
#         'vllmNumRequestsRunning', 
#         'vllmNumRequestsWaiting',
#         'podMetricsLastSecond', 
#         'numPrefillTokensForAllPods',
#         'numDecodeTokensForAllPods'
#     ]

#     expected_last_second_pod_metrics_keys = [
#         'last_second_avg_ttft_ms', 
#         'last_second_min_ttft_ms', 
#         'last_second_max_ttft_ms', 
#         'last_second_p50_ttft_ms', 
#         'last_second_p90_ttft_ms', 
#         'last_second_p95_ttft_ms', 
#         'last_second_p99_ttft_ms', 
#         'last_second_ttft_samples', 
#         'last_second_avg_tpot_ms', 
#         'last_second_min_tpot_ms', 
#         'last_second_max_tpot_ms', 
#         'last_second_p50_tpot_ms', 
#         'last_second_p90_tpot_ms', 
#         'last_second_p95_tpot_ms', 
#         'last_second_p99_tpot_ms', 
#         'last_second_tpot_samples', 
#         # 'last_second_early_tokens_tpot_ms', 
#         # 'last_second_mid_tokens_tpot_ms', 
#         # 'last_second_late_tokens_tpot_ms', 
#         'last_second_total_requests', 
#         'last_second_total_decode_tokens', 
#         'last_second_total_prefill_tokens', 
#         'last_second_total_tokens'
#     ]
    
#     # Check for missing expected columns
#     missing_columns = [col for col in expected_columns if col not in df.columns]
#     if missing_columns:
#         logger.error(f"Error: Missing expected columns: {missing_columns}")
#         assert False
    
#     # Check for unknown columns
#     unknown_columns = [col for col in df.columns if col not in expected_columns]
#     if unknown_columns:
#         logger.warning(f"Warning: Unused columns: {unknown_columns}")

#     # Filter out rows with empty 'podMetricsLastSecond'
#     df = df[df['podMetricsLastSecond'].notna()]
    
#     # Additional filtering for empty dictionaries
#     valid_rows = []
#     num_filter = 0
#     for idx, row in df.iterrows():
#         pod_metrics = safe_parse_json(row['podMetricsLastSecond'])
#         if pod_metrics and len(pod_metrics) > 0:  # Check if dictionary is non-empty
#             valid_rows.append(idx)
#         else:
#             logger.warning(f"Warning: Empty podMetricsLastSecond for row {idx}. request_id: {row['requestID']}")
#             num_filter += 1
#     logger.info(f"Filtered out {num_filter} rows with empty podMetricsLastSecond.")
#     df = df.loc[valid_rows]

#     # Process first row to check podMetricsLastSecond structure
#     if 'podMetricsLastSecond' in df.columns and len(df) > 0:
#         first_row = df.iloc[0]
#         logger.warning(f"WARNING: We are using the first row only to check podMetricsLastSecond structure")
#         pod_metrics = safe_parse_json(first_row['podMetricsLastSecond'])
#         logger.info(f"features in pod_metrics: {pod_metrics.keys()}")
#         try:
#             logger.info(f"features in pod_metrics: {pod_metrics[list(pod_metrics.keys())[0]].keys()}")
#         except Exception as e:
#             logger.error(f"Error: {e}")
#             logger.error(f"first_row['podMetricsLastSecond']: {first_row['podMetricsLastSecond']}")
#             logger.error(f"pod_metrics: {pod_metrics}")
#             logger.error(f"first_row: {first_row}")
#             assert False
#         if pod_metrics:
#             # Check structure for each pod
#             for pod_id, metrics in pod_metrics.items():
#                 # logger.info(f"Checking metrics for pod {pod_id}")
#                 logger.debug(f"metrics: {metrics}")
#                 # Check for missing expected keys
#                 missing_keys = [key for key in expected_last_second_pod_metrics_keys if key not in metrics]
#                 if missing_keys:
#                     logger.error(f"Error: Missing expected keys in podMetricsLastSecond for pod {pod_id}: {missing_keys}")
#                     assert False
                
#                 # Check for unknown keys
#                 unknown_keys = [key for key in metrics.keys() if key not in expected_last_second_pod_metrics_keys]
#                 if unknown_keys:
#                     logger.error(f"Error: Found unknown keys in podMetricsLastSecond for pod {pod_id}: {unknown_keys}")
#                     assert False
#     else:
#         logger.error("Error: podMetricsLastSecond column not found in the DataFrame.")
#         assert False
    
#     # Convert string columns to appropriate types
#     numeric_columns = [
#                         # 'normalized_start_time', 
#                         # 'normalized_end_time', 
#                         'ttft', 
#                         'avg_tpot', 
#                         'total_decode_time', 
#                         'e2e', 
#                         'numInputTokens', 
#                         'numOutputTokens', 
#                         'numTotalTokens',
#                     ]
    
#     for col in numeric_columns:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     # Parse JSON fields if they are strings
#     json_columns = [
#                     'allPodsKvCacheHitRatios', 
#                     'numInflightRequestsAllPods', 
#                     'vllmGPUKVCacheUsage', 
#                     'vllmCPUKVCacheUsage', 
#                     'vllmNumRequestsRunning', 
#                     'vllmNumRequestsWaiting', 
#                     'podMetricsLastSecond', 
#                     'numPrefillTokensForAllPods', 
#                     'numDecodeTokensForAllPods',
#                    ]
    
#     for col in json_columns:
#         if col in df.columns:
#             # logger.info(f"Processing column: {col}")
#             sample_val = df[col].iloc[0]
#             if isinstance(sample_val, str):
#                 # logger.info(f"  Parsing as JSON string")
#                 df[col] = df[col].apply(safe_parse_json)
#             elif isinstance(sample_val, dict):
#                 logger.debug(f"  Already parsed as dictionary")
#             else:
#                 logger.info(f"  Unknown type: {type(sample_val)}")
#                 # Try to parse anyway
#                 df[col] = df[col].apply(safe_parse_json)
    
#     # Create a new list to store processed records
#     processed_records = []
#     pod_gpu_models = {pod_id: "NVIDIA-L20" for pod_id in all_pods}
#     for _, row in df.iterrows():
#         base_features = {
#             'request_id': row['requestID'],
#             # 'request_start_time': row['request_start_time'],
#             # 'request_end_time': row['request_end_time'],
#             'selected_pod': row['selectedpod'],
#             'input_tokens': row['numInputTokens'],
#             'output_tokens': row['numOutputTokens'],
#             'total_tokens': row['numTotalTokens'],
#             'ttft': row['ttft'],
#             'avg_tpot': row['avg_tpot'],
#             'e2e_latency': row['e2e'],
#         }
#         pod_metrics = safe_parse_json(row['podMetricsLastSecond'])
        
#         # Here's the fixed part - getting pod-specific dictionaries for each metric
#         all_pods_kv_cache_hit_ratios = safe_parse_json(row['allPodsKvCacheHitRatios'])
#         all_pods_inflight_requests = safe_parse_json(row['numInflightRequestsAllPods'])
#         all_pods_gpu_kv_cache = safe_parse_json(row['vllmGPUKVCacheUsage'])
#         all_pods_cpu_kv_cache = safe_parse_json(row['vllmCPUKVCacheUsage'])
#         all_pods_running_requests = safe_parse_json(row['vllmNumRequestsRunning'])
#         all_pods_waiting_requests = safe_parse_json(row['vllmNumRequestsWaiting'])
#         all_pods_prefill_tokens = safe_parse_json(row['numPrefillTokensForAllPods'])
#         all_pods_decode_tokens = safe_parse_json(row['numDecodeTokensForAllPods'])
        
#         # Get pod-specific features for each pod
#         pod_features = {}
#         for pod_id in all_pods:
#             # Now extract just the value for this pod from each dictionary
#             pod_features[pod_id] = {
#                 'kv_hit_ratio': all_pods_kv_cache_hit_ratios.get(pod_id, 0),
#                 'inflight_requests': all_pods_inflight_requests.get(pod_id, 0),
#                 'gpu_kv_cache': all_pods_gpu_kv_cache.get(pod_id, 0),
#                 'cpu_kv_cache': all_pods_cpu_kv_cache.get(pod_id, 0),
#                 'running_requests': all_pods_running_requests.get(pod_id, 0),
#                 'waiting_requests': all_pods_waiting_requests.get(pod_id, 0),
#                 'prefill_tokens': all_pods_prefill_tokens.get(pod_id, 0),
#                 'decode_tokens': all_pods_decode_tokens.get(pod_id, 0),
#                 'gpu_model': pod_gpu_models[pod_id],
#             }
            
#             # Add key metrics from pod_metrics
#             key_metrics = extract_key_pod_metrics(pod_metrics, pod_id)
#             pod_features[pod_id].update(key_metrics)
        
#         # # Create a flattened record with all features
#         # record = base_features.copy()
        
#         # Add pod features with prefixed column names
#         for pod_id, features in pod_features.items():
#             pod_prefix = f"pod_{pod_id}"
#             for feature_name, feature_value in features.items():
#                 base_features[f"{pod_prefix}-{feature_name}"] = feature_value
        
#         processed_records.append(base_features)

#     # Create a new DataFrame with processed records
#     processed_df = pd.DataFrame(processed_records)
    
#     processed_df = processed_df.fillna(0)
    
#     # Map pod IDs to integer indices for the action space
#     unique_pods = processed_df['selected_pod'].unique()
#     # pod_to_index = {pod: idx for idx, pod in enumerate(unique_pods)}
#     pod_to_index = {str(pod): idx for idx, pod in enumerate(unique_pods)}

#     # Add the action column (the pod index)
#     processed_df['action'] = processed_df['selected_pod'].map(pod_to_index)
#     index_to_pod = {int(idx): str(pod) for pod, idx in pod_to_index.items()}

#     processed_df['avg_tpot_slo_satisfied'] = processed_df['avg_tpot'].apply(
#         lambda x: x <= avg_tpot_slo)
#     processed_df['avg_ttft_slo_satisfied'] = processed_df['ttft'].apply(
#         lambda x: x <= ttft_slo)

#     processed_df['ttft_reward'] = processed_df.apply(
#         lambda row: calculate_ttft_reward(row, ttft_slo=ttft_slo), axis=1)
    
#     processed_df['tpot_reward'] = processed_df.apply(
#         lambda row: calculate_tpot_reward(row, avg_tpot_slo=avg_tpot_slo), axis=1)
    
#     # Combined reward
#     processed_df['reward'] = processed_df['ttft_reward'] + processed_df['tpot_reward']
    
#     # Add normalized metrics for analysis
#     processed_df['ttft_normalized'] = processed_df['ttft'].apply(
#         lambda x: min(1.0, max(0.0, float(x) / 500)) if x > 0 else 0)
    
#     processed_df['tpot_normalized'] = processed_df['avg_tpot'].apply(
#         lambda x: min(1.0, max(0.0, float(x) / 25)) if x > 0 else 0)

#     # Save mapping information
#     # mapping_info = {
#     #     'pod_to_index': pod_to_index,
#     #     'index_to_pod': {idx: pod for pod, idx in pod_to_index.items()},
#     #     'pod_gpu_models': pod_gpu_models,
#     # }
#     mapping_info = {
#         'pod_to_index': pod_to_index,
#         'index_to_pod': index_to_pod,
#         'pod_gpu_models': pod_gpu_models,
#     }
    
#     logger.info(f"Processed dataset shape: {processed_df.shape}")
#     logger.info(f"Processed columns: {processed_df.columns[:10].tolist()}...")

#     # logger.info GPU model mapping
#     logger.info("\nPod GPU model mapping:")
#     for pod_id, gpu_model in pod_gpu_models.items():
#         logger.info(f"  Pod {pod_id} -> GPU model {gpu_model}")
    
#     return processed_df, mapping_info, all_pods

## new
def preprocess_dataset(df, ttft_slo, avg_tpot_slo):
    # Pre-parse all JSON columns once to avoid repeated parsing
    logger.info("Pre-parsing JSON columns...")
    json_columns = [
        'allPodsKvCacheHitRatios', 
        'numInflightRequestsAllPods', 
        'vllmGPUKVCacheUsage', 
        'vllmCPUKVCacheUsage', 
        'vllmNumRequestsRunning', 
        'vllmNumRequestsWaiting', 
        'podMetricsLastSecond', 
        'numPrefillTokensForAllPods', 
        'numDecodeTokensForAllPods',
    ]
    
    json_parse_start_time = time.time()
    for col in json_columns:
        if col in df.columns:
            sample_val = df[col].iloc[0]
            if isinstance(sample_val, str):
                df[col] = df[col].apply(safe_parse_json)
    json_parse_overhead = time.time() - json_parse_start_time

    # Collect all unique pod IDs in a single pass
    all_pods_set = set()
    logger.info("Collecting all unique pod IDs across the dataset...")
    
    # Vectorized approach - get all unique pods from all relevant columns at once
    for col in ['allPodsKvCacheHitRatios', 'numInflightRequestsAllPods', 'podMetricsLastSecond']:
        if col in df.columns:
            for data in df[col]:
                if data:
                    all_pods_set.update(data.keys())
    
    all_pods = list(all_pods_set)
    logger.info(f"Identified {len(all_pods)} pods: {all_pods}")

    logger.info(f"Original dataset shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    expected_columns = [
        'requestID', 
        'selectedpod', 
        'ttft', 
        'avg_tpot', 
        'total_decode_time', 
        'e2e',
        'numInputTokens', 
        'numOutputTokens', 
        'numTotalTokens',
        'allPodsKvCacheHitRatios', 
        'numInflightRequestsAllPods',
        'vllmGPUKVCacheUsage', 
        'vllmCPUKVCacheUsage',
        'vllmNumRequestsRunning', 
        'vllmNumRequestsWaiting',
        'podMetricsLastSecond', 
        'numPrefillTokensForAllPods',
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
        'last_second_total_requests', 
        'last_second_total_decode_tokens', 
        'last_second_total_prefill_tokens', 
        'last_second_total_tokens'
    ]
    
    column_check_start_time = time.time()
    # Check for missing expected columns
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Error: Missing expected columns: {missing_columns}")
        assert False
    
    # Check for unknown columns
    unknown_columns = [col for col in df.columns if col not in expected_columns]
    if unknown_columns:
        logger.warning(f"Warning: Unused columns: {unknown_columns}")

    # Filter out rows with empty 'podMetricsLastSecond' - vectorized approach
    valid_mask = df['podMetricsLastSecond'].notna()
    
    # Additional filtering for empty dictionaries - vectorized
    # non_empty_mask = df['podMetricsLastSecond'].apply(lambda x: x and len(x) > 0 if isinstance(x, dict) else False)

    non_empty_mask = df['podMetricsLastSecond'].apply(lambda x: isinstance(x, dict) and len(x) > 0)
    
    num_filter = len(df) - non_empty_mask.sum()
    logger.info(f"Filtered out {num_filter} rows with empty podMetricsLastSecond.")
    
    df = df[valid_mask & non_empty_mask].copy()
    column_check_overhead = time.time() - column_check_start_time # 0-4ms

    podmetrics_parse_start_time = time.time()
    # Process first row to check podMetricsLastSecond structure (same as before)
    if 'podMetricsLastSecond' in df.columns and len(df) > 0:
        first_row = df.iloc[0]
        logger.warning(f"WARNING: We are using the first row only to check podMetricsLastSecond structure")
        pod_metrics = first_row['podMetricsLastSecond']  # Already parsed
        logger.info(f"features in pod_metrics: {pod_metrics.keys()}")
        try:
            logger.info(f"features in pod_metrics: {pod_metrics[list(pod_metrics.keys())[0]].keys()}")
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error(f"first_row['podMetricsLastSecond']: {first_row['podMetricsLastSecond']}")
            logger.error(f"pod_metrics: {pod_metrics}")
            logger.error(f"first_row: {first_row}")
            assert False
        if pod_metrics:
            # Check structure for each pod
            for pod_id, metrics in pod_metrics.items():
                logger.debug(f"metrics: {metrics}")
                # Check for missing expected keys
                missing_keys = [key for key in expected_last_second_pod_metrics_keys if key not in metrics]
                if missing_keys:
                    logger.error(f"Error: Missing expected keys in podMetricsLastSecond for pod {pod_id}: {missing_keys}")
                    assert False
                
                # Check for unknown keys
                unknown_keys = [key for key in metrics.keys() if key not in expected_last_second_pod_metrics_keys]
                if unknown_keys:
                    logger.error(f"Error: Found unknown keys in podMetricsLastSecond for pod {pod_id}: {unknown_keys}")
                    assert False
    else:
        logger.error("Error: podMetricsLastSecond column not found in the DataFrame.")
        assert False
    podmetrics_parse_overhead = time.time() - podmetrics_parse_start_time # 0-1ms


    numeric_conversion_start_time = time.time()
    # Convert string columns to appropriate types - vectorized
    numeric_columns = [
        'ttft', 
        'avg_tpot', 
        'total_decode_time', 
        'e2e', 
        'numInputTokens', 
        'numOutputTokens', 
        'numTotalTokens',
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    numeric_conversion_overhead = time.time() - numeric_conversion_start_time # 0-1ms
    

    # Pre-create pod_gpu_models and pod features structure
    pod_gpu_models = {pod_id: "NVIDIA-L20" for pod_id in all_pods}
    
    # Vectorized processing using pandas operations
    logger.info("Processing records in vectorized manner...")
    
    get_value_start_time = time.time()
    # Extract base features
    base_data = {
        'request_id': df['requestID'].values,
        'selected_pod': df['selectedpod'].values,
        'input_tokens': df['numInputTokens'].values,
        'output_tokens': df['numOutputTokens'].values,
        'total_tokens': df['numTotalTokens'].values,
        'ttft': df['ttft'].values,
        'avg_tpot': df['avg_tpot'].values,
        'e2e_latency': df['e2e'].values,
    }
    
    # Pre-extract all JSON data to avoid repeated parsing
    all_kv_cache = df['allPodsKvCacheHitRatios'].values
    all_inflight = df['numInflightRequestsAllPods'].values  
    all_gpu_cache = df['vllmGPUKVCacheUsage'].values
    all_cpu_cache = df['vllmCPUKVCacheUsage'].values
    all_running = df['vllmNumRequestsRunning'].values
    all_waiting = df['vllmNumRequestsWaiting'].values
    all_prefill = df['numPrefillTokensForAllPods'].values
    all_decode = df['numDecodeTokensForAllPods'].values
    all_pod_metrics = df['podMetricsLastSecond'].values
    
    # Process pod features for all rows at once
    for pod_id in all_pods:
        # Vectorized extraction for each pod across all rows
        base_data[f"pod_{pod_id}-kv_hit_ratio"] = [data.get(pod_id, 0) for data in all_kv_cache]
        base_data[f"pod_{pod_id}-inflight_requests"] = [data.get(pod_id, 0) for data in all_inflight]
        base_data[f"pod_{pod_id}-gpu_kv_cache"] = [data.get(pod_id, 0) for data in all_gpu_cache]
        base_data[f"pod_{pod_id}-cpu_kv_cache"] = [data.get(pod_id, 0) for data in all_cpu_cache]
        base_data[f"pod_{pod_id}-running_requests"] = [data.get(pod_id, 0) for data in all_running]
        base_data[f"pod_{pod_id}-waiting_requests"] = [data.get(pod_id, 0) for data in all_waiting]
        base_data[f"pod_{pod_id}-prefill_tokens"] = [data.get(pod_id, 0) for data in all_prefill]
        base_data[f"pod_{pod_id}-decode_tokens"] = [data.get(pod_id, 0) for data in all_decode]
        base_data[f"pod_{pod_id}-gpu_model"] = ["NVIDIA-L20"] * len(df)
        
        # Extract key metrics for this pod across all rows
        for metric_key in ['last_second_avg_ttft_ms', 'last_second_avg_tpot_ms', 'last_second_p99_ttft_ms', 
                          'last_second_p99_tpot_ms', 'last_second_total_requests', 'last_second_total_tokens',
                          'last_second_total_decode_tokens', 'last_second_total_prefill_tokens']:
            base_data[f"pod_{pod_id}-{metric_key}"] = [
                metrics.get(pod_id, {}).get(metric_key, 0) for metrics in all_pod_metrics
            ]
    get_value_overhead = time.time() - get_value_start_time # 0ms

    ##################################################################
    # # Create DataFrame directly from the dictionary
    # create_df_start_time = time.time()
    # processed_df = pd.DataFrame(base_data)
    # processed_df = processed_df.fillna(0)
    
    # # Map pod IDs to integer indices for the action space
    # unique_pods = processed_df['selected_pod'].unique()
    # pod_to_index = {str(pod): idx for idx, pod in enumerate(unique_pods)}

    # # Add the action column (the pod index) - vectorized
    # processed_df['action'] = processed_df['selected_pod'].map(pod_to_index)
    # index_to_pod = {int(idx): str(pod) for pod, idx in pod_to_index.items()}

    # # Vectorized reward calculations
    # processed_df['avg_tpot_slo_satisfied'] = processed_df['avg_tpot'] <= avg_tpot_slo
    # processed_df['avg_ttft_slo_satisfied'] = processed_df['ttft'] <= ttft_slo

    # # Vectorized TTFT reward calculation
    # ttft_values = processed_df['ttft'].values
    # ttft_rewards = np.where(
    #     ttft_values <= 0, 
    #     0.5,  # Maximum reward for perfect performance
    #     np.where(
    #         ttft_values <= ttft_slo,
    #         0.5 - (0.4 * ttft_values / ttft_slo),  # Linear scaling
    #         -0.1 - (0.4 * np.minimum(1.0, (ttft_values - ttft_slo) / ttft_slo))  # Negative reward
    #     )
    # )
    # processed_df['ttft_reward'] = ttft_rewards
    
    # # Vectorized TPOT reward calculation  
    # tpot_values = processed_df['avg_tpot'].values
    # tpot_rewards = np.where(
    #     tpot_values <= 0,
    #     -0.5,  # Penalize invalid values
    #     np.where(
    #         tpot_values <= avg_tpot_slo,
    #         0.1 + (0.4 * (1 - tpot_values / avg_tpot_slo)),  # Linear scaling
    #         -0.1 - (0.4 * np.minimum(1.0, (tpot_values - avg_tpot_slo) / avg_tpot_slo))  # Negative reward
    #     )
    # )
    # processed_df['tpot_reward'] = tpot_rewards
    
    # # Combined reward
    # processed_df['reward'] = processed_df['ttft_reward'] + processed_df['tpot_reward']
    
    # # Vectorized normalized metrics
    # processed_df['ttft_normalized'] = np.where(
    #     processed_df['ttft'] > 0,
    #     np.minimum(1.0, np.maximum(0.0, processed_df['ttft'] / 500)),
    #     0
    # )
    
    # processed_df['tpot_normalized'] = np.where(
    #     processed_df['avg_tpot'] > 0,
    #     np.minimum(1.0, np.maximum(0.0, processed_df['avg_tpot'] / 25)),
    #     0
    # )

    # # Save mapping information
    # mapping_info = {
    #     'pod_to_index': pod_to_index,
    #     'index_to_pod': index_to_pod,
    #     'pod_gpu_models': pod_gpu_models,
    # }
    
    # logger.info(f"Processed dataset shape: {processed_df.shape}")
    # logger.info(f"Processed columns: {processed_df.columns[:10].tolist()}...")

    # # logger.info GPU model mapping
    # logger.info("\nPod GPU model mapping:")
    # for pod_id, gpu_model in pod_gpu_models.items():
    #     logger.info(f"  Pod {pod_id} -> GPU model {gpu_model}")
    # create_value_overhead = time.time() - create_df_start_time



    # Create DataFrame directly from the dictionary - OPTIMIZED VERSION

    # Pre-calculate all derived values before DataFrame creation
    num_rows = len(base_data['request_id'])

    pod_index_start_time = time.time()
    # Map pod IDs to integer indices for the action space - do this early with numpy arrays
    unique_pods = np.unique(base_data['selected_pod'])
    pod_to_index = {str(pod): idx for idx, pod in enumerate(unique_pods)}
    index_to_pod = {int(idx): str(pod) for pod, idx in pod_to_index.items()}

    # Pre-calculate all derived columns as numpy arrays (much faster than pandas operations)
    selected_pods_array = np.array(base_data['selected_pod'])
    action_values = np.array([pod_to_index[str(pod)] for pod in selected_pods_array])

    ttft_values = np.array(base_data['ttft'], dtype=np.float64)
    tpot_values = np.array(base_data['avg_tpot'], dtype=np.float64)
    pod_index_overhead = time.time() - pod_index_start_time

    # Vectorized reward calculations using numpy (faster than pandas)
    reward_calc_start_time = time.time()
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
    reward_calc_overhead = time.time() - reward_calc_start_time

    slo_update_start_time = time.time()
    # Add all the computed columns to base_data before DataFrame creation
    base_data.update({
        'action': action_values,
        'avg_tpot_slo_satisfied': tpot_values <= avg_tpot_slo,
        'avg_ttft_slo_satisfied': ttft_values <= ttft_slo,
        'ttft_reward': ttft_rewards,
        'tpot_reward': tpot_rewards,
        'reward': ttft_rewards + tpot_rewards,
        'ttft_normalized': np.where(
            ttft_values > 0,
            np.minimum(1.0, np.maximum(0.0, ttft_values / 500)),
            0
        ),
        'tpot_normalized': np.where(
            tpot_values > 0,
            np.minimum(1.0, np.maximum(0.0, tpot_values / 25)),
            0
        )
    })
    slo_update_overhead = time.time() - slo_update_start_time

    # Create DataFrame only once with all data
    create_df_start_time = time.time()
    processed_df = pd.DataFrame(base_data)
    create_df_overhead = time.time() - create_df_start_time


    # Replace fillna(0) with a more targeted approach since most values should already be handled
    # Only fill NaN values in specific columns that might have them
    nan_columns = processed_df.columns[processed_df.isnull().any()].tolist()
    if nan_columns:
        processed_df[nan_columns] = processed_df[nan_columns].fillna(0)

    # Save mapping information
    mapping_info = {
        'pod_to_index': pod_to_index,
        'index_to_pod': index_to_pod,
        'pod_gpu_models': pod_gpu_models,
    }

    logger.debug(f"Processed dataset shape: {processed_df.shape}")
    logger.debug(f"Processed columns: {processed_df.columns[:10].tolist()}...")

    # logger.debug GPU model mapping
    logger.debug("\nPod GPU model mapping:")
    for pod_id, gpu_model in pod_gpu_models.items():
        logger.debug(f"  Pod {pod_id} -> GPU model {gpu_model}")

    ##################################################################

    preprocess_dataset_overhead_summary = {
        'preprocess.preprocess_dataset_json_parse_overhead': json_parse_overhead*1000,
        'preprocess.preprocess_dataset_column_check_overhead': column_check_overhead*1000,
        'preprocess.preprocess_dataset_podmetrics_parse_overhead': podmetrics_parse_overhead*1000,
        'preprocess.preprocess_dataset_numeric_conversion_overhead': numeric_conversion_overhead*1000,
        'preprocess.preprocess_dataset_get_value_overhead': get_value_overhead*1000,
        'preprocess.preprocess_dataset_create_df_overhead': create_df_overhead*1000,
        'preprocess.preprocess_dataset_pod_index_overhead': pod_index_overhead*1000,
        'preprocess.preprocess_dataset_reward_calc_overhead': reward_calc_overhead*1000,
        'preprocess.preprocess_dataset_slo_update_overhead': slo_update_overhead*1000,
    }
    
    return processed_df, mapping_info, all_pods, preprocess_dataset_overhead_summary


# Optimized version - just replace your existing parse_log_message function with this
def parse_log_message(log_message):
    """
    Ultra-fast log message parser - drop-in replacement for parse_log_message.
    Returns: (DataFrame, json_columns) for compatibility
    """
    # Fast check without string operations
    if "latency_metrics" not in log_message:
        logging.error(f"Invalid line. {log_message}")
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
                logging.error(f"Error decoding JSON, column: {key}, value: {value}")
                logging.error(f"Error: {e}")
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
        return df, json_columns
    else:
        return pd.DataFrame(), []


def main(input_file, log_message, TTFT_SLO, AVG_TPOT_SLO):
    # input_file is None for inference workload, valid only for training workflow.
    parse_log_file_start_time = time.time()
    
    if input_file is not None:
        df, json_columns = parse_log_file(input_file)
        if len(df) == 0:
            logger.error("No data found in the log file.")
    else:
        # Use the optimized parser (same function name)
        df, json_columns = parse_log_message(log_message)
        if len(df) == 0:
            logger.error("No data found in the log message.")
    
    logger.info(f"df.columns: {list(df.columns)}, json_columns: {json_columns}")
    
    # REMOVED: No need for parse_json_columns since JSON is already parsed
    # df = parse_json_columns(df, json_columns)
    
    if len(df) == 0:
        logger.error("No data found after parsing JSON columns.")
        logger.info(f"Parsed {len(df)} records from {input_file}")
        assert False
    
    parse_log_file_overhead = time.time() - parse_log_file_start_time
    
    preprocess_dataset_start_time = time.time()
    processed_df, mapping_info, all_pods, preprocess_dataset_overhead_summary = preprocess_dataset(df, TTFT_SLO, AVG_TPOT_SLO)
    total_preprocess_dataset_overhead = time.time() - preprocess_dataset_start_time

    mapping_info_write_start_time = time.time()
    output_file = None
    if input_file is not None:
        try:
            input_dir = os.path.dirname(input_file)
            # Save mapping information
            output_file = os.path.join(input_dir, "processed_dataset.csv")
            mapping_file = output_file.replace('.csv', '_mapping.json')
            if not os.path.exists(mapping_file):
                with open(mapping_file, 'w') as f:
                    try:
                        json.dump(mapping_info, f, indent=2)
                        logger.info("JSON serialization successful")
                    except TypeError as e:
                        logger.error(f"JSON serialization failed: {e}")
                        # Try to identify the problematic part by serializing each part separately
                        logger.info("Trying to identify the problematic part:")
                        try:
                            json.dumps(mapping_info['pod_to_index'])
                            logger.info("pod_to_index serialization: OK")
                        except TypeError as e:
                            logger.error(f"pod_to_index serialization failed: {e}")
                        
                        try:
                            json.dumps(mapping_info['index_to_pod'])
                            logger.info("index_to_pod serialization: OK")
                        except TypeError as e:
                            logger.error(f"index_to_pod serialization failed: {e}")
                        
                        try:
                            json.dumps(mapping_info['pod_gpu_models'])
                            logger.info("pod_gpu_models serialization: OK")
                        except TypeError as e:
                            logger.error(f"pod_gpu_models serialization failed: {e}")
                        
                        raise
                
            logger.info(f"Mapping information saved to {mapping_file}")
            logger.info("\nPod mapping (for action space):")
            for pod, idx in mapping_info['pod_to_index'].items():
                logger.info(f"  Pod {pod} -> Action {idx}")
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            assert False
    mapping_info_write_overhead = time.time() - mapping_info_write_start_time

    preprocess_dataset_overhead_summary["preprocess.main.total_overhead"] = total_preprocess_dataset_overhead*1000
    preprocess_dataset_overhead_summary["preprocess.main.mapping_info_write_overhead"] = mapping_info_write_overhead*1000
    preprocess_dataset_overhead_summary["preprocess.main.parse_log_file_overhead"] = parse_log_file_overhead*1000

    return processed_df, output_file, all_pods, preprocess_dataset_overhead_summary
