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

AVG_TPOT_SLO = 50
TTFT_SLO = 1000

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
                logger.warning(f"Warning: Could not parse JSON: {str(json_str)[:50]}...")
                return {}

def calculate_ttft_reward(row, ttft_slo_threshold):
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

def calculate_tpot_reward(row, AVG_TPOT_SLO):
    try:
        avg_tpot = float(row['avg_tpot'])
        
        if avg_tpot <= 0:
            return -0.5  # Penalize invalid values
        elif avg_tpot <= AVG_TPOT_SLO:
            # Linear scaling from 0.5 (best) to 0.1 (at threshold)
            return 0.1 + (0.4 * (1 - avg_tpot / AVG_TPOT_SLO))
        else:
            # Negative reward scaling with how much it exceeds threshold
            excess_factor = min(1.0, (avg_tpot - AVG_TPOT_SLO) / AVG_TPOT_SLO)
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

def preprocess_dataset(df):
    all_pods_set = set()
    logger.info("Collecting all unique pod IDs across the dataset...")
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
            logger.info(f"Found all EIGHT pods: {all_pods_set}. break")
            break
    all_pods = list(all_pods_set)
    logger.info(f"Identified {len(all_pods)} pods: {all_pods}")


    logger.info(f"Original dataset shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    expected_columns = [
        'normalized_start_time', 'time_bucket', 'normalized_end_time', 
        'requestID', 'request_start_time', 'request_end_time', 
        'selectedpod', 'ttft', 'avg_tpot', 'total_decode_time', 'e2e',
        'numInputTokens', 'numOutputTokens', 'numTotalTokens',
        'allPodsKvCacheHitRatios', 'numInflightRequestsAllPods',
        'vllmGPUKVCacheUsage', 'vllmCPUKVCacheUsage',
        'vllmNumRequestsRunning', 'vllmNumRequestsWaiting',
        'podMetricsLastSecond', 'numPrefillTokensForAllPods',
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
        # 'last_second_early_tokens_tpot_ms', 
        # 'last_second_mid_tokens_tpot_ms', 
        # 'last_second_late_tokens_tpot_ms', 
        'last_second_total_requests', 
        'last_second_total_decode_tokens', 
        'last_second_total_prefill_tokens', 
        'last_second_total_tokens'
    ]
    
    # Check for missing expected columns
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Error: Missing expected columns: {missing_columns}")
        assert False
    
    # Check for unknown columns
    unknown_columns = [col for col in df.columns if col not in expected_columns]
    if unknown_columns:
        logger.error(f"Error: Found unknown columns: {unknown_columns}")

    # Process first row to check podMetricsLastSecond structure
    if 'podMetricsLastSecond' in df.columns and len(df) > 0:
        first_row = df.iloc[0]
        logger.warning(f"WARNING: We are using the first row only to check podMetricsLastSecond structure")
        pod_metrics = safe_parse_json(first_row['podMetricsLastSecond'])
        # logger.info(f"features in pod_metrics: {pod_metrics.keys()}")
        logger.info(f"features in pod_metrics: {pod_metrics[list(pod_metrics.keys())[0]].keys()}")
        if pod_metrics:
            # Check structure for each pod
            for pod_id, metrics in pod_metrics.items():
                # logger.info(f"Checking metrics for pod {pod_id}")
                logger.info(f"metrics: {metrics}")
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
            # logger.info(f"Processing column: {col}")
            sample_val = df[col].iloc[0]
            if isinstance(sample_val, str):
                # logger.info(f"  Parsing as JSON string")
                df[col] = df[col].apply(safe_parse_json)
            elif isinstance(sample_val, dict):
                logger.info(f"  Already parsed as dictionary")
            else:
                logger.info(f"  Unknown type: {type(sample_val)}")
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
            pod_prefix = f"pod_{pod_id}"
            for feature_name, feature_value in features.items():
                record[f"{pod_prefix}-{feature_name}"] = feature_value
        
        processed_records.append(record)
    
    # Create a new DataFrame with processed records
    processed_df = pd.DataFrame(processed_records)
    
    all_pod_ids = all_pods
    processed_df = processed_df.fillna(0)
    
    # Map pod IDs to integer indices for the action space
    unique_pods = processed_df['selected_pod'].unique()
    pod_to_index = {pod: idx for idx, pod in enumerate(unique_pods)}
    
    # Add the action column (the pod index)
    processed_df['action'] = processed_df['selected_pod'].map(pod_to_index)

    processed_df['avg_tpot_slo_satisfied'] = processed_df['avg_tpot'].apply(
        lambda x: x <= AVG_TPOT_SLO)
    processed_df['avg_ttft_slo_satisfied'] = processed_df['ttft'].apply(
        lambda x: x <= TTFT_SLO)

    processed_df['ttft_reward'] = processed_df.apply(
        lambda row: calculate_ttft_reward(row, ttft_slo_threshold=TTFT_SLO), axis=1)
    
    processed_df['tpot_reward'] = processed_df.apply(
        lambda row: calculate_tpot_reward(row, AVG_TPOT_SLO=AVG_TPOT_SLO), axis=1)
    
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
    
    logger.info(f"Processed dataset shape: {processed_df.shape}")
    logger.info(f"Processed columns: {processed_df.columns[:10].tolist()}...")

    # logger.info GPU model mapping
    logger.info("\nPod GPU model mapping:")
    for pod_id, gpu_model in pod_gpu_models.items():
        logger.info(f"  Pod {pod_id} -> GPU model {gpu_model}")
    
    return processed_df, mapping_info, all_pods

def main(input_file):
    df, json_columns = parse_log_file(input_file)
    if len(df) == 0:
        logger.error("No data found in the log file.")
    df = parse_json_columns(df, json_columns)
    df = normalize_time(df)
    input_dir = os.path.dirname(input_file)
    try:
        processed_df, mapping_info, all_pods = preprocess_dataset(df)
        
        # Save the processed dataset
        output_file = os.path.join(input_dir, "processed_dataset.csv")
        logger.info(f"Saving processed dataset to {output_file}...")
        processed_df.to_csv(output_file, index=False)
        
        # Save mapping information
        mapping_file = output_file.replace('.csv', '_mapping.json')
        with open(mapping_file, 'w') as f:
            json.dump(mapping_info, f, indent=2)
        logger.info(f"Mapping information saved to {mapping_file}")
        logger.info("\nPod mapping (for action space):")
        for pod, idx in mapping_info['pod_to_index'].items():
            logger.info(f"  Pod {pod} -> Action {idx}")
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        assert False
    return processed_df, output_file, all_pods

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python reorg.py <input_dir>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        logger.error("ERROR: Input file does not exist. exiting...")
        exit()
    processed_df, output_file = main(input_file)
    logger.info(f"* AVG_TPOT_SLO: {AVG_TPOT_SLO} ms")
    logger.info(f"* TTFT_SLO: {TTFT_SLO} ms")
    logger.info(f"Processed dataset saved to {output_file}")