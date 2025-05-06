import pandas as pd
import numpy as np
import json
import ast
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import argparse

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
        return {
            'avg_ttft_ms': 0,
            'avg_tpot_ms': 0, 
            'total_requests': 0,
            'total_tokens': 0
        }
    
    metrics = pod_metrics[pod_id]
    return {
        'avg_ttft_ms': metrics.get('avg_ttft_ms', 0),
        'avg_tpot_ms': metrics.get('avg_tpot_ms', 0),
        'total_requests': metrics.get('total_requests', 0),
        'total_tokens': metrics.get('total_tokens', 0)
    }

def preprocess_dataset(input_file, output_file=None):
    """
    Preprocess the dataset for offline RL training
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file
    
    Returns:
        Processed DataFrame and pod mapping information
    """
    print(f"Reading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    

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

    expected_pod_metrics_keys = [
        'avg_ttft_ms', 'min_ttft_ms', 'max_ttft_ms', 'p50_ttft_ms', 
        'p90_ttft_ms', 'p95_ttft_ms', 'p99_ttft_ms', 'ttft_samples', 
        'avg_tpot_ms', 'min_tpot_ms', 'max_tpot_ms', 'p50_tpot_ms', 
        'p90_tpot_ms', 'p95_tpot_ms', 'p99_tpot_ms', 'tpot_samples', 
        'early_tokens_tpot_ms', 'mid_tokens_tpot_ms', 'late_tokens_tpot_ms', 
        'total_requests', 'total_decode_tokens', 'total_prefill_tokens', 
        'total_tokens'
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
        pod_metrics = safe_parse_json(first_row['podMetricsLastSecond'])
        
        if pod_metrics:
            print(f"Checking podMetricsLastSecond structure...")
            # Check structure for each pod
            for pod_id, metrics in pod_metrics.items():
                # print(f"Checking metrics for pod {pod_id}")
                
                # Check for missing expected keys
                missing_keys = [key for key in expected_pod_metrics_keys if key not in metrics]
                if missing_keys:
                    print(f"Error: Missing expected keys in podMetricsLastSecond for pod {pod_id}: {missing_keys}")
                    assert False
                
                # Check for unknown keys
                unknown_keys = [key for key in metrics.keys() if key not in expected_pod_metrics_keys]
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
    
    # Get unique pods from the first row
    first_row = df.iloc[0]
    first_row_kv_cache = first_row['allPodsKvCacheHitRatios']
    
    if isinstance(first_row_kv_cache, dict):
        all_pods = list(first_row_kv_cache.keys())
    else:
        # Try to parse it again
        all_pods = list(safe_parse_json(first_row_kv_cache).keys())
        
    print(f"Identified {len(all_pods)} pods: {all_pods}")
    
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
        
        
        # Parse JSON fields to get pod-specific data
        kv_cache_hit_ratios = row['allPodsKvCacheHitRatios']
        inflight_requests = row['numInflightRequestsAllPods']
        gpu_kv_cache = row['vllmGPUKVCacheUsage']
        cpu_kv_cache = row['vllmCPUKVCacheUsage']
        running_requests = row['vllmNumRequestsRunning']
        waiting_requests = row['vllmNumRequestsWaiting']
        pod_metrics = row['podMetricsLastSecond']
        prefill_tokens = row['numPrefillTokensForAllPods']
        decode_tokens = row['numDecodeTokensForAllPods']
        
        # Get pod-specific features for each pod
        pod_features = {}
        for pod_id in all_pods:
            pod_features[pod_id] = {
                'kv_hit_ratio': kv_cache_hit_ratios.get(pod_id, 0),
                'inflight_requests': inflight_requests.get(pod_id, 0),
                'gpu_kv_cache': gpu_kv_cache.get(pod_id, 0),
                'cpu_kv_cache': cpu_kv_cache.get(pod_id, 0),
                'running_requests': running_requests.get(pod_id, 0),
                'waiting_requests': waiting_requests.get(pod_id, 0),
                'prefill_tokens': prefill_tokens.get(pod_id, 0),
                'decode_tokens': decode_tokens.get(pod_id, 0),
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
                record[f"{pod_prefix}_{feature_name}"] = feature_value
        
        processed_records.append(record)
    
    # Create a new DataFrame with processed records
    processed_df = pd.DataFrame(processed_records)
    
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

def create_train_test_split(processed_df, train_ratio=0.8, output_dir=None):
    """
    Split the processed dataset into training and testing sets
    """
    if output_dir is None:
        output_dir = '.'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Shuffle the dataset
    shuffled_df = processed_df.sample(frac=1, random_state=42)
    
    # Split into train and test
    train_size = int(len(shuffled_df) * train_ratio)
    train_df = shuffled_df[:train_size]
    test_df = shuffled_df[train_size:]
    
    # Save the splits
    train_file = os.path.join(output_dir, 'train_data.csv')
    test_file = os.path.join(output_dir, 'test_data.csv')
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"Training set shape: {train_df.shape}, saved to {train_file}")
    print(f"Testing set shape: {test_df.shape}, saved to {test_file}")
    
    return train_df, test_df

def main(input_dir):
    output_dir = input_dir
    input_file = os.path.join(input_dir, "parsed-gateway-plugins.log.csv")
    output_file = os.path.join(output_dir, "processed_dataset.csv")
    try:
        processed_df, mapping_info = preprocess_dataset(input_file, output_file)
        
        # Create train-test split
        train_df, test_df = create_train_test_split(processed_df, output_dir=output_dir)
        print("\nPod mapping (for action space):")
        for pod, idx in mapping_info['pod_to_index'].items():
            print(f"  Pod {pod} -> Action {idx}")
            
    except Exception as e:
        print(f"Error processing dataset: {e}")
        assert False

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python reorg.py <input_dir>")
        sys.exit(1)
    input_dir = sys.argv[1]
    if not os.path.exists(input_dir):
        print("Input dir does not exist. exiting...")
        exit()

    global avg_tpot_slo_threshold, avg_ttft_slo_threshold
    avg_tpot_slo_threshold = 25
    avg_ttft_slo_threshold = 500
    main(input_dir)
    print(f"* avg_tpot_slo_threshold: {avg_tpot_slo_threshold} ms")
    print(f"* avg_ttft_slo_threshold: {avg_ttft_slo_threshold} ms")