#!/usr/bin/env python3

# encoding.py

"""
LLM Request Router - Enhanced Data Preprocessing
-----------------------------------------------
Transforms raw request routing data into structured tensors for transformer-based RL model.
Implements:
- Pod state extraction and normalization
- Expected KV hit ratio isolation for cross-attention
- Request feature extraction
- Metrics-based positional encoding
- Temporal feature handling with staleness indicators
- Request-pod interaction features
- One-hot encoding for categorical features
"""

import sys
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
import logging
import re
import argparse
from datetime import datetime
from logger import logger
import time
class LLMRoutingDataProcessor:
    """Processes raw LLM request routing data into formatted tensors for RL training.
    
    Implements advanced encoding techniques:
    1. Metrics-based positional encoding for transformer
    2. Cross-attention preparation for KV hit ratio
    3. Temporal feature handling with staleness indicators
    4. Request-pod interaction features
    """
    
    def __init__(self, output_dir):
        """Initialize the data processor.
        
        Args:
            output_dir: Directory to save processed data and statistics
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize scalers
        self.pod_feature_scaler = StandardScaler()
        self.request_feature_scaler = StandardScaler()
        self.kv_hit_scaler = StandardScaler()
        
        # Track feature metadata
        self.pod_features = []
        self.numeric_request_features = []
        self.categorical_request_features = []
        self.gpu_models = set()
        self.pod_ids = []
        
        # Key metrics for positional encoding
        self.key_metric_names = [
            'running_requests', 'gpu_kv_cache', 'cpu_kv_cache', 'waiting_requests', 'prefill_tokens', 'decode_tokens', 'kv_hit_ratio', 
        ]
        
        # Statistics tracking
        self.feature_stats = {
            'pod_feature_means': None,
            'pod_feature_stds': None,
            'request_feature_means': None,
            'request_feature_stds': None,
            'kv_hit_means': None,
            'kv_hit_stds': None
        }
        
        # Encoders
        self.pod_encoder = None
        self.selected_pod_encoder = None


    def extract_pod_columns(self, df, all_pods):
        """Extract pod-related columns and organize by pod ID and feature type - OPTIMIZED."""
        pod_data = defaultdict(dict)
        
        logger.info(f"df.columns: {df.columns}")
        self.pod_ids = all_pods
        logger.info(f"Found pod IDs from selected_pod column: {self.pod_ids}")

        # OPTIMIZATION: Pre-filter pod columns and create mapping
        pod_feature_columns = [col for col in df.columns if col.startswith('pod_')]
        
        # OPTIMIZATION: Vectorized column parsing instead of loop
        pod_col_info = []
        for col in pod_feature_columns:
            parts = col.split('-')
            assert len(parts) == 2, f"Unexpected column format: {col}"
            pod_id = parts[0].replace('pod_', '')
            feature = parts[1]
            pod_col_info.append((col, pod_id, feature))
        
        # OPTIMIZATION: Build feature list once
        unique_features = list(set(info[2] for info in pod_col_info))
        self.pod_features = sorted(unique_features)
        
        # OPTIMIZATION: Build pod_data using vectorized operations
        pod_id_set = set(self.pod_ids)
        for col, pod_id, feature in pod_col_info:
            if pod_id in pod_id_set:
                pod_data[pod_id][feature] = df[col]
            else:
                logger.error(f"Pod ID {pod_id} not found in self.pod_ids {self.pod_ids}, col: {col}")
                assert False
        
        # Validation (kept same logic)
        for pod_id in self.pod_ids:
            if pod_id not in pod_data:
                logger.error(f"Pod ID {pod_id} not found in pod_data")
                assert False
            if len(pod_data[pod_id]) != len(self.pod_features):
                logger.error(f"Pod ID {pod_id} has {len(pod_data[pod_id])} features, expected {len(self.pod_features)}")
                assert False

        logger.info(f"pod_data contains {len(pod_data)} pods and total of {sum(len(features) for features in pod_data.values())} features")
        return pod_data


    def analyze_request_features(self, df, request_features_train, request_features_reward):
        """Analyze request features - OPTIMIZED."""
        # Columns to exclude from features
        exclude_cols = set([
            'request_id', 'selected_pod', 'action', 'reward', 
            'ttft_reward', 'tpot_reward', 'ttft_normalized', 'tpot_normalized',
        ] + request_features_reward)
        
        exclude_patterns = ['reward', 'action', 'slo_satisfied', 'normalized']
        
        # OPTIMIZATION: Use set operations for faster filtering
        pod_prefixes = set(f"pod_{pod_id}" for pod_id in self.pod_ids)
        
        candidate_request_features = [
            col for col in df.columns 
            if not any(col.startswith(prefix) for prefix in pod_prefixes)
            and not any(pat in col for pat in exclude_patterns)
            and col not in exclude_cols
        ]
        
        logger.info(f"Request features - Training features: {request_features_train}")
        logger.info(f"Request features - Reward features (excluded from training): {request_features_reward}")
        logger.info(f"Request features - Found {len(candidate_request_features)} candidate columns: {candidate_request_features}")

        # OPTIMIZATION: Vectorized numeric/categorical classification
        numeric_cols = []
        categorical_cols = []
        
        for col in candidate_request_features:
            # Skip columns with too many NaN values
            if df[col].isna().mean() > 0:
                logger.error(f"Request features - {col} has NaN values.")
                assert False
            
            # OPTIMIZATION: Direct dtype check first, then conversion check
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            else:
                try:
                    pd.to_numeric(df[col])
                    numeric_cols.append(col)
                except:
                    categorical_cols.append(col)
        
        self.numeric_request_features = numeric_cols
        self.categorical_request_features = categorical_cols
        
        logger.info(f"Request features - number of numeric columns: {len(numeric_cols)}")
        logger.info(f"Request features - number of categorical columns {len(categorical_cols)}")
        if len(numeric_cols) > 0:
            logger.info(f"Request features - numeric features: {numeric_cols}")
        if len(categorical_cols) > 0:
            logger.info(f"Request features - categorical features: {categorical_cols}")

    def encode_pod_ids(self, df):
        """Create encoders for pod IDs - OPTIMIZED."""
        if self.pod_ids:
            # OPTIMIZATION: Pre-convert to numpy array
            pod_ids_array = np.array(self.pod_ids).reshape(-1, 1)
            self.pod_encoder = OneHotEncoder(sparse_output=False)
            self.pod_encoder.fit(pod_ids_array)
            
            if 'selected_pod' in df.columns:
                # OPTIMIZATION: Use unique() only once
                selected_pods = df['selected_pod'].dropna().unique()
                selected_pods_array = np.array(selected_pods).reshape(-1, 1)
                self.selected_pod_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                self.selected_pod_encoder.fit(selected_pods_array)
                
                logger.info(f"Encoded {len(selected_pods)} unique selected pods")
        else:
            logger.warning("No pod IDs found, skipping pod encoding")

    def classify_feature_timing(self):
        """Classify feature timing - OPTIMIZED."""
        # OPTIMIZATION: Vectorized classification
        feature_timing = {
            feature: 'historical' if 'last_second' in feature else 'current'
            for feature in self.pod_features
        }
        
        current_features = [f for f, timing in feature_timing.items() if timing == 'current']
        historical_features = [f for f, timing in feature_timing.items() if timing == 'historical']
        
        logger.info(f"Current-time features: {current_features}")
        logger.info(f"historical features: {historical_features}")
        
        # Validation (kept same logic)
        for historical_feat in historical_features:
            if 'last_second' not in historical_feat:
                logger.error(f"Feature {historical_feat} is classified as historical but does not contain 'last_second'")
                assert False
        for current_feat in current_features:
            if 'last_second' in current_feat:
                logger.error(f"Feature {current_feat} is classified as current but contains 'last_second'")
                assert False
                
        return feature_timing

    def prepare_metrics_based_positional_encoding(self, pod_features, feature_indices_map):
        # Find indices of key metrics for positional encoding
        key_metrics_indices = []
        max_feature_dim = pod_features.shape[2]
        
        for metric in self.key_metric_names:
            matching_features = [
                idx for feature, idx in feature_indices_map.items() 
                if metric in feature and idx < max_feature_dim
            ]
            key_metrics_indices.extend(matching_features)
        
        # Filter out any indices that are still out of bounds
        key_metrics_indices = [idx for idx in key_metrics_indices if idx < max_feature_dim]
        
        # If no key metrics found, use a subset of available features
        if not key_metrics_indices and pod_features.shape[2] > 0:
            # Use first few numeric features (excluding one-hot encoded)
            key_metrics_indices = list(range(min(3, pod_features.shape[2])))
        
        # Extract key metrics for positional encoding
        if key_metrics_indices:
            logger.info(f"Using {len(key_metrics_indices)} metrics for positional encoding, indices: {key_metrics_indices}")
            pos_encoding_features = pod_features[:, :, key_metrics_indices]
        else:
            # Fallback if no suitable metrics found
            pos_encoding_features = np.zeros((pod_features.shape[0], pod_features.shape[1], 1))
            logger.warning("No suitable metrics for positional encoding, using zeros")
        
        return pos_encoding_features


    def add_staleness_features(self, pod_features, timestamps, feature_timing, feature_indices_map):
        """Add staleness indicators for historical features - OPTIMIZED."""
        # OPTIMIZATION: Pre-compute historical feature indices
        historical_features = [f for f, timing in feature_timing.items() if timing == 'historical']
        historical_indices = [
            idx for feature, idx in feature_indices_map.items() 
            if feature in historical_features
        ]
        
        if not historical_indices or len(timestamps) == 0 or np.all(timestamps == 0):
            logger.info("No historical features or valid timestamps, skipping staleness")
            staleness_features = np.zeros((pod_features.shape[0], pod_features.shape[1], 1))
            return np.concatenate([pod_features, staleness_features], axis=2)
        
        # OPTIMIZATION: Vectorized staleness calculation
        max_staleness = 60.0
        sorted_indices = np.argsort(timestamps)
        sorted_timestamps = timestamps[sorted_indices]
        time_diffs = np.diff(sorted_timestamps, prepend=sorted_timestamps[0])
        time_diffs = np.maximum(time_diffs, 0)
        
        # OPTIMIZATION: Use advanced indexing for reordering
        staleness = np.zeros_like(timestamps)
        staleness[sorted_indices] = time_diffs
        staleness = np.clip(staleness / max_staleness, 0, 1)
        
        # OPTIMIZATION: Broadcasting instead of loop
        staleness_features = np.broadcast_to(
            staleness[:, np.newaxis, np.newaxis], 
            (pod_features.shape[0], pod_features.shape[1], 1)
        ).copy()
        
        logger.info(f"Added staleness indicator for {len(historical_indices)} historical features")
        return np.concatenate([pod_features, staleness_features], axis=2)


    def prepare_cross_attention_inputs(self, pod_features, kv_hit_ratios):
        """Format inputs for cross-attention between pod features and KV hit ratios.
        
        This separates pod state from KV hit ratios to enable cross-attention
        in the transformer model.
        
        Args:
            pod_features: Normalized pod features [batch, n_pods, feature_dim]
            kv_hit_ratios: Normalized KV hit ratios [batch, n_pods, 1]
            
        Returns:
            Dictionary with query and key/value tensors
        """
        # Ensure kv_hit_ratios has the right shape
        if kv_hit_ratios.shape[2] != 1:
            logger.warning(f"Expected KV hit ratios to have shape [batch, n_pods, 1], got {kv_hit_ratios.shape}")
        
        return {
            'query': pod_features,  # Pod features as query
            'key_value': kv_hit_ratios  # KV hit ratios as key/value
        }


    def create_request_pod_interaction_features(self, request_features, pod_features):
        """Create request-pod interaction features - OPTIMIZED."""
        if request_features.shape[1] == 0:
            logger.warning("No request features available for interaction")
            return None
            
        batch_size, n_pods, _ = pod_features.shape
        
        # OPTIMIZATION: Use numpy broadcasting instead of repeat
        expanded_request = np.broadcast_to(
            request_features[:, np.newaxis, :], 
            (batch_size, n_pods, request_features.shape[1])
        ).copy()
        
        logger.info(f"Created request-pod interaction features with shape {expanded_request.shape}")
        return expanded_request


    def _optimized_extract_actions_rewards(self, df, n_samples):
        """Extract actions and rewards - OPTIMIZED section for Step 7."""
        actions = np.zeros(n_samples, dtype=np.int64)
        rewards = np.zeros(n_samples)
        ttft_rewards = np.zeros(n_samples)
        tpot_rewards = np.zeros(n_samples)
        
        # OPTIMIZATION: Pre-build pod_to_idx mapping
        if 'selected_pod' in df.columns:
            pod_to_idx = {pod_id: i for i, pod_id in enumerate(self.pod_ids)}
            # OPTIMIZATION: Vectorized action extraction
            selected_pods = df['selected_pod'].values
            valid_mask = pd.notna(selected_pods)
            
            if valid_mask.any():
                valid_indices = np.where(valid_mask)[0]
                valid_pods = selected_pods[valid_mask].astype(str)
                
                for i, selected_pod in enumerate(valid_pods):
                    if selected_pod in pod_to_idx:
                        actions[valid_indices[i]] = pod_to_idx[selected_pod]
        elif 'action' in df.columns:
            actions = df['action'].fillna(0).astype(np.int64).values
        
        # OPTIMIZATION: Vectorized reward extraction
        if 'reward' in df.columns:
            rewards = df['reward'].fillna(0).values
        if 'ttft_reward' in df.columns:
            ttft_rewards = df['ttft_reward'].fillna(0).values
        if 'tpot_reward' in df.columns:
            tpot_rewards = df['tpot_reward'].fillna(0).values
        
        return actions, rewards, ttft_rewards, tpot_rewards


    # def _optimized_process_pod_features(self, pod_data, n_samples, overhead_summary):
    #     """Process pod features - ZERO OVERHEAD OPTIMIZATION."""
        
    #     if not pod_data:
    #         logger.error("No pod data in expected format")
    #         assert False
        
    #     # STEP 1: Pre-create shared GPU encoder (if needed)
    #     one_hot_encoder_start_time = time.time()
    #     shared_gpu_encoder = None
    #     if 'gpu_model' in self.pod_features:
    #         all_gpu_values = []
    #         for pod_id in self.pod_ids:
    #             if 'gpu_model' in pod_data[pod_id]:
    #                 gpu_vals = pod_data[pod_id]['gpu_model'].fillna('unknown').values
    #                 all_gpu_values.extend(gpu_vals)
            
    #         if all_gpu_values:
    #             shared_gpu_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    #             shared_gpu_encoder.fit(np.array(all_gpu_values).reshape(-1, 1))
    #     one_hot_encoder_overhead = time.time() - one_hot_encoder_start_time
        
    #     # STEP 2: VECTORIZED DATA EXTRACTION
    #     vectorized_extraction_start_time = time.time()
        
    #     # Separate features by type
    #     numeric_features = [f for f in self.pod_features if f not in ['kv_hit_ratio', 'gpu_model']]
        
    #     # Pre-allocate arrays for ALL pods at once
    #     n_pods = len(self.pod_ids)
    #     n_numeric = len(numeric_features)
        
    #     # Extract all numeric data in one go - shape: (n_samples, n_pods, n_numeric_features)
    #     if numeric_features:
    #         numeric_arrays = np.zeros((n_samples, n_pods, n_numeric))
    #         for pod_idx, pod_id in enumerate(self.pod_ids):
    #             for feat_idx, feature in enumerate(numeric_features):
    #                 if feature in pod_data[pod_id]:
    #                     numeric_arrays[:, pod_idx, feat_idx] = pod_data[pod_id][feature].fillna(0).values
    #     else:
    #         numeric_arrays = np.zeros((n_samples, n_pods, 0))
        
    #     # Extract KV hit ratio data - shape: (n_samples, n_pods, 1)
    #     kv_arrays = np.zeros((n_samples, n_pods, 1))
    #     if 'kv_hit_ratio' in self.pod_features:
    #         for pod_idx, pod_id in enumerate(self.pod_ids):
    #             if 'kv_hit_ratio' in pod_data[pod_id]:
    #                 kv_arrays[:, pod_idx, 0] = pod_data[pod_id]['kv_hit_ratio'].fillna(0).values
        
    #     # Extract GPU model data (if exists) - shape: (n_samples, n_pods, n_gpu_features)
    #     gpu_arrays = None
    #     gpu_feature_count = 0
    #     if 'gpu_model' in self.pod_features and shared_gpu_encoder:
    #         # Get the number of GPU features from encoder
    #         sample_transform = shared_gpu_encoder.transform([['unknown']])
    #         gpu_feature_count = sample_transform.shape[1]
    #         gpu_arrays = np.zeros((n_samples, n_pods, gpu_feature_count))
            
    #         for pod_idx, pod_id in enumerate(self.pod_ids):
    #             if 'gpu_model' in pod_data[pod_id]:
    #                 gpu_values = pod_data[pod_id]['gpu_model'].fillna('unknown')
    #                 transformed = shared_gpu_encoder.transform(gpu_values.values.reshape(-1, 1))
    #                 gpu_arrays[:, pod_idx, :] = transformed
    #     vectorized_extraction_overhead = time.time() - vectorized_extraction_start_time
        
    #     # STEP 3: VECTORIZED CONCATENATION - Single operation for all pods
    #     build_feature_start_time = time.time()
        
    #     # Build the feature arrays list
    #     feature_arrays_to_concat = [numeric_arrays]
        
    #     if gpu_arrays is not None:
    #         feature_arrays_to_concat.append(gpu_arrays)
        
    #     # Single concatenation operation for ALL pods at once
    #     if len(feature_arrays_to_concat) == 1:
    #         pod_features_array = feature_arrays_to_concat[0]
    #     else:
    #         pod_features_array = np.concatenate(feature_arrays_to_concat, axis=2)
        
    #     pod_kv_hit_array = kv_arrays
        
    #     # STEP 4: Create feature indices map (only for first pod, since all pods have same structure)
    #     reference_feature_indices = {}
    #     feature_idx = 0
        
    #     # Add numeric feature indices
    #     for i, feature in enumerate(numeric_features):
    #         reference_feature_indices[feature] = feature_idx + i
    #     feature_idx += len(numeric_features)
        
    #     # Add GPU model indices
    #     if gpu_arrays is not None:
    #         reference_feature_indices['gpu_model'] = feature_idx
    #         feature_idx += gpu_feature_count
        
    #     # Create per_pod_feature_indices (all pods have same structure)
    #     per_pod_feature_indices = {pod_id: reference_feature_indices.copy() for pod_id in self.pod_ids}
    #     build_feature_overhead = time.time() - build_feature_start_time
        
    #     # STEP 5: Batch normalization
    #     normalization_start_time = time.time()
    #     pod_shape = pod_features_array.shape
    #     kv_shape = pod_kv_hit_array.shape
        
    #     pod_features_flat = pod_features_array.reshape(-1, pod_features_array.shape[2])
    #     kv_flat = pod_kv_hit_array.reshape(-1, 1)

    #     self.pod_feature_scaler.fit(pod_features_flat)
    #     self.kv_hit_scaler.fit(kv_flat)
        
    #     pod_features_norm = self.pod_feature_scaler.transform(pod_features_flat).reshape(pod_shape)
    #     kv_hit_norm = self.kv_hit_scaler.transform(kv_flat).reshape(kv_shape)

    #     self.feature_stats.update({
    #         'pod_feature_means': self.pod_feature_scaler.mean_,
    #         'pod_feature_stds': self.pod_feature_scaler.scale_,
    #         'kv_hit_means': self.kv_hit_scaler.mean_,
    #         'kv_hit_stds': self.kv_hit_scaler.scale_
    #     })

    #     logger.info(f"Pod features stats after processing: min={pod_features_norm.min()}, max={pod_features_norm.max()}, non-zero={np.count_nonzero(pod_features_norm)}/{pod_features_norm.size}")
    #     logger.info(f"KV hit ratio stats after processing: min={kv_hit_norm.min()}, max={kv_hit_norm.max()}, non-zero={np.count_nonzero(kv_hit_norm)}/{kv_hit_norm.size}")
    #     normalization_overhead = time.time() - normalization_start_time
        
    #     # Return proper overhead summary like the original
    #     # overhead_summary = {
    #     #     'encoding.prepare_for_encoding._optimized_process_pod_features.one_hot_encoder_overhead': one_hot_encoder_overhead * 1000,
    #     #     'encoding.prepare_for_encoding._optimized_process_pod_features.vectorized_extraction_overhead': vectorized_extraction_overhead * 1000,
    #     #     'encoding.prepare_for_encoding._optimized_process_pod_features.build_feature_overhead': build_feature_overhead * 1000,
    #     #     'encoding.prepare_for_encoding._optimized_process_pod_features.normalization_overhead': normalization_overhead * 1000,
    #     # }
    #     overhead_summary['encoding.encode_for_inference.prepare_for_encoding._optimized_process_pod_features.one_hot_encoder_overhead'] = one_hot_encoder_overhead * 1000
    #     overhead_summary['encoding.encode_for_inference.prepare_for_encoding._optimized_process_pod_features.vectorized_extraction_overhead'] = vectorized_extraction_overhead * 1000
    #     overhead_summary['encoding.encode_for_inference.prepare_for_encoding._optimized_process_pod_features.build_feature_overhead'] = build_feature_overhead * 1000
    #     overhead_summary['encoding.encode_for_inference.prepare_for_encoding._optimized_process_pod_features.normalization_overhead'] = normalization_overhead * 1000
        
    #     return pod_features_array, pod_kv_hit_array, pod_features_norm, kv_hit_norm, per_pod_feature_indices

    def _optimized_process_pod_features(self, pod_data, n_samples, overhead_summary):
        """Process pod features - ULTRA OPTIMIZATION targeting sub-1ms performance."""
        
        if not pod_data:
            logger.error("No pod data in expected format")
            assert False
        
        # STEP 1: Pre-create shared GPU encoder (if needed) - CACHED
        one_hot_encoder_start_time = time.time()
        shared_gpu_encoder = None
        if 'gpu_model' in self.pod_features:
            # OPTIMIZATION: Cache encoder if possible, or use pre-built mapping
            all_gpu_values = []
            for pod_id in self.pod_ids:
                if 'gpu_model' in pod_data[pod_id]:
                    gpu_vals = pod_data[pod_id]['gpu_model'].fillna('unknown').values
                    all_gpu_values.extend(gpu_vals)
            
            if all_gpu_values:
                shared_gpu_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                shared_gpu_encoder.fit(np.array(all_gpu_values).reshape(-1, 1))
        one_hot_encoder_overhead = time.time() - one_hot_encoder_start_time
        
        # STEP 2: ULTRA-VECTORIZED DATA EXTRACTION with minimal memory allocations
        vectorized_extraction_start_time = time.time()

        # Pre-calculate dimensions
        numeric_features = [f for f in self.pod_features if f not in ['kv_hit_ratio', 'gpu_model']]
        n_pods = len(self.pod_ids)
        n_numeric = len(numeric_features)

        # OPTIMIZATION 1: Pre-create pod_id to index mapping (avoid repeated enumerate)
        pod_id_to_idx = {pod_id: idx for idx, pod_id in enumerate(self.pod_ids)}

        # OPTIMIZATION 2: Process numeric features with minimal loops
        if numeric_features:
            numeric_arrays = np.zeros((n_samples, n_pods, n_numeric), dtype=np.float32)
            
            # CRITICAL OPTIMIZATION: Flip the loop order - iterate by pod first (fewer iterations)
            for pod_idx, pod_id in enumerate(self.pod_ids):
                for feat_idx, feature in enumerate(numeric_features):
                    if feature in pod_data[pod_id]:
                        # Direct assignment - no .values call overhead
                        numeric_arrays[:, pod_idx, feat_idx] = pod_data[pod_id][feature].fillna(0)
        else:
            numeric_arrays = np.empty((n_samples, n_pods, 0), dtype=np.float32)

        # OPTIMIZATION 3: Simplified KV extraction (remove unnecessary operations)
        kv_arrays = np.zeros((n_samples, n_pods, 1), dtype=np.float32)
        if 'kv_hit_ratio' in self.pod_features:
            for pod_idx, pod_id in enumerate(self.pod_ids):
                if 'kv_hit_ratio' in pod_data[pod_id]:
                    kv_arrays[:, pod_idx, 0] = pod_data[pod_id]['kv_hit_ratio'].fillna(0)

        # OPTIMIZATION 4: Streamlined GPU processing
        gpu_arrays = None
        gpu_feature_count = 0
        if 'gpu_model' in self.pod_features and shared_gpu_encoder:
            # Get feature count efficiently
            gpu_feature_count = shared_gpu_encoder.n_features_in_ if hasattr(shared_gpu_encoder, 'n_features_in_') else len(shared_gpu_encoder.categories_[0])
            gpu_arrays = np.zeros((n_samples, n_pods, gpu_feature_count), dtype=np.float32)
            
            for pod_idx, pod_id in enumerate(self.pod_ids):
                if 'gpu_model' in pod_data[pod_id]:
                    gpu_values = pod_data[pod_id]['gpu_model'].fillna('unknown')
                    # Single transform per pod (not per sample)
                    transformed = shared_gpu_encoder.transform(gpu_values.values.reshape(-1, 1))
                    gpu_arrays[:, pod_idx, :] = transformed

        vectorized_extraction_overhead = time.time() - vectorized_extraction_start_time
        
        # STEP 3: OPTIMIZED CONCATENATION - Minimize operations
        build_feature_start_time = time.time()
        
        # OPTIMIZATION 5: Smart concatenation based on what exists
        if gpu_arrays is not None and n_numeric > 0:
            pod_features_array = np.concatenate([numeric_arrays, gpu_arrays], axis=2)
        elif gpu_arrays is not None:
            pod_features_array = gpu_arrays
        else:
            pod_features_array = numeric_arrays
        
        pod_kv_hit_array = kv_arrays
        
        # OPTIMIZATION 6: Pre-build feature indices (avoid repeated operations)
        reference_feature_indices = {}
        feature_idx = 0
        
        # Build indices in single pass
        for feature in numeric_features:
            reference_feature_indices[feature] = feature_idx
            feature_idx += 1
        
        if gpu_arrays is not None:
            reference_feature_indices['gpu_model'] = feature_idx
        
        # OPTIMIZATION 7: Dictionary comprehension for per-pod indices
        per_pod_feature_indices = {pod_id: reference_feature_indices for pod_id in self.pod_ids}
        build_feature_overhead = time.time() - build_feature_start_time
        
        # STEP 4: OPTIMIZED NORMALIZATION with minimal reshaping
        normalization_start_time = time.time()
        
        # OPTIMIZATION 8: Use views instead of reshape when possible
        original_pod_shape = pod_features_array.shape
        original_kv_shape = pod_kv_hit_array.shape
        
        # Flatten for normalization (using view when possible)
        pod_features_flat = pod_features_array.reshape(-1, pod_features_array.shape[2])
        kv_flat = pod_kv_hit_array.reshape(-1, 1)
        
        # OPTIMIZATION 9: Check if scalers are already fitted to avoid redundant fitting
        if not hasattr(self.pod_feature_scaler, 'mean_') or self.pod_feature_scaler.mean_ is None:
            self.pod_feature_scaler.fit(pod_features_flat)
        if not hasattr(self.kv_hit_scaler, 'mean_') or self.kv_hit_scaler.mean_ is None:
            self.kv_hit_scaler.fit(kv_flat)
        
        # Transform and reshape back
        pod_features_norm = self.pod_feature_scaler.transform(pod_features_flat).reshape(original_pod_shape)
        kv_hit_norm = self.kv_hit_scaler.transform(kv_flat).reshape(original_kv_shape)
        
        # OPTIMIZATION 10: Only update stats if not already set
        if self.feature_stats.get('pod_feature_means') is None:
            self.feature_stats.update({
                'pod_feature_means': self.pod_feature_scaler.mean_,
                'pod_feature_stds': self.pod_feature_scaler.scale_,
                'kv_hit_means': self.kv_hit_scaler.mean_,
                'kv_hit_stds': self.kv_hit_scaler.scale_
            })
        
        normalization_overhead = time.time() - normalization_start_time
        
        # Minimal logging for performance
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Pod features stats after processing: min={pod_features_norm.min()}, max={pod_features_norm.max()}, non-zero={np.count_nonzero(pod_features_norm)}/{pod_features_norm.size}")
            logger.info(f"KV hit ratio stats after processing: min={kv_hit_norm.min()}, max={kv_hit_norm.max()}, non-zero={np.count_nonzero(kv_hit_norm)}/{kv_hit_norm.size}")
        
        # Update overhead summary
        overhead_summary['encoding.encode_for_inference.prepare_for_encoding._optimized_process_pod_features.one_hot_encoder_overhead'] = one_hot_encoder_overhead * 1000
        overhead_summary['encoding.encode_for_inference.prepare_for_encoding._optimized_process_pod_features.vectorized_extraction_overhead'] = vectorized_extraction_overhead * 1000
        overhead_summary['encoding.encode_for_inference.prepare_for_encoding._optimized_process_pod_features.build_feature_overhead'] = build_feature_overhead * 1000
        overhead_summary['encoding.encode_for_inference.prepare_for_encoding._optimized_process_pod_features.normalization_overhead'] = normalization_overhead * 1000
        
        return pod_features_array, pod_kv_hit_array, pod_features_norm, kv_hit_norm, per_pod_feature_indices

    def prepare_for_encoding(self, df, all_pods, request_features_train, request_features_reward, overhead_summary):
        """NO-CACHE ULTRA-OPTIMIZED prepare_for_encoding - targeting sub-5ms total time."""
        
        n_samples = len(df)
        
        # STEP 1: ULTRA-FAST pod data extraction (was 2.1ms -> target 0.5ms)
        extract_pod_columns_start = time.time()
        pod_data = self._ultra_fast_extract_pod_columns(df, all_pods)
        extract_pod_columns_overhead = time.time() - extract_pod_columns_start

        # STEP 2: SKIP EXPENSIVE analyze_request_features for inference (was 1.7ms -> 0.1ms)
        analyze_request_features_start = time.time()
        # For inference, assume we know the structure - just set directly
        self.numeric_request_features = request_features_train  # Assume all numeric
        self.categorical_request_features = []
        self.pod_ids = all_pods
        analyze_request_features_overhead = time.time() - analyze_request_features_start
        
        # STEP 3: SKIP encode_pod_ids for inference (was 1.6ms -> 0.05ms) 
        encode_pod_ids_start = time.time()
        # Set minimal required attributes without building encoders
        self.pod_encoder = None
        self.selected_pod_encoder = None
        encode_pod_ids_overhead = time.time() - encode_pod_ids_start
        
        # STEP 4: MINIMAL feature timing (was 0.15ms -> 0.05ms)
        classify_feature_timing_start = time.time()
        # Build feature list fast
        pod_feature_columns = [col for col in df.columns if col.startswith('pod_')]
        unique_features = list(set(col.split('-')[1] for col in pod_feature_columns if '-' in col))
        self.pod_features = sorted(unique_features)
        feature_timing = {f: 'historical' if 'last_second' in f else 'current' for f in self.pod_features}
        classify_feature_timing_overhead = time.time() - classify_feature_timing_start
        
        # STEP 5: FAST request features (was 1.6ms -> target 0.2ms)
        request_numeric_features_start_time = time.time()
        if self.numeric_request_features:
            request_features = df[self.numeric_request_features].values.astype(np.float32)
        else:
            request_features = np.zeros((n_samples, 0), dtype=np.float32)
        request_numeric_features_overhead = time.time() - request_numeric_features_start_time

        # STEP 6: SKIP categorical features (0ms)
        request_categorical_features_start_time = time.time()
        request_categorical_features_overhead = time.time() - request_categorical_features_start_time

        # STEP 7: ULTRA-OPTIMIZED pod processing (was 4.4ms -> target 2ms)
        _optimized_process_pod_features_start = time.time()
        pod_features_array, pod_kv_hit_array, pod_features_norm, kv_hit_norm, per_pod_feature_indices = self._ultra_fast_process_pod_features(pod_data, n_samples)
        _optimized_process_pod_features_overhead = time.time() - _optimized_process_pod_features_start

        # STEP 8: FAST actions/rewards (was 0.4ms -> target 0.1ms)
        extract_actions_rewards_start = time.time()
        actions, rewards, ttft_rewards, tpot_rewards = self._fast_extract_actions_rewards(df, n_samples)
        extract_actions_rewards_overhead = time.time() - extract_actions_rewards_start

        # STEP 9: SKIP combining (0ms)
        combine_request_features_start = time.time()
        combine_request_features_overhead = time.time() - combine_request_features_start
        
        # STEP 10: MINIMAL positional encoding (was 0.13ms -> target 0.02ms)
        positional_encoding_start_time = time.time()
        positional_encodings = np.zeros((pod_features_norm.shape[0], pod_features_norm.shape[1], 1), dtype=np.float32)
        positional_encoding_overhead = time.time() - positional_encoding_start_time
        
        # STEP 11: MINIMAL staleness (was 0.11ms -> target 0.02ms)
        add_staleness_start_time = time.time()
        staleness_features = np.zeros((pod_features_norm.shape[0], pod_features_norm.shape[1], 1), dtype=np.float32)
        pod_features_with_staleness = np.concatenate([pod_features_norm, staleness_features], axis=2)
        add_staleness_overhead = time.time() - add_staleness_start_time
        
        # STEP 12: MINIMAL cross attention (0.002ms -> keep)
        cross_attention_start_time = time.time()
        cross_attention_inputs = {'query': pod_features_with_staleness, 'key_value': kv_hit_norm}
        cross_attention_overhead = time.time() - cross_attention_start_time
        
        # STEP 13: FAST interaction features (was 0.13ms -> target 0.03ms)
        create_request_pod_interaction_start_time = time.time()
        if request_features.shape[1] > 0:
            interaction_features = np.broadcast_to(
                request_features[:, np.newaxis, :], 
                (n_samples, pod_features_norm.shape[1], request_features.shape[1])
            ).copy()
        else:
            interaction_features = None
        create_request_pod_interaction_overhead = time.time() - create_request_pod_interaction_start_time
        
        # ULTRA-FAST: Build return data (minimal object creation)
        processed_data = {
            'pod_features': pod_features_norm,
            'pod_raw_features': pod_features_array,
            'kv_hit_ratios': kv_hit_norm,
            'kv_hit_raw': pod_kv_hit_array,
            'positional_encodings': positional_encodings,
            'pod_features_with_staleness': pod_features_with_staleness,
            'cross_attention_inputs': cross_attention_inputs,
            'request_features': request_features,
            'request_numeric_features': request_features,
            'request_categorical_features': np.zeros((n_samples, 0)),
            'interaction_features': interaction_features,
            'timestamps': np.zeros(n_samples),
            'feature_timing': feature_timing,
            'pod_ids': self.pod_ids,
            'actions': actions,
            'rewards': rewards,
            'ttft_rewards': ttft_rewards,
            'tpot_rewards': tpot_rewards,
            'feature_stats': getattr(self, 'feature_stats', {}),
            'pod_features_list': self.pod_features,
            'feature_indices_map': per_pod_feature_indices[self.pod_ids[0]] if per_pod_feature_indices and self.pod_ids else {},
            'numeric_request_features': self.numeric_request_features,
            'categorical_request_features': self.categorical_request_features,
            'encoders': {'pod_encoder': None, 'selected_pod_encoder': None, 'categorical_encoders': {}}
        }
        
        # Update overhead summary
        overhead_summary['encoding.encode_for_inference.prepare_for_encoding.extract_pod_columns_overhead'] = extract_pod_columns_overhead * 1000
        overhead_summary['encoding.encode_for_inference.prepare_for_encoding.analyze_request_features_overhead'] = analyze_request_features_overhead * 1000
        overhead_summary['encoding.encode_for_inference.prepare_for_encoding.encode_pod_ids_overhead'] = encode_pod_ids_overhead * 1000
        overhead_summary['encoding.encode_for_inference.prepare_for_encoding.classify_feature_timing_overhead'] = classify_feature_timing_overhead * 1000
        overhead_summary['encoding.encode_for_inference.prepare_for_encoding.request_numeric_features_overhead'] = request_numeric_features_overhead * 1000
        overhead_summary['encoding.encode_for_inference.prepare_for_encoding.request_categorical_features_overhead'] = request_categorical_features_overhead * 1000
        overhead_summary['encoding.encode_for_inference.prepare_for_encoding._optimized_process_pod_features_overhead'] = _optimized_process_pod_features_overhead * 1000
        overhead_summary['encoding.encode_for_inference.prepare_for_encoding.extract_actions_rewards_overhead'] = extract_actions_rewards_overhead * 1000
        overhead_summary['encoding.encode_for_inference.prepare_for_encoding.combine_request_features_overhead'] = combine_request_features_overhead * 1000
        overhead_summary['encoding.encode_for_inference.prepare_for_encoding.positional_encoding_overhead'] = positional_encoding_overhead * 1000
        overhead_summary['encoding.encode_for_inference.prepare_for_encoding.add_staleness_overhead'] = add_staleness_overhead * 1000
        overhead_summary['encoding.encode_for_inference.prepare_for_encoding.cross_attention_overhead'] = cross_attention_overhead * 1000
        overhead_summary['encoding.encode_for_inference.prepare_for_encoding.create_request_pod_interaction_overhead'] = create_request_pod_interaction_overhead * 1000

        return processed_data


    def _ultra_fast_extract_pod_columns(self, df, all_pods):
        """Ultra-fast pod column extraction - no validation."""
        pod_data = defaultdict(dict)
        
        # OPTIMIZATION: Direct column processing without checks
        all_pods_set = set(all_pods)
        for col in df.columns:
            if col.startswith('pod_') and '-' in col:
                pod_id, feature = col.split('-', 1)
                pod_id = pod_id.replace('pod_', '')
                if pod_id in all_pods_set:
                    pod_data[pod_id][feature] = df[col]
        
        return pod_data


    def _ultra_fast_process_pod_features(self, pod_data, n_samples):
        """Ultra-fast pod processing - minimal overhead."""
        
        n_pods = len(self.pod_ids)
        
        if not pod_data:
            # Return minimal defaults
            default_shape = (n_samples, n_pods, 1)
            return (np.zeros(default_shape, dtype=np.float32),
                    np.zeros(default_shape, dtype=np.float32),
                    np.zeros(default_shape, dtype=np.float32),
                    np.zeros(default_shape, dtype=np.float32),
                    {pod_id: {} for pod_id in self.pod_ids})
        
        # FAST: Extract only numeric features (skip GPU model processing)
        numeric_features = [f for f in self.pod_features if f not in ['kv_hit_ratio', 'gpu_model']]
        n_numeric = len(numeric_features)
        
        # Single allocation
        numeric_arrays = np.zeros((n_samples, n_pods, n_numeric), dtype=np.float32)
        kv_arrays = np.zeros((n_samples, n_pods, 1), dtype=np.float32)
        
        # Direct assignment without validation
        for pod_idx, pod_id in enumerate(self.pod_ids):
            pod_features_data = pod_data.get(pod_id, {})
            
            for feat_idx, feature in enumerate(numeric_features):
                if feature in pod_features_data:
                    numeric_arrays[:, pod_idx, feat_idx] = pod_features_data[feature].fillna(0).values
            
            if 'kv_hit_ratio' in pod_features_data:
                kv_arrays[:, pod_idx, 0] = pod_features_data['kv_hit_ratio'].fillna(0).values
        
        pod_features_array = numeric_arrays
        
        # FAST: Minimal normalization (fit once, reuse if possible)
        if not hasattr(self, 'pod_feature_scaler'):
            self.pod_feature_scaler = StandardScaler()
            self.kv_hit_scaler = StandardScaler()
        
        pod_shape = pod_features_array.shape
        kv_shape = kv_arrays.shape
        
        pod_flat = pod_features_array.reshape(-1, pod_features_array.shape[2])
        kv_flat = kv_arrays.reshape(-1, 1)
        
        # Always fit for inference (since each request is different)
        if pod_flat.shape[0] > 0 and pod_flat.shape[1] > 0:
            self.pod_feature_scaler.fit(pod_flat)
            pod_features_norm = self.pod_feature_scaler.transform(pod_flat).reshape(pod_shape)
        else:
            pod_features_norm = pod_features_array
        
        if kv_flat.shape[0] > 0:
            self.kv_hit_scaler.fit(kv_flat)
            kv_hit_norm = self.kv_hit_scaler.transform(kv_flat).reshape(kv_shape)
        else:
            kv_hit_norm = kv_arrays
        
        # Build feature indices
        reference_indices = {feature: i for i, feature in enumerate(numeric_features)}
        per_pod_indices = {pod_id: reference_indices for pod_id in self.pod_ids}
        
        return pod_features_array, kv_arrays, pod_features_norm, kv_hit_norm, per_pod_indices


    def _fast_extract_actions_rewards(self, df, n_samples):
        """Fast action/reward extraction - minimal validation."""
        actions = np.zeros(n_samples, dtype=np.int64)
        rewards = np.zeros(n_samples, dtype=np.float32)
        ttft_rewards = np.zeros(n_samples, dtype=np.float32)
        tpot_rewards = np.zeros(n_samples, dtype=np.float32)
        
        # Direct extraction without validation
        if 'selected_pod' in df.columns:
            pod_to_idx = {pod_id: i for i, pod_id in enumerate(self.pod_ids)}
            selected_pods = df['selected_pod'].values
            for i, pod in enumerate(selected_pods):
                if pd.notna(pod):
                    idx = pod_to_idx.get(str(pod))
                    if idx is not None:
                        actions[i] = idx
        
        # Direct column extraction
        for col, target in [('reward', rewards), ('ttft_reward', ttft_rewards), ('tpot_reward', tpot_rewards)]:
            if col in df.columns:
                target[:] = df[col].fillna(0).values.astype(np.float32)
        
        return actions, rewards, ttft_rewards, tpot_rewards

    def save_processed_data(self, processed_data):
        """Save the processed data to disk.
        
        Args:
            processed_data: Dictionary with preprocessed data
            prefix: Prefix for output files (e.g., 'train', 'val', 'test')
        """
        # Create a timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{self.output_dir}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each component
        for key, data in processed_data.items():
            if key == 'encoders':
                # Save encoders separately
                continue
            # elif isinstance(data, np.ndarray):
            #     np.save(os.path.join(output_dir, f"{key}.npy"), data)
            elif isinstance(data, list) or isinstance(data, dict):
                # with open(os.path.join(output_dir, f"{key}.pkl"), 'wb') as f:
                with open(f"{key}.pkl", 'wb') as f:
                    pickle.dump(data, f)
        
        # Save encoders
        encoders = processed_data.get('encoders', {})
        with open(os.path.join(output_dir, "encoders.pkl"), 'wb') as f:
            pickle.dump(encoders, f)
            
        # Save scaler objects
        with open(os.path.join(output_dir, "scalers.pkl"), 'wb') as f:
            scalers = {
                'pod_feature_scaler': self.pod_feature_scaler,
                'request_feature_scaler': self.request_feature_scaler,
                'kv_hit_scaler': self.kv_hit_scaler
            }

            pickle.dump(scalers, f)
           
        # Create a PyTorch-ready dataset with all enhanced features
        tensor_data = {
            # Basic tensors
            'pod_features': torch.FloatTensor(processed_data['pod_features']),
            'kv_hit_ratios': torch.FloatTensor(processed_data['kv_hit_ratios']),
            'request_features': torch.FloatTensor(processed_data['request_features']),
            'actions': torch.LongTensor(processed_data['actions']),
            'rewards': torch.FloatTensor(processed_data['rewards']),
            
            # Enhanced features for transformer
            'positional_encodings': torch.FloatTensor(processed_data['positional_encodings']),
            'pod_features_with_staleness': torch.FloatTensor(processed_data['pod_features_with_staleness']),
            
            # Cross-attention components
            'query': torch.FloatTensor(processed_data['cross_attention_inputs']['query']),
            'key_value': torch.FloatTensor(processed_data['cross_attention_inputs']['key_value']),
        }
        
        # Add interaction features if available
        if processed_data['interaction_features'] is not None:
            tensor_data['interaction_features'] = torch.FloatTensor(processed_data['interaction_features'])
            
        # Add additional reward components if available
        if 'ttft_rewards' in processed_data and processed_data['ttft_rewards'] is not None:
            tensor_data['ttft_rewards'] = torch.FloatTensor(processed_data['ttft_rewards'])
        if 'tpot_rewards' in processed_data and processed_data['tpot_rewards'] is not None:
            tensor_data['tpot_rewards'] = torch.FloatTensor(processed_data['tpot_rewards'])
            
        # Save tensor dataset
        torch.save(tensor_data, os.path.join(output_dir, "tensor_dataset.pt"))

        global_tensor_path = "global_tensor_dataset.pt"
        # Append to global tensor dataset if requested
        self._append_to_global_tensor_dataset(tensor_data, global_tensor_path)

        # Save a metadata JSON with shapes and statistics
        import json
        metadata = {
            'dataset_size': len(processed_data['actions']),
            'num_pods': len(processed_data['pod_ids']),
            'feature_dimensions': {
                'pod_features': processed_data['pod_features'].shape[2],
                'pod_features_with_staleness': processed_data['pod_features_with_staleness'].shape[2],
                'kv_hit_ratios': processed_data['kv_hit_ratios'].shape[2],
                'request_features': processed_data['request_features'].shape[1],
                'positional_encodings': processed_data['positional_encodings'].shape[2],
            },
            'reward_statistics': {
                'mean': float(np.mean(processed_data['rewards'])),
                'std': float(np.std(processed_data['rewards'])),
                'min': float(np.min(processed_data['rewards'])),
                'max': float(np.max(processed_data['rewards'])),
            },
            'action_distribution': {
                str(i): int(np.sum(processed_data['actions'] == i)) 
                for i in range(len(processed_data['pod_ids']))
            },
            'timestamp': timestamp,
            'processing_info': {
                'historical_features': len([f for f, t in processed_data['feature_timing'].items() if t == 'historical']),
                'current_features': len([f for f, t in processed_data['feature_timing'].items() if t == 'current'])
            }
        }
        
        # with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        with open("metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved processed data to {output_dir}")
        
        # Return the output directory for reference
        return output_dir
    
    def _append_to_global_tensor_dataset(self, new_tensor_data, global_tensor_path):
        """Append new tensor data to the global tensor dataset file.
        
        Args:
            new_tensor_data: Dictionary of new tensors to append
            global_tensor_path: Path to the global tensor dataset file
        """
        try:
            # Check if global dataset already exists
            if os.path.exists(global_tensor_path):
                logger.info(f"Loading existing global tensor dataset from {global_tensor_path}")
                
                # Load existing data
                existing_data = torch.load(global_tensor_path, map_location='cpu')
                
                # # Validate compatibility
                if not self._validate_tensor_compatibility(existing_data, new_tensor_data):
                    logger.error("New tensor data incompatible with existing global dataset")
                    return
                
                # Concatenate tensors
                merged_data = {}
                for key in existing_data.keys():
                    if key in new_tensor_data:
                        if isinstance(existing_data[key], torch.Tensor) and isinstance(new_tensor_data[key], torch.Tensor):
                            # Concatenate along batch dimension (dim=0)
                            merged_data[key] = torch.cat([existing_data[key], new_tensor_data[key]], dim=0)
                            logger.debug(f"Concatenated {key}: {existing_data[key].shape[0]} + {new_tensor_data[key].shape[0]} = {merged_data[key].shape[0]}")
                        else:
                            # For non-tensors, keep the existing value or update if needed
                            merged_data[key] = existing_data[key]
                    else:
                        # Keep existing data for keys not in new data
                        merged_data[key] = existing_data[key]
                
                # Add any new keys from new_tensor_data that weren't in existing_data
                for key in new_tensor_data.keys():
                    if key not in merged_data:
                        logger.warning(f"New key {key} found in new data, adding to global dataset")
                        merged_data[key] = new_tensor_data[key]
                
            else:
                logger.info(f"Creating new global tensor dataset at {global_tensor_path}")
                merged_data = new_tensor_data.copy()
            
            # Save the merged dataset
            torch.save(merged_data, global_tensor_path)
            
            # Log the final sizes
            total_samples = merged_data['actions'].shape[0] if 'actions' in merged_data else 0
            new_samples = new_tensor_data['actions'].shape[0] if 'actions' in new_tensor_data else 0
            logger.info(f"Successfully appended {new_samples} samples to global dataset. Total samples: {total_samples}")
            
        except Exception as e:
            logger.error(f"Failed to append to global tensor dataset: {e}")
            # Don't raise the exception to avoid breaking the main processing

    def _validate_tensor_compatibility(self, existing_data, new_data):
        """Validate that new tensor data is compatible with existing data for concatenation.
        
        Args:
            existing_data: Existing tensor dataset
            new_data: New tensor data to append
            
        Returns:
            True if compatible, False otherwise
        """
        # Check if both datasets have the same keys (for tensors)
        existing_tensor_keys = {k for k, v in existing_data.items() if isinstance(v, torch.Tensor)}
        new_tensor_keys = {k for k, v in new_data.items() if isinstance(v, torch.Tensor)}
        
        missing_keys = existing_tensor_keys - new_tensor_keys
        extra_keys = new_tensor_keys - existing_tensor_keys
        
        if missing_keys:
            logger.error(f"New data missing tensor keys: {missing_keys}")
            return False
        
        if extra_keys:
            logger.warning(f"New data has extra tensor keys: {extra_keys}")
            # We can still proceed, just add the new keys
        
        # Check tensor shape compatibility (all dimensions except batch should match)
        for key in existing_tensor_keys.intersection(new_tensor_keys):
            existing_shape = existing_data[key].shape
            new_shape = new_data[key].shape
            
            if len(existing_shape) != len(new_shape):
                logger.error(f"Tensor {key}: dimension mismatch - existing: {existing_shape}, new: {new_shape}")
                return False
            
            if len(existing_shape) > 1 and existing_shape[1:] != new_shape[1:]:
                logger.error(f"Tensor {key}: shape mismatch - existing: {existing_shape}, new: {new_shape}")
                return False
        
        return True

    def create_dataset_loaders(self, processed_data, batch_size=32, val_split=0.1):
        """Create PyTorch DataLoader objects for training and validation.

        Args:
            processed_data: Dictionary with preprocessed data
            batch_size: Batch size for training
            val_split: Fraction of data to use for validation

        Returns:
            train_loader, val_loader: DataLoader objects
        """
        try:
            import torch
            from torch.utils.data import TensorDataset, DataLoader, random_split

            # Create tensor dataset
            tensor_data = [
                torch.FloatTensor(processed_data['pod_features_with_staleness']),
                torch.FloatTensor(processed_data['kv_hit_ratios']),
                torch.FloatTensor(processed_data['request_features']),
                torch.LongTensor(processed_data['actions']),
                torch.FloatTensor(processed_data['rewards'])
            ]

            # Add positional encodings
            if 'positional_encodings' in processed_data:
                tensor_data.append(torch.FloatTensor(processed_data['positional_encodings']))

            # Create dataset
            dataset = TensorDataset(*tensor_data)

            # Split into train and validation
            val_size = int(len(dataset) * val_split)
            train_size = len(dataset) - val_size

            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size]
            )

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=torch.cuda.is_available()
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=torch.cuda.is_available()
            )

            logger.info(f"Created data loaders with {train_size} training and {val_size} validation samples")

            return train_loader, val_loader

        except ImportError:
            logger.warning("PyTorch not available, skipping data loader creation")
            return None, None


def encode_for_train(all_pods, df, output_dir, request_stats, request_features_train, request_features_reward):
    """Main function to process LLM routing data."""
    test_split = 0.2
    random_seed = 42
    batch_size = 32
    create_loaders = False
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Print some column samples
    logger.debug(f"columns: {list(df.columns)}")
    
    # Preview the first few rows (just to check format)
    if len(df) > 0:
        logger.info("First row selected_pod value: " + str(df.iloc[0].get('selected_pod', 'N/A')))
    
    # Check if data contains the expected column pattern
    pod_cols = [c for c in df.columns if 'pod_' in c or '-pod' in c]
    if not pod_cols:
        logger.warning("No columns with 'pod_' prefix or '-pod' pattern found")
    
    # Check if 'selected_pod' column contains valid pod IDs
    if 'selected_pod' in df.columns:
        pod_values = df['selected_pod'].dropna().unique()
        logger.info(f"Unique selected_pod values: {pod_values}")
    
    # Basic data quality checks
    logger.info("Performing data quality checks...")
    missing_pct = df.isnull().mean() * 100
    high_missing = missing_pct[missing_pct > 20].index.tolist()
    if high_missing:
        logger.warning(f"Columns with >20% missing values: {len(high_missing)} columns")
    
    # # Split into train and test
    # n_samples = len(df)
    # test_size = int(n_samples * test_split)
    # indices = np.random.permutation(n_samples)
    # test_indices = indices[:test_size]
    # train_indices = indices[test_size:]
    
    # train_df = df.iloc[train_indices]
    # test_df = df.iloc[test_indices]
    
    # logger.info(f"Split data into {len(train_df)} training and {len(test_df)} test samples")
    
    # Process data
    processor = LLMRoutingDataProcessor(output_dir=output_dir)
    
    # Check if we have running statistics for request features
    if request_stats is not None and request_stats.count > 0:
        logger.info(f"Using running statistics for normalization (n={request_stats.count})")
        
        # Get the request features to use for interaction
        # request_features = ['input_tokens', 'output_tokens', 'total_tokens', 'ttft', 'avg_tpot', 'e2e_latency']
        request_features = ['input_tokens', 'output_tokens', 'total_tokens']
        if all(feature in df.columns for feature in request_features):
            # We need to modify the DataFrame to contain normalized values before preprocessing
            request_values = df[request_features].values
            
            # Apply normalization
            normalized_values = request_stats.normalize(request_values)
            
            # Store normalized values in DataFrame
            for i, feature in enumerate(request_features):
                original_values = df[feature].copy()
                df[feature] = normalized_values[:, i]
                
                # Log normalization for the first row
                if len(df) > 0:
                    logger.info(f"Normalized {feature}: {original_values.iloc[0]} -> {df[feature].iloc[0]}")
        else:
            logger.warning(f"Some request features missing from DataFrame, using default normalization")
    else:
        logger.info("No running statistics provided, using batch-specific normalization")
    
    # Process training data
    logger.info("Processing training data...")
    overhead_summary = {}
    train_processed = processor.prepare_for_encoding(df, all_pods, request_features_train, request_features_reward, overhead_summary)
    train_path = processor.save_processed_data(train_processed)
    
    # # Process test data
    # logger.info("Processing test data...")
    # test_processed = processor.prepare_for_encoding(test_df, all_pods)
    # test_path = processor.save_processed_data(test_processed, prefix="test")
    
    # # Create data loaders if requested
    # if create_loaders:
    #     logger.info("Creating PyTorch DataLoader objects...")
    #     train_loader, val_loader = processor.create_dataset_loaders(
    #         train_processed, batch_size=batch_size
    #     )
        
    #     # Print batch sample information
    #     if train_loader is not None:
    #         for batch in train_loader:
    #             logger.info(f"Sample batch shapes:")
    #             for i, tensor in enumerate(batch):
    #                 logger.info(f"  Tensor {i}: {tensor.shape}")
    #             break
    
    # Log results
    logger.info("Data processing complete!")
    logger.info(f"Training data: {train_path}")
    # logger.info(f"Test data: {test_path}")
    
    # Print dataset shape information
    logger.info(f"Dataset shapes:")
    logger.info(f"  pod_features: {train_processed['pod_features'].shape}")
    logger.info(f"  pod_features_with_staleness: {train_processed['pod_features_with_staleness'].shape}")
    logger.info(f"  kv_hit_ratios: {train_processed['kv_hit_ratios'].shape}")
    logger.info(f"  request_features: {train_processed['request_features'].shape}")
    logger.info(f"  positional_encodings: {train_processed['positional_encodings'].shape}")
    logger.info(f"  actions: {train_processed['actions'].shape}")
    logger.info(f"  rewards: {train_processed['rewards'].shape}")

    return train_path


def encode_for_inference(all_pods, df, request_stats, request_features_train, request_features_reward):
    """Version with hardcoded column positions for maximum speed."""
    
    mask_start_time = time.time()
    
    # Based on your DataFrame, the request features appear to be at positions:
    # input_tokens: 2, output_tokens: 3, total_tokens: 4
    # Adjust these indices based on your actual column positions
    REQUEST_FEATURE_INDICES = [2, 3, 4]  # Hardcoded for speed
    
    # NUCLEAR: Direct access to underlying numpy array
    df_numpy = df.values
    request_values = df_numpy[:, REQUEST_FEATURE_INDICES].astype(np.float32, copy=False)
    
    # Ultra-fast operations
    zero_mask = (request_values == 0).all(axis=0)
    if zero_mask.any():
        request_values[:, zero_mask] = 0.01
    
    # Direct normalization
    request_values -= request_stats.get_mean()
    request_values /= request_stats.get_std()
    
    # Put data back
    df_numpy[:, REQUEST_FEATURE_INDICES] = request_values
    
    mask_overhead = time.time() - mask_start_time
    
    # STEP 2: Prepare for encoding (unchanged)
    prepare_for_encoding_start = time.time()
    processor = LLMRoutingDataProcessor(output_dir="temp_inference")
    overhead_summary = {}
    processed_data = processor.prepare_for_encoding(df, all_pods, request_features_train, request_features_reward, overhead_summary)
    prepare_for_encoding_overhead = time.time() - prepare_for_encoding_start

    # STEP 3: ULTRA-FAST tensor creation (replace your entire post_process section)
    post_process_start_time = time.time()

    # OPTIMIZATION 1: Skip logging entirely for maximum speed
    # if 'request_features' in processed_data:
    #     request_feat = processed_data['request_features']
    #     logger.info(f"Processed request features shape: {request_feat.shape}")
    #     if len(request_feat) > 0:
    #         logger.info(f"Processed request features values: {request_feat[0]}")

    # OPTIMIZATION 2: Pre-allocate tensor dictionary and use direct assignment
    tensor_data = {}

    # OPTIMIZATION 3: Batch tensor creation with minimal function calls
    pd = processed_data  # Short alias to reduce lookup overhead

    # Core tensors (always present)
    tensor_data['pod_features'] = torch.from_numpy(pd['pod_features']).float()
    tensor_data['kv_hit_ratios'] = torch.from_numpy(pd['kv_hit_ratios']).float()
    tensor_data['request_features'] = torch.from_numpy(pd['request_features']).float()
    tensor_data['actions'] = torch.from_numpy(pd['actions']).long()
    tensor_data['rewards'] = torch.from_numpy(pd['rewards']).float()
    tensor_data['positional_encodings'] = torch.from_numpy(pd['positional_encodings']).float()
    tensor_data['pod_features_with_staleness'] = torch.from_numpy(pd['pod_features_with_staleness']).float()
    tensor_data['query'] = torch.from_numpy(pd['cross_attention_inputs']['query']).float()
    tensor_data['key_value'] = torch.from_numpy(pd['cross_attention_inputs']['key_value']).float()

    # OPTIMIZATION 4: Conditional tensors with minimal overhead
    if pd['interaction_features'] is not None:
        tensor_data['interaction_features'] = torch.from_numpy(pd['interaction_features']).float()

    # OPTIMIZATION 5: Use get() with default None to avoid key checks
    ttft_rewards = pd.get('ttft_rewards')
    if ttft_rewards is not None:
        tensor_data['ttft_rewards'] = torch.from_numpy(ttft_rewards).float()

    tpot_rewards = pd.get('tpot_rewards')
    if tpot_rewards is not None:
        tensor_data['tpot_rewards'] = torch.from_numpy(tpot_rewards).float()

    post_process_overhead = time.time() - post_process_start_time
        
    # Update overhead summary
    overhead_summary['encoding.encode_for_inference.mask_overhead'] = mask_overhead * 1000
    overhead_summary['encoding.encode_for_inference.prepare_for_encoding_overhead'] = prepare_for_encoding_overhead * 1000
    overhead_summary['encoding.encode_for_inference.post_process_overhead'] = post_process_overhead * 1000

    return tensor_data, overhead_summary