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

    # ## old
    # def extract_pod_columns(self, df, all_pods):
    #     """Extract pod-related columns and organize by pod ID and feature type."""
    #     pod_data = defaultdict(dict)
        
    #     logger.info(f"df.columns: {df.columns}")
    #     self.pod_ids = all_pods
    #     logger.info(f"Found pod IDs from selected_pod column: {self.pod_ids}")

    #     # Direct check for column patterns
    #     pod_feature_columns = [col for col in df.columns if col.startswith('pod_')]
    #     for col in pod_feature_columns:
    #         # Look for columns like 'pod_10.0.1.32-feature_name'
    #         parts = col.split('-')
    #         assert len(parts) == 2, f"Unexpected column format: {col}"
    #         pod_id = parts[0].replace('pod_', '')
    #         feature = parts[1]
    #         if feature not in self.pod_features:
    #             self.pod_features.append(feature)
    #         logger.debug(f"pod_id: {pod_id}, self.pod_ids: {self.pod_ids}")
    #         if pod_id in self.pod_ids:
    #             pod_data[pod_id][feature] = df[col]
    #         else:
    #             logger.error(f"Pod ID {pod_id} not found in self.pod_ids {self.pod_ids}, col: {col}")
    #             assert False
    #     # all pods must have the same number of columns for pod features
    #     for pod_id in self.pod_ids:
    #         if pod_id not in pod_data:
    #             logger.error(f"Pod ID {pod_id} not found in pod_data")
    #             assert False
    #         # Check if all pods have the same features
    #         if len(pod_data[pod_id]) != len(self.pod_features):
    #             logger.error(f"Pod ID {pod_id} has {len(pod_data[pod_id])} features, expected {len(self.pod_features)}")
    #             assert False

    #     logger.info(f"pod_data contains {len(pod_data)} pods and total of {sum(len(features) for features in pod_data.values())} features")
    #     self.pod_features = sorted(self.pod_features)
    #     return pod_data
        
    #     # # Log features
    #     # self.pod_features = sorted(self.pod_features)
    #     # logger.info(f"Extracted {len(self.pod_ids)} pod IDs and {len(self.pod_features)} pod features")
    #     # logger.info(f"Pod IDs: {self.pod_ids[:5]}{'...' if len(self.pod_ids) > 5 else ''}")
    #     # logger.info(f"Pod features: {self.pod_features[:5]}{'...' if len(self.pod_features) > 5 else ''}")
        
    #     # # Log last_second features specifically
    #     # last_second_features = [f for f in self.pod_features if 'last_second' in f]
    #     # logger.info(f"Found {len(last_second_features)} last_second features: {last_second_features}")
        
    #     # return pod_data

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

    # ## old
    # def analyze_request_features(self, df, request_features_train, request_features_reward):
    #     # Columns to exclude from features
    #     exclude_cols = [
    #         'request_id',           # Identifier, not a feature
    #         'selected_pod',         # Target, not a feature
    #         'action',               # Target, not a feature
    #         'reward',               # Target, not a feature
    #         'ttft_reward',          # Component of reward, not a feature
    #         'tpot_reward',          # Component of reward, not a feature
    #         'ttft_normalized',      # Derived from reward
    #         'tpot_normalized',      # Derived from reward
    #     ]
    #     exclude_cols.extend(request_features_reward)
    #     exclude_patterns = ['reward', 'action', 'slo_satisfied', 'normalized']
        
    #     # Any column not starting with "pod_" and not in exclude list
    #     candidate_request_features = [
    #         col for col in df.columns 
    #         if not any(col.startswith(f"pod_{pod_id}") for pod_id in self.pod_ids) 
    #         and not any(pat in col for pat in exclude_patterns)
    #         and col not in exclude_cols
    #     ]
        
    #     # Log what we're doing
    #     logger.info(f"Request features - Training features: {request_features_train}")
    #     logger.info(f"Request features - Reward features (excluded from training): {request_features_reward}")
    #     logger.info(f"Request features - Found {len(candidate_request_features)} candidate columns: {candidate_request_features}")

    #     # Test each column to see if it's numeric or categorical
    #     numeric_cols = []
    #     categorical_cols = []
        
    #     for col in candidate_request_features:
    #         # Skip columns with too many NaN values
    #         if df[col].isna().mean() > 0:
    #             logger.error(f"Request features - {col} has NaN values.")
    #             assert False
                
    #         # Try to convert to numeric
    #         try:
    #             # Check if already numeric
    #             if pd.api.types.is_numeric_dtype(df[col]):
    #                 numeric_cols.append(col)
    #                 continue
                
    #             # Try to convert
    #             pd.to_numeric(df[col])
    #             numeric_cols.append(col)
    #         except:
    #             # If conversion fails, it's categorical
    #             categorical_cols.append(col)
        
    #     self.numeric_request_features = numeric_cols
    #     self.categorical_request_features = categorical_cols
        
    #     logger.info(f"Request features - number of numeric columns: {len(numeric_cols)}")
    #     logger.info(f"Request features - number of categorical columns {len(categorical_cols)}")
    #     if len(numeric_cols) > 0:
    #         logger.info(f"Request features - numeric features: {numeric_cols}")
    #     if len(categorical_cols) > 0:
    #         logger.info(f"Request features - categorical features: {categorical_cols}")

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

    # ## old
    # def encode_pod_ids(self, df):
    #     """Create encoders for pod IDs.
        
    #     Args:
    #         df: Pandas DataFrame with routing data
    #     """
    #     # If we have pod IDs, create an encoder
    #     if self.pod_ids:
    #         # Fit an encoder for pod IDs
    #         self.pod_encoder = OneHotEncoder(sparse_output=False)
    #         self.pod_encoder.fit(np.array(self.pod_ids).reshape(-1, 1))
            
    #         # Also fit an encoder for selected_pod column
    #         if 'selected_pod' in df.columns:
    #             selected_pods = df['selected_pod'].dropna().unique()
    #             self.selected_pod_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    #             self.selected_pod_encoder.fit(np.array(selected_pods).reshape(-1, 1))
                
    #             logger.info(f"Encoded {len(selected_pods)} unique selected pods")
    #     else:
    #         logger.warning("No pod IDs found, skipping pod encoding")

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


    # ## old
    # def classify_feature_timing(self):
    #     feature_timing = {}
    #     logger.info(f"Classifying timing for pod features: {self.pod_features}")
    #     for feature in self.pod_features:
    #         if 'last_second' in feature:
    #             feature_timing[feature] = 'historical'
    #         else:
    #             feature_timing[feature] = 'current'
    #     current_features = [f for f, timing in feature_timing.items() if timing == 'current']
    #     historical_features = [f for f, timing in feature_timing.items() if timing == 'historical']
    #     logger.info(f"Current-time features: {current_features}")
    #     logger.info(f"historical features: {historical_features}")
    #     for historical_feat in historical_features:
    #         if 'last_second' not in historical_feat:
    #             logger.error(f"Feature {historical_feat} is classified as historical but does not contain 'last_second'")
    #             assert False
    #     for current_feat in current_features:
    #         if 'last_second' in current_feat:
    #             logger.error(f"Feature {current_feat} is classified as current but contains 'last_second'")
    #             assert False
    #     return feature_timing

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

    # ## old
    # def add_staleness_features(self, pod_features, timestamps, feature_timing, feature_indices_map):
    #     """Add staleness indicators for historical features.
        
    #     Args:
    #         pod_features: Pod feature tensor [batch, n_pods, feature_dim]
    #         timestamps: Request timestamps
    #         feature_timing: Dictionary mapping features to timing category
    #         feature_indices_map: Dictionary mapping feature names to indices
            
    #     Returns:
    #         Pod features with staleness indicators added
    #     """
    #     # Get indices of historical features
    #     historical_features = [f for f, timing in feature_timing.items() if timing == 'historical']
    #     historical_indices = [
    #         idx for feature, idx in feature_indices_map.items() 
    #         if feature in historical_features
    #     ]
        
    #     if not historical_indices or len(timestamps) == 0 or np.all(timestamps == 0):
    #         logger.info("No historical features or valid timestamps, skipping staleness")
    #         # Add dummy staleness feature (all zeros)
    #         staleness_features = np.zeros((pod_features.shape[0], pod_features.shape[1], 1))
    #         return np.concatenate([pod_features, staleness_features], axis=2)
        
    #     # Calculate staleness based on timestamps (normalize to [0,1] range)
    #     # Assuming timestamps are in seconds from reference point
    #     max_staleness = 60.0  # 60 seconds max staleness
        
    #     # Calculate time differences between consecutive requests
    #     # Sort timestamps and get differences
    #     sorted_indices = np.argsort(timestamps)
    #     sorted_timestamps = timestamps[sorted_indices]
    #     time_diffs = np.diff(sorted_timestamps, prepend=sorted_timestamps[0])
    #     time_diffs = np.maximum(time_diffs, 0)  # Ensure positive
        
    #     # Reorder to original sequence and normalize
    #     staleness = np.zeros_like(timestamps)
    #     staleness[sorted_indices] = time_diffs
    #     staleness = np.clip(staleness / max_staleness, 0, 1)
        
    #     # Create staleness feature for each pod
    #     staleness_features = np.zeros((pod_features.shape[0], pod_features.shape[1], 1))
    #     for i in range(pod_features.shape[1]):  # For each pod
    #         staleness_features[:, i, 0] = staleness
        
    #     logger.info(f"Added staleness indicator for {len(historical_indices)} historical features")
        
    #     # Concatenate with original features
    #     return np.concatenate([pod_features, staleness_features], axis=2)

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

    # ## old
    # def create_request_pod_interaction_features(self, request_features, pod_features):
    #     """Create features capturing interactions between request and pod characteristics.
        
    #     Args:
    #         request_features: Request feature tensor [batch, request_dim]
    #         pod_features: Pod feature tensor [batch, n_pods, pod_dim]
            
    #     Returns:
    #         Interaction features for each pod
    #     """
    #     if request_features.shape[1] == 0:
    #         logger.warning("No request features available for interaction")
    #         return None
            
    #     batch_size, n_pods, _ = pod_features.shape
        
    #     # Expand request features to match pod dimensions
    #     expanded_request = np.repeat(
    #         request_features[:, np.newaxis, :], n_pods, axis=1
    #     )
        
    #     logger.info(f"Created request-pod interaction features with shape {expanded_request.shape}")
        
    #     return expanded_request

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

    ## old
    # def _optimized_process_pod_features(self, pod_data, n_samples):
    #     """Process pod features - FURTHER OPTIMIZED section for Step 6."""
    #     pod_features_list = []
    #     pod_kv_hit_ratios = []
    #     per_pod_feature_indices = {}
        
    #     if pod_data:
    #         # OPTIMIZATION 1: Pre-create shared GPU encoder if gpu_model exists
    #         one_hot_encoder_start_time = time.time()
    #         shared_gpu_encoder = None
    #         if 'gpu_model' in self.pod_features:
    #             # Collect all GPU values from all pods at once
    #             all_gpu_values = []
    #             for pod_id in self.pod_ids:
    #                 if 'gpu_model' in pod_data[pod_id]:
    #                     gpu_vals = pod_data[pod_id]['gpu_model'].fillna('unknown').values
    #                     all_gpu_values.extend(gpu_vals)
                
    #             if all_gpu_values:
    #                 # Create one encoder for all pods
    #                 shared_gpu_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    #                 shared_gpu_encoder.fit(np.array(all_gpu_values).reshape(-1, 1))
    #         one_hot_encoder_overhead = time.time() - one_hot_encoder_start_time

    #         # OPTIMIZATION 2: Pre-process all data by feature type
    #         kv_hit_data = {}
    #         numeric_data = {}
    #         gpu_model_data = {}
            
    #         nested_loop_start_time = time.time()
    #         for pod_id in self.pod_ids:
    #             kv_hit_data[pod_id] = None
    #             numeric_data[pod_id] = []
    #             gpu_model_data[pod_id] = None
                
    #             for feature in self.pod_features:
    #                 if feature in pod_data[pod_id]:
    #                     if feature == 'kv_hit_ratio':
    #                         kv_hit_data[pod_id] = pod_data[pod_id][feature].fillna(0).values.reshape(-1, 1)
    #                     elif feature == 'gpu_model':
    #                         gpu_values = pod_data[pod_id][feature].fillna('unknown')
    #                         if shared_gpu_encoder:
    #                             gpu_model_data[pod_id] = shared_gpu_encoder.transform(gpu_values.values.reshape(-1, 1))
    #                     else:
    #                         # Batch process numeric features
    #                         values = pod_data[pod_id][feature].fillna(0).values.reshape(-1, 1)
    #                         numeric_data[pod_id].append(values)
    #         nested_loop_overhead = time.time() - nested_loop_start_time

    #         build_feature_start_time = time.time()
    #         # OPTIMIZATION 3: Build features for each pod using pre-processed data
    #         for pod_id in self.pod_ids:
    #             pod_features = []
    #             feature_indices_map = {}
    #             feature_idx = 0
                
    #             # Add numeric features first (batch processed)
    #             numeric_features = [f for f in self.pod_features if f not in ['kv_hit_ratio', 'gpu_model']]
    #             if numeric_data[pod_id]:
    #                 # Single hstack for all numeric features per pod
    #                 numeric_stack = np.hstack(numeric_data[pod_id])
    #                 pod_features.append(numeric_stack)
                    
    #                 # Update feature indices
    #                 for i, feature in enumerate(numeric_features):
    #                     feature_indices_map[feature] = feature_idx + i
    #                 feature_idx += len(numeric_features)
                
    #             # Add GPU model if exists
    #             if gpu_model_data[pod_id] is not None:
    #                 pod_features.append(gpu_model_data[pod_id])
    #                 feature_indices_map['gpu_model'] = feature_idx
    #                 feature_idx += gpu_model_data[pod_id].shape[1]
                
    #             # Store KV hit ratio separately
    #             if kv_hit_data[pod_id] is not None:
    #                 pod_kv_hit_ratios.append(kv_hit_data[pod_id])
    #             else:
    #                 # Default zero values if missing
    #                 pod_kv_hit_ratios.append(np.zeros((n_samples, 1)))
                
    #             per_pod_feature_indices[pod_id] = feature_indices_map
                
    #             if pod_features:
    #                 # Single hstack per pod
    #                 pod_features_combined = np.hstack(pod_features)
    #                 pod_features_list.append(pod_features_combined)
    #             else:
    #                 logger.error(f"No features found for pod {pod_id}")
    #                 assert False
            
    #         if pod_features_list:
    #             # OPTIMIZATION 4: Batch operations (unchanged as already optimized)
    #             pod_features_array = np.stack(pod_features_list, axis=1)
    #             pod_kv_hit_array = np.stack(pod_kv_hit_ratios, axis=1)
                
    #             pod_shape = pod_features_array.shape
    #             kv_shape = pod_kv_hit_array.shape
                
    #             pod_features_flat = pod_features_array.reshape(-1, pod_features_array.shape[2])
    #             kv_flat = pod_kv_hit_array.reshape(-1, 1)

    #             self.pod_feature_scaler.fit(pod_features_flat)
    #             self.kv_hit_scaler.fit(kv_flat)
                
    #             pod_features_norm = self.pod_feature_scaler.transform(pod_features_flat).reshape(pod_shape)
    #             kv_hit_norm = self.kv_hit_scaler.transform(kv_flat).reshape(kv_shape)

    #             self.feature_stats.update({
    #                 'pod_feature_means': self.pod_feature_scaler.mean_,
    #                 'pod_feature_stds': self.pod_feature_scaler.scale_,
    #                 'kv_hit_means': self.kv_hit_scaler.mean_,
    #                 'kv_hit_stds': self.kv_hit_scaler.scale_
    #             })

    #             logger.info(f"Pod features stats after processing: min={pod_features_norm.min()}, max={pod_features_norm.max()}, non-zero={np.count_nonzero(pod_features_norm)}/{pod_features_norm.size}")
    #             logger.info(f"KV hit ratio stats after processing: min={kv_hit_norm.min()}, max={kv_hit_norm.max()}, non-zero={np.count_nonzero(kv_hit_norm)}/{kv_hit_norm.size}")
    #             build_feature_overhead = time.time() - build_feature_start_time

    #             process_pod_features_overhead_summary = {
    #                 'encoding_process_pod_features_overhead_one_hot_encoder_overhead': int(one_hot_encoder_overhead*1000),
    #                 'encoding_process_pod_features_overhead_nested_loop_overhead': int(nested_loop_overhead*1000),
    #                 'encoding_process_pod_features_overhead_build_feature_overhead': int(build_feature_overhead*1000)
    #             }
    #             return pod_features_array, pod_kv_hit_array, pod_features_norm, kv_hit_norm, per_pod_feature_indices, process_pod_features_overhead_summary
    #         else:
    #             logger.error("No pod features found in expected format")
    #             assert False
    #     else:
    #         logger.error("No pod data in expected format, creating default pod features")
    #         assert False

    def _optimized_process_pod_features(self, pod_data, n_samples):
        """Process pod features - ZERO OVERHEAD OPTIMIZATION."""
        
        if not pod_data:
            logger.error("No pod data in expected format")
            assert False
        
        # STEP 1: Pre-create shared GPU encoder (if needed)
        one_hot_encoder_start_time = time.time()
        shared_gpu_encoder = None
        if 'gpu_model' in self.pod_features:
            all_gpu_values = []
            for pod_id in self.pod_ids:
                if 'gpu_model' in pod_data[pod_id]:
                    gpu_vals = pod_data[pod_id]['gpu_model'].fillna('unknown').values
                    all_gpu_values.extend(gpu_vals)
            
            if all_gpu_values:
                shared_gpu_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                shared_gpu_encoder.fit(np.array(all_gpu_values).reshape(-1, 1))
        one_hot_encoder_overhead = time.time() - one_hot_encoder_start_time
        
        # STEP 2: VECTORIZED DATA EXTRACTION
        vectorized_extraction_start_time = time.time()
        
        # Separate features by type
        numeric_features = [f for f in self.pod_features if f not in ['kv_hit_ratio', 'gpu_model']]
        
        # Pre-allocate arrays for ALL pods at once
        n_pods = len(self.pod_ids)
        n_numeric = len(numeric_features)
        
        # Extract all numeric data in one go - shape: (n_samples, n_pods, n_numeric_features)
        if numeric_features:
            numeric_arrays = np.zeros((n_samples, n_pods, n_numeric))
            for pod_idx, pod_id in enumerate(self.pod_ids):
                for feat_idx, feature in enumerate(numeric_features):
                    if feature in pod_data[pod_id]:
                        numeric_arrays[:, pod_idx, feat_idx] = pod_data[pod_id][feature].fillna(0).values
        else:
            numeric_arrays = np.zeros((n_samples, n_pods, 0))
        
        # Extract KV hit ratio data - shape: (n_samples, n_pods, 1)
        kv_arrays = np.zeros((n_samples, n_pods, 1))
        if 'kv_hit_ratio' in self.pod_features:
            for pod_idx, pod_id in enumerate(self.pod_ids):
                if 'kv_hit_ratio' in pod_data[pod_id]:
                    kv_arrays[:, pod_idx, 0] = pod_data[pod_id]['kv_hit_ratio'].fillna(0).values
        
        # Extract GPU model data (if exists) - shape: (n_samples, n_pods, n_gpu_features)
        gpu_arrays = None
        gpu_feature_count = 0
        if 'gpu_model' in self.pod_features and shared_gpu_encoder:
            # Get the number of GPU features from encoder
            sample_transform = shared_gpu_encoder.transform([['unknown']])
            gpu_feature_count = sample_transform.shape[1]
            gpu_arrays = np.zeros((n_samples, n_pods, gpu_feature_count))
            
            for pod_idx, pod_id in enumerate(self.pod_ids):
                if 'gpu_model' in pod_data[pod_id]:
                    gpu_values = pod_data[pod_id]['gpu_model'].fillna('unknown')
                    transformed = shared_gpu_encoder.transform(gpu_values.values.reshape(-1, 1))
                    gpu_arrays[:, pod_idx, :] = transformed
        vectorized_extraction_overhead = time.time() - vectorized_extraction_start_time
        
        # STEP 3: VECTORIZED CONCATENATION - Single operation for all pods
        build_feature_start_time = time.time()
        
        # Build the feature arrays list
        feature_arrays_to_concat = [numeric_arrays]
        
        if gpu_arrays is not None:
            feature_arrays_to_concat.append(gpu_arrays)
        
        # Single concatenation operation for ALL pods at once
        if len(feature_arrays_to_concat) == 1:
            pod_features_array = feature_arrays_to_concat[0]
        else:
            pod_features_array = np.concatenate(feature_arrays_to_concat, axis=2)
        
        pod_kv_hit_array = kv_arrays
        
        # STEP 4: Create feature indices map (only for first pod, since all pods have same structure)
        reference_feature_indices = {}
        feature_idx = 0
        
        # Add numeric feature indices
        for i, feature in enumerate(numeric_features):
            reference_feature_indices[feature] = feature_idx + i
        feature_idx += len(numeric_features)
        
        # Add GPU model indices
        if gpu_arrays is not None:
            reference_feature_indices['gpu_model'] = feature_idx
            feature_idx += gpu_feature_count
        
        # Create per_pod_feature_indices (all pods have same structure)
        per_pod_feature_indices = {pod_id: reference_feature_indices.copy() for pod_id in self.pod_ids}
        build_feature_overhead = time.time() - build_feature_start_time
        
        # STEP 5: Batch normalization
        normalization_start_time = time.time()
        pod_shape = pod_features_array.shape
        kv_shape = pod_kv_hit_array.shape
        
        pod_features_flat = pod_features_array.reshape(-1, pod_features_array.shape[2])
        kv_flat = pod_kv_hit_array.reshape(-1, 1)

        self.pod_feature_scaler.fit(pod_features_flat)
        self.kv_hit_scaler.fit(kv_flat)
        
        pod_features_norm = self.pod_feature_scaler.transform(pod_features_flat).reshape(pod_shape)
        kv_hit_norm = self.kv_hit_scaler.transform(kv_flat).reshape(kv_shape)

        self.feature_stats.update({
            'pod_feature_means': self.pod_feature_scaler.mean_,
            'pod_feature_stds': self.pod_feature_scaler.scale_,
            'kv_hit_means': self.kv_hit_scaler.mean_,
            'kv_hit_stds': self.kv_hit_scaler.scale_
        })

        logger.info(f"Pod features stats after processing: min={pod_features_norm.min()}, max={pod_features_norm.max()}, non-zero={np.count_nonzero(pod_features_norm)}/{pod_features_norm.size}")
        logger.info(f"KV hit ratio stats after processing: min={kv_hit_norm.min()}, max={kv_hit_norm.max()}, non-zero={np.count_nonzero(kv_hit_norm)}/{kv_hit_norm.size}")
        normalization_overhead = time.time() - normalization_start_time
        
        # Return proper overhead summary like the original
        process_pod_features_overhead_summary = {
            'prepare_for_encoding._optimized_process_pod_features.one_hot_encoder_overhead': int(one_hot_encoder_overhead * 1000),
            'prepare_for_encoding._optimized_process_pod_features.vectorized_extraction_overhead': int(vectorized_extraction_overhead * 1000),
            'prepare_for_encoding._optimized_process_pod_features.build_feature_overhead': int(build_feature_overhead * 1000),
            'prepare_for_encoding._optimized_process_pod_features.normalization_overhead': int(normalization_overhead * 1000)
        }
        
        return pod_features_array, pod_kv_hit_array, pod_features_norm, kv_hit_norm, per_pod_feature_indices, process_pod_features_overhead_summary

    def prepare_for_encoding(self, df, all_pods, request_features_train, request_features_reward):
        extract_pod_columns_start = time.time()
        # Step 1: Extract pod-related columns
        pod_data = self.extract_pod_columns(df, all_pods)
        extract_pod_columns_overhead = time.time() - extract_pod_columns_start

        # Step 2 Analyze request features
        analyze_request_features_start = time.time()
        self.analyze_request_features(df, request_features_train, request_features_reward)
        analyze_request_features_overhead = time.time() - analyze_request_features_start
        
        # Step 3: Encode pod IDs
        encode_pod_ids_start = time.time()
        self.encode_pod_ids(df)
        encode_pod_ids_overhead = time.time() - encode_pod_ids_start
        
        # Step 4: Classify feature timing (historical vs current)
        classify_feature_timing_start = time.time()
        feature_timing = self.classify_feature_timing()
        classify_feature_timing_overhead = time.time() - classify_feature_timing_start
        
        # Step 5: Process request features
        n_samples = len(df)
        
        # Process numeric request features
        request_numeric_features_start_time = time.time()
        request_numeric_features = None
        if self.numeric_request_features:
            request_numeric_features = df[self.numeric_request_features].fillna(0).values
            # Still store the statistics even if not normalizing
            self.feature_stats['request_feature_means'] = np.mean(request_numeric_features, axis=0)
            self.feature_stats['request_feature_stds'] = np.std(request_numeric_features, axis=0)
        else:
            request_numeric_features = np.zeros((n_samples, 0))
        request_numeric_features_overhead = time.time() - request_numeric_features_start_time

        # Process categorical request features
        request_categorical_features_start_time = time.time()
        request_categorical_features = []
        categorical_encoders = {}
        
        for col in self.categorical_request_features:
            # Skip columns with all NaN
            if df[col].isna().all():
                continue
                
            # Fill NaN values
            filled_col = df[col].fillna('unknown')
            
            # Create encoder
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoder.fit(filled_col.values.reshape(-1, 1))
            categorical_encoders[col] = encoder
            
            # Transform
            encoded = encoder.transform(filled_col.values.reshape(-1, 1))
            request_categorical_features.append(encoded)
        
        # Combine categorical features
        if request_categorical_features:
            request_categorical_features = np.hstack(request_categorical_features)
        else:
            request_categorical_features = np.zeros((n_samples, 0))
        request_categorical_features_overhead = time.time() - request_categorical_features_start_time

        ## new
        process_pod_features_start = time.time()
        pod_features_array, pod_kv_hit_array, pod_features_norm, kv_hit_norm, per_pod_feature_indices, process_pod_features_overhead_summary = self._optimized_process_pod_features(pod_data, n_samples)
        process_pod_features_overhead = time.time() - process_pod_features_start

        ## old
        # # Step 6: Process pod features
        # pod_features_list = []
        # pod_kv_hit_ratios = []
        
        # # Create a per-pod feature indices map instead of a global one
        # per_pod_feature_indices = {}
        
        # # If we have pod data in the expected format
        # if pod_data:
        #     for pod_id in self.pod_ids:
        #         pod_features = []
        #         # Start fresh for each pod
        #         feature_indices_map = {}
        #         feature_idx = 0
                
        #         # Process each feature for this pod
        #         for feature in self.pod_features:
        #             if feature in pod_data[pod_id]:
        #                 # Special handling for kv_hit_ratio
        #                 if feature == 'kv_hit_ratio':
        #                     kv_values = pod_data[pod_id][feature].fillna(0).values.reshape(-1, 1)
        #                     pod_kv_hit_ratios.append(kv_values)
        #                 elif feature == 'gpu_model':
        #                     # One-hot encode GPU model
        #                     gpu_values = pod_data[pod_id][feature].fillna('unknown')
        #                     encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        #                     encoder.fit(gpu_values.values.reshape(-1, 1))
        #                     encoded = encoder.transform(gpu_values.values.reshape(-1, 1))
        #                     pod_features.append(encoded)
        #                     # Add to feature indices map for this pod
        #                     feature_indices_map[feature] = feature_idx
        #                     feature_idx += encoded.shape[1]
        #                 else:
        #                     # Regular numeric feature
        #                     values = pod_data[pod_id][feature].fillna(0).values.reshape(-1, 1)
        #                     pod_features.append(values)
        #                     # Add to feature indices map for this pod
        #                     feature_indices_map[feature] = feature_idx
        #                     feature_idx += 1
        #             else:
        #                 logger.error(f"Feature {feature} not found for pod {pod_id}")
        #                 assert False
                
        #         # Store feature indices map for this pod
        #         per_pod_feature_indices[pod_id] = feature_indices_map
                
        #         # Combine all features for this pod
        #         if pod_features:
        #             pod_features = np.hstack(pod_features)
        #             pod_features_list.append(pod_features)
        #         else:
        #             logger.error(f"No features found for pod {pod_id}")
        #             assert False
            
        #     # Stack features for all pods
        #     if pod_features_list:
        #         pod_features_array = np.stack(pod_features_list, axis=1)
        #         pod_kv_hit_array = np.stack(pod_kv_hit_ratios, axis=1)
                
        #         ## NOTE: normalization can cause issue, making all values zero if raw values in different pods are similar
        #         pod_shape = pod_features_array.shape
        #         kv_shape = pod_kv_hit_array.shape
        #         pod_features_flat = pod_features_array.reshape(-1, pod_features_array.shape[2])
        #         kv_flat = pod_kv_hit_array.reshape(-1, 1)

        #         self.pod_feature_scaler.fit(pod_features_flat)
        #         self.kv_hit_scaler.fit(kv_flat)
        #         pod_features_norm = self.pod_feature_scaler.transform(pod_features_flat).reshape(pod_shape)
        #         kv_hit_norm = self.kv_hit_scaler.transform(kv_flat).reshape(kv_shape)

        #         self.feature_stats['pod_feature_means'] = self.pod_feature_scaler.mean_
        #         self.feature_stats['pod_feature_stds'] = self.pod_feature_scaler.scale_
        #         self.feature_stats['kv_hit_means'] = self.kv_hit_scaler.mean_
        #         self.feature_stats['kv_hit_stds'] = self.kv_hit_scaler.scale_

        #         logger.info(f"Pod features stats after processing: min={pod_features_norm.min()}, max={pod_features_norm.max()}, non-zero={np.count_nonzero(pod_features_norm)}/{pod_features_norm.size}")
        #         logger.info(f"KV hit ratio stats after processing: min={kv_hit_norm.min()}, max={kv_hit_norm.max()}, non-zero={np.count_nonzero(kv_hit_norm)}/{kv_hit_norm.size}")
        #     else:
        #         logger.error("No pod features found in expected format")
        #         assert False
        # else:
        #     logger.error("No pod data in expected format, creating default pod features")
        #     assert False
        
        ## new
        extract_actions_rewards_start = time.time()
        actions, rewards, ttft_rewards, tpot_rewards = self._optimized_extract_actions_rewards(df, n_samples)
        extract_actions_rewards_overhead = time.time() - extract_actions_rewards_start

        ## old
        # # Step 7: Extract actions and rewards
        # actions = np.zeros(n_samples, dtype=np.int64)
        # rewards = np.zeros(n_samples)
        # ttft_rewards = np.zeros(n_samples)
        # tpot_rewards = np.zeros(n_samples)
        
        # # Extract selected pod and convert to action index
        # if 'selected_pod' in df.columns:
        #     pod_to_idx = {pod_id: i for i, pod_id in enumerate(self.pod_ids)}
        #     for i, selected_pod in enumerate(df['selected_pod'].values):
        #         if pd.notna(selected_pod) and str(selected_pod) in pod_to_idx:
        #             actions[i] = pod_to_idx[str(selected_pod)]
        # elif 'action' in df.columns:
        #     actions = df['action'].fillna(0).astype(np.int64).values
        
        # # Extract rewards
        # if 'reward' in df.columns:
        #     rewards = df['reward'].fillna(0).values
        # if 'ttft_reward' in df.columns:
        #     ttft_rewards = df['ttft_reward'].fillna(0).values
        # if 'tpot_reward' in df.columns:
        #     tpot_rewards = df['tpot_reward'].fillna(0).values
        
        # Step 8: Combine request features
        combine_request_features_start = time.time()
        request_features = np.hstack([
            request_numeric_features, 
            request_categorical_features
        ]) if request_categorical_features.size > 0 else request_numeric_features
        combine_request_features_overhead = time.time() - combine_request_features_start
        
        # # Step 9: Create timestamps
        timestamps = np.zeros(n_samples)
        # if 'request_start_time' in df.columns:
        #     try:
        #         timestamps = pd.to_numeric(df['request_start_time']).values
        #         min_timestamp = timestamps.min() if len(timestamps) > 0 else 0
        #         timestamps = (timestamps - min_timestamp) / 1000.0  # Convert to seconds from reference
        #     except:
        #         logger.warning("Could not convert request_start_time to numeric values")
        
        # # Step 10: Generate metrics-based positional encoding
        # positional_encodings = self.prepare_metrics_based_positional_encoding(
        #     pod_features_norm, feature_indices_map
        # )
        # Use the first pod's feature indices map as reference for positional encoding
        if per_pod_feature_indices and self.pod_ids:
            reference_feature_indices = per_pod_feature_indices[self.pod_ids[0]]
        else:
            reference_feature_indices = {}

        # Step 10: Generate metrics-based positional encoding
        positional_encoding_start_time = time.time()
        positional_encodings = self.prepare_metrics_based_positional_encoding(pod_features_norm, reference_feature_indices)
        positional_encoding_overhead = time.time() - positional_encoding_start_time
        
        # # Step 11: Add staleness features for historical metrics
        # pod_features_with_staleness = self.add_staleness_features(
        #     pod_features_norm, timestamps, feature_timing, feature_indices_map
        # )
        add_staleness_start_time = time.time()
        pod_features_with_staleness = self.add_staleness_features(
            pod_features_norm, timestamps, feature_timing, reference_feature_indices
        )
        add_staleness_overhead = time.time() - add_staleness_start_time
        
        # Step 12: Prepare cross-attention inputs
        cross_attention_start_time = time.time()
        cross_attention_inputs = self.prepare_cross_attention_inputs(
            pod_features_with_staleness, kv_hit_norm
        )
        cross_attention_overhead = time.time() - cross_attention_start_time
        
        # Step 13: Create request-pod interaction features
        create_request_pod_interaction_start_time = time.time()
        interaction_features = self.create_request_pod_interaction_features(
            request_features, pod_features_norm
        )
        create_request_pod_interaction_overhead = time.time() - create_request_pod_interaction_start_time
        
        if interaction_features is not None:
            logger.info(f"Created interaction features with shape {interaction_features.shape}")
            logger.info(f"Interaction features min={interaction_features.min()}, max={interaction_features.max()}")
    
        
        # Return processed data
        processed_data = {
            # Original pod features
            'pod_features': pod_features_norm,
            'pod_raw_features': pod_features_array if 'pod_features_array' in locals() else np.zeros((n_samples, len(self.pod_ids), 1)),
            
            # KV hit ratio (for cross-attention)
            'kv_hit_ratios': kv_hit_norm,
            'kv_hit_raw': pod_kv_hit_array if 'pod_kv_hit_array' in locals() else np.zeros((n_samples, len(self.pod_ids), 1)),
            
            # Enhanced features for transformer model
            'positional_encodings': positional_encodings,
            'pod_features_with_staleness': pod_features_with_staleness,
            'cross_attention_inputs': cross_attention_inputs,
            
            # Request features
            'request_features': request_features,
            'request_numeric_features': request_numeric_features,
            'request_categorical_features': request_categorical_features,
            
            # Request-pod interaction
            'interaction_features': interaction_features,
            
            # Timestamps and feature timing
            'timestamps': timestamps,
            'feature_timing': feature_timing,
            
            # Identifiers and targets
            'pod_ids': self.pod_ids,
            'actions': actions,
            'rewards': rewards,
            'ttft_rewards': ttft_rewards,
            'tpot_rewards': tpot_rewards,
            
            # Statistics and metadata
            'feature_stats': self.feature_stats,
            'pod_features_list': self.pod_features,
            # 'feature_indices_map': feature_indices_map,
            'feature_indices_map': reference_feature_indices,
            'numeric_request_features': self.numeric_request_features,
            'categorical_request_features': self.categorical_request_features,
            'encoders': {
                'pod_encoder': self.pod_encoder,
                'selected_pod_encoder': self.selected_pod_encoder,
                'categorical_encoders': categorical_encoders
            }
        }
        
        prepare_for_encoding_overhead_summary = {
            'encoding.prepare_for_encoding.prepare_extract_pod_columns_overhead': extract_pod_columns_overhead*1000,
            'encoding.prepare_for_encoding.prepare_analyze_request_features_overhead': analyze_request_features_overhead*1000,
            'encoding.prepare_for_encoding.prepare_encode_pod_ids_overhead': encode_pod_ids_overhead*1000,
            'encoding.prepare_for_encoding.prepare_classify_feature_timing_overhead': classify_feature_timing_overhead*1000,
            'encoding.prepare_for_encoding.prepare_request_numeric_features_overhead': request_numeric_features_overhead*1000,
            'encoding.prepare_for_encoding.prepare_request_categorical_features_overhead': request_categorical_features_overhead*1000,
            'encoding.prepare_for_encoding.prepare_process_pod_features_overhead': process_pod_features_overhead*1000,
            'encoding.prepare_for_encoding.prepare_extract_actions_rewards_overhead': extract_actions_rewards_overhead*1000,
            'encoding.prepare_for_encoding.prepare_combine_request_features_overhead': combine_request_features_overhead*1000,
            'encoding.prepare_for_encoding.prepare_positional_encoding_overhead': positional_encoding_overhead*1000,
            'encoding.prepare_for_encoding.prepare_add_staleness_overhead': add_staleness_overhead*1000,
            'encoding.prepare_for_encoding.prepare_cross_attention_overhead': cross_attention_overhead*1000,
            'encoding.prepare_for_encoding.prepare_create_request_pod_interaction_overhead': create_request_pod_interaction_overhead*1000
        }
        return processed_data, prepare_for_encoding_overhead_summary, process_pod_features_overhead_summary

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
    train_processed, _, _ = processor.prepare_for_encoding(df, all_pods, request_features_train, request_features_reward)
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

# ## old
# def encode_for_inference(all_pods, df, request_stats, request_features_train, request_features_reward):
#     encode_for_inference_start_time = time.time()
#     # Check if we have running statistics for request features
#     if request_stats is not None and request_stats.count > 0:
#         logger.info(f"Using running statistics for normalization during inference (n={request_stats.count})")
        
#         if all(feature in df.columns for feature in request_features_train):
#             # Check for required features
#             missing_features = [f for f in request_features_train if f not in df.columns]
#             if missing_features:
#                 logger.error(f"Missing required request features: {missing_features}")
#                 assert False, f"Required features {missing_features} not found in DataFrame"
            
#             # Check for zero values
#             for feature in request_features_train:
#                 if feature in df.columns and (df[feature] == 0).all():
#                     logger.warning(f"Feature {feature} has all zero values. Setting to 0.01 to avoid scaling issues.")
#                     df[feature] = 0.01
            
#             # Log original feature values for debugging
#             logger.info("Request features before normalization:")
#             for feature in request_features_train:
#                 if len(df) > 0:
#                     logger.info(f"  {feature}: {df[feature].iloc[0]}")

#             request_values = df[request_features_train].values
#             normalized_values = request_stats.normalize(request_values)
            
#             # Store normalized values in DataFrame
#             for i, feature in enumerate(request_features_train):
#                 original_values = df[feature].copy()
#                 df[feature] = normalized_values[:, i]
                
#                 # Log normalization for the first row
#                 if len(df) > 0:
#                     logger.info(f"Normalized {feature}: {original_values.iloc[0]} -> {df[feature].iloc[0]}")
#         else:
#             logger.error(f"Some request features missing from DataFrame, using default normalization")
#             assert False
#     else:
#         logger.error(f"No running statistics provided. request_stats: {request_stats}, request_stats.count: {request_stats.count}")
#         assert False
    
#     processor = LLMRoutingDataProcessor(output_dir="temp_inference")
#     encoder_preprocess_start = time.time()
#     processed_data = processor.prepare_for_encoding(df, all_pods, request_features_train, request_features_reward)
#     encoder_preprocess_overhead = time.time() - encoder_preprocess_start

#     # Log processed request features
#     if 'request_features' in processed_data:
#         request_feat = processed_data['request_features']
#         logger.info(f"Processed request features shape: {request_feat.shape}")
#         if len(request_feat) > 0:
#             logger.info(f"Processed request features values: {request_feat[0]}")
    
#     tensor_data = {
#         # Basic tensors
#         'pod_features': torch.FloatTensor(processed_data['pod_features']),
#         'kv_hit_ratios': torch.FloatTensor(processed_data['kv_hit_ratios']),
#         'request_features': torch.FloatTensor(processed_data['request_features']),
#         'actions': torch.LongTensor(processed_data['actions']),
#         'rewards': torch.FloatTensor(processed_data['rewards']),
        
#         # Enhanced features for transformer
#         'positional_encodings': torch.FloatTensor(processed_data['positional_encodings']),
#         'pod_features_with_staleness': torch.FloatTensor(processed_data['pod_features_with_staleness']),
        
#         # Cross-attention components
#         'query': torch.FloatTensor(processed_data['cross_attention_inputs']['query']),
#         'key_value': torch.FloatTensor(processed_data['cross_attention_inputs']['key_value']),
#     }
    
#     # Add interaction features if available
#     if processed_data['interaction_features'] is not None:
#         tensor_data['interaction_features'] = torch.FloatTensor(processed_data['interaction_features'])
        
#     # Add additional reward components if available
#     if 'ttft_rewards' in processed_data and processed_data['ttft_rewards'] is not None:
#         tensor_data['ttft_rewards'] = torch.FloatTensor(processed_data['ttft_rewards'])
#     if 'tpot_rewards' in processed_data and processed_data['tpot_rewards'] is not None:
#         tensor_data['tpot_rewards'] = torch.FloatTensor(processed_data['tpot_rewards'])
#     total_overhead = time.time() - encode_for_inference_start_time
#     encoder_other_overhead = total_overhead - encoder_preprocess_overhead
#     return tensor_data, encoder_other_overhead, encoder_preprocess_overhead


def encode_for_inference(all_pods, df, request_stats, request_features_train, request_features_reward):
    """OPTIMIZED inference encoding function."""
    encode_for_inference_start_time = time.time()
    
    if request_stats is not None and request_stats.count > 0:
        logger.info(f"Using running statistics for normalization during inference (n={request_stats.count})")
        
        if all(feature in df.columns for feature in request_features_train):
            missing_features = [f for f in request_features_train if f not in df.columns]
            if missing_features:
                logger.error(f"Missing required request features: {missing_features}")
                assert False, f"Required features {missing_features} not found in DataFrame"
            
            # OPTIMIZATION: Vectorized zero-value handling
            zero_mask = (df[request_features_train] == 0).all(axis=0)
            if zero_mask.any():
                zero_features = df[request_features_train].columns[zero_mask].tolist()
                logger.warning(f"Features {zero_features} have all zero values. Setting to 0.01 to avoid scaling issues.")
                df.loc[:, zero_features] = 0.01
            
            logger.info("Request features before normalization:")
            if len(df) > 0:
                for feature in request_features_train:
                    logger.info(f"  {feature}: {df[feature].iloc[0]}")

            # OPTIMIZATION: Single vectorized normalization operation
            request_values = df[request_features_train].values
            normalized_values = request_stats.normalize(request_values)
            
            # OPTIMIZATION: Bulk assignment
            original_values = request_values.copy() if len(df) > 0 else None
            df[request_features_train] = normalized_values
            
            if len(df) > 0:
                for i, feature in enumerate(request_features_train):
                    logger.info(f"Normalized {feature}: {original_values[0, i]} -> {normalized_values[0, i]}")
        else:
            logger.error(f"Some request features missing from DataFrame, using default normalization")
            assert False
    else:
        logger.error(f"No running statistics provided. request_stats: {request_stats}, request_stats.count: {request_stats.count}")
        assert False
    
    processor = LLMRoutingDataProcessor(output_dir="temp_inference")
    prepare_for_encoding_start = time.time()
    processed_data, prepare_for_encoding_overhead_summary, process_pod_features_overhead_summary = processor.prepare_for_encoding(df, all_pods, request_features_train, request_features_reward)
    total_prepare_for_encoding_overhead = time.time() - prepare_for_encoding_start

    if 'request_features' in processed_data:
        request_feat = processed_data['request_features']
        logger.info(f"Processed request features shape: {request_feat.shape}")
        if len(request_feat) > 0:
            logger.info(f"Processed request features values: {request_feat[0]}")
    
    # OPTIMIZATION: Single tensor creation operation
    tensor_data = {
        'pod_features': torch.FloatTensor(processed_data['pod_features']),
        'kv_hit_ratios': torch.FloatTensor(processed_data['kv_hit_ratios']),
        'request_features': torch.FloatTensor(processed_data['request_features']),
        'actions': torch.LongTensor(processed_data['actions']),
        'rewards': torch.FloatTensor(processed_data['rewards']),
        'positional_encodings': torch.FloatTensor(processed_data['positional_encodings']),
        'pod_features_with_staleness': torch.FloatTensor(processed_data['pod_features_with_staleness']),
        'query': torch.FloatTensor(processed_data['cross_attention_inputs']['query']),
        'key_value': torch.FloatTensor(processed_data['cross_attention_inputs']['key_value']),
    }
    
    # OPTIMIZATION: Conditional tensor creation
    if processed_data['interaction_features'] is not None:
        tensor_data['interaction_features'] = torch.FloatTensor(processed_data['interaction_features'])
        
    if 'ttft_rewards' in processed_data and processed_data['ttft_rewards'] is not None:
        tensor_data['ttft_rewards'] = torch.FloatTensor(processed_data['ttft_rewards'])
    if 'tpot_rewards' in processed_data and processed_data['tpot_rewards'] is not None:
        tensor_data['tpot_rewards'] = torch.FloatTensor(processed_data['tpot_rewards'])
        
    total_overhead = time.time() - encode_for_inference_start_time
    encoder_other_overhead = total_overhead - total_prepare_for_encoding_overhead
    return tensor_data, encoder_other_overhead, total_prepare_for_encoding_overhead, prepare_for_encoding_overhead_summary, process_pod_features_overhead_summary

