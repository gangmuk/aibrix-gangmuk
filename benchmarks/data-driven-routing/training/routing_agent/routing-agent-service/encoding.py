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
        """Extract pod-related columns and organize by pod ID and feature type."""
        pod_data = defaultdict(dict)
        
        logger.info(f"df.columns: {df.columns}")
        self.pod_ids = all_pods
        logger.info(f"Found pod IDs from selected_pod column: {self.pod_ids}")

        # Direct check for column patterns
        pod_feature_columns = [col for col in df.columns if col.startswith('pod_')]
        for col in pod_feature_columns:
            # Look for columns like 'pod_10.0.1.32-feature_name'
            parts = col.split('-')
            assert len(parts) == 2, f"Unexpected column format: {col}"
            pod_id = parts[0].replace('pod_', '')
            feature = parts[1]
            if feature not in self.pod_features:
                self.pod_features.append(feature)
            logger.debug(f"pod_id: {pod_id}, self.pod_ids: {self.pod_ids}")
            if pod_id in self.pod_ids:
                pod_data[pod_id][feature] = df[col]
            else:
                logger.error(f"Pod ID {pod_id} not found in self.pod_ids {self.pod_ids}, col: {col}")
                assert False
        # all pods must have the same number of columns for pod features
        for pod_id in self.pod_ids:
            if pod_id not in pod_data:
                logger.error(f"Pod ID {pod_id} not found in pod_data")
                assert False
            # Check if all pods have the same features
            if len(pod_data[pod_id]) != len(self.pod_features):
                logger.error(f"Pod ID {pod_id} has {len(pod_data[pod_id])} features, expected {len(self.pod_features)}")
                assert False

        logger.info(f"pod_data contains {len(pod_data)} pods and total of {sum(len(features) for features in pod_data.values())} features")
        self.pod_features = sorted(self.pod_features)
        return pod_data
        
        # # Log features
        # self.pod_features = sorted(self.pod_features)
        # logger.info(f"Extracted {len(self.pod_ids)} pod IDs and {len(self.pod_features)} pod features")
        # logger.info(f"Pod IDs: {self.pod_ids[:5]}{'...' if len(self.pod_ids) > 5 else ''}")
        # logger.info(f"Pod features: {self.pod_features[:5]}{'...' if len(self.pod_features) > 5 else ''}")
        
        # # Log last_second features specifically
        # last_second_features = [f for f in self.pod_features if 'last_second' in f]
        # logger.info(f"Found {len(last_second_features)} last_second features: {last_second_features}")
        
        # return pod_data

    def analyze_request_features(self, df):
        # Columns to exclude from features
        exclude_cols = [
            'request_id',           # Identifier, not a feature
            'selected_pod',         # Target, not a feature
            'action',               # Target, not a feature
            'reward',               # Target, not a feature
            'ttft_reward',          # Component of reward, not a feature
            'tpot_reward',          # Component of reward, not a feature
            'ttft_normalized',      # Derived from reward
            'tpot_normalized',      # Derived from reward
        ]
        exclude_patterns = ['reward', 'action', 'slo_satisfied', 'normalized']
        
        # Any column not starting with "pod_" and not in exclude list
        candidate_request_features = [
            col for col in df.columns 
            if not any(col.startswith(f"pod_{pod_id}") for pod_id in self.pod_ids) 
            and not any(pat in col for pat in exclude_patterns)
            and col not in exclude_cols
        ]
        
        logger.info(f"Request features - found {len(candidate_request_features)} candidate columns: {candidate_request_features}")

        # Test each column to see if it's numeric or categorical
        numeric_cols = []
        categorical_cols = []
        
        for col in candidate_request_features:
            # Skip columns with too many NaN values
            if df[col].isna().mean() > 0:
                logger.error(f"Request features - {col} has NaN values.")
                assert False
                
            # Try to convert to numeric
            try:
                # Check if already numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
                    continue
                
                # Try to convert
                pd.to_numeric(df[col])
                numeric_cols.append(col)
            except:
                # If conversion fails, it's categorical
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
        """Create encoders for pod IDs.
        
        Args:
            df: Pandas DataFrame with routing data
        """
        # If we have pod IDs, create an encoder
        if self.pod_ids:
            # Fit an encoder for pod IDs
            self.pod_encoder = OneHotEncoder(sparse_output=False)
            self.pod_encoder.fit(np.array(self.pod_ids).reshape(-1, 1))
            
            # Also fit an encoder for selected_pod column
            if 'selected_pod' in df.columns:
                selected_pods = df['selected_pod'].dropna().unique()
                self.selected_pod_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                self.selected_pod_encoder.fit(np.array(selected_pods).reshape(-1, 1))
                
                logger.info(f"Encoded {len(selected_pods)} unique selected pods")
        else:
            logger.warning("No pod IDs found, skipping pod encoding")

    def classify_feature_timing(self):
        feature_timing = {}
        logger.info(f"Classifying timing for pod features: {self.pod_features}")
        for feature in self.pod_features:
            if 'last_second' in feature:
                feature_timing[feature] = 'historical'
            else:
                feature_timing[feature] = 'current'
        current_features = [f for f, timing in feature_timing.items() if timing == 'current']
        historical_features = [f for f, timing in feature_timing.items() if timing == 'historical']
        logger.info(f"Current-time features: {current_features}")
        logger.info(f"historical features: {historical_features}")
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
        """Add staleness indicators for historical features.
        
        Args:
            pod_features: Pod feature tensor [batch, n_pods, feature_dim]
            timestamps: Request timestamps
            feature_timing: Dictionary mapping features to timing category
            feature_indices_map: Dictionary mapping feature names to indices
            
        Returns:
            Pod features with staleness indicators added
        """
        # Get indices of historical features
        historical_features = [f for f, timing in feature_timing.items() if timing == 'historical']
        historical_indices = [
            idx for feature, idx in feature_indices_map.items() 
            if feature in historical_features
        ]
        
        if not historical_indices or len(timestamps) == 0 or np.all(timestamps == 0):
            logger.info("No historical features or valid timestamps, skipping staleness")
            # Add dummy staleness feature (all zeros)
            staleness_features = np.zeros((pod_features.shape[0], pod_features.shape[1], 1))
            return np.concatenate([pod_features, staleness_features], axis=2)
        
        # Calculate staleness based on timestamps (normalize to [0,1] range)
        # Assuming timestamps are in seconds from reference point
        max_staleness = 60.0  # 60 seconds max staleness
        
        # Calculate time differences between consecutive requests
        # Sort timestamps and get differences
        sorted_indices = np.argsort(timestamps)
        sorted_timestamps = timestamps[sorted_indices]
        time_diffs = np.diff(sorted_timestamps, prepend=sorted_timestamps[0])
        time_diffs = np.maximum(time_diffs, 0)  # Ensure positive
        
        # Reorder to original sequence and normalize
        staleness = np.zeros_like(timestamps)
        staleness[sorted_indices] = time_diffs
        staleness = np.clip(staleness / max_staleness, 0, 1)
        
        # Create staleness feature for each pod
        staleness_features = np.zeros((pod_features.shape[0], pod_features.shape[1], 1))
        for i in range(pod_features.shape[1]):  # For each pod
            staleness_features[:, i, 0] = staleness
        
        logger.info(f"Added staleness indicator for {len(historical_indices)} historical features")
        
        # Concatenate with original features
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
        """Create features capturing interactions between request and pod characteristics.
        
        Args:
            request_features: Request feature tensor [batch, request_dim]
            pod_features: Pod feature tensor [batch, n_pods, pod_dim]
            
        Returns:
            Interaction features for each pod
        """
        if request_features.shape[1] == 0:
            logger.warning("No request features available for interaction")
            return None
            
        batch_size, n_pods, _ = pod_features.shape
        
        # Expand request features to match pod dimensions
        expanded_request = np.repeat(
            request_features[:, np.newaxis, :], n_pods, axis=1
        )
        
        logger.info(f"Created request-pod interaction features with shape {expanded_request.shape}")
        
        return expanded_request

    def preprocess_data(self, df, all_pods):
        # Step 1: Extract pod-related columns
        pod_data = self.extract_pod_columns(df, all_pods)
        
        # Step 2: Analyze request features
        self.analyze_request_features(df)
        
        # Step 3: Encode pod IDs
        self.encode_pod_ids(df)
        
        # Step 4: Classify feature timing (historical vs current)
        feature_timing = self.classify_feature_timing()
        
        # Step 5: Process request features
        n_samples = len(df)
        
        # Process numeric request features
        request_numeric_features = None
        if self.numeric_request_features:
            request_numeric_features = df[self.numeric_request_features].fillna(0).values
            # Fit scaler
            self.request_feature_scaler.fit(request_numeric_features)
            # Store statistics
            self.feature_stats['request_feature_means'] = self.request_feature_scaler.mean_
            self.feature_stats['request_feature_stds'] = self.request_feature_scaler.scale_
            # Transform
            request_numeric_features = self.request_feature_scaler.transform(request_numeric_features)
        else:
            request_numeric_features = np.zeros((n_samples, 0))
            
        # Process categorical request features
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
        
        # Step 6: Process pod features
        pod_features_list = []
        pod_kv_hit_ratios = []
        
        # Create a per-pod feature indices map instead of a global one
        per_pod_feature_indices = {}
        
        # If we have pod data in the expected format
        if pod_data:
            for pod_id in self.pod_ids:
                pod_features = []
                # Start fresh for each pod
                feature_indices_map = {}
                feature_idx = 0
                
                # Process each feature for this pod
                for feature in self.pod_features:
                    if feature in pod_data[pod_id]:
                        # Special handling for kv_hit_ratio
                        if feature == 'kv_hit_ratio':
                            kv_values = pod_data[pod_id][feature].fillna(0).values.reshape(-1, 1)
                            pod_kv_hit_ratios.append(kv_values)
                        elif feature == 'gpu_model':
                            # One-hot encode GPU model
                            gpu_values = pod_data[pod_id][feature].fillna('unknown')
                            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                            encoder.fit(gpu_values.values.reshape(-1, 1))
                            encoded = encoder.transform(gpu_values.values.reshape(-1, 1))
                            pod_features.append(encoded)
                            # Add to feature indices map for this pod
                            feature_indices_map[feature] = feature_idx
                            feature_idx += encoded.shape[1]
                        else:
                            # Regular numeric feature
                            values = pod_data[pod_id][feature].fillna(0).values.reshape(-1, 1)
                            pod_features.append(values)
                            # Add to feature indices map for this pod
                            feature_indices_map[feature] = feature_idx
                            feature_idx += 1
                    else:
                        logger.error(f"Feature {feature} not found for pod {pod_id}")
                        assert False
                
                # Store feature indices map for this pod
                per_pod_feature_indices[pod_id] = feature_indices_map
                
                # Combine all features for this pod
                if pod_features:
                    pod_features = np.hstack(pod_features)
                    pod_features_list.append(pod_features)
                else:
                    logger.error(f"No features found for pod {pod_id}")
                    assert False
            
            # Stack features for all pods
            if pod_features_list:
                pod_features_array = np.stack(pod_features_list, axis=1)
                pod_kv_hit_array = np.stack(pod_kv_hit_ratios, axis=1)
                
                ## Normalize pod features
                ## NOTE: normalization can cause issue, making all values zero if raw values in different pods are similar
                pod_shape = pod_features_array.shape
                kv_shape = pod_kv_hit_array.shape
                pod_features_flat = pod_features_array.reshape(-1, pod_features_array.shape[2])
                kv_flat = pod_kv_hit_array.reshape(-1, 1)
                self.pod_feature_scaler.fit(pod_features_flat)
                self.kv_hit_scaler.fit(kv_flat)
                pod_features_norm = self.pod_feature_scaler.transform(pod_features_flat).reshape(pod_shape)
                kv_hit_norm = self.kv_hit_scaler.transform(kv_flat).reshape(kv_shape)

                # pod_features_norm = pod_features_array  # Just use raw values
                # kv_hit_norm = pod_kv_hit_array  # Just use raw values

                # Add after processing pod features and KV hit ratio:
                logger.info(f"Pod features stats after processing: min={pod_features_norm.min()}, max={pod_features_norm.max()}, non-zero={np.count_nonzero(pod_features_norm)}/{pod_features_norm.size}")
                logger.info(f"KV hit ratio stats after processing: min={kv_hit_norm.min()}, max={kv_hit_norm.max()}, non-zero={np.count_nonzero(kv_hit_norm)}/{kv_hit_norm.size}")

                self.feature_stats['pod_feature_means'] = self.pod_feature_scaler.mean_
                self.feature_stats['pod_feature_stds'] = self.pod_feature_scaler.scale_
                self.feature_stats['kv_hit_means'] = self.kv_hit_scaler.mean_
                self.feature_stats['kv_hit_stds'] = self.kv_hit_scaler.scale_
            else:
                logger.error("No pod features found in expected format")
                assert False
        else:
            logger.error("No pod data in expected format, creating default pod features")
            assert False
        
        # Step 7: Extract actions and rewards
        actions = np.zeros(n_samples, dtype=np.int64)
        rewards = np.zeros(n_samples)
        ttft_rewards = np.zeros(n_samples)
        tpot_rewards = np.zeros(n_samples)
        
        # Extract selected pod and convert to action index
        if 'selected_pod' in df.columns:
            pod_to_idx = {pod_id: i for i, pod_id in enumerate(self.pod_ids)}
            for i, selected_pod in enumerate(df['selected_pod'].values):
                if pd.notna(selected_pod) and str(selected_pod) in pod_to_idx:
                    actions[i] = pod_to_idx[str(selected_pod)]
        elif 'action' in df.columns:
            actions = df['action'].fillna(0).astype(np.int64).values
        
        # Extract rewards
        if 'reward' in df.columns:
            rewards = df['reward'].fillna(0).values
        if 'ttft_reward' in df.columns:
            ttft_rewards = df['ttft_reward'].fillna(0).values
        if 'tpot_reward' in df.columns:
            tpot_rewards = df['tpot_reward'].fillna(0).values
        
        # Step 8: Combine request features
        request_features = np.hstack([
            request_numeric_features, 
            request_categorical_features
        ]) if request_categorical_features.size > 0 else request_numeric_features
        
        # Step 9: Create timestamps
        timestamps = np.zeros(n_samples)
        if 'request_start_time' in df.columns:
            try:
                timestamps = pd.to_numeric(df['request_start_time']).values
                min_timestamp = timestamps.min() if len(timestamps) > 0 else 0
                timestamps = (timestamps - min_timestamp) / 1000.0  # Convert to seconds from reference
            except:
                logger.warning("Could not convert request_start_time to numeric values")
        
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
        positional_encodings = self.prepare_metrics_based_positional_encoding(
            pod_features_norm, reference_feature_indices
        )
        
        # # Step 11: Add staleness features for historical metrics
        # pod_features_with_staleness = self.add_staleness_features(
        #     pod_features_norm, timestamps, feature_timing, feature_indices_map
        # )
        pod_features_with_staleness = self.add_staleness_features(
            pod_features_norm, timestamps, feature_timing, reference_feature_indices
        )
        
        # Step 12: Prepare cross-attention inputs
        cross_attention_inputs = self.prepare_cross_attention_inputs(
            pod_features_with_staleness, kv_hit_norm
        )
        
        # Step 13: Create request-pod interaction features
        interaction_features = self.create_request_pod_interaction_features(
            request_features, pod_features_norm
        )
        
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
            'feature_indices_map': feature_indices_map,
            'numeric_request_features': self.numeric_request_features,
            'categorical_request_features': self.categorical_request_features,
            'encoders': {
                'pod_encoder': self.pod_encoder,
                'selected_pod_encoder': self.selected_pod_encoder,
                'categorical_encoders': categorical_encoders
            }
        }
        
        
        return processed_data

    def save_processed_data(self, processed_data, prefix="train"):
        """Save the processed data to disk.
        
        Args:
            processed_data: Dictionary with preprocessed data
            prefix: Prefix for output files (e.g., 'train', 'val', 'test')
        """
        # Create a timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_dir, f"{prefix}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each component
        for key, data in processed_data.items():
            if key == 'encoders':
                # Save encoders separately
                continue
            # elif isinstance(data, np.ndarray):
            #     np.save(os.path.join(output_dir, f"{key}.npy"), data)
            elif isinstance(data, list) or isinstance(data, dict):
                with open(os.path.join(output_dir, f"{key}.pkl"), 'wb') as f:
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
        
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved processed data to {output_dir}")
        
        # Return the output directory for reference
        return output_dir

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


def encode(all_pods, df, output_dir):
    """Main function to process LLM routing data."""
    test_split = 0.2
    random_seed = 42
    batch_size = 32
    create_loaders = False
    
    # Set random seed
    np.random.seed(random_seed)
    
    # logger.info(f"Processing input file: {input_file}")
    
    # # Load the data
    # try:
    #     df = pd.read_csv(input_file)
    #     logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    # except Exception as e:
    #     logger.error(f"Failed to load data: {e}")
    #     sys.exit(1)
    
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
    
    # Process training data
    logger.info("Processing training data...")
    train_processed = processor.preprocess_data(df, all_pods)
    train_path = processor.save_processed_data(train_processed, prefix="train")
    
    # # Process test data
    # logger.info("Processing test data...")
    # test_processed = processor.preprocess_data(test_df, all_pods)
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

if __name__ == "__main__":
    encode() # this should be updated