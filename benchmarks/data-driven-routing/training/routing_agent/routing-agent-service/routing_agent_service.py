# routing_agent_service.py

# import threading
# import joblib
import pandas as pd
import numpy as np
# import uvicorn
# from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union
import os
import logging
import time
# import asyncio
# from concurrent.futures import ThreadPoolExecutor
import sys
# import concurrent.futures
import encoding
# import sac
import ppo
import contextual_bandit
from flask import Flask, request, jsonify
# from apscheduler.schedulers.background import BackgroundScheduler
# import atexit
from logger import logger
import preprocess
import pickle

app = Flask(__name__)

BATCH_ID = 0
ENCODED_DATA_DIR = "encoded_data"
STATS_FILE = "request_feature_stats.pkl"  # Add this near the top with your other constants

class RunningStats:
    """Maintains running mean and standard deviation for feature normalization"""
    def __init__(self, feature_names=None):
        self.count = 0
        self.mean = None
        self.var = None  # Variance
        self.feature_names = feature_names
        
    def update(self, new_data):
        """Update statistics with new batch of data"""
        if new_data is None or len(new_data) == 0:
            return
        
        # Convert to numpy array
        new_data = np.array(new_data, dtype=np.float64)
        new_count = len(new_data)
        
        # First update
        if self.count == 0:
            self.mean = np.mean(new_data, axis=0)
            self.var = np.var(new_data, axis=0) * new_count
            self.count = new_count
            logger.info(f"Initialized running stats with {new_count} samples")
            return
        
        # Compute batch statistics
        batch_mean = np.mean(new_data, axis=0)
        batch_var = np.var(new_data, axis=0) * new_count
        
        # Update running statistics using Welford's algorithm
        new_count = len(new_data)
        new_total = self.count + new_count
        
        # Update mean
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * new_count / new_total
        
        # Update variance
        self.var = self.var + batch_var + delta**2 * self.count * new_count / new_total
        
        # Update count
        self.count = new_total
        
        logger.info(f"Updated running stats, now have {self.count} samples")
        
    def get_mean(self):
        """Get current mean"""
        return self.mean if self.mean is not None else 0
        
    def get_std(self):
        """Get current standard deviation"""
        if self.count <= 1 or self.var is None:
            return np.ones_like(self.mean) if self.mean is not None else 1.0
        std = np.sqrt(self.var / self.count)
        # Ensure no zeros to prevent division by zero during normalization
        if isinstance(std, np.ndarray):
            std[std < 1.0] = 1.0
        return std
        
    def normalize(self, data):
        """Normalize data using current statistics"""
        if self.count == 0:
            logger.warning("No statistics available, returning original data")
            return data
        
        mean = self.get_mean()
        std = self.get_std()
        
        return (data - mean) / std
        
    def save(self, filename):
        """Save statistics to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'count': self.count,
                'mean': self.mean,
                'var': self.var,
                'feature_names': self.feature_names
            }, f)
        logger.info(f"Saved running statistics to {filename}")
        
    @classmethod
    def load(cls, filename):
        """Load statistics from file"""
        if not os.path.exists(filename):
            logger.info(f"Statistics file {filename} not found, initializing new stats")
            return cls()
            
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            
        stats = cls(feature_names=data.get('feature_names'))
        stats.count = data.get('count', 0)
        stats.mean = data.get('mean')
        stats.var = data.get('var')
        
        logger.info(f"Loaded running statistics from {filename} with {stats.count} samples")
        return stats

request_stats = None

def get_request_stats():
    """Get or initialize request feature statistics"""
    global request_stats
    if request_stats is None:
        request_stats = RunningStats.load(STATS_FILE)
    return request_stats

def write_to_file(log_data, raw_data):
    with open(raw_data, "w") as log_file:
        for request_id, log_message in log_data.items():
            log_file.write(f"{log_message}\n")
    logger.info(f"Successfully wrote {len(log_data)} entries to {raw_data}")

request_features_train = ['input_tokens', 'output_tokens', 'total_tokens']
request_features_reward = ['ttft', 'avg_tpot', 'e2e_latency']

@app.route("/flush", methods=["POST"])
def handle_flush():
    global BATCH_ID, ENCODED_DATA_DIR
    log_data = request.json
    try:
        logger.info(f"Received log data with {len(log_data) if log_data else 0} entries")
        raw_data = f"raw_data_batch_{BATCH_ID}.csv"
        BATCH_ID += 1
        
        # Write raw data to file
        write_to_file(log_data, raw_data)

        # Preprocess raw data
        df, preprocessed_file, all_pods = preprocess.main(raw_data)
        logger.info(f"Successfully parsed data. (writte in  {preprocessed_file})")
        
        # Update running statistics
        # request_features = ['input_tokens', 'output_tokens', 'total_tokens', 'ttft', 'avg_tpot', 'e2e_latency']
        request_features = ['input_tokens', 'output_tokens', 'total_tokens']
        stats = get_request_stats()
        stats.update(df[request_features].values)
        stats.save(STATS_FILE)
        
        # Apply normalization using the updated running statistics
        # normalized_values = stats.normalize(df[request_features].values)
        normalized_values = stats.normalize(df[request_features_train].values)
        for i, feature in enumerate(request_features):
            df[feature] = normalized_values[:, i]

        # Encode preprocessed data
        encoded_data_subdir = f"{ENCODED_DATA_DIR}/batch_{BATCH_ID}"
        encoding.encode_for_train(all_pods, df, encoded_data_subdir, stats, request_features_train, request_features_reward)
        logger.info(f"Successfully encoded data to {encoded_data_subdir}")

        # sac.train(ENCODED_DATA_DIR)
        # ppo.train(ENCODED_DATA_DIR)
        contextual_bandit.train(ENCODED_DATA_DIR)
        logger.info("Successfully trained routing agent")
            
        return jsonify({"status": "success", "message": f"Successfully processed {len(log_data)} log messages"}), 200
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        return jsonify({"status": "error", "message": str(e), "traceback": error_traceback}), 500

@app.route("/infer", methods=["POST"])
def handle_infer():
    try:
        # Get the log message as a string from the request body
        log_message = request.data.decode('utf-8')
        logger.info(f"Received inference request: {log_message[:100]}...")
        
        # Extract request ID for logging purposes
        request_id = "unknown"
        if "requestID@" in log_message:
            parts = log_message.split("requestID@")
            if len(parts) > 1:
                request_id_parts = parts[1].split("@")
                if request_id_parts:
                    request_id = request_id_parts[0]
        
        logger.info(f"Processing inference request for request ID: {request_id}")
        
        # Create a temporary file with the single log message
        raw_data = f"infer_request_{request_id}.csv"
        with open(raw_data, "w") as log_file:
            log_file.write(f"{log_message}\n")
        
        # Use the existing preprocessing function to parse the log
        processed_df, _, all_pods = preprocess.main(raw_data)
        logger.info(f"Successfully parsed data for request_{request_id}")
        
        # Print essential request features immediately after preprocessing
        logger.info("Important request features after preprocessing:")
        for feature in ['input_tokens', 'output_tokens', 'total_tokens', 'ttft', 'avg_tpot']:
            if feature in processed_df.columns:
                value = processed_df[feature].iloc[0] if len(processed_df) > 0 else "N/A"
                logger.info(f"  {feature}: {value}")

        # Get running statistics
        stats = get_request_stats()

        ## new approach. in memory tensor dataset
        tensor_dataset = encoding.encode_for_inference(all_pods, processed_df, stats, request_features_train, request_features_reward)
        logger.info(f"Successfully encoded data in memory for inference")
        result = contextual_bandit.infer_from_tensor(tensor_dataset)

        # ## debugging
        # tensor_dataset = encoding.fix_encode_for_inference_with_feature_info(all_pods, processed_df, request_features_train, request_features_reward)
        # logger.info(f"Successfully encoded data in memory for inference with feature information")
        # encoding.debug_request_feature_encoding(processed_df, tensor_dataset)

        logger.info(f"Inference result: {result}")
        
        # Map the pod index back to the actual pod ID
        selected_pod_index = result.get('selected_pod_index', 0)
        if selected_pod_index >= len(all_pods):
            logger.warning(f"Selected pod index {selected_pod_index} out of range, defaulting to first pod")
            selected_pod_index = 0
            
        selected_pod = all_pods[selected_pod_index]
        confidence = result.get('confidence', 1.0)
        
        # Return the result
        response = {
            "selected_pod": selected_pod,
            "confidence": confidence,
            "request_id": request_id
        }
        
        logger.info(f"Selected pod {selected_pod} with confidence {confidence}")
        return jsonify(response), 200
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error in handle_infer: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    
    # scheduler = BackgroundScheduler()
    # scheduler.add_job(func=train, trigger="interval", seconds=60)
    # scheduler.start()
    # atexit.register(lambda: scheduler.shutdown())

    app.run(host="0.0.0.0", port=port, debug=False)