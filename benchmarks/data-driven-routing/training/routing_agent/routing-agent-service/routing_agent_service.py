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

app = Flask(__name__)

BATCH_ID = 0
ENCODED_DATA_DIR = "encoded_data"

def write_to_file(log_data, raw_data):
    with open(raw_data, "w") as log_file:
        for request_id, log_message in log_data.items():
            log_file.write(f"{log_message}\n")
    logger.info(f"Successfully wrote {len(log_data)} entries to {raw_data}")

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

        # Encode preprocessed data
        encoded_data_subdir = f"{ENCODED_DATA_DIR}/batch_{BATCH_ID}"
        encoding.encode(all_pods, df, encoded_data_subdir)
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
        # This will extract all pod info and create the processed dataframe
        processed_df, _, all_pods = preprocess.main(raw_data)
        logger.info(f"Successfully parsed data for request_{request_id}")
        
        # Encode the preprocessed data
        # This will create tensor_dataset.pt in the output directory

        ## previous approach. file based    
        # encoded_data_subdir = f"inference_encoded/request_{request_id}"
        # encoding.encode(all_pods, processed_df, encoded_data_subdir)
        # logger.info(f"Successfully encoded data to {encoded_data_subdir}")
        # tensor_dataset_path = os.path.join(encoded_data_subdir, "train", "tensor_dataset.pt")
        # if not os.path.exists(tensor_dataset_path):
        #     logger.error(f"Tensor dataset not found at {tensor_dataset_path}")
        #     return jsonify({"error": "Failed to encode data"}), 500
        # result = contextual_bandit.infer(tensor_dataset_path)

        ## new approach. in memory tensor dataset
        tensor_dataset = encoding.encode_for_inference(all_pods, processed_df)
        logger.info(f"Successfully encoded data in memory for inference")
        result = contextual_bandit.infer_from_tensor(tensor_dataset)
        

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