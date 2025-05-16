import threading
import joblib
import pandas as pd
import numpy as np
import uvicorn
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union
import os
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys
import concurrent.futures
import encoding
import sac
import ppo
from flask import Flask, request, jsonify
from logger import logger
import preprocess

# Initialize thread pool for background processing
thread_pool = ThreadPoolExecutor(max_workers=16)

# Global variables for model and data management
training_data = []
model = None
model_lock = threading.Lock()
training_in_progress = False
training_lock = threading.Lock()

app = Flask(__name__)

batch_id = 0

@app.route("/flush", methods=["POST"])
def handle_flush():
    global batch_id
    try:
        # Parse the incoming JSON data
        log_data = request.json
        logger.info(f"Received log data with {len(log_data) if log_data else 0} entries")
        
        # Step 1: Write to file
        logger.info("Step 1: Writing log data to file...")
        raw_data = f"raw_data_batch_{batch_id}.csv"
        batch_id += 1
        try:
            with open(raw_data, "w") as log_file:
                for request_id, log_message in log_data.items():
                    log_file.write(f"{log_message}\n")
            logger.info(f"Successfully wrote {len(log_data)} entries to {raw_data}")
        except Exception as e:
            logger.error(f"Failed at step 1 (writing to file): {str(e)}")
            raise

        # # Step 2: Parse log file
        # logger.info("Step 2: Parsing log file...")
        # try:
        #     df, json_columns = preprocess.parse_log_file(raw_data)
        #     logger.info(f"Successfully parsed log file, got DataFrame with {len(df)} rows and {len(json_columns)} JSON columns")
        # except Exception as e:
        #     logger.error(f"Failed at step 2 (parsing log file): {str(e)}")
        #     raise
            
        # # Step 3: Parse JSON columns
        # logger.info("Step 3: Parsing JSON columns...")
        # try:
        #     df = preprocess.parse_json_columns(df, json_columns)
        #     logger.info(f"Successfully parsed JSON columns")
        # except Exception as e:
        #     logger.error(f"Failed at step 3 (parsing JSON columns): {str(e)}")
        #     raise
            
        # # Step 4: Normalize time
        # logger.info("Step 4: Normalizing time...")
        # try:
        #     df = preprocess.normalize_time(df)
        #     logger.info(f"Successfully normalized time, DataFrame now has {len(df)} rows")
        # except Exception as e:
        #     logger.error(f"Failed at step 4 (normalizing time): {str(e)}")
        #     raise
            
        # # Step 5: Preprocess dataset
        # logger.info("Step 5: Preprocessing dataset...")
        # preprocessed_file = f"processed_data_batch_{batch_id}.csv"
        # try:
        #     df, mapping_info, all_pods = preprocess.preprocess_dataset(df)
        #     logger.info(f"Successfully preprocessed dataset to {preprocessed_file}")
        #     logger.info(f"Pod mapping: {mapping_info['pod_to_index']}")
        # except Exception as e:
        #     logger.error(f"Failed at step 5 (preprocessing dataset): {str(e)}")
        #     raise

        df, preprocessed_file, all_pods = preprocess.main(raw_data)
        #########################################################
            
        # Step 6: Encoding data
        logger.info("Step 6: Encoding data...")
        encoded_data_dir = "encoded_data"
        encoded_data_subdir = f"{encoded_data_dir}/batch_{batch_id}"
        try:
            encoding.encode(all_pods, df, encoded_data_subdir)
            logger.info(f"Successfully encoded data to {encoded_data_subdir}")
        except Exception as e:
            logger.error(f"Failed at step 6 (encoding data): {str(e)}")
            raise
            
        # Step 7: Training routing agent
        logger.info("Step 7: Training routing agent...")
        try:
            # sac.train(encoded_data_dir)
            ppo.train(encoded_data_dir)
            logger.info("Successfully trained routing agent")
        except Exception as e:
            logger.error(f"Failed at step 7 (training routing agent): {str(e)}")
            raise
            
        return jsonify({"status": "success", "message": f"Successfully processed {len(log_data)} log messages"}), 200
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        return jsonify({"status": "error", "message": str(e), "traceback": error_traceback}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)