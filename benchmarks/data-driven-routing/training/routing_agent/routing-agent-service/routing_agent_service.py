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

app = Flask(__name__)

BATCH_ID = 0

def write_to_file(log_data, raw_data):
    with open(raw_data, "w") as log_file:
        for request_id, log_message in log_data.items():
            log_file.write(f"{log_message}\n")
    logger.info(f"Successfully wrote {len(log_data)} entries to {raw_data}")
    
@app.route("/flush", methods=["POST"])
def handle_flush():
    global BATCH_ID
    log_data = request.json
    try:
        logger.info(f"Received log data with {len(log_data) if log_data else 0} entries")
        raw_data = f"raw_data_batch_{BATCH_ID}.csv"
        BATCH_ID += 1
        
        # Write raw data to file
        write_to_file(log_data, raw_data)

        # Preprocess raw data
        df, preprocessed_file, all_pods = preprocess.main(raw_data)
        logger.info(f"Successfully parsed data.  (writte in  {preprocessed_file})")

        # Encode preprocessed data
        encoded_data_dir = "encoded_data"
        encoded_data_subdir = f"{encoded_data_dir}/batch_{BATCH_ID}"
        encoding.encode(all_pods, df, encoded_data_subdir)
        logger.info(f"Successfully encoded data to {encoded_data_subdir}")
            
        # Train the model
        train(encoded_data_dir)
        logger.info(f"Successfully train the model")
            
        return jsonify({"status": "success", "message": f"Successfully processed {len(log_data)} log messages"}), 200
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        return jsonify({"status": "error", "message": str(e), "traceback": error_traceback}), 500

def train(encoded_data_dir):
    try:
        # sac.train(encoded_data_dir)
        ppo.train(encoded_data_dir)
        logger.info("Successfully trained routing agent")
    except Exception as e:
        logger.error(f"Failed at step 7 (training routing agent): {str(e)}")
        raise

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)