import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def make_predictions(model_file, input_file, output_file=None):
    """
    Make predictions using a trained model.
    
    Parameters:
    -----------
    model_file : str
        Path to the trained model file
    input_file : str
        Path to the input CSV file with features
    output_file : str, optional
        Path to save the predictions
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with predictions
    """
    # Check if files exist
    if not os.path.exists(model_file):
        logger.error(f"Model file not found: {model_file}")
        return None
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return None
    
    # Load model
    logger.info(f"Loading model from {model_file}")
    try:
        model_data = joblib.load(model_file)
        models = model_data['models']
        logger.info(f"Loaded models for targets: {list(models.keys())}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None
    
    # Load input data
    logger.info(f"Loading data from {input_file}")
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None
    
    # Make predictions for each target
    preds_df = df.copy()
    
    for target, model in models.items():
        logger.info(f"Making predictions for {target}")
        try:
            # Make predictions
            preds = model.predict(df)
            preds_df[f"predicted_{target}"] = preds
            
            # Print summary statistics
            logger.info(f"Prediction stats for {target}:")
            logger.info(f"  Min: {preds.min():.2f}")
            logger.info(f"  Max: {preds.max():.2f}")
            logger.info(f"  Mean: {preds.mean():.2f}")
            logger.info(f"  Std: {preds.std():.2f}")
        except Exception as e:
            logger.error(f"Error making predictions for {target}: {e}")
    
    # Save predictions
    if output_file:
        # Save as CSV
        if output_file.endswith('.csv'):
            preds_df.to_csv(output_file, index=False)
        # Save as JSON
        elif output_file.endswith('.json'):
            # Convert to list of records
            records = []
            for i, row in preds_df.iterrows():
                record = row.to_dict()
                records.append(record)
            
            # Write to file
            with open(output_file, 'w') as f:
                json.dump(records, f, indent=2)
        else:
            # Default to CSV
            preds_df.to_csv(output_file, index=False)
        
        logger.info(f"Saved predictions to {output_file}")
    
    return preds_df

if __name__ == "__main__":
    # Parse arguments
    if len(sys.argv) < 3:
        print("Usage: python simple_inference.py <model_file> <input_file> [output_file]")
        sys.exit(1)
    
    model_file = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "predictions.csv"
    
    # Make predictions
    preds_df = make_predictions(model_file, input_file, output_file)
    
    if preds_df is None:
        sys.exit(1)