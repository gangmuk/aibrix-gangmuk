#!/usr/bin/env python3
import argparse
import pandas as pd
import requests
import json
import sys
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
from tabulate import tabulate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Validate the Latency Predictor Service')
    parser.add_argument('--test-file', '-f', required=True, help='Path to the test CSV file')
    parser.add_argument('--service-url', '-u', default='http://localhost:8080', help='Base URL of the latency predictor service')
    parser.add_argument('--num-samples', '-n', type=int, default=0, help='Number of samples to test with (0 for all)')
    parser.add_argument('--output', '-o', help='Path to save validation results (optional)')
    parser.add_argument('--visualize', '-v', action='store_true', help='Create visualizations of results')
    return parser.parse_args()

def check_health(service_url: str) -> bool:
    """Check if the service is healthy"""
    try:
        response = requests.get(f"{service_url}/health")
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Service is healthy. Available models: {data.get('models', [])}")
            return True
        else:
            logger.error(f"Health check failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error connecting to service: {e}")
        return False

def prepare_test_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Prepare test data from DataFrame"""
    pods = []
    
    for idx, row in df.iterrows():
        # Basic required fields
        selected_pod = row.get("selected_pod", f"10.1.0.{idx+1}")  # Use selected_pod if available, otherwise create fake IP
        
        pod_features = {
            "selected_pod": selected_pod,
            "input_tokens": int(row["input_tokens"]),
            "output_tokens": int(row["output_tokens"]),
            "total_tokens": int(row["total_tokens"])
        }
        
        # Add all other fields that aren't in the exclude list
        exclude_columns = ["input_tokens", "output_tokens", "total_tokens", "set", "selected_pod"]
        
        for col in df.columns:
            if col in exclude_columns:
                continue
            
            if pd.notna(row[col]):
                value = row[col]
                
                # Handle different data types
                if isinstance(value, (int, float, str)):
                    pod_features[col] = value
                else:
                    # Try to convert to appropriate type
                    try:
                        if isinstance(value, str) and '.' in value:
                            pod_features[col] = float(value)
                        elif isinstance(value, str):
                            pod_features[col] = int(value)
                        else:
                            pod_features[col] = str(value)
                    except (ValueError, TypeError):
                        pod_features[col] = str(value)
        
        pods.append(pod_features)
    
    return pods

def test_batch_prediction(service_url: str, pods: List[Dict[str, Any]]) -> Optional[Dict]:
    """Test the prediction endpoint with a batch of pods"""
    try:
        request_data = {"pods": pods}
        logger.info(f"Sending prediction request with {len(pods)} pods")
        
        response = requests.post(
            f"{service_url}/predict",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info("Prediction successful")
            return result
        else:
            logger.error(f"Prediction failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return None

def calculate_metrics(df: pd.DataFrame, predictions: Dict) -> Tuple[Dict, pd.DataFrame]:
    """Calculate validation metrics comparing predictions to actual values"""
    metrics = {}
    results_df = pd.DataFrame()
    
    # Expected targets in predictions
    targets = ["ttft", "avg_tpot"]
    
    # Prepare arrays for actual and predicted values
    actuals = {}
    preds = {}
    
    for target in targets:
        actuals[target] = []
        preds[target] = []
    
    # Extract pod IPs and predictions
    rows = []
    for idx, row in df.iterrows():
        pod_ip = row.get("selected_pod", f"10.1.0.{idx+1}")
        
        row_data = {
            "pod_ip": pod_ip,
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"]
        }
        
        # Get predictions for this pod
        if pod_ip in predictions.get("predictions", {}):
            pod_preds = predictions["predictions"][pod_ip]
            
            for target in targets:
                if target in pod_preds and target in row:
                    # Add to arrays for metric calculation
                    actuals[target].append(float(row[target]))
                    preds[target].append(float(pod_preds[target]))
                    
                    # Add to row data
                    row_data[f"actual_{target}"] = float(row[target])
                    row_data[f"predicted_{target}"] = float(pod_preds[target])
                    row_data[f"error_{target}"] = abs(float(row[target]) - float(pod_preds[target]))
                    
                    # Calculate percentage error if actual value is not zero
                    if float(row[target]) != 0:
                        pct_error = abs(float(row[target]) - float(pod_preds[target])) / float(row[target]) * 100
                        row_data[f"pct_error_{target}"] = pct_error
                    else:
                        row_data[f"pct_error_{target}"] = float('nan')
        
        rows.append(row_data)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(rows)
    
    # Calculate metrics for each target
    for target in targets:
        if len(actuals[target]) > 0:
            # Basic metrics
            metrics[f"{target}_mae"] = np.mean(np.abs(np.array(actuals[target]) - np.array(preds[target])))
            metrics[f"{target}_mse"] = np.mean((np.array(actuals[target]) - np.array(preds[target])) ** 2)
            metrics[f"{target}_rmse"] = np.sqrt(metrics[f"{target}_mse"])
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            non_zero_actuals = np.array([a for a in actuals[target] if a != 0])
            non_zero_preds = np.array([p for i, p in enumerate(preds[target]) if actuals[target][i] != 0])
            
            if len(non_zero_actuals) > 0:
                mape = np.mean(np.abs((non_zero_actuals - non_zero_preds) / non_zero_actuals)) * 100
                metrics[f"{target}_mape"] = mape
            
            # Calculate correlation coefficient
            if len(actuals[target]) > 1:
                corr = np.corrcoef(actuals[target], preds[target])[0, 1]
                metrics[f"{target}_correlation"] = corr
    
    # Calculate combined score (weighted average of TTFT and TPOT)
    if "ttft_mae" in metrics and "avg_tpot_mae" in metrics:
        ttft_weight = 0.4
        tpot_weight = 0.6
        metrics["combined_mae"] = ttft_weight * metrics["ttft_mae"] + tpot_weight * metrics["avg_tpot_mae"]
    
    return metrics, results_df

def print_metrics(metrics: Dict):
    """Print metrics in a nicely formatted table"""
    metric_rows = []
    
    # Group metrics by target
    targets = ["ttft", "avg_tpot", "combined"]
    metric_types = ["mae", "rmse", "mape", "correlation"]
    
    for target in targets:
        row = [target.upper()]
        for metric_type in metric_types:
            metric_key = f"{target}_{metric_type}"
            if metric_key in metrics:
                if metric_type == "mape" or metric_type == "correlation":
                    # Format as percentage or correlation coefficient
                    row.append(f"{metrics[metric_key]:.2f}")
                else:
                    # Format as milliseconds
                    row.append(f"{metrics[metric_key]:.2f} ms")
            else:
                row.append("N/A")
        
        metric_rows.append(row)
    
    # Print table
    headers = ["Target", "MAE", "RMSE", "MAPE (%)", "Correlation"]
    print("\nModel Performance Metrics:")
    print(tabulate(metric_rows, headers=headers, tablefmt="pretty"))

def create_visualizations(results_df: pd.DataFrame, output_path: Optional[str] = None):
    """Create visualizations of prediction results"""
    if results_df.empty:
        logger.warning("No results to visualize")
        return
    
    targets = ["ttft", "avg_tpot"]
    
    for target in targets:
        actual_col = f"actual_{target}"
        pred_col = f"predicted_{target}"
        
        if actual_col in results_df.columns and pred_col in results_df.columns:
            # Scatter plot of actual vs predicted
            plt.figure(figsize=(8, 6))
            plt.scatter(results_df[actual_col], results_df[pred_col], alpha=0.7)
            
            # Add identity line (perfect predictions)
            min_val = min(results_df[actual_col].min(), results_df[pred_col].min())
            max_val = max(results_df[actual_col].max(), results_df[pred_col].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.xlabel(f'Actual {target} (ms)')
            plt.ylabel(f'Predicted {target} (ms)')
            plt.title(f'Actual vs Predicted {target.upper()}')
            plt.grid(True, alpha=0.3)
            
            # Add correlation coefficient
            if len(results_df) > 1:
                corr = results_df[[actual_col, pred_col]].corr().iloc[0, 1]
                plt.annotate(f'Correlation: {corr:.2f}', 
                             xy=(0.05, 0.95), xycoords='axes fraction',
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # Save or show
            if output_path:
                plot_path = f"{output_path}_{target}_scatter.png"
                plt.savefig(plot_path)
                logger.info(f"Saved scatter plot to {plot_path}")
            else:
                plt.show()
            
            plt.close()
            
            # Histogram of errors
            error_col = f"error_{target}"
            if error_col in results_df.columns:
                plt.figure(figsize=(8, 6))
                plt.hist(results_df[error_col], bins=20, alpha=0.7)
                plt.xlabel(f'Absolute Error in {target} (ms)')
                plt.ylabel('Frequency')
                plt.title(f'Distribution of Prediction Errors for {target.upper()}')
                plt.grid(True, alpha=0.3)
                
                # Add mean error line
                mean_error = results_df[error_col].mean()
                plt.axvline(mean_error, color='r', linestyle='--')
                plt.annotate(f'Mean Error: {mean_error:.2f} ms', 
                             xy=(0.05, 0.95), xycoords='axes fraction',
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                
                # Save or show
                if output_path:
                    plot_path = f"{output_path}_{target}_error_hist.png"
                    plt.savefig(plot_path)
                    logger.info(f"Saved error histogram to {plot_path}")
                else:
                    plt.show()
                
                plt.close()

def main():
    args = parse_arguments()
    
    # Check service health
    logger.info(f"Checking health of service at {args.service_url}")
    if not check_health(args.service_url):
        logger.error("Health check failed. Exiting.")
        sys.exit(1)
    
    # Load test data
    logger.info(f"Loading test data from {args.test_file}")
    try:
        df = pd.read_csv(args.test_file)
        logger.info(f"Loaded {len(df)} rows from {args.test_file}")
        
        # Take subset if specified
        if args.num_samples > 0 and args.num_samples < len(df):
            df = df.head(args.num_samples)
            logger.info(f"Using {len(df)} samples for testing")
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        sys.exit(1)
    
    # Prepare test data
    pods = prepare_test_data(df)
    if not pods:
        logger.error("Failed to prepare test data. Exiting.")
        sys.exit(1)
    
    # Test prediction
    logger.info("Testing prediction endpoint")
    predictions = test_batch_prediction(args.service_url, pods)
    if not predictions:
        logger.error("Prediction test failed. Exiting.")
        sys.exit(1)
    
    # Calculate validation metrics
    logger.info("Calculating validation metrics")
    metrics, results_df = calculate_metrics(df, predictions)
    
    # Print metrics
    print_metrics(metrics)
    
    # Print sample of results
    print("\nSample of Prediction Results:")
    display_cols = ["pod_ip", "input_tokens", "output_tokens", 
                  "actual_ttft", "predicted_ttft", "error_ttft",
                  "actual_avg_tpot", "predicted_avg_tpot", "error_avg_tpot"]
    display_cols = [col for col in display_cols if col in results_df.columns]
    print(tabulate(results_df[display_cols].head(5).to_dict('records'), 
                 headers='keys', tablefmt="pretty", floatfmt=".2f"))
    
    # Create visualizations if requested
    if args.visualize:
        logger.info("Creating visualizations")
        create_visualizations(results_df, args.output)
    
    # Save results if output path specified
    if args.output:
        metrics_path = f"{args.output}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
        
        results_path = f"{args.output}_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved results to {results_path}")
    
    logger.info("Validation completed successfully!")

if __name__ == "__main__":
    main()