import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import logging
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train and test LLM latency prediction model')
    
    parser.add_argument('input_file', type=str,
                        help='Path to the CSV file with pod-specific features')
    parser.add_argument('--output_dir', type=str, default='model_output',
                        help='Directory to save model and test results')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for testing (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of trees in XGBoost model (default: 100)')
    parser.add_argument('--max_depth', type=int, default=5,
                        help='Maximum depth of trees in XGBoost model (default: 5)')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for XGBoost model (default: 0.1)')
    parser.add_argument('--cv', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--no_plots', action='store_true',
                        help='Do not generate plots')
    
    return parser.parse_args()

def preprocess_and_extract_features(input_file, target_performance_metrics):
    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return None
    
    # Load data
    logger.info(f"Loading data from {input_file}")
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None
    
    # Check for required columns
    if 'selected_pod' not in df.columns:
        logger.error("Required column 'selected_pod' not found in input data")
        return None
    
    # Create new DataFrame for extracted features
    records = []
    
    # Process each row
    for _, row in df.iterrows():
        # Get the selected pod for this request
        selected_pod = row['selected_pod']
        pod_prefix = f"pod_{selected_pod.replace('.', '_')}"
        
        # Extract pod-specific features
        pod_features = {}
        for col in df.columns:
            if col.startswith(pod_prefix):
                # Get the feature name without the pod prefix
                feature_name = col.replace(f"{pod_prefix}_", "")
                pod_features[feature_name] = row[col]
        
        # print(f"Pod features: {pod_features}")

        # Add request-specific features
        request_features = {
            'input_tokens': row['input_tokens'],
            'output_tokens': row['output_tokens'],
            'total_tokens': row['total_tokens'],
            'selected_pod': selected_pod
        }
        
        # # Add system-wide metrics that could affect performance
        # system_features = {
        #     'cluster_total_gpu_kv_cache': row['cluster_total_gpu_kv_cache'],
        #     'cluster_total_cpu_kv_cache': row['cluster_total_cpu_kv_cache'],
        #     'cluster_total_kv_hit_ratio': row['cluster_total_kv_hit_ratio'],
        #     'cluster_total_prefill_tokens': row['cluster_total_prefill_tokens'],
        #     'cluster_total_decode_tokens': row['cluster_total_decode_tokens'],
        # }
        system_features = {}
        
        # Add target variables if they exist
        target_vars = {}
        for target in target_performance_metrics:
            if target in row:
                target_vars[target] = row[target]
        
        # Combine all features
        records.append({**pod_features, **request_features, **system_features, **target_vars})
        # records.append({**pod_features, **request_features, **target_vars})
    
    # Convert to DataFrame
    trainig_df = pd.DataFrame(records)
    logger.info(f"Extracted features with shape: {trainig_df.shape}")
    
    # Handle missing values
    numeric_cols = trainig_df.select_dtypes(include=['number']).columns
    trainig_df[numeric_cols] = trainig_df[numeric_cols].fillna(0)
    
    categorical_cols = trainig_df.select_dtypes(include=['object']).columns
    trainig_df[categorical_cols] = trainig_df[categorical_cols].fillna('unknown')
    
    trainig_df.drop(columns=['selected_pod'], inplace=True, errors='ignore')
    
    return trainig_df

def train_and_evaluate(df, output_dir, target_performance_metrics, test_size=0.2, random_state=42, 
                       n_estimators=100, max_depth=5, learning_rate=0.1, cv=5, no_plots=False):
    """
    Train XGBoost models for latency prediction and evaluate their performance.
    
    Parameters:
    -----------
    training_file : str
        Path to the input CSV file with preprocessed features
    output_dir : str
        Path to save the model and test results
    test_size : float
        Fraction of data to use for testing
    random_state : int
        Random state for reproducibility
    n_estimators : int
        Number of trees in XGBoost model
    max_depth : int
        Maximum depth of trees in XGBoost model
    learning_rate : float
        Learning rate for XGBoost model
    cv : int
        Number of cross-validation folds
    no_plots : bool
        If True, do not generate plots
    
    Returns:
    --------
    dict
        Dictionary with trained models, evaluation metrics, and test results
    """
    
    # Check for target variables
    available_targets = [var for var in target_performance_metrics if var in df.columns]
    
    if not available_targets:
        logger.error("No target variables found in input data")
        return None
    
    logger.info(f"Found target variables: {available_targets}")
    
    # Split data into features and targets
    X = df.drop(available_targets, axis=1, errors='ignore')
    y = df[available_targets]
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    logger.info(f"Found {len(numeric_cols)} numeric features and {len(categorical_cols)} categorical features")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # train_file = 'train_dataset.csv'
    # X_train.to_csv(train_file, index=False)
    # logger.info(f"Saved training dataset to {train_file})")

    test_df = X_test.copy()
    for target in available_targets:
        test_df[target] = y_test[target]
    test_df['set'] = 'test'  # Mark as test set
    test_file = 'test_dataset.csv'
    test_df.to_csv(test_file, index=False)
    logger.info(f"Saved test dataset to {test_file}")
    
    logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown'))
            ]), categorical_cols)
        ]
    )
    
    # Results dictionary
    result = {
        'models': {},
        'metrics': {},
        'feature_importances': {},
        'test_results': {
            'metrics': {},
            'predictions': {}
        },
        'cv_results': {},
        'model_params': {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate
        }
    }
    
    # Train models for each target
    for target in available_targets:
        logger.info(f"\n=== Training and evaluating model for {target} ===")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state
            ))
        ])
        
        # Cross-validation
        if cv > 1:
            logger.debug(f"Performing {cv}-fold cross-validation")
            cv_scores = cross_val_score(
                pipeline, X_train, y_train[target],
                cv=cv, 
                scoring='neg_root_mean_squared_error',
                n_jobs=-1
            )
            
            # Convert negative RMSE to positive
            cv_scores = -cv_scores
            
            result['cv_results'][target] = {
                'rmse_mean': cv_scores.mean(),
                'rmse_std': cv_scores.std(),
                'fold_scores': cv_scores.tolist()
            }
            
            logger.debug(f"Cross-validation RMSE: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Train model on full training set
        logger.debug(f"Training final model on full training set")
        pipeline.fit(X_train, y_train[target])
        
        # Store model
        result['models'][target] = pipeline
        
        # Extract feature importances
        if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
            # Get feature names (this is a simplification)
            feature_names = numeric_cols + categorical_cols
            
            # Store feature importances
            importances = dict(zip(feature_names, pipeline.named_steps['model'].feature_importances_))
            result['feature_importances'][target] = importances
            
            # Print top features
            sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
            logger.debug(f"Top 10 features for {target}:")
            for feature, importance in sorted_importances:
                logger.debug(f"  {feature}: {importance:.4f}")
        
        # Evaluate on test set
        logger.debug(f"Evaluating on test set")
        y_pred = pipeline.predict(X_test)
        
        # Store predictions
        result['test_results']['predictions'][target] = {
            'true': y_test[target].tolist(),
            'pred': y_pred.tolist()
        }
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test[target], y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test[target], y_pred)),
            'mae': mean_absolute_error(y_test[target], y_pred),
            'r2': r2_score(y_test[target], y_pred)
        }
        
        # Calculate MAPE only if no zero values in y_test
        if not (y_test[target] == 0).any():
            metrics['mape'] = mean_absolute_percentage_error(y_test[target], y_pred) * 100
        
        result['test_results']['metrics'][target] = metrics
        
        # Print metrics
        logger.debug(f"Test metrics for {target}:")
        for metric_name, metric_value in metrics.items():
            logger.debug(f"  {metric_name}: {metric_value:.4f}")
        
        # Generate plots
        if not no_plots:
            # Actual vs Predicted plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test[target], y_pred, alpha=0.5)
            plt.plot([y_test[target].min(), y_test[target].max()], 
                    [y_test[target].min(), y_test[target].max()], 'r--')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'{target} - Actual vs Predicted (R² = {metrics["r2"]:.4f})')
            plt.grid(True, alpha=0.3)
            fn = f'{output_dir}/{target}_predictions.pdf'
            plt.savefig(fn)
            print(f"** Saved {target} prediction plot: {fn}")
            plt.close()
            
            # Residuals plot
            residuals = y_test[target] - y_pred
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted')
            plt.ylabel('Residuals')
            plt.title(f'{target} - Residuals Plot (RMSE = {metrics["rmse"]:.4f})')
            plt.grid(True, alpha=0.3)
            fn = f'{output_dir}/{target}_residuals.pdf'
            plt.savefig(fn)
            print(f"** Saved {target} residuals plot: {fn}")
            plt.close()
            
            # Feature importance plot
            if target in result['feature_importances']:
                importance = result['feature_importances'][target]
                if importance:
                    # Sort by importance
                    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20])
                    
                    plt.figure(figsize=(12, 8))
                    sns.barplot(x=list(importance.values()), y=list(importance.keys()))
                    plt.title(f'Top 20 Features for {target}')
                    plt.xlabel('Importance')
                    plt.tight_layout()
                    fn = f'{output_dir}/{target}_feature_importance.pdf'
                    plt.savefig(fn)
                    print(f"** Saved {target} feature importance plot: {fn}")
                    plt.close()
    
    # Save model
    model_file = os.path.join(output_dir, f'latency_predictor.joblib')
    joblib.dump(result, model_file)
    logger.info(f"Saved model to {model_file}")
    
    # Create summary report
    with open(os.path.join(output_dir, f'summary.txt'), 'w') as f:
        f.write("LLM Latency Prediction Model Summary\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Target variables: {', '.join(available_targets)}\n\n")
        
        f.write("Model Parameters:\n")
        f.write("-" * 40 + "\n")
        f.write(f"n_estimators: {n_estimators}\n")
        f.write(f"max_depth: {max_depth}\n")
        f.write(f"learning_rate: {learning_rate}\n\n")
        
        if cv > 1:
            f.write("Cross-Validation Results:\n")
            f.write("-" * 40 + "\n")
            for target in available_targets:
                if target in result['cv_results']:
                    f.write(f"\n{target}:\n")
                    cv_results = result['cv_results'][target]
                    f.write(f"  RMSE: {cv_results['rmse_mean']:.4f} ± {cv_results['rmse_std']:.4f}\n")
                    
                    # Stability assessment
                    cv_ratio = cv_results['rmse_std'] / cv_results['rmse_mean'] if cv_results['rmse_mean'] > 0 else float('inf')
                    if cv_ratio <= 0.1:
                        stability = "Very stable"
                    elif cv_ratio <= 0.2:
                        stability = "Stable"
                    elif cv_ratio <= 0.3:
                        stability = "Moderately stable"
                    else:
                        stability = "Unstable"
                    
                    f.write(f"  Stability: {stability} (CV variation ratio: {cv_ratio:.2f})\n")
        
        f.write("\nTest Results:\n")
        f.write("-" * 40 + "\n")
        
        for target in available_targets:
            if target in result['test_results']['metrics']:
                f.write(f"\n{target}:\n")
                metrics = result['test_results']['metrics'][target]
                
                for metric_name, metric_value in metrics.items():
                    f.write(f"  {metric_name}: {metric_value:.4f}\n")
                
                # Model quality assessment
                r2 = metrics.get('r2', 0)
                if r2 >= 0.9:
                    assessment = "Excellent - Model explains >90% of variance"
                elif r2 >= 0.8:
                    assessment = "Good - Model explains 80-90% of variance"
                elif r2 >= 0.7:
                    assessment = "Acceptable - Model explains 70-80% of variance"
                elif r2 >= 0.6:
                    assessment = "Marginal - Model explains 60-70% of variance"
                elif r2 >= 0.5:
                    assessment = "Poor - Model explains 50-60% of variance"
                else:
                    assessment = "Unacceptable - Model explains <50% of variance"
                
                f.write(f"  Quality: {assessment}\n")
                
                # Check for potential overfitting
                if cv > 1 and target in result['cv_results']:
                    test_rmse = metrics['rmse']
                    cv_rmse = result['cv_results'][target]['rmse_mean']
                    
                    # Calculate difference ratio
                    diff_ratio = (cv_rmse - test_rmse) / cv_rmse if cv_rmse > 0 else float('inf')
                    
                    if diff_ratio > 0.2:
                        f.write("  Warning: Potential overfitting detected\n")
                    elif diff_ratio < -0.2:
                        f.write("  Warning: Test set may be easier than training data\n")
                    else:
                        f.write("  No signs of overfitting detected\n")
        
        f.write("\nTop Features:\n")
        f.write("-" * 40 + "\n")
        
        for target in available_targets:
            if target in result['feature_importances']:
                f.write(f"\n{target}:\n")
                
                # Get top 5 features
                importance = result['feature_importances'][target]
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                
                for feature, importance in top_features:
                    f.write(f"  {feature}: {importance:.4f}\n")
    
    logger.info(f"Summary report saved to {os.path.join(output_dir, f'summary.txt')}")
    
    return result

def main():
    args = parse_args()
    target_performance_metrics = ['avg_tpot', 'ttft']
    training_df = preprocess_and_extract_features(args.input_file, target_performance_metrics)
    input_dir = '/'.join(args.input_file.split('/')[:-1])
    print(f"Input dir: {input_dir}")
    training_file = f"{input_dir}/pod_specific_data.csv"
    training_df.to_csv(training_file, index=False)
    logger.info(f"Saved extracted features to {training_file}")
    if training_df is None:
        logger.error("Feature extraction failed. Exiting.")
        sys.exit(1)
    output_dir = f"{input_dir}/{args.output_dir}"
    os.makedirs(output_dir, exist_ok=True)
    result = train_and_evaluate(
        df=training_df,
        output_dir=output_dir,
        target_performance_metrics=target_performance_metrics,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        cv=args.cv,
        no_plots=args.no_plots
    )
    
    if result is None:
        sys.exit(1)
    
    print(f"** Target performance metrics: {target_performance_metrics} **")
    logger.info("Training and evaluation completed successfully!")

if __name__ == "__main__":
    main()