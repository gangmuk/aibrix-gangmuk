#!/usr/bin/env python3
"""
LLM Request Router - Data Analysis and Simple Baseline
----------------------------------------------------
Analyzes the preprocessed data to understand feature distributions
and creates simple baseline models for comparison.
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_dir):
    """Load processed data from the specified directory"""
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    # Load tensor datasets
    train_data = torch.load(os.path.join(train_dir, 'tensor_dataset.pt'))
    test_data = None
    if os.path.exists(test_dir):
        test_data = torch.load(os.path.join(test_dir, 'tensor_dataset.pt'))
    
    return train_data, test_data

def analyze_data(data):
    """Analyze the processed data to understand feature distributions"""
    logger.info("Analyzing data...")
    
    # Extract key components
    pod_features = data['pod_features_with_staleness'].numpy()
    kv_hit_ratios = data['kv_hit_ratios'].numpy()
    request_features = data['request_features'].numpy()
    actions = data['actions'].numpy()
    rewards = data['rewards'].numpy()
    
    # Basic statistics
    n_samples = len(actions)
    n_pods = pod_features.shape[1]
    pod_feature_dim = pod_features.shape[2]
    request_feature_dim = request_features.shape[1]
    
    logger.info(f"Number of samples: {n_samples}")
    logger.info(f"Number of pods: {n_pods}")
    logger.info(f"Pod feature dimension: {pod_feature_dim}")
    logger.info(f"Request feature dimension: {request_feature_dim}")
    
    # Action distribution
    action_counts = np.bincount(actions, minlength=n_pods)
    logger.info(f"Action distribution: {action_counts}")
    
    # Reward statistics
    logger.info(f"Reward statistics: min={rewards.min():.4f}, max={rewards.max():.4f}, mean={rewards.mean():.4f}, std={rewards.std():.4f}")
    
    # Feature statistics
    logger.info(f"Pod feature statistics: min={pod_features.min():.4f}, max={pod_features.max():.4f}, mean={pod_features.mean():.4f}, std={pod_features.std():.4f}")
    logger.info(f"KV hit ratio statistics: min={kv_hit_ratios.min():.4f}, max={kv_hit_ratios.max():.4f}, mean={kv_hit_ratios.mean():.4f}, std={kv_hit_ratios.std():.4f}")
    logger.info(f"Request feature statistics: min={request_features.min():.4f}, max={request_features.max():.4f}, mean={request_features.mean():.4f}, std={request_features.std():.4f}")
    
    # Check for NaN and Inf values
    logger.info(f"Pod features contain NaN: {np.isnan(pod_features).any()}")
    logger.info(f"Pod features contain Inf: {np.isinf(pod_features).any()}")
    logger.info(f"KV hit ratios contain NaN: {np.isnan(kv_hit_ratios).any()}")
    logger.info(f"KV hit ratios contain Inf: {np.isinf(kv_hit_ratios).any()}")
    logger.info(f"Request features contain NaN: {np.isnan(request_features).any()}")
    logger.info(f"Request features contain Inf: {np.isinf(request_features).any()}")
    
    # Visualize pod features
    plt.figure(figsize=(15, 10))
    
    # Plot pod feature distributions
    plt.subplot(2, 2, 1)
    for i in range(min(4, pod_feature_dim)):
        sns.kdeplot(pod_features[:, :, i].flatten(), label=f'Feature {i}')
    plt.title('Pod Feature Distributions')
    plt.legend()
    
    # Plot KV hit ratio distribution
    plt.subplot(2, 2, 2)
    sns.kdeplot(kv_hit_ratios.flatten())
    plt.title('KV Hit Ratio Distribution')
    
    # Plot reward distribution
    plt.subplot(2, 2, 3)
    sns.histplot(rewards, bins=30)
    plt.title('Reward Distribution')
    
    # Plot action distribution
    plt.subplot(2, 2, 4)
    plt.bar(range(n_pods), action_counts)
    plt.title('Action Distribution')
    plt.xticks(range(n_pods))
    
    plt.tight_layout()
    plt.savefig('data_analysis_basic.png')
    logger.info("Saved basic data analysis plot to data_analysis_basic.png")
    
    # Analyze feature-action relationships
    
    # Reshape pod features for easier analysis 
    # Combine pod features and kv hit ratio for the selected pod
    selected_pod_features = np.zeros((n_samples, pod_feature_dim))
    selected_pod_kv_hit = np.zeros(n_samples)
    
    for i in range(n_samples):
        selected_pod = actions[i]
        selected_pod_features[i] = pod_features[i, selected_pod]
        selected_pod_kv_hit[i] = kv_hit_ratios[i, selected_pod]
    
    # Perform PCA on request features (high-dimensional)
    if request_feature_dim > 2:
        logger.info("Performing PCA on request features...")
        pca = PCA(n_components=min(10, request_feature_dim))
        request_features_pca = pca.fit_transform(request_features)
        logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        
        # Visualize PCA components
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
        plt.title('PCA Explained Variance Ratio')
        plt.xlabel('Component')
        plt.ylabel('Explained Variance Ratio')
        plt.savefig('pca_explained_variance.png')
        logger.info("Saved PCA explained variance plot to pca_explained_variance.png")
    else:
        request_features_pca = request_features
    
    # Use t-SNE to visualize high-dimensional data in 2D
    logger.info("Performing t-SNE visualization...")
    # Combine all features
    combined_features = np.hstack([
        selected_pod_features,
        selected_pod_kv_hit.reshape(-1, 1),
        request_features_pca[:, :min(2, request_features_pca.shape[1])]  # Use first 2 PCA components
    ])
    
    # Sample data if too large
    max_samples = min(1000, n_samples)
    if n_samples > max_samples:
        indices = np.random.choice(n_samples, max_samples, replace=False)
        combined_features_sample = combined_features[indices]
        actions_sample = actions[indices]
    else:
        combined_features_sample = combined_features
        actions_sample = actions
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(combined_features_sample)
    
    # Plot t-SNE
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=actions_sample, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Selected Pod')
    plt.title('t-SNE Visualization of Features Colored by Selected Pod')
    plt.savefig('tsne_visualization.png')
    logger.info("Saved t-SNE visualization to tsne_visualization.png")
    
    # Analyze predictive power of KV hit ratio
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x=actions, y=selected_pod_kv_hit)
    plt.title('KV Hit Ratio by Selected Pod')
    plt.xlabel('Selected Pod')
    plt.ylabel('KV Hit Ratio')
    
    # Compute average KV hit ratio for each pod
    avg_kv_hit_by_pod = np.zeros((n_samples, n_pods))
    for i in range(n_samples):
        for j in range(n_pods):
            avg_kv_hit_by_pod[i, j] = kv_hit_ratios[i, j]
    
    avg_kv_hit_by_pod = avg_kv_hit_by_pod.mean(axis=0)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(n_pods), avg_kv_hit_by_pod)
    plt.title('Average KV Hit Ratio by Pod')
    plt.xlabel('Pod')
    plt.ylabel('Average KV Hit Ratio')
    plt.xticks(range(n_pods))
    
    plt.tight_layout()
    plt.savefig('kv_hit_analysis.png')
    logger.info("Saved KV hit ratio analysis to kv_hit_analysis.png")
    
    # Correlation between features and actions
    corr_features = np.zeros((pod_feature_dim + 1, n_pods))
    
    for pod in range(n_pods):
        # Create dummy variable for this pod
        pod_dummy = np.zeros(n_samples)
        pod_dummy[actions == pod] = 1
        
        # Compute correlation with pod features
        for i in range(pod_feature_dim):
            # Reshape pod features for this dimension
            feat = pod_features[:, pod, i].flatten()
            corr_features[i, pod] = np.corrcoef(feat, pod_dummy)[0, 1]
        
        # Compute correlation with KV hit ratio
        corr_features[pod_feature_dim, pod] = np.corrcoef(kv_hit_ratios[:, pod].flatten(), pod_dummy)[0, 1]
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 8))
    feature_names = [f'Pod Feature {i}' for i in range(pod_feature_dim)] + ['KV Hit Ratio']
    pod_names = [f'Pod {i}' for i in range(n_pods)]
    
    sns.heatmap(corr_features, annot=True, fmt='.2f', 
                xticklabels=pod_names, 
                yticklabels=feature_names,
                cmap='coolwarm')
    plt.title('Feature-Action Correlations')
    plt.tight_layout()
    plt.savefig('feature_action_correlations.png')
    logger.info("Saved feature-action correlations to feature_action_correlations.png")
    
    return {
        'pod_features': pod_features,
        'kv_hit_ratios': kv_hit_ratios,
        'request_features': request_features,
        'actions': actions,
        'rewards': rewards
    }

def train_simple_models(train_data, test_data=None):
    """Train simple baseline models for comparison"""
    logger.info("Training simple baseline models...")
    
    # Extract training data
    train_pod_features = train_data['pod_features_with_staleness'].numpy()
    train_kv_hit_ratios = train_data['kv_hit_ratios'].numpy()
    train_request_features = train_data['request_features'].numpy()
    train_actions = train_data['actions'].numpy()
    
    n_samples = len(train_actions)
    n_pods = train_pod_features.shape[1]
    pod_feature_dim = train_pod_features.shape[2]
    
    # Reshape features for traditional ML models
    # Strategy: Create features for each pod and combine with request features
    X_train = np.zeros((n_samples, n_pods * (pod_feature_dim + 1) + min(10, train_request_features.shape[1])))
    
    for i in range(n_samples):
        # Pod features and KV hit ratios
        for j in range(n_pods):
            X_train[i, j * (pod_feature_dim + 1):(j + 1) * (pod_feature_dim + 1) - 1] = train_pod_features[i, j]
            X_train[i, (j + 1) * (pod_feature_dim + 1) - 1] = train_kv_hit_ratios[i, j]
        
        # Add some request features (limit to avoid too high dimensionality)
        request_dim = min(10, train_request_features.shape[1])
        X_train[i, n_pods * (pod_feature_dim + 1):n_pods * (pod_feature_dim + 1) + request_dim] = train_request_features[i, :request_dim]
    
    # Check for NaN or infinity
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        logger.warning("Features contain NaN or Inf values, replacing with zeros")
        X_train = np.nan_to_num(X_train)
    
    y_train = train_actions
    
    # Train a Random Forest classifier
    logger.info("Training Random Forest classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Train a Logistic Regression classifier
    logger.info("Training Logistic Regression classifier...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    try:
        lr_model.fit(X_train, y_train)
    except Exception as e:
        logger.warning(f"Logistic Regression training failed: {e}")
        lr_model = None
    
    # Random baseline (randomly predict according to training distribution)
    logger.info("Creating random baseline...")
    action_probs = np.bincount(y_train) / len(y_train)
    
    # Feature importance from Random Forest
    logger.info("Computing feature importances...")
    feature_importance = rf_model.feature_importances_
    
    # Group feature importances by pod and by type
    pod_importances = np.zeros(n_pods)
    feature_type_importances = np.zeros(pod_feature_dim + 1 + 1)  # Pod features + KV hit ratio + request features
    
    for i in range(n_pods):
        # Sum importances for this pod
        start_idx = i * (pod_feature_dim + 1)
        end_idx = (i + 1) * (pod_feature_dim + 1)
        pod_importances[i] = feature_importance[start_idx:end_idx].sum()
        
        # Sum importances by feature type across all pods
        for j in range(pod_feature_dim):
            feature_type_importances[j] += feature_importance[start_idx + j]
        
        # KV hit ratio importance
        feature_type_importances[pod_feature_dim] += feature_importance[end_idx - 1]
    
    # Request features importance
    request_start_idx = n_pods * (pod_feature_dim + 1)
    feature_type_importances[-1] = feature_importance[request_start_idx:].sum()
    
    # Visualize feature importances
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.bar(range(n_pods), pod_importances)
    plt.title('Feature Importance by Pod')
    plt.xlabel('Pod')
    plt.ylabel('Importance')
    plt.xticks(range(n_pods))
    
    plt.subplot(2, 1, 2)
    feature_type_names = [f'Pod Feature {i}' for i in range(pod_feature_dim)] + ['KV Hit Ratio', 'Request Features']
    plt.bar(range(len(feature_type_importances)), feature_type_importances)
    plt.title('Feature Importance by Type')
    plt.xlabel('Feature Type')
    plt.ylabel('Importance')
    plt.xticks(range(len(feature_type_importances)), feature_type_names, rotation=45)
    
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    logger.info("Saved feature importances to feature_importances.png")
    
    # Evaluate on training data
    logger.info("Evaluating models on training data...")
    
    # Random baseline
    random_pred = np.random.choice(n_pods, size=len(y_train), p=action_probs)
    random_acc = accuracy_score(y_train, random_pred)
    logger.info(f"Random baseline training accuracy: {random_acc:.4f}")
    
    # Random Forest
    rf_pred = rf_model.predict(X_train)
    rf_acc = accuracy_score(y_train, rf_pred)
    logger.info(f"Random Forest training accuracy: {rf_acc:.4f}")
    
    # Logistic Regression
    if lr_model is not None:
        lr_pred = lr_model.predict(X_train)
        lr_acc = accuracy_score(y_train, lr_pred)
        logger.info(f"Logistic Regression training accuracy: {lr_acc:.4f}")
    else:
        lr_acc = None
    
    # Evaluate on test data if available
    test_results = None
    if test_data is not None:
        logger.info("Evaluating models on test data...")
        
        # Extract test data
        test_pod_features = test_data['pod_features_with_staleness'].numpy()
        test_kv_hit_ratios = test_data['kv_hit_ratios'].numpy()
        test_request_features = test_data['request_features'].numpy()
        test_actions = test_data['actions'].numpy()
        
        n_test_samples = len(test_actions)
        
        # Reshape features
        X_test = np.zeros((n_test_samples, n_pods * (pod_feature_dim + 1) + min(10, test_request_features.shape[1])))
        
        for i in range(n_test_samples):
            # Pod features and KV hit ratios
            for j in range(n_pods):
                X_test[i, j * (pod_feature_dim + 1):(j + 1) * (pod_feature_dim + 1) - 1] = test_pod_features[i, j]
                X_test[i, (j + 1) * (pod_feature_dim + 1) - 1] = test_kv_hit_ratios[i, j]
            
            # Add some request features
            request_dim = min(10, test_request_features.shape[1])
            X_test[i, n_pods * (pod_feature_dim + 1):n_pods * (pod_feature_dim + 1) + request_dim] = test_request_features[i, :request_dim]
        
        # Check for NaN or infinity
        if np.isnan(X_test).any() or np.isinf(X_test).any():
            logger.warning("Test features contain NaN or Inf values, replacing with zeros")
            X_test = np.nan_to_num(X_test)
        
        y_test = test_actions
        
        # Random baseline
        random_test_pred = np.random.choice(n_pods, size=len(y_test), p=action_probs)
        random_test_acc = accuracy_score(y_test, random_test_pred)
        logger.info(f"Random baseline test accuracy: {random_test_acc:.4f}")
        
        # Random Forest
        rf_test_pred = rf_model.predict(X_test)
        rf_test_acc = accuracy_score(y_test, rf_test_pred)
        logger.info(f"Random Forest test accuracy: {rf_test_acc:.4f}")
        
        # Detailed evaluation for Random Forest
        rf_report = classification_report(y_test, rf_test_pred)
        rf_cm = confusion_matrix(y_test, rf_test_pred)
        logger.info(f"Random Forest test classification report:\n{rf_report}")
        
        # Visualize confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Random Forest Confusion Matrix (Test)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('rf_confusion_matrix.png')
        logger.info("Saved Random Forest confusion matrix to rf_confusion_matrix.png")
        
        # Logistic Regression
        if lr_model is not None:
            lr_test_pred = lr_model.predict(X_test)
            lr_test_acc = accuracy_score(y_test, lr_test_pred)
            logger.info(f"Logistic Regression test accuracy: {lr_test_acc:.4f}")
        else:
            lr_test_acc = None
        
        test_results = {
            'random_acc': random_test_acc,
            'rf_acc': rf_test_acc,
            'lr_acc': lr_test_acc,
            'rf_report': rf_report,
            'rf_cm': rf_cm
        }
    
    return {
        'training_results': {
            'random_acc': random_acc,
            'rf_acc': rf_acc,
            'lr_acc': lr_acc
        },
        'test_results': test_results,
        'feature_importances': {
            'pod_importances': pod_importances,
            'feature_type_importances': feature_type_importances,
            'feature_type_names': feature_type_names
        },
        'models': {
            'rf_model': rf_model,
            'lr_model': lr_model
        }
    }

def main():
    """Main function for analyzing data and training simple models"""
    parser = argparse.ArgumentParser(description='Analyze LLM routing data and train simple baseline models')
    parser.add_argument('data_dir', type=str, help='Directory containing processed data')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data_dir}...")
    train_data, test_data = load_data(args.data_dir)
    
    # Analyze data
    logger.info("Analyzing data...")
    analysis_results = analyze_data(train_data)
    
    # Train simple models
    logger.info("Training simple models...")
    model_results = train_simple_models(train_data, test_data)
    
    logger.info("Analysis and training complete!")

if __name__ == "__main__":
    main()