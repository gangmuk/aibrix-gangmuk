#!/usr/bin/env python3
"""
LLM Request Router - Behavioral Cloning Training
-----------------------------------------------
Trains a supervised learning model to imitate routing decisions from
historical data, serving as a baseline and potential pre-training
for reinforcement learning.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle
import time
import logging
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
from logger import logger

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


class ParallelFeatureProcessor(nn.Module):
    """
    Processes pod features and KV hit ratios in parallel pathways,
    then combines them rather than using cross-attention.
    """
    def __init__(self, pod_feature_dim, kv_hit_dim, hidden_dim):
        super().__init__()
        
        # Pod features pathway
        self.pod_encoder = nn.Sequential(
            nn.Linear(pod_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # KV hit ratio pathway (separate)
        self.kv_encoder = nn.Sequential(
            nn.Linear(kv_hit_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # Final combination
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, pod_features, kv_hit_ratios):
        # Process pod features
        pod_encoding = self.pod_encoder(pod_features)
        
        # Process KV hit ratios separately
        kv_encoding = self.kv_encoder(kv_hit_ratios)
        
        # Combine through concatenation
        combined = torch.cat([pod_encoding, kv_encoding], dim=-1)
        output = self.combiner(combined)
        
        return output


class LLMRoutingModel(nn.Module):
    """
    Neural network for LLM request routing using behavioral cloning.
    Simpler architecture focused on supervised learning.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        
        # Define dimensions from state_dim dict
        pod_feature_dim = state_dim['pod_features']
        kv_hit_dim = state_dim['kv_hit_ratios']
        request_dim = state_dim['request_features']
        
        # Request encoder with dropout for regularization
        self.request_encoder = nn.Sequential(
            nn.Linear(request_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Feature processors - parallel pathways for pods and KV hit ratios
        self.feature_processor = ParallelFeatureProcessor(
            pod_feature_dim, kv_hit_dim, hidden_dim
        )
        
        # Output layer for routing decisions
        self.routing_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, pod_features, kv_hit_ratios, request_features):
        batch_size, num_pods, _ = pod_features.shape
        
        # 1. Encode request features
        request_encoding = self.request_encoder(request_features)
        request_encoding = request_encoding.unsqueeze(1).expand(-1, num_pods, -1)
        
        # 2. Process each pod's features with KV hit ratio
        pod_encodings = self.feature_processor(pod_features, kv_hit_ratios)
        
        # 3. Add request context through addition
        contextualized_pods = pod_encodings + request_encoding
        
        # 4. Final routing logits for each pod
        routing_logits = self.routing_head(contextualized_pods).squeeze(-1)
        
        return routing_logits


def load_data(data_dir):
    """Load processed data from the specified directory"""
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    # Check if directories exist
    if not os.path.exists(train_dir):
        raise ValueError(f"Training data directory not found: {train_dir}")
    
    # Load tensor datasets
    train_data = torch.load(os.path.join(train_dir, 'tensor_dataset.pt'))
    
    # Load test data if available
    test_data = None
    if os.path.exists(test_dir) and os.path.exists(os.path.join(test_dir, 'tensor_dataset.pt')):
        test_data = torch.load(os.path.join(test_dir, 'tensor_dataset.pt'))
    
    # Load metadata
    if os.path.exists(os.path.join(train_dir, 'metadata.json')):
        with open(os.path.join(train_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
    else:
        metadata = None
        
    return train_data, test_data, metadata


def train_behavioral_cloning(train_data, test_data=None, config=None):
    """Train a behavioral cloning model on the given data"""
    # Default configuration
    if config is None:
        config = {
            'hidden_dim': 256,
            'batch_size': 64,
            'learning_rate': 3e-4,
            'weight_decay': 1e-4,
            'num_epochs': 50,
            'dropout': 0.2,
            'eval_interval': 5,
        }
    
    # Create output directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"bc_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Determine state dimensions
    state_dim = {
        'pod_features': train_data['pod_features_with_staleness'].shape[2],
        'kv_hit_ratios': train_data['kv_hit_ratios'].shape[2],
        'request_features': train_data['request_features'].shape[1],
        'num_pods': train_data['pod_features'].shape[1]
    }
    
    # Determine action dimension (number of pods)
    action_dim = train_data['pod_features'].shape[1]
    
    logger.info(f"State dimensions: {state_dim}")
    logger.info(f"Action dimension: {action_dim}")
    
    # Create model
    model = LLMRoutingModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    ).to(device)
    
    # Create optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(
        train_data['pod_features_with_staleness'],
        train_data['kv_hit_ratios'],
        train_data['request_features'],
        train_data['actions']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Create validation dataloader if test data is available
    val_loader = None
    if test_data is not None:
        val_dataset = TensorDataset(
            test_data['pod_features_with_staleness'],
            test_data['kv_hit_ratios'],
            test_data['request_features'],
            test_data['actions']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available()
        )
    
    # Training loop
    logger.info("Starting behavioral cloning training...")
    best_accuracy = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(config['num_epochs']):
        # Training
        model.train()
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move data to device
            pod_features, kv_hit_ratios, request_features, actions = [b.to(device) for b in batch]
            
            # Forward pass
            logits = model(pod_features, kv_hit_ratios, request_features)
            
            # Compute loss
            loss = F.cross_entropy(logits, actions)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if num_batches % 100 == 0:
                logger.info(f"Epoch {epoch+1}/{config['num_epochs']}, "
                          f"Batch {num_batches}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}")
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        # Log epoch statistics
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']} completed in {epoch_time:.2f}s, "
                  f"Average Loss: {avg_loss:.4f}")
        
        # Evaluation
        if val_loader is not None and ((epoch + 1) % config['eval_interval'] == 0 or epoch == config['num_epochs'] - 1):
            model.eval()
            correct = 0
            total = 0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    # Move data to device
                    pod_features, kv_hit_ratios, request_features, actions = [b.to(device) for b in batch]
                    
                    # Forward pass
                    logits = model(pod_features, kv_hit_ratios, request_features)
                    
                    # Get predictions
                    _, predicted = torch.max(logits, 1)
                    
                    # Update statistics
                    total += actions.size(0)
                    correct += (predicted == actions).sum().item()
                    
                    # Save predictions and targets for later analysis
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(actions.cpu().numpy())
            
            # Calculate accuracy
            accuracy = correct / total
            val_accuracies.append(accuracy)
            
            # Update learning rate scheduler
            scheduler.step(1.0 - accuracy)  # Use (1 - accuracy) as the metric to minimize
            
            # Log evaluation results
            logger.info(f"Validation Accuracy: {accuracy:.4f}")
            
            # Generate confusion matrix and classification report
            cm = confusion_matrix(all_targets, all_preds)
            report = classification_report(all_targets, all_preds)
            
            # Log detailed evaluation results
            logger.info(f"Confusion Matrix:\n{cm}")
            logger.info(f"Classification Report:\n{report}")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                logger.info(f"New best model saved with accuracy: {accuracy:.4f}")
            
            # Additional diagnostics
            action_distribution = np.bincount(all_preds, minlength=action_dim)
            logger.info(f"Predicted Action Distribution: {action_distribution}")
            
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    
    # Plot and save training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    if val_loader is not None:
        plt.subplot(1, 2, 2)
        # Convert evaluation interval to actual epochs
        eval_epochs = [(i + 1) * config['eval_interval'] - 1 for i in range(len(val_accuracies))]
        if len(eval_epochs) > len(val_accuracies):
            eval_epochs = eval_epochs[:len(val_accuracies)]
        plt.plot(eval_epochs, val_accuracies)
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_accuracy': val_accuracies
    }
    
    with open(os.path.join(output_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    logger.info(f"Training completed, models and history saved to {output_dir}")
    
    return model, output_dir


def evaluate_model(model, test_data, output_dir=None):
    """Evaluate the model's performance on test data"""
    # Create dataset and dataloader
    test_dataset = TensorDataset(
        test_data['pod_features_with_staleness'],
        test_data['kv_hit_ratios'],
        test_data['request_features'],
        test_data['actions']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize counters
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_logits = []
    
    # Evaluate model
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            pod_features, kv_hit_ratios, request_features, actions = [b.to(device) for b in batch]
            
            # Forward pass
            logits = model(pod_features, kv_hit_ratios, request_features)
            
            # Get predictions
            _, predicted = torch.max(logits, 1)
            
            # Update statistics
            total += actions.size(0)
            correct += (predicted == actions).sum().item()
            
            # Save predictions, targets, and logits for later analysis
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(actions.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
    
    # Calculate accuracy
    accuracy = correct / total
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    
    # Generate confusion matrix and classification report
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds)
    
    # Log detailed evaluation results
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{report}")
    
    # Additional diagnostics
    action_dim = model.routing_head[-1].out_features if hasattr(model, 'routing_head') else len(np.unique(all_targets))
    action_distribution = np.bincount(all_preds, minlength=action_dim)
    true_distribution = np.bincount(all_targets, minlength=action_dim)
    
    logger.info(f"True Action Distribution: {true_distribution}")
    logger.info(f"Predicted Action Distribution: {action_distribution}")
    
    # Calculate average logits for each action class
    all_logits = np.vstack(all_logits)
    avg_logits_per_class = []
    for i in range(action_dim):
        class_indices = np.where(np.array(all_targets) == i)[0]
        if len(class_indices) > 0:
            avg_logits = np.mean(all_logits[class_indices], axis=0)
            avg_logits_per_class.append(avg_logits)
    
    # Save evaluation results if output directory is provided
    if output_dir is not None:
        # Save metrics
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'predicted_distribution': action_distribution.tolist(),
            'true_distribution': true_distribution.tolist()
        }
        
        with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add labels
        classes = [f"Pod {i}" for i in range(action_dim)]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        
        # Plot action distributions
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.bar(range(action_dim), true_distribution)
        plt.title('True Action Distribution')
        plt.xlabel('Action')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        plt.bar(range(action_dim), action_distribution)
        plt.title('Predicted Action Distribution')
        plt.xlabel('Action')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'action_distributions.png'))
        
        # Plot average logits for each class
        if len(avg_logits_per_class) > 0:
            plt.figure(figsize=(12, 6 * len(avg_logits_per_class)))
            for i, avg_logits in enumerate(avg_logits_per_class):
                plt.subplot(len(avg_logits_per_class), 1, i+1)
                plt.bar(range(action_dim), avg_logits)
                plt.title(f'Average Logits for Class {i}')
                plt.xlabel('Action')
                plt.ylabel('Logit Value')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'average_logits.png'))
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predicted_distribution': action_distribution,
        'true_distribution': true_distribution
    }


def main():
    """Main function to train the LLM routing model"""
    parser = argparse.ArgumentParser(description='Train LLM routing model with behavioral cloning')
    parser.add_argument('data_dir', type=str, help='Directory containing processed data (with train and test subdirectories)')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension of neural networks')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for regularization')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--eval-interval', type=int, default=5, help='Evaluation interval (epochs)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data
    logger.info(f"Loading data from {args.data_dir}...")
    train_data, test_data, metadata = load_data(args.data_dir)
    
    # Print data statistics
    logger.info(f"Loaded {len(train_data['rewards'])} training samples")
    if test_data is not None:
        logger.info(f"Loaded {len(test_data['rewards'])} test samples")
    
    # Create configuration
    config = {
        'hidden_dim': args.hidden_dim,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'num_epochs': args.epochs,
        'dropout': args.dropout,
        'eval_interval': args.eval_interval,
        'seed': args.seed,
        'output_dir': args.output_dir
    }
    
    # Train model
    logger.info("Training model...")
    model, output_dir = train_behavioral_cloning(train_data, test_data, config)
    
    # Evaluate model
    if test_data is not None:
        logger.info("Evaluating model on test data...")
        evaluate_model(model, test_data, output_dir)
    
    logger.info(f"Training completed, results saved to {output_dir}")


if __name__ == "__main__":
    main()