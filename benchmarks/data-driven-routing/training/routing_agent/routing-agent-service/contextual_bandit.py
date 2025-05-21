#!/usr/bin/env python3

# contextual_bandit.py

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.distributions import Categorical
import pickle
import time
import matplotlib.pyplot as plt
from datetime import datetime
import glob
from logger import logger
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
training_results_dir = "training_results"
final_model_path = "final_model"

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


class PolicyNetwork(nn.Module):
    """
    Policy Network for Contextual Bandit, outputs action probabilities
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Define dimensions from state_dim dict
        pod_feature_dim = state_dim['pod_features']
        kv_hit_dim = state_dim['kv_hit_ratios']
        request_dim = state_dim['request_features']
        
        # Request encoder
        self.request_encoder = nn.Sequential(
            nn.Linear(request_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Feature processors - parallel pathways for pods and KV hit ratios
        self.feature_processor = ParallelFeatureProcessor(
            pod_feature_dim, kv_hit_dim, hidden_dim
        )
        
        # Attention mechanism across pods 
        self.pod_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Output layer for action probabilities
        self.policy_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, pod_features, kv_hit_ratios, request_features, return_attention=False):
        batch_size, num_pods, _ = pod_features.shape
        
        # 1. Encode request features
        request_encoding = self.request_encoder(request_features)
        request_encoding = request_encoding.unsqueeze(1).expand(-1, num_pods, -1)
        
        # 2. Process each pod's features with KV hit ratio
        pod_encodings = self.feature_processor(pod_features, kv_hit_ratios)
        
        # 3. Add request context through addition
        contextualized_pods = pod_encodings + request_encoding
        
        # 4. Apply attention across pods to capture inter-pod relationships
        attn_output, attn_weights = self.pod_attention(
            query=contextualized_pods,
            key=contextualized_pods,
            value=contextualized_pods,
            need_weights=return_attention
        )
        
        # 5. Final policy logits
        policy_logits = self.policy_head(attn_output).squeeze(-1)
        
        # Get action probabilities
        action_probs = F.softmax(policy_logits, dim=1)
        
        if return_attention:
            return action_probs, attn_weights
        
        return action_probs
    
    def get_action(self, pod_features, kv_hit_ratios, request_features, explore=True, epsilon=0.1):
        action_probs = self.forward(pod_features, kv_hit_ratios, request_features)
        
        if not explore:
            # Purely exploit - select the pod with highest probability
            return torch.argmax(action_probs, dim=1)
        
        # Epsilon-greedy exploration
        batch_size = pod_features.shape[0]
        random_actions = torch.randint(0, action_probs.shape[1], (batch_size,), device=device)
        greedy_actions = torch.argmax(action_probs, dim=1)
        
        # Random mask for exploration
        explore_mask = (torch.rand(batch_size, device=device) < epsilon).long()
        
        # Choose either exploration or exploitation based on epsilon
        actions = (1 - explore_mask) * greedy_actions + explore_mask * random_actions
        
        # Calculate log probabilities for the chosen actions
        log_probs = torch.log(torch.gather(action_probs, 1, actions.unsqueeze(1)).squeeze(1) + 1e-10)
        
        return actions, log_probs


class ContextualBandit:
    """
    Contextual Bandit for LLM request routing
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, 
                 batch_size=64, exploration_rate=0.1):
        self.batch_size = batch_size
        self.exploration_rate = exploration_rate
        
        # Initialize policy network
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Initialize memory attributes
        self.pod_features = []
        self.kv_hit_ratios = []
        self.request_features = []
        self.actions = []
        self.rewards = []
        
        # Metrics for tracking
        self.loss_history = []
        self.reward_history = []
        self.entropy_history = []
        
    def remember(self, pod_features, kv_hit_ratios, request_features, action, reward):
        """Store context-action-reward tuple in memory"""
        self.pod_features.append(pod_features)
        self.kv_hit_ratios.append(kv_hit_ratios)
        self.request_features.append(request_features)
        self.actions.append(action)
        self.rewards.append(reward)
        
    def choose_action(self, pod_features, kv_hit_ratios, request_features, evaluate=False):
        """Select an action (pod) for the given context"""
        with torch.no_grad():
            if evaluate:
                # Pure exploitation during evaluation
                action_probs = self.policy(pod_features, kv_hit_ratios, request_features)
                action = torch.argmax(action_probs, dim=1)
                return action
            else:
                # Exploration-exploitation during training
                action, log_prob = self.policy.get_action(
                    pod_features, kv_hit_ratios, request_features, 
                    explore=True, epsilon=self.exploration_rate
                )
                return action, log_prob
    
    def learn(self):
        """Update policy using rewards"""
        if len(self.pod_features) == 0:
            return {
                'loss': 0.0,
                'reward': 0.0,
                'entropy': 0.0
            }
        
        # Stack all tensors
        pod_features = torch.cat(self.pod_features, dim=0)
        kv_hit_ratios = torch.cat(self.kv_hit_ratios, dim=0)
        request_features = torch.cat(self.request_features, dim=0)
        actions = torch.cat(self.actions, dim=0)
        rewards = torch.cat(self.rewards, dim=0).view(-1, 1)
        
        # Normalize rewards for stable learning (zero mean, unit variance)
        normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Create batches
        n_samples = len(self.pod_features)
        batch_size = min(self.batch_size, n_samples)
        batch_start = np.arange(0, n_samples, batch_size)
        indices = np.arange(n_samples, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + batch_size] for i in batch_start]
        
        logger.info(f"Starting learning with {n_samples} experiences in memory")
        
        epoch_loss = 0
        epoch_entropy = 0
        num_updates = 0
        
        # Process each batch
        for batch_idx, batch_indices in enumerate(batches):
            # Get batch data
            batch_pod_features = pod_features[batch_indices]
            batch_kv_hit_ratios = kv_hit_ratios[batch_indices]
            batch_request_features = request_features[batch_indices]
            batch_actions = actions[batch_indices]
            batch_rewards = normalized_rewards[batch_indices]
            
            # Get current policy distributions
            action_probs = self.policy(batch_pod_features, batch_kv_hit_ratios, batch_request_features)
            dist = Categorical(action_probs)
            
            # Calculate log probabilities of the actions taken
            log_probs = dist.log_prob(batch_actions)
            
            # Calculate entropy for monitoring exploration
            entropy = dist.entropy().mean()
            
            # Calculate loss (negative because we're maximizing)
            # This is the policy gradient loss: -log_prob * reward
            # We want to increase probability of actions that led to high rewards
            loss = -(log_probs * batch_rewards.squeeze()).mean()
            
            # Update policy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            epoch_entropy += entropy.item()
            num_updates += 1
        
        # Clear memory
        self.clear_memory()
        
        # Store metrics
        avg_loss = epoch_loss / max(1, num_updates)
        avg_reward = rewards.mean().item()
        avg_entropy = epoch_entropy / max(1, num_updates)
        
        self.loss_history.append(avg_loss)
        self.reward_history.append(avg_reward)
        self.entropy_history.append(avg_entropy)
        
        return {
            'loss': avg_loss,
            'reward': avg_reward,
            'entropy': avg_entropy
        }

    def clear_memory(self):
        """Clear memory buffers"""
        self.pod_features = []
        self.kv_hit_ratios = []
        self.request_features = []
        self.actions = []
        self.rewards = []

    def save(self, directory):
        """Save the agent's parameters to the specified directory"""
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Creating directory: {directory}")
        
        # Save policy network
        torch.save(self.policy.state_dict(), os.path.join(directory, 'policy.pth'))
        # Save training history
        history = {
            'loss': self.loss_history,
            'reward': self.reward_history,
            'entropy': self.entropy_history
        }
        
        with open(os.path.join(directory, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)
            
        os.makedirs(final_model_path, exist_ok=True)
        os.system(f"cp {directory}/* {final_model_path}")
        logger.info(f"Saved agent to {directory}")
    
    def load(self, directory):
        """Load the agent's parameters from the specified directory"""
        # Load policy network
        self.policy.load_state_dict(torch.load(os.path.join(directory, 'policy.pth')))
        
        # Load training history if available
        if os.path.exists(os.path.join(directory, 'history.pkl')):
            with open(os.path.join(directory, 'history.pkl'), 'rb') as f:
                history = pickle.load(f)
                
                self.loss_history = history.get('loss', [])
                self.reward_history = history.get('reward', [])
                self.entropy_history = history.get('entropy', [])
                
        logger.info(f"Loaded agent from {directory}")


class RoutingDataset(Dataset):
    """
    Dataset for LLM routing data, creates batches for contextual bandit training
    """
    def __init__(self, tensor_data):
        self.pod_features = tensor_data['pod_features_with_staleness']
        self.kv_hit_ratios = tensor_data['kv_hit_ratios']
        self.request_features = tensor_data['request_features']
        self.actions = tensor_data['actions']
        self.rewards = tensor_data['rewards']
        
    def __len__(self):
        return len(self.rewards)
    
    def __getitem__(self, idx):
        return {
            'pod_features': self.pod_features[idx],
            'kv_hit_ratios': self.kv_hit_ratios[idx],
            'request_features': self.request_features[idx],
            'action': self.actions[idx],
            'reward': self.rewards[idx]
        }


def evaluate_agent(agent, eval_data, num_samples=100):
    # Extract data
    pod_features = eval_data['pod_features_with_staleness'].to(device)
    kv_hit_ratios = eval_data['kv_hit_ratios'].to(device)
    request_features = eval_data['request_features'].to(device)
    true_actions = eval_data['actions'].to(device)
    rewards = eval_data['rewards'].to(device)
    
    # Limit to specified number of samples
    if len(rewards) > num_samples:
        indices = torch.randperm(len(rewards))[:num_samples]
        pod_features = pod_features[indices]
        kv_hit_ratios = kv_hit_ratios[indices]
        request_features = request_features[indices]
        true_actions = true_actions[indices]
        rewards = rewards[indices]
    
    # Evaluate agent
    agent.policy.eval()
    with torch.no_grad():
        # Get agent's actions
        pred_actions = agent.choose_action(pod_features, kv_hit_ratios, request_features, evaluate=True)
        
        # Get action probabilities
        action_probs = agent.policy(pod_features, kv_hit_ratios, request_features)
        
        # Calculate accuracy
        accuracy = (pred_actions == true_actions).float().mean().item()
        
        # Calculate reward for predicted actions
        # This is a simplification - in a real environment, we'd get the actual reward
        # Here we compare with the reward for the actual action taken
        true_reward = rewards.mean().item()
        
    # Additional metrics
    metrics = {
        'accuracy': accuracy,
        'true_reward': true_reward,
        'probs': action_probs.cpu().numpy(),
        'pred_actions': pred_actions.cpu().numpy(),
        'true_actions': true_actions.cpu().numpy()
    }
    
    agent.policy.train()
    return metrics


def plot_training_metrics(agent, eval_metrics, output_dir):
    """Plot training metrics and save figures"""
    plt.figure(figsize=(15, 10))
    
    # Plot policy loss
    plt.subplot(2, 2, 1)
    plt.plot(agent.loss_history)
    plt.title('Policy Loss')
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    
    # Plot rewards
    plt.subplot(2, 2, 2)
    plt.plot(agent.reward_history)
    plt.title('Average Reward')
    plt.xlabel('Updates')
    plt.ylabel('Reward')
    
    # Plot entropy
    plt.subplot(2, 2, 3)
    plt.plot(agent.entropy_history)
    plt.title('Policy Entropy')
    plt.xlabel('Updates')
    plt.ylabel('Entropy')
    
    # Plot evaluation accuracy
    plt.subplot(2, 2, 4)
    plt.plot([m['accuracy'] for m in eval_metrics])
    plt.title('Evaluation Accuracy')
    plt.xlabel('Evaluations')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    fn = f"{output_dir}/training_metrics.pdf"
    plt.savefig(fn)
    os.system(f"cp {fn} {final_model_path}")
    # Plot action distribution
    if len(eval_metrics) > 0:
        last_eval = eval_metrics[-1]
        plt.figure(figsize=(12, 6))
        
        # Plot action distribution
        plt.subplot(1, 2, 1)
        action_counts = np.bincount(last_eval['true_actions'], minlength=agent.policy.policy_head.out_features)
        plt.bar(np.arange(agent.policy.policy_head.out_features), action_counts)
        plt.title('True Action Distribution')
        plt.xlabel('Action')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        action_counts = np.bincount(last_eval['pred_actions'], minlength=agent.policy.policy_head.out_features)
        plt.bar(np.arange(agent.policy.policy_head.out_features), action_counts)
        plt.title('Predicted Action Distribution')
        plt.xlabel('Action')
        plt.ylabel('Count')
        
        plt.tight_layout()
        fn = f"{output_dir}/action_distribution.pdf"
        plt.savefig(fn)
        os.system(f"cp {fn} {final_model_path}")
        
        # Plot predicted probabilities
        if 'probs' in last_eval:
            plt.figure(figsize=(10, 8))
            avg_probs = last_eval['probs'].mean(axis=0)
            plt.bar(np.arange(len(avg_probs)), avg_probs)
            plt.title('Average Action Probabilities')
            plt.xlabel('Action')
            plt.ylabel('Probability')
            fn = f"{output_dir}/action_probabilities.pdf"
            plt.savefig(fn)
            os.system(f"cp {fn} {final_model_path}")
    
    logger.info(f"Saved training metrics plots to {output_dir}")


def load_all_encoded_data(encoded_data_dir):
    logger.info(f"Loading data from {encoded_data_dir}")
    
    # Find all batch subdirectories
    batch_dirs = glob.glob(os.path.join(encoded_data_dir, "batch_*"))
    logger.info(f"Found {len(batch_dirs)} batch directories: {batch_dirs}")
    
    combined_data = None
    total_samples = 0
    
    # Process each batch directory
    for batch_dir in batch_dirs:
        # Look for tensor_dataset.pt in the batch directory or its train subdirectory
        tensor_path = os.path.join(batch_dir, "tensor_dataset.pt")
        if not os.path.exists(tensor_path):
            train_dir = os.path.join(batch_dir, "train")
            if os.path.exists(train_dir):
                tensor_path = os.path.join(train_dir, "tensor_dataset.pt")
                
        if not os.path.exists(tensor_path):
            logger.warning(f"No tensor_dataset.pt found in {batch_dir} or its train subdirectory")
            continue
            
        try:
            # Load tensor data
            logger.debug(f"Loading tensor data from {tensor_path}")
            batch_data = torch.load(tensor_path)
            
            # If this is the first valid batch, use it as the base
            if combined_data is None:
                combined_data = batch_data
                total_samples = batch_data['rewards'].size(0)
                logger.info(f"First batch has {total_samples} samples")
            else:
                # Check that the tensors have compatible shapes (except for batch dimension)
                compatible = True
                for key in combined_data:
                    if key in batch_data:
                        # The shapes should be the same except for the first dimension (batch size)
                        combined_shape = combined_data[key].shape[1:]
                        batch_shape = batch_data[key].shape[1:]
                        
                        if combined_shape != batch_shape:
                            logger.warning(f"Incompatible shapes for key {key}: {combined_shape} vs {batch_shape}")
                            compatible = False
                            break
                
                if compatible:
                    # Combine the data by concatenating along the batch dimension
                    for key in combined_data:
                        if key in batch_data:
                            combined_data[key] = torch.cat([combined_data[key], batch_data[key]], dim=0)
                            
                    batch_samples = batch_data['rewards'].size(0)
                    total_samples += batch_samples
                    logger.debug(f"Added {batch_samples} samples from batch {os.path.basename(batch_dir)}")
                else:
                    logger.warning(f"Skipping incompatible batch {os.path.basename(batch_dir)}")
                    
        except Exception as e:
            logger.error(f"Error loading data from {batch_dir}: {e}")
            continue
    
    if combined_data is None:
        logger.error("No valid data could be loaded from any batch")
        raise ValueError("No valid data found in the encoded_data directory")
        
    logger.info(f"Successfully combined data from multiple batches, total samples: {total_samples}")
    
    # Log tensor shapes for debugging
    logger.info("Combined data tensor shapes:")
    for key, tensor in combined_data.items():
        logger.info(f"  {key}: {tensor.shape}")
    
    return combined_data

def load_previous_model():
    global final_model_path
    if os.path.exists(final_model_path):
        logger.info(f"Found previous model at {final_model_path}")
        return final_model_path

# def load_previous_model(encoded_data_dir):
#     global results_dir
#     # Look for results directory
#     results_dir = os.path.join(os.path.dirname(encoded_data_dir), "results")
#     if not os.path.exists(results_dir):
#         return None
        
#     # Find all model directories sorted by timestamp
#     model_dirs = sorted(glob.glob(os.path.join(results_dir, "cb_*")), reverse=True)
    
#     if not model_dirs:
#         return None
        
#     # Get the most recent model directory
#     latest_model_dir = model_dirs[0]
    
#     # Check if it has a final_model subdirectory
#     final_model_path = os.path.join(latest_model_dir, "final_model")
#     if os.path.exists(final_model_path):
#         logger.info(f"Found previous model at {final_model_path}")
#         return final_model_path
        
#     # If no final_model, look for the latest checkpoint
#     checkpoints = sorted(glob.glob(os.path.join(latest_model_dir, "checkpoint_epoch_*")), key=lambda x: int(x.split("_")[-1]), reverse=True)
    
#     if checkpoints:
#         logger.info(f"Found previous checkpoint at {checkpoints[0]}")
#         return checkpoints[0]
        
#     return None


def train(encoded_data_dir):
    global training_results_dir
    # Hyperparameters
    hidden_dim = 256
    batch_size = 64
    lr = 3e-4
    exploration_rate = 0.1
    training_epochs = 2
    max_updates_per_epoch = 1000
    eval_interval = 10
    seed = 42
    continue_training = False
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(training_results_dir, exist_ok=True)
    output_dir = os.path.join(training_results_dir, f"cb_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and combine data from all batches
    combined_data = load_all_encoded_data(encoded_data_dir)
    
    # Check if we should continue training from a previous model
    previous_model_path = None
    if continue_training:
        # previous_model_path = load_previous_model(encoded_data_dir)
        previous_model_path = load_previous_model()
        logger.info(f"Continue training from previous model: {previous_model_path}")
    
    # Create configuration
    config = {
        'hidden_dim': hidden_dim,
        'batch_size': batch_size,
        'learning_rate': lr,
        'exploration_rate': exploration_rate,
        'num_training_epochs': training_epochs,
        'max_updates_per_epoch': max_updates_per_epoch,
        'eval_interval': eval_interval,
        'seed': seed
    }
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Determine state dimensions
    state_dim = {
        'pod_features': combined_data['pod_features_with_staleness'].shape[2],
        'kv_hit_ratios': combined_data['kv_hit_ratios'].shape[2],
        'request_features': combined_data['request_features'].shape[1],
        'num_pods': combined_data['pod_features'].shape[1]
    }
    
    # Determine action dimension (number of pods)
    action_dim = combined_data['pod_features'].shape[1]
    
    logger.info(f"State dimensions: {state_dim}")
    logger.info(f"Action dimension: {action_dim}")
    
    # Create Contextual Bandit agent
    agent = ContextualBandit(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config['hidden_dim'],
        lr=config['learning_rate'],
        batch_size=config['batch_size'],
        exploration_rate=config['exploration_rate']
    )
    
    # Load previous model if available
    if previous_model_path:
        try:
            agent.load(previous_model_path)
            logger.info(f"Successfully loaded previous model from {previous_model_path}")
        except Exception as e:
            logger.error(f"Error loading previous model: {e}")
    
    # Create dataset
    dataset = RoutingDataset(combined_data)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    number_of_batches = len(dataloader)
    logger.info(f"Loaded dataset with {len(dataset)} samples")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Number of batches in training data: {number_of_batches}")

    # Training loop
    logger.info("Starting training...")
    total_updates = 0
    eval_metrics = []
    logger.info(f"Total number of training epochs: {config['num_training_epochs']}")
    
    for epoch in range(config['num_training_epochs']):
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_reward = 0
        epoch_entropy = 0
        epoch_updates = 0
        
        dataloader_iter = iter(dataloader)
        num_iter_per_data = 5  # Process each batch multiple times for better learning
        total_iter = number_of_batches * num_iter_per_data
        final_total_num_iteration = min(config['max_updates_per_epoch'], total_iter)
        total_num_data = len(dataset)
        
        logger.info(f"Epoch: {epoch+1}/{config['num_training_epochs']}, "
                   f"Total iterations: {final_total_num_iteration}. "
                   f"Total data: {total_num_data}, "
                   f"Number of batches: {number_of_batches}, "
                   f"Iterations per data: {num_iter_per_data}")
        
        for batch_iter_idx in range(final_total_num_iteration):
            try:
                # Get next batch
                batch = next(dataloader_iter)
            except StopIteration:
                # Restart iterator if we've gone through all batches
                logger.debug(f"Batch iter: {batch_iter_idx+1}/{final_total_num_iteration}. "
                            f"Consumed all batches, reiterate from beginning")
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            
            # Process batch data
            pod_features = batch['pod_features'].to(device)
            kv_hit_ratios = batch['kv_hit_ratios'].to(device)
            request_features = batch['request_features'].to(device)
            actions = batch['action'].to(device)
            rewards = batch['reward'].to(device).unsqueeze(1)
            
            # Store all data of this batch in agent memory
            logger.debug(f"Batch iter: {batch_iter_idx+1}/{final_total_num_iteration}. Storing {len(rewards)} experiences in memory")
            
            for j in range(len(rewards)):
                agent.remember(
                    pod_features[j:j+1], 
                    kv_hit_ratios[j:j+1], 
                    request_features[j:j+1], 
                    actions[j:j+1], 
                    rewards[j:j+1]
                )
            
            # Trigger learning every 5th batch iteration
            trigger_learning = (batch_iter_idx+1) % 5 == 0 or batch_iter_idx == final_total_num_iteration - 1
            if trigger_learning:
                logger.debug(f"Learning triggered! (Batch iter: {batch_iter_idx+1}/{final_total_num_iteration}), Memory size: {len(agent.pod_features)}")
                if len(agent.pod_features) > 0:  # Only learn if we have collected experiences
                    try:
                        update_metrics = agent.learn()
                        total_updates += 1
                        epoch_updates += 1
                        epoch_loss += update_metrics['loss']
                        epoch_reward += update_metrics['reward']
                        epoch_entropy += update_metrics['entropy']
                        
                        # Log progress
                        if batch_iter_idx % (final_total_num_iteration//5) == 0:
                            logger.info(f"Batch: {batch_iter_idx+1}/{final_total_num_iteration}, "
                                       f"Loss: {update_metrics['loss']:.4f}, "
                                       f"Reward: {update_metrics['reward']:.4f}, "
                                       f"Entropy: {update_metrics['entropy']:.4f}")
                    except Exception as e:
                        logger.error(f"Error during learning: {e}")
            
            # Evaluate the agent periodically
            temp = min(config['eval_interval'], final_total_num_iteration)
            if (batch_iter_idx + 1) % (final_total_num_iteration // temp) == 0 or batch_iter_idx == final_total_num_iteration - 1:
                logger.info(f"Evaluating agent at batch {batch_iter_idx+1}/{final_total_num_iteration}")
                
                try:
                    # Create a validation subset for evaluation
                    eval_indices = torch.randperm(len(dataset))[:min(1000, len(dataset))]
                    eval_data = {
                        'pod_features_with_staleness': combined_data['pod_features_with_staleness'][eval_indices],
                        'kv_hit_ratios': combined_data['kv_hit_ratios'][eval_indices],
                        'request_features': combined_data['request_features'][eval_indices],
                        'actions': combined_data['actions'][eval_indices],
                        'rewards': combined_data['rewards'][eval_indices]
                    }
                    
                    metrics = evaluate_agent(agent, eval_data)
                    eval_metrics.append(metrics)
                    
                    logger.info(f"Evaluation metrics - Accuracy: {metrics['accuracy']:.4f}, "
                               f"True Reward: {metrics['true_reward']:.4f}")
                    
                    ## Save checkpoint
                    # checkpoint_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}_batch_{batch_iter_idx+1}")
                    # os.makedirs(checkpoint_dir, exist_ok=True)
                    # agent.save(checkpoint_dir)
                    # logger.info(f"Saved checkpoint to {checkpoint_dir}")
                    
                except Exception as e:
                    logger.error(f"Error during evaluation: {e}")
        
        # End of epoch
        epoch_duration = time.time() - epoch_start_time
        
        if epoch_updates > 0:
            avg_loss = epoch_loss / epoch_updates
            avg_reward = epoch_reward / epoch_updates
            avg_entropy = epoch_entropy / epoch_updates
            
            logger.info(f"Epoch {epoch+1}/{config['num_training_epochs']} completed in {epoch_duration:.2f}s - "
                       f"Avg Loss: {avg_loss:.4f}, "
                       f"Avg Reward: {avg_reward:.4f}, "
                       f"Avg Entropy: {avg_entropy:.4f}")
        else:
            logger.warning(f"Epoch {epoch+1}/{config['num_training_epochs']} completed with no updates")
    
    # End of training
    logger.info(f"Training completed with {total_updates} total updates")
    
    # Save final model
    agent.save(output_dir)
    logger.info(f"Saved final model to {output_dir}")
    
    # Plot training metrics
    try:
        plot_training_metrics(agent, eval_metrics, output_dir)
    except Exception as e:
        logger.error(f"Error plotting training metrics: {e}")
    
    return {
        'agent': agent,
        'model_dir': output_dir,
        'output_dir': output_dir,
        'config': config,
        'eval_metrics': eval_metrics
    }

# Add this new function to contextual_bandit.py
def infer_from_tensor(tensor_data, exploration_enabled=False, exploration_rate=0.1):
    global final_model_path
    # Find the latest model if not specified
    # if model_dir is None:
    #     if not os.path.exists(results_dir):
    #         raise ValueError("No trained models found")
            
    #     if not final_model_path:
    #         raise ValueError("No trained contextual bandit models found")
            
        # # Get the most recent model directory
        # latest_model_dir = model_dirs[0]
        
        # # Check if it has a final_model subdirectory
        # final_model_path = os.path.join(latest_model_dir, "final_model")
        # if os.path.exists(final_model_path):
        #     model_dir = final_model_path
        # else:
        #     # If no final_model, look for the latest checkpoint
        #     checkpoints = sorted(glob.glob(os.path.join(latest_model_dir, "checkpoint_epoch_*")), 
        #                         key=lambda x: int(x.split("_")[-1]), 
        #                         reverse=True)
        #     if checkpoints:
        #         model_dir = checkpoints[0]
        #     else:
        #         raise ValueError("No trained model checkpoints found")
    
    logger.info(f"Using model from {final_model_path} for inference")
    
    ###########################################################
    # Print all available keys in tensor_data
    logger.info("Available tensor data keys:")
    for key in tensor_data.keys():
        if isinstance(tensor_data[key], torch.Tensor):
            logger.info(f"  {key}: shape={tensor_data[key].shape}, dtype={tensor_data[key].dtype}")
        else:
            logger.info(f"  {key}: type={type(tensor_data[key])}")
    
    # Try to load metadata to get feature names if available
    metadata_file = "metadata.json"
    pod_features_list_file = "pod_features_list.pkl"
    feature_indices_map_file = "feature_indices_map.pkl"
    
    try:
        feature_names = {
            "pod_features": [],
            "request_features": []
        }
        
        # Load metadata if available
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                logger.info("Loaded feature dimensions from metadata:")
                for key, value in metadata.get('feature_dimensions', {}).items():
                    logger.info(f"  {key}: {value}")
        
        # Try to load pod feature names
        if os.path.exists(pod_features_list_file):
            with open(pod_features_list_file, 'rb') as f:
                pod_features_list = pickle.load(f)
                feature_names["pod_features"] = pod_features_list
                logger.info("Pod features used in inference:")
                for i, feature in enumerate(pod_features_list):
                    logger.info(f"  {i}: {feature}")
        
        # Try to load feature indices map
        if os.path.exists(feature_indices_map_file):
            with open(feature_indices_map_file, 'rb') as f:
                feature_indices_map = pickle.load(f)
                logger.info("Feature indices map:")
                for feature, idx in feature_indices_map.items():
                    logger.info(f"  {feature}: index={idx}")
    
    except Exception as e:
        logger.error(f"Error loading feature metadata: {e}")
        logger.info("Continuing with inference without feature names")
    
    ###########################################################
    
    # Extract data from tensor dataset and move to device
    try:
        pod_features = tensor_data['pod_features_with_staleness'].to(device)
        kv_hit_ratios = tensor_data['kv_hit_ratios'].to(device)
        request_features = tensor_data['request_features'].to(device)
    except KeyError as e:
        logger.error(f"Missing key in tensor data: {e}")
        raise ValueError(f"Missing key in tensor data: {e}")
    
    # Ensure data is in batch format (add batch dimension if needed)
    if len(pod_features.shape) == 2:
        pod_features = pod_features.unsqueeze(0)
    if len(kv_hit_ratios.shape) == 2:
        kv_hit_ratios = kv_hit_ratios.unsqueeze(0)
    if len(request_features.shape) == 1:
        request_features = request_features.unsqueeze(0)


     # Analyze request features
    if request_features is not None:
        logger.info("Analyzing request features used for inference:")
        try:
            # Get information about the request features from tensor_data
            if 'feature_info' in tensor_data:
                feature_info = tensor_data['feature_info']
                if 'numeric_request_features' in feature_info:
                    numeric_features = feature_info['numeric_request_features']
                    logger.info(f"Numeric request features ({len(numeric_features)}):")
                    for i, feature in enumerate(numeric_features):
                        logger.info(f"  {i}: {feature}")
                
                if 'categorical_request_features' in feature_info:
                    categorical_features = feature_info['categorical_request_features']
                    logger.info(f"Categorical request features ({len(categorical_features)}):")
                    for i, feature in enumerate(categorical_features):
                        logger.info(f"  {i}: {feature}")
            
            # Print the actual values in the tensor
            logger.info("Request features tensor values:")
            if request_features.shape[0] > 0:
                # Print the first row of features
                feature_values = request_features[0].cpu().numpy().flatten()
                logger.info(f"  Values (first row, {len(feature_values)} features): {feature_values}")
                
                # Check for values that might indicate important features
                non_zero_indices = np.nonzero(feature_values)[0]
                logger.info(f"  Non-zero features (may be most important): {non_zero_indices}")
                for idx in non_zero_indices:
                    logger.info(f"    Feature index {idx}: Value = {feature_values[idx]}")
        
        except Exception as e:
            logger.error(f"Error analyzing request features: {e}")
            logger.error(traceback.format_exc())
            logger.info("Continuing with inference despite feature analysis error")
    
    
    
    
    # Determine state dimensions
    state_dim = {
        'pod_features': pod_features.shape[2],
        'kv_hit_ratios': kv_hit_ratios.shape[2],
        'request_features': request_features.shape[1],
        'num_pods': pod_features.shape[1]
    }
    
    # Determine action dimension (number of pods)
    action_dim = pod_features.shape[1]
    
    # Create agent and load model
    agent = ContextualBandit(
        state_dim=state_dim,
        action_dim=action_dim,
        exploration_rate=exploration_rate
    )
    agent.load(final_model_path)
    logger.info(f"Loaded model from {final_model_path}")

    # Set to evaluation mode
    agent.policy.eval()
    with torch.no_grad():
        # Get action probabilities
        action_probs = agent.policy(pod_features, kv_hit_ratios, request_features)
        
        if exploration_enabled:
            # Use exploration strategy (epsilon-greedy)
            action, _ = agent.policy.get_action(
                pod_features, kv_hit_ratios, request_features, 
                explore=True, 
                epsilon=exploration_rate
            )
            selected_action = action.item()
            confidence = action_probs[0, selected_action].item()
        else:
            # Use pure exploitation (select best pod)
            selected_action = torch.argmax(action_probs, dim=1).item()
            confidence = action_probs[0, selected_action].item()
    
    # Return inference results
    return {
        'selected_pod_index': selected_action,
        'confidence': confidence,
        'pod_probabilities': action_probs[0].cpu().numpy().tolist(),
        'final_model_path': final_model_path,
        'exploration_enabled': exploration_enabled
    }