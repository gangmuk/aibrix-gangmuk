#!/usr/bin/env python3
"""
LLM Request Router - RL Training
--------------------------------
Trains a Proximal Policy Optimization (PPO) agent for routing LLM inference requests
to optimal pods in a GPU cluster.

Usage:
  python train_routing_agent_ppo.py /path/to/encoded_data [--continue-training]
  
The script loads preprocessed data from the specified directory and trains
a reinforcement learning agent using the PPO algorithm.
"""

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
import logging
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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


class ActorNetwork(nn.Module):
    """
    Actor Network for PPO, outputs action probabilities
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
    
    def get_action(self, pod_features, kv_hit_ratios, request_features, evaluate=False):
        action_probs = self.forward(pod_features, kv_hit_ratios, request_features)
        
        if evaluate:
            # During evaluation, select the pod with highest probability
            return torch.argmax(action_probs, dim=1)
        else:
            # During training, sample from the distribution
            dist = Categorical(action_probs)
            action = dist.sample()
            return action, dist.log_prob(action)


class CriticNetwork(nn.Module):
    """
    Critic Network for PPO, outputs state value
    """
    def __init__(self, state_dim, hidden_dim=256):
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
        
        # Output layer for value
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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
        
        # 4. Apply attention across pods to capture inter-pod relationships
        attn_output, _ = self.pod_attention(
            query=contextualized_pods,
            key=contextualized_pods,
            value=contextualized_pods,
            need_weights=False
        )
        
        # 5. Pool across pods to get a global state representation
        global_state = torch.mean(attn_output, dim=1)
        
        # 6. Compute state value
        value = self.value_head(global_state)
        
        return value


class PPOMemory:
    """
    Memory buffer for PPO training
    """
    def __init__(self, batch_size):
        self.pod_features = []
        self.kv_hit_ratios = []
        self.request_features = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        
    def store_memory(self, pod_features, kv_hit_ratios, request_features, action, probs, vals, reward, done):
        self.pod_features.append(pod_features)
        self.kv_hit_ratios.append(kv_hit_ratios)
        self.request_features.append(request_features)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def clear_memory(self):
        self.pod_features = []
        self.kv_hit_ratios = []
        self.request_features = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        
    def sample_buffer(self):
        pod_features = torch.cat(self.pod_features, dim=0)
        kv_hit_ratios = torch.cat(self.kv_hit_ratios, dim=0)
        request_features = torch.cat(self.request_features, dim=0)
        actions = torch.cat(self.actions, dim=0)
        old_probs = torch.cat(self.probs, dim=0)
        vals = torch.cat(self.vals, dim=0)
        rewards = torch.cat(self.rewards, dim=0)
        dones = torch.cat(self.dones, dim=0)
        
        return pod_features, kv_hit_ratios, request_features, actions, old_probs, vals, rewards, dones
    
    def get_batches(self):
        n_states = len(self.pod_features)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        
        pod_features = torch.cat(self.pod_features, dim=0)
        kv_hit_ratios = torch.cat(self.kv_hit_ratios, dim=0)
        request_features = torch.cat(self.request_features, dim=0)
        actions = torch.cat(self.actions, dim=0)
        old_probs = torch.cat(self.probs, dim=0)
        vals = torch.cat(self.vals, dim=0)
        rewards = torch.cat(self.rewards, dim=0)
        dones = torch.cat(self.dones, dim=0)
        
        return pod_features, kv_hit_ratios, request_features, actions, old_probs, vals, rewards, dones, batches


class PPO:
    """
    Proximal Policy Optimization (PPO) for LLM request routing
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr_actor=3e-4, lr_critic=3e-4,
                 gamma=0.99, gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        
        # Initialize actor and critic networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(device)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize memory
        self.memory = PPOMemory(batch_size)
        
        # Metrics for tracking
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.entropy_history = []
        
    def remember(self, pod_features, kv_hit_ratios, request_features, action, probs, val, reward, done):
        self.memory.store_memory(pod_features, kv_hit_ratios, request_features, action, probs, val, reward, done)
        
    def choose_action(self, pod_features, kv_hit_ratios, request_features, evaluate=False):
        """Select an action (pod) for the given state"""
        with torch.no_grad():
            if evaluate:
                # During evaluation, select the pod with highest probability
                action_probs = self.actor(pod_features, kv_hit_ratios, request_features)
                action = torch.argmax(action_probs, dim=1)
                return action
            else:
                # During training, sample from the distribution
                action, log_prob = self.actor.get_action(pod_features, kv_hit_ratios, request_features)
                value = self.critic(pod_features, kv_hit_ratios, request_features)
                return action, log_prob, value
    
    def learn(self):
        """Update policy and value function using PPO"""
        # Get all data from memory
        pod_features, kv_hit_ratios, request_features, actions, old_log_probs, values, rewards, dones, batches = self.memory.get_batches()
        
        # Initialize advantage
        advantages = torch.zeros_like(rewards).to(device)
        
        # Compute advantages using GAE
        for t in range(len(rewards) - 1):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * (
                advantages[t + 1] if t < len(rewards) - 2 else 0)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        
        # Compute target values
        target_values = rewards + self.gamma * values * (1 - dones)
        
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        epoch_entropy = 0
        
        # PPO epochs
        for _ in range(self.n_epochs):
            # Process each batch
            for batch_indices in batches:
                batch_pod_features = pod_features[batch_indices]
                batch_kv_hit_ratios = kv_hit_ratios[batch_indices]
                batch_request_features = request_features[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_target_values = target_values[batch_indices]
                
                # Get current policy distributions
                action_probs = self.actor(batch_pod_features, batch_kv_hit_ratios, batch_request_features)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Get current value estimates
                critic_value = self.critic(batch_pod_features, batch_kv_hit_ratios, batch_request_features).squeeze()
                # Reshape critic_value and batch_target_values to ensure they match
                critic_value = critic_value.view(-1)
                batch_target_values = batch_target_values.view(-1)
                
                # Compute policy ratio and clip
                prob_ratio = torch.exp(new_log_probs - batch_old_log_probs)
                weighted_probs = batch_advantages * prob_ratio
                weighted_clipped_probs = batch_advantages * torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                
                # Compute actor loss
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                # Compute critic loss
                critic_loss = F.mse_loss(critic_value, batch_target_values)
                
                # Update networks
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                
                # Track metrics
                epoch_actor_loss += actor_loss.item()
                epoch_critic_loss += critic_loss.item()
                epoch_entropy += entropy.item()
        
        # Clear memory
        self.memory.clear_memory()
        
        # Store metrics
        avg_actor_loss = epoch_actor_loss / (len(batches) * self.n_epochs)
        avg_critic_loss = epoch_critic_loss / (len(batches) * self.n_epochs)
        avg_entropy = epoch_entropy / (len(batches) * self.n_epochs)
        
        self.actor_loss_history.append(avg_actor_loss)
        self.critic_loss_history.append(avg_critic_loss)
        self.entropy_history.append(avg_entropy)
        
        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'entropy': avg_entropy
        }
    
    def save(self, directory):
        """Save the agent's parameters to the specified directory"""
        os.makedirs(directory, exist_ok=True)
        
        # Save networks
        torch.save(self.actor.state_dict(), os.path.join(directory, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(directory, 'critic.pth'))
        
        # Save training history
        history = {
            'actor_loss': self.actor_loss_history,
            'critic_loss': self.critic_loss_history,
            'entropy': self.entropy_history
        }
        
        with open(os.path.join(directory, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)
            
        logger.info(f"Saved agent to {directory}")
    
    def load(self, directory):
        """Load the agent's parameters from the specified directory"""
        # Load networks
        self.actor.load_state_dict(torch.load(os.path.join(directory, 'actor.pth')))
        self.critic.load_state_dict(torch.load(os.path.join(directory, 'critic.pth')))
        
        # Load training history if available
        if os.path.exists(os.path.join(directory, 'history.pkl')):
            with open(os.path.join(directory, 'history.pkl'), 'rb') as f:
                history = pickle.load(f)
                
                self.actor_loss_history = history.get('actor_loss', [])
                self.critic_loss_history = history.get('critic_loss', [])
                self.entropy_history = history.get('entropy', [])
                
        logger.info(f"Loaded agent from {directory}")


class RoutingDataset(Dataset):
    """
    Dataset for LLM routing data, creates batches for PPO training
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


def evaluate_agent(agent, eval_data, num_episodes=100):
    """
    Evaluate the agent's performance on a validation set.
    
    Args:
        agent: PPO agent to evaluate
        eval_data: Validation data in tensor format
        num_episodes: Number of evaluation episodes
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Extract data
    pod_features = eval_data['pod_features_with_staleness'].to(device)
    kv_hit_ratios = eval_data['kv_hit_ratios'].to(device)
    request_features = eval_data['request_features'].to(device)
    true_actions = eval_data['actions'].to(device)
    rewards = eval_data['rewards'].to(device)
    
    # Limit to specified number of episodes
    if len(rewards) > num_episodes:
        indices = torch.randperm(len(rewards))[:num_episodes]
        pod_features = pod_features[indices]
        kv_hit_ratios = kv_hit_ratios[indices]
        request_features = request_features[indices]
        true_actions = true_actions[indices]
        rewards = rewards[indices]
    
    # Evaluate agent
    agent.actor.eval()
    agent.critic.eval()
    with torch.no_grad():
        # Get agent's actions
        pred_actions = agent.choose_action(pod_features, kv_hit_ratios, request_features, evaluate=True)
        
        # Get action probabilities and state values
        action_probs = agent.actor(pod_features, kv_hit_ratios, request_features)
        state_values = agent.critic(pod_features, kv_hit_ratios, request_features)
        
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
        'avg_value': state_values.mean().item(),
        'probs': action_probs.cpu().numpy(),
        'pred_actions': pred_actions.cpu().numpy(),
        'true_actions': true_actions.cpu().numpy()
    }
    
    agent.actor.train()
    agent.critic.train()
    return metrics


def plot_training_metrics(agent, eval_metrics, output_dir):
    """Plot training metrics and save figures"""
    plt.figure(figsize=(15, 10))
    
    # Plot actor loss
    plt.subplot(2, 2, 1)
    plt.plot(agent.actor_loss_history)
    plt.title('Actor Loss')
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    
    # Plot critic loss
    plt.subplot(2, 2, 2)
    plt.plot(agent.critic_loss_history)
    plt.title('Critic Loss')
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    
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
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    
    # Plot action distribution
    if len(eval_metrics) > 0:
        last_eval = eval_metrics[-1]
        plt.figure(figsize=(12, 6))
        
        # Plot action distribution
        plt.subplot(1, 2, 1)
        action_counts = np.bincount(last_eval['true_actions'], minlength=agent.actor.policy_head.out_features)
        plt.bar(np.arange(agent.actor.policy_head.out_features), action_counts)
        plt.title('True Action Distribution')
        plt.xlabel('Action')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        action_counts = np.bincount(last_eval['pred_actions'], minlength=agent.actor.policy_head.out_features)
        plt.bar(np.arange(agent.actor.policy_head.out_features), action_counts)
        plt.title('Predicted Action Distribution')
        plt.xlabel('Action')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'action_distribution.png'))
        
        # Plot predicted probabilities
        if 'probs' in last_eval:
            plt.figure(figsize=(10, 8))
            avg_probs = last_eval['probs'].mean(axis=0)
            plt.bar(np.arange(len(avg_probs)), avg_probs)
            plt.title('Average Action Probabilities')
            plt.xlabel('Action')
            plt.ylabel('Probability')
            plt.savefig(os.path.join(output_dir, 'action_probabilities.png'))
    
    logger.info(f"Saved training metrics plots to {output_dir}")


def load_all_encoded_data(encoded_data_dir):
    """
    Load and combine data from all batch subdirectories in the encoded_data directory.
    
    Args:
        encoded_data_dir: Path to the directory containing batch_* subdirectories
        
    Returns:
        combined_data: Dictionary with combined tensor data
    """
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
            logger.info(f"Loading tensor data from {tensor_path}")
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
                    logger.info(f"Added {batch_samples} samples from batch {os.path.basename(batch_dir)}")
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


def load_previous_model(encoded_data_dir):
    """
    Try to load the previously trained model if it exists.
    
    Args:
        encoded_data_dir: Path to the encoded_data directory
        
    Returns:
        model_path: Path to the latest model, or None if not found
    """
    # Look for results directory
    results_dir = os.path.join(os.path.dirname(encoded_data_dir), "results")
    if not os.path.exists(results_dir):
        return None
        
    # Find all PPO model directories sorted by timestamp
    model_dirs = sorted(glob.glob(os.path.join(results_dir, "ppo_*")), reverse=True)
    
    if not model_dirs:
        return None
        
    # Get the most recent model directory
    latest_model_dir = model_dirs[0]
    
    # Check if it has a final_model subdirectory
    final_model_path = os.path.join(latest_model_dir, "final_model")
    if os.path.exists(final_model_path):
        logger.info(f"Found previous model at {final_model_path}")
        return final_model_path
        
    # If no final_model, look for the latest checkpoint
    checkpoints = sorted(glob.glob(os.path.join(latest_model_dir, "checkpoint_epoch_*")), 
                         key=lambda x: int(x.split("_")[-1]), 
                         reverse=True)
    
    if checkpoints:
        logger.info(f"Found previous checkpoint at {checkpoints[0]}")
        return checkpoints[0]
        
    return None


def train(encoded_data_dir):
    """Main function to train the LLM routing agent using PPO on all available data"""
    parser = argparse.ArgumentParser(description='Train LLM routing agent with PPO')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension of neural networks')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda parameter')
    parser.add_argument('--policy-clip', type=float, default=0.2, help='PPO clip parameter')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--updates-per-epoch', type=int, default=1000, help='Number of updates per epoch')
    parser.add_argument('--n-epochs', type=int, default=10, help='Number of PPO epochs per update')
    parser.add_argument('--eval-interval', type=int, default=5, help='Evaluation interval (epochs)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--continue-training', action='store_true', help='Continue training from the latest model')
    
    try:
        args = parser.parse_args()
    except:
        # If called from another script, use default arguments
        args = parser.parse_args([])
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "training_results"
        os.makedirs(results_dir, exist_ok=True)
        args.output_dir = os.path.join(results_dir, f"ppo_{timestamp}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and combine data from all batches
    combined_data = load_all_encoded_data(encoded_data_dir)
    
    # Check if we should continue training from a previous model
    previous_model_path = None
    if args.continue_training:
        previous_model_path = load_previous_model(encoded_data_dir)
        logger.info(f"Continue training from previous model: {previous_model_path}")
    
    # Create configuration
    config = {
        'hidden_dim': args.hidden_dim,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'policy_clip': args.policy_clip,
        'num_epochs': args.epochs,
        'n_epochs': args.n_epochs,
        'updates_per_epoch': args.updates_per_epoch,
        'eval_interval': args.eval_interval,
        'seed': args.seed
    }
    
    # Save configuration
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
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
    
    # Create PPO agent
    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config['hidden_dim'],
        lr_actor=config['learning_rate'],
        lr_critic=config['learning_rate'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        policy_clip=config['policy_clip'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs']
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
    
    # Training loop
    logger.info("Starting training...")
    total_updates = 0
    eval_metrics = []
    
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        epoch_entropy = 0
        epoch_updates = 0
        
        # Training updates for this epoch
        for i in range(min(config['updates_per_epoch'], len(dataloader))):
            # Get batch data
            batch = next(iter(dataloader))
            
            pod_features = batch['pod_features'].to(device)
            kv_hit_ratios = batch['kv_hit_ratios'].to(device)
            request_features = batch['request_features'].to(device)
            actions = batch['action'].to(device)
            rewards = batch['reward'].to(device).unsqueeze(1)
            
            # Get current policy probabilities, values
            with torch.no_grad():
                old_action_probs = agent.actor(pod_features, kv_hit_ratios, request_features)
                old_log_probs = torch.log(torch.gather(old_action_probs, 1, actions.unsqueeze(1)) + 1e-10).squeeze()
                old_values = agent.critic(pod_features, kv_hit_ratios, request_features)
            
            # Store transitions in memory
            for j in range(len(rewards)):
                # Fake done flags (all False since we don't have episode boundaries in offline data)
                done = torch.zeros(1, device=device)
                
                agent.remember(
                    pod_features[j:j+1], 
                    kv_hit_ratios[j:j+1], 
                    request_features[j:j+1], 
                    actions[j:j+1], 
                    old_log_probs[j:j+1], 
                    old_values[j:j+1], 
                    rewards[j:j+1], 
                    done
                )
            
            # Learn if we've collected enough data
            if i % 10 == 0 or i == config['updates_per_epoch'] - 1:
                update_metrics = agent.learn()
                
                total_updates += 1
                epoch_updates += 1
                epoch_actor_loss += update_metrics['actor_loss']
                epoch_critic_loss += update_metrics['critic_loss']
                epoch_entropy += update_metrics['entropy']
                
                # Log progress
                if i % 100 == 0:
                    logger.info(f"Epoch {epoch+1}/{config['num_epochs']}, "
                              f"Update {i+1}/{config['updates_per_epoch']}, "
                              f"Actor Loss: {update_metrics['actor_loss']:.4f}, "
                              f"Critic Loss: {update_metrics['critic_loss']:.4f}, "
                              f"Entropy: {update_metrics['entropy']:.4f}")
        
        # Calculate average metrics for the epoch
        avg_actor_loss = epoch_actor_loss / max(1, epoch_updates)
        avg_critic_loss = epoch_critic_loss / max(1, epoch_updates)
        avg_entropy = epoch_entropy / max(1, epoch_updates)
        epoch_time = time.time() - epoch_start_time
        
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']} completed in {epoch_time:.2f}s, "
                  f"Avg Actor Loss: {avg_actor_loss:.4f}, "
                  f"Avg Critic Loss: {avg_critic_loss:.4f}, "
                  f"Avg Entropy: {avg_entropy:.4f}")
        
        # Evaluate agent
        if (epoch + 1) % config['eval_interval'] == 0 or epoch == config['num_epochs'] - 1:
            # Use the combined data for evaluation
            eval_data = combined_data
            logger.info("Evaluating on combined data...")
            
            metrics = evaluate_agent(agent, eval_data)
            eval_metrics.append(metrics)
            
            logger.info(f"Evaluation: Accuracy: {metrics['accuracy']:.4f}, "
                      f"True Reward: {metrics['true_reward']:.4f}, "
                      f"Avg Value: {metrics['avg_value']:.4f}")
            
            # Save agent checkpoint
            agent.save(os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}"))
            
            # Plot training metrics
            plot_training_metrics(agent, eval_metrics, args.output_dir)
    
    # Final save
    agent.save(os.path.join(args.output_dir, "final_model"))
    
    # Save evaluation metrics
    with open(os.path.join(args.output_dir, 'eval_metrics.pkl'), 'wb') as f:
        pickle.dump(eval_metrics, f)
    
    logger.info(f"Training completed, model saved to {args.output_dir}")
    return args.output_dir


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_routing_agent_ppo.py /path/to/encoded_data_dir [--continue-training]")
        sys.exit(1)
    
    encoded_data_dir = sys.argv[1]
    train(encoded_data_dir)