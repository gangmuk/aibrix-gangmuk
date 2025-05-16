#!/usr/bin/env python3
"""
LLM Request Router - RL Training
--------------------------------
Trains a Soft Actor-Critic (SAC) agent for routing LLM inference requests
to optimal pods in a GPU cluster.

Usage:
  python train_routing_agent.py /path/to/processed_data
  
The script loads preprocessed data from the specified directory and trains
a reinforcement learning agent using the SAC algorithm.
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
from logger import logger
import glob

# Configure logging
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


class LLMRoutingAgent(nn.Module):
    """
    Neural network architecture for LLM request routing.
    Implements parallel processing for pod features and KV hit ratios,
    with attention mechanisms across pods.
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
        
        # Output layer for routing decisions
        self.routing_head = nn.Linear(hidden_dim, 1)
        
        # Value head for critic
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
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
        
        # 5. Final routing logits
        routing_logits = self.routing_head(attn_output).squeeze(-1)
        
        # 6. State value (for critic function)
        state_value = self.value_head(torch.mean(attn_output, dim=1))
        
        if return_attention:
            return routing_logits, state_value, attn_weights
        
        return routing_logits, state_value


class ReplayBuffer:
    """
    Experience replay buffer for off-policy learning.
    Stores transitions and supports prioritized sampling.
    """
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.pod_features = None
        self.kv_hit_ratios = None
        self.request_features = None
        self.actions = None
        self.rewards = None
        self.next_pod_features = None
        self.next_kv_hit_ratios = None
        self.next_request_features = None
        self.dones = None
        
        # For prioritized replay
        self.priorities = None
        
    def init_from_batch(self, batch):
        """Initialize buffer structure from a batch of data"""
        pod_features, kv_hit_ratios, request_features, actions, rewards = batch[:5]
        
        # Get dimensions
        batch_size, num_pods, pod_dim = pod_features.shape
        _, _, kv_dim = kv_hit_ratios.shape
        _, request_dim = request_features.shape
        
        # Initialize arrays
        self.pod_features = np.zeros((self.max_size, num_pods, pod_dim), dtype=np.float32)
        self.kv_hit_ratios = np.zeros((self.max_size, num_pods, kv_dim), dtype=np.float32)
        self.request_features = np.zeros((self.max_size, request_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size, 1), dtype=np.int64)
        self.rewards = np.zeros((self.max_size, 1), dtype=np.float32)
        self.next_pod_features = np.zeros((self.max_size, num_pods, pod_dim), dtype=np.float32)
        self.next_kv_hit_ratios = np.zeros((self.max_size, num_pods, kv_dim), dtype=np.float32)
        self.next_request_features = np.zeros((self.max_size, request_dim), dtype=np.float32)
        self.dones = np.zeros((self.max_size, 1), dtype=np.float32)
        
        # Initialize priorities with ones
        self.priorities = np.ones((self.max_size, 1), dtype=np.float32)
        
    def add(self, pod_features, kv_hit_ratios, request_features, action, reward,
           next_pod_features, next_kv_hit_ratios, next_request_features, done):
        """Add a new transition to the buffer"""
        # Initialize buffer if not done yet
        if self.pod_features is None:
            _, num_pods, pod_dim = pod_features.shape
            _, _, kv_dim = kv_hit_ratios.shape
            _, request_dim = request_features.shape
            
            self.pod_features = np.zeros((self.max_size, num_pods, pod_dim), dtype=np.float32)
            self.kv_hit_ratios = np.zeros((self.max_size, num_pods, kv_dim), dtype=np.float32)
            self.request_features = np.zeros((self.max_size, request_dim), dtype=np.float32)
            self.actions = np.zeros((self.max_size, 1), dtype=np.int64)
            self.rewards = np.zeros((self.max_size, 1), dtype=np.float32)
            self.next_pod_features = np.zeros((self.max_size, num_pods, pod_dim), dtype=np.float32)
            self.next_kv_hit_ratios = np.zeros((self.max_size, num_pods, kv_dim), dtype=np.float32)
            self.next_request_features = np.zeros((self.max_size, request_dim), dtype=np.float32)
            self.dones = np.zeros((self.max_size, 1), dtype=np.float32)
            self.priorities = np.ones((self.max_size, 1), dtype=np.float32)
        
        # Store transition
        self.pod_features[self.ptr] = pod_features
        self.kv_hit_ratios[self.ptr] = kv_hit_ratios
        self.request_features[self.ptr] = request_features
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_pod_features[self.ptr] = next_pod_features
        self.next_kv_hit_ratios[self.ptr] = next_kv_hit_ratios
        self.next_request_features[self.ptr] = next_request_features
        self.dones[self.ptr] = done
        
        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size, prioritized=True, alpha=0.6, beta=0.4):
        """Sample a batch of transitions, with prioritized replay if enabled"""
        if prioritized and self.size > 1:
            # Prioritized sampling
            probs = self.priorities[:self.size].squeeze()
            probs = probs ** alpha
            probs = probs / probs.sum()
            
            indices = np.random.choice(self.size, batch_size, replace=False, p=probs)
            
            # Compute importance sampling weights
            weights = (self.size * probs[indices]) ** (-beta)
            weights = weights / weights.max()
            weights = np.array(weights, dtype=np.float32).reshape(-1, 1)
        else:
            # Uniform sampling
            indices = np.random.randint(0, self.size, size=batch_size)
            weights = np.ones((batch_size, 1), dtype=np.float32)
        
        # Get batch
        pod_features = torch.FloatTensor(self.pod_features[indices]).to(device)
        kv_hit_ratios = torch.FloatTensor(self.kv_hit_ratios[indices]).to(device)
        request_features = torch.FloatTensor(self.request_features[indices]).to(device)
        actions = torch.LongTensor(self.actions[indices]).to(device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(device)
        next_pod_features = torch.FloatTensor(self.next_pod_features[indices]).to(device)
        next_kv_hit_ratios = torch.FloatTensor(self.next_kv_hit_ratios[indices]).to(device)
        next_request_features = torch.FloatTensor(self.next_request_features[indices]).to(device)
        dones = torch.FloatTensor(self.dones[indices]).to(device)
        weights = torch.FloatTensor(weights).to(device)
        
        return (pod_features, kv_hit_ratios, request_features, actions, rewards,
                next_pod_features, next_kv_hit_ratios, next_request_features, dones,
                weights, indices)
    
    def update_priorities(self, indices, priorities):
        """Update priorities for prioritized replay"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    @classmethod
    def from_offline_data(cls, tensor_data, buffer_size=100000):
        """Create a replay buffer from offline data"""
        buffer = cls(max_size=buffer_size)
        
        # Extract data
        pod_features = tensor_data['pod_features_with_staleness'].cpu().numpy()
        kv_hit_ratios = tensor_data['kv_hit_ratios'].cpu().numpy()
        request_features = tensor_data['request_features'].cpu().numpy()
        actions = tensor_data['actions'].cpu().numpy().reshape(-1, 1)
        rewards = tensor_data['rewards'].cpu().numpy().reshape(-1, 1)
        
        # Initialize buffer
        buffer.init_from_batch((pod_features, kv_hit_ratios, request_features, actions, rewards))
        
        # For offline data, we construct "next states" by rolling the states
        # This is a simplification but works for initial training on offline data
        num_samples = len(pod_features)
        
        # Add transitions
        buffer.size = min(num_samples - 1, buffer_size)
        for i in range(buffer.size):
            buffer.pod_features[i] = pod_features[i]
            buffer.kv_hit_ratios[i] = kv_hit_ratios[i]
            buffer.request_features[i] = request_features[i]
            buffer.actions[i] = actions[i]
            buffer.rewards[i] = rewards[i]
            
            # Next state (use the next sample in the dataset)
            buffer.next_pod_features[i] = pod_features[i+1]
            buffer.next_kv_hit_ratios[i] = kv_hit_ratios[i+1]
            buffer.next_request_features[i] = request_features[i+1]
            buffer.dones[i] = 0  # Assume all transitions are not terminal
            
        buffer.ptr = buffer.size % buffer.max_size
        logger.info(f"Created replay buffer with {buffer.size} transitions")
        
        return buffer


class SAC:
    """
    Soft Actor-Critic implementation for LLM request routing.
    Uses twin critics, automatic entropy tuning, and prioritized replay.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256,
                lr_actor=3e-4, lr_critic=3e-4, gamma=0.99,
                tau=0.005, alpha=0.2, auto_alpha=True):
        # State and action dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor network (policy)
        self.actor = LLMRoutingAgent(state_dim, action_dim, hidden_dim).to(device)
        
        # Twin critics
        self.critic1 = LLMRoutingAgent(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = LLMRoutingAgent(state_dim, action_dim, hidden_dim).to(device)
        
        # Target networks
        self.critic1_target = LLMRoutingAgent(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = LLMRoutingAgent(state_dim, action_dim, hidden_dim).to(device)
        
        # Copy parameters to target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)
        
        # Automatic entropy tuning
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -np.log(1.0 / action_dim) * 0.98  # slightly lower than uniform
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_actor)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
        
        # Training parameters
        self.gamma = gamma
        self.tau = tau
        
        # Tracking metrics
        self.critic_loss_history = []
        self.actor_loss_history = []
        self.alpha_loss_history = []
        self.entropy_history = []
        
    def select_action(self, pod_features, kv_hit_ratios, request_features, evaluate=False):
        """Select an action (pod) for the given state"""
        with torch.no_grad():
            routing_logits, _ = self.actor(pod_features, kv_hit_ratios, request_features)
            
            if evaluate:
                # During evaluation, select the pod with highest probability
                return torch.argmax(routing_logits, dim=1)
            else:
                # During training, sample from the distribution
                probs = F.softmax(routing_logits, dim=1)
                dist = torch.distributions.Categorical(probs)
                return dist.sample()
    
    def calculate_q_values(self, critic, pod_features, kv_hit_ratios, request_features, actions=None):
        """Calculate Q-values for a batch of states and optionally specific actions"""
        # Get Q-values for all actions
        q_logits, _ = critic(pod_features, kv_hit_ratios, request_features)
        
        if actions is not None:
            # Return Q-values for specific actions
            return torch.gather(q_logits, 1, actions)
        else:
            # Return Q-values for all actions
            return q_logits
    
    def update(self, replay_buffer, batch_size, update_alpha=True):
        """Update the agent's parameters using a batch of transitions"""
        # Sample a batch from the replay buffer
        (pod_features, kv_hit_ratios, request_features, actions, rewards,
         next_pod_features, next_kv_hit_ratios, next_request_features, dones,
         weights, indices) = replay_buffer.sample(batch_size)
        
        #=====================================================================
        # Update Critic Networks
        #=====================================================================
        
        # Compute target Q-values
        with torch.no_grad():
            # Get policy probabilities for next state
            next_logits, _ = self.actor(next_pod_features, next_kv_hit_ratios, next_request_features)
            next_probs = F.softmax(next_logits, dim=1)
            next_log_probs = F.log_softmax(next_logits, dim=1)
            
            # Compute entropy term
            entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
            
            # Get Q-values for next state from both target critics
            next_q1_all = self.calculate_q_values(self.critic1_target, 
                                               next_pod_features, 
                                               next_kv_hit_ratios, 
                                               next_request_features)
            next_q2_all = self.calculate_q_values(self.critic2_target, 
                                               next_pod_features, 
                                               next_kv_hit_ratios, 
                                               next_request_features)
            
            # Take minimum of the two Q-values for each action
            next_q_all = torch.min(next_q1_all, next_q2_all)
            
            # Compute expected Q-value and add entropy term
            expected_q = torch.sum(next_probs * next_q_all, dim=1, keepdim=True)
            target_q = rewards + (1 - dones) * self.gamma * (expected_q + self.alpha * entropy)
        
        # Compute current Q-values
        current_q1 = self.calculate_q_values(self.critic1, pod_features, kv_hit_ratios, request_features, actions)
        current_q2 = self.calculate_q_values(self.critic2, pod_features, kv_hit_ratios, request_features, actions)
        
        # Compute critic losses with importance sampling weights
        critic1_loss = F.mse_loss(current_q1, target_q, reduction='none') * weights
        critic2_loss = F.mse_loss(current_q2, target_q, reduction='none') * weights
        
        critic1_loss = critic1_loss.mean()
        critic2_loss = critic2_loss.mean()
        
        # Update priorities in the replay buffer
        with torch.no_grad():
            td_errors = torch.abs(target_q - current_q1).cpu().numpy()
            replay_buffer.update_priorities(indices, td_errors + 1e-6)  # Add small constant for stability
        
        # Update critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        # Update critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Record losses
        critic_loss = (critic1_loss.item() + critic2_loss.item()) / 2
        self.critic_loss_history.append(critic_loss)
        
        #=====================================================================
        # Update Actor Network
        #=====================================================================
        
        # Compute actor loss
        logits, _ = self.actor(pod_features, kv_hit_ratios, request_features)
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        
        # Get Q-values for all actions
        q1_all = self.calculate_q_values(self.critic1, pod_features, kv_hit_ratios, request_features)
        q2_all = self.calculate_q_values(self.critic2, pod_features, kv_hit_ratios, request_features)
        q_all = torch.min(q1_all, q2_all)
        
        # Compute entropy
        entropy = -torch.sum(probs * log_probs, dim=1)
        
        # Compute actor loss: maximize Q-value and entropy
        actor_loss = -torch.mean(torch.sum(probs * (q_all - self.alpha * log_probs), dim=1))
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Record loss and entropy
        self.actor_loss_history.append(actor_loss.item())
        self.entropy_history.append(entropy.mean().item())
        
        #=====================================================================
        # Update Alpha (Temperature Parameter)
        #=====================================================================
        
        if self.auto_alpha and update_alpha:
            # Compute alpha loss
            alpha_loss = -torch.mean(self.log_alpha * (entropy.detach() + self.target_entropy))
            
            # Update alpha
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # Update alpha value
            self.alpha = self.log_alpha.exp().item()
            
            # Record alpha loss
            self.alpha_loss_history.append(alpha_loss.item())
        
        #=====================================================================
        # Update Target Networks
        #=====================================================================
        
        # Soft update of target networks
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss.item(),
            'entropy': entropy.mean().item(),
            'alpha': self.alpha
        }
    
    def save(self, directory):
        """Save the agent's parameters to the specified directory"""
        os.makedirs(directory, exist_ok=True)
        
        # Save networks
        torch.save(self.actor.state_dict(), os.path.join(directory, 'actor.pth'))
        torch.save(self.critic1.state_dict(), os.path.join(directory, 'critic1.pth'))
        torch.save(self.critic2.state_dict(), os.path.join(directory, 'critic2.pth'))
        torch.save(self.critic1_target.state_dict(), os.path.join(directory, 'critic1_target.pth'))
        torch.save(self.critic2_target.state_dict(), os.path.join(directory, 'critic2_target.pth'))
        
        # Save training history
        history = {
            'critic_loss': self.critic_loss_history,
            'actor_loss': self.actor_loss_history,
            'entropy': self.entropy_history
        }
        
        if self.auto_alpha:
            history['alpha_loss'] = self.alpha_loss_history
            torch.save(self.log_alpha, os.path.join(directory, 'log_alpha.pth'))
            
        with open(os.path.join(directory, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)
            
        logger.info(f"Saved agent to {directory}")
    
    def load(self, directory):
        """Load the agent's parameters from the specified directory"""
        # Load networks
        self.actor.load_state_dict(torch.load(os.path.join(directory, 'actor.pth')))
        self.critic1.load_state_dict(torch.load(os.path.join(directory, 'critic1.pth')))
        self.critic2.load_state_dict(torch.load(os.path.join(directory, 'critic2.pth')))
        self.critic1_target.load_state_dict(torch.load(os.path.join(directory, 'critic1_target.pth')))
        self.critic2_target.load_state_dict(torch.load(os.path.join(directory, 'critic2_target.pth')))
        
        # Load alpha if auto_alpha is enabled
        if self.auto_alpha and os.path.exists(os.path.join(directory, 'log_alpha.pth')):
            self.log_alpha = torch.load(os.path.join(directory, 'log_alpha.pth'))
            self.alpha = self.log_alpha.exp().item()
            
        # Load training history if available
        if os.path.exists(os.path.join(directory, 'history.pkl')):
            with open(os.path.join(directory, 'history.pkl'), 'rb') as f:
                history = pickle.load(f)
                
                self.critic_loss_history = history.get('critic_loss', [])
                self.actor_loss_history = history.get('actor_loss', [])
                self.entropy_history = history.get('entropy', [])
                
                if 'alpha_loss' in history:
                    self.alpha_loss_history = history['alpha_loss']
                    
        logger.info(f"Loaded agent from {directory}")


def evaluate_agent(agent, eval_data, num_episodes=100):
    """
    Evaluate the agent's performance on a validation set.
    
    Args:
        agent: SAC agent to evaluate
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
    with torch.no_grad():
        # Get agent's actions
        pred_actions = agent.select_action(pod_features, kv_hit_ratios, request_features, evaluate=True)
        
        # Get routing logits and values
        routing_logits, state_values = agent.actor(pod_features, kv_hit_ratios, request_features)
        
        # Calculate accuracy
        accuracy = (pred_actions == true_actions).float().mean().item()
        
        # Calculate reward for predicted actions
        # This is a simplification - in a real environment, we'd get the actual reward
        # Here we compare with the reward for the actual action taken
        true_reward = rewards.mean().item()
        
        # Get probabilities for visualization
        probs = F.softmax(routing_logits, dim=1)
        
    # Additional metrics
    metrics = {
        'accuracy': accuracy,
        'true_reward': true_reward,
        'avg_value': state_values.mean().item(),
        'probs': probs.cpu().numpy(),
        'pred_actions': pred_actions.cpu().numpy(),
        'true_actions': true_actions.cpu().numpy()
    }
    
    agent.actor.train()
    return metrics


def plot_training_metrics(agent, eval_metrics, output_dir):
   """Plot training metrics and save figures"""
   plt.figure(figsize=(15, 10))
   
   # Plot critic loss
   plt.subplot(2, 2, 1)
   plt.plot(agent.critic_loss_history)
   plt.title('Critic Loss')
   plt.xlabel('Updates')
   plt.ylabel('Loss')
   
   # Plot actor loss
   plt.subplot(2, 2, 2)
   plt.plot(agent.actor_loss_history)
   plt.title('Actor Loss')
   plt.xlabel('Updates')
   plt.ylabel('Loss')
   
   # Plot entropy
   plt.subplot(2, 2, 3)
   plt.plot(agent.entropy_history)
   plt.title('Policy Entropy')
   plt.xlabel('Updates')
   plt.ylabel('Entropy')
   
   # Plot alpha if auto_alpha is enabled
   if agent.auto_alpha and len(agent.alpha_loss_history) > 0:
       plt.subplot(2, 2, 4)
       plt.plot(agent.alpha_loss_history)
       plt.title('Alpha Loss')
       plt.xlabel('Updates')
       plt.ylabel('Loss')
   else:
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
       action_counts = np.bincount(last_eval['true_actions'], minlength=agent.action_dim)
       plt.bar(np.arange(agent.action_dim), action_counts)
       plt.title('True Action Distribution')
       plt.xlabel('Action')
       plt.ylabel('Count')
       
       plt.subplot(1, 2, 2)
       action_counts = np.bincount(last_eval['pred_actions'], minlength=agent.action_dim)
       plt.bar(np.arange(agent.action_dim), action_counts)
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


def train_agent(train_data, test_data=None, config=None):
   """Train a SAC agent on the given data"""
   # Default configuration
   if config is None:
       config = {
           'hidden_dim': 256,
           'batch_size': 64,
           'buffer_size': 100000,
           'learning_rate': 3e-4,
           'gamma': 0.99,
           'tau': 0.005,
           'alpha': 0.2,
           'auto_alpha': True,
           'num_epochs': 100,
           'updates_per_epoch': 1000,
           'eval_interval': 5,
           'prioritized_replay': True
       }
   
   # Create output directory for results
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   output_dir = os.path.join("results", f"sac_{timestamp}")
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
   
   # Create agent
   agent = SAC(
       state_dim=state_dim,
       action_dim=action_dim,
       hidden_dim=config['hidden_dim'],
       lr_actor=config['learning_rate'],
       lr_critic=config['learning_rate'],
       gamma=config['gamma'],
       tau=config['tau'],
       alpha=config['alpha'],
       auto_alpha=config['auto_alpha']
   )
   
   # Create replay buffer from offline data
   replay_buffer = ReplayBuffer.from_offline_data(
       train_data, buffer_size=config['buffer_size']
   )
   
   # Training loop
   logger.info("Starting training...")
   total_updates = 0
   eval_metrics = []
   
   for epoch in range(config['num_epochs']):
       epoch_start_time = time.time()
       epoch_critic_loss = 0
       epoch_actor_loss = 0
       epoch_entropy = 0
       epoch_updates = 0
       
       # Training updates for this epoch
       for i in range(config['updates_per_epoch']):
           update_metrics = agent.update(
               replay_buffer=replay_buffer,
               batch_size=config['batch_size'],
               update_alpha=(i % 10 == 0)  # Update alpha less frequently
           )
           
           total_updates += 1
           epoch_updates += 1
           epoch_critic_loss += update_metrics['critic_loss']
           epoch_actor_loss += update_metrics['actor_loss']
           epoch_entropy += update_metrics['entropy']
           
           # Log progress
           if i % 100 == 0:
               logger.info(f"Epoch {epoch+1}/{config['num_epochs']}, "
                         f"Update {i+1}/{config['updates_per_epoch']}, "
                         f"Critic Loss: {update_metrics['critic_loss']:.4f}, "
                         f"Actor Loss: {update_metrics['actor_loss']:.4f}, "
                         f"Entropy: {update_metrics['entropy']:.4f}, "
                         f"Alpha: {update_metrics['alpha']:.4f}")
       
       # Calculate average metrics for the epoch
       avg_critic_loss = epoch_critic_loss / epoch_updates
       avg_actor_loss = epoch_actor_loss / epoch_updates
       avg_entropy = epoch_entropy / epoch_updates
       epoch_time = time.time() - epoch_start_time
       
       logger.info(f"Epoch {epoch+1}/{config['num_epochs']} completed in {epoch_time:.2f}s, "
                 f"Avg Critic Loss: {avg_critic_loss:.4f}, "
                 f"Avg Actor Loss: {avg_actor_loss:.4f}, "
                 f"Avg Entropy: {avg_entropy:.4f}")
       
       # Evaluate agent
       if (epoch + 1) % config['eval_interval'] == 0 or epoch == config['num_epochs'] - 1:
           if test_data is not None:
               eval_data = test_data
               logger.info("Evaluating on test data...")
           else:
               eval_data = train_data
               logger.info("Evaluating on training data...")
           
           metrics = evaluate_agent(agent, eval_data)
           eval_metrics.append(metrics)
           
           logger.info(f"Evaluation: Accuracy: {metrics['accuracy']:.4f}, "
                     f"True Reward: {metrics['true_reward']:.4f}, "
                     f"Avg Value: {metrics['avg_value']:.4f}")
           
           # Save agent
           agent.save(os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}"))
           
           # Plot training metrics
           plot_training_metrics(agent, eval_metrics, output_dir)
   
   # Final save
   agent.save(os.path.join(output_dir, "final_model"))
   
   # Save evaluation metrics
   with open(os.path.join(output_dir, 'eval_metrics.pkl'), 'wb') as f:
       pickle.dump(eval_metrics, f)
   
   logger.info(f"Training completed, model saved to {output_dir}")
   return agent, eval_metrics, output_dir


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
        
    # Find all SAC model directories sorted by timestamp
    model_dirs = sorted(glob.glob(os.path.join(results_dir, "sac_*")), reverse=True)
    
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
    """Main function to train the LLM routing agent on all available data"""
    parser = argparse.ArgumentParser(description='Train LLM routing agent')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension of neural networks')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--buffer-size', type=int, default=100000, help='Size of replay buffer')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Target network update rate')
    parser.add_argument('--alpha', type=float, default=0.2, help='Initial entropy coefficient')
    parser.add_argument('--no-auto-alpha', action='store_true', help='Disable automatic entropy tuning')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--updates-per-epoch', type=int, default=1000, help='Number of updates per epoch')
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
        args.output_dir = os.path.join(results_dir, f"sac_{timestamp}")
    
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
        'buffer_size': args.buffer_size,
        'learning_rate': args.lr,
        'gamma': args.gamma,
        'tau': args.tau,
        'alpha': args.alpha,
        'auto_alpha': not args.no_auto_alpha,
        'num_epochs': args.epochs,
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
    
    # Create agent
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config['hidden_dim'],
        lr_actor=config['learning_rate'],
        lr_critic=config['learning_rate'],
        gamma=config['gamma'],
        tau=config['tau'],
        alpha=config['alpha'],
        auto_alpha=config['auto_alpha']
    )
    
    # Load previous model if available
    if previous_model_path:
        try:
            agent.load(previous_model_path)
            logger.info(f"Successfully loaded previous model from {previous_model_path}")
        except Exception as e:
            logger.error(f"Error loading previous model: {e}")
    
    # Create replay buffer from combined data
    replay_buffer = ReplayBuffer.from_offline_data(
        combined_data, buffer_size=config['buffer_size']
    )
    
    # Training loop
    logger.info("Starting training...")
    total_updates = 0
    eval_metrics = []
    
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        epoch_critic_loss = 0
        epoch_actor_loss = 0
        epoch_entropy = 0
        epoch_updates = 0
        
        # Training updates for this epoch
        for i in range(config['updates_per_epoch']):
            update_metrics = agent.update(
                replay_buffer=replay_buffer,
                batch_size=config['batch_size'],
                update_alpha=(i % 10 == 0)  # Update alpha less frequently
            )
            
            total_updates += 1
            epoch_updates += 1
            epoch_critic_loss += update_metrics['critic_loss']
            epoch_actor_loss += update_metrics['actor_loss']
            epoch_entropy += update_metrics['entropy']
            
            # Log progress
            if i % 100 == 0:
                logger.info(f"Epoch {epoch+1}/{config['num_epochs']}, "
                          f"Update {i+1}/{config['updates_per_epoch']}, "
                          f"Critic Loss: {update_metrics['critic_loss']:.4f}, "
                          f"Actor Loss: {update_metrics['actor_loss']:.4f}, "
                          f"Entropy: {update_metrics['entropy']:.4f}, "
                          f"Alpha: {update_metrics['alpha']:.4f}")
        
        # Calculate average metrics for the epoch
        avg_critic_loss = epoch_critic_loss / epoch_updates
        avg_actor_loss = epoch_actor_loss / epoch_updates
        avg_entropy = epoch_entropy / epoch_updates
        epoch_time = time.time() - epoch_start_time
        
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']} completed in {epoch_time:.2f}s, "
                  f"Avg Critic Loss: {avg_critic_loss:.4f}, "
                  f"Avg Actor Loss: {avg_actor_loss:.4f}, "
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
        print("Usage: python train_routing_agent.py /path/to/encoded_data_dir [--continue-training]")
        sys.exit(1)
    
    encoded_data_dir = sys.argv[1]
    train(encoded_data_dir)