import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import json
import os
from tqdm import tqdm

# Configuration
class Config:
    def __init__(self, input_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = None  # Will be set after data loading
        self.action_dim = None  # Will be set after data loading
        self.hidden_dim = 256
        self.learning_rate = 0.0003
        self.gamma = 0.99  # Discount factor
        self.batch_size = 256
        self.num_epochs = 200
        self.alpha = 0.1  # CQL regularization parameter - higher more conservative
        self.tau = 0.005  # Target network update rate
        self.input_dir = input_dir
        self.model_save_path = f"{input_dir}/models/cql_model.pt"

def create_essential_relative_features(df, pod_ids, metrics=None):
    """
    Create essential relative features for key metrics
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with pod metrics
    pod_ids : list
        List of pod identifiers
    metrics : list, optional
        List of metrics to process. If None, will detect automatically.
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with added relative features
    """
    if metrics is None:
        # Detect metrics by finding columns that appear for multiple pods
        all_cols = set(df.columns)
        metrics = set()
        
        for pod_id in pod_ids:
            pod_prefix = f"pod_{pod_id}_"
            pod_cols = [col[len(pod_prefix):] for col in all_cols if col.startswith(pod_prefix)]
            
            if not metrics:
                metrics = set(pod_cols)
            else:
                metrics = metrics.intersection(pod_cols)
        
        metrics = list(metrics)
        print(f"Detected {len(metrics)} common metrics across pods: {metrics}")
    
    # Process each metric
    for metric in metrics:
        # Get all pod columns for this metric
        pod_metric_cols = [f"pod_{p}_{metric}" for p in pod_ids if f"pod_{p}_{metric}" in df.columns]
        
        if len(pod_metric_cols) <= 1:
            continue  # Skip if not enough pods have this metric
        
        # Calculate total and normalized values (percentage of cluster total)
        df[f"total_{metric}"] = df[pod_metric_cols].sum(axis=1)
        
        for col in pod_metric_cols:
            pod_id = col.split("_")[1]
            df[f"pct_{pod_id}_{metric}"] = df[col] / (df[f"total_{metric}"] + 1e-6)
        
        # Calculate ranks (1 = lowest value, which is typically better for load metrics)
        ranks = df[pod_metric_cols].rank(axis=1)
        for col in pod_metric_cols:
            pod_id = col.split("_")[1]
            df[f"rank_{pod_id}_{metric}"] = ranks[col]
            
            # Normalized rank (0-1 scale)
            df[f"norm_rank_{pod_id}_{metric}"] = (df[f"rank_{pod_id}_{metric}"] - 1) / (len(pod_metric_cols) - 1)
    
    return df

def load_and_preprocess_data(data_path, mapping_path, config):
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Load the pod mapping
    with open(mapping_path, 'r') as f:
        mapping_info = json.load(f)
    
    print(f"Dataset shape before feature engineering: {df.shape}")
    
    # Extract pod IDs from the mapping info
    pod_ids = list(mapping_info['pod_to_index'].keys())
    print(f"Found {len(pod_ids)} pods from mapping")
    
    # Create relative features
    df = create_essential_relative_features(df, pod_ids)
    
    print(f"Dataset shape after feature engineering: {df.shape}")
    
    # Define which columns are features and which are metadata
    metadata_cols = ['request_id', 'request_start_time', 'request_end_time', 
                     'selected_pod', 'action', 'reward', 'ttft_reward', 'tpot_reward',
                     'avg_tpot_slo_satisfied', 'avg_ttft_slo_satisfied', 
                     'ttft_normalized', 'tpot_normalized']
    
    # Identify GPU model columns
    gpu_model_cols = [col for col in df.columns if 'gpu_model' in col]
    
    # Numeric feature columns (exclude metadata and handle GPU model columns as before)
    numeric_feature_cols = [col for col in df.columns if col not in metadata_cols and col not in gpu_model_cols]
    
    # Handle one-hot encoding of GPU models as in your original code
    unique_gpu_models = set()
    for col in gpu_model_cols:
        unique_gpu_models.update(df[col].unique())
    
    # One-hot encode GPU models
    encoded_gpu_features = {}
    for col in gpu_model_cols:
        pod_id = col.replace('_gpu_model', '')
        for gpu_model in unique_gpu_models:
            feature_name = f"{pod_id}_gpu_{gpu_model.replace('-', '_')}"
            encoded_gpu_features[feature_name] = (df[col] == gpu_model).astype(int)
    
    # Create a DataFrame with encoded GPU features
    encoded_df = pd.DataFrame(encoded_gpu_features)
    
    # Combine with numeric features
    combined_df = pd.concat([df[numeric_feature_cols], encoded_df], axis=1)
    
    # Get list of all feature columns after one-hot encoding
    all_feature_cols = list(combined_df.columns)
    
    print(f"Using {len(numeric_feature_cols)} numeric features and {len(encoded_gpu_features)} encoded GPU features")
    
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(combined_df.values)
    
    # Set state and action dimensions
    config.state_dim = X.shape[1]
    config.action_dim = len(mapping_info['pod_to_index'])
    
    print(f"State dimension: {config.state_dim}")
    print(f"Action dimension: {config.action_dim}")
    
    # Extract actions and rewards
    actions = df['action'].values
    rewards = df['reward'].values
    
    # Convert to PyTorch tensors
    states = torch.FloatTensor(X)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    
    print(f"Data shapes - States: {states.shape}, Actions: {actions.shape}, Rewards: {rewards.shape}")
    
    return states, actions, rewards, mapping_info, all_feature_cols, scaler

# Q-Network model
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Conservative Q-Learning
class CQLAgent:
    def __init__(self, config):
        self.device = config.device
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha  # Set to 0.01 for less conservative behavior
        
        # Q-Networks
        self.q_network = QNetwork(self.state_dim, self.action_dim, config.hidden_dim).to(self.device)
        self.target_q_network = QNetwork(self.state_dim, self.action_dim, config.hidden_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
    def update(self, states, actions, rewards, next_states, dones):
        # Get current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values from target network
        next_q = self.target_q_network(next_states).max(1)[0].unsqueeze(1).detach()
        
        # Compute target Q values
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Standard TD loss
        td_loss = nn.MSELoss()(current_q, target_q)
        
        # CQL regularization: minimize Q values for random actions
        random_actions = torch.randint(0, self.action_dim, (states.shape[0], 10)).to(self.device)
        random_q = torch.gather(
            self.q_network(states).unsqueeze(1).repeat(1, 10, 1),
            2,
            random_actions.unsqueeze(2)
        ).mean(1)
        
        # Minimize Q values for random actions, maximize Q values for dataset actions
        cql_loss = random_q.mean() - current_q.mean()
        
        # Total loss with reduced alpha
        loss = td_loss + self.alpha * cql_loss
        
        # Update Q-Network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return loss.item(), td_loss.item(), cql_loss.item()
    
    def select_action(self, state):
        with torch.no_grad():
            q_values = self.q_network(state)
            return q_values.argmax(dim=1).item()
        
    def save(self, path):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_q_network': self.target_q_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

# Training function
def train_cql(agent, states, actions, rewards, config):
    # Create dataset of (s, a, r, s') tuples
    dataset_size = states.shape[0] - 1
    state_dataset = states[:dataset_size]
    action_dataset = actions[:dataset_size]
    reward_dataset = rewards[:dataset_size]
    next_state_dataset = states[1:dataset_size+1]
    done_dataset = torch.zeros((dataset_size, 1))  # Assuming no terminal states in the dataset
    
    # Create DataLoader
    dataset = TensorDataset(state_dataset, action_dataset, reward_dataset, next_state_dataset, done_dataset)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Training loop
    losses = []
    td_losses = []
    cql_losses = []
    
    for epoch in range(config.num_epochs):
        epoch_loss = 0
        epoch_td_loss = 0
        epoch_cql_loss = 0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [b.to(config.device) for b in batch]
            
            loss, td_loss, cql_loss = agent.update(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)
            
            epoch_loss += loss
            epoch_td_loss += td_loss
            epoch_cql_loss += cql_loss
            num_batches += 1
        
        # Average loss for the epoch
        avg_loss = epoch_loss / num_batches
        avg_td_loss = epoch_td_loss / num_batches
        avg_cql_loss = epoch_cql_loss / num_batches
        
        losses.append(avg_loss)
        td_losses.append(avg_td_loss)
        cql_losses.append(avg_cql_loss)
        
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {avg_loss:.6f}, TD Loss: {avg_td_loss:.6f}, CQL Loss: {avg_cql_loss:.6f}")
        
        # Save model periodically
        if (epoch + 1) % 20 == 0 or epoch == config.num_epochs - 1:
            os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
            agent.save(config.model_save_path)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(td_losses)
    plt.title('TD Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 3)
    plt.plot(cql_losses)
    plt.title('CQL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(f'{config.input_dir}/training_curves.pdf')
    plt.show()
    
    return losses, td_losses, cql_losses

# Evaluation function - focusing on reward metrics rather than accuracy
def evaluate(agent, states, actions, rewards, mapping_info, config):
    agent.q_network.eval()
    
    with torch.no_grad():
        q_values = agent.q_network(states.to(config.device))
        predicted_actions = q_values.argmax(dim=1).cpu().numpy()
        actual_actions = actions.cpu().numpy()
        
        # Calculate SLO statistics based on the model's predictions
        expected_rewards = q_values.max(dim=1)[0].mean().item()
        actual_rewards = rewards.mean().item()
        
        # Analyze pod selection patterns
        pod_selections = {}
        index_to_pod = mapping_info['index_to_pod']
        for action in predicted_actions:
            pod_id = index_to_pod.get(str(action), str(action))
            pod_selections[pod_id] = pod_selections.get(pod_id, 0) + 1
        
        # Sort by frequency
        pod_selections = {k: v for k, v in sorted(pod_selections.items(), 
                                                 key=lambda item: item[1], 
                                                 reverse=True)}
        
        # Calculate improvements over dataset decisions
        q_values_of_selected = q_values.gather(1, actions.unsqueeze(1).to(config.device))
        q_values_of_best = q_values.max(dim=1)[0].unsqueeze(1)
        improvement = (q_values_of_best - q_values_of_selected).mean().item()
        
        # Calculate how often model makes different decisions than dataset
        deviation_rate = (predicted_actions != actual_actions).mean()
        
        return {
            'deviation_rate': deviation_rate,
            'expected_rewards': expected_rewards,
            'actual_rewards': actual_rewards,
            'pod_selections': pod_selections,
            'improvement': improvement
        }

# Feature importance analysis
def analyze_feature_importance(agent, scaler, feature_names):
    # Get weights from the first layer
    weights = agent.q_network.fc1.weight.data.cpu().numpy()
    
    # Calculate feature importance based on the absolute sum of weights for each feature
    importance = np.abs(weights).sum(axis=0)
    
    # Scale importance relative to feature scales (if applicable)
    if scaler is not None and hasattr(scaler, 'scale_') and len(scaler.scale_) == len(importance):
        # Get feature scales (standard deviations)
        feature_scales = scaler.scale_
        # Adjust importance by feature scale
        importance = importance * feature_scales
    
    # Normalize to sum to 100%
    importance = 100.0 * (importance / importance.sum())
    
    # Create a dictionary of feature importance
    feature_importance = {name: imp for name, imp in zip(feature_names, importance)}
    
    # Sort by importance
    feature_importance = {k: v for k, v in sorted(feature_importance.items(), 
                                                 key=lambda item: item[1], 
                                                 reverse=True)}
    
    return feature_importance
def analyze_feature_correlations(states, feature_names):
    """Analyze correlations between features"""
    # Convert to numpy for correlation calculation
    states_np = states.cpu().numpy()
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(states_np.T)
    
    # Find highly correlated features with pod_10_0_0_4_prefill_tokens
    # First, find the index of this feature
    target_feature = "pod_10_0_0_4_prefill_tokens"
    if target_feature in feature_names:
        target_idx = feature_names.index(target_feature)
        
        # Get correlations with this feature
        correlations = []
        for i, name in enumerate(feature_names):
            if i != target_idx:
                correlations.append((name, corr_matrix[target_idx, i]))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Return top correlations
        return correlations[:10]
    else:
        return "Target feature not found"

def improved_feature_importance(agent, states, actions, feature_names):
    """Permutation-based feature importance analysis"""
    # Get baseline performance
    agent.q_network.eval()
    with torch.no_grad():
        baseline_q = agent.q_network(states).gather(1, actions.unsqueeze(1)).mean().item()
    
    # Calculate importance by permuting each feature
    importance = []
    for i in range(states.shape[1]):
        # Create a copy with the i-th feature permuted
        permuted_states = states.clone()
        permuted_states[:, i] = permuted_states[torch.randperm(states.shape[0]), i]
        
        # Measure impact on Q-values
        with torch.no_grad():
            permuted_q = agent.q_network(permuted_states).gather(1, actions.unsqueeze(1)).mean().item()
        
        # Importance is the drop in performance
        feature_importance = baseline_q - permuted_q
        importance.append(feature_importance)
    
    # Convert to numpy and normalize
    importance = np.array(importance)
    importance = np.abs(importance)  # Take absolute value
    if importance.sum() > 0:  # Avoid division by zero
        importance = 100.0 * (importance / importance.sum())
    
    # Create dictionary mapping features to importance
    feature_importance_dict = {}
    for name, imp in zip(feature_names, importance):
        feature_importance_dict[name] = imp
    
    # Sort by importance
    sorted_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    return dict(sorted_importance)  # Return as dictionary

# Main function
def main(input_dir):
    # Data paths
    train_data_path = f"{input_dir}/train_data.csv"
    test_data_path = f"{input_dir}/test_data.csv"
    mapping_path = f"{input_dir}/processed_dataset_mapping.json"
    
    config = Config(input_dir)

    # Load and preprocess data
    train_states, train_actions, train_rewards, mapping_info, feature_cols, scaler = load_and_preprocess_data(train_data_path, mapping_path, config)
    test_states, test_actions, test_rewards, _, _, _ = load_and_preprocess_data(test_data_path, mapping_path, config)

    # Analyze feature correlations before training
    print("\nAnalyzing feature correlations:")
    correlations = analyze_feature_correlations(train_states, feature_cols)
    if isinstance(correlations, list):
        print("Top correlations with pod_10_0_0_4_prefill_tokens:")
        for feature, corr in correlations:
            print(f"  {feature}: {corr:.4f}")
    else:
        print(correlations)
    
    # Initialize agent
    agent = CQLAgent(config)
    
    # Train agent
    train_cql(agent, train_states, train_actions, train_rewards, config)
    
    # Evaluate agent
    print("\nEvaluating on training data:")
    train_results = evaluate(agent, train_states, train_actions, train_rewards, mapping_info, config)
    print(f"Deviation rate: {train_results['deviation_rate']:.4f}")
    print(f"Expected rewards: {train_results['expected_rewards']:.4f}")
    print(f"Actual rewards: {train_results['actual_rewards']:.4f}")
    print(f"Potential improvement: {train_results['improvement']:.4f}")
    
    print("\nPod selection distribution:")
    for pod, count in list(train_results['pod_selections'].items())[:5]:
        print(f"  {pod}: {count} ({count/len(train_states)*100:.2f}%)")
    
    print("\nEvaluating on test data:")
    test_results = evaluate(agent, test_states, test_actions, test_rewards, mapping_info, config)
    print(f"Deviation rate: {test_results['deviation_rate']:.4f}")
    print(f"Expected rewards: {test_results['expected_rewards']:.4f}")
    print(f"Actual rewards: {test_results['actual_rewards']:.4f}")
    print(f"Potential improvement: {test_results['improvement']:.4f}")
    
    # # Standard feature importance
    # print("\nStandard feature importance analysis:")
    # feature_importance = analyze_feature_importance(agent, scaler, feature_cols)
    # for feature, importance in list(feature_importance.items())[:10]:
    #     print(f"  {feature}: {importance:.2f}%")
    # # Standard importance
    # plt.subplot(2, 1, 1)
    # top_features = dict(list(feature_importance.items())[:20])
    # plt.barh(list(top_features.keys()), list(top_features.values()))
    # plt.xlabel('Importance (%)')
    # plt.title('Top 20 Feature Importance (Standard Method)')
    
    # Improved feature importance
    print("\nImproved permutation-based feature importance:")
    improved_importance = improved_feature_importance(agent, train_states.to(config.device), train_actions.to(config.device), feature_cols)
    for feature, importance in list(improved_importance.items())[:10]:
        print(f"  {feature}: {importance:.2f}%")
    
    # Visualize feature importance (using both methods)
    plt.figure(figsize=(12, 6))
    improved_top_features = dict(list(improved_importance.items())[:20])
    plt.barh(list(improved_top_features.keys()), list(improved_top_features.values()))
    plt.xlabel('Importance (%)')
    plt.title('Top 20 Feature Importance (Permutation Method)')
    
    plt.tight_layout()
    plt.savefig(f'{config.input_dir}/feature_importance_comparison.pdf')
    plt.show()
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python training.py <input_dir>")
        sys.exit(1)
    input_dir = sys.argv[1]
    main(input_dir)