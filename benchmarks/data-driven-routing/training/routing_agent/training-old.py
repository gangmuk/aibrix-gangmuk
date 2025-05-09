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
    def __init__(self, output_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = None  # Will be set after data loading
        self.action_dim = None  # Will be set after data loading
        self.hidden_dim = 256
        self.learning_rate = 0.0003
        self.gamma = 0.99  # Discount factor
        self.batch_size = 256
        self.num_epochs = 1000
        self.alpha = 0.01  # CQL regularization parameter - higher more conservative
        self.tau = 0.005  # Target network update rate
        self.output_dir = output_dir
        self.model_save_path = f"{output_dir}/models/cql_model.pt"

def create_train_test_split(processed_df, train_ratio, output_dir):
    """
    Split the processed dataset into training and testing sets
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Shuffle the dataset
    shuffled_df = processed_df.sample(frac=1, random_state=42)
    
    # Split into train and test
    train_size = int(len(shuffled_df) * train_ratio)
    train_df = shuffled_df[:train_size]
    test_df = shuffled_df[train_size:]
    
    # Save the splits
    train_file = os.path.join(output_dir, 'train_data.csv')
    test_file = os.path.join(output_dir, 'test_data.csv')
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"Training set shape: {train_df.shape}, saved to {train_file}")
    print(f"Testing set shape: {test_df.shape}, saved to {test_file}")
    
    return train_file, test_file, train_df, test_df

def create_essential_relative_features(df, pod_ids, metrics=None):
    if metrics is None:
        # Detect metrics by finding columns that appear for multiple pods
        all_cols = set(df.columns)
        metrics = set()

        print(f"all_cols: {all_cols}")
        
        for pod_id in pod_ids:
            pod_prefix = f"pod_{pod_id}_"
            pod_cols = [col[len(pod_prefix):] for col in all_cols if col.startswith(pod_prefix)]
            print(f"pod_id: {pod_id}, pod_cols: {pod_cols}")
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
    
    print(f"Dataset shape: {df.shape}")
    
    # Extract pod IDs from the mapping info
    pod_ids = list(mapping_info['pod_to_index'].keys())
    pod_ids = [pod_id.replace(".", "_") for pod_id in pod_ids]  # Replace '.' with '_' for compatibility
    print(f"Found {len(pod_ids)} pods from mapping")

    # THIS IS WHERE create_essential_relative_features SHOULD BE USED:
    df = create_essential_relative_features(df, pod_ids)
    print(f"Created relative features, new shape: {df.shape}")
    
    # Define which columns are features and which are metadata
    metadata_cols = ['request_id', 'request_start_time', 'request_end_time', 
                     'selected_pod', 'input_tokens', 'output_tokens', 'total_tokens',
                     'ttft', 'avg_tpot', 'e2e_latency', 'action', 'reward', 
                     'ttft_reward', 'tpot_reward', 'avg_tpot_slo_satisfied', 
                     'avg_ttft_slo_satisfied', 'ttft_normalized', 'tpot_normalized']
    
    # Identify columns by patterns based on the new structure
    pct_cols = [col for col in df.columns if col.startswith('pct_')]
    total_cols = [col for col in df.columns if col.startswith('total_') and col != 'total_tokens']
    
    # All feature columns (excluding metadata)
    feature_cols = pct_cols + total_cols
    
    print(f"Using {len(feature_cols)} feature columns")
    print(f"Feature columns sample: {feature_cols[:5]}")
    
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].values)
    
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
    
    return states, actions, rewards, mapping_info, feature_cols, scaler

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
    plt.savefig(f'{config.output_dir}/training_curves.pdf')
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
def main(training_input_file, input_dir, output_dir):
    # Data paths
    df = pd.read_csv(training_input_file)
    train_data_path, test_data_path, _, _ = create_train_test_split(df, train_ratio=0.8, output_dir=output_dir)
    
    config = Config(output_dir)

    # Load and preprocess data
    mapping_path = f"{input_dir}/processed_dataset_mapping.json"
    train_states, train_actions, train_rewards, mapping_info, feature_cols, scaler = load_and_preprocess_data(train_data_path, mapping_path, config)
    test_states, test_actions, test_rewards, _, _, _ = load_and_preprocess_data(test_data_path, mapping_path, config)

    # Initialize agent
    agent = CQLAgent(config)
    
    # Train agent
    train_cql(agent, train_states, train_actions, train_rewards, config)
    
    # # Evaluate agent
    print("\nEvaluating on test data:")
    test_results = evaluate(agent, test_states, test_actions, test_rewards, mapping_info, config)
    print(f"Deviation rate: {test_results['deviation_rate']:.4f}")
    print(f"Expected rewards: {test_results['expected_rewards']:.4f}")
    print(f"Actual rewards: {test_results['actual_rewards']:.4f}")
    print(f"Potential improvement: {test_results['improvement']:.4f}")
    
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
    plt.savefig(f'{config.output_dir}/feature_importance_comparison.pdf')
    plt.show()
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python training.py <training_input_file>")
        sys.exit(1)
    training_input_file = sys.argv[1]
    if not os.path.exists(training_input_file):
        print(f"Training input file {training_input_file} does not exist.")
        sys.exit(1)
    input_dir = os.path.dirname(training_input_file)
    output_dir = f"{input_dir}/routing_model_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist.")
        sys.exit(1)
    main(training_input_file, input_dir, output_dir)