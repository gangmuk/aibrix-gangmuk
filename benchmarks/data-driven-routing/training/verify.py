import numpy as np
import torch
import pickle
import json
import os
import matplotlib.pyplot as plt

def verify_encoding(data_dir):
    """Verify that the data is correctly encoded with all techniques."""
    print(f"Verifying data in: {data_dir}")
    
    # Change to the data directory
    os.chdir(data_dir)
    
    # 1. Check available files
    files = os.listdir('.')
    print(f"Found {len(files)} files")
    
    # 2. Load tensor dataset
    tensor_data = torch.load("tensor_dataset.pt")
    print("\n=== Tensor Dataset ===")
    print("Keys:", list(tensor_data.keys()))
    
    for key, tensor in tensor_data.items():
        print(f"{key}: {tensor.shape}")
    
    # 3. Check positional encodings
    positional_encodings = np.load("positional_encodings.npy")
    pod_features = np.load("pod_features.npy")
    
    print("\n=== Positional Encodings ===")
    print(f"Positional encodings shape: {positional_encodings.shape}")
    print(f"Pod features shape: {pod_features.shape}")
    print(f"Positional encoding stats: min={positional_encodings.min():.4f}, max={positional_encodings.max():.4f}, mean={positional_encodings.mean():.4f}")
    
    # 4. Check feature timing
    feature_timing = pickle.load(open("feature_timing.pkl", "rb"))
    
    print("\n=== Feature Timing ===")
    historical = [f for f, t in feature_timing.items() if t == 'historical']
    current = [f for f, t in feature_timing.items() if t == 'current']
    
    print(f"Historical features: {len(historical)}")
    print(f"Current features: {len(current)}")
    
    # Check last_second features
    last_second_features = [f for f in feature_timing.keys() if 'last_second' in f]
    correctly_classified = all(feature_timing[f] == 'historical' for f in last_second_features)
    print(f"Last_second features correctly classified: {correctly_classified}")
    
    # 5. Check staleness features
    pod_features_with_staleness = np.load("pod_features_with_staleness.npy")
    
    print("\n=== Staleness Features ===")
    print(f"Pod features shape: {pod_features.shape}")
    print(f"Pod features with staleness shape: {pod_features_with_staleness.shape}")
    
    extra_dims = pod_features_with_staleness.shape[2] - pod_features.shape[2]
    print(f"Extra dimensions for staleness: {extra_dims}")
    
    if extra_dims > 0:
        staleness = pod_features_with_staleness[:, :, -1]
        print(f"Staleness feature stats: min={staleness.min():.4f}, max={staleness.max():.4f}, mean={staleness.mean():.4f}")
        
        # Plot histogram of staleness values
        plt.figure(figsize=(8, 4))
        plt.hist(staleness.flatten(), bins=20)
        plt.title("Distribution of Staleness Values")
        plt.xlabel("Staleness")
        plt.ylabel("Frequency")
        plt.savefig("staleness_distribution.png")
        print(f"Saved staleness distribution plot to staleness_distribution.png")
    
    # 6. Check cross-attention inputs
    cross_attention_inputs = pickle.load(open("cross_attention_inputs.pkl", "rb"))
    kv_hit_ratios = np.load("kv_hit_ratios.npy")
    
    print("\n=== Cross-Attention Inputs ===")
    print("Cross-attention keys:", list(cross_attention_inputs.keys()))
    print(f"Query shape: {cross_attention_inputs['query'].shape}")
    print(f"Key/Value shape: {cross_attention_inputs['key_value'].shape}")
    
    kv_match = np.allclose(cross_attention_inputs['key_value'], kv_hit_ratios)
    print(f"Key/Value matches KV hit ratios: {kv_match}")
    
    # 7. Check interaction features
    if "interaction_features.npy" in files:
        interaction_features = np.load("interaction_features.npy")
        request_features = np.load("request_features.npy")
        
        print("\n=== Interaction Features ===")
        print(f"Request features shape: {request_features.shape}")
        print(f"Interaction features shape: {interaction_features.shape}")
        
        if len(interaction_features.shape) == 3:
            num_samples, num_pods, interaction_dim = interaction_features.shape
            _, request_dim = request_features.shape
            
            match = (interaction_dim == request_dim)
            print(f"Interaction dimension ({interaction_dim}) matches request dimension ({request_dim}): {match}")
    else:
        print("\n=== Interaction Features ===")
        print("Interaction features file not found")


    # Add this to verify.py
    print("\nChecking for last_second features in pod_features_list:")
    pod_features_list = pickle.load(open("pod_features_list.pkl", "rb"))
    last_second_features = [f for f in pod_features_list if any(pattern in f for pattern in 
                            ['last_second', 'previous', 'historical', 'avg_'])]
    print(f"Found {len(last_second_features)} potential historical features: {last_second_features}")
    
    # 8. Check metadata
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
    
    print("\n=== Metadata ===")
    print("Sections:", list(metadata.keys()))
    print("Feature dimensions:", metadata["feature_dimensions"])
    print("Processing info:", metadata["processing_info"])
    print("Action distribution:", metadata["action_distribution"])
    
    # 9. Final summary
    print("\n=== Verification Summary ===")
    
    checks = {
        "Positional Encodings": positional_encodings.shape[2] > 0,
        "Staleness Features": extra_dims > 0,
        "Cross-Attention Setup": 'query' in cross_attention_inputs and 'key_value' in cross_attention_inputs,
        "Historical Features Detected": len(historical) > 0,
        "Last-Second Features Classified": correctly_classified,
        "Metadata Complete": len(metadata["feature_dimensions"]) >= 3
    }
    
    all_passed = True
    for check, passed in checks.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{check}: {status}")
        all_passed = all_passed and passed
    
    return all_passed

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify LLM routing data encoding")
    parser.add_argument("data_dir", help="Directory containing the processed data")
    
    args = parser.parse_args()
    
    success = verify_encoding(args.data_dir)
    
    if success:
        print("\nAll verification checks passed! Your data is correctly encoded with all techniques.")
    else:
        print("\nSome verification checks failed. Please review the output above.")