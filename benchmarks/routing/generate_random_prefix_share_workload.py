#!/usr/bin/env python3
import random
import string
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

def generate_random_string(tokenizer, token_length: int) -> str:
    random_string = "".join(
        random.choices(string.ascii_letters + string.digits, k=token_length * 100)
    )
    tokenized_output = tokenizer.encode(random_string, add_special_tokens=False)[
        :token_length
    ]
    if len(tokenized_output) < token_length:
        tokenized_output = tokenized_output + [tokenizer.pad_token_id] * (
            token_length - len(tokenized_output)
        )
    decoded_string = tokenizer.decode(tokenized_output, skip_special_tokens=False)
    return decoded_string

def generate_unique_prefix(base_text, index):
    return str(index) + base_text[len(str(index)):]

def prepare_prompts(tokenizer, config):
    """
    Prepare prompts based on the provided configuration
    
    Args:
        tokenizer: The tokenizer to use
        config: Dictionary with prefix_length, suffix_length, num_samples_per_prefix, num_prefix
        
    Returns:
        Tuple of (all_prompts, tot_input_len, prompts_token_counts)
    """
    prefix_length = config["prefix_length"]
    suffix_length = config["suffix_length"]
    num_samples_per_prefix = config["num_samples_per_prefix"]
    num_prefix = config["num_prefix"]
    
    base_prefix = generate_random_string(tokenizer, prefix_length)
    tot_input_len = 0
    all_prompts = []
    prompts_token_counts = []  # Store token counts for each prompt
    
    for i in tqdm(range(num_prefix), desc=f"Preparing prompts for config {config['id']}"):
        unique_prefix = generate_unique_prefix(base_prefix, i)
        prompt_list = []
        token_count_list = []
        
        for j in range(num_samples_per_prefix):
            suffix = generate_random_string(tokenizer, suffix_length)
            prompt = unique_prefix + suffix
            
            # Count tokens
            token_count = len(tokenizer.encode(prompt))
            tot_input_len += token_count
            
            prompt_list.append(prompt)
            token_count_list.append(token_count)
        
        all_prompts.append(prompt_list)
        prompts_token_counts.append(token_count_list)
    
    return all_prompts, tot_input_len, prompts_token_counts

def calculate_prefix_sharing_ratio(tokenizer, all_prompts, prompts_token_counts, prefix_length):
    """
    Calculate the prefix sharing ratio in the entire workload based on token counts
    
    Prefix sharing ratio = (total tokens in shared prefixes) / (total tokens in all prompts)
    
    Args:
        tokenizer: The tokenizer to use
        all_prompts: List of prompt lists
        prompts_token_counts: List of list of token counts corresponding to all_prompts
        prefix_length: Length of the prefix in tokens
        
    Returns:
        Prefix sharing ratio (float)
    """
    # Flatten the token counts
    flat_token_counts = [
        token_count 
        for token_count_list in prompts_token_counts 
        for token_count in token_count_list
    ]
    total_prompt_tokens = sum(flat_token_counts)
    
    # Count the unique prefixes
    unique_prefixes = []
    for prompt_list in all_prompts:
        if prompt_list and len(prompt_list) > 0:
            # Take first prompt from each list to get the unique prefix
            first_prompt = prompt_list[0]
            prefix = first_prompt[:len(str(len(unique_prefixes))) + prefix_length]
            unique_prefixes.append(prefix)
    
    # Calculate token counts for each unique prefix
    unique_prefix_token_counts = [len(tokenizer.encode(prefix)) for prefix in unique_prefixes]
    total_shared_prefix_tokens = sum(unique_prefix_token_counts)
    
    # Calculate how many tokens would be needed if each prompt had its own prefix
    total_prefix_tokens_if_not_shared = 0
    for i, prompt_list in enumerate(all_prompts):
        prefix_token_count = unique_prefix_token_counts[i] if i < len(unique_prefix_token_counts) else 0
        total_prefix_tokens_if_not_shared += prefix_token_count * len(prompt_list)
    
    # Calculate tokens saved by sharing
    tokens_saved_by_sharing = total_prefix_tokens_if_not_shared - total_shared_prefix_tokens
    
    # Calculate sharing ratio
    sharing_ratio = tokens_saved_by_sharing / total_prompt_tokens
    
    return sharing_ratio

def generate_poisson_arrival_times(num_requests, rps, start_time=0):
    """
    Generate arrival times based on Poisson distribution
    
    Args:
        num_requests: Total number of requests
        rps: Requests per second (lambda parameter for Poisson)
        start_time: Starting timestamp (in milliseconds)
        
    Returns:
        List of timestamps in milliseconds
    """
    # For Poisson process, inter-arrival times follow exponential distribution
    # with mean = 1/lambda, where lambda = rps
    inter_arrival_times = np.random.exponential(scale=1.0/rps, size=num_requests)
    
    # Convert to cumulative times (in seconds)
    arrival_times = np.cumsum(inter_arrival_times)
    
    # Convert to millisecond timestamps and add start_time
    timestamps = [int(start_time + t * 1000) for t in arrival_times]
    
    return timestamps

def process_workload_configs(tokenizer, configs):
    """
    Process multiple workload configurations and combine them
    
    Args:
        tokenizer: The tokenizer to use
        configs: List of workload configuration dictionaries
        
    Returns:
        Dictionary with combined workload data
    """
    all_prompts_combined = []
    all_timestamps_combined = []
    total_tokens = 0
    config_stats = []
    
    # Variables to track overall prefix sharing
    total_prompts_count = 0
    total_unique_prefixes = 0
    total_prefix_tokens = 0
    
    # Variables for overall prefix sharing calculation
    all_prompts_for_sharing = []
    all_prompts_token_counts = []
    all_prefix_lengths = []
    
    current_time = 0  # Track current time for sequential workloads
    
    # Process each configuration
    for i, config in enumerate(configs):
        # Add an ID to the config for reference
        config["id"] = i+1
        
        # Generate prompts for this config
        prompts, tokens, token_counts = prepare_prompts(tokenizer, config)
        total_tokens += tokens
        
        # Calculate prefix sharing ratio for this config
        sharing_ratio = calculate_prefix_sharing_ratio(
            tokenizer, prompts, token_counts, config["prefix_length"]
        )
        
        # Flatten prompts for this config
        flat_prompts = [prompt for prompt_list in prompts for prompt in prompt_list]
        flat_token_counts = [count for count_list in token_counts for count in count_list]
        
        # Generate timestamps for this config
        rps = config.get("rps", 1)
        timestamps = generate_poisson_arrival_times(
            num_requests=len(flat_prompts),
            rps=rps,
            start_time=current_time
        )
        
        # Update current_time for next config
        if timestamps:
            current_time = max(timestamps) + 1000  # Add a 1-second gap between configs
        
        # Update overall prefix sharing tracking
        total_prompts_count += len(flat_prompts)
        
        # Store config data for overall prefix calculation
        all_prompts_for_sharing.extend(prompts)
        all_prompts_token_counts.extend(token_counts)
        all_prefix_lengths.extend([config["prefix_length"]] * len(prompts))
        
        # Store stats for this config
        total_num_req = config["num_prefix"] * config["num_samples_per_prefix"]
        total_duration = total_num_req / rps
        
        config_stats.append({
            "config_id": config["id"],
            "prefix_length": config["prefix_length"],
            "suffix_length": config["suffix_length"],
            "num_samples_per_prefix": config["num_samples_per_prefix"],
            "num_prefix": config["num_prefix"],
            "rps": rps,
            "num_requests": len(flat_prompts),
            "total_tokens": tokens,
            "total_duration": total_duration,
            "prefix_sharing_ratio": sharing_ratio,
            "start_time": min(timestamps) if timestamps else 0,
            "end_time": max(timestamps) if timestamps else 0
        })
        
        # Add to combined lists
        for j, prompt in enumerate(flat_prompts):
            all_prompts_combined.append({
                "config_id": config["id"],
                "prompt": prompt,
                "token_count": flat_token_counts[j],
                "timestamp": timestamps[j]
            })
    
    # Calculate overall prefix sharing ratio using the same token-based method
    # Since we may have different prefix lengths, we'll use a weighted average
    overall_sharing_ratio = 0
    if len(configs) == 1:
        # If there's only one config, use its sharing ratio
        overall_sharing_ratio = config_stats[0]["prefix_sharing_ratio"]
    else:
        # For multiple configs, calculate an overall ratio based on all prompts
        # This is more complex and would need special handling for different prefix lengths
        # For now, we'll use a weighted average based on token counts
        total_config_tokens = sum(cfg["total_tokens"] for cfg in config_stats)
        overall_sharing_ratio = sum(
            cfg["prefix_sharing_ratio"] * cfg["total_tokens"] / total_config_tokens
            for cfg in config_stats
        ) if total_config_tokens > 0 else 0
    
    # Sort combined data by timestamp
    all_prompts_combined.sort(key=lambda x: x["timestamp"])
    
    return {
        "prompts": all_prompts_combined,
        "stats": config_stats,
        "total_tokens": total_tokens,
        "overall_sharing_ratio": overall_sharing_ratio
    }

def save_to_jsonl(workload_data, output_file):
    """
    Save the combined workload to a JSONL file
    
    Args:
        workload_data: Dictionary with prompts and stats
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        for item in workload_data["prompts"]:
            entry = {
                "timestamp": item["timestamp"],
                "requests": [
                    {
                        "Prompt Length": item["token_count"],  # Use token count instead of character length
                        "Output Length": 8,  # Fixed value as per example
                        "prompt": item["prompt"]
                    }
                ]
            }
            f.write(json.dumps(entry) + '\n')

def save_stats(workload_data, stats_file):
    """
    Save workload statistics to a JSON file
    
    Args:
        workload_data: Dictionary with prompts and stats
        stats_file: Output file path for stats
    """
    with open(stats_file, 'w') as f:
        json.dump({
            "config_stats": workload_data["stats"],
            "num_tokens": workload_data["total_tokens"],
            "num_requests": len(workload_data["prompts"]),
            "overall_sharing_ratio": workload_data["overall_sharing_ratio"],
        }, f, indent=2)
    
    total_duration = 0
    total_num_requests = 0
    print("\nConfiguration details:")
    for cfg in workload_data["stats"]:
        num_req = cfg['num_prefix'] * cfg['num_samples_per_prefix']
        duration = num_req / cfg['rps']
        total_duration += duration
        total_num_requests += num_req
        print(f"Config {cfg['config_id']}:")
        print(f"  - Prefix length: {cfg['prefix_length']}")
        print(f"  - Suffix length: {cfg['suffix_length']}")
        print(f"  - Number of requests per prefix: {cfg['num_samples_per_prefix']}")
        print(f"  - Number of different prefixes: {cfg['num_prefix']}")
        print(f"  - RPS: {cfg['rps']}")
        print(f"  - Duration: {duration:.0f} seconds")
        print(f"  - Number of requests {cfg['num_requests']}")
        print(f"  - Prefix sharing ratio: {cfg['prefix_sharing_ratio']*100:.2f}%")
        print(f"  - Time range: {int(cfg['start_time']/1000)}s to {int(cfg['end_time']/1000)}s")

    print("\nWorkload Summary:")
    print(f"Total number of requests: {total_num_requests}")
    print(f"Total duration: {total_duration:.0f} seconds")
    print(f"Total prompts: {len(workload_data['prompts'])}")
    print(f"Total tokens: {workload_data['total_tokens']}")
    print(f"Overall prefix sharing ratio: {workload_data['overall_sharing_ratio']*100:.2f}%")

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    
    # Define workload configurations
    prefix_workload_configs = [
        {
            "prefix_length": 3968,
            "suffix_length": 128,
            "num_samples_per_prefix": 32,
            "num_prefix": 10,
            "rps": 5
        },
        # Uncomment to add more configurations
        # {
        #     "prefix_length": 2048,
        #     "suffix_length": 128,
        #     "num_samples_per_prefix": 32,
        #     "num_prefix": 10,
        #     "rps": 5
        # }
    ]
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "hf-internal-testing/llama-tokenizer", 
        legacy=True,
        model_max_length=4096,  # Increased to handle longer prefixes
        padding_side="right",
        truncation_side="right",
        use_fast=True
    )
    base_filename = "prefix-share-workload"
    print("Generating multi-configuration workload...")
    workload_data = process_workload_configs(tokenizer, prefix_workload_configs)
    
    # Save results
    output_file = f"{base_filename}.jsonl"
    stats_file = f"{base_filename}-stats.json"
    save_to_jsonl(workload_data, output_file)
    save_stats(workload_data, stats_file)
    print(f"Saving workload statistics to {stats_file}")
    print(f"Saving workload traces to {output_file}")