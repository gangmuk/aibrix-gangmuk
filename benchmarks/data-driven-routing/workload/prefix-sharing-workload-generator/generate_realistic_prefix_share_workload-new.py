#!/usr/bin/env python3
import random
import string
import json
import numpy as np
from tqdm import tqdm
import time
import os
import re
import concurrent.futures

# A simple tokenizer implementation that doesn't require downloading
class SimpleTokenizer:
    def __init__(self):
        """
        Simple whitespace and punctuation tokenizer
        """
        self.token_pattern = re.compile(r'\w+|[^\w\s]')
    
    def encode(self, text):
        """
        Simple encoding function that returns token IDs
        """
        if not text:
            return []
        
        # Simple tokenization strategy: split by whitespace and punctuation
        tokens = self.token_pattern.findall(text)
        # For our purposes, just use the position as a token ID
        token_ids = list(range(len(tokens)))
        return token_ids
    
    def decode(self, token_ids, skip_special_tokens=True):
        """
        Since we're not really decoding to get the original text back
        (we're just using the tokenizer to count tokens),
        this implementation will just join tokens with spaces.
        """
        # For our simple purpose, we'll just return a dummy string of appropriate length
        return " ".join(["token"] * len(token_ids))

# A collection of realistic text templates for generating prompts
REALISTIC_TEMPLATES = [
    # Question-answering templates
    "Can you explain how {topic} works in simple terms?",
    "What are the main differences between {topic_a} and {topic_b}?",
    "I need to understand {topic} for my {purpose}. Can you help?",
    "Could you provide a step-by-step guide on how to {action}?",
    "What are the best practices for {activity} in {field}?",
    
    # Creative writing templates
    "Write a short story about {character} who discovers {object} in {location}.",
    "Create a poem about {theme} using the style of {author}.",
    "Describe a scene where {character_a} meets {character_b} at {location}.",
    "Write a dialogue between {character_a} and {character_b} discussing {topic}.",
    "Develop a plot outline for a story about {theme} set in {setting}.",
    
    # Professional content templates
    "Draft an email to {recipient} regarding {subject}.",
    "Write a product description for {product} highlighting its {feature}.",
    "Create a marketing copy for {service} targeting {audience}.",
    "Compose a social media post announcing {event} for {platform}.",
    "Draft a professional bio for {person} who specializes in {expertise}.",
    
    # Information retrieval templates
    "Summarize the key points about {topic} in bullet points.",
    "What are the latest developments in {field} as of 2024?",
    "Provide a comparison table of {item_a}, {item_b}, and {item_c}.",
    "What are the pros and cons of {subject}?",
    "Give me 5 tips for improving {skill}."
]

# Domain-specific vocabulary to make the prompts more realistic
TOPICS = [
    "machine learning", "artificial intelligence", "neural networks", "deep learning", 
    "natural language processing", "computer vision", "reinforcement learning",
    "blockchain", "cryptocurrency", "smart contracts", "decentralized finance",
    "cloud computing", "serverless architecture", "microservices", "containerization",
    "cybersecurity", "ethical hacking", "network security", "encryption",
    "data science", "big data", "data visualization", "statistical analysis",
    "software development", "agile methodology", "DevOps", "continuous integration"
]

ACTIONS = [
    "deploy a machine learning model", "optimize database queries", "secure a web application",
    "build a responsive website", "create a mobile app", "implement an API",
    "analyze data using Python", "set up a cloud infrastructure", "configure a firewall",
    "develop a recommendation system", "train a neural network", "perform sentiment analysis"
]

CHARACTERS = [
    "a software engineer", "a data scientist", "a startup founder", "a cybersecurity expert",
    "an AI researcher", "a product manager", "a UX designer", "a digital nomad",
    "a tech entrepreneur", "a blockchain developer", "a virtual reality designer"
]

LOCATIONS = [
    "Silicon Valley", "a tech conference", "a coworking space", "a virtual reality world",
    "a futuristic city", "a remote island with high-speed internet", "a hackathon",
    "an innovation lab", "a digital marketplace", "an AI research center"
]

# Generate random strings for unique prefixes
RANDOM_WORDS = []
for _ in range(100000):
    # Generate a random string of 8 digits
    rand_str = str(random.randint(10000000, 99999999))
    RANDOM_WORDS.append(rand_str)

def generate_realistic_prompt(tokenizer, target_token_length):
    """
    Generate a realistic prompt using templates and domain-specific vocabulary
    
    Args:
        tokenizer: The tokenizer to use
        target_token_length: Desired length in tokens
        
    Returns:
        A realistic prompt string
    """
    # Start with a random template
    template = random.choice(REALISTIC_TEMPLATES)
    
    # Fill in the template with random relevant content
    filled_template = template.format(
        topic=random.choice(TOPICS),
        topic_a=random.choice(TOPICS),
        topic_b=random.choice(TOPICS),
        purpose=random.choice(["project", "research", "presentation", "startup idea", "blog post"]),
        action=random.choice(ACTIONS),
        activity=random.choice(["coding", "designing", "analyzing", "implementing", "testing"]),
        field=random.choice(["tech", "finance", "healthcare", "education", "e-commerce"]),
        character=random.choice(CHARACTERS),
        character_a=random.choice(CHARACTERS),
        character_b=random.choice(CHARACTERS),
        object=random.choice(["a quantum computer", "an AI assistant", "a time machine", "a virtual reality device"]),
        location=random.choice(LOCATIONS),
        theme=random.choice(["innovation", "digital transformation", "future of work", "technological singularity"]),
        author=random.choice(["a tech visionary", "a sci-fi writer", "a futurist", "a digital artist"]),
        setting=random.choice(["a smart city", "a space colony", "a digital universe", "a post-AI world"]),
        recipient=random.choice(["a potential client", "a team member", "a project stakeholder", "a tech investor"]),
        subject=random.choice(["project proposal", "software update", "partnership opportunity", "technical issue"]),
        product=random.choice(["AI software", "smart device", "cloud service", "tech gadget"]),
        feature=random.choice(["innovative features", "user-friendly interface", "cutting-edge technology", "performance"]),
        service=random.choice(["consulting service", "tech solution", "software as a service", "digital platform"]),
        audience=random.choice(["tech enthusiasts", "business professionals", "developers", "startups"]),
        event=random.choice(["product launch", "tech conference", "software release", "hackathon"]),
        platform=random.choice(["LinkedIn", "Twitter", "Facebook", "Instagram"]),
        person=random.choice(CHARACTERS),
        expertise=random.choice(TOPICS),
        item_a=random.choice(TOPICS),
        item_b=random.choice(TOPICS),
        item_c=random.choice(TOPICS),
        skill=random.choice(["programming", "data analysis", "system design", "technical writing", "debugging"])
    )
    
    # Check token length
    token_count = len(tokenizer.encode(filled_template))
    
    # If the template is too short, extend it with additional relevant content
    while token_count < target_token_length:
        # Add more content to the prompt
        additional_content = [
            f" Additionally, I'm interested in learning about {random.choice(TOPICS)}.",
            f" Could you also explain how this relates to {random.choice(TOPICS)}?",
            f" I'm asking because I need to {random.choice(ACTIONS)} for {random.choice(['my work', 'a client', 'a project', 'my research'])}.",
            f" For context, I have experience with {random.choice(TOPICS)} but I'm new to this specific area.",
            f" I've been trying to understand this concept for {random.choice(['days', 'weeks', 'months'])} and would appreciate a clear explanation."
        ]
        
        filled_template += random.choice(additional_content)
        token_count = len(tokenizer.encode(filled_template))
    
    # If the prompt is too long, truncate it to roughly the desired length
    # This simple approach may not be as precise as the original but should work for our purposes
    if token_count > target_token_length:
        # Estimate the ratio of tokens to characters for simple truncation
        ratio = len(filled_template) / token_count
        estimated_char_count = int(target_token_length * ratio)
        filled_template = filled_template[:estimated_char_count]
        
        # Re-check token length to make small adjustments if needed
        token_count = len(tokenizer.encode(filled_template))
        
        # If still too long, continue truncating
        while token_count > target_token_length and filled_template:
            filled_template = filled_template[:-10]  # Remove 10 chars at a time
            token_count = len(tokenizer.encode(filled_template))
    
    return filled_template

def generate_unique_prefix(base_text, index):
    return RANDOM_WORDS[index] + " " + base_text

def parallelize_prompts_preparation(config, tokenizer, start_index, end_index):
    """
    Process a chunk of prefixes for a single configuration
    
    Args:
        config: Configuration dictionary
        tokenizer: The tokenizer to use
        start_index: Starting index for this chunk
        end_index: Ending index (exclusive) for this chunk
        
    Returns:
        Tuple of (prompts chunk, token counts chunk, total tokens)
    """
    prefix_length = config["prefix_length"]
    suffix_length = config["suffix_length"]
    num_samples_per_prefix = config["num_samples_per_prefix"]
    
    # Use the same base prefix for all chunks to maintain prefix sharing benefits
    base_prefix = generate_realistic_prompt(tokenizer, prefix_length)
    tot_input_len = 0
    prompts_chunk = []
    prompts_token_counts_chunk = []
    
    # Process only the assigned range of prefixes
    for i in range(start_index, end_index):
        unique_prefix = generate_unique_prefix(base_prefix, i)
        prompt_list = []
        token_count_list = []
        
        for j in range(num_samples_per_prefix):
            suffix = generate_realistic_prompt(tokenizer, suffix_length)
            prompt = unique_prefix + " " + suffix
            
            token_count = len(tokenizer.encode(prompt))
            tot_input_len += token_count
            
            prompt_list.append(prompt)
            token_count_list.append(token_count)
        
        prompts_chunk.append(prompt_list)
        prompts_token_counts_chunk.append(token_count_list)
    
    return prompts_chunk, prompts_token_counts_chunk, tot_input_len

def prepare_prompts_parallel(tokenizer, config, num_workers=4):
    """
    Prepare prompts based on the provided configuration using parallel processing
    
    Args:
        tokenizer: The tokenizer to use
        config: Dictionary with prefix_length, suffix_length, num_samples_per_prefix, num_prefix
        num_workers: Number of parallel workers
        
    Returns:
        Tuple of (all_prompts, tot_input_len, prompts_token_counts)
    """
    num_diff_prefix = config["num_diff_prefix"]
    
    # Determine chunk size for each worker
    chunk_size = max(1, num_diff_prefix // num_workers)
    chunks = []
    
    for i in range(0, num_diff_prefix, chunk_size):
        end = min(i + chunk_size, num_diff_prefix)
        chunks.append((i, end))
    
    all_prompts = []
    all_token_counts = []
    total_tokens = 0
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_chunk = {
            executor.submit(
                parallelize_prompts_preparation, 
                config, 
                tokenizer, 
                start, 
                end
            ): (start, end) for start, end in chunks
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_chunk), 
                          total=len(future_to_chunk),
                          desc=f"Processing prefixes for config {config['id']}"):
            start, end = future_to_chunk[future]
            try:
                prompts_chunk, token_counts_chunk, tokens = future.result()
                all_prompts.extend(prompts_chunk)
                all_token_counts.extend(token_counts_chunk)
                total_tokens += tokens
                
            except Exception as exc:
                print(f"Chunk {start}-{end} generated an exception: {exc}")
    
    return all_prompts, total_tokens, all_token_counts

def prepare_prompts(tokenizer, config):
    """
    Prepare prompts based on the provided configuration (non-parallel version)
    
    Args:
        tokenizer: The tokenizer to use
        config: Dictionary with prefix_length, suffix_length, num_samples_per_prefix, num_prefix
        
    Returns:
        Tuple of (all_prompts, tot_input_len, prompts_token_counts)
    """
    prefix_length = config["prefix_length"]
    suffix_length = config["suffix_length"]
    num_samples_per_prefix = config["num_samples_per_prefix"]
    num_diff_prefix = config["num_diff_prefix"]
    
    # Generate a base prefix using realistic content
    base_prefix = generate_realistic_prompt(tokenizer, prefix_length)
    tot_input_len = 0
    all_prompts = []
    prompts_token_counts = []  # Store token counts for each prompt
    
    for i in tqdm(range(num_diff_prefix), desc=f"Preparing prompts for config {config['id']}"):
        unique_prefix = generate_unique_prefix(base_prefix, i)
        prompt_list = []
        token_count_list = []
        
        for j in range(num_samples_per_prefix):
            # Generate a realistic suffix
            suffix = generate_realistic_prompt(tokenizer, suffix_length)
            prompt = unique_prefix + " " + suffix
            
            # Count tokens
            token_count = len(tokenizer.encode(prompt))
            tot_input_len += token_count
            
            prompt_list.append(prompt)
            token_count_list.append(token_count)
        
        all_prompts.append(prompt_list)
        prompts_token_counts.append(token_count_list)
    
    return all_prompts, tot_input_len, prompts_token_counts

def calculate_prefix_proportion(prefix_length, suffix_length):
    """
    Calculate the proportion of the prompt that is prefix.
    
    Prefix proportion = prefix_length / (prefix_length + suffix_length)
    
    Args:
        prefix_length: Length of the prefix in tokens
        suffix_length: Length of the suffix in tokens
        
    Returns:
        Prefix proportion (float)
    """
    return prefix_length / (prefix_length + suffix_length)

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

def process_workload_configs(tokenizer, configs, num_workers=4):
    """
    Process multiple workload configurations and combine them using parallel processing
    where appropriate
    
    Args:
        tokenizer: The tokenizer to use
        configs: List of workload configuration dictionaries
        num_workers: Number of parallel workers
        
    Returns:
        Dictionary with combined workload data
    """
    all_prompts_combined = []
    total_tokens = 0
    config_stats = []
    
    # Variables for overall prefix sharing calculation
    all_prompts_for_sharing = []
    all_prompts_token_counts = []
    all_prefix_lengths = []
    
    current_time = 0  # Track current time for sequential workloads
    
    # Process each configuration - we process configs sequentially to maintain
    # the time ordering, but parallelize within each config
    for i, config in enumerate(configs):
        # Add an ID to the config for reference
        config["id"] = i+1
        
        print(f"\nProcessing config {config['id']} in parallel:")
        # Generate prompts for this config using parallel processing
        prompts, tokens, token_counts = prepare_prompts_parallel(tokenizer, config, num_workers)
        total_tokens += tokens
        
        # Calculate prefix sharing ratio for this config
        sharing_ratio = calculate_prefix_sharing_ratio(
            tokenizer, prompts, token_counts, config["prefix_length"]
        )
        
        # Calculate prefix proportion
        prefix_proportion = calculate_prefix_proportion(
            config["prefix_length"], config["suffix_length"]
        )
        
        # Create flattened prompt data with prefix group information
        flat_prompts_data = []
        for prefix_idx, prompt_list in enumerate(prompts):
            for j, prompt in enumerate(prompt_list):
                flat_prompts_data.append({
                    "prompt": prompt,
                    "token_count": token_counts[prefix_idx][j],
                    "prefix_group": prefix_idx,
                    "config_id": config["id"]
                })
        
        # Generate timestamps for this config
        rps = config.get("rps", 1)
        timestamps = generate_poisson_arrival_times(
            num_requests=len(flat_prompts_data),
            rps=rps,
            start_time=current_time
        )
        
        # Update current_time for next config
        if timestamps:
            current_time = max(timestamps) + 1000  # Add a 1-second gap between configs
        
        # Add timestamps to prompt data
        for j, prompt_data in enumerate(flat_prompts_data):
            prompt_data["timestamp"] = timestamps[j]
            all_prompts_combined.append(prompt_data)
        
        # Store config data for overall prefix calculation
        all_prompts_for_sharing.extend(prompts)
        all_prompts_token_counts.extend(token_counts)
        all_prefix_lengths.extend([config["prefix_length"]] * len(prompts))
        
        # Store stats for this config
        total_num_req = config["num_diff_prefix"] * config["num_samples_per_prefix"]
        total_duration = total_num_req / rps
        
        config_stats.append({
            "config_id": config["id"],
            "prefix_length": config["prefix_length"],
            "suffix_length": config["suffix_length"],
            "num_samples_per_prefix": config["num_samples_per_prefix"],
            "num_diff_prefix": config["num_diff_prefix"],
            "rps": rps,
            "num_requests": len(flat_prompts_data),
            "total_tokens": tokens,
            "total_duration": total_duration,
            "prefix_sharing_ratio": sharing_ratio,
            "prefix_proportion": prefix_proportion,
            "start_time": min(timestamps) if timestamps else 0,
            "end_time": max(timestamps) if timestamps else 0
        })
    
    # Calculate overall prefix sharing ratio
    overall_sharing_ratio = 0
    if len(configs) == 1:
        overall_sharing_ratio = config_stats[0]["prefix_sharing_ratio"]
        overall_prefix_proportion = config_stats[0]["prefix_proportion"]
    else:
        total_config_tokens = sum(cfg["total_tokens"] for cfg in config_stats)
        overall_sharing_ratio = sum(
            cfg["prefix_sharing_ratio"] * cfg["total_tokens"] / total_config_tokens
            for cfg in config_stats
        ) if total_config_tokens > 0 else 0
        
        overall_prefix_proportion = sum(
            cfg["prefix_proportion"] * cfg["total_tokens"] / total_config_tokens
            for cfg in config_stats
        ) if total_config_tokens > 0 else 0
    
    # Global randomization of all prompts
    if len(all_prompts_combined) > 1:
        # Extract all timestamps
        all_timestamps = [prompt["timestamp"] for prompt in all_prompts_combined]
        
        # Shuffle the timestamps
        random.shuffle(all_timestamps)
        
        # Reassign the shuffled timestamps to the prompts
        for i, prompt in enumerate(all_prompts_combined):
            prompt["timestamp"] = all_timestamps[i]
        
        # Sort combined data by timestamp - this keeps the shuffled order
        all_prompts_combined.sort(key=lambda x: x["timestamp"])

        print(f"Shuffled timestamps for all prompts to ensure randomness in prompt order.")
    return {
        "prompts": all_prompts_combined,
        "stats": config_stats,
        "total_tokens": total_tokens,
        "overall_sharing_ratio": overall_sharing_ratio,
        "overall_prefix_proportion": overall_prefix_proportion
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
                        "prompt": item["prompt"],
                        "prefix_group": item["prefix_group"],  # Add prefix group info for analysis
                        "config_id": item["config_id"]
                    }
                ]
            }
            f.write(json.dumps(entry) + '\n')

def process_workload_with_named_patterns(tokenizer, workload_configs, num_workers=4):
    """
    Process named workload patterns and calculate sharing ratios at multiple levels
    
    Args:
        tokenizer: The tokenizer to use
        workload_configs: Dictionary of named patterns with configurations
        num_workers: Number of parallel workers
        
    Returns:
        Dictionary with processed workloads and statistics
    """
    results = {}
    
    for pattern_name, configs in workload_configs.items():
        print(f"\nProcessing pattern: {pattern_name}")
        
        pattern_prompts = []
        pattern_total_tokens = 0
        workload_stats = []
        
        # Process each workload unit within the pattern
        for i, workload in enumerate(configs):
            workload_id = i + 1
            print(f"  Processing workload unit {workload_id} in {pattern_name}")
            
            # Add identifiers to the workload
            workload["id"] = workload_id
            workload["pattern_name"] = pattern_name
            
            # Generate prompts for this workload
            prompts, tokens, token_counts = prepare_prompts_parallel(tokenizer, workload, num_workers)
            pattern_total_tokens += tokens
            
            # Calculate prefix sharing ratio for this specific workload
            workload_sharing_ratio = calculate_prefix_sharing_ratio(
                tokenizer, prompts, token_counts, workload["prefix_length"]
            )
            
            # Calculate prefix proportion
            prefix_proportion = calculate_prefix_proportion(
                workload["prefix_length"], workload["suffix_length"]
            )
            
            # Create flattened prompt data with workload information
            flat_prompts_data = []
            for prefix_idx, prompt_list in enumerate(prompts):
                for j, prompt in enumerate(prompt_list):
                    flat_prompts_data.append({
                        "prompt": prompt,
                        "token_count": token_counts[prefix_idx][j],
                        "prefix_group": prefix_idx,
                        "workload_id": workload_id,
                        "pattern_name": pattern_name
                    })
            
            # Generate timestamps for this workload
            rps = workload.get("rps", 1)
            timestamps = generate_poisson_arrival_times(
                num_requests=len(flat_prompts_data),
                rps=rps,
                start_time=0  # Start at 0 for each workload
            )
            
            # Add timestamps to prompt data
            for j, prompt_data in enumerate(flat_prompts_data):
                prompt_data["timestamp"] = timestamps[j]
                pattern_prompts.append(prompt_data)
            
            # Store stats for this workload
            total_num_req = workload["num_diff_prefix"] * workload["num_samples_per_prefix"]
            total_duration = total_num_req / rps
            
            workload_stat = {
                "workload_id": workload_id,
                "pattern_name": pattern_name,
                "prefix_length": workload["prefix_length"],
                "suffix_length": workload["suffix_length"],
                "num_samples_per_prefix": workload["num_samples_per_prefix"],
                "num_diff_prefix": workload["num_diff_prefix"],
                "rps": rps,
                "output_length": workload.get("output_length", 8),
                "num_requests": len(flat_prompts_data),
                "total_tokens": tokens,
                "total_duration": total_duration,
                "prefix_sharing_ratio": workload_sharing_ratio,
                "prefix_proportion": prefix_proportion,
                "start_time": min(timestamps) if timestamps else 0,
                "end_time": max(timestamps) if timestamps else 0
            }
            
            workload_stats.append(workload_stat)
        
        # Shuffle timestamps within the pattern
        if len(pattern_prompts) > 1:
            # Extract all timestamps
            pattern_timestamps = [prompt["timestamp"] for prompt in pattern_prompts]
            
            # Shuffle the timestamps
            random.shuffle(pattern_timestamps)
            
            # Reassign the shuffled timestamps to the prompts
            for i, prompt in enumerate(pattern_prompts):
                prompt["timestamp"] = pattern_timestamps[i]
            
            print(f"  Shuffled timestamps for prompts within pattern {pattern_name}")
        
        # Sort pattern data by timestamp
        pattern_prompts.sort(key=lambda x: x["timestamp"])
        
        # Calculate pattern-level prefix sharing ratio
        pattern_prefix_sharing = 0
        if workload_stats:
            total_pattern_tokens = sum(stat["total_tokens"] for stat in workload_stats)
            pattern_prefix_sharing = sum(
                stat["prefix_sharing_ratio"] * stat["total_tokens"] / total_pattern_tokens
                for stat in workload_stats
            ) if total_pattern_tokens > 0 else 0
        
        # Calculate pattern-level prefix proportion
        pattern_prefix_proportion = 0
        if workload_stats:
            total_pattern_tokens = sum(stat["total_tokens"] for stat in workload_stats)
            pattern_prefix_proportion = sum(
                stat["prefix_proportion"] * stat["total_tokens"] / total_pattern_tokens
                for stat in workload_stats
            ) if total_pattern_tokens > 0 else 0
        
        # Store pattern results
        results[pattern_name] = {
            "prompts": pattern_prompts,
            "workload_stats": workload_stats,
            "total_tokens": pattern_total_tokens,
            "pattern_prefix_sharing_ratio": pattern_prefix_sharing,
            "pattern_prefix_proportion": pattern_prefix_proportion,
            "num_requests": len(pattern_prompts)
        }
    
    return results

def save_pattern_stats(pattern_results, pattern_name, stats_file):
    """
    Save statistics for a specific pattern to a JSON file
    
    Args:
        pattern_results: Dictionary with pattern results
        pattern_name: Name of the pattern
        stats_file: Output file path for stats
    """
    stats = {
        "pattern_name": pattern_name,
        "workload_stats": pattern_results["workload_stats"],
        "num_tokens": pattern_results["total_tokens"],
        "num_requests": pattern_results["num_requests"],
        "pattern_prefix_sharing_ratio": pattern_results["pattern_prefix_sharing_ratio"],
        "pattern_prefix_proportion": pattern_results["pattern_prefix_proportion"],
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nPattern Summary for {pattern_name}:")
    print(f"- Total requests: {pattern_results['num_requests']}")
    print(f"- Total tokens: {pattern_results['total_tokens']}")
    print(f"- Pattern prefix proportion: {pattern_results['pattern_prefix_proportion']*100:.2f}%")
    print(f"- Pattern prefix sharing ratio: {pattern_results['pattern_prefix_sharing_ratio']*100:.2f}%")
    
    print("\nWorkload units details:")
    for workload in pattern_results["workload_stats"]:
        print(f"  Workload {workload['workload_id']}:")
        print(f"    - Prefix length: {workload['prefix_length']}")
        print(f"    - Suffix length: {workload['suffix_length']}")
        print(f"    - Number of requests per prefix: {workload['num_samples_per_prefix']}")
        print(f"    - Number of different prefixes: {workload['num_diff_prefix']}")
        print(f"    - RPS: {workload['rps']}")
        print(f"    - Output length: {workload['output_length']}")
        print(f"    - Requests: {workload['num_requests']}")
        print(f"    - Duration: {workload['total_duration']:.0f} seconds")
        print(f"    - Prefix proportion: {workload['prefix_proportion']*100:.2f}%")
        print(f"    - Prefix sharing ratio: {workload['prefix_sharing_ratio']*100:.2f}%")


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    
    # Define workload configurations with groups
    workload_configs = {
    "production-like-mixed-load": [
            {
                "prefix_length": 1024,
                "suffix_length": 256,
                "num_samples_per_prefix": 15,
                "num_diff_prefix": 20,
                "rps": 15,
                "output_length": 128
            },
            {
                "prefix_length": 4096,
                "suffix_length": 512,
                "num_samples_per_prefix": 12,
                "num_diff_prefix": 15,
                "rps": 8,
                "output_length": 512
            },
            {
                "prefix_length": 8192,
                "suffix_length": 1024,
                "num_samples_per_prefix": 8,
                "num_diff_prefix": 10,
                "rps": 3,
                "output_length": 1024
            }
        ],
        
        "basic-load-patterns": [
            {
                "prefix_length": 2048,
                "suffix_length": 512,
                "num_samples_per_prefix": 20,
                "num_diff_prefix": 15,
                "rps": 3,
                "output_length": 256
            },
            {
                "prefix_length": 2048,
                "suffix_length": 512,
                "num_samples_per_prefix": 20,
                "num_diff_prefix": 15,
                "rps": 10,
                "output_length": 256
            },
            {
                "prefix_length": 2048,
                "suffix_length": 512,
                "num_samples_per_prefix": 20,
                "num_diff_prefix": 15,
                "rps": 25,
                "output_length": 256
            }
        ],
        
        "prefix-sharing-efficiency": [
            {
                "prefix_length": 4096,
                "suffix_length": 1024,
                "num_samples_per_prefix": 5,
                "num_diff_prefix": 25,
                "rps": 8,
                "output_length": 512
            },
            {
                "prefix_length": 4096,
                "suffix_length": 1024,
                "num_samples_per_prefix": 40,
                "num_diff_prefix": 10,
                "rps": 8,
                "output_length": 512
            }
        ],
        
        "input-size-impact": [
            {
                "prefix_length": 1024,
                "suffix_length": 256,
                "num_samples_per_prefix": 15,
                "num_diff_prefix": 20,
                "rps": 20,
                "output_length": 256
            },
            {
                "prefix_length": 4096,
                "suffix_length": 1024,
                "num_samples_per_prefix": 15,
                "num_diff_prefix": 15,
                "rps": 10,
                "output_length": 256
            },
            {
                "prefix_length": 8192,
                "suffix_length": 2048,
                "num_samples_per_prefix": 15,
                "num_diff_prefix": 10,
                "rps": 3,
                "output_length": 256
            }
        ],
        
        "output-size-impact": [
            {
                "prefix_length": 2048,
                "suffix_length": 512,
                "num_samples_per_prefix": 15,
                "num_diff_prefix": 10,
                "rps": 15,
                "output_length": 64
            },
            {
                "prefix_length": 2048,
                "suffix_length": 512,
                "num_samples_per_prefix": 15,
                "num_diff_prefix": 10,
                "rps": 8,
                "output_length": 512
            },
            {
                "prefix_length": 2048,
                "suffix_length": 512,
                "num_samples_per_prefix": 15,
                "num_diff_prefix": 10,
                "rps": 3,
                "output_length": 1536
            }
        ],
        
        "chatbot-simulation": [
            {
                "prefix_length": 4096,
                "suffix_length": 256,
                "num_samples_per_prefix": 30,
                "num_diff_prefix": 10,
                "rps": 12,
                "output_length": 384
            }
        ],
        
        "content-generation": [
            {
                "prefix_length": 6144,
                "suffix_length": 1024,
                "num_samples_per_prefix": 8,
                "num_diff_prefix": 15,
                "rps": 5,
                "output_length": 1024
            }
        ],
        
        "quick-qa": [
            {
                "prefix_length": 1024,
                "suffix_length": 256,
                "num_samples_per_prefix": 10,
                "num_diff_prefix": 25,
                "rps": 25,
                "output_length": 128
            }
        ],
        
        "burst-patterns": [
            {
                "prefix_length": 2048,
                "suffix_length": 512,
                "num_samples_per_prefix": 20,
                "num_diff_prefix": 50,
                "rps": 30,
                "output_length": 256,
                "burst_duration": 60
            },
            {
                "prefix_length": 2048,
                "suffix_length": 512,
                "num_samples_per_prefix": 20,
                "num_diff_prefix": 15,
                "rps": 8,
                "output_length": 256
            }
        ],
        
        "large-context-testing": [
            {
                "prefix_length": 16384,
                "suffix_length": 1024,
                "num_samples_per_prefix": 10,
                "num_diff_prefix": 5,
                "rps": 2,
                "output_length": 512
            },
            {
                "prefix_length": 32768,
                "suffix_length": 2048,
                "num_samples_per_prefix": 5,
                "num_diff_prefix": 3,
                "rps": 1,
                "output_length": 768
            }
        ]
    }

    # Initialize tokenizer
    print("Initializing the SimpleTokenizer...")
    tokenizer = SimpleTokenizer()
    
    # Use multiple threads for processing
    num_workers = 4
    print(f"Using {num_workers} worker threads")
    
    # Process all patterns
    pattern_results = process_workload_with_named_patterns(tokenizer, workload_configs, num_workers)
    
    # Save results for each pattern
    for pattern_name, results in pattern_results.items():
        output_dir = f"comprehensive_set/{pattern_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/workload.jsonl"
        stats_file = f"{output_dir}/stats.json"
        
        # Save the JSONL file
        with open(output_file, 'w') as f:
            for item in results["prompts"]:
                entry = {
                    "timestamp": item["timestamp"],
                    "requests": [
                        {
                            "Prompt Length": item["token_count"],
                            "Output Length": workload_configs[pattern_name][item["workload_id"]-1].get("output_length", 8),
                            "prompt": item["prompt"],
                            "prefix_group": item["prefix_group"],
                            "workload_id": item["workload_id"],
                            "pattern_name": item["pattern_name"]
                        }
                    ]
                }
                f.write(json.dumps(entry) + '\n')
        
        # Save the stats file
        save_pattern_stats(results, pattern_name, stats_file)
        
        print(f"Saved workload trace to {output_file}")
        print(f"Saved workload statistics to {stats_file}")
    
    print("\nAll workload patterns generated successfully!")