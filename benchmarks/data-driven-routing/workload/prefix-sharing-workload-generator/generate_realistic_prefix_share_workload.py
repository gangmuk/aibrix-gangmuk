#!/usr/bin/env python3
import random
import string
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

# RANDOM_WORDS = [
#     "Hello, ", "Sounds good, ", "Okay, ",
#     "Nice to meet you, ", "Let me start, ", "I will start, ", "Beginning, ", "However, ", "Therefore, ", "Well, ", "So, ", "In addition, ", "Moreover, ", "Furthermore, ", "Additionally, ", "On the other hand, ", "In contrast, ", "Conversely, ", "Nevertheless, ", "Nonetheless, ", "Despite that, ", "Even so, ", "In spite of that, ", "Yet, ", "Still, ", "But, ", "Although, ", "Though, ", "Even though, ", "While, ", "Whereas, ", "As a result, ", "Consequently, ", "Thus, ", "Hence, ", "Therefore, ", "For this reason, ", "Because of that, ", "Due to that, ", "As a consequence, ", "In conclusion, ", "To sum up, ", "In summary, ", "Overall, ", "All in all, ", "To summarize, ", "To conclude, ", "In short, ", "In brief, ", "Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy", "Kevin", "Liam", "Mia", "Nina", "Oscar", "Paul", "Quinn", "Rita", "Sam", "Tina", "Uma", "Vera", "Will", "Xena", "Yara", "Zane", "Alex", "Blake", "Casey", "Drew", "Erin", "Finn", "Gale", "Hugo", "Ivy", "Jax", "Kai", "Luna", "Max", "Nora", "Owen", "Piper", "Quinn", "Riley", "Sage", "Tess", "Uriah", "Violet", "Wyatt", "Xander", "Yara", "Zane", "Aria", "Bella", "Cora", "Daisy", "Ella", "Fiona", "Gia", "Hannah", "Iris", "Jade", "Kira", "Lila", "Maya", "Nina", "Opal", "Poppy", "Quinn", "Rhea", "Sienna", "Tara", "Uma", "Vera", "Willa", "Yara", "Zara", "Aiden", "Brooke", "Carter", "Dylan", "Ethan", "Faith", "Gavin", "Holly", "Ian", "Jasper", "Kylie", "Logan", "Mason", "Nolan", "Olivia", "Parker", "Quinn", "Ryder", "Sophie", "Tyler", "Ulysses", "Violet"
# ]

RANDOM_WORDS = []
for _ in range(100000):
    # Generate a random string of 5-10 characters
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
    
    # If the prompt is too long, truncate it to the desired length
    if token_count > target_token_length:
        tokenized = tokenizer.encode(filled_template)[:target_token_length]
        filled_template = tokenizer.decode(tokenized, skip_special_tokens=True)
    
    return filled_template

def generate_unique_prefix(base_text, index):
    return RANDOM_WORDS[index] + base_text

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

# def process_workload_configs(tokenizer, configs):
#     """
#     Process multiple workload configurations and combine them
    
#     Args:
#         tokenizer: The tokenizer to use
#         configs: List of workload configuration dictionaries
        
#     Returns:
#         Dictionary with combined workload data
#     """
#     all_prompts_combined = []
#     all_timestamps_combined = []
#     total_tokens = 0
#     config_stats = []
    
#     # Variables to track overall prefix sharing
#     total_prompts_count = 0
#     total_unique_prefixes = 0
#     total_prefix_tokens = 0
    
#     # Variables for overall prefix sharing calculation
#     all_prompts_for_sharing = []
#     all_prompts_token_counts = []
#     all_prefix_lengths = []
    
#     current_time = 0  # Track current time for sequential workloads
    
#     # Process each configuration
#     for i, config in enumerate(configs):
#         # Add an ID to the config for reference
#         config["id"] = i+1
        
#         # Generate prompts for this config
#         prompts, tokens, token_counts = prepare_prompts(tokenizer, config)
#         total_tokens += tokens
        
#         # Calculate prefix sharing ratio for this config
#         sharing_ratio = calculate_prefix_sharing_ratio(
#             tokenizer, prompts, token_counts, config["prefix_length"]
#         )
        
#         # Calculate prefix proportion
#         prefix_proportion = calculate_prefix_proportion(
#             config["prefix_length"], config["suffix_length"]
#         )
        
#         # Create flattened prompt data with prefix group information
#         flat_prompts_data = []
#         for prefix_idx, prompt_list in enumerate(prompts):
#             for j, prompt in enumerate(prompt_list):
#                 flat_prompts_data.append({
#                     "prompt": prompt,
#                     "token_count": token_counts[prefix_idx][j],
#                     "prefix_group": prefix_idx,
#                     "config_id": config["id"]
#                 })
        
#         # Determine if we should randomize the order
#         randomize_order = config.get("randomize_order", False)
        
#         # If randomize_order is True, shuffle the prompts across different prefix groups
#         if randomize_order:
#             random.shuffle(flat_prompts_data)
        
#         # Generate timestamps for this config
#         rps = config.get("rps", 1)
#         timestamps = generate_poisson_arrival_times(
#             num_requests=len(flat_prompts_data),
#             rps=rps,
#             start_time=current_time
#         )
        
#         # Update current_time for next config
#         if timestamps:
#             current_time = max(timestamps) + 1000  # Add a 1-second gap between configs
        
#         # Add timestamps to prompt data
#         for j, prompt_data in enumerate(flat_prompts_data):
#             prompt_data["timestamp"] = timestamps[j]
#             all_prompts_combined.append(prompt_data)
        
#         # Update overall prefix sharing tracking
#         total_prompts_count += len(flat_prompts_data)
        
#         # Store config data for overall prefix calculation
#         all_prompts_for_sharing.extend(prompts)
#         all_prompts_token_counts.extend(token_counts)
#         all_prefix_lengths.extend([config["prefix_length"]] * len(prompts))
        
#         # Store stats for this config
#         total_num_req = config["num_diff_prefix"] * config["num_samples_per_prefix"]
#         total_duration = total_num_req / rps
        
#         config_stats.append({
#             "config_id": config["id"],
#             "prefix_length": config["prefix_length"],
#             "suffix_length": config["suffix_length"],
#             "num_samples_per_prefix": config["num_samples_per_prefix"],
#             "num_diff_prefix": config["num_diff_prefix"],
#             "rps": rps,
#             "randomize_order": randomize_order,
#             "num_requests": len(flat_prompts_data),
#             "total_tokens": tokens,
#             "total_duration": total_duration,
#             "prefix_sharing_ratio": sharing_ratio,
#             "prefix_proportion": prefix_proportion,
#             "start_time": min(timestamps) if timestamps else 0,
#             "end_time": max(timestamps) if timestamps else 0
#         })
    
#     # Calculate overall prefix sharing ratio using the same token-based method
#     overall_sharing_ratio = 0
#     if len(configs) == 1:
#         # If there's only one config, use its sharing ratio
#         overall_sharing_ratio = config_stats[0]["prefix_sharing_ratio"]
#         overall_prefix_proportion = config_stats[0]["prefix_proportion"]
#     else:
#         # For multiple configs, calculate an overall ratio based on all prompts
#         # This is more complex and would need special handling for different prefix lengths
#         # For now, we'll use a weighted average based on token counts
#         total_config_tokens = sum(cfg["total_tokens"] for cfg in config_stats)
#         overall_sharing_ratio = sum(
#             cfg["prefix_sharing_ratio"] * cfg["total_tokens"] / total_config_tokens
#             for cfg in config_stats
#         ) if total_config_tokens > 0 else 0
        
#         # Calculate weighted average of prefix proportions
#         overall_prefix_proportion = sum(
#             cfg["prefix_proportion"] * cfg["total_tokens"] / total_config_tokens
#             for cfg in config_stats
#         ) if total_config_tokens > 0 else 0
    

#     # With this approach:
#     if len(all_prompts_combined) > 1:
#         # Extract all timestamps
#         all_timestamps = [prompt["timestamp"] for prompt in all_prompts_combined]
        
#         # Shuffle the timestamps
#         random.shuffle(all_timestamps)
        
#         # Reassign the shuffled timestamps to the prompts
#         for i, prompt in enumerate(all_prompts_combined):
#             prompt["timestamp"] = all_timestamps[i]
        
#         # Sort combined data by timestamp - this keeps the shuffled order
#         all_prompts_combined.sort(key=lambda x: x["timestamp"])
    
#     return {
#         "prompts": all_prompts_combined,
#         "stats": config_stats,
#         "total_tokens": total_tokens,
#         "overall_sharing_ratio": overall_sharing_ratio,
#         "overall_prefix_proportion": overall_prefix_proportion
#     }


import concurrent.futures
from functools import partial



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
    
    # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid tokenizer issues
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
        for future in concurrent.futures.as_completed(future_to_chunk):
            start, end = future_to_chunk[future]
            try:
                prompts_chunk, token_counts_chunk, tokens = future.result()
                all_prompts.extend(prompts_chunk)
                all_token_counts.extend(token_counts_chunk)
                total_tokens += tokens
                
                # Update progress
                print(f"Processed {end-start} prefixes from {start} to {end}")
                
            except Exception as exc:
                print(f"Chunk {start}-{end} generated an exception: {exc}")
    
    return all_prompts, total_tokens, all_token_counts

def process_single_config(config, tokenizer, current_time=0):
    """
    Process a single configuration
    
    Args:
        config: Configuration dictionary
        tokenizer: The tokenizer to use
        current_time: Starting timestamp for this config
        
    Returns:
        Tuple of (
            flat_prompts_data, 
            config_stats, 
            tokens, 
            prompts_for_sharing,
            prompts_token_counts, 
            prefix_lengths, 
            end_time
        )
    """
    # Generate prompts for this config
    prompts, tokens, token_counts = prepare_prompts(tokenizer, config)
    
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
    
    # Calculate end time for this config
    end_time = max(timestamps) + 1000 if timestamps else current_time
    
    # Add timestamps to prompt data
    for j, prompt_data in enumerate(flat_prompts_data):
        prompt_data["timestamp"] = timestamps[j]
    
    # Store prompts for overall prefix calculation
    prompts_for_sharing = prompts
    prompts_token_counts = token_counts
    prefix_lengths = [config["prefix_length"]] * len(prompts)
    
    # Calculate stats for this config
    total_num_req = config["num_diff_prefix"] * config["num_samples_per_prefix"]
    total_duration = total_num_req / rps
    
    config_stats = {
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
    }
    
    return (
        flat_prompts_data, 
        config_stats, 
        tokens, 
        prompts_for_sharing,
        prompts_token_counts, 
        prefix_lengths, 
        end_time
    )


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
        
        # Skip within-config randomization since we'll do global randomization
        # at the end which is sufficient
        
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
            "randomize_order": config.get("randomize_order", False),
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
            "overall_prefix_proportion": workload_data["overall_prefix_proportion"],
        }, f, indent=2)
    
    total_duration = 0
    total_num_requests = 0
    print("\nConfiguration details:")
    for cfg in workload_data["stats"]:
        num_req = cfg['num_diff_prefix'] * cfg['num_samples_per_prefix']
        duration = num_req / cfg['rps']
        total_duration += duration
        total_num_requests += num_req
        print(f"Config {cfg['config_id']}:")
        print(f"  - Prefix length: {cfg['prefix_length']}")
        print(f"  - Suffix length: {cfg['suffix_length']}")
        print(f"  - Number of requests per prefix: {cfg['num_samples_per_prefix']}")
        print(f"  - Number of different prefixes: {cfg['num_diff_prefix']}")
        print(f"  - RPS: {cfg['rps']}")
        print(f"  - Duration: {duration:.0f} seconds")
        print(f"  - Number of requests {cfg['num_requests']}")
        print(f"  - Prefix proportion: {cfg['prefix_proportion']*100:.2f}% (portion of each prompt that is shared)")
        print(f"  - Efficiency gain: {cfg['prefix_sharing_ratio']*100:.2f}% (computational savings from prefix sharing)")
        print(f"  - Time range: {int(cfg['start_time']/1000)}s to {int(cfg['end_time']/1000)}s")

    print("\nWorkload Summary:")
    print(f"Total number of requests: {total_num_requests}")
    print(f"Total duration: {total_duration:.0f} seconds")
    print(f"Total prompts: {len(workload_data['prompts'])}")
    print(f"Total tokens: {workload_data['total_tokens']}")
    print(f"Overall prefix proportion: {workload_data['overall_prefix_proportion']*100:.2f}% (portion of each prompt that is shared)")
    print(f"Overall efficiency gain: {workload_data['overall_sharing_ratio']*100:.2f}% (computational savings from prefix sharing)")

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    
    # Define workload configurations
    prefix_workload_configs = [
        {
            "prefix_length": 2048,
            "suffix_length": 512,
            "num_samples_per_prefix": 50,
            "num_diff_prefix": 50,
            "rps": 10,
            "randomize_order": True  # This flag is no longer used within configs
        },
        {
            "prefix_length": 4096,
            "suffix_length": 1024,
            "num_samples_per_prefix": 50,
            "num_diff_prefix": 50,
            "rps": 10,
            "randomize_order": True
        },
        {
            "prefix_length": 8096,
            "suffix_length": 2048,
            "num_samples_per_prefix": 50,
            "num_diff_prefix": 50,
            "rps": 10,
            "randomize_order": True
        },
        {
            "prefix_length": 16192,
            "suffix_length": 4096,
            "num_samples_per_prefix": 50,
            "num_diff_prefix": 50,
            "rps": 10,
            "randomize_order": True
        },
    ]
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "hf-internal-testing/llama-tokenizer", 
        legacy=True,
        model_max_length=30000,  # Increased to handle longer prefixes
        padding_side="right",
        truncation_side="right",
        use_fast=True
    )
    
    # Calculate number of workers based on CPU count
    num_cpus = os.cpu_count()
    num_workers = max(1, num_cpus - 1) if num_cpus else 4
    print(f"Using {num_workers} worker threads")
    
    # Generate filename
    output_filename = ""
    for config in prefix_workload_configs:
        output_filename += f"p{config['prefix_length']}_s{config['suffix_length']}_rps{config['rps']}-"
    if output_filename.endswith("-"):
        output_filename = output_filename[:-1]
    
    print("Generating multi-configuration workload...")
    workload_data = process_workload_configs(tokenizer, prefix_workload_configs, num_workers)
    
    # Save results
    output_file = f"{output_filename}.jsonl"
    stats_file = f"{output_filename}-stats.json"
    save_to_jsonl(workload_data, output_file)
    save_stats(workload_data, stats_file)
    print(f"Saving workload statistics to {stats_file}")
    print(f"Saving workload traces to {output_file}")