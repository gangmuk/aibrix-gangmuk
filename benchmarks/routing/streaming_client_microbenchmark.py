import json
import time
import random
import requests
import transformers

url = "http://localhost:8888/v1/chat/completions"
custom_headers1 = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi',
    'routing-strategy': 'least-request'
}

# routing_strategy = 'prefix-cache'
routing_strategy = 'prefix-cache-and-load'

custom_headers2 = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi',
    'routing-strategy': f'{routing_strategy}'
}

# read prompt file
with open('/Users/bytedance/Downloads/long_prompt.txt', 'r') as file:
    message = file.read()

def run_non_streaming_test():
    print("\n=== NON-STREAMING TEST ===")
    latency_list = []
    target_pod_list = []
    ttft_list = []
    tpot_list = []
    
    # Non-streaming data
    non_streaming_data = {
        "model": "deepseek-llm-7b-chat",
        "messages": [
            {
                "role": "user",
                "content": message,
            },
        ],
        "max_tokens": 100,
        "temperature": 0.0,
    }

    # First non-streaming request with least-request routing
    start_time = time.time()
    response = requests.post(url, headers=custom_headers1, json=non_streaming_data)
    end_time = time.time()

    print(f"Response status: {response.status_code}")
    print(f"Target pod: {response.headers.get('target-pod', 'unknown')}")
    print(f"Latency: {end_time - start_time:.4f}s")

    # Check for timing headers
    if 'x-timing-ttft-ms' in response.headers:
        ttft = float(response.headers['x-timing-ttft-ms']) / 1000.0
        print(f"TTFT: {ttft:.4f}s")
        ttft_list.append(ttft)
        
    if 'x-timing-tpot-ms' in response.headers:
        tpot = float(response.headers['x-timing-tpot-ms']) / 1000.0
        print(f"TPOT: {tpot:.4f}s")
        tpot_list.append(tpot)

    latency_list.append(end_time - start_time)
    target_pod_list.append(response.headers.get('target-pod', 'unknown'))
    time.sleep(2)

    # Additional non-streaming requests with different routing
    prefix_request_repeat = 3
    for i in range(prefix_request_repeat):
        start_time = time.time()
        response = requests.post(url, headers=custom_headers2, json=non_streaming_data)
        end_time = time.time()
        
        print(f"\nRequest {i+1}")
        print(f"Response status: {response.status_code}")
        print(f"Target pod: {response.headers.get('target-pod', 'unknown')}")
        print(f"Latency: {end_time - start_time:.4f}s")
        
        # Check for timing headers
        if 'x-timing-ttft-ms' in response.headers:
            ttft = float(response.headers['x-timing-ttft-ms']) / 1000.0
            print(f"TTFT: {ttft:.4f}s")
            ttft_list.append(ttft)
            
        if 'x-timing-tpot-ms' in response.headers:
            tpot = float(response.headers['x-timing-tpot-ms']) / 1000.0
            print(f"TPOT: {tpot:.4f}s")
            tpot_list.append(tpot)
        
        latency_list.append(end_time - start_time)
        target_pod_list.append(response.headers.get('target-pod', 'unknown'))
        time.sleep(2)
    
    return {
        "latency_list": latency_list,
        "target_pod_list": target_pod_list,
        "ttft_list": ttft_list,
        "tpot_list": tpot_list
    }

def run_streaming_test():
    print("\n=== STREAMING TEST ===")
    latency_list = []
    target_pod_list = []
    ttft_list = []
    tpot_list = []
    
    # Streaming data
    streaming_data = {
        "model": "deepseek-llm-7b-chat",
        "messages": [
            {
                "role": "user",
                "content": message,
            },
        ],
        "max_tokens": 100,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {
            "include_usage": True
        }
    }

    # First streaming request with least-request routing
    start_time = time.time()
    first_token_time = None
    token_count = 0
    response = requests.post(url, headers=custom_headers1, json=streaming_data, stream=True)
    target_pod_list.append(response.headers.get('target-pod', 'unknown'))

    print(f"Response status: {response.status_code}")
    print(f"Target pod: {response.headers.get('target-pod', 'unknown')}")

    for line in response.iter_lines():
        if line:
            # Skip empty lines or keep-alive new lines
            line = line.decode('utf-8')
            if line.startswith('data: '):
                line = line[6:]  # Remove 'data: ' prefix
                
                if line == "[DONE]":
                    break
                    
                try:
                    data = json.loads(line)
                    
                    # Track first token time
                    if first_token_time is None and len(data.get('choices', [])) > 0:
                        if data['choices'][0].get('delta', {}).get('content'):
                            first_token_time = time.time()
                            print(f"First token received after {first_token_time - start_time:.4f}s")
                    
                    # Count tokens (simplified)
                    if len(data.get('choices', [])) > 0 and data['choices'][0].get('delta', {}).get('content'):
                        token_count += 1
                except json.JSONDecodeError:
                    pass

    end_time = time.time()
    total_time = end_time - start_time
    latency_list.append(total_time)

    # Calculate streaming metrics
    ttft = first_token_time - start_time if first_token_time else None
    tpot = (end_time - first_token_time) / (token_count - 1) if first_token_time and token_count > 1 else None

    print(f"Latency: {total_time:.4f}s")
    print(f"TTFT (calculated): {ttft:.4f}s" if ttft else "TTFT: N/A")
    print(f"TPOT (calculated): {tpot:.4f}s" if tpot else "TPOT: N/A")
    print(f"Token count: {token_count}")

    # Check for timing headers (though they might not be accessible in streaming mode)
    if 'x-timing-ttft-ms' in response.headers:
        header_ttft = float(response.headers['x-timing-ttft-ms']) / 1000.0
        print(f"TTFT (header): {header_ttft:.4f}s")
        ttft_list.append(header_ttft)
        
    if 'x-timing-tpot-ms' in response.headers:
        header_tpot = float(response.headers['x-timing-tpot-ms']) / 1000.0
        print(f"TPOT (header): {header_tpot:.4f}s")
        tpot_list.append(header_tpot)
    
    if ttft:
        ttft_list.append(ttft)
    if tpot:
        tpot_list.append(tpot)

    time.sleep(2)

    # Additional streaming requests with different routing
    prefix_request_repeat = 3
    for i in range(prefix_request_repeat):
        start_time = time.time()
        first_token_time = None
        token_count = 0
        response = requests.post(url, headers=custom_headers2, json=streaming_data, stream=True)
        target_pod_list.append(response.headers.get('target-pod', 'unknown'))
        
        print(f"\nStreaming Request {i+1}")
        print(f"Response status: {response.status_code}")
        print(f"Target pod: {response.headers.get('target-pod', 'unknown')}")
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    line = line[6:]
                    
                    if line == "[DONE]":
                        break
                        
                    try:
                        data = json.loads(line)
                        
                        # Track first token time
                        if first_token_time is None and len(data.get('choices', [])) > 0:
                            if data['choices'][0].get('delta', {}).get('content'):
                                first_token_time = time.time()
                                print(f"First token received after {first_token_time - start_time:.4f}s")
                        
                        # Count tokens
                        if len(data.get('choices', [])) > 0 and data['choices'][0].get('delta', {}).get('content'):
                            token_count += 1
                    except json.JSONDecodeError:
                        pass
        
        end_time = time.time()
        total_time = end_time - start_time
        latency_list.append(total_time)
        
        # Calculate streaming metrics
        ttft = first_token_time - start_time if first_token_time else None
        tpot = (end_time - first_token_time) / (token_count - 1) if first_token_time and token_count > 1 else None
        
        print(f"Latency: {total_time:.4f}s")
        print(f"TTFT (calculated): {ttft:.4f}s" if ttft else "TTFT: N/A")
        print(f"TPOT (calculated): {tpot:.4f}s" if tpot else "TPOT: N/A")
        print(f"Token count: {token_count}")
        
        if ttft:
            ttft_list.append(ttft)
        if tpot:
            tpot_list.append(tpot)
        
        # Check for timing headers
        if 'x-timing-ttft-ms' in response.headers:
            header_ttft = float(response.headers['x-timing-ttft-ms']) / 1000.0
            print(f"TTFT (header): {header_ttft:.4f}s")
        
        if 'x-timing-tpot-ms' in response.headers:
            header_tpot = float(response.headers['x-timing-tpot-ms']) / 1000.0
            print(f"TPOT (header): {header_tpot:.4f}s")
        
        time.sleep(2)
    
    return {
        "latency_list": latency_list,
        "target_pod_list": target_pod_list,
        "ttft_list": ttft_list,
        "tpot_list": tpot_list
    }

def print_summary(results, is_streaming=False):
    test_type = "Streaming" if is_streaming else "Non-streaming"
    print(f"\n=== {test_type.upper()} SUMMARY ===")
    
    latency_list = results["latency_list"]
    ttft_list = results["ttft_list"]
    tpot_list = results["tpot_list"]
    target_pod_list = results["target_pod_list"]
    
    print(f"Average latency: {sum(latency_list) / len(latency_list):.4f}s")
    if ttft_list:
        print(f"Average TTFT: {sum(ttft_list) / len(ttft_list):.4f}s")
    if tpot_list:
        print(f"Average TPOT: {sum(tpot_list) / len(tpot_list):.4f}s")
    
    print(f"Target pods:", target_pod_list)

# # Run the tests
# non_streaming_results = run_non_streaming_test()
# print_summary(non_streaming_results, is_streaming=False)

streaming_results = run_streaming_test()
print_summary(streaming_results, is_streaming=True)