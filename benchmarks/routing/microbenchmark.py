import json
import time
import random
import requests
import time
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

# Open the JSON file and load its data
# with open('/Users/bytedance/Downloads/message-threads.json', 'r') as file:
#     workload_data = json.load(file)

# random.shuffle(data)

# read ~/Downloads/long_prompt.txt
# input_path = '/Users/bytedance/Downloads/medium_prompt.txt'
input_path = '/Users/bytedance/Downloads/long_prompt.txt'
# input_path = '/Users/bytedance/Downloads/long_prompt-2.txt'
with open(input_path, 'r') as file:
    message = file.read()
# message = "I like apple."

data = {
    "model": "deepseek-llm-7b-chat",
    "messages": [
        {
            "role": "user",
            "content": message,
        },
    ],
    # "max_tokens": 2000,
    "temperature": 0.0,
    "stream": True,
    "stream_options": {
        "include_usage": True
    }
}

latency_list = []
target_pod_list = []

prefix_request_repeat = 3
for i in range(prefix_request_repeat):
    start_time = time.time()
    print(f"Sending request {i+1}")
    response = requests.post(url, headers=custom_headers2, json=data)
    end_time = time.time()
    latency_list.append(end_time - start_time)
    target_pod_list.append(response.headers['target-pod'])
    # print(response.text)
    print(end_time - start_time)
    # print(response.json())
    time.sleep(2)