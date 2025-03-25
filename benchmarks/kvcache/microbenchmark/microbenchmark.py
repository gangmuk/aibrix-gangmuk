from openai import OpenAI
import tiktoken
import json
import time
import random
import requests
import os

def tokenize(text, model_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(text))
    return encoding.encode(text), num_tokens

# routing_strategy = 'prefix-cache'
# routing_strategy = 'prefix-cache-and-load'
# routing_strategy = 'least-request'
# routing_strategy = 'random'
routing_strategy = 'roundrobin'

url = "http://localhost:8888/v1/chat/completions"

headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi',
    'routing-strategy': f'{routing_strategy}'
}

# Open the JSON file and load its data
# with open('data/message-threads.json', 'r') as file:
    # data = json.load(file)
# random.shuffle(data)

model = "llama-3-8b-instruct"

def print_response_all_headers(response):
    print(f"Response: {response}")
    print("Response Headers:")
    for key, value in response.headers.items():
        print(f"{key}: {value}")

# input_path = '/Users/bytedance/Downloads/medium_prompt.txt'
# input_path = '/Users/bytedance/Downloads/long_prompt.txt'
# input_path_list = ['/Users/bytedance/Downloads/medium_prompt.txt', '/Users/bytedance/Downloads/long_prompt.txt', '/Users/bytedance/Downloads/long_prompt-2.txt', , '/Users/bytedance/Downloads/long_prompt-3.txt']
# input_path_list = ['/Users/bytedance/Downloads/long_prompt-3.txt']
input_path_list = ['/Users/bytedance/Downloads/long_prompt-apple.txt']

prepend_random_string = True

request_id = 0
for input_path in input_path_list:
    seq_length_list = [4000]
    for seq_length in seq_length_list:
        aibrix_repo = "/Users/bytedance/projects/aibrix-routing-experiment"
        os.system(f"python3 {aibrix_repo}/benchmarks/utils/check_k8s_is_ready.py aibrix-controller-manager")
        os.system(f"python3 {aibrix_repo}/benchmarks/utils/check_k8s_is_ready.py aibrix-gateway-plugins")
        os.system(f"python3 {aibrix_repo}/benchmarks/utils/check_k8s_is_ready.py llama-3-8b-instruct")
        time.sleep(5)
        for repeat in range(2):
            with open(input_path, 'r') as file:
                prompt = file.read()
                if prepend_random_string:
                    random_string = "".join(random.choices("0123456789", k=8))
                    prompt = random_string + prompt
                prompt = prompt.split(' ')[:seq_length]
                prompt = ' '.join(prompt)
            encoded_requests, token_length = tokenize(prompt)
            print(f"word_length,{len(prompt.split(' '))},token_length,{token_length},routing,{routing_strategy}")
            for i in range(4):
                os.system("port-forward")
                req_data = {"model": model, "messages": [{"role": "user","content": prompt,},],"max_tokens": 1,"temperature": 0.0,}
                start_time = time.time()
                response = requests.post(url, headers=headers, json=req_data)
                if response.status_code != 200:
                    print(f"Error: {response.status_code}, {response.text}")
                    exit()
                end_time = time.time()
                print(f"request_id,{request_id},#{i},enable-prefix-caching,0,token_length,{token_length},routig,{routing_strategy},TTFT,{end_time - start_time:.2f},target-pod,{response.headers['target-pod']}")
                if i < 1:
                    # ((num of tokens / block size) * (a single persist iteration overhead)) / 1000 for ms to s
                    sleep_time = ((seq_length/16)*500)/1000
                    for i in range(10):
                        print(f"sleep for {sleep_time / 10}s (sleep idx: {i})")
                        time.sleep(sleep_time / 10)
                else:
                    time.sleep(10)
                request_id += 1

        os.system(f"kubectl rollout restart deploy aibrix-gateway-plugins -n aibrix-system")
        manifest_path = "/Users/bytedance/projects/huggingface-models/deployment_yamls/kv_cache_offloading/llama-8b-kv-cache-offloading-fix.yaml"
        os.system(f"kubectl delete -f {manifest_path}")
        time.sleep(5)
        os.system(f"kubectl apply -f {manifest_path}")