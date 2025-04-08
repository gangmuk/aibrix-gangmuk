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

def send_request(model_, prompt_, url_, headers_):
    req_data = {"model": model_, "messages": [{"role": "user", "content": prompt_}], "max_tokens": 1, "temperature": 0.0}
    response = requests.post(url_, headers=headers_, json=req_data)
    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")
        exit()
    return response

manifest_path = "./deployment/llama-8b-kv-cache-offloading-fix.yaml"
py_pod_ready_check = "/Users/bytedance/projects/aibrix-routing-experiment/benchmarks/utils/check_k8s_is_ready.py"
def reset_deployment():
    os.system(f"kubectl rollout restart deploy aibrix-gateway-plugins -n aibrix-system")
    if not os.path.exists(manifest_path):
        print(f"Error: {manifest_path} does not exist")
        exit()
    os.system(f"kubectl delete -f {manifest_path}")
    time.sleep(5)
    os.system(f"kubectl apply -f {manifest_path}")
    time.sleep(20)
    os.system(f"python3 {py_pod_ready_check} aibrix-controller-manager")
    os.system(f"python3 {py_pod_ready_check} aibrix-gateway-plugins")
    os.system(f"python3 {py_pod_ready_check} llama-3-8b-instruct")
    time.sleep(5)

def prepend_random_string_to_prompt(prompt):
    random_string = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))
    return f"{random_string} {prompt}"

# input_path = '/Users/bytedance/Downloads/medium_prompt.txt'
# input_path = '/Users/bytedance/Downloads/long_prompt.txt'
# input_path_list = ['/Users/bytedance/Downloads/medium_prompt.txt', '/Users/bytedance/Downloads/long_prompt.txt', '/Users/bytedance/Downloads/long_prompt-2.txt', , '/Users/bytedance/Downloads/long_prompt-3.txt']
input_path_list = ['/Users/bytedance/Downloads/long_prompt-3.txt']
# input_path_list = ['/Users/bytedance/Downloads/long_prompt-apple.txt']

os.system(f"python3 {py_pod_ready_check} llama-3-8b-instruct")
prepend_random_string = True
request_id = 0
for input_path in input_path_list:
    seq_length_list = [5000]
    for seq_length in seq_length_list:
        # reset_deployment()
        for repeat in range(1):
            prompt = ""
            with open(input_path, 'r') as file:
                prompt = file.read()
            if prepend_random_string:
                prompt = prepend_random_string_to_prompt(prompt)
            if len(prompt) == 0:
                print(f"Error: {input_path} is empty")
                exit()
            prompt = prompt.split(' ')
            prompt = prompt[:seq_length]
            prompt = ' '.join(prompt)
            encoded_requests, token_length = tokenize(prompt)
            warmup_prompt = "What is the capital of France?"
            _ = send_request(model, warmup_prompt, url, headers)
            _ = send_request(model, warmup_prompt, url, headers)
            time.sleep(5)
            for i in range(4):
                os.system("port-forward")
                start_time = time.time()
                response = send_request(model, prompt, url, headers)
                end_time = time.time()
                print(f"request_id,{request_id},#{i},enable-prefix-caching,0,token_length,{token_length},routig,{routing_strategy},TTFT,{end_time - start_time:.2f},target-pod,{response.headers['target-pod']}")
                if i < 1:
                    sleep_time = ((seq_length/16)*400)/1000 # ((num of tokens / block size) * (a single persist iteration overhead)) / 1000 for ms to s
                    print(f"sleep for {sleep_time}s (sleep idx: {i}), ", end="", flush=True)
                    for i in range(10):
                        print(f"{i} ", end="", flush=True)
                        time.sleep(sleep_time / 10)
                    print()
                else:
                    time.sleep(10)
                request_id += 1
            print("-"*50)
