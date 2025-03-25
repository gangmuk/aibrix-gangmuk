from openai import OpenAI
import tiktoken
import json
import time
import random
import requests
import time

def count_tokens(text, model_name="gpt-3.5-turbo"):
    """
    Counts the number of tokens in a text string using the tiktoken library.

    Args:
        text (str): The text string to count tokens in.
        model_name (str, optional): The name of the model to use for tokenization. 
                                     Defaults to "gpt-3.5-turbo".

    Returns:
        int: The number of tokens in the text string.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens

routing_strategy = 'prefix-cache-and-load'

url = "http://localhost:8888/v1/chat/completions"
custom_headers1 = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi',
    'routing-strategy': 'random'
}

custom_headers2 = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi',
    'routing-strategy': f'{routing_strategy}'
}

prefix_cache_headers = {
    'routing-strategy': f'{routing_strategy}'
}

# Open the JSON file and load its data
# with open('data/message-threads.json', 'r') as file:
    # data = json.load(file)
# random.shuffle(data)

client = OpenAI(
    api_key="sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi",
    base_url="http://localhost:8888/v1",
    default_headers={'routing-strategy': f'{routing_strategy}'},
)

# input_path = '/Users/bytedance/Downloads/medium_prompt.txt'
# input_path = '/Users/bytedance/Downloads/long_prompt.txt'
# input_path_list = ['/Users/bytedance/Downloads/medium_prompt.txt', '/Users/bytedance/Downloads/long_prompt.txt', '/Users/bytedance/Downloads/long_prompt-2.txt', , '/Users/bytedance/Downloads/long_prompt-3.txt']
input_path_list = ['/Users/bytedance/Downloads/long_prompt-3.txt']
for input_path in input_path_list:
    with open(input_path, 'r') as file:
        prompt = file.read()

    print('token length: ', count_tokens(prompt))
    req_data = {
        "model": "deepseek-llm-7b-chat",
        "messages": [
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "max_tokens": 1,
        "temperature": 0.0,
    }

    start_time = time.time()
    response = requests.post(url, headers=custom_headers1, json=req_data)
    end_time = time.time()
    print('TTFT random routing', end_time - start_time, "target-pod", response.headers['target-pod'])

    # ----------------------------------------

    req_data = {
        "model": "deepseek-llm-7b-chat",
        "messages": [
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "max_tokens": 1,
        "temperature": 0.0,
    }

    start_time = time.time()
    response = requests.post(url, headers=custom_headers2, json=req_data)
    end_time = time.time()
    print("#1: TTFT prefix-cache routing", end_time - start_time, "target-pod", response.headers['target-pod'])

    # ----------------------------------------

    req_data = {
        "model": "deepseek-llm-7b-chat",
        "messages": [
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "max_tokens": 1,
        "temperature": 0.0,
    }

    start_time = time.time()
    response = requests.post(url, headers=custom_headers2, json=req_data)
    end_time = time.time()
    print("#2: TTFT prefix-cache routing", end_time - start_time, "target-pod", response.headers['target-pod'])

    # ----------------------------------------

    start_time = time.time()
    response_stream = client.chat.completions.create(
        model="deepseek-llm-7b-chat",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in response_stream:
        continue
        # if chunk.choices:
        #     end_time = time.time()
        #     print('stream based ttft for prefix-cache routing ', end_time - start_time)
        #     break

    print('\n\n')
    time.sleep(2)



# for item in data:
#     data = {
#         "model": "deepseek-llm-7b-chat",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": item["userMessages"][0],
#             },
#         ],
#         "max_tokens": 1,
#         "temperature": 0.0,
#     }

#     start_time = time.time()
#     response = requests.post(url, headers=custom_headers1, json=data)
#     end_time = time.time()
#     ttft1 = end_time - start_time
#     tp1 = response.headers['target-pod']

#     start_time = time.time()
#     response = requests.post(url, headers=custom_headers2, json=data)
#     end_time = time.time()
#     ttft2 = end_time - start_time
#     tp2 = response.headers['target-pod']

#     start_time = time.time()
#     response = requests.post(url, headers=custom_headers2, json=data)
#     end_time = time.time()
#     ttft3 = end_time - start_time
#     tp3 = response.headers['target-pod']

#     print("data", item["userMessages"][0])
#     print("\n")
#     print("w/o prefix cache", tp1, ttft1, "with prefix cache", tp2, ttft2, "--", tp3, ttft3)
#     print("\n\n")
#     time.sleep(5)

#     # start_time = time.time()
#     # response = requests.post(url, headers=custom_headers2, json=data)
#     # end_time = time.time()
#     # ttft4 = end_time - start_time
#     # tp4 = response.headers['target-pod']

#     # start_time = time.time()
#     # response = requests.post(url, headers=custom_headers2, json=data)
#     # end_time = time.time()
#     # ttft5 = end_time - start_time
#     # tp5 = response.headers['target-pod']

#     # start_time = time.time()
#     # response = requests.post(url, headers=custom_headers2, json=data)
#     # end_time = time.time()
#     # ttft6 = end_time - start_time
#     # tp6 = response.headers['target-pod']

#     # print("data", item["userMessages"][0])
#     # print("\n")
#     # print("w/o prefix cache", tp1, ttft1, "with prefix cache", tp2, ttft2, "--", tp3, ttft3, "--", tp4, ttft4, "--", tp5, ttft5, "--", tp6, ttft6)
#     # print("\n\n")
#     # time.sleep(5)