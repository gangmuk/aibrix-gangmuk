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

routing = 'prefix-cache-and-load'
# routing = 'prefix-cache'

url = "http://localhost:8888/v1/chat/completions"
custom_headers1 = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi',
    'routing-strategy': 'random'
}

custom_headers2 = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi',
    'routing-strategy': f'{routing}'
}

# Open the JSON file and load its data
with open('/Users/bytedance/Downloads/message-threads.json', 'r') as file:
    data = json.load(file)

random.shuffle(data)

client = OpenAI(
    api_key="sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi",
    base_url="http://localhost:8888/v1",
    default_headers={'routing-strategy': f'{routing}'},
    # default_headers={'routing-strategy': 'prefix-cache'},
)

token_length_bucket = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
num_samples_per_bucket = 5
random_index = 0
num_trials = 0
for i in range(len(token_length_bucket)-1):
    j = 0
    while j < num_samples_per_bucket:
        random_index = random.randint(0, len(data)-1)
        prompt = data[random_index]["userMessages"][0]
        if token_length_bucket[i] > count_tokens(prompt) or count_tokens(prompt) >= token_length_bucket[i+1]:
            num_trials += 1
            if num_trials > 100:
                print(f"Could not find a sample of the target length bucket after {num_trials} trials ({token_length_bucket[i]}-{token_length_bucket[i+1]}), skipping. this bucket")
                break
            continue
        print(f"token_length_bucket[{i}]: {token_length_bucket[i]}, token_length_bucket[{i+1}]: {token_length_bucket[i+1]}, token length, {count_tokens(prompt)}")
        num_trials = 0
        j += 1
        # req_data = {
        #     "model": "deepseek-llm-7b-chat",
        #     "messages": [
        #         {
        #             "role": "user",
        #             "content": prompt,
        #         },
        #     ],
        #     "max_tokens": 1,
        #     "temperature": 0.0,
        # }

        # start_time = time.time()
        # response = requests.post(url, headers=custom_headers1, json=req_data)
        # end_time = time.time()
        # print('TTFT random routing', end_time - start_time, "target-pod", response.headers['target-pod'])

        # # ----------------------------------------

        # req_data = {
        #     "model": "deepseek-llm-7b-chat",
        #     "messages": [
        #         {
        #             "role": "user",
        #             "content": prompt,
        #         },
        #     ],
        #     "max_tokens": 1,
        #     "temperature": 0.0,
        # }

        # start_time = time.time()
        # response = requests.post(url, headers=custom_headers2, json=req_data)
        # end_time = time.time()
        # print("#1: TTFT prefix-cache routing", end_time - start_time, "target-pod", response.headers['target-pod'])

        # # ----------------------------------------

        # req_data = {
        #     "model": "deepseek-llm-7b-chat",
        #     "messages": [
        #         {
        #             "role": "user",
        #             "content": prompt,
        #         },
        #     ],
        #     "max_tokens": 1,
        #     "temperature": 0.0,
        # }

        # start_time = time.time()
        # response = requests.post(url, headers=custom_headers2, json=req_data)
        # end_time = time.time()
        # print("#2: TTFT prefix-cache routing", end_time - start_time, "target-pod", response.headers['target-pod'])

        # # ----------------------------------------

        # start_time = time.time()
        # response_stream = client.chat.completions.create(
        #     model="deepseek-llm-7b-chat",
        #     messages=[{"role": "user", "content": prompt}],
        #     stream=True,
        #     stream_options={"include_usage": True},
        # )
        # for chunk in response_stream:
        #     continue
        #     # if chunk.choices:
        #     #     end_time = time.time()
        #     #     print('stream based ttft for prefix-cache routing ', end_time - start_time)
        #     #     break
        # print('\n\n')
        # time.sleep(2)