#!/bin/bash

# input_workload_path="./workload/one_request.jsonl"
# input_workload_path="./workload/ten_requests.jsonl"
# input_workload_path="./workload/5s.jsonl"
# input_workload_path="./workload/5min-later-part-init.jsonl"
input_workload_path="./workload/prefix-sharing-workload/p1024_s128_rps5-p2048_s128_rps5-p4096_s128_rps5.jsonl"
# input_workload_path="./workload/prefix-sharing-workload/prefixsharingworkload-p1024_s128_rps5.jsonl"
max_tokens=100

api_key="sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi" # set your api key
output_jsonl_path="./output.jsonl"
model="llama-3-8b-instruct"
port=80
ipaddr=10.0.3.21
# port=8888
# ipaddr=localhost
routing="prefix-cache-and-load"
python3 async-client.py \
        --workload_path ${input_workload_path} \
        --model ${model} \
        --endpoint http://${ipaddr}:${port} \
        --api_key ${api_key} \
        --output_file_path ${output_jsonl_path} \
        --routing_strategy ${routing} \
        --max_tokens ${max_tokens} \
        --streaming