#!/bin/bash

workload=$1

##########################################

max_tokens=4096
if [ "${workload}" == "one" ]; then
    input_workload_path="./workload/one_request.jsonl"
    max_tokens=100
elif [ "${workload}" == "ten" ]; then
    input_workload_path="./workload/ten_requests.jsonl"
elif [ "${workload}" == "5s" ]; then
    input_workload_path="./workload/5s.jsonl"
elif [ "${workload}" == "5min" ]; then
    input_workload_path="./workload/5min-later-part-init.jsonl"
elif [ "${workload}" == "prefix-sharing" ]; then
    input_workload_path="./workload/prefix-sharing-workload/realistic-prefix-share-workload.jsonl"
else
    # input_workload_path="./workload/one_request.jsonl"
    input_workload_path="./workload/simple_ten_requests.jsonl"
fi
echo "workload: ${input_workload_path}"

##########################################

api_key="sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi" # set your api key
output_jsonl_path="./output.jsonl"

# model="deepseek-llm-7b-chat"
# model="llama2-7b"
model="llama-3-8b-instruct"
# port=80 # local k8s cluster context
port=8888 # remote k8s cluster context
# routing="prefix-cache-and-load"
routing="random"


python3 client-new.py \
    --workload-path ${input_workload_path} \
    --endpoint http://localhost:${port} \
    --model ${model} \
    --api-key ${api_key} \
    --output-file-path ${output_jsonl_path} \
    --routing-strategy ${routing} \
    --max-tokens ${max_tokens}
    # --streaming