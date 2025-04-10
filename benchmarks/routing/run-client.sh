#!/bin/bash

aibrix_repo="/Users/bytedance/projects/aibrix/aibrix-gangmuk-kvcache"
input_workload_path="/Users/bytedance/projects/aibrix/aibrix-gangmuk-kvcache/benchmarks/routing/workload/one_request.jsonl"
target_ai_model="deepseek-llm-7b-chat"
api_key="sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi" # set your api key
output_jsonl_path="/Users/bytedance/projects/aibrix/aibrix-gangmuk-kvcache/benchmarks/routing/output.jsonl"
routing="random"

python3 ${aibrix_repo}/benchmarks/client/client.py \
    --workload-path ${input_workload_path} \
    --endpoint "http://localhost:8888" \
    --model ${target_ai_model} \
    --api-key ${api_key} \
    --output-file-path ${output_jsonl_path} \
    --routing-strategy ${routing} \
    --streaming