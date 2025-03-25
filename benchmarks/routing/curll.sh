#!/bin/bash

curl -i -v http://localhost:8888/v1/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi" \
    -H "routing-strategy: prefix-cache-and-load" \
    -d '{"model": "deepseek-llm-7b-chat", "prompt": "Where is Beijing", "temperature": 0.0, "max_tokens": 4000}'


# curl -i -v http://localhost:8888/v1/completions \
#     -H "Content-Type: application/json" \
#     -H "Authorization: Bearer sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi" \
#     -H "routing-strategy: prefix-cache-and-load" \
#     -d '{"model": "qwen2-5-7b-instruct", "prompt": "Where is Beijing", "temperature": 0.0}'