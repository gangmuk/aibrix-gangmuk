#!/bin/bash

request="$1"
routing=$2
model=llama-3-8b-instruct
temp=0.0
max_tokens=10
auth_token="sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi"

port=80
ipaddr=10.0.3.21 # external ip address of envoy-aibrix-system-aibrix-eg-903790dc service in envoy-gateway-system

# port=8888
# ipaddr=localhost

if [ -z "$routing" ]; then
    routing="random"
fi

if [ -z "$request" ]; then
    probability=$(($RANDOM % 100))
    if [ $probability -le 33 ]; then
        request="I like apple"
    elif [ $probability -le 66 ]; then
        request="I like orange"
    else
        request="I like orange very much"
    fi
fi


curl_command='curl -v -i "http://${ipaddr}:${port}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${auth_token}" \
    -H "routing-strategy: ${routing}" \
    -H "model: ${model}" \
    -d "{
        \"model\": \"${model}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"${request}\"}], \
        \"temperature\": ${temp}, \
        \"max_tokens\": ${max_tokens} \
    }"'

# Echo the curl command
echo "The curl command is: $curl_command"

# Execute the curl command
eval "$curl_command"
echo ""
