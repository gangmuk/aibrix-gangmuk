#!/bin/bash

input_workload_path=$1
routing=$2
restart_deployment=$3
if [ -z "$restart_deployment" ]; then
    restart_deployment="true"
fi
num_replicas=8
aibrix_repo="/Users/bytedance/projects/aibrix-routing-experiment" # root dir of aibrix repo
api_key="sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi" # set your api key
# target_deployment="deepseek-llm-7b-chat"
# target_ai_model="deepseek-llm-7b-chat"
target_deployment="qwen2-5-7b-instruct"
target_ai_model="qwen2-5-7b-instruct"

echo "Make sure ${target_deployment} is the right deployment."
sleep 3

# Input validation
if [ -z "$aibrix_repo" ]; then
    echo aibrix_repo is empty. Set it to root dir of aibrix repo
    exit 1
fi
if [ -z "$api_key" ]; then
    echo "API key is not set. Please set the API key in the script"
    exit 1
fi
if [ -z "$input_workload_path" ]; then
    echo "input_workload_path is not given"
    echo "Usage: $0 <input_workload_path> <routing-policy>"
    exit 1
fi

# Setup experiment directory
workload_name=$(echo $input_workload_path | tr '/' '\n' | grep .jsonl | cut -d '.' -f 1)
experiment_result_dir="experiment_results/${target_ai_model}/${workload_name}/${routing}-$(date +%Y%m%d-%H%M%S)"
if [ ! -d ${experiment_result_dir} ]; then
    echo "output directory does not exist. Create the output directory (${experiment_result_dir})"
    mkdir -p ${experiment_result_dir}
fi

echo "----------------------------------------"
echo "workload_name: $workload_name"
echo "target_deployment: $target_deployment"
echo "routing: $routing"
echo "autoscaler: none"
echo "target_deployment: $target_deployment"
echo "input_workload_path: $input_workload_path"
echo "experiment_result_dir: $experiment_result_dir"
echo "----------------------------------------"

# Port-forwarding
# It is needed only when you run the client in the laptop not inside the K8S cluster.
kubectl -n envoy-gateway-system port-forward service/envoy-aibrix-system-aibrix-eg-903790dc 8888:80 &
PORT_FORWARD_PID=$!
echo "started port-forwarding with PID: $PORT_FORWARD_PID"

# Clean up any existing autoscalers
kubectl delete podautoscaler --all --all-namespaces
kubectl delete hpa --all --all-namespaces

if [ "$num_replicas" -eq -1 ]; then
    num_replicas=$(kubectl get deploy ${target_deployment} -n default -o jsonpath='{.spec.replicas}')
    echo "Current number of replicas: ${num_replicas}"
else
    echo "Set number of replicas to ${num_replicas}"
    python3 ${aibrix_repo}/benchmarks/utils/set_num_replicas.py --deployment ${target_deployment} --replicas ${num_replicas}
fi

if [ "$restart_deployment" == "true" ]; then
    echo "Restart aibrix-controller-manager deployment"
    kubectl rollout restart deploy aibrix-controller-manager -n aibrix-system

    echo "Restart aibrix-gateway-plugins deployment"
    kubectl rollout restart deploy aibrix-gateway-plugins -n aibrix-system
    echo "Restart ${target_deployment} deployment"

    kubectl rollout restart deploy ${target_deployment} -n default
    sleep_before_pod_check=20
    echo "Sleep for ${sleep_before_pod_check} seconds after restarting deployment"
    sleep ${sleep_before_pod_check}
fi

python3 ${aibrix_repo}/benchmarks/utils/check_k8s_is_ready.py aibrix-controller-manager
python3 ${aibrix_repo}/benchmarks/utils/check_k8s_is_ready.py aibrix-gateway-plugins
python3 ${aibrix_repo}/benchmarks/utils/check_k8s_is_ready.py ${target_deployment}

# Start pod log monitoring
pod_log_dir="${experiment_result_dir}/pod_logs"
mkdir -p ${pod_log_dir}

# Copy the snapshot of input workload
cp ${input_workload_path} ${experiment_result_dir}

# Start pod counter. It will run on background until the end of the experiment.
python3 ${aibrix_repo}/benchmarks/utils/count_num_pods.py ${target_deployment} ${experiment_result_dir} &
COUNT_NUM_POD_PID=$!
echo "started count_num_pods.py with PID: $COUNT_NUM_POD_PID"

# Streaming pod logs to files on the background
exclude="200 OK"
python3 ${aibrix_repo}/benchmarks/utils/streaming_pod_log_to_file.py ${target_deployment} default ${pod_log_dir} none ${exclude}  & pid_1=$!
python3 ${aibrix_repo}/benchmarks/utils/streaming_pod_log_to_file.py aibrix-controller-manager aibrix-system ${pod_log_dir} none none & pid_2=$!
python3 ${aibrix_repo}/benchmarks/utils/streaming_pod_log_to_file.py aibrix-gateway-plugins aibrix-system ${pod_log_dir} none none & pid_3=$!

# Run experiment!!!
output_jsonl_path=${experiment_result_dir}/output.jsonl
python3 ${aibrix_repo}/benchmarks/client/client.py \
    --workload-path ${input_workload_path} \
    --endpoint "http://localhost:8888" \
    --model ${target_ai_model} \
    --api-key ${api_key} \
    --output-file-path ${output_jsonl_path} \
    --routing-strategy ${routing} \
    --max-tokens 10 \
    --streaming

echo "Experiment is done. date: $(date)"

sleep 5
kill ${pid_1}
kill ${pid_2}
kill ${pid_3}
sleep 1

# Cleanup
kubectl delete podautoscaler --all --all-namespaces
# python3 ${aibrix_repo}/benchmarks/utils/set_num_replicas.py --deployment ${target_deployment} --replicas 1

# Stop monitoring processes
echo "Stopping monitoring processes..."
kill $COUNT_NUM_POD_PID
echo "killed count_num_pods.py with PID: $COUNT_NUM_POD_PID"

kill $PORT_FORWARD_PID
echo "killed port-forwarding with PID: $PORT_FORWARD_PID"

# Copy output file
cp output.txt ${experiment_result_dir}
echo "copied output.txt to ${experiment_result_dir}"
echo "Experiment completed."

# Cleanup function for handling interruption
cleanup() {
    echo "Cleaning up..."
    kill $PORT_FORWARD_PID 2>/dev/null
    kill $COUNT_NUM_POD_PID 2>/dev/null
    kill $pid_1 2>/dev/null
    kill $pid_2 2>/dev/null
    kill $pid_3 2>/dev/null
    kubectl delete podautoscaler --all --all-namespaces
    echo "Cleanup completed"
    exit
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM
