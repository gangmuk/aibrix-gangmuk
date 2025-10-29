#!/bin/bash

# Script to run the client in a K8s pod using kubectl exec

set -e

# Configuration
NAMESPACE=${NAMESPACE:-default}
POD_NAME=${POD_NAME:-client-service}
CONTAINER_NAME=${CONTAINER_NAME:-client}
k8s_cluster="vke"

routing_policy_list=(
    # "prefix_cache_1"
    "latency_predictor"
    # "random"
    # "least-latency"
    # "least-request"
    # "least-kv-cache"
    # "preble"
    # # "prefix_cache_2"
    # "scalable_rl_agent"
)

# workload_name="ten_request"
workload_name_list=(
    # "ten_request" # 20
    "hundred_request" # 99
    # "SharingRatio9%" # 2053, 265875 (5min)
    # "SharingRatio28%" # 1999, 259697 (5min)
    # "SharingRatio47%" # 2313, 299803
    # "SharingRatio71%" # 1500, 346602
    # "MixedSharingRatio10_30_50_70%" # 4000
    # "multiturn-chat" # 4752, avg token per turn: 1141, input: 100-3700, avg input len: 1141, sharing ratio: 0.7

    # "mooncake-conversation" # 2674, input: 200-8000, avg input: 2354, sharing ratio: 0.04
    # "mooncake-toolagent" # 2713, input: 200-8000, avg input: 1738, sharing ratio: 0.33
    # "text_to_sql"

    # "old-version/input10002000/rps2030/SharingRatio10%_avginput1500_globalrps20"
    # "old-version/input10002000/rps2030/SharingRatio30%_avginput1500_globalrps20"
    # "old-version/input10002000/rps2030/SharingRatio50%_avginput1500_globalrps25"
    # "old-version/input10002000/rps2030/SharingRatio70%_avginput1500_globalrps35"

    # "old-version/input100020004000/SharingRatio10%_avginput2333_globalrps10/"
    # "old-version/input100020004000/SharingRatio30%_avginput2333_globalrps10/"
    # "old-version/input100020004000/SharingRatio50%_avginput2333_globalrps10/"
    # "old-version/input100020004000/SharingRatio70%_avginput2333_globalrps12/"

    # "old-version/SharingRatio10%_avginput2333_globalrps15/"
    # "old-version/SharingRatio30%_avginput2333_globalrps15/"
    # "old-version/SharingRatio50%_avginput2333_globalrps16/"
    # "old-version/SharingRatio70%_avginput2333_globalrps20/"

    # "config_sharing10%"
    # "config_sharing30%"
    # "config_sharing50%"
    # "config_sharing70%"
)

# max_input_tokens=8000
target_gpu="L20" # "A30"
EXPLORATION_RATE="0.1"

ENABLE_ONLINE_LEARNING="1"
MIN_NUM_TRAINING_DATA="50"
MIN_NUM_UPDATE_DATA="10"
ENABLE_FLUSH="1"
FLUSH_PERIOD="10"
MIN_NUM_LOG_MESSAGES_TO_FLUSH="100"
EXPLORATION_ENABLED="0"


# Configuration for the client
api_key="sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi"
# POD_LABEL_SELECTOR="llama2-7b"
# POD_LABEL_SELECTOR="tinyllama"
POD_LABEL_SELECTOR="model.aibrix.ai/name=llama-3-8b-instruct"
llm_model="llama-3-8b-instruct"
# POD_LABEL_SELECTOR="model.aibrix.ai/name=llama3-1-8b"
# llm_model="llama3-1-8b"


## External IP of client-service svc
ipaddr="115.190.180.7" # vke cluster
## CLUSTER-IP of client-service svc
# ipaddr="10.102.24.174" # local cluster
port=80

rps_list=(
    # 8 # SharingRatio71
    # 10 # MixedSharingRatio10_30_50_70, mooncake conversation and toolagent
    # 10 # MixedSharingRatio10_30_50_70, mooncake conversation and toolagent
    # 16 # Multiturn Chat tried but not saturate
    # 18 # Multiturn Chat, saturate but not working...
    12
)

for rps in "${rps_list[@]}"; do
    cut_done=0
    for workload_name in "${workload_name_list[@]}"; do
        if [ "${workload_name}" == "SharingRatio9%" ]; then
            rps=7.5 # works in 7*L20
            # rps=12 # 7*L20 + 8*A30
            max_tokens=50
            max_tokens_std=5
            total_num_episodes=4
        elif [ "${workload_name}" == "SharingRatio28%" ]; then
            # rps=8 # works
            # rps=12 # 7*L20 + 8*A30
            max_tokens=50
            max_tokens_std=5
            total_num_episodes=6
        elif [ "${workload_name}" == "SharingRatio47%" ]; then
            # rps=8 # works
            # rps=12 # 7*L20 + 8*A30
            max_tokens=50
            max_tokens_std=5
            total_num_episodes=6
        elif [ "${workload_name}" == "SharingRatio71%" ]; then
            # rps=10 # 8 is same as prefix cache. 9, 10, etc do not work due to kvcache usage hitting 100%.... fuck
            # rps=12 # 7*L20 + 8*A30
            max_tokens=50
            max_tokens_std=5
            total_num_episodes=8
        elif [ "${workload_name}" == "MixedSharingRatio10_30_50_70%" ]; then
            # rps=8
            # rps=12 # 7*L20 + 8*A30
            max_tokens=50
            max_tokens_std=5
            total_num_episodes=4
        elif [ "${workload_name}" == "multiturn-chat" ]; then
            # rps=12
            max_tokens=50
            max_tokens_std=10
            total_num_episodes=6
        elif [ "${workload_name}" == "mooncake-conversation" ]; then
            # rps=10
            max_tokens=50
            max_tokens_std=10
            total_num_episodes=4
        elif [ "${workload_name}" == "mooncake-toolagent" ]; then
            # rps=7
            max_tokens=50
            max_tokens_std=5
            total_num_episodes=4
        elif [ "${workload_name}" == "hundred_request" ]; then
            rps=10
            max_tokens=50
            max_tokens_std=10
            total_num_episodes=50
        else
            # rps=8
            max_tokens=50
            max_tokens_std=10
            total_num_episodes=6
        fi

        for routing_policy in "${routing_policy_list[@]}"; do
            delimiter="+"
            if [ "${routing_policy}" == "preble" ]; then
                config="preble${delimiter}${routing_policy}"
            else
                config="rl-online-router${delimiter}${routing_policy}"
            fi
            routing="${config%%${delimiter}*}"
            subAlgorithm="${config#*${delimiter}}"
            if [ "${subAlgorithm}" != "latency_predictor" ] && [ "${cut_done}" == "0" ]; then
                cut_done=1
                total_num_episodes=$((total_num_episodes/2))
                echo "subAlgorithm: ${subAlgorithm}, cut half total_num_episodes: ${total_num_episodes}"
            else
                echo "subAlgorithm: ${subAlgorithm}, no need to cut total_num_episodes: ${total_num_episodes}"
            fi

            ship_model=0
            ship_code=1
            if [ "${subAlgorithm}" == "latency_predictor" ]; then
                ship_offline_training_data=1
            else
                ship_offline_training_data=0
            fi
            ship_offline_training_data=0

            if [ "${routing_policy}" == "scalable_rl_agent" ]; then
                final_model_dir="../training_data/scalable_rl_agent/final_model"
            else
                if [ "${target_gpu}" == "L20" ]; then
                    if [ "${workload_name}" == "multiturn-chat" ]; then
                        final_model_dir="../workload-and-experiment_results/multiturn-chat/training_data/final_model-latency_predictor_ttft-20251027_233030"
                    else
                        final_model_dir="../training_data/L20-7/merged-data/all-with-mixed/final_model-latency_predictor_ttft"
                        # final_model_dir="../training_data/L20-7/merged-data/all/final_model-latency_predictor_ttft"
                    fi
                elif [ "${target_gpu}" == "A30" ]; then
                    final_model_dir="../training_data/A30-8/old-version/final_model-latency_predictor_ttft-20251025_231816"
                else
                    final_model_dir="../training_data/L20-7/merged-data/all/final_model-latency_predictor_ttft"
                fi
            fi

            if [ "${ship_model}" == "1" ] && [ ! -d "${final_model_dir}" ]; then
                echo "Error: Final model directory does not exist: ${final_model_dir}"
                echo "Exiting..."
                exit 1
            fi
            if [ ! -f "${final_model_dir}/model_config.json" ]; then
                echo "Error: model_config.json does not exist: ${final_model_dir}/model_config.json"
                echo "Exiting..."
                exit 1
            fi
            if [ ! -f "${final_model_dir}/latency_predictor.pth" ]; then
                echo "Error: latency_predictor.pth does not exist: ${final_model_dir}/latency_predictor.pth"
                echo "Exiting..."
                exit 1
            fi
            if [ ! -f "${final_model_dir}/feature_normalization_statistics.csv" ]; then
                echo "Error: feature_normalization_statistics.csv does not exist: ${final_model_dir}/feature_normalization_statistics.csv"
                echo "Exiting..."
                exit 1
            fi
            echo "========================================="
            echo "!!! workload_name: ${workload_name} !!!"
            echo "!!! final_model_dir: ${final_model_dir} !!!"
            echo "========================================="

            if [ "${routing_policy}" == "scalable_rl_agent" ]; then
                training_epochs=10
                num_requests_per_episode=500
                num_iterations=20
                num_episodes_per_iteration=5
                n_eval_episodes=5
                total_num_episode_for_training=$((num_iterations * num_episodes_per_iteration))
                total_num_episode_for_evaluation=$((num_iterations * n_eval_episodes))
                total_num_episodes=$((total_num_episode_for_training + total_num_episode_for_evaluation))
                batch_size=256
                echo "*******************************************"
                echo "num_iterations: ${num_iterations}"
                echo "num_episodes_per_iteration: ${num_episodes_per_iteration}"
                echo "n_eval_episodes: ${n_eval_episodes}"
                echo "total_num_episode_for_training: ${total_num_episode_for_training}"
                echo "total_num_episode_for_evaluation: ${total_num_episode_for_evaluation}"
                echo "total_num_episodes: ${total_num_episodes}"
                echo "*******************************************"
                python update_model_config.py \
                    --final_model_dir "../training_data/scalable_rl_agent/final_model" \
                    --num_requests_per_episode ${num_requests_per_episode} \
                    --num_iterations ${num_iterations} \
                    --num_episodes_per_iteration ${num_episodes_per_iteration} \
                    --n_eval_episodes ${n_eval_episodes} \
                    --training_epochs ${training_epochs} \
                    --batch_size ${batch_size}
            fi

            echo "Starting to update k8s env for routing-agent-service"
            python3 update_k8s_env.py \
                --deployment routing-agent-service \
                --namespace default \
                --container routing-agent \
                --env EXPLORATION_ENABLED=${EXPLORATION_ENABLED} \
                --env MIN_NUM_TRAINING_DATA=${MIN_NUM_TRAINING_DATA} \
                --env ENABLE_ONLINE_LEARNING=${ENABLE_ONLINE_LEARNING} \
                --env POD_LABEL_SELECTOR=${POD_LABEL_SELECTOR} \
                --env EXPLORATION_RATE=${EXPLORATION_RATE}
            
            echo "Starting to update k8s env for aibrix-gateway-plugins"
            python3 update_k8s_env.py \
                --deployment aibrix-gateway-plugins \
                --namespace aibrix-system \
                --container gateway-plugin \
                --env ENABLE_FLUSH=${ENABLE_FLUSH} \
                --env FLUSH_PERIOD=${FLUSH_PERIOD} \
                --env MIN_NUM_LOG_MESSAGES_TO_FLUSH=${MIN_NUM_LOG_MESSAGES_TO_FLUSH} \
                --env useRealRequest=1
                # --env LATENCY_METRICS_LOG_PATH=/path/to/your/metrics.log

            ship_start_time=$(date +%s)
            echo "Starting to ship all files to pods"
            python ship_all.py --ship_code ${ship_code} --ship_model ${ship_model} --final_model_dir ${final_model_dir} --k8s_cluster ${k8s_cluster} --ship_offline_training_data ${ship_offline_training_data}
            
            if [ "${routing_policy}" == "scalable_rl_agent" ]; then
                scalable_rl_agent_init_model_dir="../training_data/scalable_rl_agent/init_model"
                python kubectl_cp_from_host_to_pod.py ${scalable_rl_agent_init_model_dir} /app/final_model routing-agent-service default
            fi
            
            python kubectl_cp_from_host_to_pod.py async-client.py /app client-service default

            ship_end_time=$(date +%s)
            ship_took=$((ship_end_time - ship_start_time))
            echo "* ship_all took: ${ship_took}s"

            python3 check_ready.py --deployment aibrix-gateway-plugins --namespace aibrix-system
            python3 check_ready.py --deployment routing-agent-service --namespace default
            python3 check_ready.py --deployment llama-3-8b-instruct --namespace default
            sleep 5

            echo "========================================="
            echo "Running Client in K8s Pod"
            echo "========================================="
            echo "Namespace:           ${NAMESPACE}"
            echo "Pod:                 ${POD_NAME}"
            echo "Container:           ${CONTAINER_NAME}"
            echo "Routing Strategy:    ${routing}"
            echo "Sub-Algorithm:       ${subAlgorithm}"
            echo "Workload:            ${workload_name}"
            echo "Online Learning:     ${ENABLE_ONLINE_LEARNING}"
            echo "total_num_episodes:          ${total_num_episodes}"
            echo "Max Tokens:          ${max_tokens}"
            echo "Max Tokens Std:      ${max_tokens_std}"
            echo "========================================="

            # Find the actual pod name (in case of deployment with generated suffix)
            ACTUAL_POD=$(kubectl get pods -n ${NAMESPACE} -l app=${POD_NAME} -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

            if [ -z "$ACTUAL_POD" ]; then
                echo "Error: No pod found with label app=${POD_NAME} in namespace ${NAMESPACE}"
                echo "Trying to use pod name directly: ${POD_NAME}"
                ACTUAL_POD=${POD_NAME}
            fi

            echo "Using pod: ${ACTUAL_POD}"

            # Wait for pod to be ready
            echo "Waiting for pod to be ready..."
            kubectl wait --for=condition=ready pod/${ACTUAL_POD} -n ${NAMESPACE} --timeout=60s || {
                echo "Error: Pod did not become ready within 60 seconds"
                kubectl describe pod ${ACTUAL_POD} -n ${NAMESPACE}
                exit 1
            }

            workload_path="/app/workload/${workload_name}/workload.jsonl"
            output_dir="/app/output/${workload_name}-${subAlgorithm}-$(date +%Y%m%d_%H%M%S)"
            output_jsonl_path="${output_dir}/output.jsonl"

            # Create output directory in pod
            echo "Creating output directory in pod..."
            kubectl exec -n ${NAMESPACE} ${ACTUAL_POD} -c ${CONTAINER_NAME} -- mkdir -p ${output_dir}

            # Check if workload file exists
            echo "Checking if workload file exists..."
            kubectl exec -n ${NAMESPACE} ${ACTUAL_POD} -c ${CONTAINER_NAME} -- test -f ${workload_path} || {
                echo "Error: Workload file ${workload_path} not found in pod"
                echo "Available workloads:"
                kubectl exec -n ${NAMESPACE} ${ACTUAL_POD} -c ${CONTAINER_NAME} -- find /app/workload -name "*.jsonl"
                exit 1
            }

            

            # Create local experiment result output directory
            timestamp=$(date +%Y%m%d_%H%M%S)
            experiment_result_output_dir="../workload-and-experiment_results/hetero/${workload_name}/${subAlgorithm}"
            if [ "${subAlgorithm}" == "rl_agent" ]; then
                postfix="total_num_episodes${total_num_episodes}"
                experiment_result_output_dir="${experiment_result_output_dir}-${postfix}"
            elif [ "${subAlgorithm}" == "rl_naive" ]; then
                trained_model_data_name=$(echo "$final_model_dir" | awk -F'training_data/' '{print $2}' | cut -d'/' -f1)
                used_data_name=$(echo "$final_model_dir" | awk -F'training_data/' '{print $2}' | cut -d'/' -f2)
                hyperparameter_name=$(echo "$final_model_dir" | awk -F'processed-' '{print $2}')
                hyperparameter_name="${hyperparameter_name}-explr_${EXPLORATION_ENABLED}"
                postfix="onlinelearning_${ENABLE_ONLINE_LEARNING}-trained_on_${trained_model_data_name}_${used_data_name}-${hyperparameter_name}-total_num_episodes${total_num_episodes}"
                experiment_result_output_dir="${experiment_result_output_dir}-${postfix}"
            elif [ "${subAlgorithm}" == "latency_predictor" ]; then
                trained_model_data_name=$(echo "$final_model_dir" | awk -F'training_data/' '{print $2}' | cut -d'/' -f1)
                prediction_metric=$(echo "$final_model_dir" | awk -F'latency_predictor_' '{print $2}')
                used_data_name=$(echo "$final_model_dir" | awk -F'training_data/' '{print $2}' | cut -d'/' -f2)
                postfix="trained_on_${trained_model_data_name}_${used_data_name}"
                experiment_result_output_dir="${experiment_result_output_dir}_${prediction_metric}"
            fi
            experiment_result_output_dir="${experiment_result_output_dir}-${target_gpu}-rps${rps}-${postfix}-iter${total_num_episodes}-${timestamp}"

            echo "* experiment_result_output_dir: ${experiment_result_output_dir}"
            if [ ! -d "${experiment_result_output_dir}" ]; then
                mkdir -p "${experiment_result_output_dir}"
            fi

            echo "Starting log collection..."
            
            # Function to collect logs with timestamp tracking to avoid duplicates
            collect_logs_continuously() {
                local namespace=$1
                local pod_pattern=$2
                local output_file=$3
                local interval=30  # Collect logs every 30 seconds
                
                local since_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
                
                while true; do
                    pod_name=$(kubectl get pods -n ${namespace} | grep ${pod_pattern} | awk '{print $1}' | head -n 1)
                    if [ -n "$pod_name" ]; then
                        # Get logs since last collection time
                        kubectl logs --since-time="${since_time}" -n ${namespace} ${pod_name} >> ${output_file} 2>&1
                        # Update timestamp for next iteration
                        since_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
                    fi
                    sleep ${interval}
                done
            }
            
            # Function to copy checkpoints periodically
            copy_checkpoints_periodically() {
                local output_dir=$1
                local interval=300  # Copy checkpoints every 5 minutes (300 seconds)
                
                # Wait a bit before first copy to let training start
                sleep 60
                
                while true; do
                    checkpoint_ts=$(date +%Y%m%d_%H%M%S)
                    echo "[$(date)] Copying checkpoints at ${checkpoint_ts}..."
                    python kubectl_cp_from_pod_to_host.py /app/final_model/checkpoints "${output_dir}/checkpoints_${checkpoint_ts}" routing-agent-service default 2>&1 | grep -v "tar: Removing leading"
                    echo "[$(date)] Checkpoint copy completed: checkpoints_${checkpoint_ts}"
                    sleep ${interval}
                done
            }
            
            # Start log collection in background
            collect_logs_continuously "aibrix-system" "aibrix-gateway-plugins" "${experiment_result_output_dir}/all-aibrix-gateway-plugins.log.txt" &
            pid_1=$!
            collect_logs_continuously "default" "routing-agent-service" "${experiment_result_output_dir}/all-routing-agent-service.log.txt" &
            pid_2=$!
            
            # Start periodic checkpoint copying for scalable_rl_agent
            if [ "${subAlgorithm}" == "scalable_rl_agent" ]; then
                echo "Starting periodic checkpoint copying (every 5 minutes)..."
                copy_checkpoints_periodically "${experiment_result_output_dir}" &
                pid_checkpoint=$!
            fi

            echo "Starting client in pod..."
            echo "Output will be saved to: ${output_dir}"

            # ./curl.sh && sleep 0.5 && ./curl.sh && sleep 0.5 && ./curl.sh && sleep 3

            # Run the client using kubectl exec
            kubectl exec -n ${NAMESPACE} ${ACTUAL_POD} -c ${CONTAINER_NAME} -- \
                python3 /app/async-client.py \
                    --workload_path ${workload_path} \
                    --model ${llm_model} \
                    --endpoint http://${ipaddr}:${port} \
                    --api_key ${api_key} \
                    --output_file_path ${output_jsonl_path} \
                    --routing_strategy ${routing} \
                    --subAlgorithm ${subAlgorithm} \
                    --max_tokens ${max_tokens} \
                    --max-tokens-std ${max_tokens_std} \
                    --output_dir ${output_dir} \
                    --prompt-type chat \
                    --rps ${rps} \
                    --poisson-arrivals \
                    --shuffle-requests \
                    --iterations ${total_num_episodes} \
                    --streaming \
                    2>&1 | tee ${experiment_result_output_dir}/client.log.txt


            sleep 5
            # kubectl rollout restart deployment llama-3-8b-instruct

            
            # Copy final model
            echo "Copying final_model from pod..."
            python kubectl_cp_from_pod_to_host.py /app/final_model "${experiment_result_output_dir}/final_model" routing-agent-service default

            # python kubectl_cp_from_pod_to_host.py /tmp/latency_metrics.log "${experiment_result_output_dir}/latency_metrics.log.txt" gateway-plugins aibrix-system

            # Copy checkpoints (for scalable_rl_agent)
            if [ "${subAlgorithm}" == "scalable_rl_agent" ]; then
                checkpoint_ts=$(date +%Y%m%d_%H%M%S)
                echo "Copying scalable RL agent checkpoints..."
                python kubectl_cp_from_pod_to_host.py /app/final_model/checkpoints "${experiment_result_output_dir}/checkpoints_${checkpoint_ts}" routing-agent-service default || echo "⚠️  No checkpoints found (agent may not have trained enough steps)"
            fi

            # Process logs
            cat ${experiment_result_output_dir}/all-aibrix-gateway-plugins.log.txt | grep "**@latency_metrics" | grep -v "infer:" > ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins.log.csv
            echo "* all gateway log: ${experiment_result_output_dir}/all-aibrix-gateway-plugins.log.txt"
            echo "* filtered gateway log: ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins.log.csv"
            echo "* routing agent log: ${experiment_result_output_dir}/all-routing-agent-service.log.txt"
            echo "* client log: ${experiment_result_output_dir}/client.log.txt"
            if [ "${subAlgorithm}" == "scalable_rl_agent" ]; then
                echo "* checkpoints (periodic snapshots): ${experiment_result_output_dir}/checkpoints_*/"
                echo "* final checkpoint: ${experiment_result_output_dir}/checkpoints_${checkpoint_ts}/"
            fi
            python plot_latency_timeseries.py ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins.log.csv
            
            # Kill background processes
            kill $pid_1 2>/dev/null || true
            kill $pid_2 2>/dev/null || true
            if [ "${subAlgorithm}" == "scalable_rl_agent" ] && [ -n "${pid_checkpoint}" ]; then
                kill $pid_checkpoint 2>/dev/null || true
            fi

            kubectl rollout restart deployment client-service
        done
    done
done