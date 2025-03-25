#!/bin/bash


#### choose workloads
# workload_path="prefix-workload/prefix-share-workload-p1024-s128-rps10.jsonl"
# workload_path="prefix-workload/prefix-share-workload-p3968-s128-rps10.jsonl"
# workload_path="prefix-workload/prefix-share-workload-p3968-s128-rps5.jsonl"
# workload_path="prefix-workload/prefix-share-workload-p3968-s128-rps5.jsonl"
# workload_path="prefix-workload/realistic-prefix-share-workload-p1024-s128-rps10.jsonl"
# workload_path="prefix-workload/realistic-prefix-share-workload-p3968-s128-rps5.jsonl"
# workload_path="prefix-workload/realistic-prefix-share-workload-p3968-s128-rps5-randomized.jsonl"
# workload_path="prefix-workload/realistic-prefix-share-workload-p3968-s128-rps10-randomized.jsonl"
# workload_path="prefix-workload/realistic-prefix-share-workload-p8192-s128-rps5-randomized.jsonl"
# workload_path="prefix-workload/realistic-prefix-share-workload-p8192-s128-rps8-randomized.jsonl"
workload_path="prefix-workload/realistic-prefix-share-workload-p8192-s128-rps10-randomized.jsonl"
# workload_path="prefix-workload/one_request.jsonl"
# workload_path="temp.jsonl"

#### choose routing policies
# routing_policies="random prefix-cache-and-load prefix-cache least-request least-kv-cache least-latency throughput"
routing_policies="least-request random prefix-cache-and-load prefix-cache"
# routing_policies="prefix-cache-and-load prefix-cache"
# routing_policies="random"


restart_deployment="true" # "true" or "false"
for wrk in ${workload_path}; do
    for routing in ${routing_policies}; do
        start_time=$(date +%s)
        echo "--------------------------------"
        echo "started experiment at $(date)"
        echo workload: ${wrk} 
        echo "The stdout/stderr is being logged in ./output.txt"
        echo "routing: ${routing}"
        echo "autoscaler: none"
        ./run-test.sh ${wrk} ${routing} ${restart_deployment} &> output.txt
        end_time=$(date +%s)
        echo "Done: Time taken: $((end_time-start_time)) seconds"
        echo "--------------------------------"
        sleep 10
    done
done
