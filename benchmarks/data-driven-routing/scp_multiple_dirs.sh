#!/bin/bash

dirs=(
    "basic-load-patterns"
    "large-context-testing"
    "burst-patterns"
    "output-size-impact"
    # "chatbot-simulation"
    "prefix-sharing-efficiency"
    "production-like-mixed-load"
    "content-generation"
    "quick-qa"
    "input-size-impact"
)

for dir in "${dirs[@]}"; do
    python scp_filtered_gateway_log.py root@180.184.82.203 /mnt/vdb/data-driven-routing/workload/prefix-sharing-workload/comprehensive_set/random/${dir} --local-dir filtered_logs
done