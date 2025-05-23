#!/bin/bash

target=$1

if [ -z "$target" ]; then
  echo "Please provide a target (e.g., 'remote' or 'local')."
  exit 1
fi
if [ "$target" == "remote" ]; then
    platform=linux/amd64
elif [ "$target" == "local" ]; then
    platform=linux/arm64
else
    echo "Invalid target. Use 'remote' or 'local'."
    exit 1
fi

# docker buildx build --platform ${platform} --no-cache -t aibrix/gangmuk-routing-agent:latest .
docker buildx build --platform ${platform} -t aibrix/gangmuk-routing-agent:latest .


if [ "$target" == "remote" ]; then
    docker tag aibrix/gangmuk-routing-agent:latest aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gangmuk-routing-agent:latest
    docker push aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gangmuk-routing-agent:latest
elif [ "$target" == "local" ]; then
    docker tag aibrix/gangmuk-routing-agent:latest aibrix/gangmuk-routing-agent:latest
    kubectl rollout restart deploy routing-agent-service
    kubectl rollout restart deploy aibrix-gateway-plugins -n aibrix-system
fi

# kubectl rollout restart deploy routing-agent-service
