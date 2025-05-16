#!/bin/bash

build=$1
tag=gangmuk-20250415-gatewaylog-00572df

if [ -z "$build" ]; then
    echo "build argument is empty"
    echo "Usage: ./build-gateway.sh <local|remote>"
    echo "exiting..."
    exit 1
fi

if [ "$build" == "local" ]; then
    ##############################
    ## for local docker registry only
    make docker-build-gateway-plugins
    docker tag aibrix/gateway-plugins:nightly aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gateway-plugins:$tag
else
    ##############################
    ## Remote push
    # build
    # make docker-build-gateway-plugins-amd64 # Use it when you build it on a mac but for intel server (vke).

    docker buildx build --platform linux/amd64 -t aibrix/gateway-plugins:nightly -f build/container/Dockerfile.gateway .

    # tag
    docker tag aibrix/gateway-plugins:nightly aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gateway-plugins:$tag
    # push
    docker push aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gateway-plugins:$tag
fi

# krrdgateway