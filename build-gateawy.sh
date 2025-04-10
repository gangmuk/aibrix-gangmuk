#!/bin/bash

build=$1
tag=gangmuk-test

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
    docker tag aibrix/gateway-plugins:nightly aibrix/gateway-plugins:$tag
else
    ##############################
    ## Remote push
    # build
    make docker-build-gateway-plugins
    # tag
    docker tag aibrix/gateway-plugins:nightly aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gateway-plugins:$tag
    # push
    docker push aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gateway-plugins:$tag
fi

krrdgateway