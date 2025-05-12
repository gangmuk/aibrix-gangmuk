platform=linux/amd64
# platform=linux/arm64 

docker buildx build --platform ${platform} -t aibrix/gangmuk-latency-predictor:latest .

docker tag aibrix/gangmuk-latency-predictor:latest aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gangmuk-latency-predictor:latest

docker push aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gangmuk-latency-predictor:latest