platform=linux/amd64
# platform=linux/arm64 

# docker buildx build --platform ${platform} --no-cache -t aibrix/gangmuk-latency-predictor:latest .
docker buildx build --platform ${platform} -t aibrix/gangmuk-latency-predictor:latest .

docker tag aibrix/gangmuk-latency-predictor:latest aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gangmuk-latency-predictor:latest

docker push aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gangmuk-latency-predictor:latest

kubectl rollout restart deploy latency-predictor-service