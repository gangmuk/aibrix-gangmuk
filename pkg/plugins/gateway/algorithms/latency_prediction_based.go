package routingalgorithms

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/aibrix/pkg/cache"
	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
	"github.com/vllm-project/aibrix/pkg/utils/prefixcacheindexer"
	"github.com/vllm-project/aibrix/pkg/utils/tokenizer"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

var (
	predictorServiceURL = "http://latency-predictor-service.default.svc.cluster.local:8080/predict"
	gpuStatServiceURL   = "http://latency-predictor-service.default.svc.cluster.local:8080/gpu-status"
	requestTimeout      = 1000 * time.Millisecond
	ttftWeight          = 0.4
	tpotWeight          = 0.6

	// Enable HTTP/2
	customTransport = &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 20,
		IdleConnTimeout:     90 * time.Second,
		DisableCompression:  false,
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Millisecond,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		TLSHandshakeTimeout: 30 * time.Millisecond,
		ForceAttemptHTTP2:   true, // Enable HTTP/2
	}
	gpuMonitoringClient = &http.Client{
		Timeout:   5 * time.Second, // Longer timeout for GPU stats
		Transport: customTransport,
	}
)

func init() {
	RegisterDelayedConstructor("latency-prediction-based", NewLatencyPredictionRouter)

	klog.InfoS("latency_prediction_based_configurations",
		"predictor_service_url", predictorServiceURL)
}

// PodFeatures represents the features for a single pod
type PodFeatures struct {
	RequestID string `json:"request_id"`

	// Core request features
	SelectedPod  string `json:"selected_pod"`
	InputTokens  int    `json:"input_tokens"`
	OutputTokens int    `json:"output_tokens"`
	TotalTokens  int    `json:"total_tokens"`

	// Pod metrics
	KVHitRatio int `json:"kv_hit_ratio"`

	InflightRequests int `json:"inflight_requests"`

	GpuKVCache float64 `json:"gpu_kv_cache"`
	CpuKVCache float64 `json:"cpu_kv_cache"`

	RunningRequests int `json:"running_requests"`
	WaitingRequests int `json:"waiting_requests"`

	PrefillTokens int `json:"prefill_tokens"`
	DecodeTokens  int `json:"decode_tokens"`

	GpuModel string `json:"gpu_model"`

	LastSecondAvgTTFTMs float64 `json:"last_second_avg_ttft_ms"`
	LastSecondP99TTFTMs int     `json:"last_second_p99_ttft_ms"`
	LastSecondAvgTPOTMs float64 `json:"last_second_avg_tpot_ms"`
	LastSecondP99TPOTMs int     `json:"last_second_p99_tpot_ms"`

	LastSecondTotalRequests      int `json:"last_second_total_requests"`
	LastSecondTotalTokens        int `json:"last_second_total_tokens"`
	LastSecondTotalDecodeTokens  int `json:"last_second_total_decode_tokens"`
	LastSecondTotalPrefillTokens int `json:"last_second_total_prefill_tokens"`
}

// func (pf *PodFeatures) CheckAllFieldsSet() bool {
// }

// PredictionRequest is the request body sent to the predictor service
type PredictionRequest struct {
	Pods []PodFeatures `json:"pods"`
}

// PredictionResponse is the response from the predictor service
type PredictionResponse struct {
	Predictions map[string]map[string]float64 `json:"predictions"`
}

// Cache entry for prediction results
type predictionCacheEntry struct {
	Predictions map[string]map[string]float64
	Timestamp   time.Time
}

// LatencyPredictionRouter provides routing based on predicted latency
type latencyPredictionRouter struct {
	cache              cache.Cache
	httpClient         *http.Client
	predictionCache    map[string]predictionCacheEntry
	ttftWeight         float64
	tpotWeight         float64
	prefixCacheIndexer *prefixcacheindexer.PrefixHashTable
	tokenizer          tokenizer.Tokenizer
}

// warmupConnection sends a minimal request to keep the connection alive
func warmupConnection(client *http.Client, url string) {
	req, err := http.NewRequest("HEAD", url, nil)
	if err != nil {
		klog.V(4).Infof("Failed to create warmup request: %v", err)
		return
	}

	resp, err := client.Do(req)
	if err != nil {
		klog.V(4).Infof("Warmup request failed: %v", err)
		return
	}
	defer resp.Body.Close()

	klog.V(4).Infof("Connection warmup successful, status: %d", resp.StatusCode)
}

// NewLatencyPredictionRouter creates a new latency prediction-based router
func NewLatencyPredictionRouter() (types.Router, error) {
	c, err := cache.Get()
	if err != nil {
		klog.Error("fail to get cache store in latency prediction router")
		return nil, err
	}

	var tokenizerObj tokenizer.Tokenizer
	// TODO: refactor initilization
	// supported tokenizers: ["character", "tiktoken"]
	if tokenizerType == "tiktoken" {
		tokenizerObj = tokenizer.NewTiktokenTokenizer()
	} else {
		tokenizerObj = tokenizer.NewCharacterTokenizer()
	}

	// Create HTTP client with timeout
	httpClient := &http.Client{
		Timeout:   requestTimeout,
		Transport: customTransport,
	}

	// Set up connection warmup to avoid cold connection penalties
	go func() {
		ticker := time.NewTicker(15 * time.Second)
		defer ticker.Stop()

		warmupURL := predictorServiceURL

		// Initial warmup
		warmupConnection(httpClient, warmupURL)

		// Regular warmup
		for range ticker.C {
			warmupConnection(httpClient, warmupURL)
		}
	}()

	klog.InfoS("Created latency prediction router",
		"requestTimeout", requestTimeout,
		"ttftWeight", ttftWeight,
		"tpotWeight", tpotWeight)

	router := &latencyPredictionRouter{
		cache:              c,
		httpClient:         httpClient,
		predictionCache:    make(map[string]predictionCacheEntry),
		ttftWeight:         ttftWeight,
		tpotWeight:         tpotWeight,
		tokenizer:          tokenizerObj,
		prefixCacheIndexer: prefixcacheindexer.NewPrefixHashTable(),
	}
	return router, nil
}

// TimingResult contains detailed timing information for HTTP requests
type TimingResult struct {
	RequestID      string
	DNSTime        time.Duration
	ConnectionTime time.Duration
	RequestTime    time.Duration
	SendTime       time.Duration
	ReadTime       time.Duration
	ParseTime      time.Duration
	TotalTime      time.Duration
	Success        bool
	Error          error
}

// String returns a formatted string representation of the timing result
func (tr TimingResult) String() string {
	return fmt.Sprintf("DNS: %dms, Conn: %dms, Req: %dms, Send: %dms, Read: %dms, Parse: %dms, Total: %dms",
		tr.DNSTime.Milliseconds(),
		tr.ConnectionTime.Milliseconds(),
		tr.RequestTime.Milliseconds(),
		tr.SendTime.Milliseconds(),
		tr.ReadTime.Milliseconds(),
		tr.ParseTime.Milliseconds(),
		tr.TotalTime.Milliseconds())
}

// SendRequestWithTiming performs an HTTP request with detailed timing measurements
func SendRequestWithTiming(
	client *http.Client,
	url string,
	method string,
	reqBody []byte,
	headers map[string]string,
	requestID string,
	timeout time.Duration,
) ([]byte, TimingResult, error) {

	result := TimingResult{
		RequestID: requestID,
		Success:   false,
	}

	startTime := time.Now()

	// 1. DNS Resolution
	dnsStart := time.Now()
	hostParts := strings.Split(url, "://")
	var hostName string
	if len(hostParts) > 1 {
		hostWithPath := strings.Split(hostParts[1], "/")[0]
		hostWithPort := strings.Split(hostWithPath, ":")
		hostName = hostWithPort[0]
	} else {
		hostName = strings.Split(strings.Split(url, "/")[0], ":")[0]
	}

	hosts, dnsErr := net.LookupHost(hostName)
	result.DNSTime = time.Since(dnsStart)

	if dnsErr != nil {
		klog.V(4).Infof("DNS lookup error for %s: %v", hostName, dnsErr)
	}

	// 2. Connection Test
	result.ConnectionTime = 0
	if len(hosts) > 0 {
		connStart := time.Now()
		// Extract port if included in URL
		port := "80"
		if strings.HasPrefix(url, "https://") {
			port = "443"
		}
		if strings.Contains(hostName, ":") {
			portParts := strings.Split(hostName, ":")
			if len(portParts) > 1 {
				port = portParts[1]
			}
		}

		conn, dialErr := net.DialTimeout("tcp", hosts[0]+":"+port, timeout/2)
		result.ConnectionTime = time.Since(connStart)

		if dialErr != nil {
			klog.V(4).Infof("Connection test error to %s:%s: %v", hosts[0], port, dialErr)
		} else if conn != nil {
			conn.Close()
		}
	}

	// 3. Request Creation
	reqStart := time.Now()
	req, reqErr := http.NewRequest(method, url, bytes.NewBuffer(reqBody))
	result.RequestTime = time.Since(reqStart)

	if reqErr != nil {
		result.Error = reqErr
		result.TotalTime = time.Since(startTime)
		return nil, result, reqErr
	}

	// Add headers
	for key, value := range headers {
		req.Header.Set(key, value)
	}

	// Add request ID header for tracking
	if requestID != "" {
		req.Header.Set("X-Request-ID", requestID)
	}

	// Set context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	req = req.WithContext(ctx)

	// 4. Send Request
	sendStart := time.Now()
	resp, sendErr := client.Do(req)
	result.SendTime = time.Since(sendStart)

	if sendErr != nil {
		result.Error = sendErr
		result.TotalTime = time.Since(startTime)
		return nil, result, sendErr
	}
	defer resp.Body.Close()

	// 5. Read Response
	readStart := time.Now()
	body, readErr := ioutil.ReadAll(resp.Body)
	result.ReadTime = time.Since(readStart)

	if readErr != nil {
		result.Error = readErr
		result.TotalTime = time.Since(startTime)
		return nil, result, readErr
	}

	// Check status code
	if resp.StatusCode != http.StatusOK {
		err := fmt.Errorf("non-200 status code: %d", resp.StatusCode)
		result.Error = err
		result.TotalTime = time.Since(startTime)
		return body, result, err
	}

	// Skip parsing timing since we don't know the response type
	result.ParseTime = 0
	result.Success = true
	result.TotalTime = time.Since(startTime)

	return body, result, nil
}

// SendJSONRequestWithParsing sends a JSON request and parses the response with timing
func SendJSONRequestWithParsing(
	client *http.Client,
	url string,
	reqBody []byte,
	responseObj interface{},
	requestID string,
	timeout time.Duration,
) (TimingResult, error) {

	headers := map[string]string{
		"Content-Type": "application/json",
		"Accept":       "application/json",
	}

	body, timingResult, err := SendRequestWithTiming(
		client,
		url,
		"POST",
		reqBody,
		headers,
		requestID,
		timeout,
	)

	if err != nil {
		return timingResult, err
	}

	// Parse JSON response
	parseStart := time.Now()
	err = json.Unmarshal(body, responseObj)
	timingResult.ParseTime = time.Since(parseStart)

	if err != nil {
		timingResult.Error = err
		timingResult.Success = false
	}

	return timingResult, err
}

// Route selects the optimal pod based on latency predictions
func (r *latencyPredictionRouter) Route(ctx *types.RoutingContext, pods types.PodList) (string, error) {
	// Get all ready pods
	readyPods := pods.All()
	if len(readyPods) == 0 {
		return "", fmt.Errorf("no ready pods available")
	}

	// If only one pod is available, just return it
	if len(readyPods) == 1 {
		ctx.SetTargetPod(readyPods[0])
		return ctx.TargetAddress(), nil
	}

	var prefixHashes []uint64
	var matchedPods map[string]int
	readyPodsMap := map[string]struct{}{}
	for _, pod := range readyPods {
		readyPodsMap[pod.Status.PodIP] = struct{}{}
	}

	ts := time.Now()
	tokens, err := r.tokenizer.TokenizeInputText(ctx.Message)
	klog.Infof("requestID: %s - prediction_based, tokenization took %dms", ctx.RequestID, time.Since(ts).Milliseconds())
	if err != nil {
		klog.Errorf("requestID: %s, Tokenization failed: %v", ctx.RequestID, err)
		return "", err
	}

	matchedPods, prefixHashes = r.prefixCacheIndexer.MatchPrefix(tokens, ctx.Model, readyPodsMap)
	utils.StoreKVCacheHitRatio(ctx.RequestID, matchedPods)
	inputTokens := utils.GetNumPrefillTokensForRequest(ctx.RequestID)
	outputTokens := 128 // NOTE: This is a placeholder. It should be replaced with output token count prediction logic.
	totalTokens := inputTokens + outputTokens

	// Build the request to the prediction service
	podFeatures := make([]PodFeatures, 0, len(readyPods))

	// Get detailed pod metrics
	metricsWindow := utils.MetricsTracker.WindowSize
	klog.Infof("metricsWindow: %s", metricsWindow)
	ts = time.Now()
	podMetricsMap := utils.MetricsTracker.GetDetailedMetrics(time.Now().Add(-metricsWindow))
	klog.Infof("requestID: %s - prediction_based, GetDetailedMetrics took %dms", ctx.RequestID, time.Since(ts).Milliseconds())

	ts = time.Now()
	// Create feature set for each pod
	for _, pod := range readyPods {
		podIP := pod.Status.PodIP

		// Create pod features
		features := PodFeatures{
			RequestID:    ctx.RequestID,
			SelectedPod:  podIP,
			InputTokens:  inputTokens,
			OutputTokens: outputTokens,
			TotalTokens:  totalTokens,
		}

		// Add KV cache hit ratio - use per-pod accessor
		if val, exists := utils.GetKVCacheHitRatioForPod(ctx.RequestID, podIP); exists {
			features.KVHitRatio = val
		} else {
			// klog.Errorf("ERROR! KVHitRatio is not available in podMetrics, requestID: %s, for pod %s", ctx.RequestID, podIP)
		}

		// Add inflight requests - use per-pod accessor
		if val, exists := utils.GetNumInflightRequestsForPod(podIP); exists {
			features.InflightRequests = val
		} else {
			// klog.Errorf("ERROR! InflightRequests is not available in podMetrics, requestID: %s, for pod %s", ctx.RequestID, podIP)
			features.InflightRequests = 0
		}

		// Add GPU KV cache - use per-pod accessor
		if val, exists := utils.GetVLLMGPUKVCacheUsageForPod(ctx.RequestID, podIP); exists && val >= 0 {
			features.GpuKVCache = val
		} else {
			klog.Errorf("ERROR! GpuKVCache is not available in podMetrics, requestID: %s, for pod %s", ctx.RequestID, podIP)
		}

		// Add CPU KV cache - use per-pod accessor
		if val, exists := utils.GetVLLMCPUKVCacheUsageForPod(ctx.RequestID, podIP); exists && val >= 0 {
			features.CpuKVCache = val
		} else {
			features.CpuKVCache = 0.0
		}

		// Add running requests - use per-pod accessor
		if val, exists := utils.GetVLLMNumRequestsRunningForPod(ctx.RequestID, podIP); exists && val >= 0 {
			features.RunningRequests = int(val)
		} else {
			klog.Errorf("ERROR! RunningRequests is not available in podMetrics, requestID: %s, for pod %s", ctx.RequestID, podIP)
		}

		// Add waiting requests - use per-pod accessor
		if val, exists := utils.GetVLLMNumRequestsWaitingForPod(ctx.RequestID, podIP); exists && val >= 0 {
			features.WaitingRequests = int(val)
		} else {
			klog.Errorf("ERROR! WaitingRequests is not available in podMetrics, requestID: %s, for pod %s", ctx.RequestID, podIP)
		}

		// Add prefill tokens - use per-pod accessor
		if val, exists := utils.GetNumPrefillTokensForPod(podIP); exists {
			features.PrefillTokens = val
		} else {
			klog.Errorf("ERROR! PrefillTokens is not available in podMetrics, requestID: %s, for pod %s", ctx.RequestID, podIP)
		}

		// Add decode tokens - use per-pod accessor
		if val, exists := utils.GetNumDecodeTokensForPod(podIP); exists {
			features.DecodeTokens = val
		} else {
			klog.Errorf("ERROR! DecodeTokens is not available in podMetrics, requestID: %s, for pod %s", ctx.RequestID, podIP)
		}

		// Add GPU model
		features.GpuModel = "NVIDIA-L20"

		// Add pod performance metrics if available
		if podMetrics, exists := podMetricsMap[podIP]; exists {
			if podMetrics.AvgTTFT > 0 {
				features.LastSecondAvgTTFTMs = podMetrics.AvgTTFT
			} else {
				klog.Errorf("ERROR! AvgTTFT is not available in podMetrics, requestID: %s, for pod %s", ctx.RequestID, podIP)
			}

			if podMetrics.AvgTPOT > 0 {
				features.LastSecondAvgTPOTMs = podMetrics.AvgTPOT
			} else {
				klog.Errorf("ERROR! AvgTPOT is not available in podMetrics, requestID: %s, for pod %s", ctx.RequestID, podIP)
			}

			if podMetrics.P99TTFT > 0 {
				p99TTFT := int(podMetrics.P99TTFT)
				features.LastSecondP99TTFTMs = p99TTFT
			} else {
				klog.Errorf("ERROR! P99TTFT is not available in podMetrics, requestID: %s, for pod %s", ctx.RequestID, podIP)
			}

			if podMetrics.P99TPOT > 0 {
				p99TPOT := int(podMetrics.P99TPOT)
				features.LastSecondP99TPOTMs = p99TPOT
			} else {
				klog.Errorf("ERROR! P99TPOT is not available in podMetrics, requestID: %s, for pod %s", ctx.RequestID, podIP)
			}

			if podMetrics.TotalRequests >= 0 {
				features.LastSecondTotalRequests = podMetrics.TotalRequests
			} else {
				klog.Errorf("ERROR! TotalRequests is not available in podMetrics, requestID: %s, for pod %s", ctx.RequestID, podIP)
			}

			if podMetrics.TotalTokens >= 0 {
				features.LastSecondTotalTokens = podMetrics.TotalTokens
			} else {
				klog.Errorf("ERROR! TotalTokens is not available in podMetrics, requestID: %s, for pod %s", ctx.RequestID, podIP)
			}

			if podMetrics.TotalDecodeTokens >= 0 {
				features.LastSecondTotalDecodeTokens = podMetrics.TotalDecodeTokens
			} else {
				klog.Errorf("ERROR! TotalDecodeTokens is not available in podMetrics, requestID: %s, for pod %s", ctx.RequestID, podIP)
			}

			if podMetrics.TotalPrefillTokens >= 0 {
				features.LastSecondTotalPrefillTokens = podMetrics.TotalPrefillTokens
			} else {
				klog.Errorf("ERROR! TotalPrefillTokens is not available in podMetrics, requestID: %s, for pod %s", ctx.RequestID, podIP)
			}
		}

		podFeatures = append(podFeatures, features)

		featureJSON, err := json.MarshalIndent(features, "\t", "  ")
		if err != nil {
			klog.Errorf("Failed to marshal pod features: %v", err)
		} else {
			klog.Infof("requestID: %s, Pod features for: %s\n%s", ctx.RequestID, podIP, string(featureJSON))
		}
	}
	klog.Infof("requestID: %s - prediction_based, reading features took %dms", ctx.RequestID, time.Since(ts).Milliseconds())

	// Create the request
	predRequest := PredictionRequest{
		Pods: podFeatures,
	}
	var targetPod *v1.Pod

	// Convert to JSON
	ts = time.Now()
	reqBody, err := json.Marshal(predRequest)
	klog.Infof("requestID: %s - prediction_based, marshaling took %dms", ctx.RequestID, time.Since(ts).Milliseconds())
	if err != nil {
		klog.Errorf("Failed to marshal prediction request: %v", err)
		targetPod, _ = r.fallbackRouting(ctx, readyPods)
	}

	// Send request to prediction service
	// klog.Infof("Sending prediction request to %s", predictorServiceURL)
	req, err := http.NewRequest("POST", predictorServiceURL, bytes.NewBuffer(reqBody))
	if err != nil {
		klog.Errorf("Failed to create prediction request: %v", err)
		targetPod, _ = r.fallbackRouting(ctx, readyPods)
	}

	req.Header.Set("Content-Type", "application/json")

	// Set a short context timeout to avoid slowing down routing
	ts = time.Now()
	httpCtx, cancel := context.WithTimeout(context.Background(), r.httpClient.Timeout)
	klog.Infof("requestID: %s - prediction_based, context timeout took %dms", ctx.RequestID, time.Since(ts).Milliseconds())
	defer cancel()
	req = req.WithContext(httpCtx)

	var predResponse PredictionResponse
	timingResult, err := SendJSONRequestWithParsing(
		r.httpClient,
		predictorServiceURL,
		reqBody,
		&predResponse,
		ctx.RequestID,
		r.httpClient.Timeout,
	)

	klog.Infof("requestID: %s - prediction_based, prediction took %dms", ctx.RequestID, timingResult.TotalTime.Milliseconds())
	klog.Infof("requestID: %s -  Timing breakdown took, %s", ctx.RequestID, timingResult.String())

	if err != nil {
		klog.Errorf("Failed to send prediction request: %v", err)
		targetPod, _ = r.fallbackRouting(ctx, readyPods)
		ctx.SetTargetPod(targetPod)
		return ctx.TargetAddress(), nil
	}

	if targetPod == nil {
		targetPod = r.selectBestPod(ctx.RequestID, readyPods, predResponse.Predictions)
		if targetPod != nil {
			klog.Infof("success, requestID: %s, Selected pod %s based on latency predictions", ctx.RequestID, targetPod.Status.PodIP)
		} else {
			klog.Warningf("Failed to select pod based on predictions, falling back to default routing")
			targetPod, _ = r.fallbackRouting(ctx, readyPods)
		}
	}

	if len(prefixHashes) > 0 {
		klog.Infof("Adding prefix hashes to cache. pod: %s", targetPod.Status.PodIP)
		r.prefixCacheIndexer.AddPrefix(prefixHashes, ctx.Model, targetPod.Status.PodIP)
	}

	ctx.SetTargetPod(targetPod)
	return ctx.TargetAddress(), nil
}

func (r *latencyPredictionRouter) selectBestPod(requestID string, readyPods []*v1.Pod, predictions map[string]map[string]float64) *v1.Pod {
	if len(predictions) == 0 {
		return nil
	}

	bestPod := readyPods[0]
	bestScore := float64(^uint(0) >> 1) // Max int as initial value

	for _, pod := range readyPods {
		podIP := pod.Status.PodIP

		// Skip if we don't have predictions for this pod
		podPreds, exists := predictions[podIP]
		if !exists {
			continue
		}

		// Get TTFT and TPOT predictions
		predicted_ttft, ttftExists := podPreds["ttft"]
		predicted_tpot, tpotExists := podPreds["avg_tpot"]

		// Skip if predictions are missing or invalid
		if !ttftExists || !tpotExists || predicted_ttft < 0 || predicted_tpot < 0 {
			continue
		}

		// Calculate weighted score
		score := r.ttftWeight*predicted_ttft + r.tpotWeight*predicted_tpot

		// For debugging
		klog.Infof("success, requestID: %s, Pod %s, predicted TTFT: %.2f, predicted TPOT: %.2f, score: %.2f", requestID, podIP, predicted_ttft, predicted_tpot, score)

		// Update best pod if this one has a better score
		if score < bestScore {
			bestScore = score
			bestPod = pod
		}
	}

	// If we couldn't find a valid pod (all had invalid predictions), return nil
	if bestScore == float64(^uint(0)>>1) {
		klog.Errorf("No valid predictions found for any pod")
		return nil
	}

	return bestPod
}

func (r *latencyPredictionRouter) fallbackRouting(ctx *types.RoutingContext, readyPods []*v1.Pod) (*v1.Pod, error) {
	klog.Infof("Using fallback routing for request %s", ctx.RequestID)
	var err error
	targetPod, err := selectRandomPod(readyPods, rand.Intn)
	if err != nil {
		klog.Errorf("Failed to select random pod: %v", err)
		return nil, err
	}
	if targetPod == nil {
		klog.Errorf("No suitable pod found for fallback routing")
		return nil, fmt.Errorf("no suitable pod found for fallback routing")
	}
	return targetPod, nil
}
