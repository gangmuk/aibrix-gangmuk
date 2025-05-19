package routingalgorithms

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"net"
	"net/http"
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
	requestTimeout      = 1000 * time.Millisecond
	ttftWeight          = 0.4
	tpotWeight          = 0.6

	// Enable HTTP/2
	customTransport = &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 100,
		IdleConnTimeout:     180 * time.Second,
		DisableCompression:  false,
		DialContext: (&net.Dialer{
			Timeout:   200 * time.Millisecond,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		TLSHandshakeTimeout:   200 * time.Millisecond,
		ForceAttemptHTTP2:     true, // Enable HTTP/2
		ResponseHeaderTimeout: 1 * time.Second,
	}
)

func init() {
	RegisterDelayedConstructor("latency-prediction-based", NewLatencyPredictionRouter)

	klog.InfoS("latency_prediction_based_configurations",
		"predictor_service_url", predictorServiceURL)
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
	podFeatures := make([]utils.PodFeatures, 0, len(readyPods))

	// Get detailed pod metrics
	ts = time.Now()
	podMetricsMap := utils.MetricsTracker.GetDetailedMetrics(time.Now().Add(-utils.MetricsTracker.WindowSize))
	klog.Infof("requestID: %s - prediction_based, GetDetailedMetrics took %dms", ctx.RequestID, time.Since(ts).Milliseconds())

	ts = time.Now()
	// Create feature set for each pod
	for _, pod := range readyPods {
		podIP := pod.Status.PodIP

		// Create pod features
		features := utils.PodFeatures{
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
	predRequest := utils.PredictionRequest{
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

	var predResponse utils.PredictionResponse
	timingResult, err := utils.SendJSONRequestWithParsing(
		r.httpClient,
		predictorServiceURL,
		reqBody,
		&predResponse,
		ctx.RequestID,
		r.httpClient.Timeout,
	)

	klog.Infof("requestID: %s -  Timing breakdown took, %s", ctx.RequestID, timingResult.String())

	if err != nil {
		klog.Errorf("Failed to send prediction request: %v", err)
		targetPod, _ = r.fallbackRouting(ctx, readyPods)
		ctx.SetTargetPod(targetPod)
		return ctx.TargetAddress(), nil
	}

	targetPod = r.selectBestPod(ctx.RequestID, readyPods, predResponse.Predictions)
	klog.Infof("success, requestID: %s, Selected pod %s based on latency predictions", ctx.RequestID, targetPod.Status.PodIP)

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
