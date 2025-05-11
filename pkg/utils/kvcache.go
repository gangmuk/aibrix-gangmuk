// In pkg/utils/kvcache.go
package utils

import (
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/vllm-project/aibrix/pkg/metrics"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

// PodDetailedMetrics provides detailed statistics for a pod's performance
type PodDetailedMetrics struct {
	// TTFT metrics
	AvgTTFT     float64 `json:"last_second_avg_ttft_ms"`
	MinTTFT     int64   `json:"last_second_min_ttft_ms"`
	MaxTTFT     int64   `json:"last_second_max_ttft_ms"`
	P50TTFT     int64   `json:"last_second_p50_ttft_ms"` // Median TTFT
	P90TTFT     int64   `json:"last_second_p90_ttft_ms"` // 90th percentile TTFT
	P95TTFT     int64   `json:"last_second_p95_ttft_ms"` // 95th percentile TTFT
	P99TTFT     int64   `json:"last_second_p99_ttft_ms"` // 99th percentile TTFT
	TTFTSamples int     `json:"last_second_ttft_samples"`

	// TPOT metrics
	AvgTPOT     float64 `json:"last_second_avg_tpot_ms"`
	MinTPOT     int64   `json:"last_second_min_tpot_ms"`
	MaxTPOT     int64   `json:"last_second_max_tpot_ms"`
	P50TPOT     int64   `json:"last_second_p50_tpot_ms"` // Median TPOT
	P90TPOT     int64   `json:"last_second_p90_tpot_ms"` // 90th percentile TPOT
	P95TPOT     int64   `json:"last_second_p95_tpot_ms"` // 95th percentile TPOT
	P99TPOT     int64   `json:"last_second_p99_tpot_ms"` // 99th percentile TPOT
	TPOTSamples int     `json:"last_second_tpot_samples"`

	// // Token position-based TPOT metrics (average TPOT for tokens 2-10)
	EarlyTokensTPOT float64 `json:"last_second_early_tokens_tpot_ms"`
	MidTokensTPOT   float64 `json:"last_second_mid_tokens_tpot_ms"`
	LateTokensTPOT  float64 `json:"last_second_late_tokens_tpot_ms"`

	// Overall metrics
	TotalRequests      int `json:"last_second_total_requests"`
	TotalDecodeTokens  int `json:"last_second_total_decode_tokens"`
	TotalPrefillTokens int `json:"last_second_total_prefill_tokens"`
	TotalTokens        int `json:"last_second_total_tokens"`
}

// **NOTE**: the name PodMetric is very confusing. one PodMetric instance is one response token (first token will create one PodMetric and also one decode response(token) will create one PodMetric instance).
type PodMetric struct {
	RequestID       string
	Timestamp       time.Time
	TTFT            int64 // Time to first token
	TPOT            int64 // Time per output token (for single token)
	PrefillTokenNum int64 // number of tokens in the decode phase
	DecodeTokenNum  int64 // number of decoded tokens generated up to Timestamp
}

func (t *PodMetricsTracker) InitPodKey(podIP string) {
	// Trim port from podIP if present
	if colonIndex := len(podIP) - 1; podIP[colonIndex] == ':' {
		podIP = podIP[:colonIndex]
	}
	t.mutex.Lock()
	defer t.mutex.Unlock()
	if _, exists := t.podMetrics[podIP]; !exists {
		t.podMetrics[podIP] = []PodMetric{CreatePodMetric()}
		klog.Infof("Initialized pod metrics for pod %s", podIP)
	}
}

func (t *PodMetricsTracker) AddPodMetric(podIP string, metric PodMetric) {
	// Trim port from podIP if present
	if colonIndex := len(podIP) - 1; podIP[colonIndex] == ':' {
		podIP = podIP[:colonIndex]
	}
	t.mutex.Lock()
	defer t.mutex.Unlock()

	// Add the new metric
	t.podMetrics[podIP] = append(t.podMetrics[podIP], metric)

	// Clean up old metrics outside our window
	t.cleanupOldMetrics(podIP, time.Now())
}

// GetAverages calculates the average TTFT and TPOT for all pods over the window
func (t *PodMetricsTracker) GetAverages() map[string]map[string]float64 {
	t.mutex.RLock()
	defer t.mutex.RUnlock()

	now := time.Now()
	cutoff := now.Add(-t.WindowSize)
	result := make(map[string]map[string]float64)

	for podIP, metrics := range t.podMetrics {
		var ttftSum, tpotSum int64
		var ttftCount, tpotCount int

		// Calculate sums of valid metrics
		for _, m := range metrics {
			if m.Timestamp.After(cutoff) {
				if m.TTFT > 0 {
					ttftSum += m.TTFT
					ttftCount++
				}
				if m.TPOT > 0 {
					tpotSum += m.TPOT
					tpotCount++
				}
			}
		}

		// Calculate averages
		podAvg := make(map[string]float64)
		if ttftCount > 0 {
			podAvg["avg_ttft"] = float64(ttftSum) / float64(ttftCount)
		}
		if tpotCount > 0 {
			podAvg["avg_tpot"] = float64(tpotSum) / float64(tpotCount)
		}
		podAvg["sample_count"] = float64(len(metrics))

		result[podIP] = podAvg
	}

	return result
}

// NewPodMetricsTracker creates a new metrics tracker with the specified window size
func NewPodMetricsTracker(windowSize time.Duration) *PodMetricsTracker {
	return &PodMetricsTracker{
		podMetrics: make(map[string][]PodMetric),
		WindowSize: windowSize,
	}
}

func (t *PodMetricsTracker) CleanupAllMetrics() {
	t.mutex.Lock()
	defer t.mutex.Unlock()

	now := time.Now()
	cutoff := now.Add(-t.WindowSize)

	for podIP, metrics := range t.podMetrics {
		var newMetrics []PodMetric
		for _, m := range metrics {
			if m.Timestamp.After(cutoff) {
				newMetrics = append(newMetrics, m)
			}
		}

		// Always keep the pod entry, even if it has no valid metrics
		t.podMetrics[podIP] = newMetrics

		// Optionally, if you want to maintain at least one entry
		// to mark that the pod exists, you could add a placeholder:
		if len(newMetrics) == 0 {
			// Add a placeholder metric with zero values
			t.podMetrics[podIP] = []PodMetric{CreatePodMetric()}
		}
	}
}

func CreatePodMetric() PodMetric {
	return PodMetric{
		RequestID:       "",
		Timestamp:       time.Now(),
		TTFT:            0,
		TPOT:            0,
		PrefillTokenNum: 0,
		DecodeTokenNum:  0,
	}
}

type PodMetricsTracker struct {
	mutex      sync.RWMutex
	podMetrics map[string][]PodMetric // Map of pod IP -> slice of metrics
	WindowSize time.Duration          // How long to keep metrics
}

func (t *PodMetricsTracker) GetMutex() *sync.RWMutex {
	return &t.mutex
}

// cleanupOldMetrics removes metrics that are older than the window size
func (t *PodMetricsTracker) cleanupOldMetrics(podIP string, now time.Time) {
	cutoff := now.Add(-t.WindowSize)
	metrics := t.podMetrics[podIP]

	var newMetrics []PodMetric
	for _, m := range metrics {
		if m.Timestamp.After(cutoff) {
			newMetrics = append(newMetrics, m)
		}
	}

	t.podMetrics[podIP] = newMetrics
}

func percentile(sortedValues []int64, p int) int64 {
	if len(sortedValues) == 0 {
		return 0
	}

	if len(sortedValues) == 1 {
		return sortedValues[0]
	}

	// Calculate the rank
	rank := float64(p) / 100.0 * float64(len(sortedValues)-1)
	rankInt := int(rank)

	// If rank is an integer, return that value
	if rank == float64(rankInt) {
		return sortedValues[rankInt]
	}

	// Otherwise, interpolate between two values
	fraction := rank - float64(rankInt)
	return int64(float64(sortedValues[rankInt]) + fraction*(float64(sortedValues[rankInt+1])-float64(sortedValues[rankInt])))
}

func (t *PodMetricsTracker) GetDetailedMetrics(log_window_start_time time.Time) map[string]PodDetailedMetrics {
	t.mutex.RLock()
	defer t.mutex.RUnlock()
	result := make(map[string]PodDetailedMetrics)
	for podIP, metrics := range t.podMetrics {
		// podMetrics should have all pods in its entry
		// Init here to record all pods even if one does not have metrics after log_window_start_time
		detailedMetrics := PodDetailedMetrics{
			TotalRequests:      -1,
			TotalDecodeTokens:  -1,
			TotalPrefillTokens: -1,
			TotalTokens:        -1,
			TTFTSamples:        -1,
			TPOTSamples:        -1,
			AvgTTFT:            -1,
			MinTTFT:            -1,
			MaxTTFT:            -1,
			P50TTFT:            -1,
			P90TTFT:            -1,
			P95TTFT:            -1,
			P99TTFT:            -1,
			AvgTPOT:            -1,
			MinTPOT:            -1,
			MaxTPOT:            -1,
			P50TPOT:            -1,
			P90TPOT:            -1,
			P95TPOT:            -1,
			P99TPOT:            -1,
		}

		var validMetrics []PodMetric
		for _, m := range metrics {
			if m.Timestamp.After(log_window_start_time) {
				validMetrics = append(validMetrics, m)
			}
		}
		var ttftValues []int64
		var tpotValues []int64
		var ttftSum, tpotSum int64
		// var earlyTokensTPOT, midTokensTPOT, lateTokensTPOT []int64
		uniqueRequests := make(map[string]bool)
		totalDecodeTokens := 0
		totalPrefillTokens := 0
		for _, m := range validMetrics {
			if m.TTFT > 0 {
				ttftValues = append(ttftValues, m.TTFT)
				ttftSum += m.TTFT
				uniqueKey := fmt.Sprintf("%s-%d", podIP, m.Timestamp.UnixNano())
				uniqueRequests[uniqueKey] = true
				totalPrefillTokens += int(m.PrefillTokenNum)
			}
			// if m.TPOT > 0 {
			// 	tpotValues = append(tpotValues, m.TPOT)
			// 	tpotSum += m.TPOT
			// 	totalDecodeTokens++
			// 	early_token_index := numOutputTokens / 3
			// 	mid_token_index := (numOutputTokens / 3) * 2
			// 	switch {
			// 	case m.DecodeTokenNum <= early_token_index:
			// 		earlyTokensTPOT = append(earlyTokensTPOT, m.TPOT)
			// 	case m.DecodeTokenNum > early_token_index && m.DecodeTokenNum <= mid_token_index:
			// 		midTokensTPOT = append(midTokensTPOT, m.TPOT)
			// 	case m.DecodeTokenNum > mid_token_index:
			// 		lateTokensTPOT = append(lateTokensTPOT, m.TPOT)
			// 	}
			// }
		}
		sort.Slice(ttftValues, func(i, j int) bool { return ttftValues[i] < ttftValues[j] })
		sort.Slice(tpotValues, func(i, j int) bool { return tpotValues[i] < tpotValues[j] })

		detailedMetrics.TotalRequests = len(uniqueRequests)
		detailedMetrics.TotalDecodeTokens = totalDecodeTokens
		detailedMetrics.TotalPrefillTokens = totalPrefillTokens
		detailedMetrics.TotalTokens = totalDecodeTokens + totalPrefillTokens
		detailedMetrics.TTFTSamples = len(ttftValues)
		detailedMetrics.TPOTSamples = len(tpotValues)
		if len(ttftValues) > 0 {
			detailedMetrics.AvgTTFT = float64(ttftSum) / float64(len(ttftValues))
			detailedMetrics.MinTTFT = ttftValues[0]
			detailedMetrics.MaxTTFT = ttftValues[len(ttftValues)-1]
			detailedMetrics.P50TTFT = percentile(ttftValues, 50)
			detailedMetrics.P90TTFT = percentile(ttftValues, 90)
			detailedMetrics.P95TTFT = percentile(ttftValues, 95)
			detailedMetrics.P99TTFT = percentile(ttftValues, 99)
		}
		if len(tpotValues) > 0 {
			detailedMetrics.AvgTPOT = float64(tpotSum) / float64(len(tpotValues))
			detailedMetrics.MinTPOT = tpotValues[0]
			detailedMetrics.MaxTPOT = tpotValues[len(tpotValues)-1]
			detailedMetrics.P50TPOT = percentile(tpotValues, 50)
			detailedMetrics.P90TPOT = percentile(tpotValues, 90)
			detailedMetrics.P95TPOT = percentile(tpotValues, 95)
			detailedMetrics.P99TPOT = percentile(tpotValues, 99)
		}

		// if len(earlyTokensTPOT) > 0 {
		// 	var sum int64
		// 	for _, v := range earlyTokensTPOT {
		// 		sum += v
		// 	}
		// 	detailedMetrics.EarlyTokensTPOT = float64(sum) / float64(len(earlyTokensTPOT))
		// }
		// if len(midTokensTPOT) > 0 {
		// 	var sum int64
		// 	for _, v := range midTokensTPOT {
		// 		sum += v
		// 	}
		// 	detailedMetrics.MidTokensTPOT = float64(sum) / float64(len(midTokensTPOT))
		// }
		// if len(lateTokensTPOT) > 0 {
		// 	var sum int64
		// 	for _, v := range lateTokensTPOT {
		// 		sum += v
		// 	}
		// 	detailedMetrics.LateTokensTPOT = float64(sum) / float64(len(lateTokensTPOT))
		// }

		result[podIP] = detailedMetrics
	}
	return result
}

var (
	podMetricsMutex     sync.RWMutex
	requestToPodMetrics = make(map[string]map[string]PodDetailedMetrics)

	// Global map to store KV cache hit ratios per request
	podInflightRequests map[string]int // podIP -> num inflight requests
	PodInflightMutex    sync.RWMutex

	podTotalPrefillTokensMutex sync.RWMutex
	podTotalPrefillTokens      map[string]int // podIP -> total number of active prefill tokens

	podTotalPrefillRequestsMutex sync.RWMutex
	podTotalPrefillRequests      map[string]int // podIP -> total number of active prefill requests

	podTotalDecodeTokensMutex sync.RWMutex
	podTotalDecodeTokens      map[string]int // podIP -> total number of active decode tokens

	podTotalDecodeRequestsMutex sync.RWMutex
	podTotalDecodeRequests      map[string]int // podIP -> total number of active decode requests

	podToTotalTokensMutex sync.RWMutex
	podToTotalTokens      map[string]int // podIP -> total number of active tokens

	requestAllPodsKVCacheMutex sync.RWMutex
	requestAllPodsKVCache      map[string]map[string]int // requestID -> (podIP -> hit ratio)

	requestInflightMutex sync.RWMutex
	requestInflight      map[string]map[string]int // requestID -> (podIP -> num inflight requests

	requestToNumPrefillTokensMutex sync.RWMutex
	requestToNumPrefillTokens      map[string]int // requestID -> num prefill tokens

	requestToPrefillTokensMutex sync.RWMutex
	requestToPrefillTokens      map[string][]int // requestID -> num prefill tokens

	requestToNumDecodeTokensMutex sync.RWMutex
	requestToNumDecodeTokens      map[string]int // requestID -> num decode tokens

	requestToPodIPMutex sync.RWMutex
	requestToPodIP      map[string]string // requestID -> podIP

	vllmGPUKVCacheUsageMutex sync.RWMutex
	vllmGPUKVCacheUsage      map[string]map[string]float64 // requestID -> (podIP -> gpu kv cache usage)

	vllmCPUKVCacheUsageMutex sync.RWMutex
	vllmCPUKVCacheUsage      map[string]map[string]float64 // requestID -> (podIP -> cpu kv cache usage)

	vllmNumRequestsRunningMutex sync.RWMutex
	vllmNumRequestsRunning      map[string]map[string]float64 // requestID -> (podIP -> num requests running)

	vllmNumRequestsWaitingMutex sync.RWMutex
	vllmNumRequestsWaiting      map[string]map[string]float64 // requestID -> (podIP -> num requests waiting

)

const (
	MetricGPUCacheUsagePerc  = "gpu_cache_usage_perc"
	MetricCPUCacheUsagePerc  = "cpu_cache_usage_perc"
	MetricNumRequestsRunning = "num_requests_running"
	MetricNumRequestsWaiting = "num_requests_waiting"
	PodPort                  = 8000 // Same as in the metrics code
)

func init() {
	podMetricsMutex = sync.RWMutex{}
	requestToPodMetrics = make(map[string]map[string]PodDetailedMetrics)

	PodInflightMutex = sync.RWMutex{}
	podInflightRequests = make(map[string]int)

	podTotalPrefillTokens = make(map[string]int)
	podTotalPrefillTokensMutex = sync.RWMutex{}

	podTotalPrefillRequestsMutex = sync.RWMutex{}
	podTotalPrefillRequests = make(map[string]int)

	podTotalDecodeTokensMutex = sync.RWMutex{}
	podTotalDecodeTokens = make(map[string]int)

	podTotalDecodeRequestsMutex = sync.RWMutex{}
	podTotalDecodeRequests = make(map[string]int)

	podToTotalTokensMutex = sync.RWMutex{}
	podToTotalTokens = make(map[string]int)

	requestAllPodsKVCacheMutex = sync.RWMutex{}
	requestAllPodsKVCache = make(map[string]map[string]int)

	requestInflightMutex = sync.RWMutex{}
	requestInflight = make(map[string]map[string]int)

	requestToNumPrefillTokensMutex = sync.RWMutex{}
	requestToNumPrefillTokens = make(map[string]int)

	requestToPrefillTokensMutex = sync.RWMutex{}
	requestToPrefillTokens = make(map[string][]int)

	requestToNumDecodeTokensMutex = sync.RWMutex{}
	requestToNumDecodeTokens = make(map[string]int)

	requestToPodIPMutex = sync.RWMutex{}
	requestToPodIP = make(map[string]string)

	vllmGPUKVCacheUsage = make(map[string]map[string]float64)
	vllmGPUKVCacheUsageMutex = sync.RWMutex{}

	vllmCPUKVCacheUsage = make(map[string]map[string]float64)
	vllmCPUKVCacheUsageMutex = sync.RWMutex{}

	vllmNumRequestsRunning = make(map[string]map[string]float64)
	vllmNumRequestsRunningMutex = sync.RWMutex{}

	vllmNumRequestsWaiting = make(map[string]map[string]float64)
	vllmNumRequestsWaitingMutex = sync.RWMutex{}
}

func AddRequestPodMetrics(requestID string, detailedpodmetrics map[string]PodDetailedMetrics) {
	podMetricsMutex.Lock()
	defer podMetricsMutex.Unlock()

	if _, exists := requestToPodMetrics[requestID]; !exists {
		requestToPodMetrics[requestID] = make(map[string]PodDetailedMetrics)
	}
	for podIP, metrics := range detailedpodmetrics {
		if _, exists := requestToPodMetrics[requestID][podIP]; !exists {
			// requestToPodMetrics[requestID][podIP] = PodDetailedMetrics{}
			requestToPodMetrics[requestID][podIP] = metrics
		}
	}
}

func GetAndCleanupRequestPodMetrics(requestID string) map[string]PodDetailedMetrics {
	podMetricsMutex.Lock()
	defer podMetricsMutex.Unlock()

	metrics, exists := requestToPodMetrics[requestID]
	if !exists {
		klog.ErrorS(nil, "Failed to find metrics for request ID", "requestID", requestID)
		return make(map[string]PodDetailedMetrics)
	}

	delete(requestToPodMetrics, requestID)
	return metrics // Return directly, no need for copying
}

func GetrequestToPrefillTokensMutex() *sync.RWMutex {
	return &requestToPrefillTokensMutex
}

func GetPrefillTokensForRequest(requestID string) []int {
	requestToPrefillTokensMutex.RLock()
	defer requestToPrefillTokensMutex.RUnlock()
	if _, ok := requestToPrefillTokens[requestID]; !ok {
		return nil
	}
	return requestToPrefillTokens[requestID]
}

func SetPrefillTokensForRequest(requestID string, prefillTokens []int) {
	requestToPrefillTokensMutex.Lock()
	defer requestToPrefillTokensMutex.Unlock()
	if _, ok := requestToPrefillTokens[requestID]; !ok {
		requestToPrefillTokens[requestID] = make([]int, 0)
	}
	requestToPrefillTokens[requestID] = prefillTokens
}

func GetrequestAllPodsKVCacheMutex() *sync.RWMutex {
	return &requestAllPodsKVCacheMutex
}

func CleanuprequestToPodIP(requestID string) {
	requestToPodIPMutex.Lock()
	defer requestToPodIPMutex.Unlock()
	delete(requestToPodIP, requestID)
}

func StoreKVCacheHitRatio(requestID string, allPodsRatios map[string]int) {
	requestAllPodsKVCacheMutex.Lock()
	defer requestAllPodsKVCacheMutex.Unlock()
	requestAllPodsKVCache[requestID] = allPodsRatios
	klog.Infof("Stored KV cache hit ratios for request %s: %v", requestID, allPodsRatios)
}

func GetAllPodsKVCacheHitRatios(requestID string) map[string]int {
	requestAllPodsKVCacheMutex.RLock()
	defer requestAllPodsKVCacheMutex.RUnlock()
	if ratios, ok := requestAllPodsKVCache[requestID]; ok {
		return ratios
	}
	return make(map[string]int)
}

func CleanupKVCacheHitRatio(requestID string) {
	requestAllPodsKVCacheMutex.Lock()
	defer requestAllPodsKVCacheMutex.Unlock()
	delete(requestAllPodsKVCache, requestID)
}

func StoreRequestToPodIP(requestID string, podIP string) {
	requestToPodIPMutex.Lock()
	defer requestToPodIPMutex.Unlock()
	if _, exists := requestToPodIP[requestID]; exists {
		klog.Errorf("requestID already exists in requestToPodIP: %s", requestID)
	}
	requestToPodIP[requestID] = podIP
}

func GetPodIPForRequest(requestID string) (string, bool) {
	requestToPodIPMutex.RLock()
	defer requestToPodIPMutex.RUnlock()
	podIP, exists := requestToPodIP[requestID]
	if !exists {
		klog.Errorf("requestID not found in requestToPodIP: %s", requestID)
		return "", false
	}
	return podIP, exists
}

func SetNumPrefillTokensForRequest(requestID string, numTokens int) {
	requestToNumPrefillTokensMutex.Lock()
	defer requestToNumPrefillTokensMutex.Unlock()
	if _, ok := requestToNumPrefillTokens[requestID]; !ok {
		requestToNumPrefillTokens[requestID] = 0
	}
	requestToNumPrefillTokens[requestID] += numTokens
	klog.V(5).Infof("TokenCount, Increment prefill tokens for request %s: by %d, %d", requestID, numTokens, requestToNumPrefillTokens[requestID])
}

func GetNumPrefillTokensForRequest(requestID string) int {
	requestToNumPrefillTokensMutex.RLock()
	defer requestToNumPrefillTokensMutex.RUnlock()
	if _, ok := requestToNumPrefillTokens[requestID]; !ok {
		return 0
	}
	return requestToNumPrefillTokens[requestID]
}

// Increment the number of inflight requests for a specific pod
func IncrementNumInflightForPod(requestID string) {
	PodInflightMutex.Lock()
	defer PodInflightMutex.Unlock()
	podIP, exists := GetPodIPForRequest(requestID)
	if !exists {
		klog.Errorf("Pod name not found for request ID: %s", requestID)
		return
	}
	if _, ok := podInflightRequests[podIP]; !ok {
		podInflightRequests[podIP] = 0
	}

	podInflightRequests[podIP]++
	klog.V(5).Infof("Incremented inflight requests for pod %s: %d", podIP, podInflightRequests[podIP])
}

// Decrement the number of inflight requests for a specific pod
func DecrementNumInflightForPod(requestID string) {
	PodInflightMutex.Lock()
	defer PodInflightMutex.Unlock()
	podIP, exists := GetPodIPForRequest(requestID)
	if !exists {
		klog.Errorf("Pod name not found for request ID: %s", requestID)
		return
	}
	if _, ok := podInflightRequests[podIP]; !ok {
		klog.Errorf("Pod name not found in podInflightRequests: %s", podIP)
		return
	}

	podInflightRequests[podIP]--
	if podInflightRequests[podIP] < 0 {
		klog.Errorf("podInflightRequests[%s]: %d is negative!", podIP, podInflightRequests[podIP])
	}
	klog.V(5).Infof("Decremented inflight requests for pod %s: %d", podIP, podInflightRequests[podIP])
}

func GetNumInflightRequestsForPod(podIP string) int {
	PodInflightMutex.RLock()
	defer PodInflightMutex.RUnlock()
	if _, ok := podInflightRequests[podIP]; !ok {
		return 0
	}
	return podInflightRequests[podIP]
}

func StoreInflightRequestsForTheRequest(requestID string) {
	requestInflightMutex.Lock()
	defer requestInflightMutex.Unlock()
	if _, exists := requestInflight[requestID]; exists {
		klog.Errorf("requestID already exists in requestInflight: %s", requestID)
		return
	}
	requestInflight[requestID] = make(map[string]int)
	PodInflightMutex.RLock()
	defer PodInflightMutex.RUnlock()
	for podIP, numinflightrequests := range podInflightRequests {
		requestInflight[requestID][podIP] = numinflightrequests
	}
}

func GetrequestInflightMutex() *sync.RWMutex {
	return &requestInflightMutex
}

func GetInflightRequestsForAllPods(requestID string) map[string]int {
	requestInflightMutex.RLock()
	defer requestInflightMutex.RUnlock()
	if inflightRequests, ok := requestInflight[requestID]; !ok {
		return make(map[string]int)
	} else {
		// Convert pod names to pod IPs before returning
		return inflightRequests
	}
}

func CleanupInflightRequests(requestID string) {
	requestInflightMutex.Lock()
	defer requestInflightMutex.Unlock()
	delete(requestInflight, requestID)
}

func IncrementNumPrefillTokensForPod(podIP string, numTokens int) int {
	podTotalPrefillTokensMutex.Lock()
	defer podTotalPrefillTokensMutex.Unlock()
	if _, ok := podTotalPrefillTokens[podIP]; !ok {
		podTotalPrefillTokens[podIP] = 0
	}
	old_numTokens := podTotalPrefillTokens[podIP]
	podTotalPrefillTokens[podIP] += numTokens
	klog.V(5).Infof("TokenCount, Incremented prefill tokens for pod %s by %d. from %d to %d", podIP, numTokens, old_numTokens, podTotalPrefillTokens[podIP])
	return podTotalPrefillTokens[podIP]
}

func DecrementNumPrefillTokensForPod(podIP string, numTokens int) int {
	podTotalPrefillTokensMutex.Lock()
	defer podTotalPrefillTokensMutex.Unlock()
	if _, ok := podTotalPrefillTokens[podIP]; !ok {
		klog.Errorf("Pod name not found in podTotalPrefillTokens: %s", podIP)
		return -1
	}
	if podTotalPrefillTokens[podIP] < numTokens {
		klog.Errorf("podTotalPrefillTokens[%s]: %d is less than numTokens: %d", podIP, podTotalPrefillTokens[podIP], numTokens)
		return -1
	}
	old_numTokens := podTotalPrefillTokens[podIP]
	podTotalPrefillTokens[podIP] -= numTokens
	klog.V(5).Infof("TokenCount, Decremented prefill tokens for pod %s by %d. from %d to %d", podIP, numTokens, old_numTokens, podTotalPrefillTokens[podIP])
	if podTotalPrefillTokens[podIP] < 0 {
		klog.Errorf("podTotalPrefillTokens[%s]: %d is negative!", podIP, podTotalPrefillTokens[podIP])
	}
	return podTotalPrefillTokens[podIP]
}

func GetNumPrefillTokensForPod(podIP string) int {
	podTotalPrefillTokensMutex.RLock()
	defer podTotalPrefillTokensMutex.RUnlock()
	if _, ok := podTotalPrefillTokens[podIP]; !ok {
		return 0
	}
	return podTotalPrefillTokens[podIP]
}

func IncrementNumTotalTokensForPod(podIP string, numTokens int) {
	podToTotalTokensMutex.Lock()
	defer podToTotalTokensMutex.Unlock()
	if _, ok := podToTotalTokens[podIP]; !ok {
		podToTotalTokens[podIP] = 0
	}
	podToTotalTokens[podIP] += numTokens
	klog.V(5).Infof("TokenCount, Incremented total tokens for pod %s: %d and became %d", podIP, numTokens, podToTotalTokens[podIP])
}

func DecrementNumTotalTokensForPod(podIP string, numTokens int) int {
	podToTotalTokensMutex.Lock()
	defer podToTotalTokensMutex.Unlock()
	if _, ok := podToTotalTokens[podIP]; !ok {
		klog.Errorf("Pod name not found in podToTotalTokens: %s", podIP)
		return -1
	}
	podToTotalTokens[podIP] -= numTokens
	if podToTotalTokens[podIP] < 0 {
		klog.Errorf("podToTotalTokens[%s]: %d is negative!", podIP, podToTotalTokens[podIP])
	}
	klog.V(5).Infof("TokenCount, Decremented total tokens for pod %s: %d", podIP, podToTotalTokens[podIP])
	return podToTotalTokens[podIP]
}

func GetNumTotalTokensForPod(podIP string) int {
	podToTotalTokensMutex.RLock()
	defer podToTotalTokensMutex.RUnlock()
	if _, ok := podToTotalTokens[podIP]; !ok {
		return 0
	}
	return podToTotalTokens[podIP]
}

func GetNumTotalRequestsForAllPods() map[string]int {
	podToTotalTokensMutex.RLock()
	defer podToTotalTokensMutex.RUnlock()
	return podToTotalTokens
}

func IncrementNumDecodeTokensForPod(podIP string, numTokens int) int {
	podTotalDecodeTokensMutex.Lock()
	defer podTotalDecodeTokensMutex.Unlock()
	if _, ok := podTotalDecodeTokens[podIP]; !ok {
		podTotalDecodeTokens[podIP] = 0
	}
	old_numTokens := podTotalDecodeTokens[podIP]
	klog.V(5).Infof("TokenCount, Incremented decode tokens for pod %s by %d, from %d to %d", podIP, numTokens, old_numTokens, podTotalDecodeTokens[podIP])
	podTotalDecodeTokens[podIP] += numTokens
	return podTotalDecodeTokens[podIP]
}

func IncrementNumDecodeTokensForRequest(requestID string, numTokens int) int {
	requestToNumDecodeTokensMutex.Lock()
	defer requestToNumDecodeTokensMutex.Unlock()
	if _, ok := requestToNumDecodeTokens[requestID]; !ok {
		requestToNumDecodeTokens[requestID] = 0
	}
	requestToNumDecodeTokens[requestID] += numTokens
	return requestToNumDecodeTokens[requestID]
}

func DecrementNumDecodeTokensForPod(podIP string, numTokens int) int {
	podTotalDecodeTokensMutex.Lock()
	defer podTotalDecodeTokensMutex.Unlock()
	if _, ok := podTotalDecodeTokens[podIP]; !ok {
		klog.Errorf("Pod name not found in podTotalDecodeTokens: %s", podIP)
		return -1
	}
	old_numTokens := podTotalDecodeTokens[podIP]
	podTotalDecodeTokens[podIP] -= numTokens
	if podTotalDecodeTokens[podIP] < 0 {
		klog.Errorf("DecrementNumDecodeTokensForPod ,podTotalDecodeTokens[%s]: %d is negative!", podIP, podTotalDecodeTokens[podIP])
	}
	klog.V(5).Infof("TokenCount, Decremented decode tokens for pod %s by %d, from %d to %d", podIP, numTokens, old_numTokens, podTotalDecodeTokens[podIP])
	return podTotalDecodeTokens[podIP]
}

func GetNumDecodeTokensForPod(podIP string) int {
	podTotalDecodeTokensMutex.RLock()
	defer podTotalDecodeTokensMutex.RUnlock()
	if _, ok := podTotalDecodeTokens[podIP]; !ok {
		return 0
	}
	return podTotalDecodeTokens[podIP]
}

func GetpodTotalDecodeTokensMutex() *sync.RWMutex {
	return &podTotalDecodeTokensMutex
}

func GetNumDecodeTokensForAllPods() map[string]int {
	podTotalDecodeTokensMutex.RLock()
	defer podTotalDecodeTokensMutex.RUnlock()
	return podTotalDecodeTokens
}

func CleanupNumDecodeTokensForRequest(requestID string) {
	requestToNumDecodeTokensMutex.Lock()
	defer requestToNumDecodeTokensMutex.Unlock()
	if _, ok := requestToNumDecodeTokens[requestID]; ok {
		delete(requestToNumDecodeTokens, requestID)
	} else {
		klog.Errorf("requestToNumDecodeTokens not found for request ID %s", requestID)
	}
}

func CleanupNumPrefillTokensForRequest(requestID string) {
	requestToNumPrefillTokensMutex.Lock()
	defer requestToNumPrefillTokensMutex.Unlock()
	if _, ok := requestToNumPrefillTokens[requestID]; ok {
		delete(requestToNumPrefillTokens, requestID)
	} else {
		klog.Errorf("requestToNumPrefillTokens not found for request ID %s", requestID)
	}
}

func GetpodTotalPrefillTokensMutex() *sync.RWMutex {
	return &podTotalPrefillTokensMutex
}

func GetNumPrefillTokensForAllPods() map[string]int {
	podTotalPrefillTokensMutex.RLock()
	defer podTotalPrefillTokensMutex.RUnlock()
	return podTotalPrefillTokens
}

// /////////////////////////////////////////////////////

func ReadAndStorevLLMGPUKVCacheUsage(requestID string, pod *v1.Pod) error {
	url := fmt.Sprintf("http://%s:%d/metrics", pod.Status.PodIP, PodPort)
	allMetrics, err := metrics.ParseMetricsURL(url)
	if err != nil {
		err := fmt.Errorf("error parsing metric families from pod %s: %v", pod.Status.PodIP, err)
		return err
	}
	metricFamily, exists := allMetrics[fmt.Sprintf("vllm:%s", MetricGPUCacheUsagePerc)]
	if !exists {
		klog.Errorf("Metric %s not found for pod %s", MetricGPUCacheUsagePerc, pod.Status.PodIP)
		vllmGPUKVCacheUsageMutex.Lock()
		vllmGPUKVCacheUsage[requestID][pod.Status.PodIP] = -1
		vllmGPUKVCacheUsageMutex.Unlock()
	}
	for _, familyMetric := range metricFamily.Metric {
		modelName, _ := metrics.GetLabelValueForKey(familyMetric, "model_name")
		metricValue, err := metrics.GetCounterGaugeValue(familyMetric, metricFamily.GetType())
		if err != nil {
			klog.Errorf("Failed to parse metric %s from pod %s: %v", MetricGPUCacheUsagePerc, pod.Status.PodIP, err)
			continue
		}
		vllmGPUKVCacheUsageMutex.Lock()
		if _, ok := vllmGPUKVCacheUsage[requestID]; !ok {
			vllmGPUKVCacheUsage[requestID] = make(map[string]float64)
		}
		vllmGPUKVCacheUsage[requestID][pod.Status.PodIP] = metricValue
		vllmGPUKVCacheUsageMutex.Unlock()
		klog.V(5).Infof("Read metric %s for model %s from pod %s: %f", MetricGPUCacheUsagePerc, modelName, pod.Status.PodIP, metricValue)
	}
	return nil
}

func ReadAndStorevLLMCPUKVCacheUsage(requestID string, pod *v1.Pod) error {
	url := fmt.Sprintf("http://%s:%d/metrics", pod.Status.PodIP, PodPort)
	allMetrics, err := metrics.ParseMetricsURL(url)
	if err != nil {
		err := fmt.Errorf("error parsing metric families from pod %s: %v", pod.Status.PodIP, err)
		return err
	}
	metricFamily, exists := allMetrics[fmt.Sprintf("vllm:%s", MetricCPUCacheUsagePerc)]
	if !exists {
		klog.Errorf("Metric %s not found for pod %s", MetricCPUCacheUsagePerc, pod.Status.PodIP)
		vllmCPUKVCacheUsageMutex.Lock()
		vllmCPUKVCacheUsage[requestID][pod.Status.PodIP] = -1
		vllmCPUKVCacheUsageMutex.Unlock()
	}
	for _, familyMetric := range metricFamily.Metric {
		modelName, _ := metrics.GetLabelValueForKey(familyMetric, "model_name")
		metricValue, err := metrics.GetCounterGaugeValue(familyMetric, metricFamily.GetType())
		if err != nil {
			klog.Errorf("Failed to parse metric %s from pod %s: %v", MetricCPUCacheUsagePerc, pod.Status.PodIP, err)
			continue
		}
		vllmCPUKVCacheUsageMutex.Lock()
		if _, ok := vllmCPUKVCacheUsage[requestID]; !ok {
			vllmCPUKVCacheUsage[requestID] = make(map[string]float64)
		}
		vllmCPUKVCacheUsage[requestID][pod.Status.PodIP] = metricValue
		vllmCPUKVCacheUsageMutex.Unlock()
		klog.V(5).Infof("Read metric %s for model %s from pod %s: %f", MetricCPUCacheUsagePerc, modelName, pod.Status.PodIP, metricValue)
	}
	return nil
}

func ReadAndStorevLLMNumRequestsRunning(requestID string, pod *v1.Pod) error {
	url := fmt.Sprintf("http://%s:%d/metrics", pod.Status.PodIP, PodPort)
	allMetrics, err := metrics.ParseMetricsURL(url)
	if err != nil {
		err := fmt.Errorf("error parsing metric families from pod %s: %v", pod.Status.PodIP, err)
		return err
	}
	metricFamily, exists := allMetrics[fmt.Sprintf("vllm:%s", MetricNumRequestsRunning)]
	if !exists {
		klog.Errorf("Metric %s not found for pod %s", MetricNumRequestsRunning, pod.Status.PodIP)
		vllmNumRequestsRunningMutex.Lock()
		vllmNumRequestsRunning[requestID][pod.Status.PodIP] = -1
		vllmNumRequestsRunningMutex.Unlock()
	}
	for _, familyMetric := range metricFamily.Metric {
		modelName, _ := metrics.GetLabelValueForKey(familyMetric, "model_name")
		metricValue, err := metrics.GetCounterGaugeValue(familyMetric, metricFamily.GetType())
		if err != nil {
			klog.Errorf("Failed to parse metric %s from pod %s: %v", MetricNumRequestsRunning, pod.Status.PodIP, err)
			continue
		}
		vllmNumRequestsRunningMutex.Lock()
		if _, ok := vllmNumRequestsRunning[requestID]; !ok {
			vllmNumRequestsRunning[requestID] = make(map[string]float64)
		}
		vllmNumRequestsRunning[requestID][pod.Status.PodIP] = metricValue
		vllmNumRequestsRunningMutex.Unlock()
		klog.V(5).Infof("Read metric %s for model %s from pod %s: %f", MetricNumRequestsRunning, modelName, pod.Status.PodIP, metricValue)
	}
	return nil
}

func ReadAndStorevLLMNumRequestsWaiting(requestID string, pod *v1.Pod) error {
	url := fmt.Sprintf("http://%s:%d/metrics", pod.Status.PodIP, PodPort)
	allMetrics, err := metrics.ParseMetricsURL(url)
	if err != nil {
		err := fmt.Errorf("error parsing metric families from pod %s: %v", pod.Status.PodIP, err)
		return err
	}
	metricFamily, exists := allMetrics[fmt.Sprintf("vllm:%s", MetricNumRequestsWaiting)]
	if !exists {
		klog.Errorf("Metric %s not found for pod %s", MetricNumRequestsWaiting, pod.Status.PodIP)
		vllmNumRequestsWaitingMutex.Lock()
		vllmNumRequestsWaiting[requestID][pod.Status.PodIP] = -1
		vllmNumRequestsWaitingMutex.Unlock()
	}
	for _, familyMetric := range metricFamily.Metric {
		modelName, _ := metrics.GetLabelValueForKey(familyMetric, "model_name")
		metricValue, err := metrics.GetCounterGaugeValue(familyMetric, metricFamily.GetType())
		if err != nil {
			klog.Errorf("Failed to parse metric %s from pod %s: %v", MetricNumRequestsWaiting, pod.Status.PodIP, err)
			continue
		}
		vllmNumRequestsWaitingMutex.Lock()
		if _, ok := vllmNumRequestsWaiting[requestID]; !ok {
			vllmNumRequestsWaiting[requestID] = make(map[string]float64)
		}
		vllmNumRequestsWaiting[requestID][pod.Status.PodIP] = metricValue
		vllmNumRequestsWaitingMutex.Unlock()
		klog.V(5).Infof("Read metric %s for model %s from pod %s: %f", MetricNumRequestsWaiting, modelName, pod.Status.PodIP, metricValue)
	}
	return nil
}

func GetvllmGPUKVCacheUsageMutex() *sync.RWMutex {
	return &vllmGPUKVCacheUsageMutex
}

func GetvLLMGPUKVCacheUsageForTheRequestForAllPods(requestID string) (map[string]float64, error) {
	vllmGPUKVCacheUsageMutex.RLock()
	defer vllmGPUKVCacheUsageMutex.RUnlock()

	if usage, ok := vllmGPUKVCacheUsage[requestID]; ok {
		// Convert pod names to pod IPs before returning
		return usage, nil
	}
	return nil, fmt.Errorf("vLLM GPU KV cache usage not found for request ID %s", requestID)
}

func GetvllmCPUKVCacheUsageMutex() *sync.RWMutex {
	return &vllmCPUKVCacheUsageMutex
}

func GetvLLMCPUKVCacheUsageForTheRequestForAllPods(requestID string) (map[string]float64, error) {
	vllmCPUKVCacheUsageMutex.RLock()
	defer vllmCPUKVCacheUsageMutex.RUnlock()

	if usage, ok := vllmCPUKVCacheUsage[requestID]; ok {
		return usage, nil
	}
	return nil, fmt.Errorf("vLLM CPU KV cache usage not found for request ID %s", requestID)
}

func GetvllmNumRequestsRunningMutex() *sync.RWMutex {
	return &vllmNumRequestsRunningMutex
}

func GetvLLMNumRequestsRunningForTheRequestForAllPods(requestID string) (map[string]float64, error) {
	vllmNumRequestsRunningMutex.RLock()
	defer vllmNumRequestsRunningMutex.RUnlock()

	if requests, ok := vllmNumRequestsRunning[requestID]; ok {
		return requests, nil
	}
	return nil, fmt.Errorf("vLLM Num requests running not found for request ID %s", requestID)
}

func GetvllmNumRequestsWaitingMutex() *sync.RWMutex {
	return &vllmNumRequestsWaitingMutex
}

func GetvLLMNumRequestsWaitingForTheRequestForAllPods(requestID string) (map[string]float64, error) {
	vllmNumRequestsWaitingMutex.RLock()
	defer vllmNumRequestsWaitingMutex.RUnlock()

	if requests, ok := vllmNumRequestsWaiting[requestID]; ok {
		return requests, nil
	}
	return nil, fmt.Errorf("vLLM Num requests waiting not found for request ID %s", requestID)
}

func CleanupvLLMGPUKVCacheUsage(requestID string) {
	vllmGPUKVCacheUsageMutex.Lock()
	defer vllmGPUKVCacheUsageMutex.Unlock()
	if _, ok := vllmGPUKVCacheUsage[requestID]; ok {
		delete(vllmGPUKVCacheUsage, requestID)
	} else {
		klog.Errorf("vLLM GPU KV cache usage not found for request ID %s", requestID)
	}
}

func CleanupvLLMCPUKVCacheUsage(requestID string) {
	vllmCPUKVCacheUsageMutex.Lock()
	defer vllmCPUKVCacheUsageMutex.Unlock()
	if _, ok := vllmCPUKVCacheUsage[requestID]; ok {
		delete(vllmCPUKVCacheUsage, requestID)
	} else {
		klog.Errorf("vLLM CPU KV cache usage not found for request ID %s", requestID)
	}
}

func CleanupvLLMNumRequestsRunning(requestID string) {
	vllmNumRequestsRunningMutex.Lock()
	defer vllmNumRequestsRunningMutex.Unlock()
	if _, ok := vllmNumRequestsRunning[requestID]; ok {
		delete(vllmNumRequestsRunning, requestID)
	} else {
		klog.Errorf("vLLM Num requests running not found for request ID %s", requestID)
	}
}

func CleanupvLLMNumRequestsWaiting(requestID string) {
	vllmNumRequestsWaitingMutex.Lock()
	defer vllmNumRequestsWaitingMutex.Unlock()
	if _, ok := vllmNumRequestsWaiting[requestID]; ok {
		delete(vllmNumRequestsWaiting, requestID)
	} else {
		klog.Errorf("vLLM Num requests waiting not found for request ID %s", requestID)
	}
}
