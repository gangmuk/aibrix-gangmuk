// In pkg/utils/kvcache.go
package utils

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net/http"
	"net/http/httptrace"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/vllm-project/aibrix/pkg/metrics"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

const (
	MetricGPUCacheUsagePerc  = "gpu_cache_usage_perc"
	MetricCPUCacheUsagePerc  = "cpu_cache_usage_perc"
	MetricNumRequestsRunning = "num_requests_running"
	MetricNumRequestsWaiting = "num_requests_waiting"
	PodPort                  = 8000 // Same as in the metrics code
)

type TimingResult struct {
	RequestID        string
	DNSTime          time.Duration
	ConnectionTime   time.Duration
	RequestTime      time.Duration
	SendTime         time.Duration
	ReadTime         time.Duration
	ParseTime        time.Duration
	TotalTime        time.Duration
	Success          bool
	Error            error
	ConnectionReused bool
}

// String returns a formatted string representation of the timing result
func (tr TimingResult) String() string {
	return fmt.Sprintf("Success:%v, DNS: %dms, Conn: %dms, ConnReuse: %v, Req: %dms, Send: %dms, Read: %dms, Parse: %dms, Total: %dms",
		tr.Success,
		tr.DNSTime.Milliseconds(),
		tr.ConnectionTime.Milliseconds(),
		tr.ConnectionReused,
		tr.RequestTime.Milliseconds(),
		tr.SendTime.Milliseconds(),
		tr.ReadTime.Milliseconds(),
		tr.ParseTime.Milliseconds(),
		tr.TotalTime.Milliseconds())
}

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

	// Variables for httptrace
	var dnsStart, dnsEnd, connectStart, connectEnd, tlsStart, tlsEnd time.Time

	// 3. Request Creation
	reqStart := time.Now()
	req, reqErr := http.NewRequest(method, url, bytes.NewBuffer(reqBody))
	result.RequestTime = time.Since(reqStart)

	if reqErr != nil {
		result.Error = reqErr
		klog.Errorf("requestID: %s - Failed to create request: %v", requestID, reqErr)
		result.TotalTime = -1
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

	// Create the HTTP trace before setting the timeout context
	trace := &httptrace.ClientTrace{
		// DNS timing
		DNSStart: func(info httptrace.DNSStartInfo) {
			dnsStart = time.Now()
			klog.Infof("requestID: %s - DNS lookup started for %s", requestID, info.Host)
		},
		DNSDone: func(info httptrace.DNSDoneInfo) {
			dnsEnd = time.Now()
			klog.Infof("requestID: %s - DNS lookup done for addresses: %v, error: %v, took: %v",
				requestID, info.Addrs, info.Err, time.Since(dnsStart))
		},

		// Connection timing
		ConnectStart: func(network, addr string) {
			connectStart = time.Now()
			klog.Infof("requestID: %s - TCP Connect started: network=%s, addr=%s",
				requestID, network, addr)
		},
		ConnectDone: func(network, addr string, err error) {
			connectEnd = time.Now()
			klog.Infof("requestID: %s - TCP Connect done: network=%s, addr=%s, err=%v, took: %v",
				requestID, network, addr, err, time.Since(connectStart))
		},

		// TLS timing
		TLSHandshakeStart: func() {
			tlsStart = time.Now()
			klog.Infof("requestID: %s - TLS handshake started", requestID)
		},
		TLSHandshakeDone: func(state tls.ConnectionState, err error) {
			tlsEnd = time.Now()
			klog.Infof("requestID: %s - TLS handshake done, version: %#x, error: %v, took: %v",
				requestID, state.Version, err, time.Since(tlsStart))
		},

		// Connection reuse information
		GotConn: func(info httptrace.GotConnInfo) {
			klog.Infof("requestID: %s - Got connection: reused=%v, was_idle=%v, idle_time=%v",
				requestID, info.Reused, info.WasIdle, info.IdleTime)

			// Also log the connection details
			klog.Infof("requestID: %s - Connection details: local=%v, remote=%v",
				requestID, info.Conn.LocalAddr(), info.Conn.RemoteAddr())

			result.ConnectionReused = info.Reused
		},

		// Request/response timing
		WroteHeaders: func() {
			klog.Infof("requestID: %s - Request headers written", requestID)
		},
		WroteRequest: func(info httptrace.WroteRequestInfo) {
			klog.Infof("requestID: %s - Request written, error: %v", requestID, info.Err)
		},
		GotFirstResponseByte: func() {
			klog.Infof("requestID: %s - Got first response byte", requestID)
		},
	}

	// Set context with timeout and trace
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	ctx = httptrace.WithClientTrace(ctx, trace)
	req = req.WithContext(ctx)

	// 4. Send Request
	sendStart := time.Now()
	resp, sendErr := client.Do(req)
	result.SendTime = time.Since(sendStart)

	// Update timing results based on httptrace data
	if !dnsStart.IsZero() && !dnsEnd.IsZero() {
		result.DNSTime = dnsEnd.Sub(dnsStart)
	}

	if !connectStart.IsZero() && !connectEnd.IsZero() {
		result.ConnectionTime = connectEnd.Sub(connectStart)
		if !tlsStart.IsZero() && !tlsEnd.IsZero() {
			// Add TLS time to connection time
			result.ConnectionTime += tlsEnd.Sub(tlsStart)
		}
	}

	if sendErr != nil {
		result.Error = sendErr
		klog.Errorf("requestID: %s - Failed to send request: %v", requestID, sendErr)
		result.TotalTime = -1
		return nil, result, sendErr
	}
	defer resp.Body.Close()

	// Check connection reuse info from headers as well
	connectionHeader := resp.Header.Get("Connection")
	if connectionHeader == "keep-alive" {
		result.ConnectionReused = true
	}

	// 5. Read Response
	readStart := time.Now()
	body, readErr := ioutil.ReadAll(resp.Body)
	result.ReadTime = time.Since(readStart)

	if readErr != nil {
		result.Error = readErr
		klog.Errorf("requestID: %s - Failed to read response: %v", requestID, readErr)
		result.TotalTime = -1
		return nil, result, readErr
	}

	// Check status code
	if resp.StatusCode != http.StatusOK {
		err := fmt.Errorf("requestID: %s - non-200 status code: %d", requestID, resp.StatusCode)
		result.Error = err
		result.TotalTime = -1
		return body, result, err
	}

	// Skip parsing timing since we don't know the response type
	result.ParseTime = 0
	result.Success = true
	result.TotalTime = time.Since(startTime)
	klog.Infof("requestID: %s - Request completed successfully, total time took %dms", requestID, result.TotalTime.Milliseconds())
	return body, result, nil
}

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

// PredictionRequest is the request body sent to the predictor service
type PredictionRequest struct {
	Pods []PodFeatures `json:"pods"`
}

// PredictionResponse is the response from the predictor service
type PredictionResponse struct {
	Predictions map[string]map[string]float64 `json:"predictions"`
}

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

func (t *PodMetricsTracker) cleanupOldMetrics(podIP string, now time.Time) {
	cutoff := now.Add(-(t.WindowSize * 2)) // Keep metrics for twice the window size
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
		klog.Infof("Number of valid metrics for pod %s: %d", podIP, len(validMetrics))
		var ttftValues []int64
		var tpotValues []int64
		var ttftSum, tpotSum int64
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
			if m.TPOT > 0 {
				tpotValues = append(tpotValues, m.TPOT)
				tpotSum += m.TPOT
				totalDecodeTokens++
			}
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
			klog.Infof("GetDetailedMetrics, Getting TTFT related values!!, %s, %d", podIP, len(ttftValues))
			detailedMetrics.AvgTTFT = float64(ttftSum) / float64(len(ttftValues))
			detailedMetrics.MinTTFT = ttftValues[0]
			detailedMetrics.MaxTTFT = ttftValues[len(ttftValues)-1]
			detailedMetrics.P50TTFT = percentile(ttftValues, 50)
			detailedMetrics.P90TTFT = percentile(ttftValues, 90)
			detailedMetrics.P95TTFT = percentile(ttftValues, 95)
			detailedMetrics.P99TTFT = percentile(ttftValues, 99)
		}
		if len(tpotValues) > 0 {
			klog.Infof("GetDetailedMetrics, Getting TPOT related values!!, %s, %d", podIP, len(tpotValues))
			detailedMetrics.AvgTPOT = float64(tpotSum) / float64(len(tpotValues))
			detailedMetrics.MinTPOT = tpotValues[0]
			detailedMetrics.MaxTPOT = tpotValues[len(tpotValues)-1]
			detailedMetrics.P50TPOT = percentile(tpotValues, 50)
			detailedMetrics.P90TPOT = percentile(tpotValues, 90)
			detailedMetrics.P95TPOT = percentile(tpotValues, 95)
			detailedMetrics.P99TPOT = percentile(tpotValues, 99)
		}
		result[podIP] = detailedMetrics
		// Print all metrics for debugging
		klog.Infof("Pod %s metrics: %+v", podIP, detailedMetrics)
	}
	return result
}

func NewPodMetricsTracker(windowSize time.Duration) *PodMetricsTracker {
	return &PodMetricsTracker{
		podMetrics: make(map[string][]PodMetric),
		WindowSize: windowSize,
	}
}

var (
	UseRealRequest = LoadEnv("AIBRIX_RL_ROUTER_USE_REAL_REQUEST", "true")

	RunningPodRegistry      = make(map[string]string) // Map to track running pods: podIP -> Pod object
	RunningPodRegistryMutex sync.RWMutex

	RequestTimings   sync.Map           // Map to track request timing information: requestID -> *RequestTiming
	MetricsTracker   *PodMetricsTracker // Track timing metrics for pods
	MetricsEnabled   atomic.Bool        // Flag to enable/disable metrics collection
	MetricsLogTicker *time.Ticker       // Ticker for periodic metrics logging

	RequestToLogMessageMutex sync.RWMutex
	RequestToLogMessage      = make(map[string]string) // requestID -> log message

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

func CleanupRoutineForpodMetrics() {
	go func() {
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				if MetricsEnabled.Load() {
					klog.Info("Running periodic metrics cleanup")
					MetricsTracker.CleanupAllMetrics()
				}
			}
		}
	}()
}

func CleanupAllRequestLogMessage() {
	RequestToLogMessageMutex.Lock()
	defer RequestToLogMessageMutex.Unlock()

	for requestID := range RequestToLogMessage {
		delete(RequestToLogMessage, requestID)
	}
	klog.Infof("Cleaned up all log messages in RequestToLogMessage")
}

func init() {
	// CleanupRoutineForpodMetrics()
	RunningPodRegistry = make(map[string]string)

	MetricsTracker = NewPodMetricsTracker(1 * time.Second)
	MetricsEnabled.Store(true)
	MetricsLogTicker = time.NewTicker(10 * time.Second)

	RequestToLogMessageMutex = sync.RWMutex{}
	RequestToLogMessage = make(map[string]string)

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

func AddPodToRegistry(podIP string, podName string) {
	RunningPodRegistryMutex.Lock()
	defer RunningPodRegistryMutex.Unlock()

	// Add the pod to the registry
	RunningPodRegistry[podIP] = podName
	klog.Infof("Registered podIP %s, podName %s", podIP, podName)
}

func SyncPodRegistry(readyPods []*v1.Pod) {
	RunningPodRegistryMutex.Lock()
	defer RunningPodRegistryMutex.Unlock()

	// Clear the registry
	RunningPodRegistry = make(map[string]string)

	// Add all ready pods to the registry
	for _, pod := range readyPods {
		podIP := pod.Status.PodIP
		podName := pod.Name
		RunningPodRegistry[podIP] = podName
		klog.Infof("Registered podIP %s, podName %s", podIP, podName)
	}
}

func DeletePodFromRegistry(podIP string) {
	RunningPodRegistryMutex.Lock()
	defer RunningPodRegistryMutex.Unlock()

	// Delete the pod from the registry
	if _, exists := RunningPodRegistry[podIP]; exists {
		delete(RunningPodRegistry, podIP)
		klog.Infof("Deleted pod with IP %s from registry", podIP)
	} else {
		klog.Errorf("Failed to delete pod with IP %s, not found in registry", podIP)
	}
}

func GetAllPodIPsFromRegistry() []string {
	RunningPodRegistryMutex.RLock()
	defer RunningPodRegistryMutex.RUnlock()

	podIPs := make([]string, 0, len(RunningPodRegistry))
	for podIP := range RunningPodRegistry {
		podIPs = append(podIPs, podIP)
	}
	return podIPs
}

func AddRequestLogMessage(requestID string, logMessage string) {
	RequestToLogMessageMutex.Lock()
	defer RequestToLogMessageMutex.Unlock()

	if _, exists := RequestToLogMessage[requestID]; !exists {
		RequestToLogMessage[requestID] = logMessage
	} else {
		klog.Errorf("Request ID %s already exists in RequestToLogMessage", requestID)
	}
}

func DeleteRequestLogMessage(requestID string) {
	RequestToLogMessageMutex.Lock()
	defer RequestToLogMessageMutex.Unlock()

	if _, exists := RequestToLogMessage[requestID]; exists {
		delete(RequestToLogMessage, requestID)
		klog.Infof("Deleted log message for request ID: %s", requestID)
	} else {
		klog.Errorf("Failed to delete log message for request ID: %s, not found", requestID)
	}
}

func GetRequestLogMessage(requestID string) (string, bool) {
	RequestToLogMessageMutex.RLock()
	defer RequestToLogMessageMutex.RUnlock()

	logMessage, exists := RequestToLogMessage[requestID]
	if !exists {
		klog.Errorf("Failed GetRequestLogMessage, Request ID %s not found in RequestToLogMessage", requestID)
		return "", false
	}
	return logMessage, true
}

func CleanupRequestLogMessage(requestID string) {
	RequestToLogMessageMutex.Lock()
	defer RequestToLogMessageMutex.Unlock()

	if _, exists := RequestToLogMessage[requestID]; exists {
		delete(RequestToLogMessage, requestID)
		klog.Infof("CleanupRequestLogMessage, Deleted log message for request ID: %s", requestID)
	} else {
		klog.Errorf("Failed CleanupRequestLogMessage, No log message found for request ID: %s", requestID)
	}
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

func GetRequestPodMetrics(requestID string) map[string]PodDetailedMetrics {
	podMetricsMutex.Lock()
	defer podMetricsMutex.Unlock()

	metrics, exists := requestToPodMetrics[requestID]
	if !exists {
		klog.ErrorS(nil, "Failed to find metrics for request ID", "requestID", requestID)
		return make(map[string]PodDetailedMetrics)
	}
	return metrics // Return directly, no need for copying
}

func CleanupRequestPodMetrics(requestID string) {
	podMetricsMutex.Lock()
	defer podMetricsMutex.Unlock()

	if _, exists := requestToPodMetrics[requestID]; exists {
		delete(requestToPodMetrics, requestID)
		klog.Infof("CleanupRequestPodMetrics, Deleted metrics for request ID: %s", requestID)
	} else {
		klog.Errorf("CleanupRequestPodMetrics, No metrics found for request ID: %s", requestID)
	}
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
	klog.Infof("StoreKVCacheHitRatio, Stored KV cache hit ratios for request %s: %v", requestID, allPodsRatios)
}

func GetAllPodsKVCacheHitRatios(requestID string) map[string]int {
	requestAllPodsKVCacheMutex.RLock()
	defer requestAllPodsKVCacheMutex.RUnlock()
	if ratios, ok := requestAllPodsKVCache[requestID]; ok {
		// Create a copy to avoid race conditions after the lock is released
		result := make(map[string]int, len(ratios))
		for k, v := range ratios {
			result[k] = v
		}
		return result
	}
	klog.Errorf("requestID not found in requestAllPodsKVCache: %s", requestID)
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
func IncrementNumInflightForPod(requestID string, podIP string) {
	PodInflightMutex.Lock()
	defer PodInflightMutex.Unlock()
	if _, ok := podInflightRequests[podIP]; !ok {
		podInflightRequests[podIP] = 0
	}

	podInflightRequests[podIP]++
	klog.V(5).Infof("Incremented inflight requests for pod %s: %d", podIP, podInflightRequests[podIP])
}

// Decrement the number of inflight requests for a specific pod
func DecrementNumInflightForPod(requestID string, podIP string) {
	PodInflightMutex.Lock()
	defer PodInflightMutex.Unlock()
	// podIP, exists := GetPodIPForRequest(requestID)
	// if !exists {
	// 	klog.Errorf("Pod name not found for request ID: %s", requestID)
	// 	return
	// }
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

func GetNumInflightRequestsForPod(podIP string) (int, bool) {
	PodInflightMutex.RLock()
	defer PodInflightMutex.RUnlock()
	val, exists := podInflightRequests[podIP]
	return val, exists
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
		// Create a copy to avoid race conditions after the lock is released
		result := make(map[string]int, len(inflightRequests))
		for k, v := range inflightRequests {
			result[k] = v
		}
		return result
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

func GetNumPrefillTokensForPod(podIP string) (int, bool) {
	podTotalPrefillTokensMutex.RLock()
	defer podTotalPrefillTokensMutex.RUnlock()
	val, exists := podTotalPrefillTokens[podIP]
	return val, exists
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

func GetNumTotalTokensForPod(podIP string) (int, bool) {
	podToTotalTokensMutex.RLock()
	defer podToTotalTokensMutex.RUnlock()
	val, exists := podToTotalTokens[podIP]
	return val, exists
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

func GetNumDecodeTokensForPod(podIP string) (int, bool) {
	podTotalDecodeTokensMutex.RLock()
	defer podTotalDecodeTokensMutex.RUnlock()
	val, exists := podTotalDecodeTokens[podIP]
	return val, exists
}

func GetpodTotalDecodeTokensMutex() *sync.RWMutex {
	return &podTotalDecodeTokensMutex
}

func GetNumDecodeTokensForAllPods() map[string]int {
	podTotalDecodeTokensMutex.RLock()
	defer podTotalDecodeTokensMutex.RUnlock()

	// Create a copy to avoid race conditions after the lock is released
	result := make(map[string]int, len(podTotalDecodeTokens))
	for k, v := range podTotalDecodeTokens {
		result[k] = v
	}
	return result
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

	// Create a copy to avoid race conditions after the lock is released
	result := make(map[string]int, len(podTotalPrefillTokens))
	for k, v := range podTotalPrefillTokens {
		result[k] = v
	}
	return result
}

// /////////////////////////////////////////////////////

func ReadAndStoreVLLMMetric(requestID string, pod *v1.Pod, metricName string) error {
	url := fmt.Sprintf("http://%s:%d/metrics", pod.Status.PodIP, PodPort)
	allMetrics, err := metrics.ParseMetricsURL(url)
	if err != nil {
		return fmt.Errorf("error parsing metric families from pod %s: %v", pod.Status.PodIP, err)
	}

	fullMetricName := fmt.Sprintf("vllm:%s", metricName)
	metricFamily, exists := allMetrics[fullMetricName]

	// Select the appropriate storage and mutex based on metricName
	var metricStorage map[string]map[string]float64
	var metricMutex *sync.RWMutex

	switch metricName {
	case MetricGPUCacheUsagePerc:
		metricStorage = vllmGPUKVCacheUsage
		metricMutex = &vllmGPUKVCacheUsageMutex
	case MetricCPUCacheUsagePerc:
		metricStorage = vllmCPUKVCacheUsage
		metricMutex = &vllmCPUKVCacheUsageMutex
	case MetricNumRequestsRunning:
		metricStorage = vllmNumRequestsRunning
		metricMutex = &vllmNumRequestsRunningMutex
	case MetricNumRequestsWaiting:
		metricStorage = vllmNumRequestsWaiting
		metricMutex = &vllmNumRequestsWaitingMutex
	default:
		return fmt.Errorf("unknown metric name: %s", metricName)
	}

	if !exists {
		klog.Errorf("Metric %s not found for pod %s", metricName, pod.Status.PodIP)
		metricMutex.Lock()
		if _, ok := metricStorage[requestID]; !ok {
			metricStorage[requestID] = make(map[string]float64)
		}
		metricStorage[requestID][pod.Status.PodIP] = -1
		metricMutex.Unlock()
		return nil
	}

	for _, familyMetric := range metricFamily.Metric {
		metricValue, err := metrics.GetCounterGaugeValue(familyMetric, metricFamily.GetType())
		if err != nil {
			klog.Errorf("Failed to parse metric %s from pod %s: %v", metricName, pod.Status.PodIP, err)
			continue
		}

		metricMutex.Lock()
		if _, ok := metricStorage[requestID]; !ok {
			metricStorage[requestID] = make(map[string]float64)
		}
		metricStorage[requestID][pod.Status.PodIP] = metricValue
		metricMutex.Unlock()

		klog.V(5).Infof("Stored requestID: %s, metric %s, pod %s: %f", requestID, metricName, pod.Status.PodIP, metricValue)
	}

	return nil
}

func GetvllmGPUKVCacheUsageMutex() *sync.RWMutex {
	return &vllmGPUKVCacheUsageMutex
}

func GetvLLMGPUKVCacheUsageForAllPods(requestID string) (map[string]float64, error) {
	vllmGPUKVCacheUsageMutex.RLock()
	defer vllmGPUKVCacheUsageMutex.RUnlock()

	if usage, ok := vllmGPUKVCacheUsage[requestID]; ok {
		// Create a copy to avoid race conditions after the lock is released
		result := make(map[string]float64, len(usage))
		for k, v := range usage {
			result[k] = v
		}
		return result, nil
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
		// Create a copy to avoid race conditions after the lock is released
		result := make(map[string]float64, len(usage))
		for k, v := range usage {
			result[k] = v
		}
		return result, nil
	}
	return nil, fmt.Errorf("vLLM CPU KV cache usage not found for request ID %s", requestID)
}

func GetvllmNumRequestsRunningMutex() *sync.RWMutex {
	return &vllmNumRequestsRunningMutex
}

func GetvLLMNumRequestsRunningForAllPods(requestID string) (map[string]float64, error) {
	vllmNumRequestsRunningMutex.RLock()
	defer vllmNumRequestsRunningMutex.RUnlock()

	if requests, ok := vllmNumRequestsRunning[requestID]; ok {
		// Create a copy to avoid race conditions after the lock is released
		result := make(map[string]float64, len(requests))
		for k, v := range requests {
			result[k] = v
		}
		return result, nil
	}
	return nil, fmt.Errorf("vLLM Num requests running not found for request ID %s", requestID)
}

func GetvllmNumRequestsWaitingMutex() *sync.RWMutex {
	return &vllmNumRequestsWaitingMutex
}

func GetvLLMNumRequestsWaitingForAllPods(requestID string) (map[string]float64, error) {
	vllmNumRequestsWaitingMutex.RLock()
	defer vllmNumRequestsWaitingMutex.RUnlock()

	if requests, ok := vllmNumRequestsWaiting[requestID]; ok {
		// Create a copy to avoid race conditions after the lock is released
		result := make(map[string]float64, len(requests))
		for k, v := range requests {
			result[k] = v
		}
		return result, nil
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

// GetKVCacheHitRatioForPod safely retrieves KV cache hit ratio for a specific pod and request
func GetKVCacheHitRatioForPod(requestID string, podIP string) (int, bool) {
	requestAllPodsKVCacheMutex.RLock()
	defer requestAllPodsKVCacheMutex.RUnlock()
	ratios, exists := requestAllPodsKVCache[requestID]
	if !exists {
		klog.Errorf("KV cache hit ratios not found for request ID %s", requestID)
		return 0, false
	}
	val, exists := ratios[podIP]
	return val, exists
}

// GetVLLMGPUKVCacheUsageForPod safely retrieves GPU KV cache usage for a specific pod and request
func GetVLLMGPUKVCacheUsageForPod(requestID string, podIP string) (float64, bool) {
	vllmGPUKVCacheUsageMutex.RLock()
	defer vllmGPUKVCacheUsageMutex.RUnlock()
	usage, exists := vllmGPUKVCacheUsage[requestID]
	if !exists {
		klog.Errorf("vLLM GPU KV cache usage not found for request ID %s", requestID)
		return 0, false
	}
	val, exists := usage[podIP]
	return val, exists
}

// GetVLLMCPUKVCacheUsageForPod safely retrieves CPU KV cache usage for a specific pod and request
func GetVLLMCPUKVCacheUsageForPod(requestID string, podIP string) (float64, bool) {
	vllmCPUKVCacheUsageMutex.RLock()
	defer vllmCPUKVCacheUsageMutex.RUnlock()
	usage, exists := vllmCPUKVCacheUsage[requestID]
	if !exists {
		klog.Errorf("vLLM CPU KV cache usage not found for request ID %s", requestID)
		return 0, false
	}
	val, exists := usage[podIP]
	return val, exists
}

// GetVLLMNumRequestsRunningForPod safely retrieves number of running requests for a specific pod and request
func GetVLLMNumRequestsRunningForPod(requestID string, podIP string) (float64, bool) {
	vllmNumRequestsRunningMutex.RLock()
	defer vllmNumRequestsRunningMutex.RUnlock()
	requests, exists := vllmNumRequestsRunning[requestID]
	if !exists {
		klog.Errorf("vLLM Num requests running not found for request ID %s", requestID)
		return 0, false
	}
	val, exists := requests[podIP]
	return val, exists
}

// GetVLLMNumRequestsWaitingForPod safely retrieves number of waiting requests for a specific pod and request
func GetVLLMNumRequestsWaitingForPod(requestID string, podIP string) (float64, bool) {
	vllmNumRequestsWaitingMutex.RLock()
	defer vllmNumRequestsWaitingMutex.RUnlock()
	requests, exists := vllmNumRequestsWaiting[requestID]
	if !exists {
		klog.Errorf("vLLM Num requests waiting not found for request ID %s", requestID)
		return 0, false
	}
	val, exists := requests[podIP]
	return val, exists
}

func GenerateLogMessages(podIPs []string, numLogs int) []string {
	logs := make([]string, numLogs)
	for i := 0; i < numLogs; i++ {
		logs[i] = generateSingleLog(podIPs, i)
	}
	return logs
}

func generateSingleLog(podIPs []string, requestID int) string {
	// Generate random metrics that make sense for a model serving cluster
	requestStartTime := time.Now().UnixNano() / 1000           // microseconds
	requestEndTime := requestStartTime + rand.Int63n(15000000) // 0-15 seconds later in microseconds

	// Select a random pod for this request
	if len(podIPs) == 0 {
		klog.Error("No pod IPs available to select from")
		return ""
	}
	selectedPod := podIPs[rand.Intn(len(podIPs))]

	// Generate realistic latency metrics
	ttft := 150 + rand.Intn(500)               // Time to first token: 150-650ms
	avgTpot := 30 + rand.Intn(50)              // Average time per output token: 30-80ms
	totalDecodeTime := 5000 + rand.Intn(15000) // Total decode time: 5-20 seconds
	e2e := totalDecodeTime + rand.Intn(2000)   // End-to-end time: total decode + 0-2 seconds

	// Generate token counts
	numInputTokens := 500 + rand.Intn(2000) // 500-2500 input tokens
	numOutputTokens := 100 + rand.Intn(800) // 100-900 output tokens
	numTotalTokens := numInputTokens + numOutputTokens

	// Build JSON objects for different metrics
	kvCacheHitRatios := buildPodJsonMap(podIPs, 0, 1)   // KV cache hit ratios are typically 0
	inflightRequests := buildPodJsonMap(podIPs, 1, 3)   // 1-3 requests per pod
	gpuKVCacheUsage := buildPodJsonMap(podIPs, 0, 0.5)  // 0-30% GPU KV cache usage
	cpuKVCacheUsage := buildPodJsonMap(podIPs, 0, 0.0)  // 0-20% CPU KV cache usage
	numRequestsRunning := buildPodJsonMap(podIPs, 0, 2) // 0-2 requests running per pod
	numRequestsWaiting := buildPodJsonMap(podIPs, 0, 1) // 0-1 requests waiting per pod

	// Build per-pod detailed metrics for the last second
	podMetricsLastSecond := buildPodMetricsLastSecond(podIPs)

	// Token counts per pod
	prefillTokens := buildPodJsonMap(podIPs, 1000, 100000) // 0-130K prefill tokens per pod
	decodeTokens := buildPodJsonMap(podIPs, 1000, 100000)  // 80K-300K decode tokens per pod

	logFormat := `**@latency_metrics@requestID@%d@request_start_time@%d@request_end_time@%d@selectedpod@%s@ttft@%d@avg_tpot@%d@total_decode_time@%d@e2e@%d@numInputTokens@%d@numOutputTokens@%d@numTotalTokens@%d@allPodsKvCacheHitRatios@%s@numInflightRequestsAllPods@%s@vllmGPUKVCacheUsage@%s@vllmCPUKVCacheUsage@%s@vllmNumRequestsRunning@%s@vllmNumRequestsWaiting@%s@podMetricsLastSecond@%s@numPrefillTokensForAllPods@%s@numDecodeTokensForAllPods@%s`

	ret := fmt.Sprintf(logFormat,
		requestID,
		requestStartTime,
		requestEndTime,
		selectedPod,
		ttft,
		avgTpot,
		totalDecodeTime,
		e2e,
		numInputTokens,
		numOutputTokens,
		numTotalTokens,
		kvCacheHitRatios,
		inflightRequests,
		gpuKVCacheUsage,
		cpuKVCacheUsage,
		numRequestsRunning,
		numRequestsWaiting,
		podMetricsLastSecond,
		prefillTokens,
		decodeTokens,
	)
	klog.V(5).Infof("Generated log message: %s", ret)
	return ret
}

// buildPodJsonMap creates a JSON object mapping pod IPs to random values
func buildPodJsonMap(podIPs []string, min float64, max float64) string {
	parts := make([]string, len(podIPs))

	for i, ip := range podIPs {
		var value interface{}
		if max <= 1 {
			// Generate a float for percentages
			value = min + rand.Float64()*(max-min)
		} else {
			// Generate an integer for counts
			value = int(min) + rand.Intn(int(max-min)+1)
		}

		// Format based on whether it's a float or int
		switch v := value.(type) {
		case float64:
			parts[i] = fmt.Sprintf(`"%s":%g`, ip, v)
		case int:
			parts[i] = fmt.Sprintf(`"%s":%d`, ip, v)
		}
	}

	return fmt.Sprintf("{%s}", strings.Join(parts, ","))
}

// buildPodMetricsLastSecond creates realistic per-pod metrics for the last second
func buildPodMetricsLastSecond(podIPs []string) string {
	parts := make([]string, len(podIPs))

	for i, ip := range podIPs {
		// Generate realistic TTFTs for this pod
		ttftAvg := 300 + rand.Float64()*3000 // 300-3300ms average TTFT
		ttftMin := ttftAvg * 0.8             // Min is ~80% of average
		ttftMax := ttftAvg * 1.2             // Max is ~120% of average
		ttftP50 := ttftAvg * 0.95            // p50 is close to average
		ttftP90 := ttftAvg * 1.1             // p90 is slightly higher
		ttftP95 := ttftAvg * 1.15            // p95 is higher still
		ttftP99 := ttftAvg * 1.18            // p99 is highest
		ttftSamples := 2 + rand.Intn(13)     // 2-15 samples

		// Generate realistic TPOTs for this pod
		tpotAvg := 40 + rand.Float64()*350 // 40-390ms average TPOT
		tpotMin := tpotAvg * 0.7           // Min is lower than average
		tpotMax := tpotAvg * 1.5           // Max can be much higher
		tpotP50 := tpotAvg * 0.9           // p50 is close to but below average
		tpotP90 := tpotAvg * 1.2           // p90 is higher
		tpotP95 := tpotAvg * 1.4           // p95 is higher still
		tpotP99 := tpotAvg * 1.45          // p99 is highest
		tpotSamples := 50 + rand.Intn(600) // 50-650 samples

		// Total tokens
		totalRequests := ttftSamples
		totalDecodeTokens := tpotSamples
		totalPrefillTokens := 2000 + rand.Intn(35000) // 2K-37K prefill tokens
		totalTokens := totalDecodeTokens + totalPrefillTokens

		metrics := fmt.Sprintf(`"last_second_avg_ttft_ms":%g,"last_second_min_ttft_ms":%g,"last_second_max_ttft_ms":%g,"last_second_p50_ttft_ms":%g,"last_second_p90_ttft_ms":%g,"last_second_p95_ttft_ms":%g,"last_second_p99_ttft_ms":%g,"last_second_ttft_samples":%d,"last_second_avg_tpot_ms":%g,"last_second_min_tpot_ms":%g,"last_second_max_tpot_ms":%g,"last_second_p50_tpot_ms":%g,"last_second_p90_tpot_ms":%g,"last_second_p95_tpot_ms":%g,"last_second_p99_tpot_ms":%g,"last_second_tpot_samples":%d,"last_second_total_requests":%d,"last_second_total_decode_tokens":%d,"last_second_total_prefill_tokens":%d,"last_second_total_tokens":%d`,
			ttftAvg, ttftMin, ttftMax, ttftP50, ttftP90, ttftP95, ttftP99, ttftSamples,
			tpotAvg, tpotMin, tpotMax, tpotP50, tpotP90, tpotP95, tpotP99, tpotSamples,
			totalRequests, totalDecodeTokens, totalPrefillTokens, totalTokens)

		parts[i] = fmt.Sprintf(`"%s":{%s}`, ip, metrics)
	}

	return fmt.Sprintf("{%s}", strings.Join(parts, ","))
}
