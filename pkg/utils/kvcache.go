// In pkg/utils/kvcache.go
package utils

import (
	"fmt"
	"sync"

	"github.com/vllm-project/aibrix/pkg/metrics"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

var (
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
