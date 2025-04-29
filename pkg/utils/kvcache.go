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
	requestKVCacheMutex    sync.RWMutex
	requestKVCacheHitRatio map[string]float64            // requestID -> hit ratio
	requestAllPodsKVCache  map[string]map[string]float64 // requestID -> (podName -> hit ratio)

	PodInflightMutex    sync.RWMutex
	podInflightRequests map[string]int // podName -> num inflight requests

	requestInflightMutex sync.RWMutex
	requestInflight      map[string]map[string]int // requestID -> (podName -> num inflight requests

	requestToPodMutex sync.RWMutex
	requestToPod      map[string]string // requestID -> podName

	vllmGPUKVCacheUsageMutex sync.RWMutex
	vllmGPUKVCacheUsage      map[string]map[string]float64 // requestID -> (podName -> gpu kv cache usage)
	vllmCPUKVCacheUsageMutex sync.RWMutex
	vllmCPUKVCacheUsage      map[string]map[string]float64 // requestID -> (podName -> cpu kv cache usage)

	vllmNumRequestsRunningMutex sync.RWMutex
	vllmNumRequestsRunning      map[string]map[string]float64 // requestID -> (podName -> num requests running)
	vllmNumRequestsWaitingMutex sync.RWMutex
	vllmNumRequestsWaiting      map[string]map[string]float64 // requestID -> (podName -> num requests waiting

	podNameToIPMutex sync.RWMutex
	podNameToIP      map[string]string // podName -> podIP:port
)

const (
	MetricGPUCacheUsagePerc  = "gpu_cache_usage_perc"
	MetricCPUCacheUsagePerc  = "cpu_cache_usage_perc"
	MetricNumRequestsRunning = "num_requests_running"
	MetricNumRequestsWaiting = "num_requests_waiting"
	PodPort                  = 8000 // Same as in the metrics code
)

func init() {
	requestKVCacheHitRatio = make(map[string]float64)
	requestAllPodsKVCache = make(map[string]map[string]float64)
	requestInflight = make(map[string]map[string]int)
	podInflightRequests = make(map[string]int)
	requestToPod = make(map[string]string)

	vllmGPUKVCacheUsage = make(map[string]map[string]float64)
	vllmCPUKVCacheUsage = make(map[string]map[string]float64)
	vllmNumRequestsRunning = make(map[string]map[string]float64)
	vllmNumRequestsWaiting = make(map[string]map[string]float64)

	podNameToIP = make(map[string]string)
}

// StorePodIPMapping stores the mapping between pod name and IP
func StorePodIPMapping(podName string, podIP string) {
	podNameToIPMutex.Lock()
	defer podNameToIPMutex.Unlock()
	podNameToIP[podName] = podIP
}

// mapPodNamesToIPs converts a map with pod names as keys to a map with pod IPs as keys
func mapPodNamesToIPs(podNameMetrics map[string]float64) map[string]float64 {
	podNameToIPMutex.RLock()
	defer podNameToIPMutex.RUnlock()

	podIPMetrics := make(map[string]float64)
	for podName, value := range podNameMetrics {
		if podIP, exists := podNameToIP[podName]; exists {
			podIPMetrics[podIP] = value
		} else {
			// Keep the original key if no mapping exists
			podIPMetrics[podName] = value
		}
	}
	return podIPMetrics
}

// mapPodNamesToIPsInt converts a map with pod names as keys to a map with pod IPs as keys for int values
func mapPodNamesToIPsInt(podNameMetrics map[string]int) map[string]int {
	podNameToIPMutex.RLock()
	defer podNameToIPMutex.RUnlock()

	podIPMetrics := make(map[string]int)
	for podName, value := range podNameMetrics {
		if podIP, exists := podNameToIP[podName]; exists {
			podIPMetrics[podIP] = value
		} else {
			// Keep the original key if no mapping exists
			podIPMetrics[podName] = value
		}
	}
	return podIPMetrics
}

func CleanupRequestToPod(requestID string) {
	requestToPodMutex.Lock()
	defer requestToPodMutex.Unlock()

	delete(requestToPod, requestID)
}

func StoreKVCacheHitRatio(requestID string, podName string, ratio float64, allPodsRatios map[string]float64) {
	requestKVCacheMutex.Lock()
	defer requestKVCacheMutex.Unlock()
	requestKVCacheHitRatio[requestID] = ratio
	// klog.Infof("Saved KV cache hit ratio: %.2f%%", ratio*100)
	requestAllPodsKVCache[requestID] = allPodsRatios
}

func GetKVCacheHitRatio(requestID string) float64 {
	requestKVCacheMutex.RLock()
	defer requestKVCacheMutex.RUnlock()

	if ratio, ok := requestKVCacheHitRatio[requestID]; ok {
		return ratio
	}
	return -1
}

func GetAllPodsKVCacheHitRatios(requestID string) map[string]float64 {
	requestKVCacheMutex.RLock()
	defer requestKVCacheMutex.RUnlock()

	if ratios, ok := requestAllPodsKVCache[requestID]; ok {
		return mapPodNamesToIPs(ratios)
	}
	return make(map[string]float64)
}

func CleanupKVCacheHitRatio(requestID string) {
	requestKVCacheMutex.Lock()
	defer requestKVCacheMutex.Unlock()

	delete(requestKVCacheHitRatio, requestID)
	delete(requestAllPodsKVCache, requestID)
}

// /////////////////////////////////////////////////////////

func StoreRequestToPod(requestID string, podName string) {
	requestToPodMutex.Lock()
	defer requestToPodMutex.Unlock()
	if _, exists := requestToPod[requestID]; exists {
		klog.Errorf("requestID already exists in requestToPod: %s", requestID)
	}
	requestToPod[requestID] = podName
}

// GetPodNameForRequest retrieves the pod name for a given request ID
func GetPodNameForRequest(requestID string) (string, bool) {
	requestToPodMutex.RLock()
	defer requestToPodMutex.RUnlock()
	podName, exists := requestToPod[requestID]
	if !exists {
		klog.Errorf("requestID not found in requestToPod: %s", requestID)
		return "", false
	}
	return podName, exists
}

// Increment the number of inflight requests for a specific pod
func IncrementNumInflightForPod(requestID string) {
	PodInflightMutex.Lock()
	defer PodInflightMutex.Unlock()
	podName, exists := GetPodNameForRequest(requestID)
	if !exists {
		klog.Errorf("Pod name not found for request ID: %s", requestID)
		return
	}
	if _, ok := podInflightRequests[podName]; !ok {
		podInflightRequests[podName] = 0
	}

	podInflightRequests[podName]++
	klog.Infof("Incremented inflight requests for pod %s: %d", podName, podInflightRequests[podName])
}

// Decrement the number of inflight requests for a specific pod
func DecrementNumInflightForPod(requestID string) {
	PodInflightMutex.Lock()
	defer PodInflightMutex.Unlock()
	podName, exists := GetPodNameForRequest(requestID)
	if !exists {
		klog.Errorf("Pod name not found for request ID: %s", requestID)
		return
	}
	if _, ok := podInflightRequests[podName]; !ok {
		klog.Errorf("Pod name not found in podInflightRequests: %s", podName)
		return
	}

	podInflightRequests[podName]--
	if podInflightRequests[podName] <= 0 {
		klog.Errorf("podInflightRequests[%s]: %d is negative!", podName, podInflightRequests[podName])
	}
	klog.Infof("Decremented inflight requests for pod %s: %d", podName, podInflightRequests[podName])
}

func GetNumInflightRequestsForPod(podName string) int {
	PodInflightMutex.RLock()
	defer PodInflightMutex.RUnlock()
	if _, ok := podInflightRequests[podName]; !ok {
		return 0
	}
	return podInflightRequests[podName]
}

// /////////////////////////////////////////////////////

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
	for podName, numinflightrequests := range podInflightRequests {
		requestInflight[requestID][podName] = numinflightrequests
	}
}

func GetInflightRequestsForAllPods(requestID string) map[string]int {
	requestInflightMutex.RLock()
	defer requestInflightMutex.RUnlock()
	if inflightRequests, ok := requestInflight[requestID]; !ok {
		return make(map[string]int)
	} else {
		// Convert pod names to pod IPs before returning
		return mapPodNamesToIPsInt(inflightRequests)
	}
}

func CleanupInflightRequests(requestID string) {
	requestInflightMutex.Lock()
	defer requestInflightMutex.Unlock()
	delete(requestInflight, requestID)
}

// /////////////////////////////////////////////////////

func ReadAndStorevLLMGPUKVCacheUsage(requestID string, pod *v1.Pod) error {
	url := fmt.Sprintf("http://%s:%d/metrics", pod.Status.PodIP, PodPort)

	StorePodIPMapping(pod.Name, pod.Status.PodIP)

	allMetrics, err := metrics.ParseMetricsURL(url)
	if err != nil {
		err := fmt.Errorf("error parsing metric families from pod %s: %v", pod.Name, err)
		return err
	}
	metricFamily, exists := allMetrics[fmt.Sprintf("vllm:%s", MetricGPUCacheUsagePerc)]
	if !exists {
		klog.Errorf("Metric %s not found for pod %s", MetricGPUCacheUsagePerc, pod.Name)
		vllmGPUKVCacheUsageMutex.Lock()
		vllmGPUKVCacheUsage[requestID][pod.Name] = -1
		vllmGPUKVCacheUsageMutex.Unlock()
	}
	for _, familyMetric := range metricFamily.Metric {
		modelName, _ := metrics.GetLabelValueForKey(familyMetric, "model_name")
		metricValue, err := metrics.GetCounterGaugeValue(familyMetric, metricFamily.GetType())
		if err != nil {
			klog.Errorf("Failed to parse metric %s from pod %s: %v", MetricGPUCacheUsagePerc, pod.Name, err)
			continue
		}
		vllmGPUKVCacheUsageMutex.Lock()
		if _, ok := vllmGPUKVCacheUsage[requestID]; !ok {
			vllmGPUKVCacheUsage[requestID] = make(map[string]float64)
		}
		vllmGPUKVCacheUsage[requestID][pod.Name] = metricValue
		vllmGPUKVCacheUsageMutex.Unlock()
		klog.V(5).Infof("Read metric %s for model %s from pod %s: %f", MetricGPUCacheUsagePerc, modelName, pod.Name, metricValue)
	}
	return nil
}

func ReadAndStorevLLMCPUKVCacheUsage(requestID string, pod *v1.Pod) error {
	url := fmt.Sprintf("http://%s:%d/metrics", pod.Status.PodIP, PodPort)
	allMetrics, err := metrics.ParseMetricsURL(url)
	if err != nil {
		err := fmt.Errorf("error parsing metric families from pod %s: %v", pod.Name, err)
		return err
	}
	metricFamily, exists := allMetrics[fmt.Sprintf("vllm:%s", MetricCPUCacheUsagePerc)]
	if !exists {
		klog.Errorf("Metric %s not found for pod %s", MetricCPUCacheUsagePerc, pod.Name)
		vllmCPUKVCacheUsageMutex.Lock()
		vllmCPUKVCacheUsage[requestID][pod.Name] = -1
		vllmCPUKVCacheUsageMutex.Unlock()
	}
	for _, familyMetric := range metricFamily.Metric {
		modelName, _ := metrics.GetLabelValueForKey(familyMetric, "model_name")
		metricValue, err := metrics.GetCounterGaugeValue(familyMetric, metricFamily.GetType())
		if err != nil {
			klog.Errorf("Failed to parse metric %s from pod %s: %v", MetricCPUCacheUsagePerc, pod.Name, err)
			continue
		}
		vllmCPUKVCacheUsageMutex.Lock()
		if _, ok := vllmCPUKVCacheUsage[requestID]; !ok {
			vllmCPUKVCacheUsage[requestID] = make(map[string]float64)
		}
		vllmCPUKVCacheUsage[requestID][pod.Name] = metricValue
		vllmCPUKVCacheUsageMutex.Unlock()
		klog.V(5).Infof("Read metric %s for model %s from pod %s: %f", MetricCPUCacheUsagePerc, modelName, pod.Name, metricValue)
	}
	return nil
}

func ReadAndStorevLLMNumRequestsRunning(requestID string, pod *v1.Pod) error {
	url := fmt.Sprintf("http://%s:%d/metrics", pod.Status.PodIP, PodPort)
	allMetrics, err := metrics.ParseMetricsURL(url)
	if err != nil {
		err := fmt.Errorf("error parsing metric families from pod %s: %v", pod.Name, err)
		return err
	}
	metricFamily, exists := allMetrics[fmt.Sprintf("vllm:%s", MetricNumRequestsRunning)]
	if !exists {
		klog.Errorf("Metric %s not found for pod %s", MetricNumRequestsRunning, pod.Name)
		vllmNumRequestsRunningMutex.Lock()
		vllmNumRequestsRunning[requestID][pod.Name] = -1
		vllmNumRequestsRunningMutex.Unlock()
	}
	for _, familyMetric := range metricFamily.Metric {
		modelName, _ := metrics.GetLabelValueForKey(familyMetric, "model_name")
		metricValue, err := metrics.GetCounterGaugeValue(familyMetric, metricFamily.GetType())
		if err != nil {
			klog.Errorf("Failed to parse metric %s from pod %s: %v", MetricNumRequestsRunning, pod.Name, err)
			continue
		}
		vllmNumRequestsRunningMutex.Lock()
		if _, ok := vllmNumRequestsRunning[requestID]; !ok {
			vllmNumRequestsRunning[requestID] = make(map[string]float64)
		}
		vllmNumRequestsRunning[requestID][pod.Name] = metricValue
		vllmNumRequestsRunningMutex.Unlock()
		klog.V(5).Infof("Read metric %s for model %s from pod %s: %f", MetricNumRequestsRunning, modelName, pod.Name, metricValue)
	}
	return nil
}

func ReadAndStorevLLMNumRequestsWaiting(requestID string, pod *v1.Pod) error {
	url := fmt.Sprintf("http://%s:%d/metrics", pod.Status.PodIP, PodPort)
	allMetrics, err := metrics.ParseMetricsURL(url)
	if err != nil {
		err := fmt.Errorf("error parsing metric families from pod %s: %v", pod.Name, err)
		return err
	}
	metricFamily, exists := allMetrics[fmt.Sprintf("vllm:%s", MetricNumRequestsWaiting)]
	if !exists {
		klog.Errorf("Metric %s not found for pod %s", MetricNumRequestsWaiting, pod.Name)
		vllmNumRequestsWaitingMutex.Lock()
		vllmNumRequestsWaiting[requestID][pod.Name] = -1
		vllmNumRequestsWaitingMutex.Unlock()
	}
	for _, familyMetric := range metricFamily.Metric {
		modelName, _ := metrics.GetLabelValueForKey(familyMetric, "model_name")
		metricValue, err := metrics.GetCounterGaugeValue(familyMetric, metricFamily.GetType())
		if err != nil {
			klog.Errorf("Failed to parse metric %s from pod %s: %v", MetricNumRequestsWaiting, pod.Name, err)
			continue
		}
		vllmNumRequestsWaitingMutex.Lock()
		if _, ok := vllmNumRequestsWaiting[requestID]; !ok {
			vllmNumRequestsWaiting[requestID] = make(map[string]float64)
		}
		vllmNumRequestsWaiting[requestID][pod.Name] = metricValue
		vllmNumRequestsWaitingMutex.Unlock()
		klog.V(5).Infof("Read metric %s for model %s from pod %s: %f", MetricNumRequestsWaiting, modelName, pod.Name, metricValue)
	}
	return nil
}

func GetvLLMGPUKVCacheUsageForTheRequestForAllPods(requestID string) (map[string]float64, error) {
	vllmGPUKVCacheUsageMutex.RLock()
	defer vllmGPUKVCacheUsageMutex.RUnlock()

	if usage, ok := vllmGPUKVCacheUsage[requestID]; ok {
		// Convert pod names to pod IPs before returning
		return mapPodNamesToIPs(usage), nil
	}
	return nil, fmt.Errorf("vLLM GPU KV cache usage not found for request ID %s", requestID)
}

func GetvLLMCPUKVCacheUsageForTheRequestForAllPods(requestID string) (map[string]float64, error) {
	vllmCPUKVCacheUsageMutex.RLock()
	defer vllmCPUKVCacheUsageMutex.RUnlock()

	if usage, ok := vllmCPUKVCacheUsage[requestID]; ok {
		return mapPodNamesToIPs(usage), nil
	}
	return nil, fmt.Errorf("vLLM CPU KV cache usage not found for request ID %s", requestID)
}

func GetvLLMNumRequestsRunningForTheRequestForAllPods(requestID string) (map[string]float64, error) {
	vllmNumRequestsRunningMutex.RLock()
	defer vllmNumRequestsRunningMutex.RUnlock()

	if requests, ok := vllmNumRequestsRunning[requestID]; ok {
		return mapPodNamesToIPs(requests), nil
	}
	return nil, fmt.Errorf("vLLM Num requests running not found for request ID %s", requestID)
}

func GetvLLMNumRequestsWaitingForTheRequestForAllPods(requestID string) (map[string]float64, error) {
	vllmNumRequestsWaitingMutex.RLock()
	defer vllmNumRequestsWaitingMutex.RUnlock()

	if requests, ok := vllmNumRequestsWaiting[requestID]; ok {
		return mapPodNamesToIPs(requests), nil
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
