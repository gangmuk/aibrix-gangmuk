// In pkg/utils/kvcache.go
package utils

import (
	"sync"

	"k8s.io/klog/v2"
)

var (
	// Global map to store KV cache hit ratios per request
	requestKVCacheMutex    sync.RWMutex
	requestKVCacheHitRatio map[string]float64            // requestID -> hit ratio
	requestAllPodsKVCache  map[string]map[string]float64 // requestID -> (podName -> hit ratio)

	PodInflightMutex    sync.RWMutex
	podInflightRequests map[string]float64 // podName -> num inflight requests

	requestInflightMutex sync.RWMutex
	requestInflight      map[string]map[string]float64 // requestID -> (podName -> num inflight requests

	requestToPodMutex sync.RWMutex
	requestToPod      map[string]string // requestID -> podName
)

func init() {
	requestKVCacheHitRatio = make(map[string]float64)
	requestAllPodsKVCache = make(map[string]map[string]float64)
	requestInflight = make(map[string]map[string]float64)
	podInflightRequests = make(map[string]float64)
	requestToPod = make(map[string]string)
}

func CleanupRequestToPod(requestID string) {
	requestToPodMutex.Lock()
	defer requestToPodMutex.Unlock()

	delete(requestToPod, requestID)
}

// StoreKVCacheHitRatio stores the KV cache hit ratio for a request
func StoreKVCacheHitRatio(requestID string, podName string, ratio float64, allPodsRatios map[string]float64) {
	requestKVCacheMutex.Lock()
	defer requestKVCacheMutex.Unlock()
	requestKVCacheHitRatio[requestID] = ratio
	// klog.Infof("Saved KV cache hit ratio: %.2f%%", ratio*100)
	requestAllPodsKVCache[requestID] = allPodsRatios
}

// GetKVCacheHitRatio retrieves the KV cache hit ratio for a request
func GetKVCacheHitRatio(requestID string) float64 {
	requestKVCacheMutex.RLock()
	defer requestKVCacheMutex.RUnlock()

	if ratio, ok := requestKVCacheHitRatio[requestID]; ok {
		return ratio
	}
	return -1
}

// GetAllPodsKVCacheHitRatios retrieves KV cache hit ratios for all pods for a request
func GetAllPodsKVCacheHitRatios(requestID string) map[string]float64 {
	requestKVCacheMutex.RLock()
	defer requestKVCacheMutex.RUnlock()

	if ratios, ok := requestAllPodsKVCache[requestID]; ok {
		// Make a copy to avoid concurrency issues
		result := make(map[string]float64, len(ratios))
		for pod, ratio := range ratios {
			result[pod] = ratio
		}
		return result
	}
	return make(map[string]float64)
}

// CleanupKVCacheHitRatio removes the KV cache hit ratio for a request
func CleanupKVCacheHitRatio(requestID string) {
	requestKVCacheMutex.Lock()
	defer requestKVCacheMutex.Unlock()

	delete(requestKVCacheHitRatio, requestID)
	delete(requestAllPodsKVCache, requestID)
}

// /////////////////////////////////////////////////////////

// StoreRequestToPod stores the mapping of requestID to podName
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
	klog.Infof("Decremented inflight requests for pod %s: %d", podName, podInflightRequests[podName])
	if podInflightRequests[podName] <= 0 {
		klog.Errorf("podInflightRequests[%s] is negative!", podName)
	}
}

// GetNumInflightRequestsForPod retrieves the number of inflight requests for a specific pod
func GetNumInflightRequestsForPod(podName string) float64 {
	PodInflightMutex.RLock()
	defer PodInflightMutex.RUnlock()
	if _, ok := podInflightRequests[podName]; !ok {
		return 0
	}
	return podInflightRequests[podName]
}

// /////////////////////////////////////////////////////
// Store number of inflight requests for all pods
// Read from the podInflightRequests map and store the values in the requestInflight map
func StoreInflightRequestsForTheRequest(requestID string) {
	requestInflightMutex.Lock()
	defer requestInflightMutex.Unlock()
	if _, exists := requestInflight[requestID]; exists {
		klog.Errorf("requestID already exists in requestInflight: %s", requestID)
		return
	}
	requestInflight[requestID] = make(map[string]float64)
	PodInflightMutex.RLock()
	defer PodInflightMutex.RUnlock()
	for podName, numinflightrequests := range podInflightRequests {
		requestInflight[requestID][podName] = numinflightrequests
	}
}

func GetInflightRequestsForAllPods(requestID string) map[string]float64 {
	requestInflightMutex.RLock()
	defer requestInflightMutex.RUnlock()
	if _, ok := requestInflight[requestID]; !ok {
		klog.Errorf("requestID not found in requestInflight: %s. Return empty slice", requestID)
		return make(map[string]float64)
	}
	return requestInflight[requestID]
}

func CleanupInflightRequests(requestID string) {
	requestInflightMutex.Lock()
	defer requestInflightMutex.Unlock()
	delete(requestInflight, requestID)
}
