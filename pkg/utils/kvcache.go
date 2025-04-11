// In pkg/utils/kvcache.go
package utils

import (
	"sync"
)

var (
	// Global map to store KV cache hit ratios per request
	requestKVCacheMutex    sync.RWMutex
	requestKVCacheHitRatio map[string]float64            // requestID -> hit ratio
	requestAllPodsKVCache  map[string]map[string]float64 // requestID -> (podName -> hit ratio)
)

func init() {
	requestKVCacheHitRatio = make(map[string]float64)
	requestAllPodsKVCache = make(map[string]map[string]float64)
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
