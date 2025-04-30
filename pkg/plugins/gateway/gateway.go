/*
Copyright 2024 The Aibrix Team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package gateway

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"

	// "github.com/vllm-project/aibrix/pkg/utils/kvcache"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/vllm-project/aibrix/pkg/cache"
	routing "github.com/vllm-project/aibrix/pkg/plugins/gateway/algorithms"
	"github.com/vllm-project/aibrix/pkg/plugins/gateway/ratelimiter"
	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
)

type RequestTiming struct {
	startTime      time.Time // When the request began processing
	firstTokenTime time.Time // When the first token was received
	lastTokenTime  time.Time // When the last token was received
	tokenCount     int64     // Count of tokens received so far

}

type PodMetric struct {
	Timestamp time.Time
	TTFT      int64 // Time to first token
	TPOT      int64 // Time per output token (for single token)
	TokenNum  int64 // Which token in the sequence (1 = first token)
}

type PodMetricsTracker struct {
	mutex      sync.RWMutex
	podMetrics map[string][]PodMetric // Map of pod IP -> slice of metrics
	windowSize time.Duration          // How long to keep metrics
}

// cleanupOldMetrics removes metrics that are older than the window size
func (t *PodMetricsTracker) cleanupOldMetrics(podIP string, now time.Time) {
	cutoff := now.Add(-t.windowSize)
	metrics := t.podMetrics[podIP]

	var newMetrics []PodMetric
	for _, m := range metrics {
		if m.Timestamp.After(cutoff) {
			newMetrics = append(newMetrics, m)
		}
	}

	t.podMetrics[podIP] = newMetrics
}

type Server struct {
	redisClient         *redis.Client
	ratelimiter         ratelimiter.RateLimiter
	client              kubernetes.Interface
	requestCountTracker map[string]int
	cache               cache.Cache

	requestTimings      sync.Map // Map to track request timing information: requestID -> *RequestTiming
	requestBuffers      sync.Map // Thread-safe map to track buffers per request
	streamingUsageCache sync.Map // Map to store usage information from streaming responses
	statusCode          sync.Map // Map to track status codes per request: requestID -> statusCode
	selectedPodIP       sync.Map // Map to track target pod per request: requestID -> podIP

	routingContexts sync.Map // Map to store routing contexts for each request: requestID -> *types.RoutingContext

	// New fields for metrics tracking
	metricsTracker   *PodMetricsTracker // Track timing metrics for pods
	metricsEnabled   atomic.Bool        // Flag to enable/disable metrics collection
	metricsLogTicker *time.Ticker       // Ticker for periodic metrics logging
}

func (t *PodMetricsTracker) InitPodKey(podIP string) {
	// Trim port from podIP if present
	if colonIndex := len(podIP) - 1; podIP[colonIndex] == ':' {
		podIP = podIP[:colonIndex]
	}
	t.mutex.Lock()
	defer t.mutex.Unlock()
	if _, exists := t.podMetrics[podIP]; !exists {
		podmetric := PodMetric{
			Timestamp: time.Now(),
			TTFT:      0,
			TPOT:      0,
			TokenNum:  0,
		}
		t.podMetrics[podIP] = []PodMetric{podmetric}
		klog.Infof("Initialized pod metrics for pod %s", podIP)
	}
}

func (t *PodMetricsTracker) AddMetric(podIP string, metric PodMetric) {
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
	cutoff := now.Add(-t.windowSize)
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
		windowSize: windowSize,
	}
}

func NewServer(redisClient *redis.Client, client kubernetes.Interface) *Server {
	c, err := cache.Get()
	if err != nil {
		panic(err)
	}
	r := ratelimiter.NewRedisAccountRateLimiter("aibrix", redisClient, 1*time.Minute)

	// Initialize the routers
	routing.Init()

	server := &Server{
		redisClient:         redisClient,
		ratelimiter:         r,
		client:              client,
		requestCountTracker: map[string]int{},
		cache:               c,
		metricsTracker:      NewPodMetricsTracker(1 * time.Second),
	}
	// Enable metrics collection by default
	server.metricsEnabled.Store(true)

	// Start metrics cleanup goroutine
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				if server.metricsEnabled.Load() {
					klog.V(4).Info("Running periodic metrics cleanup")
					server.metricsTracker.CleanupAllMetrics()
				}
			}
		}
	}()

	// Start periodic metrics logging
	server.metricsLogTicker = time.NewTicker(10 * time.Second)
	return server
}

// // CleanupAllMetrics removes all metrics outside the window for all pods
// func (t *PodMetricsTracker) CleanupAllMetrics() {
// 	t.mutex.Lock()
// 	defer t.mutex.Unlock()

// 	now := time.Now()
// 	cutoff := now.Add(-t.windowSize)

// 	for podIP, metrics := range t.podMetrics {
// 		var newMetrics []PodMetric
// 		for _, m := range metrics {
// 			if m.Timestamp.After(cutoff) {
// 				newMetrics = append(newMetrics, m)
// 			}
// 		}

// 		if len(newMetrics) > 0 {
// 			t.podMetrics[podIP] = newMetrics
// 		} else {
// 			// If no metrics are left in the window, remove the pod entry
// 			delete(t.podMetrics, podIP)
// 		}
// 	}
// }

func (t *PodMetricsTracker) CleanupAllMetrics() {
	t.mutex.Lock()
	defer t.mutex.Unlock()

	now := time.Now()
	cutoff := now.Add(-t.windowSize)

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
			t.podMetrics[podIP] = []PodMetric{{
				Timestamp: now,
				TTFT:      0,
				TPOT:      0,
				TokenNum:  0,
			}}
		}
	}
}

func (s *Server) Process(srv extProcPb.ExternalProcessor_ProcessServer) error {
	var user utils.User
	var rpm, traceTerm int64
	var respErrorCode int
	var model string
	var routingAlgorithm types.RoutingAlgorithm
	var routerCtx *types.RoutingContext
	var stream, isRespError bool
	ctx := srv.Context()
	requestID := uuid.New().String()
	completed := false

	klog.InfoS("processing request", "requestID", requestID)

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		req, err := srv.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return status.Errorf(codes.Unknown, "cannot receive stream request: %v", err)
		}

		resp := &extProcPb.ProcessingResponse{}
		switch v := req.Request.(type) {

		case *extProcPb.ProcessingRequest_RequestHeaders:
			resp, user, rpm, routingAlgorithm = s.HandleRequestHeaders(ctx, requestID, req)

		case *extProcPb.ProcessingRequest_RequestBody:
			klog.InfoS("Before HandleRequestBody", "requestID", requestID)
			resp, model, routerCtx, stream, traceTerm = s.HandleRequestBody(ctx, requestID, req, user, routingAlgorithm)
			if routerCtx != nil {
				if routerCtx.Err() != nil {
					klog.ErrorS(routerCtx.Err(), "Routing context already canceled, using original",
						"requestID", requestID)
				}
				ctx = routerCtx
			}

		case *extProcPb.ProcessingRequest_ResponseHeaders:
			resp, isRespError, respErrorCode = s.HandleResponseHeaders(ctx, requestID, model, req)
			if isRespError {
				klog.Errorf("Response headers processing error %d, requestID: %s, selectedPod: %s, model: %s", respErrorCode, requestID, routerCtx.TargetAddress(), model)
			}
		case *extProcPb.ProcessingRequest_ResponseBody:
			respBody := req.Request.(*extProcPb.ProcessingRequest_ResponseBody)
			if isRespError {
				klog.ErrorS(errors.New("request end"), string(respBody.ResponseBody.GetBody()), "requestID", requestID)
				generateErrorResponse(envoyTypePb.StatusCode(respErrorCode), nil, string(respBody.ResponseBody.GetBody()))
			} else {
				resp, completed = s.HandleResponseBody(ctx, requestID, req, user, rpm, model, stream, traceTerm, completed)
			}
		default:
			klog.Infof("Unknown Request type %+v\n", v)
		}

		if err := srv.Send(resp); err != nil && len(model) > 0 {
			s.cache.DoneRequestCount(routerCtx, requestID, model, traceTerm)
			if routerCtx != nil {
				routerCtx.Delete()
			}
			klog.ErrorS(err, "requestID", requestID)
		}
	}
}

func (s *Server) selectTargetPod(ctx *types.RoutingContext, pods types.PodList) (string, error) {
	defer func() {
		klog.InfoS("Exiting selectTargetPod", "requestID", ctx.RequestID, "ctxDone", ctx.Err() != nil)
	}()
	router, err := routing.Select(ctx.Algorithm)(ctx)
	if err != nil {
		klog.ErrorS(err, "Router selection failed", "requestID", ctx.RequestID)
		return "", err
	}

	if pods.Len() == 0 {
		return "", fmt.Errorf("no pods to forward request")
	}
	readyPods := utils.FilterRoutablePods(pods.All())
	if len(readyPods) == 0 {
		return "", fmt.Errorf("no ready pods available for fallback")
	}
	if len(readyPods) == 1 {
		for _, pod := range readyPods {
			ctx.SetTargetPod(pod)
			return ctx.TargetAddress(), nil
		}
	}
	for _, pod := range readyPods {
		s.metricsTracker.InitPodKey(pod.Status.PodIP)
	}
	return router.Route(ctx, &utils.PodArray{Pods: readyPods})
}

func NewHealthCheckServer() *HealthServer {
	return &HealthServer{}
}

type HealthServer struct{}

func (s *HealthServer) Check(ctx context.Context, in *healthPb.HealthCheckRequest) (*healthPb.HealthCheckResponse, error) {
	return &healthPb.HealthCheckResponse{Status: healthPb.HealthCheckResponse_SERVING}, nil
}

func (s *HealthServer) Watch(in *healthPb.HealthCheckRequest, srv healthPb.Health_WatchServer) error {
	return status.Error(codes.Unimplemented, "watch is not implemented")
}
