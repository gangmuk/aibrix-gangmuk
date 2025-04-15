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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"

	// "github.com/vllm-project/aibrix/pkg/utils/kvcache"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
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
	tokenCount     int       // Count of tokens received so far
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
}

func (s *Server) calculateTimingMetrics(timing *RequestTiming, currentTime time.Time, requestID string, routingCtx *types.RoutingContext, stream bool, numInputTokens int64, numOutputTokens int64, numTotalTokens int64) []*configPb.HeaderValueOption {
	// Calculate TTFT (Time To First Token)
	ttftMs := int64(0)
	if !timing.firstTokenTime.IsZero() {
		ttftMs = timing.firstTokenTime.Sub(timing.startTime).Milliseconds()
	}

	// Calculate TPOT (Time Per Output Token)
	tpotMs := int64(0)
	if !timing.firstTokenTime.IsZero() {
		totalGenerationTimeMs := currentTime.Sub(timing.firstTokenTime).Milliseconds()

		// Use the correct token count
		effectiveTokenCount := int64(0)
		if stream && timing.tokenCount > 1 {
			// For streaming, use our counted tokens
			effectiveTokenCount = int64(timing.tokenCount - 1) // Exclude first token
		} else if numOutputTokens > 1 {
			// Use the actual output tokens from usage
			effectiveTokenCount = numOutputTokens - 1
		}

		if effectiveTokenCount > 0 {
			tpotMs = totalGenerationTimeMs / effectiveTokenCount
		}
	}

	// Add timing headers
	end_to_end_latency_in_ms := time.Since(timing.startTime).Milliseconds()
	headers := []*configPb.HeaderValueOption{
		{
			Header: &configPb.HeaderValue{
				Key:      HeaderTTFT,
				RawValue: []byte(fmt.Sprintf("%d", ttftMs)),
			},
		},
		{
			Header: &configPb.HeaderValue{
				Key:      HeaderTPOT,
				RawValue: []byte(fmt.Sprintf("%d", tpotMs)),
			},
		},
		{
			Header: &configPb.HeaderValue{
				Key:      HeaderE2ELatency,
				RawValue: []byte(fmt.Sprintf("%d", end_to_end_latency_in_ms)),
			},
		},
	}

	/////////////////////////////////////
	// Get KV cache hit ratio from our global store
	kvCacheHitRatio := utils.GetKVCacheHitRatio(requestID)
	headers = append(headers, &configPb.HeaderValueOption{
		Header: &configPb.HeaderValue{
			Key:      HeaderKVCacheHitRatio,
			RawValue: []byte(fmt.Sprintf("%.4f", kvCacheHitRatio)),
		},
	})

	// Get KV cache hit ratios for all pods
	allPodsRatios := utils.GetAllPodsKVCacheHitRatios(requestID)
	allPodsJSON, err := json.Marshal(allPodsRatios)
	if err == nil {
		headers = append(headers, &configPb.HeaderValueOption{
			Header: &configPb.HeaderValue{
				Key:      HeaderKVCacheHitRatioAllPods,
				RawValue: allPodsJSON,
			},
		})
	}
	utils.CleanupKVCacheHitRatio(requestID)

	/////////////////////////////////////
	// Get inflight requests for all pods
	numInflightRequestsAllPods := utils.GetInflightRequestsForAllPods(requestID)
	numInflightRequestsAllPodsJSON, err := json.Marshal(numInflightRequestsAllPods)
	if err == nil {
		headers = append(headers, &configPb.HeaderValueOption{
			Header: &configPb.HeaderValue{
				Key:      HeaderNumInflightRequestsAllPods,
				RawValue: numInflightRequestsAllPodsJSON,
			},
		})
	}
	utils.DecrementNumInflightForPod(requestID)
	utils.CleanupInflightRequests(requestID)

	// #################################################
	vllmGPUKVCacheUsage, err := utils.GetvLLMGPUKVCacheUsageForTheRequestForAllPods(requestID)
	if err == nil {
		vllmGPUKVCacheUsageJSON, err := json.Marshal(vllmGPUKVCacheUsage)
		if err == nil {
			headers = append(headers, &configPb.HeaderValueOption{
				Header: &configPb.HeaderValue{
					Key:      HeadervLLMGPUKVCacheUsage,
					RawValue: vllmGPUKVCacheUsageJSON,
				},
			})
			klog.Infof("vllmGPUKVCacheUsageJSON: %s", string(vllmGPUKVCacheUsageJSON))
		} else {
			klog.Infof("Error marshalling vllmGPUKVCacheUsageJSON: %s", err)
		}
	}
	utils.CleanupvLLMGPUKVCacheUsage(requestID)

	vllmCPUKVCacheUsage, err := utils.GetvLLMCPUKVCacheUsageForTheRequestForAllPods(requestID)
	if err == nil {
		vllmCPUKVCacheUsageJSON, err := json.Marshal(vllmCPUKVCacheUsage)
		if err == nil {
			headers = append(headers, &configPb.HeaderValueOption{
				Header: &configPb.HeaderValue{
					Key:      HeadervLLMCPUKVCacheUsage,
					RawValue: vllmCPUKVCacheUsageJSON,
				},
			})
			klog.Infof("vllmCPUKVCacheUsageJSON: %s", string(vllmCPUKVCacheUsageJSON))
		} else {
			klog.Infof("Error marshalling vllmCPUKVCacheUsageJSON: %s", err)
		}
	}
	utils.CleanupvLLMCPUKVCacheUsage(requestID)

	vllmNumRequestsRunning, err := utils.GetvLLMNumRequestsRunningForTheRequestForAllPods(requestID)
	if err == nil {
		vllmNumRequestsRunningJSON, err := json.Marshal(vllmNumRequestsRunning)
		if err == nil {
			headers = append(headers, &configPb.HeaderValueOption{
				Header: &configPb.HeaderValue{
					Key:      HeadervLLMNumRunningRequests,
					RawValue: vllmNumRequestsRunningJSON,
				},
			})
			klog.Infof("vllmNumRequestsRunningJSON: %s", string(vllmNumRequestsRunningJSON))
		} else {
			klog.Infof("Error marshalling vllmNumRequestsRunningJSON: %s", err)
		}
	}
	utils.CleanupvLLMNumRequestsRunning(requestID)

	vllmNumRequestWaiting, err := utils.GetvLLMNumRequestsWaitingForTheRequestForAllPods(requestID)
	if err == nil {
		vllmNumRequestWaitingJSON, err := json.Marshal(vllmNumRequestWaiting)
		if err == nil {
			headers = append(headers, &configPb.HeaderValueOption{
				Header: &configPb.HeaderValue{
					Key:      HeadervLLMNumwWaitingRequests,
					RawValue: vllmNumRequestWaitingJSON,
				},
			})
			klog.Infof("vllmNumRequestWaitingJSON: %s", string(vllmNumRequestWaitingJSON))
		} else {
			klog.Infof("Error marshalling vllmNumRequestWaitingJSON: %s", err)
		}
	}
	utils.CleanupvLLMNumRequestsWaiting(requestID)

	// #################################################

	/////////////////////////////////////
	// Get target pod IP directly from routing context
	selectedPodIP := "unknown"
	if routingCtx != nil {
		selectedPodIP = routingCtx.TargetAddress()
	}

	// klog.Infof("** latency metrics, requestID: %s, selectedpod: %s, ttft: %d, tpot: %d, e2e: %d, numInputTokens: %d, numOutputTokens: %d, numTotalTokens: %d, kvCacheHitRatio: %.4f", requestID, selectedPodIP, ttftMs, tpotMs, end_to_end_latency_in_ms, numInputTokens, numOutputTokens, numTotalTokens, kvCacheHitRatio)
	klog.Infof("** latency metrics, requestID: %s, selectedpod: %s, ttft: %d, tpot: %d, e2e: %d, numInputTokens: %d, numOutputTokens: %d, numTotalTokens: %d, kvCacheHitRatio: %.4f, numInflightRequestsAllPods: %v", requestID, selectedPodIP, ttftMs, tpotMs, end_to_end_latency_in_ms, numInputTokens, numOutputTokens, numTotalTokens, kvCacheHitRatio, numInflightRequestsAllPods)
	return headers
}

func NewServer(redisClient *redis.Client, client kubernetes.Interface) *Server {
	c, err := cache.Get()
	if err != nil {
		panic(err)
	}
	r := ratelimiter.NewRedisAccountRateLimiter("aibrix", redisClient, 1*time.Minute)

	// Initialize the routers
	routing.Init()

	return &Server{
		redisClient:         redisClient,
		ratelimiter:         r,
		client:              client,
		requestCountTracker: map[string]int{},
		cache:               c,
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
			resp, model, routerCtx, stream, traceTerm = s.HandleRequestBody(ctx, requestID, req, user, routingAlgorithm)
			if routerCtx != nil {
				ctx = routerCtx
			}

		case *extProcPb.ProcessingRequest_ResponseHeaders:
			resp, isRespError, respErrorCode = s.HandleResponseHeaders(ctx, requestID, model, req)

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
	router, err := routing.Select(ctx.Algorithm)(ctx)
	if err != nil {
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
