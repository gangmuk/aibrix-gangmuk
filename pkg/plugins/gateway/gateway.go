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
	"time"

	"github.com/redis/go-redis/v9"

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
	startTime            time.Time // When the request began processing
	firstTokenTime       time.Time // When the first token was received
	firstDecodeTokenTime time.Time // When the first decode token was received
	lastTokenTime        time.Time // When the last token was received
	totalTokenCount      int64
	prefillTokenCount    int64
	decodeTokenCount     int64 // Count of tokens in the decode phase
	IsPrefill            bool
}

type Server struct {
	redisClient         *redis.Client
	ratelimiter         ratelimiter.RateLimiter
	client              kubernetes.Interface
	cache               cache.Cache
	requestBuffers      sync.Map // Thread-safe map to track buffers per request
	streamingUsageCache sync.Map // Map to store usage information from streaming responses
	statusCode          sync.Map // Map to track status codes per request: requestID -> statusCode
	selectedPodIP       sync.Map // Map to track target pod per request: requestID -> podIP
	routingContexts     sync.Map // Map to store routing contexts for each request: requestID -> *types.RoutingContext
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
		redisClient: redisClient,
		ratelimiter: r,
		client:      client,
		cache:       c,
	}
	return server
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
	// requestID := uuid.New().String()
	// klog.InfoS("processing request", "requestID", requestID)
	completed := false
	requestID := ""
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
			requestID = getRequestID(v.RequestHeaders.Headers.Headers)
			klog.V(5).Infof("Before HandleRequestHeaders, requestID: %s, ctx.Err(): %v", requestID, ctx.Err())
			resp, user, rpm, routingAlgorithm = s.HandleRequestHeaders(ctx, requestID, req)

		case *extProcPb.ProcessingRequest_RequestBody:
			klog.Infof("Before HandleRequestBody, requestID: %s, ctx.Err(): %v", requestID, ctx.Err())
			resp, model, routerCtx, stream, traceTerm = s.HandleRequestBody(ctx, requestID, req, user, routingAlgorithm)
			if routerCtx != nil {
				if routerCtx.Err() != nil {
					klog.ErrorS(routerCtx.Err(), "Routing context already canceled, using original",
						"requestID", requestID)
				}
				ctx = routerCtx
			}

		case *extProcPb.ProcessingRequest_ResponseHeaders:
			klog.V(5).Infof("Before HandleResponseHeaders, requestID: %s, ctx.Err(): %v", requestID, ctx.Err())
			resp, isRespError, respErrorCode = s.HandleResponseHeaders(ctx, requestID, model, req)
			if isRespError {
				klog.Errorf("Response headers processing error %d, requestID: %s, selectedPod: %s, model: %s", respErrorCode, requestID, routerCtx.TargetAddress(), model)
			}
		case *extProcPb.ProcessingRequest_ResponseBody:
			respBody := req.Request.(*extProcPb.ProcessingRequest_ResponseBody)
			if isRespError {
				klog.ErrorS(errors.New("request end"), string(respBody.ResponseBody.GetBody()), "requestID", requestID)
				klog.Errorf("Response body processing error %d, requestID: %s, selectedPod: %s, model: %s", respErrorCode, requestID, routerCtx.TargetAddress(), model)
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
			klog.ErrorS(err, "Error, requestID", requestID)
		}
	}
}

func (s *Server) selectTargetPod(ctx *types.RoutingContext, pods types.PodList) (string, error) {
	klog.Infof("selectTargetPod starts. context state, requestID: %s, ctx.Err(): %v", ctx.RequestID, ctx.Err())
	defer func() {
		if ctx.Err() != nil {
			klog.ErrorS(ctx.Err(), "Exiting selectTargetPod, Context error", "requestID", ctx.RequestID)
		} else {
			klog.V(5).InfoS("Exiting selectTargetPod, Context is not done successfully", "requestID", ctx.RequestID)
		}
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
		utils.MetricsTracker.InitPodKey(pod.Status.PodIP)
	}
	klog.Infof("selectTargetPod, done with InitPodKey. context state, requestID: %s, ctx.Err(): %v", ctx.RequestID, ctx.Err())

	ts := time.Now()
	selectedPodAddress, err := router.Route(ctx, &utils.PodArray{Pods: readyPods})

	klog.Infof("selectTargetPod. Routing took %s, selectedPodAddress: %s", time.Since(ts), selectedPodAddress)
	if err != nil {
		klog.ErrorS(err, "Routing failed", "requestID", ctx.RequestID)
		return "", err
	}
	return selectedPodAddress, nil
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
