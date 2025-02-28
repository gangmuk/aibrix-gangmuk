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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
	"crypto/sha256"
	"encoding/hex"

	"github.com/google/uuid"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
	"github.com/redis/go-redis/v9"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/vllm-project/aibrix/pkg/cache"
	routing "github.com/vllm-project/aibrix/pkg/plugins/gateway/algorithms"
	ratelimiter "github.com/vllm-project/aibrix/pkg/plugins/gateway/ratelimiter"
	"github.com/vllm-project/aibrix/pkg/utils"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
)

const (
	HeaderErrorInvalidRouting = "x-error-invalid-routing-strategy"

	// General Error Headers
	HeaderErrorUser                  = "x-error-user"
	HeaderErrorRouting               = "x-error-routing"
	HeaderErrorRequestBodyProcessing = "x-error-request-body-processing"
	HeaderErrorResponseUnmarshal     = "x-error-response-unmarshal"
	HeaderErrorResponseUnknown       = "x-error-response-unknown"

	// Model & Deployment Headers
	HeaderErrorNoModelInRequest = "x-error-no-model-in-request"
	HeaderErrorNoModelBackends  = "x-error-no-model-backends"

	// Streaming Headers
	HeaderErrorStreaming                 = "x-error-streaming"
	HeaderErrorNoStreamOptions           = "x-error-no-stream-options"
	HeaderErrorStreamOptionsIncludeUsage = "x-error-no-stream-options-include-usage"

	// Request & Target Headers
	HeaderWentIntoReqHeaders = "x-went-into-req-headers"
	HeaderTargetPod          = "target-pod"
	HeaderRoutingStrategy    = "routing-strategy"

	// RPM & TPM Update Errors
	HeaderUpdateTPM        = "x-update-tpm"
	HeaderUpdateRPM        = "x-update-rpm"
	HeaderErrorRPMExceeded = "x-error-rpm-exceeded"
	HeaderErrorTPMExceeded = "x-error-tpm-exceeded"
	HeaderErrorIncrRPM     = "x-error-incr-rpm"
	HeaderErrorIncrTPM     = "x-error-incr-tpm"

	HeaderTTFT = "x-timing-ttft-ms"      // Time to first token in milliseconds
	HeaderTPOT = "x-timing-tpot-ms"      // Time per output token in milliseconds

	// Rate Limiting defaults
	DefaultRPM           = 100
	DefaultTPMMultiplier = 1000

	// Envs
	EnvRoutingAlgorithm = "ROUTING_ALGORITHM"

	// Router names
	RouterRandom             = "random"
	RouterLeastRequest       = "least-request"
	RouterThroughput         = "throughput"
	RouterPrefixCache        = "prefix-cache"
	RouterPrefixCacheAndLoad = "prefix-cache-and-load"
	RouterLeastKvCache       = "least-kv-cache"
	RouterLeastBusyTime      = "least-busy-time"
	RouterLeastLatency       = "least-latency"
	
)

var (
	routingStrategies = []string{"random", "least-request", "throughput", "prefix-cache", "prefix-cache-and-load", "least-kv-cache", "least-busy-time", "least-latency"}

	ErrorUnknownResponse = errors.New("unknown response")

	requestBuffers sync.Map // Thread-safe map to track buffers per request
	streamingUsageCache sync.Map
	requestMessages sync.Map
	routingHistory sync.Map
)

// routerConstructors maps router names to their initialization functions.
var routerConstructors = map[string]func() (routing.Router, error){
	RouterRandom:             func() (routing.Router, error) { return routing.NewRandomRouter() },
	RouterLeastRequest:       func() (routing.Router, error) { return routing.NewLeastRequestRouter() },
	RouterThroughput:         func() (routing.Router, error) { return routing.NewThroughputRouter() },
	RouterPrefixCache:        func() (routing.Router, error) { return routing.NewPrefixCacheRouter() },
	RouterPrefixCacheAndLoad: func() (routing.Router, error) { return routing.NewPrefixCacheAndLoadRouter() },
	RouterLeastKvCache:       func() (routing.Router, error) { return routing.NewLeastKvCacheRouter() },
	RouterLeastBusyTime:      func() (routing.Router, error) { return routing.NewLeastBusyTimeRouter() },
	RouterLeastLatency:       func() (routing.Router, error) { return routing.NewLeastExpectedLatencyRouter() },
}

type Server struct {
	routers             map[string]routing.Router
	redisClient         *redis.Client
	ratelimiter         ratelimiter.RateLimiter
	client              kubernetes.Interface
	requestCountTracker map[string]int
	cache               *cache.Cache
	requestTimings  sync.Map // Map to track request timing information: requestID -> *RequestTiming
	streamingChunksMap  sync.Map // requestID -> int
}

type RequestTiming struct {
	startTime         time.Time  // When the request began processing
	firstTokenTime    time.Time  // When the first token was received
	tokenCount        int        // Count of tokens received so far
}

func NewServer(redisClient *redis.Client, client kubernetes.Interface) *Server {
	c, err := cache.GetCache()
	if err != nil {
		panic(err)
	}
	r := ratelimiter.NewRedisAccountRateLimiter("aibrix", redisClient, 1*time.Minute)
	routers := initializeRouters()

	return &Server{
		routers:             routers,
		redisClient:         redisClient,
		ratelimiter:         r,
		client:              client,
		requestCountTracker: map[string]int{},
		cache:               c,
	}
}

// initializeRouters initialize different routing algorithms, consider to initialize the router in lazy way
func initializeRouters() map[string]routing.Router {
	routers := make(map[string]routing.Router)
	for name, constructor := range routerConstructors {
		router, err := constructor()
		if err != nil {
			klog.Warningf("failed to initialize router %s: %v", name, err)
			continue
		}
		routers[name] = router
	}
	return routers
}

type HealthServer struct{}

func calculateMessageHash(message string) string {
	hasher := sha256.New()
	hasher.Write([]byte(message))
	return hex.EncodeToString(hasher.Sum(nil))[:8]
}

func (s *HealthServer) Check(ctx context.Context, in *healthPb.HealthCheckRequest) (*healthPb.HealthCheckResponse, error) {
	return &healthPb.HealthCheckResponse{Status: healthPb.HealthCheckResponse_SERVING}, nil
}

func (s *HealthServer) Watch(in *healthPb.HealthCheckRequest, srv healthPb.Health_WatchServer) error {
	return status.Error(codes.Unimplemented, "watch is not implemented")
}

func (s *Server) Process(srv extProcPb.ExternalProcessor_ProcessServer) error {
	var user utils.User
	var rpm, traceTerm int64
	var respErrorCode int
	var model, routingStrategy, targetPodIP string
	var stream, isRespError bool
	ctx := srv.Context()
	requestID := uuid.New().String()
	completed := false

	klog.InfoS("Processing request", "requestID", requestID)
	// start_time := time.Now()
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
			// HandleRequestBody_start_time := time.Now()
			resp, user, rpm, routingStrategy = s.HandleRequestHeaders(ctx, requestID, req)
			// klog.Infof("HandleRequestHeaders latency: %.4f", time.Since(HandleRequestBody_start_time).Seconds())

		case *extProcPb.ProcessingRequest_RequestBody:
			// HandleRequestBody_start_time := time.Now()
			resp, model, targetPodIP, stream, traceTerm = s.HandleRequestBody(ctx, requestID, req, user, routingStrategy)
			// klog.Infof("HandleRequestBody latency: %.4f", time.Since(HandleRequestBody_start_time).Seconds())

		case *extProcPb.ProcessingRequest_ResponseHeaders:
			// HandleResponseHeaders_start_time := time.Now()
			resp, isRespError, respErrorCode = s.HandleResponseHeaders(ctx, requestID, req, targetPodIP)
			// klog.Infof("HandleResponseHeaders latency: %.4f", time.Since(HandleResponseHeaders_start_time).Seconds())

		case *extProcPb.ProcessingRequest_ResponseBody:
			respBody := req.Request.(*extProcPb.ProcessingRequest_ResponseBody)
			if isRespError {
				klog.ErrorS(errors.New("request end"), string(respBody.ResponseBody.GetBody()), "requestID", requestID)
				generateErrorResponse(envoyTypePb.StatusCode(respErrorCode), nil, string(respBody.ResponseBody.GetBody()))
				} else {
					resp, completed = s.HandleResponseBody(ctx, requestID, req, user, rpm, model, targetPodIP, stream, traceTerm, completed)
					// klog.Infof("E2E latency: %.4f", time.Since(start_time).Seconds())
					// klog.Info("==========================================================================")
			}
		default:
			klog.Infof("Unknown Request type %+v\n", v)
		}

		if err := srv.Send(resp); err != nil {
			klog.Infof("send error %v", err)
		}
	}
}

func (s *Server) HandleRequestHeaders(ctx context.Context, requestID string, req *extProcPb.ProcessingRequest) (*extProcPb.ProcessingResponse, utils.User, int64, string) {
	klog.Info("latency ==========================================================================")
	klog.InfoS("-- In RequestHeaders processing ...", "requestID", requestID)
	var username string
	var user utils.User
	var rpm int64
	var err error
	var errRes *extProcPb.ProcessingResponse

	h := req.Request.(*extProcPb.ProcessingRequest_RequestHeaders)
	for _, n := range h.RequestHeaders.Headers.Headers {
		if strings.ToLower(n.Key) == "user" {
			username = string(n.RawValue)
		}
	}

	routingStrategy, routingStrategyEnabled := GetRoutingStrategy(h.RequestHeaders.Headers.Headers)
	if routingStrategyEnabled && !validateRoutingStrategy(routingStrategy) {
		klog.ErrorS(nil, "incorrect routing strategy", "routing-strategy", routingStrategy)
		return generateErrorResponse(
			envoyTypePb.StatusCode_BadRequest,
			[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
				Key: HeaderErrorInvalidRouting, RawValue: []byte(routingStrategy),
			}}}, "incorrect routing strategy"), utils.User{}, rpm, routingStrategy
	}

	if username != "" {
		user, err = utils.GetUser(utils.User{Name: username}, s.redisClient)
		if err != nil {
			klog.ErrorS(err, "unable to process user info", "requestID", requestID, "username", username)
			return generateErrorResponse(
				envoyTypePb.StatusCode_InternalServerError,
				[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
					Key: HeaderErrorUser, RawValue: []byte("true"),
				}}},
				err.Error()), utils.User{}, rpm, routingStrategy
		}

		rpm, errRes, err = s.checkLimits(ctx, user)
		if errRes != nil {
			klog.ErrorS(err, "error on checking limits", "requestID", requestID, "username", username)
			return errRes, utils.User{}, rpm, routingStrategy
		}
	}

	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_RequestHeaders{
			RequestHeaders: &extProcPb.HeadersResponse{
				Response: &extProcPb.CommonResponse{
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders: []*configPb.HeaderValueOption{
							{
								Header: &configPb.HeaderValue{
									Key:      HeaderWentIntoReqHeaders,
									RawValue: []byte("true"),
								},
							},
						},
					},
					ClearRouteCache: true,
				},
			},
		},
	}, user, rpm, routingStrategy
}

func (s *Server) HandleRequestBody(ctx context.Context, requestID string, req *extProcPb.ProcessingRequest, user utils.User, routingStrategy string) (*extProcPb.ProcessingResponse, string, string, bool, int64) {
	klog.InfoS("-- In RequestBody processing ...", "requestID", requestID)
	s.requestTimings.Store(requestID, &RequestTiming{
		startTime: time.Now(),
		tokenCount: 0,
	})
	var model, targetPodIP string
	var ok, stream bool
	var term int64 // Identify the trace window

	var jsonMap map[string]interface{}

	body := req.Request.(*extProcPb.ProcessingRequest_RequestBody)
	if err := json.Unmarshal(body.RequestBody.GetBody(), &jsonMap); err != nil {
		klog.ErrorS(err, "error to unmarshal response", "requestID", requestID, "requestBody", string(body.RequestBody.GetBody()))
		return generateErrorResponse(envoyTypePb.StatusCode_InternalServerError,
			[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
				Key: HeaderErrorRequestBodyProcessing, RawValue: []byte("true")}}},
			"error processing request body"), model, targetPodIP, stream, term
	}

	if model, ok = jsonMap["model"].(string); !ok || model == "" {
		klog.ErrorS(nil, "model error in request", "requestID", requestID, "jsonMap", jsonMap)
		return generateErrorResponse(envoyTypePb.StatusCode_InternalServerError,
			[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
				Key: HeaderErrorNoModelInRequest, RawValue: []byte(model)}}},
			"no model in request body"), model, targetPodIP, stream, term
	}

	// early reject the request if model doesn't exist.
	if !s.cache.CheckModelExists(model) {
		klog.ErrorS(nil, "model doesn't exist in cache, probably wrong model name", "requestID", requestID, "model", model)
		return generateErrorResponse(envoyTypePb.StatusCode_BadRequest,
			[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
				Key: HeaderErrorNoModelBackends, RawValue: []byte(model)}}},
			fmt.Sprintf("model %s does not exist", model)), model, targetPodIP, stream, term
	}

	// early reject if no pods are ready to accept request for a model
	pods, err := s.cache.GetPodsForModel(model)
	if len(pods) == 0 || len(utils.FilterReadyPods(pods)) == 0 || err != nil {
		klog.ErrorS(err, "no ready pod available", "requestID", requestID, "model", model)
		return generateErrorResponse(envoyTypePb.StatusCode_ServiceUnavailable,
			[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
				Key: HeaderErrorNoModelBackends, RawValue: []byte("true")}}},
			fmt.Sprintf("error on getting pods for model %s", model)), model, targetPodIP, stream, term
	}

	stream, ok = jsonMap["stream"].(bool)
	if stream && ok {
		streamOptions, ok := jsonMap["stream_options"].(map[string]interface{})
		if !ok {
			klog.ErrorS(nil, "no stream option available", "requestID", requestID, "jsonMap", jsonMap)
			return generateErrorResponse(envoyTypePb.StatusCode_InternalServerError,
				[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
					Key: HeaderErrorNoStreamOptions, RawValue: []byte("stream options not set")}}},
				"no stream option available"), model, targetPodIP, stream, term
		}
		includeUsage, ok := streamOptions["include_usage"].(bool)
		if !includeUsage || !ok {
			klog.ErrorS(nil, "no stream with usage option available", "requestID", requestID, "jsonMap", jsonMap)
			return generateErrorResponse(envoyTypePb.StatusCode_InternalServerError,
				[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
					Key: HeaderErrorStreamOptionsIncludeUsage, RawValue: []byte("include usage for stream options not set")}}},
				"no stream with usage option available"), model, targetPodIP, stream, term
		}
	}

	headers := []*configPb.HeaderValueOption{}
	klog.InfoS("request start", "routing-strategy", routingStrategy, "requestID", requestID, "model", model)
	if routingStrategy == "" {
		headers = append(headers, &configPb.HeaderValueOption{
			Header: &configPb.HeaderValue{
				Key:      "model",
				RawValue: []byte(model),
			},
		})
		klog.InfoS("routing-strategy is empty!")
	} else {
		message, extErr := getRequestMessage(jsonMap)
		if extErr != nil {
			return extErr, model, targetPodIP, stream, term
		}
		requestMessages.Store(requestID, message)
		start_time := time.Now()
		targetPodIP, err = s.selectTargetPod(ctx, routingStrategy, pods, model, message)
		routingHistory.Store(requestID, targetPodIP)
		klog.Infof("(Routing logic overhead) selectTargetPod latency: %.4f, target pod: %s", time.Since(start_time).Seconds(), targetPodIP)
		if targetPodIP == "" || err != nil {
			klog.ErrorS(err, "failed to select target pod", "requestID", requestID, "routingStrategy", routingStrategy, "model", model)
			return generateErrorResponse(
				envoyTypePb.StatusCode_ServiceUnavailable,
				[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
					Key: HeaderErrorRouting, RawValue: []byte("true")}}},
				"error on selecting target pod"), model, targetPodIP, stream, term
		}

		headers = append(headers,
			&configPb.HeaderValueOption{
				Header: &configPb.HeaderValue{
					Key:      HeaderRoutingStrategy,
					RawValue: []byte(routingStrategy),
				},
			},
			&configPb.HeaderValueOption{
				Header: &configPb.HeaderValue{
					Key:      HeaderTargetPod,
					RawValue: []byte(targetPodIP),
				},
			})
	}

	term = s.cache.AddRequestCount(requestID, model)
	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_RequestBody{
			RequestBody: &extProcPb.BodyResponse{
				Response: &extProcPb.CommonResponse{
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders: headers,
					},
				},
			},
		},
	}, model, targetPodIP, stream, term
}

func (s *Server) HandleResponseHeaders(ctx context.Context, requestID string, req *extProcPb.ProcessingRequest, targetPodIP string) (*extProcPb.ProcessingResponse, bool, int) {
	klog.InfoS("-- In ResponseHeaders processing ...", "requestID", requestID)
	HandleResponseHeaders_start_time := time.Now()
	b := req.Request.(*extProcPb.ProcessingRequest_ResponseHeaders)

	headers := []*configPb.HeaderValueOption{{
		Header: &configPb.HeaderValue{
			Key:      HeaderWentIntoReqHeaders,
			RawValue: []byte("true"),
		},
	}}
	if targetPodIP != "" {
		headers = append(headers, &configPb.HeaderValueOption{
			Header: &configPb.HeaderValue{
				Key:      HeaderTargetPod,
				RawValue: []byte(targetPodIP),
			},
		})
	}

	var isProcessingError bool
	var processingErrorCode int
	for _, headerValue := range b.ResponseHeaders.Headers.Headers {
		if headerValue.Key == ":status" {
			code, _ := strconv.Atoi(string(headerValue.RawValue))
			if code != 200 {
				isProcessingError = true
				processingErrorCode = code
			}
		}
		headers = append(headers, &configPb.HeaderValueOption{
			Header: &configPb.HeaderValue{
				Key:      headerValue.Key,
				RawValue: headerValue.RawValue,
			},
		})
	}
	klog.Infof("HandleResponseHeaders latency: %.4f", time.Since(HandleResponseHeaders_start_time).Seconds())
	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ResponseHeaders{
			ResponseHeaders: &extProcPb.HeadersResponse{
				Response: &extProcPb.CommonResponse{
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders: headers,
					},
					ClearRouteCache: true,
				},
			},
		},
	}, isProcessingError, processingErrorCode
}

func (s *Server) calculateTimingMetrics(timing *RequestTiming, currentTime time.Time, requestID string, stream bool, numInputTokens int64, numOutputTokens int64, numTotalTokens int64) []*configPb.HeaderValueOption {
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
    }
    
    // Log timing metrics with correct token counts
    end_to_end := time.Since(timing.startTime).Milliseconds()
    agg_latency := ttftMs
    if numOutputTokens > 0 {
        agg_latency += tpotMs * numOutputTokens
    }
	messageHash := ""
    if messageInterface, exists := requestMessages.Load(requestID); exists {
        message := messageInterface.(string)
        messageHash = calculateMessageHash(message)
        // Clean up the stored message
        requestMessages.Delete(requestID)
    }
	selectedPodIP, _ := routingHistory.Load(requestID)
	
	klog.Infof("** latency metrics, hash(request), %s, selectedpod, %s, ttft, %d, tpot, %d, e2e, %d, numInputTokens, %d, numOutputTokens, %d, numTotalTokens, %d", messageHash, selectedPodIP, ttftMs, tpotMs, end_to_end, numInputTokens, numOutputTokens, numTotalTokens)
    return headers
}

func (s *Server) HandleResponseBody(ctx context.Context, requestID string, req *extProcPb.ProcessingRequest, user utils.User, rpm int64, model string, targetPodIP string, stream bool, traceTerm int64, hasCompleted bool) (*extProcPb.ProcessingResponse, bool) {
    b := req.Request.(*extProcPb.ProcessingRequest_ResponseBody)
    var res openai.ChatCompletion
    var usage openai.CompletionUsage
    var promptTokens, completionTokens int64
    var headers []*configPb.HeaderValueOption
    complete := hasCompleted

    // Get timing object for this request
    timingObj, exists := s.requestTimings.Load(requestID)
    var timing *RequestTiming
    if exists {
        timing = timingObj.(*RequestTiming)
    }

    // Process response body
    currentTime := time.Now()
    if timing != nil {
        // Extract usage for streaming responses
        if stream {
            usage = s.handleStreamingResponse(requestID, b.ResponseBody.GetBody())
            
            // // Logging for verification
            // klog.InfoS("Streaming usage extraction", 
            //     "requestID", requestID,
            //     "promptTokens", usage.PromptTokens,
            //     "completionTokens", usage.CompletionTokens,
            //     "totalTokens", usage.TotalTokens)
        }

        if stream {
            t := &http.Response{
                Body: io.NopCloser(bytes.NewReader(b.ResponseBody.GetBody())),
            }
            streaming := ssestream.NewStream[openai.ChatCompletionChunk](ssestream.NewDecoder(t), nil)
            
            for streaming.Next() {
                evt := streaming.Current()
                
                // Check if this is the first token for TTFT
                if timing.firstTokenTime.IsZero() && len(evt.Choices) > 0 && evt.Choices[0].Delta.Content != "" {
                    timing.firstTokenTime = currentTime
                    klog.InfoS("First token received", "requestID", requestID, 
                        "ttft_ms", currentTime.Sub(timing.startTime).Milliseconds())
                }
                
                // Count tokens for TPOT calculation
                if len(evt.Choices) > 0 && evt.Choices[0].Delta.Content != "" {
                    timing.tokenCount++
                }
            }
            
            // Check for errors in streaming
            if err := streaming.Err(); err != nil {
                klog.ErrorS(err, "error processing streaming response", "requestID", requestID)
                complete = true
                return generateErrorResponse(
                    envoyTypePb.StatusCode_InternalServerError,
                    []*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
                        Key: HeaderErrorStreaming, RawValue: []byte("true"),
                    }}},
                    err.Error()), complete
            }
        } else {
            // For non-streaming, record first token time on final response
            if b.ResponseBody.EndOfStream && timing.firstTokenTime.IsZero() {
                timing.firstTokenTime = currentTime
            }
            
            // Process the full response for non-streaming requests
            if b.ResponseBody.EndOfStream {
                // Get the full body
                buf, _ := requestBuffers.LoadOrStore(requestID, &bytes.Buffer{})
                buffer := buf.(*bytes.Buffer)
                buffer.Write(b.ResponseBody.Body)
                finalBody := buffer.Bytes()
                requestBuffers.Delete(requestID)
                
                // Parse the response
                if err := json.Unmarshal(finalBody, &res); err != nil {
                    klog.ErrorS(err, "error unmarshaling response", "requestID", requestID)
                    complete = true
                    return generateErrorResponse(
                        envoyTypePb.StatusCode_InternalServerError,
                        []*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
                            Key: HeaderErrorResponseUnmarshal, RawValue: []byte("true"),
                        }}},
                        err.Error()), complete
                } else if len(res.Model) == 0 {
                    msg := ErrorUnknownResponse.Error()
                    responseBodyContent := string(finalBody)
                    if len(responseBodyContent) != 0 {
                        msg = responseBodyContent
                    }
                    klog.ErrorS(nil, "unexpected response", "requestID", requestID)
                    complete = true
                    return generateErrorResponse(
                        envoyTypePb.StatusCode_InternalServerError,
                        []*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
                            Key: HeaderErrorResponseUnknown, RawValue: []byte("true"),
                        }}},
                        msg), complete
                }
                
                // Get usage data
                usage = res.Usage
            } else {
                // Just append data for partial responses
                buf, _ := requestBuffers.LoadOrStore(requestID, &bytes.Buffer{})
                buffer := buf.(*bytes.Buffer)
                buffer.Write(b.ResponseBody.Body)
                
                // Return early for partial responses
                return &extProcPb.ProcessingResponse{
                    Response: &extProcPb.ProcessingResponse_ResponseBody{
                        ResponseBody: &extProcPb.BodyResponse{
                            Response: &extProcPb.CommonResponse{},
                        },
                    },
                }, complete
            }
        }
        
        // Only calculate and add timing metrics at the end of the stream
        if b.ResponseBody.EndOfStream {
            // Calculate timing metrics and add headers
            timingHeaders := s.calculateTimingMetrics(timing, currentTime, requestID, stream, usage.PromptTokens, usage.CompletionTokens, usage.TotalTokens)
            headers = append(headers, timingHeaders...)
            
            // Clean up timing data when we're done
            s.requestTimings.Delete(requestID)
        }
    }

    // Set completion flag if we have token usage data
    if usage.TotalTokens > 0 {
        complete = true
        promptTokens = usage.PromptTokens
        completionTokens = usage.CompletionTokens
        
        // Count token per user if needed
        if user.Name != "" {
            tpm, err := s.ratelimiter.Incr(ctx, fmt.Sprintf("%v_TPM_CURRENT", user), usage.TotalTokens)
            if err != nil {
                return generateErrorResponse(
                    envoyTypePb.StatusCode_InternalServerError,
                    []*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
                        Key: HeaderErrorIncrTPM, RawValue: []byte("true"),
                    }}},
                    err.Error()), complete
            }

            headers = append(headers,
                &configPb.HeaderValueOption{
                    Header: &configPb.HeaderValue{
                        Key:      HeaderUpdateRPM,
                        RawValue: []byte(fmt.Sprintf("%d", rpm)),
                    },
                },
                &configPb.HeaderValueOption{
                    Header: &configPb.HeaderValue{
                        Key:      HeaderUpdateTPM,
                        RawValue: []byte(fmt.Sprintf("%d", tpm)),
                    },
                },
            )
        }

        // Add target pod information
        if targetPodIP != "" {
            headers = append(headers,
                &configPb.HeaderValueOption{
                    Header: &configPb.HeaderValue{
                        Key:      HeaderTargetPod,
                        RawValue: []byte(targetPodIP),
                    },
                },
            )
        }

        // Only log completion on the final chunk
        if b.ResponseBody.EndOfStream {
            klog.InfoS("Request completed", 
                "requestID", requestID,
                "targetPod", targetPodIP,
                "promptTokens", promptTokens,
                "completionTokens", completionTokens,
                "totalTokens", promptTokens + completionTokens)
        }
    }
    
    // Call DoneRequestTrace when the request is complete
    if !hasCompleted && complete && b.ResponseBody.EndOfStream {
        s.cache.DoneRequestTrace(requestID, model, promptTokens, completionTokens, traceTerm)
    }

    // Only log completion on the final chunk
    if b.ResponseBody.EndOfStream {
        klog.InfoS("HandleResponseBody completed", 
            "requestID", requestID,
            "promptTokens", promptTokens, 
            "completionTokens", completionTokens,
            "stream", stream)
		klog.Info("latency ==========================================================================")
    }

    return &extProcPb.ProcessingResponse{
        Response: &extProcPb.ProcessingResponse_ResponseBody{
            ResponseBody: &extProcPb.BodyResponse{
                Response: &extProcPb.CommonResponse{
                    HeaderMutation: &extProcPb.HeaderMutation{
                        SetHeaders: headers,
                    },
                },
            },
        },
    }, complete
}


func (s *Server) handleStreamingResponse(requestID string, responseBody []byte) openai.CompletionUsage {
    // Split the response into lines
    lines := strings.Split(string(responseBody), "\n")
    
    // Retrieve existing usage for this request
    existingUsageRaw, _ := streamingUsageCache.LoadOrStore(requestID, openai.CompletionUsage{})
    existingUsage := existingUsageRaw.(openai.CompletionUsage)

    for i := len(lines) - 1; i >= 0; i-- {
        line := strings.TrimSpace(lines[i])
        
        // Skip empty lines or non-data lines
        if !strings.HasPrefix(line, "data:") || line == "data: [DONE]" {
            continue
        }

        // Remove "data: " prefix
        cleanLine := strings.TrimPrefix(line, "data: ")
        
        // Try to parse the JSON
        var chunk map[string]interface{}
        if err := json.Unmarshal([]byte(cleanLine), &chunk); err != nil {
            continue
        }

        // Check for usage
        if usageMap, ok := chunk["usage"].(map[string]interface{}); ok {
            promptTokens := int64(usageMap["prompt_tokens"].(float64))
            completionTokens := int64(usageMap["completion_tokens"].(float64))
            totalTokens := int64(usageMap["total_tokens"].(float64))

            // Only update if we find meaningful usage
            if promptTokens > 0 || completionTokens > 0 || totalTokens > 0 {
                newUsage := openai.CompletionUsage{
                    PromptTokens:     promptTokens,
                    CompletionTokens: completionTokens,
                    TotalTokens:      totalTokens,
                }
                
                // Store the new usage
                streamingUsageCache.Store(requestID, newUsage)
                
                return newUsage
            }
        }
    }
    
    // Return existing usage if no new usage found
    return existingUsage
}

func (s *Server) checkLimits(ctx context.Context, user utils.User) (int64, *extProcPb.ProcessingResponse, error) {
	if user.Rpm == 0 {
		user.Rpm = int64(DefaultRPM)
	}
	if user.Tpm == 0 {
		user.Tpm = user.Rpm * int64(DefaultTPMMultiplier)
	}

	code, err := s.checkRPM(ctx, user.Name, user.Rpm)
	if err != nil {
		return 0, generateErrorResponse(
			code,
			[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
				Key: HeaderErrorRPMExceeded, RawValue: []byte("true"),
			}}},
			err.Error()), err
	}

	rpm, code, err := s.incrRPM(ctx, user.Name)
	if err != nil {
		return 0, generateErrorResponse(
			code,
			[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
				Key: HeaderErrorIncrRPM, RawValue: []byte("true"),
			}}},
			err.Error()), err
	}

	code, err = s.checkTPM(ctx, user.Name, user.Tpm)
	if err != nil {
		return 0, generateErrorResponse(
			code,
			[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
				Key: HeaderErrorTPMExceeded, RawValue: []byte("true"),
			}}},
			err.Error()), err
	}

	return rpm, nil, nil
}

func (s *Server) checkRPM(ctx context.Context, username string, rpmLimit int64) (envoyTypePb.StatusCode, error) {
	rpmCurrent, err := s.ratelimiter.Get(ctx, fmt.Sprintf("%v_RPM_CURRENT", username))
	if err != nil {
		return envoyTypePb.StatusCode_InternalServerError, fmt.Errorf("fail to get RPM for user: %v", username)
	}

	if rpmCurrent >= rpmLimit {
		return envoyTypePb.StatusCode_TooManyRequests, fmt.Errorf("user: %v has exceeded RPM: %v", username, rpmLimit)
	}

	return envoyTypePb.StatusCode_OK, nil
}

func (s *Server) incrRPM(ctx context.Context, username string) (int64, envoyTypePb.StatusCode, error) {
	rpm, err := s.ratelimiter.Incr(ctx, fmt.Sprintf("%v_RPM_CURRENT", username), 1)
	if err != nil {
		return rpm, envoyTypePb.StatusCode_InternalServerError, fmt.Errorf("fail to increment RPM for user: %v", username)
	}

	return rpm, envoyTypePb.StatusCode_OK, nil
}

func (s *Server) checkTPM(ctx context.Context, username string, tpmLimit int64) (envoyTypePb.StatusCode, error) {
	tpmCurrent, err := s.ratelimiter.Get(ctx, fmt.Sprintf("%v_TPM_CURRENT", username))
	if err != nil {
		return envoyTypePb.StatusCode_InternalServerError, fmt.Errorf("fail to get TPM for user: %v", username)
	}

	if tpmCurrent >= tpmLimit {
		return envoyTypePb.StatusCode_TooManyRequests, fmt.Errorf("user: %v has exceeded TPM: %v", username, tpmLimit)
	}

	return envoyTypePb.StatusCode_OK, nil
}

func (s *Server) selectTargetPod(ctx context.Context, routingStrategy string, pods map[string]*v1.Pod, model, message string) (string, error) {
	var route routing.Router
	switch routingStrategy {
	case "least-request":
		route = s.routers[routingStrategy]
	case "throughput":
		route = s.routers[routingStrategy]
	case "prefix-cache":
		route = s.routers[routingStrategy]
	case "prefix-cache-and-load":
		route = s.routers[routingStrategy]
	case "least-kv-cache":
		route = s.routers[routingStrategy]
	case "least-busy-time":
		route = s.routers[routingStrategy]
	case "least-latency":
		route = s.routers[routingStrategy]
	default:
		route = s.routers["random"]
	}
	return route.Route(ctx, pods, model, message)
}

func validateRoutingStrategy(routingStrategy string) bool {
	routingStrategy = strings.TrimSpace(routingStrategy)
	return slices.Contains(routingStrategies, routingStrategy)
}

func generateErrorResponse(statusCode envoyTypePb.StatusCode, headers []*configPb.HeaderValueOption, body string) *extProcPb.ProcessingResponse {
	// Set the Content-Type header to application/json
	headers = append(headers, &configPb.HeaderValueOption{
		Header: &configPb.HeaderValue{
			Key:   "Content-Type",
			Value: "application/json",
		},
	})

	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &extProcPb.ImmediateResponse{
				Status: &envoyTypePb.HttpStatus{
					Code: statusCode,
				},
				Headers: &extProcPb.HeaderMutation{
					SetHeaders: headers,
				},
				Body: generateErrorMessage(body, int(statusCode)),
			},
		},
	}
}

func getRequestMessage(jsonMap map[string]interface{}) (string, *extProcPb.ProcessingResponse) {
	messages, ok := jsonMap["messages"]
	if !ok {
		return "", generateErrorResponse(envoyTypePb.StatusCode_InternalServerError,
			[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{Key: HeaderErrorRequestBodyProcessing, RawValue: []byte("true")}}},
			"no messages in the request body")
	}
	messagesJSON, err := json.Marshal(messages)
	if err != nil {
		return "", generateErrorResponse(envoyTypePb.StatusCode_InternalServerError,
			[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{Key: HeaderErrorRequestBodyProcessing, RawValue: []byte("true")}}},
			"unable to marshal messages from request body")
	}
	return string(messagesJSON), nil
}

// GetRoutingStrategy retrieves the routing strategy from the headers or environment variable
// It returns the routing strategy value and whether custom routing strategy is enabled.
func GetRoutingStrategy(headers []*configPb.HeaderValue) (string, bool) {
	var routingStrategy string
	routingStrategyEnabled := false

	// Check headers for routing strategy
	for _, header := range headers {
		if strings.ToLower(header.Key) == HeaderRoutingStrategy {
			routingStrategy = string(header.RawValue)
			routingStrategyEnabled = true
			break // Prioritize header value over environment variable
		}
	}

	// If header not set, check environment variable
	if !routingStrategyEnabled {
		if value, exists := utils.CheckEnvExists(EnvRoutingAlgorithm); exists {
			routingStrategy = value
			routingStrategyEnabled = true
		}
	}

	return routingStrategy, routingStrategyEnabled
}

// generateErrorMessage constructs a JSON error message using fmt.Sprintf
func generateErrorMessage(message string, code int) string {
	errorStruct := map[string]interface{}{
		"error": map[string]interface{}{
			"message": message,
			"code":    code,
		},
	}
	jsonData, _ := json.Marshal(errorStruct)
	return string(jsonData)
}
