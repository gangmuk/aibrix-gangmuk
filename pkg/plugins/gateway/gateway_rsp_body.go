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
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
	"k8s.io/klog/v2"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
)

func (s *Server) handleStreamingResponse(requestID string, responseBody []byte) openai.CompletionUsage {
	// Split the response into lines
	lines := strings.Split(string(responseBody), "\n")

	// Retrieve existing usage for this request
	existingUsageRaw, _ := s.streamingUsageCache.LoadOrStore(requestID, openai.CompletionUsage{})
	existingUsage := existingUsageRaw.(openai.CompletionUsage)

	// Get timing object for this request
	timingObj, exists := s.requestTimings.Load(requestID)
	if !exists {
		return existingUsage
	}

	timing := timingObj.(*RequestTiming)
	currentTime := time.Now()

	// Get routing context to find the pod IP
	routerCtxObj, exists := s.routingContexts.Load(requestID)
	if !exists {
		return existingUsage
	}

	routerCtx := routerCtxObj.(*types.RoutingContext)
	selectedPodIP := routerCtx.TargetAddress()

	// If the pod IP contains a port, remove it
	if strings.Contains(selectedPodIP, ":") {
		selectedPodIP = strings.Split(selectedPodIP, ":")[0]
	}

	// Process the SSE stream
	t := &http.Response{
		Body: io.NopCloser(bytes.NewReader(responseBody)),
	}
	streaming := ssestream.NewStream[openai.ChatCompletionChunk](ssestream.NewDecoder(t), nil)

	for streaming.Next() {
		evt := streaming.Current()
		if len(evt.Choices) > 0 && evt.Choices[0].Delta.Content != "" {
			// If this is the first token, record TTFT
			if timing.firstTokenTime.IsZero() {
				timing.firstTokenTime = currentTime
				timing.lastTokenTime = currentTime
				ttftMs := currentTime.Sub(timing.startTime).Milliseconds()

				// Record TTFT in our metrics tracker if enabled
				if s.IsMetricsEnabled() && ttftMs > 0 {
					s.metricsTracker.AddMetric(selectedPodIP, PodMetric{
						Timestamp: currentTime,
						TTFT:      ttftMs,
						TPOT:      0, // No TPOT for first token
						TokenNum:  1,
					})
				}

				klog.InfoS("First token received", "requestID", requestID, "ttft_ms", ttftMs)
			} else {
				// For subsequent tokens, calculate and record TPOT
				tokenNum := timing.tokenCount + 1
				timeSincePrevToken := currentTime.Sub(timing.lastTokenTime).Milliseconds()

				// Record TPOT in our metrics tracker if enabled
				if s.IsMetricsEnabled() && timeSincePrevToken > 0 {
					s.metricsTracker.AddMetric(selectedPodIP, PodMetric{
						Timestamp: currentTime,
						TTFT:      0, // No TTFT for subsequent tokens
						TPOT:      timeSincePrevToken,
						TokenNum:  tokenNum,
					})
				}
			}

			// Update token count and last token time
			timing.lastTokenTime = currentTime
			timing.tokenCount++
		}
	}

	// Check for errors in processing the stream
	if err := streaming.Err(); err != nil {
		klog.ErrorS(err, "error processing streaming response", "requestID", requestID)
	}

	// Rest of the existing function for extracting usage...
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
				s.streamingUsageCache.Store(requestID, newUsage)

				return newUsage
			}
		}
	}

	// Return existing usage if no new usage found
	return existingUsage
}

func (s *Server) HandleResponseBody(ctx context.Context, requestID string, req *extProcPb.ProcessingRequest, user utils.User, rpm int64, model string, stream bool, traceTerm int64, hasCompleted bool) (*extProcPb.ProcessingResponse, bool) {
	b := req.Request.(*extProcPb.ProcessingRequest_ResponseBody)
	// statusCode := 0
	// selectedPodIP, ok := s.selectedPodIP.Load(requestID)
	// if !ok {
	// 	selectedPodIP = "unknown"
	// }
	// if storedCode, ok := s.statusCode.Load(requestID); ok { // stored in gateway_rsp_headers.go
	// 	statusCode = storedCode.(int)
	// }
	// klog.Errorf("Response status code: %d, requestID: %s, selectedPodIP: %s", statusCode, requestID, selectedPodIP)

	var res openai.ChatCompletion
	var usage openai.CompletionUsage
	var promptTokens, completionTokens int64
	var headers []*configPb.HeaderValueOption
	complete := hasCompleted
	routerCtx, _ := ctx.(*types.RoutingContext)

	////////////////////////////////////////////////////////////////////////////
	// Get timing object for this request
	timingObj, exists := s.requestTimings.Load(requestID)
	var timing *RequestTiming
	if exists {
		timing = timingObj.(*RequestTiming)
	}
	currentTime := time.Now()
	if timing != nil {
		if stream {
			usage = s.handleStreamingResponse(requestID, b.ResponseBody.GetBody())
			t := &http.Response{
				Body: io.NopCloser(bytes.NewReader(b.ResponseBody.GetBody())),
			}
			streaming := ssestream.NewStream[openai.ChatCompletionChunk](ssestream.NewDecoder(t), nil)
			for streaming.Next() {
				evt := streaming.Current()
				if timing.firstTokenTime.IsZero() && len(evt.Choices) > 0 && evt.Choices[0].Delta.Content != "" {
					timing.firstTokenTime = currentTime
					klog.InfoS("First token received", "requestID", requestID, "ttft_ms", currentTime.Sub(timing.startTime).Milliseconds())
				}
			}
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
			buf, _ := s.requestBuffers.LoadOrStore(requestID, &bytes.Buffer{})
			buffer := buf.(*bytes.Buffer)
			buffer.Write(b.ResponseBody.Body)
			if timing.firstTokenTime.IsZero() && b.ResponseBody.EndOfStream {
				timing.firstTokenTime = currentTime
			}
			if !b.ResponseBody.EndOfStream {
				return &extProcPb.ProcessingResponse{
					Response: &extProcPb.ProcessingResponse_ResponseBody{
						ResponseBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{},
						},
					},
				}, complete
			}
			finalBody := buffer.Bytes()
			s.requestBuffers.Delete(requestID)
			if err := json.Unmarshal(finalBody, &res); err != nil {
				klog.ErrorS(err, "error to unmarshal response", "requestID", requestID)
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
			usage = res.Usage
		}
		if b.ResponseBody.EndOfStream {
			if routerCtx.Algorithm == "prefix-cache-and-load" {
				timingHeaders := s.calculateTimingMetrics(timing, currentTime, requestID, routerCtx, stream, usage.PromptTokens, usage.CompletionTokens, usage.TotalTokens)
				headers = append(headers, timingHeaders...)

				// metricsData := s.CollectRequestMetrics(timing, currentTime, requestID, routerCtx, stream, usage.PromptTokens, usage.CompletionTokens, usage.TotalTokens)
				// bodyBytes := b.ResponseBody.GetBody()
				// // klog.Infof("Original body length for request %s: %d bytes", requestID, len(bodyBytes))
				// newBodyStr, err := s.injectMetricsIntoSSE(bodyBytes, metricsData, requestID)
				// newBodyBytes := []byte(newBodyStr)
				// if err != nil {
				// 	klog.Errorf("Failed to inject metrics into SSE for request %s: %v", requestID, err)
				// } else {
				// 	b.ResponseBody.Body = newBodyBytes
				// 	// klog.Infof("Modified body length for request %s: %d bytes, newBodyStr: %s", requestID, len(newBodyBytes), newBodyStr)
				// }
			}
			s.requestTimings.Delete(requestID)
			s.routingContexts.Delete(requestID)
		}
	}
	if usage.TotalTokens > 0 {
		complete = true
		promptTokens = usage.PromptTokens
		completionTokens = usage.CompletionTokens
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
		if routerCtx != nil {
			headers = append(headers,
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      HeaderTargetPod,
						RawValue: []byte(routerCtx.TargetAddress()),
					},
				},
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      HeaderRequestID,
						RawValue: []byte(requestID),
					},
				},
			)
		}
	}
	////////////////////////////////////////////////////////////////////////////

	defer func() {
		if !hasCompleted && complete {
			s.cache.DoneRequestTrace(routerCtx, requestID, model, promptTokens, completionTokens, traceTerm)
			if routerCtx != nil {
				routerCtx.Delete()
			}
		}
	}()

	if stream {
		t := &http.Response{
			Body: io.NopCloser(bytes.NewReader(b.ResponseBody.GetBody())),
		}
		streaming := ssestream.NewStream[openai.ChatCompletionChunk](ssestream.NewDecoder(t), nil)
		defer func() {
			_ = streaming.Close()
		}()
		for streaming.Next() {
			evt := streaming.Current()
			if len(evt.Choices) == 0 {
				// Do not overwrite model, res can be empty.
				usage = evt.Usage
			}
		}
		if err := streaming.Err(); err != nil {
			klog.ErrorS(err, "error to unmarshal response", "requestID", requestID, "responseBody", string(b.ResponseBody.GetBody()))
			complete = true
			return generateErrorResponse(
				envoyTypePb.StatusCode_InternalServerError,
				[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
					Key: HeaderErrorStreaming, RawValue: []byte("true"),
				}}},
				err.Error()), complete
		}
	} else {
		buf, _ := requestBuffers.LoadOrStore(requestID, &bytes.Buffer{})
		buffer := buf.(*bytes.Buffer)
		buffer.Write(b.ResponseBody.Body)
		if !b.ResponseBody.EndOfStream {
			return &extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_ResponseBody{
					ResponseBody: &extProcPb.BodyResponse{
						Response: &extProcPb.CommonResponse{},
					},
				},
			}, complete
		}
		finalBody := buffer.Bytes()
		requestBuffers.Delete(requestID)
		if err := json.Unmarshal(finalBody, &res); err != nil {
			klog.ErrorS(err, "error to unmarshal response", "requestID", requestID, "responseBody", string(b.ResponseBody.GetBody()))
			complete = true
			return generateErrorResponse(
				envoyTypePb.StatusCode_InternalServerError,
				[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
					Key: HeaderErrorResponseUnmarshal, RawValue: []byte("true"),
				}}},
				err.Error()), complete
		} else if len(res.Model) == 0 {
			msg := ErrorUnknownResponse.Error()
			responseBodyContent := string(b.ResponseBody.GetBody())
			if len(responseBodyContent) != 0 {
				msg = responseBodyContent
			}
			klog.ErrorS(err, "unexpected response", "requestID", requestID, "responseBody", responseBodyContent)
			complete = true
			return generateErrorResponse(
				envoyTypePb.StatusCode_InternalServerError,
				[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
					Key: HeaderErrorResponseUnknown, RawValue: []byte("true"),
				}}},
				msg), complete
		}
		// Do not overwrite model, res can be empty.
		usage = res.Usage
	}

	var requestEnd string
	if usage.TotalTokens != 0 {
		complete = true
		// Update promptTokens and completeTokens
		promptTokens = usage.PromptTokens
		completionTokens = usage.CompletionTokens
		// Count token per user.
		if user.Name != "" {
			tpm, err := s.ratelimiter.Incr(ctx, fmt.Sprintf("%v_TPM_CURRENT", user), res.Usage.TotalTokens)
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
			requestEnd = fmt.Sprintf(requestEnd+"rpm: %s, tpm: %s, ", rpm, tpm)
		}
		if routerCtx != nil {
			targetPodName := routerCtx.TargetName()
			targetPodIP := routerCtx.TargetAddress()
			headers = append(headers,
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      HeaderTargetPod,
						RawValue: []byte(targetPodIP),
					},
				},
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      HeaderRequestID,
						RawValue: []byte(requestID),
					},
				},
				&configPb.HeaderValueOption{
					Header: &configPb.HeaderValue{
						Key:      HeaderTargetPodName,
						RawValue: []byte(targetPodName),
					},
				},
			)
			requestEnd = fmt.Sprintf(requestEnd+"targetPod: %s", targetPodIP)
		}

		klog.Infof("request end, requestID: %s - %s", requestID, requestEnd)
	} else if b.ResponseBody.EndOfStream {
		complete = true
	}

	// klog.Infof("SetHeaders: %s", headers)

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

// MetricsData holds all the performance metrics for a request
type MetricsData struct {
	TTFT               int64              `json:"ttft_ms"`
	TPOT               int64              `json:"tpot_ms"`
	E2ELatency         int64              `json:"e2e_latency_ms"`
	KVCacheHitRatio    float64            `json:"kv_cache_hit_ratio"`
	AllPodsRatios      map[string]float64 `json:"all_pods_ratios,omitempty"`
	InflightRequests   map[string]int     `json:"inflight_requests,omitempty"`
	GPUKVCacheUsage    map[string]float64 `json:"gpu_kv_cache_usage,omitempty"`
	CPUKVCacheUsage    map[string]float64 `json:"cpu_kv_cache_usage,omitempty"`
	NumRequestsRunning map[string]float64 `json:"num_requests_running,omitempty"`
	NumRequestsWaiting map[string]float64 `json:"num_requests_waiting,omitempty"`
	InputTokens        int64              `json:"input_tokens"`
	OutputTokens       int64              `json:"output_tokens"`
	TotalTokens        int64              `json:"total_tokens"`
	SelectedPod        string             `json:"selected_pod"`
}

// Helper function to add a JSON metric to headers
func addMetricToHeaders(headers []*configPb.HeaderValueOption, key string, data interface{}) ([]*configPb.HeaderValueOption, string) {
	jsonData, err := json.Marshal(data)
	jsonStr := "{}"
	if err == nil {
		jsonStr = string(jsonData)
		headers = append(headers, &configPb.HeaderValueOption{
			Header: &configPb.HeaderValue{
				Key:      key,
				RawValue: jsonData,
			},
		})
	}
	return headers, jsonStr
}

func (s *Server) calculateTimingMetrics(timing *RequestTiming, currentTime time.Time, requestID string, routingCtx *types.RoutingContext, stream bool, numInputTokens int64, numOutputTokens int64, numTotalTokens int64) []*configPb.HeaderValueOption {
	// Calculate basic timing metrics
	ttftMs := int64(0)
	if !timing.firstTokenTime.IsZero() {
		ttftMs = timing.firstTokenTime.Sub(timing.startTime).Milliseconds()
	}

	avgTpotMs := int64(0)
	totalGenerationTimeMs := int64(0)
	if !timing.firstTokenTime.IsZero() {
		totalGenerationTimeMs = currentTime.Sub(timing.firstTokenTime).Milliseconds()
		effectiveTokenCount := int64(0)
		if stream && timing.tokenCount > 1 {
			effectiveTokenCount = int64(timing.tokenCount - 1) // Exclude first token
		} else if numOutputTokens > 1 {
			effectiveTokenCount = numOutputTokens - 1
		}
		if effectiveTokenCount > 0 {
			avgTpotMs = totalGenerationTimeMs / effectiveTokenCount
			klog.Infof("avgTpotMs: %d, totalGenerationTimeMs: %d, effectiveTokenCount: %d", avgTpotMs, totalGenerationTimeMs, effectiveTokenCount)
		}
	}

	end_to_end_latency_in_ms := time.Since(timing.startTime).Milliseconds()

	// Initialize headers with basic metrics
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
				RawValue: []byte(fmt.Sprintf("%d", avgTpotMs)),
			},
		},
		{
			Header: &configPb.HeaderValue{
				Key:      HeaderE2ELatency,
				RawValue: []byte(fmt.Sprintf("%d", end_to_end_latency_in_ms)),
			},
		},
	}

	// Prepare for JSON strings to use in logging
	var jsonStrings = make(map[string]string)

	// 1. KV cache hit ratios
	allPodsKvCacheHitRatios := utils.GetAllPodsKVCacheHitRatios(requestID)
	headers, jsonStrings["allPodsKvCacheHitRatios"] = addMetricToHeaders(headers, HeaderKVCacheHitRatioAllPods, allPodsKvCacheHitRatios)
	utils.CleanupKVCacheHitRatio(requestID)

	// 2. Inflight requests
	numInflightRequestsAllPods := utils.GetInflightRequestsForAllPods(requestID)
	headers, jsonStrings["numInflightRequestsAllPods"] = addMetricToHeaders(headers, HeaderNumInflightRequestsAllPods, numInflightRequestsAllPods)
	utils.DecrementNumInflightForPod(requestID)
	utils.CleanupInflightRequests(requestID)

	// 3. GPU KV cache usage
	vllmGPUKVCacheUsage, err := utils.GetvLLMGPUKVCacheUsageForTheRequestForAllPods(requestID)
	if err == nil {
		headers, jsonStrings["vllmGPUKVCacheUsage"] = addMetricToHeaders(headers, HeadervLLMGPUKVCacheUsage, vllmGPUKVCacheUsage)
		utils.CleanupvLLMGPUKVCacheUsage(requestID)
	} else {
		jsonStrings["vllmGPUKVCacheUsage"] = "{}"
	}

	// 4. CPU KV cache usage
	vllmCPUKVCacheUsage, err := utils.GetvLLMCPUKVCacheUsageForTheRequestForAllPods(requestID)
	if err == nil {
		headers, jsonStrings["vllmCPUKVCacheUsage"] = addMetricToHeaders(headers, HeadervLLMCPUKVCacheUsage, vllmCPUKVCacheUsage)
		utils.CleanupvLLMCPUKVCacheUsage(requestID)
	} else {
		jsonStrings["vllmCPUKVCacheUsage"] = "{}"
	}

	// 5. Number of running requests
	vllmNumRequestsRunning, err := utils.GetvLLMNumRequestsRunningForTheRequestForAllPods(requestID)
	if err == nil {
		headers, jsonStrings["vllmNumRequestsRunning"] = addMetricToHeaders(headers, HeadervLLMNumRunningRequests, vllmNumRequestsRunning)
		utils.CleanupvLLMNumRequestsRunning(requestID)
	} else {
		jsonStrings["vllmNumRequestsRunning"] = "{}"
	}

	// 6. Number of waiting requests
	vllmNumRequestWaiting, err := utils.GetvLLMNumRequestsWaitingForTheRequestForAllPods(requestID)
	if err == nil {
		headers, jsonStrings["vllmNumRequestWaiting"] = addMetricToHeaders(headers, HeadervLLMNumwWaitingRequests, vllmNumRequestWaiting)
		utils.CleanupvLLMNumRequestsWaiting(requestID)
	} else {
		jsonStrings["vllmNumRequestWaiting"] = "{}"
	}

	// Get selected pod
	selectedPodIP := "unknown"
	if routingCtx != nil {
		selectedPodIP = routingCtx.TargetAddress()
		// Trim port number if present
		if strings.Contains(selectedPodIP, ":") {
			selectedPodIP = strings.Split(selectedPodIP, ":")[0]
		}
	}

	// 7. Pod detailed metrics
	log_window_end_time := time.Now()
	log_window_start_time := time.Now().Add(-s.metricsTracker.windowSize)
	podDetailedMetrics := s.metricsTracker.GetDetailedMetrics(log_window_start_time, numInputTokens, numOutputTokens, numTotalTokens)
	headers, jsonStrings["podMetricsLastSecond"] = addMetricToHeaders(headers, HeaderPodDetailedMetrics, podDetailedMetrics)

	klog.Infof("**@latency_metrics@requestID@%s@request_start_time@%d@request_end_time@%d@selectedpod@%s@ttft@%d@avg_tpot@%d@total_decode_time@%d@e2e@%d@numInputTokens@%d@numOutputTokens@%d@numTotalTokens@%d@allPodsKvCacheHitRatios@%s@numInflightRequestsAllPods@%s@vllmGPUKVCacheUsage@%s@vllmCPUKVCacheUsage@%s@vllmNumRequestsRunning@%s@vllmNumRequestsWaiting@%s@podMetricsLastSecond@%s@log_window_start_time@%d@log_window_end_time@%d",
		requestID,
		timing.startTime.UnixMicro(),
		currentTime.UnixMicro(),
		selectedPodIP,
		ttftMs,
		avgTpotMs,
		totalGenerationTimeMs,
		end_to_end_latency_in_ms,
		numInputTokens,
		numOutputTokens,
		numTotalTokens,
		jsonStrings["allPodsKvCacheHitRatios"],
		jsonStrings["numInflightRequestsAllPods"],
		jsonStrings["vllmGPUKVCacheUsage"],
		jsonStrings["vllmCPUKVCacheUsage"],
		jsonStrings["vllmNumRequestsRunning"],
		jsonStrings["vllmNumRequestWaiting"],
		jsonStrings["podMetricsLastSecond"],
		log_window_start_time.UnixMicro(),
		log_window_end_time.UnixMicro(),
	)

	return headers
}

////////////////////////////////////////////////////////////////////////////////////////////////

// PodDetailedMetrics provides detailed statistics for a pod's performance
type PodDetailedMetrics struct {
	// TTFT metrics
	AvgTTFT     float64 `json:"avg_ttft_ms"`
	MinTTFT     int64   `json:"min_ttft_ms"`
	MaxTTFT     int64   `json:"max_ttft_ms"`
	P50TTFT     int64   `json:"p50_ttft_ms"` // Median TTFT
	P90TTFT     int64   `json:"p90_ttft_ms"` // 90th percentile TTFT
	P95TTFT     int64   `json:"p95_ttft_ms"` // 95th percentile TTFT
	P99TTFT     int64   `json:"p99_ttft_ms"` // 99th percentile TTFT
	TTFTSamples int     `json:"ttft_samples"`

	// TPOT metrics
	AvgTPOT     float64 `json:"avg_tpot_ms"`
	MinTPOT     int64   `json:"min_tpot_ms"`
	MaxTPOT     int64   `json:"max_tpot_ms"`
	P50TPOT     int64   `json:"p50_tpot_ms"` // Median TPOT
	P90TPOT     int64   `json:"p90_tpot_ms"` // 90th percentile TPOT
	P95TPOT     int64   `json:"p95_tpot_ms"` // 95th percentile TPOT
	P99TPOT     int64   `json:"p99_tpot_ms"` // 99th percentile TPOT
	TPOTSamples int     `json:"tpot_samples"`

	// // Token position-based TPOT metrics (average TPOT for tokens 2-10)
	EarlyTokensTPOT float64 `json:"early_tokens_tpot_ms"` // Avg TPOT for tokens 2-10
	MidTokensTPOT   float64 `json:"mid_tokens_tpot_ms"`   // Avg TPOT for tokens 11-100
	LateTokensTPOT  float64 `json:"late_tokens_tpot_ms"`  // Avg TPOT for tokens 101+

	// Overall metrics
	TotalRequests int `json:"total_requests"`
	TotalTokens   int `json:"total_tokens"`
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

func (t *PodMetricsTracker) GetDetailedMetrics(log_window_start_time time.Time, numInputTokens int64, numOutputTokens int64, numTotalTokens int64) map[string]PodDetailedMetrics {
	t.mutex.RLock()
	defer t.mutex.RUnlock()
	result := make(map[string]PodDetailedMetrics)
	for podIP, metrics := range t.podMetrics {
		// podMetrics should have all pods in its entry
		// Init here to record all pods even if one does not have metrics after log_window_start_time
		detailedMetrics := PodDetailedMetrics{
			TotalRequests: -1,
			TotalTokens:   -1,
			TTFTSamples:   -1,
			TPOTSamples:   -1,
			AvgTTFT:       -1,
			MinTTFT:       -1,
			MaxTTFT:       -1,
			P50TTFT:       -1,
			P90TTFT:       -1,
			P95TTFT:       -1,
			P99TTFT:       -1,
			AvgTPOT:       -1,
			MinTPOT:       -1,
			MaxTPOT:       -1,
			P50TPOT:       -1,
			P90TPOT:       -1,
			P95TPOT:       -1,
			P99TPOT:       -1,
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
		var earlyTokensTPOT, midTokensTPOT, lateTokensTPOT []int64
		uniqueRequests := make(map[string]bool)
		totalTokens := 0
		for _, m := range validMetrics {
			if m.TTFT > 0 {
				ttftValues = append(ttftValues, m.TTFT)
				ttftSum += m.TTFT
				uniqueKey := fmt.Sprintf("%s-%d", podIP, m.Timestamp.UnixNano())
				uniqueRequests[uniqueKey] = true
			}
			if m.TPOT > 0 {
				tpotValues = append(tpotValues, m.TPOT)
				tpotSum += m.TPOT
				totalTokens++
				early_token_index := numOutputTokens / 3
				mid_token_index := (numOutputTokens / 3) * 2
				switch {
				case m.TokenNum <= early_token_index:
					earlyTokensTPOT = append(earlyTokensTPOT, m.TPOT)
				case m.TokenNum > early_token_index && m.TokenNum <= mid_token_index:
					midTokensTPOT = append(midTokensTPOT, m.TPOT)
				case m.TokenNum > mid_token_index:
					lateTokensTPOT = append(lateTokensTPOT, m.TPOT)
				}
			}
		}
		sort.Slice(ttftValues, func(i, j int) bool { return ttftValues[i] < ttftValues[j] })
		sort.Slice(tpotValues, func(i, j int) bool { return tpotValues[i] < tpotValues[j] })

		detailedMetrics.TotalRequests = len(uniqueRequests)
		detailedMetrics.TotalTokens = totalTokens
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

		if len(earlyTokensTPOT) > 0 {
			var sum int64
			for _, v := range earlyTokensTPOT {
				sum += v
			}
			detailedMetrics.EarlyTokensTPOT = float64(sum) / float64(len(earlyTokensTPOT))
		}
		if len(midTokensTPOT) > 0 {
			var sum int64
			for _, v := range midTokensTPOT {
				sum += v
			}
			detailedMetrics.MidTokensTPOT = float64(sum) / float64(len(midTokensTPOT))
		}
		if len(lateTokensTPOT) > 0 {
			var sum int64
			for _, v := range lateTokensTPOT {
				sum += v
			}
			detailedMetrics.LateTokensTPOT = float64(sum) / float64(len(lateTokensTPOT))
		}

		result[podIP] = detailedMetrics
		klog.Infof("podIP: %s, detailedMetrics", podIP)
	}
	return result
}

// Helper functions for metrics tracking
// IsMetricsEnabled returns whether metrics collection is enabled
func (s *Server) IsMetricsEnabled() bool {
	return s.metricsEnabled.Load()
}

// EnableMetrics enables metrics collection
func (s *Server) EnableMetrics() {
	s.metricsEnabled.Store(true)
	klog.Info("Metrics collection enabled")
}

// DisableMetrics disables metrics collection
func (s *Server) DisableMetrics() {
	s.metricsEnabled.Store(false)
	klog.Info("Metrics collection disabled")
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//// It was trying alternative way to encode metrics into SSE but it didn't work. seems like it is overwritten by envoy or something

// // Add this function to your server
// func (s *Server) injectMetricsIntoSSE(original []byte, metrics MetricsData, requestID string) (string, error) {
// 	// First check if we have any content at all
// 	if len(original) == 0 {
// 		klog.Warningf("Empty original body for request %s, creating metrics-only SSE", requestID)

// 		// Create a minimal SSE metrics event
// 		metricsJSON, err := json.Marshal(metrics)
// 		if err != nil {
// 			return "nil", fmt.Errorf("failed to marshal metrics: %w", err)
// 		}

// 		return fmt.Sprintf("event: metrics\ndata: %s\n\n", metricsJSON), nil
// 	}

// 	originalStr := string(original)
// 	klog.Infof("Original SSE content for request %s (length: %d bytes)", requestID, len(originalStr))

// 	// Marshal metrics to JSON
// 	metricsJSON, err := json.Marshal(metrics)
// 	if err != nil {
// 		return "nil", fmt.Errorf("failed to marshal metrics: %w", err)
// 	}

// 	// Format the metrics event
// 	metricsEvent := fmt.Sprintf("event: metrics\ndata: %s\n\n", metricsJSON)

// 	// For [DONE] case, insert before [DONE]
// 	if strings.Contains(originalStr, "data: [DONE]") {
// 		newBody := strings.Replace(originalStr, "data: [DONE]", metricsEvent+"data: [DONE]", 1)
// 		klog.Infof("Injected metrics before [DONE] for request %s", requestID)
// 		return newBody, nil
// 	}

// 	// If there's no [DONE], append the metrics event
// 	newBody := originalStr
// 	if !strings.HasSuffix(newBody, "\n\n") {
// 		if !strings.HasSuffix(newBody, "\n") {
// 			newBody += "\n"
// 		}
// 		newBody += "\n"
// 	}

// 	newBody += metricsEvent
// 	klog.Infof("Appended metrics event for request %s", requestID)

// 	return newBody, nil
// }

// func (s *Server) CollectRequestMetrics(timing *RequestTiming, currentTime time.Time, requestID string,
// 	routingCtx *types.RoutingContext, stream bool,
// 	numInputTokens int64, numOutputTokens int64,
// 	numTotalTokens int64) MetricsData {
// 	// Calculate TTFT (Time To First Token)
// 	ttftMs := int64(0)
// 	if !timing.firstTokenTime.IsZero() {
// 		ttftMs = timing.firstTokenTime.Sub(timing.startTime).Milliseconds()
// 	}

// 	// Calculate TPOT (Time Per Output Token)
// 	tpotMs := int64(0)
// 	if !timing.firstTokenTime.IsZero() {
// 		totalGenerationTimeMs := currentTime.Sub(timing.firstTokenTime).Milliseconds()

// 		// Use the correct token count
// 		effectiveTokenCount := int64(0)
// 		if stream && timing.tokenCount > 1 {
// 			// For streaming, use our counted tokens
// 			effectiveTokenCount = int64(timing.tokenCount - 1) // Exclude first token
// 		} else if numOutputTokens > 1 {
// 			// Use the actual output tokens from usage
// 			effectiveTokenCount = numOutputTokens - 1
// 		}

// 		if effectiveTokenCount > 0 {
// 			tpotMs = totalGenerationTimeMs / effectiveTokenCount
// 		}
// 	}

// 	// Calculate end-to-end latency
// 	end_to_end_latency_in_ms := time.Since(timing.startTime).Milliseconds()

// 	// Get KV cache hit ratio
// 	kvCacheHitRatio := utils.GetKVCacheHitRatio(requestID)

// 	// Get KV cache hit ratios for all pods
// 	allPodsRatios := utils.GetAllPodsKVCacheHitRatios(requestID)

// 	// Get inflight requests for all pods
// 	numInflightRequestsAllPods := utils.GetInflightRequestsForAllPods(requestID)

// 	// Get vLLM GPU KV cache usage
// 	vllmGPUKVCacheUsage, err := utils.GetvLLMGPUKVCacheUsageForTheRequestForAllPods(requestID)
// 	if err != nil {
// 		vllmGPUKVCacheUsage = make(map[string]float64)
// 	}

// 	// Get vLLM CPU KV cache usage
// 	vllmCPUKVCacheUsage, err := utils.GetvLLMCPUKVCacheUsageForTheRequestForAllPods(requestID)
// 	if err != nil {
// 		vllmCPUKVCacheUsage = make(map[string]float64)
// 	}

// 	// Get vLLM number of running requests
// 	vllmNumRequestsRunning, err := utils.GetvLLMNumRequestsRunningForTheRequestForAllPods(requestID)
// 	if err != nil {
// 		vllmNumRequestsRunning = make(map[string]float64)
// 	}

// 	// Get vLLM number of waiting requests
// 	vllmNumRequestWaiting, err := utils.GetvLLMNumRequestsWaitingForTheRequestForAllPods(requestID)
// 	if err != nil {
// 		vllmNumRequestWaiting = make(map[string]float64)
// 	}

// 	// Get selected pod IP
// 	selectedPodIP := "unknown"
// 	if routingCtx != nil {
// 		selectedPodIP = routingCtx.TargetAddress()
// 	}

// 	// Create the metrics data structure
// 	metricsData := MetricsData{
// 		TTFT:               ttftMs,
// 		TPOT:               tpotMs,
// 		E2ELatency:         end_to_end_latency_in_ms,
// 		KVCacheHitRatio:    kvCacheHitRatio,
// 		AllPodsRatios:      allPodsRatios,
// 		InflightRequests:   numInflightRequestsAllPods,
// 		GPUKVCacheUsage:    vllmGPUKVCacheUsage,
// 		CPUKVCacheUsage:    vllmCPUKVCacheUsage,
// 		NumRequestsRunning: vllmNumRequestsRunning,
// 		NumRequestsWaiting: vllmNumRequestWaiting,
// 		InputTokens:        numInputTokens,
// 		OutputTokens:       numOutputTokens,
// 		TotalTokens:        numTotalTokens,
// 		SelectedPod:        selectedPodIP,
// 	}

// 	// Clean up resources
// 	utils.CleanupKVCacheHitRatio(requestID)
// 	utils.DecrementNumInflightForPod(requestID)
// 	utils.CleanupInflightRequests(requestID)
// 	utils.CleanupvLLMGPUKVCacheUsage(requestID)
// 	utils.CleanupvLLMCPUKVCacheUsage(requestID)
// 	utils.CleanupvLLMNumRequestsRunning(requestID)
// 	utils.CleanupvLLMNumRequestsWaiting(requestID)

// 	// Log the metrics
// 	klog.Infof("**,latency_metrics,requestID,%s,selectedpod,%s,ttft,%d,tpot: %d, e2e: %d, numInputTokens: %d, numOutputTokens: %d, numTotalTokens: %d, kvCacheHitRatio: %.4f, numInflightRequestsAllPods: %v",
// 		requestID, selectedPodIP, ttftMs, tpotMs, end_to_end_latency_in_ms,
// 		numInputTokens, numOutputTokens, numTotalTokens,
// 		kvCacheHitRatio, numInflightRequestsAllPods)

// 	return metricsData
// }

// // Helper function to convert map[string]float64 to JSON string
// func mapToJSONString(m map[string]float64) string {
// 	if len(m) == 0 {
// 		return "{}"
// 	}

// 	jsonBytes, err := json.Marshal(m)
// 	if err != nil {
// 		// If there's an error, fall back to empty JSON object
// 		return "{}"
// 	}

// 	return string(jsonBytes)
// }

// // Helper function to convert map[string]int to JSON string
// func mapIntToJSONString(m map[string]int) string {
// 	if len(m) == 0 {
// 		return "{}"
// 	}

// 	jsonBytes, err := json.Marshal(m)
// 	if err != nil {
// 		// If there's an error, fall back to empty JSON object
// 		return "{}"
// 	}

// 	return string(jsonBytes)
// }
