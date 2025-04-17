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
	statusCode := 0
	selectedPodIP, ok := s.selectedPodIP.Load(requestID)
	if !ok {
		selectedPodIP = "unknown"
	}
	if storedCode, ok := s.statusCode.Load(requestID); ok { // stored in gateway_rsp_headers.go
		statusCode = storedCode.(int)
	}
	klog.Errorf("Response status code: %d, requestID: %s, selectedPodIP: %s", statusCode, requestID, selectedPodIP)

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
					klog.InfoS("First token received", "requestID", requestID,
						"ttft_ms", currentTime.Sub(timing.startTime).Milliseconds())
				}
				if len(evt.Choices) > 0 && evt.Choices[0].Delta.Content != "" {
					timing.tokenCount++
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
			}
			s.requestTimings.Delete(requestID)
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
