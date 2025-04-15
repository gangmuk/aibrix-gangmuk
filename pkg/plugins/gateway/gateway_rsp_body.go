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

	// Process response body
	currentTime := time.Now()
	if timing != nil {
		// Extract usage for streaming responses
		if stream {
			usage = s.handleStreamingResponse(requestID, b.ResponseBody.GetBody())

			// Process streaming chunks for token counting
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
			// For non-streaming, process the response
			// Use request ID as a key to store per-request buffer
			buf, _ := s.requestBuffers.LoadOrStore(requestID, &bytes.Buffer{})
			buffer := buf.(*bytes.Buffer)
			// Append data to per-request buffer
			buffer.Write(b.ResponseBody.Body)

			// Record first token time when we get the first data
			if timing.firstTokenTime.IsZero() && b.ResponseBody.EndOfStream {
				timing.firstTokenTime = currentTime
			}

			if !b.ResponseBody.EndOfStream {
				// Partial data received, wait for more chunks
				return &extProcPb.ProcessingResponse{
					Response: &extProcPb.ProcessingResponse_ResponseBody{
						ResponseBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{},
						},
					},
				}, complete
			}

			// Last part received, process the full response
			finalBody := buffer.Bytes()
			// Clean up the buffer after final processing
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

			// Extract usage from complete response
			usage = res.Usage
		}

		// Only calculate and add timing metrics at the end of the stream
		if b.ResponseBody.EndOfStream {
			// Calculate timing metrics and add headers
			timingHeaders := s.calculateTimingMetrics(timing, currentTime, requestID, routerCtx, stream, usage.PromptTokens, usage.CompletionTokens, usage.TotalTokens)
			headers = append(headers, timingHeaders...)

			// Clean up timing data when we're done
			s.requestTimings.Delete(requestID)
		}
	}

	// Handle token usage and complete flag
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
		// Wrapped in a function to delay the evaluation of parameters. Using complete to make sure DoneRequestTrace only call once for a request.
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
		// Use request ID as a key to store per-request buffer
		// Retrieve or create buffer
		buf, _ := requestBuffers.LoadOrStore(requestID, &bytes.Buffer{})
		buffer := buf.(*bytes.Buffer)
		// Append data to per-request buffer
		buffer.Write(b.ResponseBody.Body)

		if !b.ResponseBody.EndOfStream {
			// Partial data received, wait for more chunks, we just return a common response here.
			return &extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_ResponseBody{
					ResponseBody: &extProcPb.BodyResponse{
						Response: &extProcPb.CommonResponse{},
					},
				},
			}, complete
		}

		// Last part received, process the full response
		finalBody := buffer.Bytes()
		// Clean up the buffer after final processing
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
