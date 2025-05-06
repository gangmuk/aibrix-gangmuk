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
	"fmt"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	routing "github.com/vllm-project/aibrix/pkg/plugins/gateway/algorithms"
	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
)

func (s *Server) HandleRequestBody(ctx context.Context, requestID string, req *extProcPb.ProcessingRequest, user utils.User, routingAlgorithm types.RoutingAlgorithm) (*extProcPb.ProcessingResponse, string, *types.RoutingContext, bool, int64) {
	klog.Infof("HandleRequestBody context state, requestID: %s, ctx.Err(): %v", requestID, ctx.Err())

	var model string
	var subAlgorithm string
	var routingCtx *types.RoutingContext
	var ok, stream bool
	var term int64 // Identify the trace window

	var jsonMap map[string]interface{}
	body := req.Request.(*extProcPb.ProcessingRequest_RequestBody)
	if err := json.Unmarshal(body.RequestBody.GetBody(), &jsonMap); err != nil {
		klog.ErrorS(err, "error to unmarshal response", "requestID", requestID, "requestBody", string(body.RequestBody.GetBody()))
		return generateErrorResponse(envoyTypePb.StatusCode_InternalServerError,
			[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
				Key: HeaderErrorRequestBodyProcessing, RawValue: []byte("true")}}},
			"error processing request body"), model, routingCtx, stream, term
	}

	if model, ok = jsonMap["model"].(string); !ok || model == "" {
		klog.ErrorS(nil, "model error in request", "requestID", requestID, "jsonMap", jsonMap)
		return generateErrorResponse(envoyTypePb.StatusCode_InternalServerError,
			[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
				Key: HeaderErrorNoModelInRequest, RawValue: []byte(model)}}},
			"no model in request body"), model, routingCtx, stream, term
	}

	if subAlgorithmValue, exists := jsonMap["subAlgorithm"]; exists {
		if subAlgorithm, ok = subAlgorithmValue.(string); !ok {
			klog.InfoS("subAlgorithm is not a string", "requestID", requestID, "value", subAlgorithmValue)
			subAlgorithm = ""
		}
	} else {
		subAlgorithm = ""
		klog.V(4).InfoS("subAlgorithm not present in request. set it empty string", "requestID", requestID)
	}

	// early reject the request if model doesn't exist.
	if !s.cache.HasModel(model) {
		klog.ErrorS(nil, "model doesn't exist in cache, probably wrong model name", "requestID", requestID, "model", model)
		return generateErrorResponse(envoyTypePb.StatusCode_BadRequest,
			[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
				Key: HeaderErrorNoModelBackends, RawValue: []byte(model)}}},
			fmt.Sprintf("model %s does not exist", model)), model, routingCtx, stream, term
	}

	// early reject if no pods are ready to accept request for a model
	podsArr, err := s.cache.ListPodsByModel(model)
	if err != nil || podsArr == nil || podsArr.Len() == 0 || utils.CountRoutablePods(podsArr.All()) == 0 {
		klog.ErrorS(err, "no ready pod available", "requestID", requestID, "model", model)
		return generateErrorResponse(envoyTypePb.StatusCode_ServiceUnavailable,
			[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
				Key: HeaderErrorNoModelBackends, RawValue: []byte("true")}}},
			fmt.Sprintf("error on getting pods for model %s", model)), model, routingCtx, stream, term
	}

	stream, ok = jsonMap["stream"].(bool)
	if ok && stream {
		if errRes := validateStreamOptions(requestID, user, jsonMap); errRes != nil {
			return errRes, model, routingCtx, stream, term
		}
	}

	headers := []*configPb.HeaderValueOption{}
	if routingAlgorithm == routing.RouterNotSet {
		headers = append(headers, &configPb.HeaderValueOption{
			Header: &configPb.HeaderValue{
				Key:      "model",
				RawValue: []byte(model),
			},
		})
		klog.InfoS("request start", "requestID", requestID, "model", model)
	} else {
		message, extErr := getRequestMessage(jsonMap)
		if extErr != nil {
			klog.Errorf("error to get request message: %v, requestID: %s", extErr, requestID)
			return extErr, model, routingCtx, stream, term
		}
		// klog.InfoS("Context state before pod selection", "requestID", requestID, "isDone", ctx.Err() != nil)
		routingCtx = routingAlgorithm.NewContext(ctx, model, message, requestID)
		routingCtx.SubAlgorithm = subAlgorithm

		// klog.V(5).InfoS("New routing context created", "requestID", requestID, "isDone", routingCtx.Err() != nil)

		if routingCtx != nil {
			s.routingContexts.Store(requestID, routingCtx)
		}

		prefill_tokens, err := utils.TokenizeInputText(routingCtx.Message)
		if err != nil {
			klog.Errorf("requestID: %s, Tokenization failed: %v", routingCtx.RequestID, err)
		}
		utils.SetPrefillTokensForRequest(routingCtx.RequestID, prefill_tokens)

		//////////////////////////////////////////////////////////
		targetPodIP, err := s.selectTargetPod(routingCtx, podsArr)
		s.selectedPodIP.Store(requestID, targetPodIP)
		//////////////////////////////////////////////////////////
		targetPodIPWithPort := routingCtx.TargetAddressWithoutPort()

		utils.SetNumPrefillTokensForRequest(routingCtx.RequestID, len(prefill_tokens))
		klog.Infof("SetNumPrefillTokensForRequest, %s, %d", routingCtx.RequestID, len(prefill_tokens))
		ret := utils.IncrementNumPrefillTokensForPod(targetPodIPWithPort, len(prefill_tokens))
		klog.Infof("IncrementNumPrefillTokensForPod(%s) by %d, %d", targetPodIPWithPort, len(prefill_tokens), ret)

		s.requestTimings.Store(requestID, &RequestTiming{
			startTime:         time.Now(),
			totalTokenCount:   int64(len(prefill_tokens)),
			prefillTokenCount: int64(len(prefill_tokens)),
			decodeTokenCount:  0,
			IsPrefill:         true,
		})
		utils.StoreRequestToPodIP(routingCtx.RequestID, targetPodIPWithPort)
		utils.IncrementNumInflightForPod(routingCtx.RequestID)
		utils.StoreInflightRequestsForTheRequest(routingCtx.RequestID)
		targetMetrics := [...]string{
			utils.MetricGPUCacheUsagePerc,
			utils.MetricCPUCacheUsagePerc,
			utils.MetricNumRequestsRunning,
			utils.MetricNumRequestsWaiting,
		}
		var wg sync.WaitGroup
		readyPods := utils.FilterRoutablePods(podsArr.All())
		for _, pod := range readyPods {
			wg.Add(1)
			go func(pod *v1.Pod) {
				defer wg.Done()
				for _, metricName := range targetMetrics {
					switch metricName {
					case utils.MetricGPUCacheUsagePerc:
						utils.ReadAndStorevLLMGPUKVCacheUsage(routingCtx.RequestID, pod)
					case utils.MetricCPUCacheUsagePerc:
						utils.ReadAndStorevLLMCPUKVCacheUsage(routingCtx.RequestID, pod)
					case utils.MetricNumRequestsRunning:
						utils.ReadAndStorevLLMNumRequestsRunning(routingCtx.RequestID, pod)
					case utils.MetricNumRequestsWaiting:
						utils.ReadAndStorevLLMNumRequestsWaiting(routingCtx.RequestID, pod)
					default:
						klog.Errorf("Unknown metric: %s", metricName)
						return
					}
				}
			}(pod)
		}
		wg.Wait()
		//////////////////////////////////////////////////////////////////

		if targetPodIP == "" || err != nil {
			klog.ErrorS(err, "failed to select target pod", "requestID", requestID, "routingAlgorithm", routingAlgorithm, "model", model)
			return generateErrorResponse(
				envoyTypePb.StatusCode_ServiceUnavailable,
				[]*configPb.HeaderValueOption{{Header: &configPb.HeaderValue{
					Key: HeaderErrorRouting, RawValue: []byte("true")}}},
				"error on selecting target pod"), model, routingCtx, stream, term
		}

		headers = append(headers,
			&configPb.HeaderValueOption{
				Header: &configPb.HeaderValue{
					Key:      HeaderRoutingStrategy,
					RawValue: []byte(routingAlgorithm),
				},
			},
			&configPb.HeaderValueOption{
				Header: &configPb.HeaderValue{
					Key:      HeaderTargetPod,
					RawValue: []byte(targetPodIP),
				},
			})
		klog.InfoS("request start", "requestID", requestID, "model", model, "routingAlgorithm", routingAlgorithm, "targetPodIP", targetPodIP)
	}

	term = s.cache.AddRequestCount(routingCtx, requestID, model)

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
	}, model, routingCtx, stream, term
}
