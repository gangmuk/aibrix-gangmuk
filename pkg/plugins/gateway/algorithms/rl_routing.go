package routingalgorithms

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
	"github.com/vllm-project/aibrix/pkg/utils/prefixcacheindexer"
	"github.com/vllm-project/aibrix/pkg/utils/tokenizer"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

var (
	flushed                    = false
	trained                    = false
	received_the_first_request = false
	flushPeriod                = 10 * time.Second
	minNumLogMessagesToFlush   = utils.LoadEnvInt("MIN_NUM_LOG_MESSAGES_TO_FLUSH", 100)
	allPodIPs                  = []string{}
	fake_request_id            = 0
	numFlush                   = 0
)

var (
	httpClientForRLAgent = &http.Client{
		Timeout: 30000 * time.Millisecond,
		Transport: &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 100,
			IdleConnTimeout:     180 * time.Second,
			DisableCompression:  false,
			DialContext: (&net.Dialer{
				Timeout:   5000 * time.Millisecond,
				KeepAlive: 30 * time.Second,
			}).DialContext,
			TLSHandshakeTimeout:   5000 * time.Millisecond,
			ForceAttemptHTTP2:     true, // Enable HTTP/2
			ResponseHeaderTimeout: 5 * time.Second,
		},
	}
	routingAgentURL = "http://routing-agent-service.default.svc.cluster.local:8080"
	inferEndpoint   = "/infer"
	flushEndpoint   = "/flush"
)

type rlOnlineRouter struct {
	prefixCacheIndexer *prefixcacheindexer.PrefixHashTable
	tokenizer          tokenizer.Tokenizer
}

func NewRLOnlineRouter() (types.Router, error) {
	var tokenizerObj tokenizer.Tokenizer
	// tokenizerObj = tokenizer.NewTiktokenTokenizer()
	tokenizerObj = tokenizer.NewCharacterTokenizer()

	router := &rlOnlineRouter{
		tokenizer:          tokenizerObj,
		prefixCacheIndexer: prefixcacheindexer.NewPrefixHashTable(),
	}
	klog.InfoS("Created RL online router")
	return router, nil
}

// // flush real request log collected
func FlushLogMessageToRLAgent() {
	if utils.UseRealRequest == "true" {
		klog.Infof("flushing real request log to RL agent")
		done := make(chan struct{})
		go func() {
			ticker := time.NewTicker(flushPeriod)
			defer ticker.Stop()
			for {
				select {
				case <-ticker.C:
					if !received_the_first_request {
						klog.Infof("The first request has not been received yet, skipping the flush. (needed to construct the running pod IPs)")
						continue // Skip this iteration and check again on the next tick
					}
					if len(utils.RequestToLogMessage) > minNumLogMessagesToFlush {
						klog.Infof("Starting %dth flushing for %d number of log messages", numFlush+1, len(utils.RequestToLogMessage))

						// utils.RequestToLogMessageMutex.Lock()
						reqBody, err := json.Marshal(utils.RequestToLogMessage)
						// utils.RequestToLogMessageMutex.Unlock()

						if err != nil {
							klog.Errorf("Failed flush. failed marshal RequestToLogMessage: %v", err)
							utils.CleanupAllRequestLogMessage()
							continue
						}

						url := fmt.Sprintf("%s%s", routingAgentURL, flushEndpoint)
						req, reqErr := http.NewRequest("POST", url, bytes.NewBuffer(reqBody))
						if reqErr != nil {
							klog.Errorf("Failed flush. failed to create request: %v", reqErr)
							utils.CleanupAllRequestLogMessage()
							continue
						}
						req.Header.Set("Content-Type", "application/json")
						resp, sendErr := httpClientForRLAgent.Do(req) // flush request
						klog.Info("flush 1")
						if sendErr != nil {
							klog.Errorf("Failed flush. failed to send request: %v", sendErr)
							utils.CleanupAllRequestLogMessage()
							continue
						}
						klog.Info("flush 2")
						if resp.StatusCode != http.StatusOK {
							klog.Errorf("Received non-200 response: %s", resp.Status)
							utils.CleanupAllRequestLogMessage()
							klog.Errorf("Failed flush. Received non-200 response: %s", resp.Status)
							continue
						}
						klog.Info("flush 3")
						body, readErr := ioutil.ReadAll(resp.Body)
						klog.Info("flush 4")
						if readErr != nil {
							klog.Errorf("Failed to read response body: %v", readErr)
							utils.CleanupAllRequestLogMessage()
							klog.Errorf("Failed flush. Failed to read response body: %v", readErr)
							continue
						}
						klog.Info("flush 5")
						resp.Body.Close()
						klog.Info("flush 7")
						utils.CleanupAllRequestLogMessage()
						klog.Infof("Successfully flushed, response: %s", string(body))
						flushed = true
						numFlush += 1
					} else {
						klog.Infof("Not enough log messages to flush: %d", len(utils.RequestToLogMessage))
					}
				case <-done:
					klog.Info("Flushing goroutine is shutting down")
					return
				}
			}
		}()
	} else {
		done := make(chan struct{})
		go func() {
			ticker := time.NewTicker(flushPeriod)
			defer ticker.Stop()
			for {
				select {
				case <-ticker.C:
					if !received_the_first_request {
						klog.Infof("The first request has not been received yet, skipping the flush")
						continue // Skip this iteration and check again on the next tick
					}

					if flushed {
						// flush only once to simplify experiment
						klog.Infof("Skip flushing. Configured to flush only once for Fake data, utils.UseRealRequest == false")
						continue
					}

					// If we got here, the first request has been received
					klog.Infof("Start flushing log messages to RL agent, %dth flush", numFlush)
					allPodIPs = utils.GetAllPodIPsFromRegistry()
					klog.Infof("All pod IPs: %v", allPodIPs)

					logs := utils.GenerateLogMessages(allPodIPs, minNumLogMessagesToFlush)
					start_request_id_of_this_batch := fake_request_id
					for _, log := range logs {
						utils.AddRequestLogMessage(fmt.Sprintf("%d", fake_request_id), log)
						fake_request_id += 1
					}
					end_request_id_of_this_batch := fake_request_id
					klog.Infof("Newly added request ids %d-%d", start_request_id_of_this_batch, end_request_id_of_this_batch)
					klog.Infof("Starting flushing process for %d logs ", len(utils.RequestToLogMessage))
					klog.V(5).Infof("logs: %v", logs)
					reqBody, err := json.Marshal(utils.RequestToLogMessage)
					if err != nil {
						klog.Errorf("Failed flush. failed to marshal RequestToLogMessage: %v", err)
						continue
					}

					url := fmt.Sprintf("%s%s", routingAgentURL, flushEndpoint)
					req, reqErr := http.NewRequest("POST", url, bytes.NewBuffer(reqBody))
					if reqErr != nil {
						klog.Errorf("Failed flush. failed to create request: %v", reqErr)
						continue
					}

					req.Header.Set("Content-Type", "application/json")
					resp, sendErr := httpClientForRLAgent.Do(req)
					if sendErr != nil {
						klog.Errorf("Failed flush. failed to send request: %v", sendErr)
						continue
					}

					// Ensure we have a valid response before proceeding
					if resp != nil {
						if resp.StatusCode != http.StatusOK {
							klog.Errorf("Received non-200 response: %s", resp.Status)
						}

						body, readErr := ioutil.ReadAll(resp.Body)
						if readErr != nil {
							klog.Errorf("Failed flush. failed to read response body: %v", readErr)
						} else {
							klog.Infof("Successfully sent RequestToLogMessage to RL agent: %s", string(body))
						}
						resp.Body.Close()
					}

					//// Delete when the RL agent is doing continuous learning.
					//// When the RL agent trains the model from scratch at every flush call, don't discard previous logs but flush all history every time.
					// utils.CleanupAllRequestLogMessage()
					flushed = true
					numFlush += 1
				case <-done:
					return
				}
			}
		}()
	}
}

func init() {
	RegisterDelayedConstructor("rl-online-router", NewRLOnlineRouter)
	FlushLogMessageToRLAgent()
}

// RouteResponse is received from the routing agent
type RouteResponse struct {
	RequestID   string  `json:"request_id"`
	SelectedPod string  `json:"selected_pod"`
	Confidence  float64 `json:"confidence"`
}

func jsonStringify(data interface{}, lock *sync.RWMutex) string {
	lock.RLock()
	defer lock.RUnlock()
	jsonData, err := json.Marshal(data)
	if err != nil {
		klog.Errorf("Error marshaling data to JSON: %v", err)
		return "{}"
	}
	return string(jsonData)
}

func GetPod(podIP string, pods []*v1.Pod) *v1.Pod {
	for _, pod := range pods {
		if pod.Status.PodIP == podIP {
			return pod
		}
	}
	return nil
}

// Route selects the optimal pod based on latency predictions
func (r *rlOnlineRouter) Route(ctx *types.RoutingContext, pods types.PodList) (string, error) {
	route_start_time := time.Now()
	// Get all ready pods
	readyPods := pods.All()
	var targetPod *v1.Pod
	if !received_the_first_request {
		klog.Infof("This is the first request, using fallback routing and return right away. Give some time for the RL agent to warm up.")
		targetPod, _ = r.fallbackRouting(ctx, readyPods)
		received_the_first_request = true
		allPodIPs = utils.GetAllPodIPsFromRegistry()
		ctx.SetTargetPod(targetPod)
		return ctx.TargetAddress(), nil
	}
	if !flushed {
		klog.Infof("At least one training is required for RL based routing. Using fallback routing and return right away.")
		targetPod, _ = r.fallbackRouting(ctx, readyPods)
		ctx.SetTargetPod(targetPod)
		return ctx.TargetAddress(), nil
	}
	if len(readyPods) == 0 {
		return "", fmt.Errorf("no ready pods available")
	}

	if len(readyPods) == 1 {
		ctx.SetTargetPod(readyPods[0])
		return ctx.TargetAddress(), nil
	}
	var prefixHashes []uint64
	var matchedPods map[string]int
	readyPodsMap := map[string]struct{}{}
	for _, pod := range readyPods {
		readyPodsMap[pod.Status.PodIP] = struct{}{}
	}
	// tokenizer_start_time := time.Now()
	tokens, err := r.tokenizer.TokenizeInputText(ctx.Message)
	// tokenizer_overhead := time.Since(tokenizer_start_time).Milliseconds()
	if err != nil {
		klog.Errorf("requestID: %s, Tokenization failed: %v", ctx.RequestID, err)
		return "", err
	}
	var log string
	log_construction_start_time := time.Now()
	if utils.UseRealRequest == "true" {
		klog.Infof("utils.UseRealRequest:%s, requestID: %s", utils.UseRealRequest, ctx.RequestID)
		numInputTokens := len(tokens)
		numOutputTokens := 128 // Placeholder for output tokens
		numTotalTokens := numInputTokens + numOutputTokens

		matchedPods, prefixHashes = r.prefixCacheIndexer.MatchPrefix(tokens, ctx.Model, readyPodsMap)
		if len(matchedPods) == 0 {
			klog.Infof("No matched pods found. Filled all readypods with 0 kv cache hit ratio")
			for _, pod := range readyPods {
				matchedPods[pod.Status.PodIP] = 0
			}
		}
		klog.Infof("matchedPods: %v", matchedPods)
		utils.StoreKVCacheHitRatio(ctx.RequestID, matchedPods)

		// Prepare for JSON strings to use in logging
		var jsonStrings = make(map[string]string)

		// 1. KV cache hit ratios
		allPodsKvCacheHitRatios := utils.GetAllPodsKVCacheHitRatios(ctx.RequestID)
		jsonStrings["allPodsKvCacheHitRatios"] = jsonStringify(allPodsKvCacheHitRatios, utils.GetrequestAllPodsKVCacheMutex())
		klog.V(5).Infof("allPodsKvCacheHitRatios: %s", jsonStrings["allPodsKvCacheHitRatios"])

		// 2. Inflight requests
		numInflightRequestsAllPods := utils.GetInflightRequestsForAllPods(ctx.RequestID)
		jsonStrings["numInflightRequestsAllPods"] = jsonStringify(numInflightRequestsAllPods, utils.GetrequestInflightMutex())

		// 3. GPU KV cache usage
		vllmGPUKVCacheUsage, err := utils.GetvLLMGPUKVCacheUsageForAllPods(ctx.RequestID)
		if err == nil {
			jsonStrings["vllmGPUKVCacheUsage"] = jsonStringify(vllmGPUKVCacheUsage, utils.GetvllmGPUKVCacheUsageMutex())
		} else {
			jsonStrings["vllmGPUKVCacheUsage"] = "{}"
		}

		// 4. CPU KV cache usage
		vllmCPUKVCacheUsage, err := utils.GetvLLMCPUKVCacheUsageForTheRequestForAllPods(ctx.RequestID)
		if err == nil {
			jsonStrings["vllmCPUKVCacheUsage"] = jsonStringify(vllmCPUKVCacheUsage, utils.GetvllmCPUKVCacheUsageMutex())
		} else {
			jsonStrings["vllmCPUKVCacheUsage"] = "{}"
		}

		// 5. Number of running requests
		vllmNumRequestsRunning, err := utils.GetvLLMNumRequestsRunningForAllPods(ctx.RequestID)
		if err == nil {
			jsonStrings["vllmNumRequestsRunning"] = jsonStringify(vllmNumRequestsRunning, utils.GetvllmNumRequestsRunningMutex())
		} else {
			jsonStrings["vllmNumRequestsRunning"] = "{}"
		}

		// 6. Number of waiting requests
		vllmNumRequestWaiting, err := utils.GetvLLMNumRequestsWaitingForAllPods(ctx.RequestID)
		if err == nil {
			jsonStrings["vllmNumRequestWaiting"] = jsonStringify(vllmNumRequestWaiting, utils.GetvllmNumRequestsWaitingMutex())
		} else {
			jsonStrings["vllmNumRequestWaiting"] = "{}"
		}

		numPrefillTokensForAllPods := utils.GetNumPrefillTokensForAllPods()
		jsonStrings["numPrefillTokensForAllPods"] = jsonStringify(numPrefillTokensForAllPods, utils.GetpodTotalPrefillTokensMutex())

		numDecodeTokensForAllPods := utils.GetNumDecodeTokensForAllPods()
		jsonStrings["numDecodeTokensForAllPods"] = jsonStringify(numDecodeTokensForAllPods, utils.GetpodTotalDecodeTokensMutex())

		podDetailedMetrics := utils.GetRequestPodMetrics(ctx.RequestID)
		jsonStrings["podMetricsLastSecond"] = jsonStringify(podDetailedMetrics, utils.MetricsTracker.GetMutex())
		logFormat := `**@latency_metrics@requestID@%s@request_start_time@%d@request_end_time@-9999@selectedpod@-9999@ttft@-9999@avg_tpot@-9999@total_decode_time@-9999@e2e@-9999@numInputTokens@%d@numOutputTokens@%d@numTotalTokens@%d@allPodsKvCacheHitRatios@%s@numInflightRequestsAllPods@%s@vllmGPUKVCacheUsage@%s@vllmCPUKVCacheUsage@%s@vllmNumRequestsRunning@%s@vllmNumRequestsWaiting@%s@podMetricsLastSecond@%s@numPrefillTokensForAllPods@%s@numDecodeTokensForAllPods@%s`
		log = fmt.Sprintf(
			logFormat,
			ctx.RequestID,
			time.Now().UnixMicro(),
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
			jsonStrings["numPrefillTokensForAllPods"],
			jsonStrings["numDecodeTokensForAllPods"],
		)
	} else {
		numLogs := 1
		log = utils.GenerateLogMessages(allPodIPs, numLogs)[0]
	}
	log_construction_overhead := time.Since(log_construction_start_time).Milliseconds()

	klog.V(5).Infof("/infer: %s", log)
	reqBody, err := json.Marshal(log)
	if err != nil {
		klog.Errorf("Failed to marshal RequestToLogMessage: %v, requestID: %s", err, ctx.RequestID)
		targetPod, _ = r.fallbackRouting(ctx, readyPods)
		ctx.SetTargetPod(targetPod)
		return ctx.TargetAddress(), nil
	}
	request_prepare_overhead := time.Since(route_start_time).Milliseconds()
	url := fmt.Sprintf("%s%s", routingAgentURL, inferEndpoint)
	req, reqErr := http.NewRequest("POST", url, bytes.NewBuffer(reqBody))
	if reqErr != nil {
		klog.Errorf("Failed to create request: %v, requestID: %s", reqErr, ctx.RequestID)
		targetPod, _ = r.fallbackRouting(ctx, readyPods)
		ctx.SetTargetPod(targetPod)
		return ctx.TargetAddress(), nil
	}
	req.Header.Set("Content-Type", "application/json")
	resp, sendErr := httpClientForRLAgent.Do(req)
	if sendErr != nil {
		klog.Errorf("Failed to send request: %v, requestID: %s", sendErr, ctx.RequestID)
		targetPod, _ = r.fallbackRouting(ctx, readyPods)
		ctx.SetTargetPod(targetPod)
		return ctx.TargetAddress(), nil
	}
	if resp.StatusCode != http.StatusOK {
		klog.Errorf("Received non-200 response: %s, requestID: %s", resp.Status, ctx.RequestID)
	}
	body, readErr := ioutil.ReadAll(resp.Body)
	if readErr != nil {
		klog.Errorf("Failed to read response body: %v, requestID: %s", readErr, ctx.RequestID)
		targetPod, _ = r.fallbackRouting(ctx, readyPods)
		ctx.SetTargetPod(targetPod)
		return ctx.TargetAddress(), nil
	}

	response_process_start := time.Now()
	// body: {"confidence":0.4398832619190216,"request_id":"10","selected_pod":"10.0.1.30"}
	var routeResponse RouteResponse
	if err := json.Unmarshal(body, &routeResponse); err != nil {
		klog.Errorf("Failed to unmarshal response body: %v, requestID: %s", err, ctx.RequestID)
		targetPod, _ = r.fallbackRouting(ctx, readyPods)
		ctx.SetTargetPod(targetPod)
		return ctx.TargetAddress(), nil
	}
	targetPod = GetPod(routeResponse.SelectedPod, readyPods)
	if targetPod == nil {
		klog.Errorf("No suitable pod found for selected pod IP: %s, requestID: %s", routeResponse.SelectedPod, ctx.RequestID)
		targetPod, _ = r.fallbackRouting(ctx, readyPods)
		ctx.SetTargetPod(targetPod)
		return ctx.TargetAddress(), nil
	}
	resp.Body.Close()

	/////////////////////////////////////////////////////////////

	ctx.SetTargetPod(targetPod)

	if len(prefixHashes) > 0 {
		klog.Infof("Adding prefix hashes to cache. pod: %s", targetPod.Status.PodIP)
		r.prefixCacheIndexer.AddPrefix(prefixHashes, ctx.Model, targetPod.Status.PodIP)
	}

	response_process_overhead := time.Since(response_process_start).Milliseconds()
	end_to_end_overhead := time.Since(route_start_time).Milliseconds()
	formattedResponseBody := formatJSONResponse(ctx.RequestID, body)
	klog.Infof("RL router, selected podIP: %s, \n"+
		"requestID: %s, Route end_to_end_overhead %dms, \n"+
		// "requestID: %s, infer_http_request took %dms, \n"+
		// "requestID: %s, tokenizer_overhead: %dms, \n"+
		"requestID: %s, log_construction_overhead: %dms, \n"+
		"requestID: %s, request_prepare_overhead: %dms, \n"+
		"requestID: %s, response_process_overhead: %dms, \n"+
		"ResponseBody: \n%s",
		ctx.TargetAddressWithoutPort(),
		ctx.RequestID, end_to_end_overhead,
		// ctx.RequestID, infer_overhead,
		// ctx.RequestID, tokenizer_overhead,
		ctx.RequestID, log_construction_overhead,
		ctx.RequestID, request_prepare_overhead,
		ctx.RequestID, response_process_overhead, // 1ms
		formattedResponseBody)
	return ctx.TargetAddress(), nil
}

func formatJSONResponse(RequestID string, jsonBytes []byte) string {
	var data map[string]interface{}
	if err := json.Unmarshal(jsonBytes, &data); err != nil {
		return string(jsonBytes) // Return original if parsing fails
	}

	var result strings.Builder
	for key, value := range data {
		result.WriteString(fmt.Sprintf("requestID: %s, %s:%v\n", RequestID, key, value))
	}
	return strings.TrimSuffix(result.String(), "\n")
}

func (r *rlOnlineRouter) fallbackRouting(ctx *types.RoutingContext, readyPods []*v1.Pod) (*v1.Pod, error) {
	klog.Infof("Using fallback routing (random) for request %s", ctx.RequestID)
	var err error
	targetPod, err := selectRandomPod(readyPods, rand.Intn)
	if err != nil {
		klog.Errorf("Failed to select random pod: %v", err)
		return nil, err
	}
	if targetPod == nil {
		klog.Errorf("No suitable pod found for fallback routing")
		return nil, fmt.Errorf("no suitable pod found for fallback routing")
	}
	return targetPod, nil
}
