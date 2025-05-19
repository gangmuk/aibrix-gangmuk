package routingalgorithms

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
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
	flushed    = false
	logMessage = `**@latency_metrics@requestID@10@request_start_time@1746735417789560@request_end_time@1746735428906305@selectedpod@10.0.1.30@ttft@249@avg_tpot@42@total_decode_time@10867@e2e@11116@numInputTokens@1253@numOutputTokens@256@numTotalTokens@1509@allPodsKvCacheHitRatios@{"10.0.0.39":0,"10.0.1.25":0,"10.0.1.30":0,"10.0.1.32":0,"10.0.1.44":0,"10.0.3.25":0,"10.0.3.27":0,"10.0.3.7":0}@numInflightRequestsAllPods@{"10.0.0.39":1,"10.0.1.25":1,"10.0.1.30":2,"10.0.1.32":1,"10.0.1.44":2,"10.0.3.25":2,"10.0.3.27":1,"10.0.3.7":1}@vllmGPUKVCacheUsage@{"10.0.0.39":0,"10.0.1.25":0,"10.0.1.30":0,"10.0.1.32":0,"10.0.1.44":0,"10.0.3.25":0,"10.0.3.27":0,"10.0.3.7":0}@vllmCPUKVCacheUsage@{"10.0.0.39":0,"10.0.1.25":0,"10.0.1.30":0,"10.0.1.32":0,"10.0.1.44":0,"10.0.3.25":0,"10.0.3.27":0,"10.0.3.7":0}@vllmNumRequestsRunning@{"10.0.0.39":0,"10.0.1.25":0,"10.0.1.30":1,"10.0.1.32":0,"10.0.1.44":0,"10.0.3.25":1,"10.0.3.27":0,"10.0.3.7":0}@vllmNumRequestsWaiting@{"10.0.0.39":0,"10.0.1.25":0,"10.0.1.30":0,"10.0.1.32":0,"10.0.1.44":0,"10.0.3.25":0,"10.0.3.27":0,"10.0.3.7":0}@podMetricsLastSecond@{"10.0.0.39":{"last_second_avg_ttft_ms":376.5,"last_second_min_ttft_ms":360,"last_second_max_ttft_ms":393,"last_second_p50_ttft_ms":376,"last_second_p90_ttft_ms":389,"last_second_p95_ttft_ms":391,"last_second_p99_ttft_ms":392,"last_second_ttft_samples":2,"last_second_avg_tpot_ms":62.147147147147145,"last_second_min_tpot_ms":31,"last_second_max_tpot_ms":205,"last_second_p50_tpot_ms":42,"last_second_p90_tpot_ms":193,"last_second_p95_tpot_ms":203,"last_second_p99_tpot_ms":203,"last_second_tpot_samples":666,"last_second_total_requests":2,"last_second_total_decode_tokens":666,"last_second_total_prefill_tokens":9782,"last_second_total_tokens":10448},"10.0.1.25":{"last_second_avg_ttft_ms":2207.5,"last_second_min_ttft_ms":1365,"last_second_max_ttft_ms":2661,"last_second_p50_ttft_ms":2298,"last_second_p90_ttft_ms":2563,"last_second_p95_ttft_ms":2612,"last_second_p99_ttft_ms":2651,"last_second_ttft_samples":6,"last_second_avg_tpot_ms":239.31578947368422,"last_second_min_tpot_ms":44,"last_second_max_tpot_ms":367,"last_second_p50_tpot_ms":214,"last_second_p90_tpot_ms":367,"last_second_p95_tpot_ms":367,"last_second_p99_tpot_ms":367,"last_second_tpot_samples":152,"last_second_total_requests":6,"last_second_total_decode_tokens":152,"last_second_total_prefill_tokens":34243,"last_second_total_tokens":34395},"10.0.1.30":{"last_second_avg_ttft_ms":310.4,"last_second_min_ttft_ms":187,"last_second_max_ttft_ms":403,"last_second_p50_ttft_ms":325,"last_second_p90_ttft_ms":383,"last_second_p95_ttft_ms":393,"last_second_p99_ttft_ms":401,"last_second_ttft_samples":5,"last_second_avg_tpot_ms":60.9815668202765,"last_second_min_tpot_ms":29,"last_second_max_tpot_ms":183,"last_second_p50_tpot_ms":44,"last_second_p90_tpot_ms":181,"last_second_p95_tpot_ms":182,"last_second_p99_tpot_ms":182,"last_second_tpot_samples":651,"last_second_total_requests":5,"last_second_total_decode_tokens":651,"last_second_total_prefill_tokens":13504,"last_second_total_tokens":14155},"10.0.1.32":{"last_second_avg_ttft_ms":306.3333333333333,"last_second_min_ttft_ms":185,"last_second_max_ttft_ms":445,"last_second_p50_ttft_ms":289,"last_second_p90_ttft_ms":413,"last_second_p95_ttft_ms":429,"last_second_p99_ttft_ms":441,"last_second_ttft_samples":3,"last_second_avg_tpot_ms":54.7773654916512,"last_second_min_tpot_ms":40,"last_second_max_tpot_ms":115,"last_second_p50_tpot_ms":47,"last_second_p90_tpot_ms":81,"last_second_p95_tpot_ms":115,"last_second_p99_tpot_ms":115,"last_second_tpot_samples":539,"last_second_total_requests":3,"last_second_total_decode_tokens":539,"last_second_total_prefill_tokens":3700,"last_second_total_tokens":4239},"10.0.1.44":{"last_second_avg_ttft_ms":2916.2,"last_second_min_ttft_ms":2604,"last_second_max_ttft_ms":3039,"last_second_p50_ttft_ms":2963,"last_second_p90_ttft_ms":3033,"last_second_p95_ttft_ms":3036,"last_second_p99_ttft_ms":3038,"last_second_ttft_samples":5,"last_second_avg_tpot_ms":375.8235294117647,"last_second_min_tpot_ms":350,"last_second_max_tpot_ms":398,"last_second_p50_tpot_ms":397,"last_second_p90_tpot_ms":398,"last_second_p95_tpot_ms":398,"last_second_p99_tpot_ms":398,"last_second_tpot_samples":68,"last_second_total_requests":5,"last_second_total_decode_tokens":68,"last_second_total_prefill_tokens":17184,"last_second_total_tokens":17252},"10.0.3.25":{"last_second_avg_ttft_ms":6065,"last_second_min_ttft_ms":5748,"last_second_max_ttft_ms":6387,"last_second_p50_ttft_ms":6037,"last_second_p90_ttft_ms":6314,"last_second_p95_ttft_ms":6350,"last_second_p99_ttft_ms":6379,"last_second_ttft_samples":6,"last_second_avg_tpot_ms":388.6727272727273,"last_second_min_tpot_ms":370,"last_second_max_tpot_ms":406,"last_second_p50_tpot_ms":387,"last_second_p90_tpot_ms":405,"last_second_p95_tpot_ms":406,"last_second_p99_tpot_ms":406,"last_second_tpot_samples":55,"last_second_total_requests":6,"last_second_total_decode_tokens":55,"last_second_total_prefill_tokens":36665,"last_second_total_tokens":36720},"10.0.3.27":{"last_second_avg_ttft_ms":489,"last_second_min_ttft_ms":336,"last_second_max_ttft_ms":576,"last_second_p50_ttft_ms":555,"last_second_p90_ttft_ms":571,"last_second_p95_ttft_ms":573,"last_second_p99_ttft_ms":575,"last_second_ttft_samples":3,"last_second_avg_tpot_ms":111.21752265861028,"last_second_min_tpot_ms":40,"last_second_max_tpot_ms":430,"last_second_p50_tpot_ms":54,"last_second_p90_tpot_ms":189,"last_second_p95_tpot_ms":430,"last_second_p99_tpot_ms":430,"last_second_tpot_samples":331,"last_second_total_requests":3,"last_second_total_decode_tokens":331,"last_second_total_prefill_tokens":19542,"last_second_total_tokens":19873},"10.0.3.7":{"last_second_avg_ttft_ms":3893.5714285714284,"last_second_min_ttft_ms":3196,"last_second_max_ttft_ms":4500,"last_second_p50_ttft_ms":3888,"last_second_p90_ttft_ms":4356,"last_second_p95_ttft_ms":4418,"last_second_p99_ttft_ms":4483,"last_second_ttft_samples":14,"last_second_avg_tpot_ms":345.787037037037,"last_second_min_tpot_ms":324,"last_second_max_tpot_ms":370,"last_second_p50_tpot_ms":344,"last_second_p90_tpot_ms":370,"last_second_p95_tpot_ms":370,"last_second_p99_tpot_ms":370,"last_second_tpot_samples":108,"last_second_total_requests":14,"last_second_total_decode_tokens":108,"last_second_total_prefill_tokens":31864,"last_second_total_tokens":31972}}@numPrefillTokensForAllPods@{"10.0.0.39":2464,"10.0.1.25":0,"10.0.1.30":4911,"10.0.1.32":15905,"10.0.1.44":67110,"10.0.3.25":128396,"10.0.3.27":19549,"10.0.3.7":82005}@numDecodeTokensForAllPods@{"10.0.0.39":141062,"10.0.1.25":215694,"10.0.1.30":83396,"10.0.1.32":152037,"10.0.1.44":166697,"10.0.3.25":132338,"10.0.3.27":188942,"10.0.3.7":181536}`

	infer_logMessage = `**@latency_metrics@requestID@10@request_start_time@1746735417789560@request_end_time@-9999@selectedpod@-9999@ttft@-9999@avg_tpot@-9999@total_decode_time@-9999@e2e@-9999@numInputTokens@1253@numOutputTokens@256@numTotalTokens@1509@allPodsKvCacheHitRatios@{"10.0.0.39":0,"10.0.1.25":0,"10.0.1.30":0,"10.0.1.32":0,"10.0.1.44":0,"10.0.3.25":0,"10.0.3.27":0,"10.0.3.7":0}@numInflightRequestsAllPods@{"10.0.0.39":1,"10.0.1.25":1,"10.0.1.30":2,"10.0.1.32":1,"10.0.1.44":2,"10.0.3.25":2,"10.0.3.27":1,"10.0.3.7":1}@vllmGPUKVCacheUsage@{"10.0.0.39":0,"10.0.1.25":0,"10.0.1.30":0,"10.0.1.32":0,"10.0.1.44":0,"10.0.3.25":0,"10.0.3.27":0,"10.0.3.7":0}@vllmCPUKVCacheUsage@{"10.0.0.39":0,"10.0.1.25":0,"10.0.1.30":0,"10.0.1.32":0,"10.0.1.44":0,"10.0.3.25":0,"10.0.3.27":0,"10.0.3.7":0}@vllmNumRequestsRunning@{"10.0.0.39":0,"10.0.1.25":0,"10.0.1.30":1,"10.0.1.32":0,"10.0.1.44":0,"10.0.3.25":1,"10.0.3.27":0,"10.0.3.7":0}@vllmNumRequestsWaiting@{"10.0.0.39":0,"10.0.1.25":0,"10.0.1.30":0,"10.0.1.32":0,"10.0.1.44":0,"10.0.3.25":0,"10.0.3.27":0,"10.0.3.7":0}@podMetricsLastSecond@{"10.0.0.39":{"last_second_avg_ttft_ms":376.5,"last_second_min_ttft_ms":360,"last_second_max_ttft_ms":393,"last_second_p50_ttft_ms":376,"last_second_p90_ttft_ms":389,"last_second_p95_ttft_ms":391,"last_second_p99_ttft_ms":392,"last_second_ttft_samples":2,"last_second_avg_tpot_ms":62.147147147147145,"last_second_min_tpot_ms":31,"last_second_max_tpot_ms":205,"last_second_p50_tpot_ms":42,"last_second_p90_tpot_ms":193,"last_second_p95_tpot_ms":203,"last_second_p99_tpot_ms":203,"last_second_tpot_samples":666,"last_second_total_requests":2,"last_second_total_decode_tokens":666,"last_second_total_prefill_tokens":9782,"last_second_total_tokens":10448},"10.0.1.25":{"last_second_avg_ttft_ms":2207.5,"last_second_min_ttft_ms":1365,"last_second_max_ttft_ms":2661,"last_second_p50_ttft_ms":2298,"last_second_p90_ttft_ms":2563,"last_second_p95_ttft_ms":2612,"last_second_p99_ttft_ms":2651,"last_second_ttft_samples":6,"last_second_avg_tpot_ms":239.31578947368422,"last_second_min_tpot_ms":44,"last_second_max_tpot_ms":367,"last_second_p50_tpot_ms":214,"last_second_p90_tpot_ms":367,"last_second_p95_tpot_ms":367,"last_second_p99_tpot_ms":367,"last_second_tpot_samples":152,"last_second_total_requests":6,"last_second_total_decode_tokens":152,"last_second_total_prefill_tokens":34243,"last_second_total_tokens":34395},"10.0.1.30":{"last_second_avg_ttft_ms":310.4,"last_second_min_ttft_ms":187,"last_second_max_ttft_ms":403,"last_second_p50_ttft_ms":325,"last_second_p90_ttft_ms":383,"last_second_p95_ttft_ms":393,"last_second_p99_ttft_ms":401,"last_second_ttft_samples":5,"last_second_avg_tpot_ms":60.9815668202765,"last_second_min_tpot_ms":29,"last_second_max_tpot_ms":183,"last_second_p50_tpot_ms":44,"last_second_p90_tpot_ms":181,"last_second_p95_tpot_ms":182,"last_second_p99_tpot_ms":182,"last_second_tpot_samples":651,"last_second_total_requests":5,"last_second_total_decode_tokens":651,"last_second_total_prefill_tokens":13504,"last_second_total_tokens":14155},"10.0.1.32":{"last_second_avg_ttft_ms":306.3333333333333,"last_second_min_ttft_ms":185,"last_second_max_ttft_ms":445,"last_second_p50_ttft_ms":289,"last_second_p90_ttft_ms":413,"last_second_p95_ttft_ms":429,"last_second_p99_ttft_ms":441,"last_second_ttft_samples":3,"last_second_avg_tpot_ms":54.7773654916512,"last_second_min_tpot_ms":40,"last_second_max_tpot_ms":115,"last_second_p50_tpot_ms":47,"last_second_p90_tpot_ms":81,"last_second_p95_tpot_ms":115,"last_second_p99_tpot_ms":115,"last_second_tpot_samples":539,"last_second_total_requests":3,"last_second_total_decode_tokens":539,"last_second_total_prefill_tokens":3700,"last_second_total_tokens":4239},"10.0.1.44":{"last_second_avg_ttft_ms":2916.2,"last_second_min_ttft_ms":2604,"last_second_max_ttft_ms":3039,"last_second_p50_ttft_ms":2963,"last_second_p90_ttft_ms":3033,"last_second_p95_ttft_ms":3036,"last_second_p99_ttft_ms":3038,"last_second_ttft_samples":5,"last_second_avg_tpot_ms":375.8235294117647,"last_second_min_tpot_ms":350,"last_second_max_tpot_ms":398,"last_second_p50_tpot_ms":397,"last_second_p90_tpot_ms":398,"last_second_p95_tpot_ms":398,"last_second_p99_tpot_ms":398,"last_second_tpot_samples":68,"last_second_total_requests":5,"last_second_total_decode_tokens":68,"last_second_total_prefill_tokens":17184,"last_second_total_tokens":17252},"10.0.3.25":{"last_second_avg_ttft_ms":6065,"last_second_min_ttft_ms":5748,"last_second_max_ttft_ms":6387,"last_second_p50_ttft_ms":6037,"last_second_p90_ttft_ms":6314,"last_second_p95_ttft_ms":6350,"last_second_p99_ttft_ms":6379,"last_second_ttft_samples":6,"last_second_avg_tpot_ms":388.6727272727273,"last_second_min_tpot_ms":370,"last_second_max_tpot_ms":406,"last_second_p50_tpot_ms":387,"last_second_p90_tpot_ms":405,"last_second_p95_tpot_ms":406,"last_second_p99_tpot_ms":406,"last_second_tpot_samples":55,"last_second_total_requests":6,"last_second_total_decode_tokens":55,"last_second_total_prefill_tokens":36665,"last_second_total_tokens":36720},"10.0.3.27":{"last_second_avg_ttft_ms":489,"last_second_min_ttft_ms":336,"last_second_max_ttft_ms":576,"last_second_p50_ttft_ms":555,"last_second_p90_ttft_ms":571,"last_second_p95_ttft_ms":573,"last_second_p99_ttft_ms":575,"last_second_ttft_samples":3,"last_second_avg_tpot_ms":111.21752265861028,"last_second_min_tpot_ms":40,"last_second_max_tpot_ms":430,"last_second_p50_tpot_ms":54,"last_second_p90_tpot_ms":189,"last_second_p95_tpot_ms":430,"last_second_p99_tpot_ms":430,"last_second_tpot_samples":331,"last_second_total_requests":3,"last_second_total_decode_tokens":331,"last_second_total_prefill_tokens":19542,"last_second_total_tokens":19873},"10.0.3.7":{"last_second_avg_ttft_ms":3893.5714285714284,"last_second_min_ttft_ms":3196,"last_second_max_ttft_ms":4500,"last_second_p50_ttft_ms":3888,"last_second_p90_ttft_ms":4356,"last_second_p95_ttft_ms":4418,"last_second_p99_ttft_ms":4483,"last_second_ttft_samples":14,"last_second_avg_tpot_ms":345.787037037037,"last_second_min_tpot_ms":324,"last_second_max_tpot_ms":370,"last_second_p50_tpot_ms":344,"last_second_p90_tpot_ms":370,"last_second_p95_tpot_ms":370,"last_second_p99_tpot_ms":370,"last_second_tpot_samples":108,"last_second_total_requests":14,"last_second_total_decode_tokens":108,"last_second_total_prefill_tokens":31864,"last_second_total_tokens":31972}}@numPrefillTokensForAllPods@{"10.0.0.39":2464,"10.0.1.25":0,"10.0.1.30":4911,"10.0.1.32":15905,"10.0.1.44":67110,"10.0.3.25":128396,"10.0.3.27":19549,"10.0.3.7":82005}@numDecodeTokensForAllPods@{"10.0.0.39":141062,"10.0.1.25":215694,"10.0.1.30":83396,"10.0.1.32":152037,"10.0.1.44":166697,"10.0.3.25":132338,"10.0.3.27":188942,"10.0.3.7":181536}`
)

var (
	flushPeriod = 10 * time.Second

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
	tokenizerObj = tokenizer.NewTiktokenTokenizer()

	router := &rlOnlineRouter{
		tokenizer:          tokenizerObj,
		prefixCacheIndexer: prefixcacheindexer.NewPrefixHashTable(),
	}
	klog.InfoS("Created RL online router")
	return router, nil
}

// func FlushLogMessageToRLAgent() {
// 	go func() {
// 		ticker := time.NewTicker(flushPeriod)
// 		defer ticker.Stop()
// 		for {
// 			select {
// 			case <-ticker.C:
// 				if len(utils.RequestToLogMessage) > 100 {
// 					utils.RequestToLogMessageMutex.Lock()
// 					klog.Infof("Starting flushing process for %d number of log messages", len(utils.RequestToLogMessage))
// 					reqBody, err := json.Marshal(utils.RequestToLogMessage)
// 					if err != nil {
// 						klog.Errorf("Failed to marshal RequestToLogMessage: %v", err)
// 						utils.RequestToLogMessageMutex.Unlock()
// 						utils.CleanupAllRequestLogMessage()
// 						continue
// 					}
// 					url := fmt.Sprintf("%s%s", routingAgentURL, flushEndpoint)
// 					req, reqErr := http.NewRequest("POST", url, bytes.NewBuffer(reqBody))
// 					if reqErr != nil {
// 						klog.Errorf("Failed to create request: %v", reqErr)
// 						utils.RequestToLogMessageMutex.Unlock()
// 						utils.CleanupAllRequestLogMessage()
// 						continue
// 					}

// 					req.Header.Set("Content-Type", "application/json")
// 					resp, sendErr := httpClientForRLAgent.Do(req)
// 					if sendErr != nil {
// 						klog.Errorf("Failed to send request: %v", sendErr)
// 						utils.RequestToLogMessageMutex.Unlock()
// 						utils.CleanupAllRequestLogMessage()
// 						continue
// 					}
// 					if resp.StatusCode != http.StatusOK {
// 						klog.Errorf("Received non-200 response: %s", resp.Status)
// 						utils.RequestToLogMessageMutex.Unlock()
// 						utils.CleanupAllRequestLogMessage()
// 						continue
// 					}
// 					body, readErr := ioutil.ReadAll(resp.Body)
// 					if readErr != nil {
// 						klog.Errorf("Failed to read response body: %v", readErr)
// 						utils.RequestToLogMessageMutex.Unlock()
// 						utils.CleanupAllRequestLogMessage()
// 						continue
// 					}
// 					klog.Infof("Successfully sent RequestToLogMessage to RL agent: %s", string(body))
// 					resp.Body.Close()
// 					utils.CleanupAllRequestLogMessage()
// 					utils.RequestToLogMessageMutex.Unlock()
// 				} else {
// 					klog.Infof("Not enough log messages to flush: %d", len(utils.RequestToLogMessage))
// 				}
// 			}
// 		}
// 	}()
// }

func FlushLogMessageToRLAgent() {
	if !flushed {
		klog.Infof("Sleep 10 seconds before flushing log messages")
		for i := 0; i < 10; i++ {
			klog.Infof("Sleeping %d seconds", i)
			time.Sleep(1 * time.Second)
		}
		for i := 0; i < 100; i++ {
			request_id := fmt.Sprintf("%d", i)
			utils.AddRequestLogMessage(request_id, logMessage)
		}

		klog.Infof("Starting flushing process for %d number of log messages", len(utils.RequestToLogMessage))
		reqBody, err := json.Marshal(utils.RequestToLogMessage)
		if err != nil {
			klog.Errorf("Failed to marshal RequestToLogMessage: %v", err)
		}

		url := fmt.Sprintf("%s%s", routingAgentURL, flushEndpoint)
		req, reqErr := http.NewRequest("POST", url, bytes.NewBuffer(reqBody))
		if reqErr != nil {
			klog.Errorf("Failed to create request: %v", reqErr)
		}

		req.Header.Set("Content-Type", "application/json")
		resp, sendErr := httpClientForRLAgent.Do(req)
		if sendErr != nil {
			klog.Errorf("Failed to send request: %v", sendErr)
		}

		if resp.StatusCode != http.StatusOK {
			klog.Errorf("Received non-200 response: %s", resp.Status)
		}

		body, readErr := ioutil.ReadAll(resp.Body)
		if readErr != nil {
			klog.Errorf("Failed to read response body: %v", readErr)
		}

		klog.Infof("Successfully sent RequestToLogMessage to RL agent: %s", string(body))
		resp.Body.Close()
	}
	flushed = true
}

func init() {
	RegisterDelayedConstructor("rl-online-router", NewRLOnlineRouter)
	FlushLogMessageToRLAgent()
}

// RouteResponse is received from the routing agent
type RouteResponse struct {
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

// Route selects the optimal pod based on latency predictions
func (r *rlOnlineRouter) Route(ctx *types.RoutingContext, pods types.PodList) (string, error) {
	// Get all ready pods
	readyPods := pods.All()
	if len(readyPods) == 0 {
		return "", fmt.Errorf("no ready pods available")
	}

	if len(readyPods) == 1 {
		ctx.SetTargetPod(readyPods[0])
		return ctx.TargetAddress(), nil
	}

	// var prefixHashes []uint64
	// var matchedPods map[string]int
	// readyPodsMap := map[string]struct{}{}
	// for _, pod := range readyPods {
	// 	readyPodsMap[pod.Status.PodIP] = struct{}{}
	// }

	// tokens, err := r.tokenizer.TokenizeInputText(ctx.Message)
	// if err != nil {
	// 	klog.Errorf("requestID: %s, Tokenization failed: %v", ctx.RequestID, err)
	// 	return "", err
	// }

	// numInputTokens := len(tokens)
	// numOutputTokens := 128 // Placeholder for output tokens
	// numTotalTokens := numInputTokens + numOutputTokens

	// matchedPods, prefixHashes = r.prefixCacheIndexer.MatchPrefix(tokens, ctx.Model, readyPodsMap)
	// utils.StoreKVCacheHitRatio(ctx.RequestID, matchedPods)

	// // Prepare for JSON strings to use in logging
	// var jsonStrings = make(map[string]string)

	// // 1. KV cache hit ratios
	// allPodsKvCacheHitRatios := utils.GetAllPodsKVCacheHitRatios(ctx.RequestID)
	// jsonStrings["allPodsKvCacheHitRatios"] = jsonStringify(allPodsKvCacheHitRatios, utils.GetrequestAllPodsKVCacheMutex())

	// // 2. Inflight requests
	// numInflightRequestsAllPods := utils.GetInflightRequestsForAllPods(ctx.RequestID)
	// jsonStrings["numInflightRequestsAllPods"] = jsonStringify(numInflightRequestsAllPods, utils.GetrequestInflightMutex())

	// // 3. GPU KV cache usage
	// vllmGPUKVCacheUsage, err := utils.GetvLLMGPUKVCacheUsageForAllPods(ctx.RequestID)
	// if err == nil {
	// 	jsonStrings["vllmGPUKVCacheUsage"] = jsonStringify(vllmGPUKVCacheUsage, utils.GetvllmGPUKVCacheUsageMutex())
	// } else {
	// 	jsonStrings["vllmGPUKVCacheUsage"] = "{}"
	// }

	// // 4. CPU KV cache usage
	// vllmCPUKVCacheUsage, err := utils.GetvLLMCPUKVCacheUsageForTheRequestForAllPods(ctx.RequestID)
	// if err == nil {
	// 	jsonStrings["vllmCPUKVCacheUsage"] = jsonStringify(vllmCPUKVCacheUsage, utils.GetvllmCPUKVCacheUsageMutex())
	// } else {
	// 	jsonStrings["vllmCPUKVCacheUsage"] = "{}"
	// }

	// // 5. Number of running requests
	// vllmNumRequestsRunning, err := utils.GetvLLMNumRequestsRunningForAllPods(ctx.RequestID)
	// if err == nil {
	// 	jsonStrings["vllmNumRequestsRunning"] = jsonStringify(vllmNumRequestsRunning, utils.GetvllmNumRequestsRunningMutex())
	// } else {
	// 	jsonStrings["vllmNumRequestsRunning"] = "{}"
	// }

	// // 6. Number of waiting requests
	// vllmNumRequestWaiting, err := utils.GetvLLMNumRequestsWaitingForAllPods(ctx.RequestID)
	// if err == nil {
	// 	jsonStrings["vllmNumRequestWaiting"] = jsonStringify(vllmNumRequestWaiting, utils.GetvllmNumRequestsWaitingMutex())
	// } else {
	// 	jsonStrings["vllmNumRequestWaiting"] = "{}"
	// }

	// numPrefillTokensForAllPods := utils.GetNumPrefillTokensForAllPods()
	// jsonStrings["numPrefillTokensForAllPods"] = jsonStringify(numPrefillTokensForAllPods, utils.GetpodTotalPrefillTokensMutex())

	// numDecodeTokensForAllPods := utils.GetNumDecodeTokensForAllPods()
	// jsonStrings["numDecodeTokensForAllPods"] = jsonStringify(numDecodeTokensForAllPods, utils.GetpodTotalDecodeTokensMutex())

	// podDetailedMetrics := utils.GetRequestPodMetrics(ctx.RequestID)
	// jsonStrings["podMetricsLastSecond"] = jsonStringify(podDetailedMetrics, utils.MetricsTracker.GetMutex())

	// klog.Infof("/infer: %s", logMessage)

	var targetPod *v1.Pod
	if flushed {
		reqBody, err := json.Marshal(logMessage)
		// reqBody, err := json.Marshal(infer_logMessage)
		if err != nil {
			klog.Errorf("Failed to marshal RequestToLogMessage: %v", err)
			targetPod, _ = r.fallbackRouting(ctx, readyPods)
			ctx.SetTargetPod(targetPod)
			return ctx.TargetAddress(), nil
		}
		url := fmt.Sprintf("%s%s", routingAgentURL, inferEndpoint)
		req, reqErr := http.NewRequest("POST", url, bytes.NewBuffer(reqBody))
		if reqErr != nil {
			klog.Errorf("Failed to create request: %v", reqErr)
		}
		req.Header.Set("Content-Type", "application/json")
		resp, sendErr := httpClientForRLAgent.Do(req)
		if sendErr != nil {
			klog.Errorf("Failed to send request: %v", sendErr)
		}
		if resp.StatusCode != http.StatusOK {
			klog.Errorf("Received non-200 response: %s", resp.Status)
		}
		body, readErr := ioutil.ReadAll(resp.Body)
		if readErr != nil {
			klog.Errorf("Failed to read response body: %v", readErr)
		}
		klog.Infof("Successfully infer targetpod: %s", string(body))
		resp.Body.Close()
	}

	/////////////////////////////////////////////////////////////

	targetPod, _ = r.fallbackRouting(ctx, readyPods)
	ctx.SetTargetPod(targetPod)

	// if len(prefixHashes) > 0 {
	// 	klog.Infof("Adding prefix hashes to cache. pod: %s", targetPod.Status.PodIP)
	// 	r.prefixCacheIndexer.AddPrefix(prefixHashes, ctx.Model, targetPod.Status.PodIP)
	// }

	return ctx.TargetAddress(), nil
}

func (r *rlOnlineRouter) fallbackRouting(ctx *types.RoutingContext, readyPods []*v1.Pod) (*v1.Pod, error) {
	klog.Infof("Using fallback routing for request %s", ctx.RequestID)
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
