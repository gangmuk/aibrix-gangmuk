package routingalgorithms

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
	"time"

	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
	"github.com/vllm-project/aibrix/pkg/utils/tokenizer"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

var (
	flushPeriod = 10 * time.Second

	httpClientForRLAgent = &http.Client{
		Timeout: 1000 * time.Millisecond,
		Transport: &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 100,
			IdleConnTimeout:     180 * time.Second,
			DisableCompression:  false,
			DialContext: (&net.Dialer{
				Timeout:   200 * time.Millisecond,
				KeepAlive: 30 * time.Second,
			}).DialContext,
			TLSHandshakeTimeout:   200 * time.Millisecond,
			ForceAttemptHTTP2:     true, // Enable HTTP/2
			ResponseHeaderTimeout: 1 * time.Second,
		},
	}
	routingAgentURL   = "http://routing-agent-service.default.svc.cluster.local:8080"
	inferenceEndpoint = "/inference"
	flushEndpoint     = "/flush"
)

type rlOnlineRouter struct {
	tokenizer tokenizer.Tokenizer
}

func NewRLOnlineRouter() (types.Router, error) {
	var tokenizerObj tokenizer.Tokenizer
	tokenizerObj = tokenizer.NewTiktokenTokenizer()

	router := &rlOnlineRouter{
		tokenizer: tokenizerObj,
	}
	klog.InfoS("Created RL online router")
	return router, nil
}

func FlushLogMessageToRLAgent() {
	go func() {
		ticker := time.NewTicker(flushPeriod)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				if len(utils.RequestToLogMessage) > 0 {
					utils.RequestToLogMessageMutex.Lock()
					klog.Infof("Starting flushing process for %d number of log messages", len(utils.RequestToLogMessage))
					reqBody, err := json.Marshal(utils.RequestToLogMessage)
					if err != nil {
						klog.Errorf("Failed to marshal RequestToLogMessage: %v", err)
						utils.RequestToLogMessageMutex.Unlock()
						utils.CleanupAllRequestLogMessage()
						continue
					}
					url := fmt.Sprintf("%s%s", routingAgentURL, flushEndpoint)
					req, reqErr := http.NewRequest("POST", url, bytes.NewBuffer(reqBody))
					if reqErr != nil {
						klog.Errorf("Failed to create request: %v", reqErr)
						utils.RequestToLogMessageMutex.Unlock()
						utils.CleanupAllRequestLogMessage()
						continue
					}

					req.Header.Set("Content-Type", "application/json")
					resp, sendErr := httpClientForRLAgent.Do(req)
					if sendErr != nil {
						klog.Errorf("Failed to send request: %v", sendErr)
						utils.RequestToLogMessageMutex.Unlock()
						utils.CleanupAllRequestLogMessage()
						continue
					}
					if resp.StatusCode != http.StatusOK {
						klog.Errorf("Received non-200 response: %s", resp.Status)
						utils.RequestToLogMessageMutex.Unlock()
						utils.CleanupAllRequestLogMessage()
						continue
					}
					body, readErr := ioutil.ReadAll(resp.Body)
					if readErr != nil {
						klog.Errorf("Failed to read response body: %v", readErr)
						utils.RequestToLogMessageMutex.Unlock()
						utils.CleanupAllRequestLogMessage()
						continue
					}
					klog.Infof("Successfully sent RequestToLogMessage to RL agent: %s", string(body))
					resp.Body.Close()
					utils.CleanupAllRequestLogMessage()
					utils.RequestToLogMessageMutex.Unlock()
				} else {
					klog.Infof("Nothing to flush to RL agent")
				}
			}
		}
	}()
}

func init() {
	RegisterDelayedConstructor("rl-online-router", NewRLOnlineRouter)
	FlushLogMessageToRLAgent()
}

// RouteRequest is sent to the routing agent for pod selection
type RouteRequest struct {
	RequestID    string                 `json:"request_id"`
	InputTokens  int                    `json:"input_tokens"`
	OutputTokens int                    `json:"output_tokens"`
	TotalTokens  int                    `json:"total_tokens"`
	Pods         map[string]PodFeatures `json:"pods"`
	Timestamp    int64                  `json:"timestamp"`
}

// RouteResponse is received from the routing agent
type RouteResponse struct {
	SelectedPod string  `json:"selected_pod"`
	Confidence  float64 `json:"confidence"`
}

// Route selects a pod for the incoming request
func (r *rlOnlineRouter) Route(ctx *types.RoutingContext, pods types.PodList) (string, error) {
	// Get all ready pods
	readyPods := pods.All()
	if len(readyPods) == 0 {
		return "", fmt.Errorf("no ready pods available")
	}

	// If only one pod is available, just return it
	if len(readyPods) == 1 {
		ctx.SetTargetPod(readyPods[0])
		return ctx.TargetAddress(), nil
	}
	targetPod, _ := r.fallbackRouting(ctx, readyPods)
	ctx.SetTargetPod(targetPod)
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
