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

package routingalgorithms

import (
	"math/rand"

	"github.com/vllm-project/aibrix/pkg/cache"
	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
	"github.com/vllm-project/aibrix/pkg/utils/prefixcacheindexer"
	"github.com/vllm-project/aibrix/pkg/utils/tokenizer"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

func init() {
	RegisterDelayedConstructor("flexible-prefix-cache", NewFlexiblePrefixCacheRouter)
}

type flexiblePrefixCacheRouter struct {
	cache              cache.Cache
	tokenizer          tokenizer.Tokenizer
	prefixCacheIndexer *prefixcacheindexer.PrefixHashTable
}

func NewFlexiblePrefixCacheRouter() (types.Router, error) {
	var tokenizerObj tokenizer.Tokenizer
	if tokenizerType == "tiktoken" {
		tokenizerObj = tokenizer.NewTiktokenTokenizer()
	} else {
		tokenizerObj = tokenizer.NewCharacterTokenizer()
	}

	c, err := cache.Get()
	if err != nil {
		klog.Error("fail to get cache store in prefix cache router")
		return nil, err
	}

	klog.InfoS("flexible_prefix_cache_configurations",
		"tokenizer_type", tokenizerType,
		"pod_running_request_imbalance_abs_count", podRunningRequestImbalanceAbsCount,
		"matched_pods_running_requests_standard_deviation_factor", standardDeviationFactor)

	return flexiblePrefixCacheRouter{
		cache:              c,
		tokenizer:          tokenizerObj,
		prefixCacheIndexer: prefixcacheindexer.NewPrefixHashTable(),
	}, nil
}

func (p flexiblePrefixCacheRouter) Route(ctx *types.RoutingContext, pods types.PodList) (string, error) {
	var prefixHashes []uint64
	var matchedPods map[string]int
	var targetPod *v1.Pod

	tokens, err := p.tokenizer.TokenizeInputText(ctx.Message)
	if err != nil {
		return "", err
	}

	readyPods := pods.All()
	readyPodsMap := map[string]struct{}{}
	for _, pod := range readyPods {
		readyPodsMap[pod.Status.PodIP] = struct{}{}
	}

	matchedPods, prefixHashes = p.prefixCacheIndexer.MatchPrefix(tokens, ctx.Model, readyPodsMap)
	klog.InfoS("matched_pods", "request_id", ctx.RequestID, "matched_pods", matchedPods)
	klog.InfoS("prefix_hashes", "request_id", ctx.RequestID, "prefix_hashes", prefixHashes)
	utils.StoreKVCacheHitRatio(ctx.RequestID, matchedPods)

	targetPod, err = selectRandomPod(pods.All(), rand.Intn)
	if err != nil {
		klog.Errorf("error to select target pod: %v, requestID: %s", err, ctx.RequestID)
		return "", err
	}
	klog.InfoS("Random routing", "request_id", ctx.RequestID, "target_pod", targetPod.Name)

	if len(prefixHashes) > 0 {
		p.prefixCacheIndexer.AddPrefix(prefixHashes, ctx.Model, targetPod.Name)
	}

	ctx.SetTargetPod(targetPod)
	return ctx.TargetAddress(), nil
}
