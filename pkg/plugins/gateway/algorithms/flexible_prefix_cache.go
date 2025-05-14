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

// const (
// 	defaultSubRoutingStrategy = "random"
// )

// var (
// 	subRoutingStrategy = utils.LoadEnv("AIBRIX_FLEXIBLE_PREFIX_CACHE_ROUTING_STRATEGY", defaultSubRoutingStrategy)
// )

func init() {
	RegisterDelayedConstructor("flexible-prefix-cache", NewFlexiblePrefixCacheRouter)
	klog.InfoS("flexible_prefix_cache_configurations",
		// "subRoutingStrategy", subRoutingStrategy,
		"pod_running_request_imbalance_abs_count", podRunningRequestImbalanceAbsCount,
		"matched_pods_running_requests_standard_deviation_factor", standardDeviationFactor)
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

	if ctx.SubAlgorithm == "random" {
		targetPod, _ = selectRandomPod(readyPods, rand.Intn)
		klog.Infof("random rouiting, request_id: %s, selectedPod: %s", ctx.RequestID, targetPod.Status.PodIP)
	} else if ctx.SubAlgorithm == "prefix-cache" {
		var isLoadImbalanced bool
		targetPod, isLoadImbalanced = getTargetPodOnLoadImbalance(p.cache, readyPods)
		if !isLoadImbalanced {
			if len(matchedPods) > 0 {
				targetPod = getTargetPodFromMatchedPods(p.cache, readyPods, matchedPods)
				klog.Infof("prefix routing - prefix routing, request_id: %s, selectedPod: %ss", ctx.RequestID, targetPod.Status.PodIP)
			}
		}
		if len(matchedPods) == 0 || targetPod == nil {
			targetPod = selectTargetPodWithLeastRequestCount(p.cache, readyPods)
			klog.Infof("prefix routing - least request count routing, request_id: %s, selectedPod: %s", ctx.RequestID, targetPod.Status.PodIP)
		}
	}

	// klog.InfoS("request_id", ctx.RequestID, "prefix_hashes", prefixHashes)
	utils.StoreKVCacheHitRatio(ctx.RequestID, matchedPods)

	// if len(prefixHashes) > 0 {
	p.prefixCacheIndexer.AddPrefix(prefixHashes, ctx.Model, targetPod.Status.PodIP)
	// }

	ctx.SetTargetPod(targetPod)
	return ctx.TargetAddress(), nil
}
