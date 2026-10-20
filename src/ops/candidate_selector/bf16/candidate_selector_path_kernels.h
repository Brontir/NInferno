#pragma once
#include "ops/candidate_selector/bf16/candidate_selector_path_plan.h"

namespace ninferno::ops::detail {
void candidate_selector_path_launch(SelectorRoute route, const Tensor& candidate_ids,
                                    const Tensor& unary_scores, const Tensor& projected_hidden,
                                    const Tensor& anchors,
                                    const SelectorCodebook& predecessor_codebook,
                                    const SelectorCodebook& successor_codebook,
                                    const Tensor& base_positions,
                                    const SamplingConfig* configs, Tensor& drafts,
                                    Tensor& proposal_q, const SelectorWorkspace& workspace,
                                    cudaStream_t stream);
}
