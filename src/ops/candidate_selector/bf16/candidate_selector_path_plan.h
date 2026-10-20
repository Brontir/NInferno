#pragma once
#include "core/tensor.h"
#include "core/layout.h"
#include "core/arena.h"
#include "ninferno/ops/sampling.h"
#include "ops/generic/generic_gemm.h"
#include <cuda_runtime.h>

namespace ninferno::ops::detail {
enum class SelectorRoute { Direct, Lattice };

struct SelectorWorkspace {
    Tensor edges;
};

// One codebook [248320 tokens, 256 rank] as the kernels read it: contiguous BF16 rows, or any
// matrix format the generic decode reads (element (token, r) is its FP32 dequantized value).
struct SelectorCodebook {
    const void* bf16 = nullptr;
    bool generic     = false;
    generic::GemmArgs args;
    generic::WeightCodec codec = generic::WeightCodec::Bf16;
    const void* begin          = nullptr; // bytes it occupies, for the overlap checks
    std::size_t bytes          = 0;
};

SelectorRoute candidate_selector_path_route(int steps, int batch);
const char* selector_route_name(SelectorRoute route);

template <class Allocator>
SelectorWorkspace allocate_selector_workspace(Allocator& allocator, SelectorRoute route, int steps,
                                              int batch) {
    SelectorWorkspace out;
    if (route == SelectorRoute::Lattice)
        out.edges = allocator.alloc(DType::FP32, {16, 16, steps, batch}, 256);
    return out;
}

void candidate_selector_path_dispatch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                      const Tensor& projected_hidden, const Tensor& anchors,
                                      const SelectorCodebook& predecessor_codebook,
                                      const SelectorCodebook& successor_codebook,
                                      const Tensor& base_positions, const SamplingConfig* configs,
                                      Tensor& drafts, Tensor& proposal_q, WorkspaceArena& workspace,
                                      cudaStream_t stream);
} // namespace ninferno::ops::detail
