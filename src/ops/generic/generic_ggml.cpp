// The generic path's calls of ggml's matrix and vector kernels: the formats they read, their
// workspaces, and the gated pair on one activation (generic_gemm.h).

#include "core/layout.h"
#include "core/tensor.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_ggml_internal.h"
#include "ops/generic/ggml_mmq/ggml_mmq.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ninferno::ops::detail::generic {

bool ggml_mmq_call(const Weight& w, LinearPolicy policy, std::int32_t rows, std::int32_t tokens) {
    return mmq_call(w, policy, rows, tokens);
}

bool ggml_mmq_format(QType qtype) {
    Weight probe;
    probe.qtype     = qtype;
    const auto type = mmq_type(probe);
    return type && ggml_mmq::matrix_supported(*type);
}

std::size_t ggml_mmq_workspace_bytes(std::int32_t k, std::int32_t plane_rows, std::int32_t tokens) {
    return mmq_workspace_bytes(k, plane_rows, tokens);
}

bool ggml_vec_format(QType qtype) {
    Weight probe;
    probe.qtype     = qtype;
    const auto type = mmq_type(probe);
    return type && ggml_mmq::vec_supported(*type);
}

std::size_t ggml_vec_workspace_bytes(std::int32_t k, std::int32_t tokens) {
    return vec_workspace_bytes(k, tokens);
}

static_assert(kGgmlVecMaxTokens == ggml_mmq::kVecMaxColumns);

bool ggml_vec_takes(const Problem& problem, const Weight& weight, std::int32_t tokens) {
    // The callers share one quantized activation of the call's tokens (kGgmlVecMaxTokens at most).
    return tokens <= kGgmlVecMaxTokens && vec_call(weight, problem.policy, tokens) &&
           select_route(problem, weight, tokens, prefill_routes_active()).family == Family::GgmlVec;
}

std::size_t ggml_vec_activation_bytes(std::int32_t k, std::int32_t tokens) {
    return ggml_mmq::vec_activation_bytes(k, tokens);
}

static_assert(kGgmlMmqMaxTokens == kMmqMaxTokens);

bool ggml_mmq_takes_tuned(const Tensor& x, const Weight& weight, LinearPolicy policy,
                          std::int32_t row, std::int32_t rows, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (!mmq_call(weight, policy, rows, tokens)) return false;
    const Problem problem{weight.qtype, policy, rows, weight.k, Epilogue::Store};
    Call call;
    call.x         = &x;
    call.weight    = &weight;
    call.row_begin = row;
    call.rows      = rows;
    call.epilogue  = Epilogue::Store;
    tune_dense(problem, call, stream);
    return select_route(problem, weight, tokens, prefill_routes_active()).family == Family::GgmlMmq;
}

std::size_t ggml_mmq_gated_pair_bytes(std::int32_t k, std::int32_t rows, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    // Two activations where the halves' formats read different q8_1 layouts.
    (void)layout.alloc_bytes(ggml_mmq::activation_bytes(k, tokens), 256);
    (void)layout.alloc_bytes(ggml_mmq::activation_bytes(k, tokens), 256);
    (void)layout.alloc_bytes(static_cast<std::size_t>(2) * rows * tokens * sizeof(float), 256);
    (void)layout.alloc_bytes(ggml_mmq::fixup_bound(tokens), 256);
    return layout.peak_bytes(1) + kGgmlArenaSlack;
}

std::int32_t ggml_mmq_slice_tokens(QType qtype) {
    return ggml_mmq_format(qtype) ? kMmqPrefillSlice : kMmqMaxTokens;
}

// FUSION-SITE: mmq_gated_pair
bool ggml_mmq_gated_pair(const Tensor& x, const PairHalf& gate, const PairHalf& up,
                         std::int32_t rows, bool gelu, Tensor& out, WorkspaceArena& workspace,
                         cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    const std::int32_t k      = gate.weight->k;
    if (up.weight->k != k || x.ne[0] != k || tokens < 1 || out.ne[0] < rows ||
        out.ne[1] != tokens || x.dtype != DType::BF16 || out.dtype != DType::BF16 ||
        !x.is_contiguous() || !out.is_contiguous()) {
        return false;
    }
    // A prefill takes the pair where a call of each half alone takes the kernel (mmq_prefill_wins
    // and the prefill route tables, as select_route decides it); the band where the tuner says.
    const auto takes = [&](const PairHalf& h) {
        if (tokens <= kMmqMaxTokens) {
            return ggml_mmq_takes_tuned(x, *h.weight, h.policy, h.row, rows, stream);
        }
        const Problem problem{h.weight->qtype, h.policy, rows, k, Epilogue::Store};
        return (rows % ggml_mmq::kRowMultiple) == 0 && mmq_admits(*h.weight, h.policy) &&
               select_route(problem, *h.weight, tokens, prefill_routes_active()).family ==
                   Family::GgmlMmq;
    };
    if (!takes(gate) || !takes(up)) return false;
    const std::int32_t slice = std::min(tokens, kMmqPrefillSlice);
    if (workspace.used() + ggml_mmq_gated_pair_bytes(k, rows, slice) > workspace.capacity()) {
        return false;
    }
    auto scope            = workspace.scope();
    const auto gate_type  = *mmq_type(*gate.weight);
    const auto up_type    = *mmq_type(*up.weight);
    const bool two        = ggml_mmq::activation_layout(up_type) !=
                     ggml_mmq::activation_layout(gate_type);
    void* gate_activation = workspace.alloc_bytes(ggml_mmq::activation_bytes(k, slice)).data;
    void* up_activation =
        two ? workspace.alloc_bytes(ggml_mmq::activation_bytes(k, slice)).data : gate_activation;
    // [gate; up] rows of one FP32 plane, the gated product in the epilogue.
    const int plane_rows = 2 * rows;
    auto* plane          = static_cast<float*>(
        workspace.alloc_bytes(static_cast<std::size_t>(plane_rows) * slice * sizeof(float)).data);
    void* fixup = workspace.alloc_bytes(ggml_mmq::fixup_bound(slice)).data;
    for (std::int32_t t0 = 0; t0 < tokens; t0 += slice) {
        const std::int32_t t = std::min(slice, tokens - t0);
        const auto* xb       = static_cast<const __nv_bfloat16*>(x.data) +
                         static_cast<std::size_t>(t0) * static_cast<std::size_t>(k);
        ggml_mmq::quantize_activation(gate_type, xb, k, k, t, gate_activation, stream);
        if (two) ggml_mmq::quantize_activation(up_type, xb, k, k, t, up_activation, stream);
        const auto product = [&](const PairHalf& h, ggml_mmq::Type type, const void* activation,
                                 float* to) {
            const int bpr = h.weight->k / static_cast<int>(ggml_block_values(h.weight->qtype));
            ggml_mmq::product(type,
                              ggml_rows_at(weight_args(*h.weight), h.weight->qtype, bpr, h.row),
                              bpr, rows, k, activation, t, to, plane_rows, fixup, stream);
        };
        product(gate, gate_type, gate_activation, plane);
        product(up, up_type, up_activation, plane + rows);
        GemmArgs args;
        args.tokens         = t;
        args.rows           = rows;
        args.gelu_gate      = gelu ? 1 : 0;
        args.segments       = 1;
        args.out[0]         = static_cast<__nv_bfloat16*>(out.data) +
                      static_cast<std::size_t>(t0) * static_cast<std::size_t>(out.ne[0]);
        args.segment_end[0] = rows;
        args.ld[0]          = static_cast<int>(out.ne[0]);
        launch_mmq_epilogue(Epilogue::SwiGlu, args, plane, plane_rows, stream);
    }
    return true;
}

bool ggml_vec_takes_tuned(const Tensor& x, const Weight& weight, LinearPolicy policy,
                          std::int32_t row, std::int32_t rows, cudaStream_t stream) {
    const Problem problem{weight.qtype, policy, rows, weight.k, Epilogue::Store};
    if (!ggml_vec_takes(problem, weight, x.ne[1])) {
        // The tuner may still put the kernel in for the heuristic's route; time the call first.
        if (!vec_admits(weight, policy) || x.ne[1] < 1 || x.ne[1] > kGgmlVecMaxTokens) {
            return false;
        }
    }
    Call call;
    call.x         = &x;
    call.weight    = &weight;
    call.row_begin = row;
    call.rows      = rows;
    call.epilogue  = Epilogue::Store;
    tune_dense(problem, call, stream);
    return ggml_vec_takes(problem, weight, x.ne[1]);
}

bool ggml_vec_gated_pair(const Tensor& x, const PairHalf& gate, const PairHalf& up,
                         std::int32_t rows, bool gelu, Tensor& out, WorkspaceArena& workspace,
                         const PreparedActivation* prepared, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    const std::int32_t k      = gate.weight->k;
    // The output is written at a stride of out.ne[0]: a contiguous [>= rows, T].
    if (up.weight->k != k || x.ne[0] != k || tokens < 1 || tokens > kGgmlVecMaxTokens ||
        out.ne[0] < rows || out.ne[1] != tokens || x.dtype != DType::BF16 ||
        out.dtype != DType::BF16 || !x.is_contiguous() || !out.is_contiguous()) {
        return false;
    }
    // Each half's route as a call of it alone gets it: the tuner times the problem first.
    const auto takes = [&](const PairHalf& h) {
        return ggml_vec_takes_tuned(x, *h.weight, h.policy, h.row, rows, stream);
    };
    if (!takes(gate) || !takes(up)) return false;
    if (workspace.used() + ggml_vec_gated_pair_bytes(k, rows, tokens) > workspace.capacity()) {
        return false;
    }
    auto scope             = workspace.scope();
    const void* activation = nullptr;
    if (prepared_serves(prepared, PreparedRepresentation::GgmlQ8_1, x.data,
                        static_cast<std::int32_t>(x.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16))),
                        k, tokens)) {
        activation = prepared->data;
    } else {
        void* own = workspace.alloc_bytes(ggml_mmq::vec_activation_bytes(k, tokens)).data;
        (void)ggml_vec_quantize(x, own, stream);
        activation = own;
    }
    const auto rows_at = [](const PairHalf& h) {
        const int bpr = h.weight->k / static_cast<int>(ggml_block_values(h.weight->qtype));
        return ggml_rows_at(weight_args(*h.weight), h.weight->qtype, bpr, h.row);
    };
    const auto bpr = [](const PairHalf& h) {
        return h.weight->k / static_cast<int>(ggml_block_values(h.weight->qtype));
    };
    ggml_mmq::VecOut o;
    o.gelu_gate      = gelu ? 1 : 0;
    o.segments       = 1;
    o.out[0]         = out.data;
    o.ld[0]          = out.ne[0];
    o.segment_end[0] = rows;
    if (gate.weight->qtype == up.weight->qtype) {
        o.epilogue = ggml_mmq::VecOut::Epilogue::SwiGlu;
        ggml_mmq::vec_product(*mmq_type(*gate.weight), rows_at(gate), rows_at(up), bpr(gate), rows,
                              k, activation, tokens, o, stream);
        return true;
    }
    auto* plane = static_cast<float*>(
        workspace.alloc_bytes(static_cast<std::size_t>(rows) * tokens * sizeof(float)).data);
    ggml_mmq::VecOut g = o;
    g.epilogue         = ggml_mmq::VecOut::Epilogue::StoreF32;
    g.f32              = plane;
    g.f32_ld           = rows;
    ggml_mmq::vec_product(*mmq_type(*gate.weight), rows_at(gate), ggml_mmq::Planes{}, bpr(gate),
                          rows, k, activation, tokens, g, stream);
    o.epilogue = ggml_mmq::VecOut::Epilogue::GateProduct;
    o.gate     = plane;
    o.gate_ld  = rows;
    ggml_mmq::vec_product(*mmq_type(*up.weight), rows_at(up), ggml_mmq::Planes{}, bpr(up), rows, k,
                          activation, tokens, o, stream);
    return true;
}

std::size_t ggml_vec_gated_pair_bytes(std::int32_t k, std::int32_t rows, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc_bytes(ggml_mmq::vec_activation_bytes(k, tokens), 256);
    (void)layout.alloc_bytes(static_cast<std::size_t>(rows) * tokens * sizeof(float), 256);
    return layout.peak_bytes(1) + kGgmlArenaSlack;
}

PreparedActivation ggml_vec_quantize(const Tensor& x, void* out, cudaStream_t stream) {
    const auto ldx = static_cast<std::int32_t>(x.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)));
    ggml_mmq::vec_quantize_activation(static_cast<const __nv_bfloat16*>(x.data), ldx, x.ne[0],
                                      x.ne[1], out, stream);
    return PreparedActivation{.representation = PreparedRepresentation::GgmlQ8_1,
                              .data           = out,
                              .source         = x.data,
                              .ldx            = ldx,
                              .k              = static_cast<std::int32_t>(x.ne[0]),
                              .tokens         = static_cast<std::int32_t>(x.ne[1])};
}

} // namespace ninferno::ops::detail::generic
