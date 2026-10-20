#include "ninferno/ops/candidate_selector.h"

#include "core/weight.h"
#include "ops/candidate_selector/bf16/candidate_selector_path_plan.h"
#include "ops/common/tensor_checks.h"
#include "ops/generic/generic_gemm.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninferno::ops {
namespace {

constexpr std::int32_t kCandidates   = 16;
constexpr std::int32_t kRank         = 256;
constexpr std::int32_t kCodebookRows = 248320;

void require_tensor(const Tensor& tensor, DType dtype, std::int32_t d0, std::int32_t d1,
                    std::int32_t d2, std::int32_t d3, const char* label) {
    if (tensor.dtype != dtype || tensor.ne[0] != d0 || tensor.ne[1] != d1 || tensor.ne[2] != d2 ||
        tensor.ne[3] != d3 || !tensor.is_contiguous() || !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("candidate_selector_path: invalid ") + label);
    }
}

void require_nonoverlap(const Tensor& candidate_ids, const Tensor& unary_scores,
                        const Tensor& projected_hidden, const Tensor& anchors,
                        const detail::SelectorCodebook& predecessor_codebook,
                        const detail::SelectorCodebook& successor_codebook,
                        const Tensor& base_positions, const SamplingConfig* configs,
                        const Tensor& drafts, const Tensor& proposal_q) {
    const std::array<OperandRange, 10> ranges{{
        {candidate_ids.data, candidate_ids.bytes(), "candidate_ids"},
        {unary_scores.data, unary_scores.bytes(), "unary_scores"},
        {projected_hidden.data, projected_hidden.bytes(), "projected_hidden"},
        {anchors.data, anchors.bytes(), "anchors"},
        {predecessor_codebook.begin, predecessor_codebook.bytes, "predecessor_codebook"},
        {successor_codebook.begin, successor_codebook.bytes, "successor_codebook"},
        {base_positions.data, base_positions.bytes(), "base_positions"},
        {configs, static_cast<std::size_t>(candidate_ids.ne[2]) * sizeof(SamplingConfig),
         "configs"},
        {drafts.data, drafts.bytes(), "drafts"},
        {proposal_q.data, proposal_q.bytes(), "proposal_q"},
    }};
    // The two codebooks are read-only and may share storage.
    constexpr std::pair<std::size_t, std::size_t> kCodebooks[] = {{4, 5}};
    require_disjoint(ranges, "candidate_selector_path", kCodebooks);
}

} // namespace

std::size_t candidate_selector_path_workspace_capacity_bytes(int min_steps, int max_steps,
                                                             int min_batch, int max_batch) {
    if (min_steps < 1 || max_steps > 15 || max_steps < min_steps || min_batch < 1 ||
        max_batch > 8 || max_batch < min_batch)
        throw std::invalid_argument("selector workspace: invalid K/B interval");
    WorkspaceLayoutBuilder layout;
    for (int k = min_steps; k <= max_steps; ++k)
        for (int b = min_batch; b <= max_batch; ++b) {
            auto scope = layout.scope();
            (void)detail::allocate_selector_workspace(
                layout, detail::candidate_selector_path_route(k, b), k, b);
        }
    return layout.peak_bytes();
}

namespace {

void run(const Tensor& candidate_ids, const Tensor& unary_scores, const Tensor& projected_hidden,
         const Tensor& anchors, const detail::SelectorCodebook& predecessor_codebook,
         const detail::SelectorCodebook& successor_codebook, const Tensor& base_positions,
         const SamplingConfig* configs, Tensor& drafts, Tensor& proposal_q,
         WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t batch_size = candidate_ids.ne[2];
    const std::int32_t kSteps     = candidate_ids.ne[1];
    if (kSteps < 1 || kSteps > 15)
        throw std::invalid_argument("candidate_selector_path: K must be in [1,15]");
    if (batch_size < 1 || batch_size > 8) {
        throw std::invalid_argument("candidate_selector_path: B must be in [1,8]");
    }
    require_tensor(candidate_ids, DType::I32, kCandidates, kSteps, batch_size, 1, "candidate_ids");
    require_tensor(unary_scores, DType::FP32, kCandidates, kSteps, batch_size, 1, "unary_scores");
    require_tensor(projected_hidden, DType::BF16, kRank, kSteps, batch_size, 1, "projected_hidden");
    require_tensor(anchors, DType::I32, batch_size, 1, 1, 1, "anchors");
    require_tensor(base_positions, DType::I32, batch_size, 1, 1, 1, "base_positions");
    require_tensor(drafts, DType::I32, kSteps, batch_size, 1, 1, "drafts");
    require_tensor(proposal_q, DType::FP32, kCandidates, kSteps, batch_size, 1, "proposal_q");
    if (!aligned_to(configs, alignof(SamplingConfig))) {
        throw std::invalid_argument("candidate_selector_path: invalid configs");
    }
    require_nonoverlap(candidate_ids, unary_scores, projected_hidden, anchors, predecessor_codebook,
                       successor_codebook, base_positions, configs, drafts, proposal_q);

    detail::candidate_selector_path_dispatch(
        candidate_ids, unary_scores, projected_hidden, anchors, predecessor_codebook,
        successor_codebook, base_positions, configs, drafts, proposal_q, workspace, stream);
}

detail::SelectorCodebook bf16_codebook(const Tensor& tensor, const char* label) {
    require_tensor(tensor, DType::BF16, kRank, kCodebookRows, 1, 1, label);
    detail::SelectorCodebook out;
    out.bf16  = tensor.data;
    out.begin = tensor.data;
    out.bytes = tensor.bytes();
    return out;
}

// A codebook Weight [248320,256]: contiguous BF16 keeps the BF16 kernels; any other format the
// generic decode reads is read element by element as its FP32 dequantized value.
detail::SelectorCodebook weight_codebook(const Weight& weight, const char* label) {
    if (weight.ndim != 2 || weight.n != kCodebookRows || weight.k != kRank ||
        weight.payload == nullptr) {
        throw std::invalid_argument(std::string("candidate_selector_path: invalid ") + label);
    }
    detail::SelectorCodebook out;
    out.begin = weight.payload;
    out.bytes = weight.payload_bytes;
    if (weight.qtype == QType::BF16 && weight.layout == QuantLayout::Contiguous) {
        if (!aligned_to(weight.qdata, 16)) {
            throw std::invalid_argument(std::string("candidate_selector_path: invalid ") + label);
        }
        out.bf16 = weight.qdata;
        return out;
    }
    if (!detail::generic::supports(weight)) {
        throw std::invalid_argument(std::string("candidate_selector_path: ") + label +
                                    ": no decode reads its format");
    }
    out.generic = true;
    out.args    = detail::generic::weight_args(weight);
    out.codec   = detail::generic::codec_of(weight);
    return out;
}

} // namespace

void candidate_selector_path(const Tensor& candidate_ids, const Tensor& unary_scores,
                             const Tensor& projected_hidden, const Tensor& anchors,
                             const Tensor& predecessor_codebook, const Tensor& successor_codebook,
                             const Tensor& base_positions, const SamplingConfig* configs,
                             Tensor& drafts, Tensor& proposal_q, WorkspaceArena& workspace,
                             cudaStream_t stream) {
    run(candidate_ids, unary_scores, projected_hidden, anchors,
        bf16_codebook(predecessor_codebook, "predecessor_codebook"),
        bf16_codebook(successor_codebook, "successor_codebook"), base_positions, configs, drafts,
        proposal_q, workspace, stream);
}

void candidate_selector_path(const Tensor& candidate_ids, const Tensor& unary_scores,
                             const Tensor& projected_hidden, const Tensor& anchors,
                             const Weight& predecessor_codebook, const Weight& successor_codebook,
                             const Tensor& base_positions, const SamplingConfig* configs,
                             Tensor& drafts, Tensor& proposal_q, WorkspaceArena& workspace,
                             cudaStream_t stream) {
    const auto predecessor = weight_codebook(predecessor_codebook, "predecessor_codebook");
    const auto successor   = weight_codebook(successor_codebook, "successor_codebook");
    if (predecessor.generic != successor.generic) {
        throw std::invalid_argument("candidate_selector_path: one codebook BF16, the other not");
    }
    run(candidate_ids, unary_scores, projected_hidden, anchors, predecessor, successor,
        base_positions, configs, drafts, proposal_q, workspace, stream);
}

} // namespace ninferno::ops
