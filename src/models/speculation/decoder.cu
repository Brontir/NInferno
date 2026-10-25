#include "models/speculation/decoder.h"

#include "ninferno/ops/argmax.h"
#include "ninferno/ops/scalar.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace ninferno::models::speculation {
namespace {

// Greedy acceptance: the longest drafted prefix the target's own argmax reproduces, then the
// target's token at the first disagreement (or after the last draft). Emits those tokens, carries
// the target's final-normed state of the column that predicted the last one (width 0: no carry),
// and advances.
__global__ void spec_accept(const std::int32_t* draft_ids, const std::int32_t* target_ids,
                            std::int32_t k, const __nv_bfloat16* hidden, std::int32_t width,
                            __nv_bfloat16* carry, std::int32_t* token, std::int32_t* position,
                            std::int32_t* out, std::int32_t* step, long long* stats) {
    __shared__ std::int32_t accepted;
    if (threadIdx.x == 0) {
        std::int32_t n = 0;
        while (n < k && draft_ids[n + 1] == target_ids[n]) { ++n; }
        accepted             = n;
        const std::int32_t s = *step;
        for (std::int32_t i = 0; i < n; ++i) { out[s + i] = draft_ids[i + 1]; }
        out[s + n] = target_ids[n];
        *step      = s + n + 1;
        *token     = target_ids[n];
        *position += n + 1;
        stats[0] += 1;
        stats[1] += k;
        stats[2] += n;
    }
    __syncthreads();
    const __nv_bfloat16* column = hidden + static_cast<std::int64_t>(accepted) * width;
    for (std::int32_t i = threadIdx.x; i < width; i += blockDim.x) { carry[i] = column[i]; }
}

} // namespace

Decoder::Decoder(DraftTarget& target, DeviceContext& device, DeviceArena& work,
                 const DecoderOptions& options, DecodeState state, DraftRound& round)
    : target_(target), device_(device), work_(work), round_(round),
      max_context_(options.max_context), vocab_size_(target.shape().vocab_size),
      hidden_size_(target.shape().hidden_size), deepest_(options.draft_tokens), state_(state) {
    if (deepest_ < 1) { throw std::invalid_argument("speculation: draft_tokens must be positive"); }
    verify_positions_ = DeviceBuffer(sizeof(std::int32_t) * (deepest_ + 1));
    target_ids_       = DeviceBuffer(sizeof(std::int32_t) * (deepest_ + 1));
    stats_            = DeviceBuffer(sizeof(long long) * 3);
    stats_.fill(0);
}

Decoder::PhaseEvents::PhaseEvents() {
    for (auto& event : e) { CUDA_CHECK(cudaEventCreate(&event)); }
}

Decoder::PhaseEvents::~PhaseEvents() {
    for (auto& event : e) {
        if (event != nullptr) { cudaEventDestroy(event); }
    }
}

void Decoder::mark(int phase) {
    if (!marking_) { return; }
    CUDA_CHECK(cudaEventRecordWithFlags(events_->e[phase], device_.stream,
                                        cudaEventRecordExternal));
}

void Decoder::round(bool accept, std::int32_t k) {
    const cudaStream_t s   = device_.stream;
    auto* verify_positions = static_cast<std::int32_t*>(verify_positions_.p);
    mark(0);
    round_.draft(state_.position, state_.token, verify_positions, k);
    mark(1);
    auto scope = work_.scope();
    Tensor ids = round_.ids(k + 1);
    Tensor pos(verify_positions, DType::I32, {k + 1});
    const ops::Gemma4AttentionEnvelope envelope{static_cast<std::uint32_t>(max_context_), k + 1};
    const std::optional<FeatureTap> tap = round_.tap(k + 1);
    Tensor normed = target_.draft_forward(ids, pos, envelope, /*a16=*/true, tap ? &*tap : nullptr);
    mark(2);
    Tensor logits = work_.alloc(DType::BF16, {vocab_size_, k + 1});
    target_.greedy_logits(normed, logits, /*a16=*/true);
    Tensor best(target_ids_.p, DType::I32, {k + 1});
    ops::argmax(logits, best, vocab_size_, s);
    mark(3);
    if (!accept) { return; }
    // Every verified column's features enter the drafter's context. Columns past the accepted
    // prefix hold rejected tokens; the next round's block overwrites exactly those positions
    // before it attends, so no round reads them as context.
    if (tap) { round_.commit(*tap, pos); }
    const Tensor carry = round_.carry();
    spec_accept<<<1, 256, 0, s>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const std::int32_t*>(target_ids_.p),
        k, static_cast<const __nv_bfloat16*>(normed.data),
        carry.data != nullptr ? hidden_size_ : 0, static_cast<__nv_bfloat16*>(carry.data),
        state_.token, state_.position, state_.out_tokens, state_.step,
        static_cast<long long*>(stats_.p));
    CUDA_CHECK(cudaGetLastError());
    mark(4);
}

DecodeGraphExecutable& Decoder::round_graph(std::int32_t k, bool profiled) {
    auto& graphs = profiled ? profiled_graphs_ : round_graphs_;
    if (graphs.size() <= static_cast<std::size_t>(k)) { graphs.resize(k + 1); }
    if (graphs[k].ready()) { return graphs[k]; }
    const cudaStream_t s = device_.stream;
    device_.synchronize();
    work_.reset();
    round(false, k); // eager warm-up, no state change
    device_.synchronize();
    work_.reset();
    if (profiled && !events_) { events_.emplace(); }
    marking_ = profiled;
    DecodeGraphDefinition definition;
    try {
        definition.capture(s, [&] { round(true, k); });
    } catch (...) {
        marking_ = false;
        throw;
    }
    marking_ = false;
    graphs[k].instantiate(definition);
    work_.reset();
    return graphs[k];
}

void Decoder::capture_rounds() {
    std::int32_t first = deepest_;
    // Defined inputs for the warm-up rounds: the prefill writes the real carried state.
    if (const Tensor carried = round_.carry(); carried.data != nullptr) {
        CUDA_CHECK(cudaMemsetAsync(carried.data, 0, carried.bytes(), device_.stream));
    }
    // A staged round verifies any draft count up to the deepest (its staged drafts start zeroed).
    if (round_.staged()) { first = 1; }
    for (std::int32_t k = first; k <= deepest_; ++k) { (void)round_graph(k); }
}

SpeculationRoundTrace Decoder::read_round(std::int32_t position, std::int32_t k, bool profiled) {
    SpeculationRoundTrace out;
    out.position       = position;
    out.depth          = k;
    const auto drafted = round_.drafted(k);
    std::vector<std::int32_t> ids(drafted + 1);
    const Tensor all = round_.ids(drafted + 1);
    CUDA_CHECK(cudaMemcpy(ids.data(), all.data, sizeof(std::int32_t) * ids.size(),
                          cudaMemcpyDeviceToHost));
    out.drafts.assign(ids.begin() + 1, ids.end());
    out.target.resize(k + 1);
    CUDA_CHECK(cudaMemcpy(out.target.data(), target_ids_.p, sizeof(std::int32_t) * (k + 1),
                          cudaMemcpyDeviceToHost));
    out.accepted = 0;
    while (out.accepted < k && out.drafts[out.accepted] == out.target[out.accepted]) {
        ++out.accepted;
    }
    if (profiled) {
        float* phase[4] = {&out.draft_ms, &out.forward_ms, &out.head_ms, &out.commit_ms};
        for (int i = 0; i < 4; ++i) {
            CUDA_CHECK(cudaEventElapsedTime(phase[i], events_->e[i], events_->e[i + 1]));
        }
    }
    return out;
}

std::vector<std::int32_t> Decoder::collect(std::int32_t& position) {
    const cudaStream_t s = device_.stream;
    std::int32_t emitted = 0, next_position = 0;
    CUDA_CHECK(
        cudaMemcpyAsync(&emitted, state_.step, sizeof(std::int32_t), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(&next_position, state_.position, sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, s));
    device_.synchronize();
    std::vector<std::int32_t> out(emitted);
    CUDA_CHECK(cudaMemcpy(out.data(), state_.out_tokens, sizeof(std::int32_t) * emitted,
                          cudaMemcpyDeviceToHost));
    position = next_position;
    return out;
}

std::vector<std::int32_t> Decoder::speculate(std::int32_t rounds, std::int32_t& position,
                                             std::vector<SpeculationRoundTrace>* trace,
                                             bool profile) {
    if (round_.staged()) {
        throw std::logic_error("speculation: staged (prompt lookup) rounds run through "
                               "verify_drafts()");
    }
    const cudaStream_t s = device_.stream;
    Tensor pos(state_.position, DType::I32, {1});
    Tensor step(state_.step, DType::I32, {1});
    ops::set_i32_scalar(pos, position, s);
    // The pending token (drawn by the prefill or the previous round) is emitted first.
    CUDA_CHECK(cudaMemcpyAsync(state_.out_tokens, state_.token, sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, s));
    ops::set_i32_scalar(step, 1, s);
    // The round is captured before the first one runs, not in the middle of a sequence.
    auto& graph = round_graph(deepest_, profile && trace != nullptr);
    if (trace != nullptr) {
        // One synchronization per round: the host reads the round back.
        for (std::int32_t r = 0; r < rounds; ++r) {
            // The round opens at the device's own pending position.
            device_.synchronize();
            std::int32_t p = 0;
            CUDA_CHECK(cudaMemcpy(&p, state_.position, sizeof(p), cudaMemcpyDeviceToHost));
            graph.launch(s);
            device_.synchronize();
            trace->push_back(read_round(p, deepest_, profile));
        }
    } else {
        for (std::int32_t r = 0; r < rounds; ++r) { graph.launch(s); }
    }
    auto out = collect(position);
    // The last emitted token is pending again: the next call emits it first, so it is not part
    // of this call's advance.
    out.pop_back();
    return out;
}

std::vector<std::int32_t> Decoder::verify_drafts(NgramRound& lookup,
                                                 std::span<const std::int32_t> drafts,
                                                 std::int32_t& position, std::int32_t& pending) {
    if (&lookup != &round_) {
        throw std::logic_error("speculation verify_drafts: not this decoder's staged round");
    }
    const auto k = static_cast<std::int32_t>(drafts.size());
    if (k < 1 || k > deepest_) {
        throw std::invalid_argument("speculation verify_drafts: draft count out of range");
    }
    const cudaStream_t s = device_.stream;
    auto& graph          = round_graph(k); // captured on first use, before the drafts are staged
    lookup.stage(drafts);
    Tensor pos(state_.position, DType::I32, {1});
    Tensor step(state_.step, DType::I32, {1});
    ops::set_i32_scalar(pos, position, s);
    CUDA_CHECK(cudaMemcpyAsync(state_.out_tokens, state_.token, sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, s));
    ops::set_i32_scalar(step, 1, s);
    graph.launch(s);
    auto out = collect(position);
    pending  = out.back();
    out.pop_back();
    return out;
}

SpeculationStats Decoder::stats() const {
    long long raw[3] = {0, 0, 0};
    CUDA_CHECK(cudaMemcpy(raw, stats_.p, sizeof(raw), cudaMemcpyDeviceToHost));
    SpeculationStats out;
    out.rounds   = raw[0];
    out.drafted  = raw[1];
    out.accepted = raw[2];
    return out;
}

} // namespace ninferno::models::speculation
