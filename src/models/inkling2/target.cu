// One sequence's Inkling 2 forward (target.h): the Pass over the sequence's own planes.
#include "models/inkling2/target.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ninferno::models::inkling2 {
namespace {

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("inkling2: ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

} // namespace

Target::Target(const Model& model, std::int32_t max_context, std::int32_t max_tokens)
    : model_(model), c_(model.config()), pass_(model, max_tokens), max_context_(max_context),
      max_tokens_(max_tokens), work_(Pass::work_bytes(model, max_tokens)) {
    if (max_tokens < 1 || max_context < max_tokens) {
        throw std::invalid_argument("inkling2: max_tokens in [1, max_context]");
    }
    slots_.resize(static_cast<std::size_t>(pass_.slots()));
    kv_.resize(slots_.size());
    conv_.resize(slots_.size());
    for (const auto& v : pass_.visits()) {
        auto& slot = slots_[static_cast<std::size_t>(v.slot)];
        if (slot.capacity) { continue; }
        const bool full = c_.kinds[static_cast<std::size_t>(v.layer)] == LayerKind::Full;
        // A sliding visit's ring keeps every key a call of max_tokens can see while it appends.
        slot.capacity =
            full ? max_context : std::min(max_context, c_.sliding_window + max_tokens - 1);
        const std::size_t plane = std::size_t(c_.head_dim) * slot.capacity * c_.kv_heads * 2;
        slot.k                  = DeviceBuffer(plane);
        slot.v                  = DeviceBuffer(plane);
        slot.conv               = DeviceBuffer(std::size_t(c_.conv_channels()) * 3 * sizeof(float));
        auto& kv                = kv_[static_cast<std::size_t>(v.slot)];
        kv.k      = Tensor(slot.k.p, DType::BF16, {c_.head_dim, slot.capacity, c_.kv_heads});
        kv.v      = Tensor(slot.v.p, DType::BF16, {c_.head_dim, slot.capacity, c_.kv_heads});
        kv.window = full ? 0 : c_.sliding_window;
        conv_[static_cast<std::size_t>(v.slot)] = {static_cast<float*>(slot.conv.p), 0, nullptr};
    }
    ids_       = DeviceBuffer(std::size_t(max_tokens) * sizeof(std::int32_t));
    positions_ = DeviceBuffer(std::size_t(max_tokens) * sizeof(std::int32_t));
    logits_    = DeviceBuffer(std::size_t(c_.vocab_rows) * 2);
    reset();
}

void Target::reset() {
    for (auto& slot : slots_) {
        slot.conv.fill(0);
        slot.k.fill(0);
        slot.v.fill(0);
    }
    position_ = 0;
}

std::vector<float> Target::forward(std::span<const std::int32_t> ids, cudaStream_t s) {
    const std::int32_t T = static_cast<std::int32_t>(ids.size());
    if (T < 1 || T > max_tokens_ || position_ + T > max_context_) {
        throw std::invalid_argument(
            "inkling2: a call takes 1..max_tokens tokens inside max_context");
    }
    const auto H = c_.hidden_size;
    work_.reset();
    std::vector<std::int32_t> pos(static_cast<std::size_t>(T));
    for (std::int32_t t = 0; t < T; ++t) { pos[static_cast<std::size_t>(t)] = position_ + t; }
    check(cudaMemcpyAsync(ids_.p, ids.data(), ids.size_bytes(), cudaMemcpyHostToDevice, s),
          "ids upload");
    check(cudaMemcpyAsync(positions_.p, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice, s),
          "positions upload");
    const Tensor ids_t(ids_.p, DType::I32, {T});
    const Tensor positions(positions_.p, DType::I32, {T});
    pass_.observe = observe;
    const Bindings bound{.kv = kv_, .conv = conv_, .batch = 1};
    const Tensor hidden = pass_.forward(work_, ids_t, positions, bound, s);

    // The head of the last token.
    const Tensor last(static_cast<__nv_bfloat16*>(hidden.data) + std::size_t(H) * (T - 1),
                      DType::BF16, {H, 1});
    if (observe) {
        check(cudaStreamSynchronize(s), "observe");
        observe("result_norm", -1, last);
    }
    Tensor logits(logits_.p, DType::BF16, {c_.vocab_rows, 1});
    pass_.head(work_, last, logits, s);
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(c_.vocab_size));
    check(cudaMemcpyAsync(bits.data(), logits_.p, bits.size() * 2, cudaMemcpyDeviceToHost, s),
          "logits download");
    check(cudaStreamSynchronize(s), "forward");
    position_ += T;
    std::vector<float> out(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
        const std::uint32_t w = std::uint32_t(bits[i]) << 16;
        std::memcpy(&out[i], &w, 4);
    }
    return out;
}

} // namespace ninferno::models::inkling2
