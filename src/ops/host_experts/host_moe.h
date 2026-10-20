#pragma once

// The routed experts of one MoE layer computed on the CPU, from the stored planes in host memory
// (MoE expert offload's decode path). The selection comes in (the GPU's router and top-k); what
// goes out is what the GPU route's expert stage hands on:
//
// - generic (ninferno/ops/moe.h): y[a] for every assignment a = t * top_k + k, BF16 [A, H]:
//   mid = act(GateUp_e x_t + b) rounded to BF16, y = Down_e mid + b rounded to BF16, as the GPU
//   route rounds them. The GPU's combine then weights and sums them.
// - gemma4 (ninferno/ops/gemma4_moe.h): out[t] = sum_k w_k Down_k(gelu_tanh(Gate_k x) * Up_k x),
//   BF16 [T, H], intermediates FP32 and the rank-ordered sum rounded once.
//
// Every output value is computed by one task from the canonical row dots (host_kernels.h), so the
// bytes depend neither on the thread count nor on the instruction set.

#include "core/weight.h"
#include "ninferno/ops/moe.h"
#include "ops/host_experts/host_kernels.h"
#include "ops/host_experts/host_pool.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ninferno::ops::host {

// A stored MXFP4 or NVFP4 plane as the CPU reads it.
struct HostPlane {
    QType qtype                 = QType::MXFP4;
    const std::uint8_t* codes   = nullptr;
    const std::uint8_t* scales  = nullptr;
    const float* divisors       = nullptr; // NVFP4: one per `divisor_rows` rows, else null
    std::int32_t divisor_rows   = 0;
    float divisor               = 1.0F;    // NVFP4 without per-source divisors
    std::int32_t n = 0, k = 0;
};

// Whether the CPU computes this plane's format and layout; `why` names what it does not.
[[nodiscard]] bool host_readable(const Weight& w, std::string* why = nullptr);
// The plane of `w` (host_readable, else std::invalid_argument naming the format).
[[nodiscard]] HostPlane host_plane(const Weight& w);

struct GenericLayer {
    HostPlane gate_up, down;
    const void* gate_up_bias = nullptr;
    bool gate_up_bias_bf16   = false;
    const void* down_bias    = nullptr;
    bool down_bias_bf16      = false;
    std::int32_t experts = 0, hidden = 0, inter = 0, top_k = 0;
    MoeActivation activation = MoeActivation::SwiGlu;
    float alpha = 1.702F, limit = 7.0F;
    bool interleaved = false;
};

struct Gemma4Layer {
    HostPlane gate_up, down;
    std::int32_t experts = 0, hidden = 0, inter = 0, top_k = 0;
};

class HostMoe {
public:
    // threads: HostPool's (0 = one per physical core). isa: the row-dot instruction set.
    HostMoe(int threads, Isa isa);
    ~HostMoe();

    [[nodiscard]] Isa isa() const noexcept { return isa_; }
    [[nodiscard]] int threads() const noexcept { return pool_.size(); }
    [[nodiscard]] const std::vector<int>& cpus() const noexcept { return pool_.cpus(); }

    // x: BF16 [T, H] (token-major); ids: [T, top_k]; y: BF16 [T * top_k, H]. 1 <= T <= 16.
    void generic(const GenericLayer& layer, int tokens, const std::uint16_t* x,
                 const std::int32_t* ids, std::uint16_t* y);
    // x: BF16 [T, H]; ids, weights: [T, top_k] (rank order); out: BF16 [T, H]. 1 <= T <= 16.
    void gemma4(const Gemma4Layer& layer, int tokens, const std::uint16_t* x,
                const std::int32_t* ids, const float* weights, std::uint16_t* out);

    struct Work; // scratch shared by the tasks of one call (host_moe.cpp)

private:
    // Work::gu for the selected experts: every gate_up row of expert(a) against x[token(a)].
    void gate_up_dots(const HostPlane& plane, int rows_per_expert, int hidden, int assignments);

    HostPool pool_;
    Isa isa_;
    RowKernels kernels_;
    std::unique_ptr<Work> work_;
};

inline constexpr int kHostMaxTokens = 16;

} // namespace ninferno::ops::host
