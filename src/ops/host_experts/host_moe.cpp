// HostMoe (host_moe.h). Three task phases per call: gate/up dots (task = one selected expert x a
// block of its gate_up rows), the activation (task = one assignment), and down (task = a block of
// hidden rows over every selected expert). Each value is written by exactly one task, with the
// same arithmetic whatever the blocks, so the partition changes no byte. An NVFP4 block is a whole
// 128-row tile of the plane: the swizzled scales of a tile's rows share their cache lines, and a
// tile split between tasks would have each task fetch every line for a quarter of its bytes.
#include "ops/host_experts/host_moe.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ninferno::ops::host {

namespace {

// Down's rows per expert-output buffer inside a task.
constexpr int kSubRows  = 16;
constexpr int kMaxTopK  = 32;
constexpr int kMaxAssignments = kHostMaxTokens * kMaxTopK;

inline float bf16_to_f32(std::uint16_t h) {
    const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline std::uint16_t f32_to_bf16(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    if ((bits & 0x7fffffffU) > 0x7f800000U) { return static_cast<std::uint16_t>((bits >> 16) | 0x40U); }
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>(bits >> 16);
}

inline float round_bf16(float f) { return bf16_to_f32(f32_to_bf16(f)); }

inline float sigmoid(float v) { return 1.0F / (1.0F + std::exp(-v)); }

inline float activate(const GenericLayer& l, float g, float u) {
    switch (l.activation) {
    case MoeActivation::GeGluTanh: {
        const float inner = 0.7978845608028654F * (g + 0.044715F * g * g * g);
        return 0.5F * g * (1.0F + std::tanh(inner)) * u;
    }
    case MoeActivation::ClampedSwiGlu: {
        const float gc = std::fmin(g, l.limit);
        const float uc = std::fmin(std::fmax(u, -l.limit), l.limit);
        return (uc + 1.0F) * gc * sigmoid(l.alpha * gc);
    }
    case MoeActivation::Relu2: {
        const float r = std::fmax(u, 0.0F);
        return r * r;
    }
    case MoeActivation::SwiGlu:
        break;
    }
    return g * sigmoid(g) * u;
}

inline float gelu_tanh_mul(float g, float u) {
    const float inner = 0.7978845608028654F * (g + 0.044715F * g * g * g);
    return 0.5F * g * (1.0F + std::tanh(inner)) * u;
}

inline float bias_at(const void* bias, bool bf16, std::int64_t index) {
    if (bias == nullptr) { return 0.0F; }
    return bf16 ? bf16_to_f32(static_cast<const std::uint16_t*>(bias)[index])
                : static_cast<const float*>(bias)[index];
}

// Row `row` of `plane` against `columns` permuted inputs, the NVFP4 divisor applied.
inline void plane_rows(const RowKernels& kernels, const HostPlane& p, std::int64_t row,
                       const float* const* x, int columns, float* out) {
    const std::uint8_t* codes = p.codes + row * (p.k / 2);
    const int chunks          = p.k / kChunk;
    if (p.qtype == QType::MXFP4) {
        kernels.mxfp4(codes, p.scales + row * (p.k / kChunk), chunks, x, columns, out);
        return;
    }
    kernels.nvfp4(codes, nvfp4_scale_row(p.scales, row, p.k), chunks, x, columns, out);
    const float divisor = p.divisors != nullptr ? p.divisors[row / p.divisor_rows] : p.divisor;
    for (int i = 0; i < columns; ++i) { out[i] = out[i] / divisor; }
}

// Rows of one task: NVFP4 whole 128-row scale tiles; MXFP4 (scales per row) 16.
inline int task_rows(const HostPlane& p) { return p.qtype == QType::NVFP4 ? 128 : 16; }

} // namespace

bool host_readable(const Weight& w, std::string* why) {
    const auto no = [&](const std::string& reason) {
        if (why) { *why = reason; }
        return false;
    };
    if (w.qtype == QType::MXFP4) {
        if (w.layout != QuantLayout::BlockScaleK32E8M0) { return no("MXFP4 in another layout"); }
        if (w.k % kChunk != 0) { return no("MXFP4 with K not a multiple of 32"); }
        return true;
    }
    if (w.qtype == QType::NVFP4) {
        if (w.layout != QuantLayout::BlockScaleK16M128x4 || w.scale_dtype != DType::FP8_E4M3FN) {
            return no("NVFP4 in another layout");
        }
        if (w.k % 64 != 0 || w.n % 128 != 0) { return no("NVFP4 with K % 64 or N % 128"); }
        return true;
    }
    return no("the CPU computes MXFP4 and NVFP4 experts; this bank is QType " +
              std::to_string(static_cast<int>(w.qtype)));
}

HostPlane host_plane(const Weight& w) {
    std::string why;
    if (!host_readable(w, &why)) { throw std::invalid_argument("host experts: " + why); }
    HostPlane p;
    p.qtype  = w.qtype;
    p.codes  = static_cast<const std::uint8_t*>(w.qdata);
    p.scales = static_cast<const std::uint8_t*>(w.scales);
    p.n      = w.n;
    p.k      = w.k;
    if (w.qtype == QType::NVFP4) {
        p.divisor = w.weight_scale_divisor;
        if (w.weight_divisors != nullptr && w.weight_divisor_rows > 0 &&
            w.weight_divisor_rows < w.n) {
            p.divisors     = static_cast<const float*>(w.weight_divisors);
            p.divisor_rows = w.weight_divisor_rows;
        }
        if (!(p.divisor > 0.0F) && p.divisors == nullptr) {
            throw std::invalid_argument("host experts: NVFP4 plane without a positive divisor");
        }
    }
    return p;
}

struct HostMoe::Work {
    std::vector<float> x;   // permuted inputs [T, H]
    std::vector<float> gu;  // gate_up dots [A, 2F] in the plane's row order
    std::vector<float> mid; // permuted intermediates [A, F]
    // The selected experts in first-seen order and their assignments.
    int unique = 0;
    std::int32_t expert[kMaxAssignments]{};
    int count[kMaxAssignments]{};
    int assignment[kMaxAssignments][kHostMaxTokens]{};
    int token[kMaxAssignments][kHostMaxTokens]{};

    void group(int tokens, int top_k, const std::int32_t* ids, std::int32_t experts) {
        unique = 0;
        for (int a = 0; a < tokens * top_k; ++a) {
            const std::int32_t e = ids[a];
            if (e < 0 || e >= experts) {
                throw std::out_of_range("host experts: expert id " + std::to_string(e) +
                                        " outside the layer's " + std::to_string(experts));
            }
            int u = 0;
            while (u < unique && expert[u] != e) { ++u; }
            if (u == unique) {
                expert[unique] = e;
                count[unique]  = 0;
                ++unique;
            }
            if (count[u] >= kHostMaxTokens) {
                throw std::logic_error("host experts: an expert selected twice for one token");
            }
            assignment[u][count[u]] = a;
            token[u][count[u]]      = a / top_k;
            ++count[u];
        }
    }
};

HostMoe::HostMoe(int threads, Isa isa)
    : pool_(threads), isa_(isa), kernels_(row_kernels(isa)), work_(std::make_unique<Work>()) {}

HostMoe::~HostMoe() = default;

namespace {

void check_call(int tokens, std::int32_t top_k, std::int32_t hidden, std::int32_t inter) {
    if (tokens < 1 || tokens > kHostMaxTokens) {
        throw std::invalid_argument("host experts: 1 to 16 tokens a call");
    }
    if (top_k < 1 || top_k > kMaxTopK) { throw std::invalid_argument("host experts: top_k 1 to 32"); }
    if (hidden % kChunk != 0 || inter % kChunk != 0) {
        throw std::invalid_argument("host experts: H and F must be multiples of 32");
    }
}

} // namespace

void HostMoe::generic(const GenericLayer& l, int tokens, const std::uint16_t* x,
                      const std::int32_t* ids, std::uint16_t* y) {
    check_call(tokens, l.top_k, l.hidden, l.inter);
    Work& w     = *work_;
    const int H = l.hidden, F = l.inter;
    w.x.resize(static_cast<std::size_t>(tokens) * H);
    w.mid.resize(static_cast<std::size_t>(tokens) * l.top_k * F);
    {
        std::vector<float> row(static_cast<std::size_t>(H));
        for (int t = 0; t < tokens; ++t) {
            for (int h = 0; h < H; ++h) { row[static_cast<std::size_t>(h)] = bf16_to_f32(x[static_cast<std::size_t>(t) * H + h]); }
            permute_input(row.data(), H, w.x.data() + static_cast<std::size_t>(t) * H);
        }
    }
    w.group(tokens, l.top_k, ids, l.experts);

    // Gate/up: mid[a][j] = bf16(act(gate_j . x + b, up_j . x + b)). Interleaved rows (gate 2j,
    // up 2j+1) pair inside every block, so a task applies the activation to its own rows;
    // concatenated ones (gate j, up F + j) are dotted first and activated in a second phase.
    if (l.interleaved) {
        const int pairs    = std::max(16, task_rows(l.gate_up) / 2);
        const int f_blocks = (F + pairs - 1) / pairs;
        auto gate_up = [&](int task) {
            const int u = task / f_blocks, block = task % f_blocks;
            const std::int64_t e = w.expert[u];
            const int columns    = w.count[u];
            const float* in[kHostMaxTokens];
            for (int c = 0; c < columns; ++c) {
                in[c] = w.x.data() + static_cast<std::size_t>(w.token[u][c]) * H;
            }
            float g[kHostMaxTokens], v[kHostMaxTokens];
            const int j_end = std::min(F, (block + 1) * pairs);
            for (int j = block * pairs; j < j_end; ++j) {
                const std::int64_t gate_row = e * 2 * F + 2 * j;
                plane_rows(kernels_, l.gate_up, gate_row, in, columns, g);
                plane_rows(kernels_, l.gate_up, gate_row + 1, in, columns, v);
                const float bg = bias_at(l.gate_up_bias, l.gate_up_bias_bf16, gate_row);
                const float bu = bias_at(l.gate_up_bias, l.gate_up_bias_bf16, gate_row + 1);
                for (int c = 0; c < columns; ++c) {
                    w.mid[static_cast<std::size_t>(w.assignment[u][c]) * F + permuted_index(j)] =
                        round_bf16(activate(l, g[c] + bg, v[c] + bu));
                }
            }
        };
        pool_.run(w.unique * f_blocks, gate_up);
    } else {
        gate_up_dots(l.gate_up, 2 * F, H, tokens * l.top_k);
        auto activation = [&](int a) {
            const std::int64_t e = ids[a];
            const float* gu      = w.gu.data() + static_cast<std::size_t>(a) * 2 * F;
            for (int j = 0; j < F; ++j) {
                const int gate = j;
                const int up   = F + j;
                const float bg = bias_at(l.gate_up_bias, l.gate_up_bias_bf16, e * 2 * F + gate);
                const float bu = bias_at(l.gate_up_bias, l.gate_up_bias_bf16, e * 2 * F + up);
                w.mid[static_cast<std::size_t>(a) * F + permuted_index(j)] =
                    round_bf16(activate(l, gu[gate] + bg, gu[up] + bu));
            }
        };
        pool_.run(tokens * l.top_k, activation);
    }

    // Down: y[a][h] = bf16(down_h . mid[a] + b).
    const int rows     = task_rows(l.down);
    const int h_blocks = (H + rows - 1) / rows;
    auto down = [&](int block) {
        const int h_end = std::min(H, (block + 1) * rows);
        for (int u = 0; u < w.unique; ++u) {
            const std::int64_t e = w.expert[u];
            const int columns    = w.count[u];
            const float* in[kHostMaxTokens];
            for (int c = 0; c < columns; ++c) { in[c] = w.mid.data() + static_cast<std::size_t>(w.assignment[u][c]) * F; }
            float out[kHostMaxTokens];
            for (int h = block * rows; h < h_end; ++h) {
                const std::int64_t row = e * H + h;
                plane_rows(kernels_, l.down, row, in, columns, out);
                const float b = bias_at(l.down_bias, l.down_bias_bf16, row);
                for (int c = 0; c < columns; ++c) {
                    y[static_cast<std::size_t>(w.assignment[u][c]) * H + h] = f32_to_bf16(out[c] + b);
                }
            }
        }
    };
    pool_.run(h_blocks, down);
}

void HostMoe::gemma4(const Gemma4Layer& l, int tokens, const std::uint16_t* x,
                     const std::int32_t* ids, const float* weights, std::uint16_t* out) {
    check_call(tokens, l.top_k, l.hidden, l.inter);
    Work& w     = *work_;
    const int H = l.hidden, F = l.inter;
    w.x.resize(static_cast<std::size_t>(tokens) * H);
    w.mid.resize(static_cast<std::size_t>(tokens) * l.top_k * F);
    {
        std::vector<float> row(static_cast<std::size_t>(H));
        for (int t = 0; t < tokens; ++t) {
            for (int h = 0; h < H; ++h) { row[static_cast<std::size_t>(h)] = bf16_to_f32(x[static_cast<std::size_t>(t) * H + h]); }
            permute_input(row.data(), H, w.x.data() + static_cast<std::size_t>(t) * H);
        }
    }
    w.group(tokens, l.top_k, ids, l.experts);
    gate_up_dots(l.gate_up, 2 * F, H, tokens * l.top_k);

    // Activation: mid[a][j] = gelu_tanh(gate_j . x) * (up_j . x), FP32.
    auto activation = [&](int a) {
        const float* gu = w.gu.data() + static_cast<std::size_t>(a) * 2 * F;
        for (int j = 0; j < F; ++j) {
            w.mid[static_cast<std::size_t>(a) * F + permuted_index(j)] = gelu_tanh_mul(gu[j], gu[F + j]);
        }
    };
    pool_.run(tokens * l.top_k, activation);

    // Down and the weighted sum: out[t][h] = bf16(sum_k w_k * (down_h . mid[t, k])), rank order;
    // a task's rows in sub-blocks of kSubRows (the expert outputs' buffer).
    const int rows     = task_rows(l.down);
    const int h_blocks = (H + rows - 1) / rows;
    auto down = [&](int block) {
        const int block_end = std::min(H, (block + 1) * rows);
        for (int h0 = block * rows; h0 < block_end; h0 += kSubRows) {
            const int h_end = std::min(block_end, h0 + kSubRows);
            float expert_out[kMaxAssignments][kSubRows];
            for (int u = 0; u < w.unique; ++u) {
                const std::int64_t e = w.expert[u];
                const int columns    = w.count[u];
                const float* in[kHostMaxTokens];
                for (int c = 0; c < columns; ++c) { in[c] = w.mid.data() + static_cast<std::size_t>(w.assignment[u][c]) * F; }
                float dots[kHostMaxTokens];
                for (int h = h0; h < h_end; ++h) {
                    plane_rows(kernels_, l.down, e * H + h, in, columns, dots);
                    for (int c = 0; c < columns; ++c) { expert_out[w.assignment[u][c]][h - h0] = dots[c]; }
                }
            }
            for (int t = 0; t < tokens; ++t) {
                for (int h = h0; h < h_end; ++h) {
                    float sum = 0.0F;
                    for (int k = 0; k < l.top_k; ++k) {
                        const int a = t * l.top_k + k;
                        sum         = std::fma(weights[a], expert_out[a][h - h0], sum);
                    }
                    out[static_cast<std::size_t>(t) * H + h] = f32_to_bf16(sum);
                }
            }
        }
    };
    pool_.run(h_blocks, down);
}

void HostMoe::gate_up_dots(const HostPlane& p, int rows_per_expert, int hidden,
                           int assignments) {
    Work& w = *work_;
    // gu[a][r] = row r of expert(a)'s gate_up . x[token(a)], every r in [0, rows_per_expert); a
    // task covers the rows of one selected expert inside one block of the plane's absolute rows
    // (expert banks need not start on a block).
    w.gu.resize(static_cast<std::size_t>(assignments) * rows_per_expert);
    const int rows  = task_rows(p);
    const int spans = (rows_per_expert + rows - 1) / rows + 1;
    auto task = [&](int index) {
        const int u               = index / spans;
        const std::int64_t first  = static_cast<std::int64_t>(w.expert[u]) * rows_per_expert;
        const std::int64_t block  = first / rows + index % spans;
        const std::int64_t r0     = std::max(first, block * rows);
        const std::int64_t r1     = std::min(first + rows_per_expert, (block + 1) * rows);
        const int columns         = w.count[u];
        const float* in[kHostMaxTokens];
        for (int c = 0; c < columns; ++c) {
            in[c] = w.x.data() + static_cast<std::size_t>(w.token[u][c]) * hidden;
        }
        float d[kHostMaxTokens];
        for (std::int64_t r = r0; r < r1; ++r) {
            plane_rows(kernels_, p, r, in, columns, d);
            for (int c = 0; c < columns; ++c) {
                w.gu[static_cast<std::size_t>(w.assignment[u][c]) * rows_per_expert + (r - first)] = d[c];
            }
        }
    };
    pool_.run(w.unique * spans, task);
}

} // namespace ninferno::ops::host
