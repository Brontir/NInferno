// Qwen 3.x prefill ops: a token's output must not depend on how many other tokens share the call,
// nor on where in the call it sits. A prefill chunk's size and start are decided by where a request
// was cut (the chunk size, a cache frontier, a restore base), not by the model, so every column of a
// call over tokens [o, o+t) has to equal, byte for byte, the same column of one wide call.
//
// Every prefill op and registered shape of the two v3 artifacts is run once over the widest call
// and again over windows that cross each route and tile frontier (T from 17 up), at the start of
// the input and at offsets inside it. Stateful ops are checked by splitting one call in two and
// carrying the state: the causal convolution at any cut, the Gated DeltaNet scan at cuts on its
// absolute 64-token block grid. A cut off that grid is reported, not failed: the scan runs a
// call's trailing partial block token by token, which rounds differently from the block form, so
// such a cut is a known, documented exception.
//
// The NVFP4 A16Only rows are the exact activation contract (--activations exact): there every width
// counts, decode and speculative verify included, so their windows start at T = 1.
#include "ops/linear/linear_test_common.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/weight.h"
#include "ninferno/ops/attn_input_proj.h"
#include "ninferno/ops/causal_conv1d_silu.h"
#include "ninferno/ops/gated_delta_net.h"
#include "ninferno/ops/gated_rmsnorm.h"
#include "ninferno/ops/gdn_gating_proj.h"
#include "ninferno/ops/gdn_input_proj.h"
#include "ninferno/ops/linear_add.h"
#include "ninferno/ops/linear_swiglu.h"
#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/rmsnorm.h"
#include "ninferno/ops/sparse_moe.h"
#include "ops/common/native_nvfp4.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninferno;
using namespace ninferno::test;
using namespace ninferno::test::linear;

constexpr std::int32_t kWidest = 8448;

struct Window {
    std::int32_t offset;
    std::int32_t tokens;
};

// Every frontier of the registered route tables is probed from both sides, plus the chunk tails
// the default chunk sizes leave, and windows that start inside the widest call.
std::vector<Window> windows() {
    std::vector<Window> out;
    for (const std::int32_t t :
         {17,   18,   20,   21,   22,   24,   25,   31,   32,   33,   34,  42,  43,  46,   47,
          48,   49,   51,   52,   63,   64,   65,   80,   96,   97,   100, 112, 127, 128,  129,
          144,  145,  191,  192,  193,  255,  256,  257,  288,  289,  300, 320, 321, 384,  385,
          448,  449,  511,  512,  513,  560,  561,  640,  641,  734,  767, 768, 769, 1000, 1023,
          1024, 1025, 1246, 2047, 2048, 2049, 2782, 4095, 4096, 4097, 8192}) {
        out.push_back({0, t});
    }
    for (const Window w : {Window{1000, 1024}, Window{777, 300}, Window{64, 2048}, Window{1, 17},
                           Window{4591, 49}, Window{8431, 17}, Window{256, 8192}, Window{300, 100},
                           Window{5000, 64}, Window{123, 128}, Window{2000, 33}}) {
        out.push_back(w);
    }
    return out;
}

// The exact contract's windows: every decode and verify width, the prefill frontiers, and narrow
// windows inside the widest call (a decode or verify step of a request whose prompt came first).
std::vector<Window> exact_windows() {
    std::vector<Window> out;
    for (std::int32_t t = 1; t <= 17; ++t) { out.push_back({0, t}); }
    for (const std::int32_t t : {20, 24, 31, 32, 33, 48, 63, 64, 65, 100, 128, 129, 1000, 4096}) {
        out.push_back({0, t});
    }
    for (const Window w : {Window{5000, 1}, Window{123, 3}, Window{64, 8}, Window{777, 16},
                           Window{8447, 1}, Window{1, 17}, Window{1000, 1024}, Window{300, 100},
                           Window{2000, 33}, Window{256, 8192}}) {
        out.push_back(w);
    }
    return out;
}

cudaStream_t stream() { return nullptr; }

std::vector<std::uint8_t> download(const void* device, std::size_t bytes) {
    std::vector<std::uint8_t> host(bytes);
    cuda_check(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost), "download");
    return host;
}

std::vector<std::uint16_t> random_bf16(std::size_t count, std::uint32_t seed, float scale) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, scale);
    std::vector<std::uint16_t> out(count);
    for (auto& v : out) { v = f32_to_bf16(normal(rng)); }
    return out;
}

std::vector<float> random_f32(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(lo, hi);
    std::vector<float> out(count);
    for (auto& v : out) { v = uniform(rng); }
    return out;
}

DeviceBuffer upload(const void* host, std::size_t bytes) {
    DeviceBuffer device(bytes);
    device.copy_from_host(host, bytes);
    return device;
}

template <class T>
DeviceBuffer upload(const std::vector<T>& host) {
    return upload(host.data(), host.size() * sizeof(T));
}

Weight dense_bf16(const DeviceBuffer& data, std::int32_t rows, std::int32_t columns) {
    Weight w{};
    w.payload         = data.p;
    w.payload_bytes   = static_cast<std::uint64_t>(rows) * columns * sizeof(std::uint16_t);
    w.qtype           = QType::BF16;
    w.qdata           = data.p;
    w.n               = rows;
    w.k               = columns;
    w.layout          = QuantLayout::Contiguous;
    w.ndim            = 2;
    w.shape[0]        = rows;
    w.shape[1]        = columns;
    w.padded_shape[0] = rows;
    w.padded_shape[1] = columns;
    return w;
}

// A packed weight from the linear test generators, resident on the device.
struct PackedDeviceWeight {
    quantized_weight::PackedWeight host;
    DeviceBuffer payload;
    Weight weight;

    PackedDeviceWeight(WeightGenerator generator, std::int32_t n, std::int32_t k,
                       std::uint32_t seed)
        : host(generator(n, k, seed)), payload(host.payload.size()) {
        payload.copy_from_host(host.payload.data(), host.payload.size());
        weight = host.device_weight(payload.p);
        host.payload.clear();
        host.payload.shrink_to_fit();
        host.dequant.clear();
        host.dequant.shrink_to_fit();
    }
};

// A row-split groupwise weight with random codes and small random scales, built on its planes
// directly: the routed expert banks are too large for the host generators.
class RandomRowSplit {
public:
    RandomRowSplit(QType qtype, std::int32_t rows, std::int32_t columns, std::uint32_t seed)
        : qtype_(qtype), rows_(rows), columns_(columns) {
        std::int32_t group = 64;
        std::size_t high   = 0;
        switch (qtype) {
        case QType::Q4_G64_FP16:
            break;
        case QType::Q5_G64_FP16:
            high = 8;
            break;
        case QType::Q6_G64_FP16:
            high = 16;
            break;
        case QType::Q8_G32_FP16:
            group = 32;
            break;
        default:
            throw std::invalid_argument("RandomRowSplit: unsupported qtype");
        }
        group_          = group;
        const auto rows64 = static_cast<std::size_t>(rows);
        const auto groups = static_cast<std::size_t>(columns / group);
        std::mt19937 rng(seed);
        std::vector<std::uint8_t> codes(rows64 * groups * 32);
        for (auto& b : codes) { b = static_cast<std::uint8_t>(rng()); }
        codes_ = upload(codes);
        if (high != 0) {
            std::vector<std::uint8_t> hi(rows64 * groups * high);
            for (auto& b : hi) { b = static_cast<std::uint8_t>(rng()); }
            high_ = std::make_unique<DeviceBuffer>(upload(hi));
        }
        std::uniform_real_distribution<float> uniform(0.002F, 0.02F);
        std::vector<std::uint16_t> scales(rows64 * groups);
        for (auto& s : scales) {
            const float value = uniform(rng);
            // FP16 bits of a small positive value.
            std::uint32_t f;
            std::memcpy(&f, &value, 4);
            const int exponent = static_cast<int>((f >> 23) & 0xff) - 127 + 15;
            s = static_cast<std::uint16_t>((exponent << 10) | ((f >> 13) & 0x3ff));
        }
        scales_ = upload(scales);
    }

    Weight weight() const {
        Weight w{};
        w.payload          = codes_.p;
        w.payload_bytes    = codes_.bytes + (high_ ? high_->bytes : 0) + scales_.bytes;
        w.high_plane_bytes = high_ ? high_->bytes : 0;
        w.qtype            = qtype_;
        w.group_size       = static_cast<std::uint32_t>(group_);
        w.qdata            = codes_.p;
        w.qhigh            = high_ ? high_->p : nullptr;
        w.scales           = scales_.p;
        w.n                = rows_;
        w.k                = columns_;
        w.group            = group_;
        w.layout           = QuantLayout::RowSplit;
        w.scale_dtype      = DType::FP16;
        w.ndim             = 2;
        w.shape[0]         = rows_;
        w.shape[1]         = columns_;
        w.padded_shape[0]  = rows_;
        w.padded_shape[1]  = columns_;
        return w;
    }

private:
    QType qtype_;
    std::int32_t rows_;
    std::int32_t columns_;
    std::int32_t group_ = 64;
    DeviceBuffer codes_;
    std::unique_ptr<DeviceBuffer> high_;
    DeviceBuffer scales_;
};

// One output of an op: a column-major device buffer of `column_bytes` per token, sized for the
// widest call.
struct Output {
    DeviceBuffer* buffer;
    std::size_t column_bytes;
};

struct OpCase {
    std::string name;
    // Runs the op over input columns [offset, offset + tokens), writing output columns [0, tokens).
    std::function<void(std::int32_t, std::int32_t)> run;
    std::vector<Output> outputs;
};

int check_case(const OpCase& op, const std::vector<Window>& spans = windows()) {
    op.run(0, kWidest);
    cuda_synchronize();
    std::vector<std::vector<std::uint8_t>> reference;
    for (const Output& out : op.outputs) {
        reference.push_back(download(out.buffer->p, out.column_bytes * kWidest));
    }
    int failures = 0;
    for (const Window w : spans) {
        if (w.offset + w.tokens > kWidest) { continue; }
        op.run(w.offset, w.tokens);
        cuda_synchronize();
        for (std::size_t o = 0; o < op.outputs.size(); ++o) {
            const Output& out = op.outputs[o];
            const auto got    = download(out.buffer->p, out.column_bytes * w.tokens);
            std::int32_t differing = 0;
            std::int32_t first     = -1;
            for (std::int32_t c = 0; c < w.tokens; ++c) {
                if (std::memcmp(got.data() + c * out.column_bytes,
                                reference[o].data() + (w.offset + c) * out.column_bytes,
                                out.column_bytes) != 0) {
                    ++differing;
                    if (first < 0) { first = c; }
                }
            }
            if (differing != 0) {
                ++failures;
                std::cout << "DIFFERS " << op.name << " output " << o << " window [" << w.offset
                          << ", +" << w.tokens << "): " << differing << " columns, first "
                          << first << '\n';
            }
        }
    }
    std::cout << (failures == 0 ? "invariant " : "VARIANT ") << op.name << '\n';
    return failures;
}

// Shared input activations of the widest call, one per K.
struct Inputs {
    std::vector<std::pair<std::int32_t, std::unique_ptr<DeviceBuffer>>> by_k;

    const DeviceBuffer& get(std::int32_t k) {
        for (auto& [key, buffer] : by_k) {
            if (key == k) { return *buffer; }
        }
        const auto host = random_bf16(static_cast<std::size_t>(k) * kWidest,
                                      0x5eedU + static_cast<std::uint32_t>(k), 1.0F);
        by_k.emplace_back(k, std::make_unique<DeviceBuffer>(upload(host)));
        return *by_k.back().second;
    }
};

Tensor columns(const DeviceBuffer& buffer, DType dtype, std::int32_t rows, std::int32_t offset,
               std::int32_t tokens) {
    const std::size_t column_bytes = static_cast<std::size_t>(rows) * dtype_size(dtype);
    return Tensor(static_cast<std::uint8_t*>(buffer.p) + column_bytes * offset, dtype,
                  {rows, tokens});
}

DeviceExecutionView execution() {
    int device = 0;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    int sms = 0;
    cuda_check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "SM count");
    return DeviceExecutionView{stream(), sms};
}

// --- stateless ops -------------------------------------------------------------------------

// every_width: the exact contract (--activations exact), under the invariant route profile from the
// workspace query on, as the engine sizes it.
int check_norm_gating(Inputs& inputs, std::int32_t heads, std::int32_t hidden, const char* name,
                      bool every_width = false) {
    const ops::ScopedInvariantRoutes invariant(every_width);
    const auto norm   = upload(random_bf16(hidden, 11U, 0.2F));
    const auto ab     = upload(random_bf16(static_cast<std::size_t>(2 * heads) * hidden, 12U,
                                           0.02F));
    const auto a_log  = upload(random_f32(heads, 13U, -1.0F, 1.0F));
    const auto bias   = upload(random_f32(heads, 14U, -1.0F, 1.0F));
    DeviceBuffer h(static_cast<std::size_t>(hidden) * kWidest * 2);
    DeviceBuffer g(static_cast<std::size_t>(heads) * kWidest * 4);
    DeviceBuffer beta(static_cast<std::size_t>(heads) * kWidest * 4);
    const Weight weight = dense_bf16(ab, 2 * heads, hidden);
    const std::size_t capacity =
        ops::gdn_norm_gating_proj_workspace_capacity_bytes(heads, hidden, 1, kWidest);
    DeviceArena workspace(std::max<std::size_t>(capacity, 256));
    const DeviceBuffer& x = inputs.get(hidden);
    OpCase op{name,
              [&](std::int32_t offset, std::int32_t t) {
                  Tensor tx = columns(x, DType::BF16, hidden, offset, t);
                  Tensor tn(norm.p, DType::BF16, {hidden});
                  Tensor ta(a_log.p, DType::FP32, {heads});
                  Tensor td(bias.p, DType::FP32, {heads});
                  Tensor th(h.p, DType::BF16, {hidden, t});
                  Tensor tg(g.p, DType::FP32, {heads, t});
                  Tensor tb(beta.p, DType::FP32, {heads, t});
                  ops::gdn_norm_gating_proj(tx, tn, 1.0e-6F, weight, ta, td, workspace, th, tg,
                                            tb, execution());
              },
              {{&h, static_cast<std::size_t>(hidden) * 2},
               {&g, static_cast<std::size_t>(heads) * 4},
               {&beta, static_cast<std::size_t>(heads) * 4}}};
    return check_case(op, every_width ? exact_windows() : windows());
}

// The fused input projections (gdn_input_proj, attn_input_proj) take NVFP4 only where the native
// kernels are built; elsewhere the model runs NVFP4 through the generic projections instead.
bool fused_runs(WeightGenerator generator, const char* name) {
    if (generator != make_nvfp4_weight || ops::detail::kNativeNvfp4) return true;
    std::cout << "SKIP " << name << ": the fused NVFP4 projection needs the sm_120a build\n";
    return false;
}

int check_gdn_input(Inputs& inputs, WeightGenerator generator, std::int32_t parent_rows,
                    std::int32_t hidden, std::int32_t qkv_rows, ops::LinearPolicy policy,
                    const char* name, bool every_width = false) {
    if (!generator_runs(generator, policy, name) || !fused_runs(generator, name)) return 0;
    PackedDeviceWeight w(generator, parent_rows, hidden, 21U);
    const std::int32_t z_rows = parent_rows - qkv_rows;
    DeviceBuffer qkv(static_cast<std::size_t>(qkv_rows) * kWidest * 2);
    DeviceBuffer z(static_cast<std::size_t>(z_rows) * kWidest * 2);
    const std::size_t capacity = ops::gdn_input_proj_workspace_capacity_bytes(
        w.weight.qtype, parent_rows, hidden, policy, 1, kWidest);
    DeviceArena workspace(std::max<std::size_t>(capacity, 256));
    const DeviceBuffer& x = inputs.get(hidden);
    OpCase op{name,
              [&](std::int32_t offset, std::int32_t t) {
                  Tensor tx = columns(x, DType::BF16, hidden, offset, t);
                  Tensor tq(qkv.p, DType::BF16, {qkv_rows, t});
                  Tensor tz(z.p, DType::BF16, {z_rows, t});
                  ops::gdn_input_proj(tx, w.weight, tq, tz, policy, workspace, stream());
              },
              {{&qkv, static_cast<std::size_t>(qkv_rows) * 2},
               {&z, static_cast<std::size_t>(z_rows) * 2}}};
    // The exact contract's rows run under the invariant route profile, as the model's calls do.
    const ops::ScopedInvariantRoutes invariant(every_width);
    return check_case(op, every_width ? exact_windows() : windows());
}

int check_attn_input(Inputs& inputs, WeightGenerator generator, std::int32_t parent_rows,
                     std::int32_t hidden, std::int32_t q_rows, std::int32_t kv_rows,
                     ops::LinearPolicy policy, const char* name, bool every_width = false) {
    if (!generator_runs(generator, policy, name) || !fused_runs(generator, name)) return 0;
    PackedDeviceWeight w(generator, parent_rows, hidden, 31U);
    DeviceBuffer q(static_cast<std::size_t>(q_rows) * kWidest * 2);
    DeviceBuffer gate(static_cast<std::size_t>(q_rows) * kWidest * 2);
    DeviceBuffer k(static_cast<std::size_t>(kv_rows) * kWidest * 2);
    DeviceBuffer v(static_cast<std::size_t>(kv_rows) * kWidest * 2);
    const std::size_t capacity = ops::attn_input_proj_workspace_capacity_bytes(
        w.weight.qtype, parent_rows, hidden, policy, 1, kWidest);
    DeviceArena workspace(std::max<std::size_t>(capacity, 256));
    const DeviceBuffer& x = inputs.get(hidden);
    OpCase op{name,
              [&](std::int32_t offset, std::int32_t t) {
                  Tensor tx = columns(x, DType::BF16, hidden, offset, t);
                  Tensor tq(q.p, DType::BF16, {q_rows, t});
                  Tensor tg(gate.p, DType::BF16, {q_rows, t});
                  Tensor tk(k.p, DType::BF16, {kv_rows, t});
                  Tensor tv(v.p, DType::BF16, {kv_rows, t});
                  ops::attn_input_proj(tx, w.weight, tq, tg, tk, tv, policy, workspace, stream());
              },
              {{&q, static_cast<std::size_t>(q_rows) * 2},
               {&gate, static_cast<std::size_t>(q_rows) * 2},
               {&k, static_cast<std::size_t>(kv_rows) * 2},
               {&v, static_cast<std::size_t>(kv_rows) * 2}}};
    // The exact contract's rows run under the invariant route profile, as the model's calls do.
    const ops::ScopedInvariantRoutes invariant(every_width);
    return check_case(op, every_width ? exact_windows() : windows());
}

// The 27B Q4/Q5 pairs (qwen3_8_27b): a Q4 parent and a Q5 parent per projection, under the
// invariant route profile (the exact contract) at every width.
int check_gdn_input_pair(Inputs& inputs, const char* name) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQkvRows = 10240;
    constexpr std::int32_t kZRows = 6144;
    const ops::ScopedInvariantRoutes invariant(true);
    RandomRowSplit qk(QType::Q4_G64_FP16, 4096, kHidden, 91U);
    RandomRowSplit value_z(QType::Q5_G64_FP16, 12288, kHidden, 92U);
    const Weight qk_weight = qk.weight();
    const Weight value_z_weight = value_z.weight();
    DeviceBuffer qkv(static_cast<std::size_t>(kQkvRows) * kWidest * 2);
    DeviceBuffer z(static_cast<std::size_t>(kZRows) * kWidest * 2);
    DeviceArena workspace(
        std::max<std::size_t>(ops::gdn_input_proj_pair_workspace_capacity_bytes(1, kWidest), 256));
    const DeviceBuffer& x = inputs.get(kHidden);
    OpCase op{name,
              [&](std::int32_t offset, std::int32_t t) {
                  Tensor tx = columns(x, DType::BF16, kHidden, offset, t);
                  Tensor tq(qkv.p, DType::BF16, {kQkvRows, t});
                  Tensor tz(z.p, DType::BF16, {kZRows, t});
                  ops::gdn_input_proj(tx, qk_weight, value_z_weight, tq, tz, workspace, stream());
              },
              {{&qkv, static_cast<std::size_t>(kQkvRows) * 2},
               {&z, static_cast<std::size_t>(kZRows) * 2}}};
    return check_case(op, exact_windows());
}

int check_attn_input_pair(Inputs& inputs, const char* name) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQRows = 6144;
    constexpr std::int32_t kKvRows = 1024;
    const ops::ScopedInvariantRoutes invariant(true);
    RandomRowSplit query_key(QType::Q4_G64_FP16, kQRows + kKvRows, kHidden, 93U);
    RandomRowSplit gate_value(QType::Q5_G64_FP16, kQRows + kKvRows, kHidden, 94U);
    const Weight query_key_weight = query_key.weight();
    const Weight gate_value_weight = gate_value.weight();
    DeviceBuffer q(static_cast<std::size_t>(kQRows) * kWidest * 2);
    DeviceBuffer gate(static_cast<std::size_t>(kQRows) * kWidest * 2);
    DeviceBuffer k(static_cast<std::size_t>(kKvRows) * kWidest * 2);
    DeviceBuffer v(static_cast<std::size_t>(kKvRows) * kWidest * 2);
    DeviceArena workspace(
        std::max<std::size_t>(ops::attn_input_proj_pair_workspace_capacity_bytes(1, kWidest), 256));
    const DeviceBuffer& x = inputs.get(kHidden);
    OpCase op{name,
              [&](std::int32_t offset, std::int32_t t) {
                  Tensor tx = columns(x, DType::BF16, kHidden, offset, t);
                  Tensor tq(q.p, DType::BF16, {kQRows, t});
                  Tensor tg(gate.p, DType::BF16, {kQRows, t});
                  Tensor tk(k.p, DType::BF16, {kKvRows, t});
                  Tensor tv(v.p, DType::BF16, {kKvRows, t});
                  ops::attn_input_proj(tx, query_key_weight, gate_value_weight, tq, tg, tk, tv,
                                       workspace, stream());
              },
              {{&q, static_cast<std::size_t>(kQRows) * 2},
               {&gate, static_cast<std::size_t>(kQRows) * 2},
               {&k, static_cast<std::size_t>(kKvRows) * 2},
               {&v, static_cast<std::size_t>(kKvRows) * 2}}};
    return check_case(op, exact_windows());
}

int check_linear_add(Inputs& inputs, WeightGenerator generator, std::int32_t n, std::int32_t k,
                     ops::LinearPolicy policy, const char* name, bool every_width = false) {
    if (!generator_runs(generator, policy, name)) return 0;
    PackedDeviceWeight w(generator, n, k, 41U);
    const auto residual_host = random_bf16(static_cast<std::size_t>(n) * kWidest, 42U, 1.0F);
    const DeviceBuffer residual = upload(residual_host);
    DeviceBuffer out(static_cast<std::size_t>(n) * kWidest * 2);
    const std::size_t capacity =
        ops::linear_add_workspace_capacity_bytes(w.weight.qtype, n, k, policy, 1, kWidest);
    DeviceArena workspace(std::max<std::size_t>(capacity, 256));
    const DeviceBuffer& x = inputs.get(k);
    OpCase op{name,
              [&](std::int32_t offset, std::int32_t t) {
                  const std::size_t column = static_cast<std::size_t>(n) * 2;
                  cuda_check(cudaMemcpy(out.p,
                                        static_cast<const std::uint8_t*>(residual.p) +
                                            column * offset,
                                        column * t, cudaMemcpyDeviceToDevice),
                             "seed residual");
                  Tensor tx = columns(x, DType::BF16, k, offset, t);
                  Tensor tr(out.p, DType::BF16, {n, t});
                  ops::linear_add(tx, w.weight, tr, policy, workspace, stream());
              },
              {{&out, static_cast<std::size_t>(n) * 2}}};
    // The exact contract's rows run under the invariant route profile, as the model's calls do.
    const ops::ScopedInvariantRoutes invariant(every_width);
    return check_case(op, every_width ? exact_windows() : windows());
}

int check_linear_swiglu(Inputs& inputs, WeightGenerator generator, std::int32_t n, std::int32_t k,
                        ops::LinearPolicy policy, const char* name, bool every_width = false) {
    if (!generator_runs(generator, policy, name)) return 0;
    PackedDeviceWeight w(generator, n, k, 51U);
    DeviceBuffer out(static_cast<std::size_t>(n / 2) * kWidest * 2);
    const std::size_t capacity =
        ops::linear_swiglu_workspace_capacity_bytes(w.weight.qtype, n, k, policy, 1, kWidest);
    DeviceArena workspace(std::max<std::size_t>(capacity, 256));
    const DeviceBuffer& x = inputs.get(k);
    OpCase op{name,
              [&](std::int32_t offset, std::int32_t t) {
                  Tensor tx = columns(x, DType::BF16, k, offset, t);
                  Tensor to(out.p, DType::BF16, {n / 2, t});
                  ops::linear_swiglu(tx, w.weight, to, policy, workspace, stream());
              },
              {{&out, static_cast<std::size_t>(n / 2) * 2}}};
    // The exact contract's rows run under the invariant route profile, as the model's calls do.
    const ops::ScopedInvariantRoutes invariant(every_width);
    return check_case(op, every_width ? exact_windows() : windows());
}

int check_rmsnorm(Inputs& inputs, std::int32_t hidden, const char* name) {
    const auto weight = upload(random_bf16(hidden, 61U, 0.2F));
    DeviceBuffer out(static_cast<std::size_t>(hidden) * kWidest * 2);
    const DeviceBuffer& x = inputs.get(hidden);
    OpCase op{name,
              [&](std::int32_t offset, std::int32_t t) {
                  Tensor tx = columns(x, DType::BF16, hidden, offset, t);
                  Tensor tw(weight.p, DType::BF16, {hidden});
                  Tensor to(out.p, DType::BF16, {hidden, t});
                  ops::rmsnorm(tx, tw, 1.0e-6F, true, to, stream());
              },
              {{&out, static_cast<std::size_t>(hidden) * 2}}};
    return check_case(op);
}

int check_gated_rmsnorm(Inputs& inputs, std::int32_t heads, const char* name) {
    constexpr std::int32_t kD = 128;
    const std::int32_t width  = kD * heads;
    const auto weight         = upload(random_bf16(kD, 71U, 0.2F));
    const auto z = upload(random_bf16(static_cast<std::size_t>(width) * kWidest, 72U, 1.0F));
    DeviceBuffer out(static_cast<std::size_t>(width) * kWidest * 2);
    const DeviceBuffer& x = inputs.get(width);
    OpCase op{name,
              [&](std::int32_t offset, std::int32_t t) {
                  const std::size_t column = static_cast<std::size_t>(width) * 2;
                  Tensor tx(static_cast<std::uint8_t*>(x.p) + column * offset, DType::BF16,
                            {kD, heads, t});
                  Tensor tz(static_cast<std::uint8_t*>(z.p) + column * offset, DType::BF16,
                            {kD, heads, t});
                  Tensor tw(weight.p, DType::BF16, {kD});
                  Tensor to(out.p, DType::BF16, {kD, heads, t});
                  ops::gated_rmsnorm(tx, tw, tz, 1.0e-6F, to, execution());
              },
              {{&out, static_cast<std::size_t>(width) * 2}}};
    return check_case(op);
}

int check_sparse_moe(Inputs& inputs, QType routed_down, const char* name,
                     bool every_width = false) {
    const ops::ScopedInvariantRoutes invariant(every_width);
    constexpr std::int32_t kHidden = 2048;
    constexpr std::int32_t kExperts = 256;
    const auto router = upload(random_bf16(static_cast<std::size_t>(kExperts + 1) * kHidden, 81U,
                                           0.05F));
    RandomRowSplit routed_gate(QType::Q4_G64_FP16, kExperts * 1024, kHidden, 82U);
    RandomRowSplit routed_down_w(routed_down, kExperts * kHidden, 512, 83U);
    RandomRowSplit shared_gate(QType::Q8_G32_FP16, 1024, kHidden, 84U);
    RandomRowSplit shared_down(QType::Q8_G32_FP16, kHidden, 512, 85U);
    const ops::SparseMoeWeights weights{dense_bf16(router, kExperts + 1, kHidden),
                                        routed_gate.weight(), routed_down_w.weight(),
                                        shared_gate.weight(), shared_down.weight()};
    const auto residual_host = random_bf16(static_cast<std::size_t>(kHidden) * kWidest, 86U, 1.0F);
    const DeviceBuffer residual = upload(residual_host);
    DeviceBuffer out(static_cast<std::size_t>(kHidden) * kWidest * 2);
    const std::size_t capacity = ops::sparse_moe_workspace_capacity_bytes(
        QType::Q4_G64_FP16, routed_down, 1, kWidest);
    DeviceArena workspace(std::max<std::size_t>(capacity, 256));
    const DeviceBuffer& x = inputs.get(kHidden);
    OpCase op{name,
              [&](std::int32_t offset, std::int32_t t) {
                  const std::size_t column = static_cast<std::size_t>(kHidden) * 2;
                  cuda_check(cudaMemcpy(out.p,
                                        static_cast<const std::uint8_t*>(residual.p) +
                                            column * offset,
                                        column * t, cudaMemcpyDeviceToDevice),
                             "seed residual");
                  Tensor tx = columns(x, DType::BF16, kHidden, offset, t);
                  Tensor to(out.p, DType::BF16, {kHidden, t});
                  ops::sparse_moe(tx, weights, ops::SparseMoeEpilogue::AddResidual, to, workspace,
                                  execution());
              },
              {{&out, static_cast<std::size_t>(kHidden) * 2}}};
    return check_case(op, every_width ? exact_windows() : windows());
}

// --- stateful ops: one call against the same tokens cut in two, the state carried ---------------

// Causal convolution with the split outputs of the GDN projection. Any cut must be invariant.
int check_conv_split(std::int32_t c0, std::int32_t c1, std::int32_t c2, const char* name) {
    const std::int32_t channels = c0 + c1 + c2;
    constexpr std::int32_t kT   = 4160;
    const auto x      = upload(random_bf16(static_cast<std::size_t>(channels) * kT, 91U, 1.0F));
    const auto weight = upload(random_bf16(static_cast<std::size_t>(channels) * 4, 92U, 0.5F));
    const auto state0 = upload(random_bf16(static_cast<std::size_t>(channels) * 3, 93U, 1.0F));
    DeviceBuffer state_a(static_cast<std::size_t>(channels) * 3 * 2);
    DeviceBuffer state_b(static_cast<std::size_t>(channels) * 3 * 2);
    DeviceBuffer out0(static_cast<std::size_t>(c0) * kT * 2), out1(static_cast<std::size_t>(c1) * kT * 2),
        out2(static_cast<std::size_t>(c2) * kT * 2);
    const auto call = [&](const DeviceBuffer& in_state, DeviceBuffer& out_state, std::int32_t begin,
                          std::int32_t t) {
        Tensor tx = columns(x, DType::BF16, channels, begin, t);
        Tensor tw(weight.p, DType::BF16, {channels, 4});
        Tensor ti(in_state.p, DType::BF16, {channels, 3});
        Tensor ts(out_state.p, DType::BF16, {channels, 3});
        Tensor o0 = columns(out0, DType::BF16, c0, begin, t);
        Tensor o1 = columns(out1, DType::BF16, c1, begin, t);
        Tensor o2 = columns(out2, DType::BF16, c2, begin, t);
        ops::causal_conv1d_silu_split(tx, tw, ti, ts, o0, o1, o2, stream());
    };
    const auto snapshot = [&] {
        cuda_synchronize();
        std::vector<std::vector<std::uint8_t>> out;
        out.push_back(download(out0.p, out0.bytes));
        out.push_back(download(out1.p, out1.bytes));
        out.push_back(download(out2.p, out2.bytes));
        return out;
    };
    call(state0, state_a, 0, kT);
    const auto reference = snapshot();
    const auto reference_state = download(state_a.p, state_a.bytes);
    int failures = 0;
    for (const std::int32_t cut : {1, 3, 17, 64, 100, 1000, 1024, 2048, 4001, 4096}) {
        call(state0, state_a, 0, cut);
        call(state_a, state_b, cut, kT - cut);
        if (snapshot() != reference || download(state_b.p, state_b.bytes) != reference_state) {
            ++failures;
            std::cout << "DIFFERS " << name << " cut " << cut << '\n';
        }
    }
    std::cout << (failures == 0 ? "invariant " : "VARIANT ") << name << '\n';
    return failures;
}

// Gated DeltaNet prefill scan. Cuts on the 64-token grid must be invariant; cuts off it are
// reported with the first differing column (known exception, see the header).
int check_gdn_split(std::int32_t qk_heads, std::int32_t v_heads, const char* name) {
    constexpr std::int32_t kD = 128;
    // Not a whole number of blocks: the reference call ends in a 20-token partial block, which a
    // cut at 4160 hands to a call of its own.
    constexpr std::int32_t kT = 4180;
    const auto q    = upload(random_bf16(static_cast<std::size_t>(kD) * qk_heads * kT, 101U, 1.0F));
    const auto k    = upload(random_bf16(static_cast<std::size_t>(kD) * qk_heads * kT, 102U, 1.0F));
    const auto v    = upload(random_bf16(static_cast<std::size_t>(kD) * v_heads * kT, 103U, 1.0F));
    const auto g    = upload(random_f32(static_cast<std::size_t>(v_heads) * kT, 104U, -0.3F, -0.01F));
    const auto beta = upload(random_f32(static_cast<std::size_t>(v_heads) * kT, 105U, 0.1F, 0.9F));
    const auto s0   = upload(random_f32(static_cast<std::size_t>(kD) * kD * v_heads, 106U, -0.05F,
                                        0.05F));
    DeviceBuffer s_a(s0.bytes), s_b(s0.bytes);
    DeviceBuffer out(static_cast<std::size_t>(kD) * v_heads * kT * 2);
    const std::size_t capacity =
        ops::gated_delta_net_workspace_capacity_bytes(qk_heads, v_heads, 1, kT);
    DeviceArena workspace(std::max<std::size_t>(capacity, 256));
    const float scale = 1.0F / std::sqrt(static_cast<float>(kD));
    const auto call   = [&](const DeviceBuffer& in_state, DeviceBuffer& out_state,
                          std::int32_t begin, std::int32_t t) {
        const auto slice3 = [&](const DeviceBuffer& b, std::int32_t heads) {
            return Tensor(static_cast<std::uint8_t*>(b.p) +
                              static_cast<std::size_t>(kD) * heads * 2 * begin,
                          DType::BF16, {kD, heads, t});
        };
        Tensor tq = slice3(q, qk_heads), tk = slice3(k, qk_heads), tv = slice3(v, v_heads);
        Tensor to = slice3(out, v_heads);
        Tensor tg(static_cast<std::uint8_t*>(g.p) + static_cast<std::size_t>(v_heads) * 4 * begin,
                  DType::FP32, {v_heads, t});
        Tensor tb(static_cast<std::uint8_t*>(beta.p) + static_cast<std::size_t>(v_heads) * 4 * begin,
                  DType::FP32, {v_heads, t});
        Tensor si(in_state.p, DType::FP32, {kD, kD, v_heads});
        Tensor so(out_state.p, DType::FP32, {kD, kD, v_heads});
        ops::gated_delta_net(tq, tk, tv, tg, tb, scale, true, workspace, si, so, to, execution());
    };
    call(s0, s_a, 0, kT);
    cuda_synchronize();
    const auto reference       = download(out.p, out.bytes);
    const auto reference_state = download(s_a.p, s_a.bytes);
    const std::size_t column   = static_cast<std::size_t>(kD) * v_heads * 2;
    int failures               = 0;
    for (const std::int32_t cut : {64, 128, 1024, 1088, 2048, 4096, 4160, 17, 100, 1000, 4001}) {
        call(s0, s_a, 0, cut);
        call(s_a, s_b, cut, kT - cut);
        cuda_synchronize();
        const auto got   = download(out.p, out.bytes);
        std::int32_t first = -1;
        for (std::int32_t c = 0; c < kT && first < 0; ++c) {
            if (std::memcmp(got.data() + c * column, reference.data() + c * column, column) != 0) {
                first = c;
            }
        }
        const bool state_same = download(s_b.p, s_b.bytes) == reference_state;
        const bool on_grid    = cut % 64 == 0;
        if (first < 0 && state_same) { continue; }
        if (on_grid) {
            ++failures;
            std::cout << "DIFFERS " << name << " cut " << cut << " first column " << first
                      << (state_same ? "" : ", final state differs") << '\n';
        } else {
            std::cout << "known off-grid exception " << name << " cut " << cut
                      << ": first differing column " << first << '\n';
        }
    }
    std::cout << (failures == 0 ? "invariant on the 64-token grid " : "VARIANT ") << name << '\n';
    return failures;
}

} // namespace

int main() {
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    // The prefill route profile is what the model runs its prefill chunks under.
    const ops::ScopedPrefillRoutes prefill_routes;
    int failures = 0;
    try {
        Inputs inputs;
        // qwen3_8_27b_nvfp4_v3
        failures += check_norm_gating(inputs, 48, 5120, "27B gdn_norm_gating_proj [96,5120]");
        failures += check_norm_gating(inputs, 48, 5120, "27B gdn_norm_gating_proj exact", true);
        failures += check_gdn_input(inputs, make_fp8_weight, 16384, 5120, 10240,
                                    ops::LinearPolicy::AllowA8, "27B gdn_input_proj FP8 A8");
        failures += check_attn_input(inputs, make_fp8_weight, 14336, 5120, 6144, 1024,
                                     ops::LinearPolicy::AllowA8, "27B attn_input_proj FP8 A8");
        failures += check_linear_add(inputs, make_fp8_weight, 5120, 6144,
                                     ops::LinearPolicy::AllowA8, "27B out linear_add FP8 A8");
        failures += check_linear_swiglu(inputs, make_nvfp4_weight, 34816, 5120,
                                        ops::LinearPolicy::AllowA4, "27B linear_swiglu NVFP4 A4");
        failures += check_linear_add(inputs, make_nvfp4_weight, 5120, 17408,
                                     ops::LinearPolicy::AllowA4, "27B down linear_add NVFP4 A4");
        failures += check_rmsnorm(inputs, 5120, "27B rmsnorm 5120");
        failures += check_gated_rmsnorm(inputs, 48, "27B gated_rmsnorm 48x128");
        failures += check_conv_split(2048, 2048, 6144, "27B causal_conv1d_silu_split");
        failures += check_gdn_split(16, 48, "27B gated_delta_net");
        // qwen3_6_27b_nvfp4 under --activations exact (A16Only): every width.
        failures += check_gdn_input(inputs, make_nvfp4_weight, 16384, 5120, 10240,
                                    ops::LinearPolicy::A16Only, "27B gdn_input_proj NVFP4 exact",
                                    true);
        failures += check_attn_input(inputs, make_nvfp4_weight, 14336, 5120, 6144, 1024,
                                     ops::LinearPolicy::A16Only, "27B attn_input_proj NVFP4 exact",
                                     true);
        failures += check_linear_add(inputs, make_nvfp4_weight, 5120, 6144,
                                     ops::LinearPolicy::A16Only, "27B out linear_add NVFP4 exact",
                                     true);
        failures += check_linear_add(inputs, make_nvfp4_weight, 5120, 17408,
                                     ops::LinearPolicy::A16Only, "27B down linear_add NVFP4 exact",
                                     true);
        failures += check_linear_swiglu(inputs, make_nvfp4_weight, 34816, 5120,
                                        ops::LinearPolicy::A16Only,
                                        "27B linear_swiglu NVFP4 exact", true);
        // qwen3_8_27b_nvfp4 FP8 inputs under --activations exact (A16Only): every width.
        failures += check_gdn_input(inputs, make_fp8_weight, 16384, 5120, 10240,
                                    ops::LinearPolicy::A16Only, "27B gdn_input_proj FP8 exact",
                                    true);
        failures += check_attn_input(inputs, make_fp8_weight, 14336, 5120, 6144, 1024,
                                     ops::LinearPolicy::A16Only, "27B attn_input_proj FP8 exact",
                                     true);
        failures += check_linear_add(inputs, make_fp8_weight, 5120, 6144,
                                     ops::LinearPolicy::A16Only, "27B out linear_add FP8 exact",
                                     true);
        // qwen3_8_27b (Q4/Q5 groupwise) under --activations exact: every width.
        failures += check_gdn_input_pair(inputs, "27B gdn_input_proj Q4/Q5 exact");
        failures += check_attn_input_pair(inputs, "27B attn_input_proj Q4/Q5 exact");
        failures += check_linear_swiglu(inputs, make_q4_g64_fp16_weight, 34816, 5120,
                                        ops::LinearPolicy::A16Only, "27B linear_swiglu Q4 exact",
                                        true);
        failures += check_linear_add(inputs, make_q5_g64_fp16_weight, 5120, 17408,
                                     ops::LinearPolicy::A16Only, "27B down linear_add Q5 exact",
                                     true);
        failures += check_linear_add(inputs, make_q5_g64_fp16_weight, 5120, 6144,
                                     ops::LinearPolicy::A16Only, "27B out linear_add Q5 exact",
                                     true);
        // qwen3_6_35b_a3b_v3
        failures += check_norm_gating(inputs, 32, 2048, "35B gdn_norm_gating_proj [64,2048]");
        failures += check_norm_gating(inputs, 32, 2048, "35B gdn_norm_gating_proj exact", true);
        failures += check_gdn_input(inputs, make_q8_g32_fp16_weight, 12288, 2048, 8192,
                                    ops::LinearPolicy::A16Only, "35B gdn_input_proj Q8");
        failures += check_attn_input(inputs, make_q8_g32_fp16_weight, 9216, 2048, 4096, 512,
                                     ops::LinearPolicy::A16Only, "35B attn_input_proj Q8");
        failures += check_linear_add(inputs, make_q8_g32_fp16_weight, 2048, 4096,
                                     ops::LinearPolicy::A16Only, "35B out linear_add Q8");
        // qwen3_6_35b_a3b_v3 under --activations exact: every width, decode and verify included.
        failures += check_gdn_input(inputs, make_q8_g32_fp16_weight, 12288, 2048, 8192,
                                    ops::LinearPolicy::A16Only, "35B gdn_input_proj Q8 exact",
                                    true);
        failures += check_attn_input(inputs, make_q8_g32_fp16_weight, 9216, 2048, 4096, 512,
                                     ops::LinearPolicy::A16Only, "35B attn_input_proj Q8 exact",
                                     true);
        failures += check_linear_add(inputs, make_q8_g32_fp16_weight, 2048, 4096,
                                     ops::LinearPolicy::A16Only, "35B out linear_add Q8 exact",
                                     true);
        failures += check_sparse_moe(inputs, QType::Q5_G64_FP16, "35B sparse_moe Q4/Q5");
        failures += check_sparse_moe(inputs, QType::Q6_G64_FP16, "35B sparse_moe Q4/Q6");
        failures +=
            check_sparse_moe(inputs, QType::Q5_G64_FP16, "35B sparse_moe Q4/Q5 exact", true);
        failures +=
            check_sparse_moe(inputs, QType::Q6_G64_FP16, "35B sparse_moe Q4/Q6 exact", true);
        failures += check_rmsnorm(inputs, 2048, "35B rmsnorm 2048");
        failures += check_gated_rmsnorm(inputs, 32, "35B gated_rmsnorm 32x128");
        failures += check_conv_split(2048, 2048, 4096, "35B causal_conv1d_silu_split");
        failures += check_gdn_split(16, 32, "35B gated_delta_net");
    } catch (const std::exception& error) {
        std::cerr << "Qwen prefill token invariance: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " Qwen prefill token invariance\n";
    return failures == 0 ? 0 : 1;
}
